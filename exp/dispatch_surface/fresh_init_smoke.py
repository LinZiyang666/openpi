"""Real-LIBERO round-trip smoke for the fresh-init generator (confirmation
plan 3.7-c, Verify stage; runs inside the LIBERO client env on each machine).

For every task of the suite it derives ONE state under the ``SMOKE``
namespace (never ``P`` / ``C``, so the frozen pool seed streams stay
untouched), through exactly the generator's state machine
(``derive_seeds`` -> ``random`` / ``numpy`` / ``env.seed`` -> ``env.reset()``
-> ``read_state``), then

1. ``env.reset(); env.set_init_state(state)`` (the client path of
   ``examples/libero/main.py``) and reads the simulator state back: it must
   equal the written state element for element;
2. steps the environment once with a zero action (env-level 1-step smoke);
3. writes the state through ``materialize`` and reads it back through
   ``load_init_states``: same content digest;
4. checks the BDDL bytes against the task manifest and, when the official
   super-pool is available, that the state width equals the official width.

The record carries host / software versions / asset rollup and a rollup of
the per-task state digests; ``compare`` checks two records from different
hosts agree on every digest (cross-machine determinism of the real env).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import tempfile

import numpy as np

from exp.dispatch_surface.generate_fresh_inits import (
    _file_sha256,
    derive_seeds,
    environment_record,
    load_init_states,
    load_task_manifest,
    make_env,
    materialize,
    official_state_dim,
    read_state,
    state_sha256,
)

PROTOCOL = "dispatch_surface_rev2_fresh_init_smoke"
NAMESPACE_POOL = "SMOKE"


def smoke_task(suite: str, name: str, bddl_file: str, k: int, tmp: pathlib.Path, width_expected: int | None) -> dict:
    seeds = derive_seeds(suite, name, NAMESPACE_POOL, k, 0)
    out = {"task_name": name, "k": k, "seeds": seeds, "ok": False, "problems": []}
    env = None
    try:
        random.seed(seeds["py_seed"])
        np.random.seed(seeds["np_seed"])
        env = make_env(bddl_file)
        env.seed(seeds["env_seed"])
        env.reset()
        state = read_state(env)
        out["width"] = int(state.shape[0])
        out["finite"] = bool(np.isfinite(state).all())
        out["state_sha256"] = state_sha256(state)
        if width_expected is not None and out["width"] != width_expected:
            out["problems"].append(f"width {out['width']} != official {width_expected}")
        # 1. client-path round trip: reset, set_init_state, read back
        env.reset()
        env.set_init_state(state)
        back = read_state(env)
        out["roundtrip_equal"] = bool(back.shape == state.shape and np.array_equal(back, state))
        if not out["roundtrip_equal"]:
            out["problems"].append(f"set_init_state round trip differs (max abs {float(np.max(np.abs(back - state))) if back.shape == state.shape else 'shape'})")
        # 2. one env step with a zero action
        env.step(np.zeros(7, dtype=np.float64))
        out["step_ok"] = True
        # 3. file round trip through materialize / load_init_states
        block = {"tasks": {name: {}}}
        digests = materialize(block, {name: [state]}, tmp)
        loaded = np.asarray(load_init_states(tmp / f"{name}.init"))
        out["file_sha256"] = digests[name]
        out["file_roundtrip_equal"] = bool(loaded.shape == (1, state.shape[0]) and state_sha256(loaded[0]) == out["state_sha256"])
        if not out["file_roundtrip_equal"]:
            out["problems"].append("materialize / load_init_states round trip differs")
        out["ok"] = not out["problems"] and out["finite"]
    except Exception as exc:  # the smoke reports, never crashes the loop
        out["problems"].append(f"exception:{type(exc).__name__}:{exc}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return out


def run(args) -> dict:
    tm = load_task_manifest(args.task_manifest)
    if tm["suite"] != args.suite:
        raise SystemExit("task manifest suite != --suite")
    bddl_root = pathlib.Path(args.bddl_root)
    tasks = sorted(tm["tasks"], key=lambda t: t["task_id"])
    results = []
    with tempfile.TemporaryDirectory(prefix="dsp_smoke_") as td:
        tmp = pathlib.Path(td)
        for t in tasks:
            bddl = bddl_root / t["bddl_file"]
            bddl_ok = bddl.is_file() and _file_sha256(bddl) == t["bddl_sha256"]
            width = official_state_dim(pathlib.Path(args.apool_dir), t["task_name"]) if args.apool_dir else None
            res = {"task_id": t["task_id"], "bddl_sha256_ok": bool(bddl_ok)}
            if not bddl_ok:
                res.update({"task_name": t["task_name"], "ok": False, "problems": ["BDDL bytes != task manifest"]})
            else:
                res.update(smoke_task(args.suite, t["task_name"], str(bddl), args.k, tmp, width))
            res["official_width"] = width
            results.append(res)
            print(json.dumps({k: res.get(k) for k in ("task_id", "task_name", "width", "ok", "problems")}))
    digests = [r.get("state_sha256") or "" for r in results]
    record = {"protocol": PROTOCOL, "suite": args.suite, "namespace_pool": NAMESPACE_POOL, "k": args.k,
              "task_manifest_sha256": _file_sha256(pathlib.Path(args.task_manifest)),
              "environment": environment_record(args.assets_dir), "tasks": results,
              "state_rollup_sha256": hashlib.sha256("\n".join(digests).encode()).hexdigest(),
              "all_ok": all(r.get("ok") for r in results)}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"all_ok": record["all_ok"], "state_rollup_sha256": record["state_rollup_sha256"],
                      "host": record["environment"]["host"]}))
    if not record["all_ok"]:
        raise SystemExit("fresh-init smoke FAILED on at least one task")
    return record


def compare(paths: list[str]) -> dict:
    recs = [json.loads(pathlib.Path(p).read_text()) for p in paths]
    problems = []
    hosts = [r["environment"]["host"] for r in recs]
    if len(set(hosts)) != len(hosts):
        problems.append(f"records are not from distinct hosts: {hosts}")
    base = recs[0]
    for r in recs[1:]:
        for key in ("suite", "namespace_pool", "k", "task_manifest_sha256", "state_rollup_sha256"):
            if r.get(key) != base.get(key):
                problems.append(f"{key} differs between {hosts[0]} and {r['environment']['host']}")
        if (r["environment"].get("assets_rollup") or {}).get("sha256") != (base["environment"].get("assets_rollup") or {}).get("sha256"):
            problems.append("asset rollups differ")
        for a, b in zip(base["tasks"], r["tasks"]):
            for key in ("task_name", "width", "state_sha256", "file_sha256", "seeds"):
                if a.get(key) != b.get(key):
                    problems.append(f"{a.get('task_name')}: {key} differs")
    if not all(r["all_ok"] for r in recs):
        problems.append("at least one record is not all_ok")
    out = {"hosts": hosts, "problems": problems, "verified": not problems,
           "state_rollup_sha256": base["state_rollup_sha256"], "software": {h: {k: r["environment"].get(k) for k in ("libero", "robosuite", "mujoco", "torch", "numpy")} for h, r in zip(hosts, recs)}}
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", required=True)
    r.add_argument("--task-manifest", required=True)
    r.add_argument("--bddl-root", required=True, help="directory holding the suite's BDDL files")
    r.add_argument("--apool-dir", default="", help="official super-pool (width check); optional")
    r.add_argument("--assets-dir", required=True)
    r.add_argument("--k", type=int, default=0)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("records", nargs="+")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        out = compare(args.records)
        if not out["verified"]:
            raise SystemExit("cross-machine smoke comparison FAILED")


if __name__ == "__main__":
    main()
