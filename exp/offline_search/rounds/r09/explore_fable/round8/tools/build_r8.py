"""Round 8 (fable): ``r09_fable_r8`` -- pure-cache look saving on the four 500-demo cells (NOT launched here).

Per cell: the R8 cache A row as same-batch control + one look-saving variant on top of A:
  pi0.5  : PaceWrist (pace-gated wrist looks) over StageFollow E1 (gated follow)      -- new prefit
  GR00T  : StageFollow E1 (gated follow) = the R8 SF1 row and its frozen R7 artifact   -- verbatim copy
No judge, no policy call anywhere in this batch.  Sub-commands: spec, prefit, check, selftest, score.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from exp.offline_search.rounds.r09.explore_fable.round5.tools import build_r5 as b5

HERE = Path(__file__).resolve()
REPO = b5.REPO
R8 = Path(os.environ.get("R9F_R8_ROOT", "/home/weiland/trace_runs/os_closed_loop/r09_fable_r8"))
R8_MAIN = Path("/home/weiland/trace_runs/os_closed_loop/r08_main/arms.json")
STORE = b5.STORE
REG = "r8_kwargs.json"
M8 = "exp.offline_search.rounds.r09.explore_fable.round8.tools.methods"
FOLLOW_BASE = "exp.offline_search.rounds.r07.c1_follow.methods:StageFollow"
R07_FITS = b5.R07_FITS
CELLS = [("pi05", "l10", 500), ("pi05", "spatial", 500), ("groot", "l10", 500), ("groot", "spatial", 500)]
SUITE_ABBR = {"l10": "l10", "spatial": "sp"}
WRIST_ARGS = ["--os-tokens", "off", "--os-request-cameras"]
CPUS = b5.CPUS


def _rows():
    return {a["arm"]: a for a in json.loads(R8_MAIN.read_text())}


def _copy(src, name):
    row = dict(name=name, model=src["model"], suite=src["suite_short"], mode="plugin", method=src["method"],
               kwargs=json.loads(json.dumps(src["kwargs"])), cost_ledger=True, client_overrides=dict(src["client_overrides"]),
               plugin_args=list(src["plugin_args"]))
    if "--os-judge" in row["plugin_args"] or "--os-policy-tail" in row["plugin_args"]:
        raise ValueError(f"{src['arm']} is not a pure-cache row")
    return row


def _fit_artifact(row):
    return row["plugin_args"][row["plugin_args"].index("--os-fit-artifact") + 1]


def build(run_root=R8):
    fits = Path(run_root) / "fits"
    rows = _rows()
    arms, prefit, prov = [], {}, {}
    for model, suite, lib in CELLS:
        cn = f"{model}_{suite}_{lib}"
        a_row = rows[f"r8_{cn}_A"]
        arms.append(_copy(a_row, f"r9f8_{cn}_A"))
        prov[f"r9f8_{cn}_A"] = dict(source=a_row["arm"], artifact=_fit_artifact(a_row), sha256=b5.sha(_fit_artifact(a_row)))
        if model == "groot":
            sf = rows[f"r8_{cn}_SF1"]
            if sf["kwargs"].get("extend_blocks") != 1 or not sf["kwargs"].get("stage_gate") or not sf["kwargs"].get("state_valve"):
                raise ValueError("R8 SF1 row is not the stage-gated one-block extension")
            arms.append(_copy(sf, f"r9f8_{cn}_fg"))
            prov[f"r9f8_{cn}_fg"] = dict(source=sf["arm"], artifact=_fit_artifact(sf), sha256=b5.sha(_fit_artifact(sf)))
        else:
            name = f"r9f8_{cn}_wpace_fg"
            base_kwargs = dict(a_row["kwargs"], extend_blocks=1, stage_gate=True, state_valve=True)
            if a_row["kwargs"] != {"lib": "big", "kref": 8, "serving": "anchor_tail", "budget": 1, "gates": "budget_only"}:
                raise ValueError(f"{cn}: unexpected A kwargs {a_row['kwargs']}")
            kw = dict(enabled=True, base_spec=FOLLOW_BASE, base_kwargs=base_kwargs, base_fit=_fit_artifact(a_row),
                      wrist_fit=str(R07_FITS / f"wrist_pi05_{SUITE_ABBR[suite]}_500.pkl"),
                      stage_fit=str(R07_FITS / f"stages_pi05_{SUITE_ABBR[suite]}_500.pkl"), pace_lag=1)
            method = f"{M8}:PaceWrist"
            prefit[name] = dict(method=method, kwargs=kw, cell=f"{model}_{suite}_cache", model=model, wrist=True)
            arms.append(dict(name=name, model=model, suite=suite, mode="plugin", method=method, kwargs=kw, cost_ledger=True,
                             client_overrides=dict(a_row["client_overrides"]),
                             plugin_args=["--os-root", STORE, "--os-no-shadow-native", "--os-blind"] + WRIST_ARGS + ["--os-fit-artifact", str(fits / f"{name}.pkl")]))
    if len({a["name"] for a in arms}) != len(arms):
        raise ValueError("duplicate arm names")
    return arms, prefit, prov


def write_spec(run_root=R8):
    run_root = Path(run_root)
    arms, prefit, prov = build(run_root)
    for d in ("fits", "prefit_logs", "manifests", "selftest"):
        (run_root / d).mkdir(parents=True, exist_ok=True)
    (run_root / "arms_in.json").write_text(json.dumps(arms, indent=1))
    (run_root / REG).write_text(json.dumps(prefit, indent=1))
    (run_root / "provenance.json").write_text(json.dumps(prov, indent=1))
    manifest = json.loads(b5.MANIFEST_SRC.read_text())
    if {(int(t), int(i)) for t, i in manifest} != {(t, i) for t in range(10) for i in b5.EVAL_INITS}:
        raise ValueError("manifest is not tasks 0-9 x inits 20-29")
    shutil.copyfile(b5.MANIFEST_SRC, run_root / "manifests" / "eval100_inits20_29.json")
    print(f"{len(arms)} arms, {len(prefit)} prefits -> {run_root}")


def check(run_root=R8):
    """Artifact-level assertions: pi0.5 PaceWrist over StageFollow E1 with R7 wrist/stage artifacts; GR00T frozen SF1."""
    run_root = Path(run_root)
    sys.path.insert(0, str(REPO))
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import fingerprint
    arms = json.loads((run_root / "arms_in.json").read_text())
    bad = []
    for a in arms:
        if a["name"].endswith("_A"):
            continue
        path = a["plugin_args"][-1]
        with open(path, "rb") as f:
            blob = pickle.load(f)
        obj = blob["method"]
        if a["model"] == "pi05":
            base = obj.base
            c = dict(cls=type(obj).__name__ == "PaceWrist", enabled=bool(obj.enabled), pace=float(obj.pace_lag) == 1.0,
                     base_follow=type(base).__name__ == "StageFollow" and int(base.follow_extend_blocks) == 1 and base.follow_stage_gate is True and base.follow_state_valve is True,
                     base_commit=base.serving == "anchor_tail" and int(base.budget) == 1 and base.gates == "budget_only" and base.lib == "big",
                     stage_fp=obj.stages.retrieval_fingerprint == fingerprint(base), follow_fp=base.follow_table.retrieval_fingerprint == fingerprint(base),
                     wrist_width=obj.wrist.B1T.shape[0] + obj.state_width == 72, camera_full=obj.next_camera_mode == "full" and obj._camera_mode == "full",
                     cell=blob.get("cell") == f"{a['model']}_{a['suite']}_cache")
        else:
            c = dict(cls=type(obj).__name__ == "StageFollow", follow=int(obj.follow_extend_blocks) == 1 and obj.follow_stage_gate is True and obj.follow_state_valve is True,
                     commit=obj.serving == "anchor_tail" and int(obj.budget) == 1 and obj.gates == "budget_only",
                     follow_fp=obj.follow_table.retrieval_fingerprint == fingerprint(obj), cell=blob.get("cell") == f"{a['model']}_{a['suite']}_cache",
                     frozen_r7=path.startswith("/home/weiland/trace_runs/os_closed_loop/r07_main/fits/"))
        ok = all(c.values())
        print(f"{a['name']:36s} {type(obj).__name__:12s} {'OK' if ok else 'FAIL ' + str([k for k, v in c.items() if not v])}")
        if not ok:
            bad.append(a["name"])
    if bad:
        raise SystemExit(f"artifact checks failed: {bad}")
    print(f"{sum(1 for a in arms if not a['name'].endswith('_A'))} variant artifacts pass all checks")


def _run(cmd, log, threads="1"):
    cmd = ["taskset", "-c", CPUS] + cmd
    env = dict(os.environ, PYTHONPATH=".:src", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode


def selftest(run_root=R8, workers=4, episodes=4):
    """Pure-cache CPU selftests (``--blind``, no judge, no policy tail) on every variant arm with its fitted artifact."""
    run_root = Path(run_root)
    arms = [a for a in json.loads((run_root / "arms_in.json").read_text()) if not a["name"].endswith("_A")]
    jobs = []
    for a in arms:
        yaml = run_root / "config" / f"{a['name']}.yaml"
        if not yaml.exists():
            raise SystemExit(f"emit the arms first: {yaml} missing")
        cmd = [str(REPO / ".venv/bin/python"), "-m", "exp.offline_search.closed_loop.selftest", "--cell", f"{a['model']}_{a['suite']}_cache", "--yaml", str(yaml),
               "--method", a["method"], "--kwargs", json.dumps(a["kwargs"]), "--fit-artifact", a["plugin_args"][-1], "--root", STORE, "--blind",
               "--episodes", str(episodes), "--no-shadow", "--out", str(run_root / "selftest" / a["name"])]
        jobs.append((a["name"], cmd))

    def one(job):
        tag, cmd = job
        rc = _run(cmd, run_root / "selftest" / f"{tag}.log")
        out = (run_root / "selftest" / f"{tag}.log").read_text()
        return tag, '"PASS": true' in out and rc == 0 and "Traceback" not in out.split("PASS")[-1]

    with ThreadPoolExecutor(min(workers, len(jobs))) as ex:
        results = list(ex.map(one, jobs))
    for tag, passed in results:
        print(f"{tag:40s} {'PASS' if passed else 'FAIL'}")
    if not all(p for _, p in results):
        raise SystemExit("selftest failures")
    print(f"{len(results)} selftests pass")


def score(run_root=R8):
    run_root = Path(run_root)
    arms = [a["name"] for a in json.loads((run_root / "arms_in.json").read_text())]
    rows = {a: b5.arm_outcomes(run_root, a) for a in arms}
    for a, v in rows.items():
        print(f"{a:36s} " + ("(missing)" if v is None else f"n={v['n']:3d} SR {v['sr']:.3f} IR {v['ir']:.3f} (full {v['look']:.3f} wrist {v['wrist']:.3f} call {v['call']:.3f} follow {v['follow']:.3f})"))
    (run_root / "score.json").write_text(json.dumps(dict(table=rows, inits=[20, 29]), indent=1))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["spec", "prefit", "check", "selftest", "score"])
    ap.add_argument("--run-root", default=str(R8))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=4)
    a = ap.parse_args(argv)
    if a.action == "spec":
        write_spec(a.run_root)
    elif a.action == "prefit":
        b5.prefit(a.run_root, a.workers, reg_name=REG)
    elif a.action == "check":
        check(a.run_root)
    elif a.action == "selftest":
        selftest(a.run_root, a.workers, a.episodes)
    else:
        score(a.run_root)


if __name__ == "__main__":
    main()
