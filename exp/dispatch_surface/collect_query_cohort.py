"""Query-cohort collection driver (150 teacher rollouts on the C init pool).

Two subcommands:

  plan    — from the split manifest, emit the per-task episode list (task_id,
            official init idx, split) and the standard collection commands the
            operator runs (serve_policy --collect + the LIBERO client on the
            materialised C pool). Collection itself uses the existing stack;
            this driver owns only the WHAT (which inits) and the audit.

  verify  — after collection, walk the produced H5 directory and check: every
            (task_id, init_idx) in the plan is present exactly once, no H5
            carries an init outside the plan, per-episode attrs carry
            task_id / init_state_idx / success, and the fit/cal counts match
            the 5/10-per-task quota. Writes a cohort manifest with per-file
            sha256 for the records ledger.

Usage:
  uv run python -m exp.dispatch_surface.collect_query_cohort plan \
      --split-manifest exp/dispatch_surface/data/init_pools/split_manifest.json \
      --pool-dir exp/dispatch_surface/data/init_pools/query_c \
      --out exp/dispatch_surface/data/query_cohort_plan.json

  uv run python -m exp.dispatch_surface.collect_query_cohort verify \
      --plan exp/dispatch_surface/data/query_cohort_plan.json \
      --h5-dir exp/dispatch_surface/data/query_cohort \
      --out exp/dispatch_surface/data/query_cohort_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import h5py

FIT_PER_TASK = 5
CAL_PER_TASK = 10


def cmd_plan(args) -> None:
    manifest = json.loads(pathlib.Path(args.split_manifest).read_text())
    episodes = []
    for tid_str, info in sorted(manifest["assignment"].items(), key=lambda kv: int(kv[0])):
        tid = int(tid_str)
        # The materialised C pool holds sorted(fit + cal) official indices;
        # the LIBERO client's loop index is the position WITHIN that file
        # (subset index). Both index spaces are planned explicitly so the
        # collector can stamp them and downstream joins never guess (G2-B4).
        pool_officials = sorted(int(i) for i in (info["fit"] + info["cal"]))
        for split, quota in (("fit", FIT_PER_TASK), ("cal", CAL_PER_TASK)):
            idxs = info[split]
            if len(idxs) != quota:
                raise SystemExit(f"task {tid}: split {split} has {len(idxs)} inits, expected {quota}")
            for idx in idxs:
                episodes.append({
                    "task_id": tid, "task_name": info["task_name"],
                    "orig_init_state_idx": int(idx),
                    "subset_init_state_idx": pool_officials.index(int(idx)),
                    "split": split,
                })
    plan = {
        "split_manifest": str(args.split_manifest),
        "pool_dir": str(args.pool_dir),
        "episodes": episodes,
        "collection_recipe": [
            # --non-concurrent is REQUIRED, not optional: serve_policy refuses
            # --collect without it, because concurrent forward hooks
            # cross-contaminate the captured embeddings. Printing the command
            # without it hands the operator something that fails at launch.
            "uv run scripts/serve_policy.py --collect --collect_dir <out> --env LIBERO "
            "--non-concurrent "
            "policy:checkpoint --policy.config pi05_libero --policy.dir <ckpt>",
            f"client: run each task's episodes on the materialised C pool at {args.pool_dir} "
            "(pure teacher, no cache), one rollout per planned (task, subset init). The "
            "client MUST pass extra_metadata with task_id / orig_init_state_idx / "
            "subset_init_state_idx / split from this plan so the collector stamps them "
            "into the H5 attrs (allowlisted persistence in EpisodeDataCollector).",
        ],
    }
    pathlib.Path(args.out).write_text(json.dumps(plan, indent=2))
    print(f"planned {len(episodes)} episodes -> {args.out}")


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def cmd_verify(args) -> None:
    plan = json.loads(pathlib.Path(args.plan).read_text())
    wanted = {
        (e["task_id"], e["orig_init_state_idx"]):
            {"split": e["split"], "subset": e["subset_init_state_idx"]}
        for e in plan["episodes"]
    }
    seen: dict[tuple[int, int], str] = {}
    files = []
    for h5_path in sorted(pathlib.Path(args.h5_dir).rglob("*.h5")):
        with h5py.File(h5_path, "r") as h5:
            for attr in ("task_id", "orig_init_state_idx", "subset_init_state_idx",
                         "split", "success"):
                if attr not in h5.attrs:
                    raise SystemExit(f"{h5_path}: missing required attr '{attr}'")
            key = (int(h5.attrs["task_id"]), int(h5.attrs["orig_init_state_idx"]))
            subset = int(h5.attrs["subset_init_state_idx"])
            split = str(h5.attrs["split"])
        if key not in wanted:
            raise SystemExit(f"{h5_path}: (task, official init)={key} is not in the plan")
        if subset != wanted[key]["subset"] or split != wanted[key]["split"]:
            raise SystemExit(
                f"{h5_path}: attrs (subset={subset}, split={split}) disagree with the "
                f"plan ({wanted[key]}) — identity chain is broken"
            )
        if key in seen:
            raise SystemExit(f"{h5_path}: duplicate collection for {key} "
                             f"(already in {seen[key]})")
        seen[key] = str(h5_path)
        files.append({"path": str(h5_path), "task_id": key[0],
                      "init_idx": key[1],  # official index — the join key downstream
                      "subset_init_state_idx": subset,
                      "split": split, "sha256": _sha256(h5_path)})
    missing = sorted(set(wanted) - set(seen))
    if missing:
        raise SystemExit(f"{len(missing)} planned episodes missing, first: {missing[:5]}")
    counts = {"fit": 0, "cal": 0}
    for f in files:
        counts[f["split"]] += 1
    expected = {"fit": FIT_PER_TASK * 10, "cal": CAL_PER_TASK * 10}
    if counts != expected:
        raise SystemExit(f"split counts {counts} != expected {expected}")
    out = {"plan": str(args.plan), "h5_dir": str(args.h5_dir),
           "counts": counts, "files": files}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"cohort verified: {counts} -> {args.out}")


def cmd_launch(args) -> None:
    """Print (or exec) the real collection client command consuming the plan.

    The LIBERO client's ``--cohort-plan`` flag (examples/libero/main.py)
    filters episodes to the planned (task, subset) pairs and sends the full
    four-field identity metadata that ``EpisodeDataCollector`` persists into
    H5 attrs — the executable end of the plan -> client -> collector chain.
    """
    plan = json.loads(pathlib.Path(args.plan).read_text())
    cmd = [
        "python", "examples/libero/main.py",
        "--host", args.host, "--port", str(args.port),
        "--task-suite-name", args.task_suite,
        "--num-trials-per-task", str(FIT_PER_TASK + CAL_PER_TASK),
        "--num-workers", "1",
        "--init-states-dir", plan["pool_dir"],
        "--cohort-plan", str(args.plan),
    ]
    print("server side:")
    for line in plan["collection_recipe"][:1]:
        print(f"  {line}")
    print("client side:")
    print("  " + " ".join(cmd))
    if args.run:
        import subprocess

        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--split-manifest", required=True)
    p_plan.add_argument("--pool-dir", required=True)
    p_plan.add_argument("--out", required=True)
    p_plan.set_defaults(func=cmd_plan)
    p_launch = sub.add_parser("launch")
    p_launch.add_argument("--plan", required=True)
    p_launch.add_argument("--host", default="127.0.0.1")
    p_launch.add_argument("--port", type=int, default=8000)
    p_launch.add_argument("--task-suite", default="libero_spatial")
    p_launch.add_argument("--run", action="store_true",
                          help="execute the client command instead of printing it")
    p_launch.set_defaults(func=cmd_launch)
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--plan", required=True)
    p_verify.add_argument("--h5-dir", required=True)
    p_verify.add_argument("--out", required=True)
    p_verify.set_defaults(func=cmd_verify)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
