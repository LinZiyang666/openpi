"""Sample and materialise the RIT-Pareto shadow cohort from the official pool.

The shadow ("cache in the shadow") calibration of the RIT-Pareto line runs the
pure teacher on a per-task random subset of the official LIBERO ``pruned_init``
evaluation pool (owner ruling 2026-09-01: calibrate and evaluate on the same
500-init test set; the owner explicitly waived the contamination concern for
this line).

``sample`` draws ``fit_per_task + cal_per_task`` distinct official indices per
task with a seeded generator, writes a split-manifest-shaped JSON carrying only
the fields the unchanged Phase 0 tools read (``assignment`` / ``quota`` /
``task_name``) and materialises the query pool through
``exp.dispatch_surface.split_init_pools.materialize_pool`` so that the LIBERO
client's loop index is the position inside the sampled file -- exactly the
subset index the cohort plan stamps into the H5 attrs.

Downstream, unchanged: ``collect_query_cohort plan|launch|verify`` and
``build_dispatch_table --split-manifest <this manifest>``.

Usage:
  uv run python -m exp.rit_pareto.shadow_cohort sample \
      --suite libero_spatial \
      --apool-dir exp/common/data/db_init/libero/libero_spatial_apool \
      --task-order-manifest exp/dispatch_surface/data/init_pools/split_manifest.json \
      --seed 20260901 --fit-per-task 5 --cal-per-task 10 \
      --pool-out <dir> --out-manifest <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from exp.dispatch_surface.split_init_pools import OFFICIAL_PER_TASK, materialize_pool

NUM_TASKS = 10


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def sample_assignment(
    task_names: dict[int, str], *, seed: int, fit_per_task: int, cal_per_task: int,
    pool_size: int = OFFICIAL_PER_TASK,
) -> dict[int, dict]:
    """Per task: ``fit_per_task + cal_per_task`` distinct official indices.

    One child generator per task (seeded from ``seed`` and the task id) so the
    draw for task ``t`` does not move when another task's quota changes.
    """
    if fit_per_task <= 0 or cal_per_task <= 0:
        raise ValueError("both quotas must be positive (the Phase 0 tools treat 0 as unset)")
    k = fit_per_task + cal_per_task
    if k > pool_size:
        raise ValueError(f"quota {k} exceeds the pool size {pool_size}")
    assignment: dict[int, dict] = {}
    for tid in sorted(task_names):
        rng = np.random.default_rng([int(seed), int(tid)])
        picked = sorted(int(i) for i in rng.choice(pool_size, size=k, replace=False))
        # Deterministic split: the first fit_per_task of the sorted draw are
        # fit, the rest cal. The label carries no weight for RIT-PL (it fits on
        # every row); it only has to satisfy the Phase 0 verify quota.
        assignment[tid] = {
            "task_name": task_names[tid],
            "fit": picked[:fit_per_task],
            "cal": picked[fit_per_task:],
        }
    return assignment


def load_task_order(manifest_path: pathlib.Path) -> dict[int, str]:
    """``task_id -> task_name`` from a split manifest (official LIBERO order)."""
    manifest = json.loads(manifest_path.read_text())
    names = {int(tid): str(info["task_name"]) for tid, info in manifest["assignment"].items()}
    if sorted(names) != list(range(NUM_TASKS)):
        raise SystemExit(f"{manifest_path}: expected task ids 0..{NUM_TASKS - 1}, got {sorted(names)}")
    return names


def cmd_sample(args) -> None:
    apool_dir = pathlib.Path(args.apool_dir)
    task_names = load_task_order(pathlib.Path(args.task_order_manifest))
    for name in task_names.values():
        if not (apool_dir / f"{name}.init").is_file():
            raise SystemExit(f"A-pool file missing for task {name!r} in {apool_dir}")
    assignment = sample_assignment(
        task_names, seed=args.seed, fit_per_task=args.fit_per_task, cal_per_task=args.cal_per_task,
    )
    pool_out = pathlib.Path(args.pool_out)
    if pool_out.exists() and any(pool_out.iterdir()):
        raise SystemExit(f"--pool-out must be empty or absent: {pool_out}")
    digests = materialize_pool(apool_dir, pool_out, assignment, ["fit", "cal"])
    manifest = {
        "protocol": "rit_pareto_shadow_v1",
        "suite": args.suite,
        "seed": int(args.seed),
        "apool_dir": str(apool_dir),
        "apool_file_sha256": {
            name: _file_sha256(apool_dir / f"{name}.init") for name in task_names.values()
        },
        "quota": {"fit": int(args.fit_per_task), "cal": int(args.cal_per_task)},
        "assignment": {str(tid): info for tid, info in sorted(assignment.items())},
        "pool_dir": str(pool_out),
        "pool_digests": {"shadow": digests},
    }
    out = pathlib.Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    n = sum(len(v["fit"]) + len(v["cal"]) for v in assignment.values())
    print(f"sampled {n} shadow episodes over {len(assignment)} tasks -> {out}; pool -> {pool_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--suite", required=True, choices=("libero_spatial", "libero_10"))
    p.add_argument("--apool-dir", required=True)
    p.add_argument("--task-order-manifest", required=True,
                   help="split manifest whose assignment carries the official task order")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--fit-per-task", type=int, default=5)
    p.add_argument("--cal-per-task", type=int, default=10)
    p.add_argument("--pool-out", required=True)
    p.add_argument("--out-manifest", required=True)
    p.set_defaults(func=cmd_sample)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
