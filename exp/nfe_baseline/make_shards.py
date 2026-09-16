"""Split each task's evaluation pool across client processes, without overlap.

One LIBERO client process runs at most one worker per task (the parallel path
hands each worker one task off a queue), so 500 episodes on ten tasks would
otherwise be ten connections deep. Sharding the pool positions lets S
processes share a suite: shard ``s`` runs every init whose position in the
task's pool file satisfies ``pos % S == s``, so the union is the whole pool and
no episode runs twice.

The filter format is the client's ``--episode-filter`` (``task_id`` /
``orig_init_state_idx`` / ``subset_init_state_idx``). The pool this line runs
on IS the official pruned pool, so the original index equals the position and
the two fields are written equal; the client stamps ``orig_init_state_idx``
into every result row, which is what the aggregator de-duplicates on.

Task ids follow the suite's official order, read from the RIT-Pareto task
order manifest; the pool file of each task is loaded to take its real length
rather than assuming fifty, and the manifest records every file's digest so a
substituted pool is visible downstream.

Usage:
  uv run python -m exp.nfe_baseline.make_shards --suite libero_spatial \
      --apool-dir exp/common/data/db_init/libero/libero_spatial_apool \
      --shards 5 --out-dir exp/nfe_baseline/data/shards/libero_spatial
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

TASK_ORDER_DIR = pathlib.Path("exp/rit_pareto/config")


def load_task_order(suite: str) -> dict[int, str]:
    """``task_id -> task_name`` in the suite's official order."""
    path = TASK_ORDER_DIR / f"task_order_{suite}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("suite") != suite:
        raise SystemExit(f"{path}: suite {d.get('suite')!r} != {suite!r}")
    return {int(k): v["task_name"] for k, v in d["assignment"].items()}


def pool_length(path: pathlib.Path) -> int:
    import torch

    states = torch.load(path, weights_only=False)
    return int(len(states))


def shard_entries(n_per_task: dict[int, int], shards: int) -> list[list[dict]]:
    """Shard ``s`` takes the positions ``pos % shards == s`` of every task."""
    if shards < 1:
        raise ValueError(f"shards must be >= 1, got {shards}")
    out: list[list[dict]] = [[] for _ in range(shards)]
    for task_id in sorted(n_per_task):
        for pos in range(n_per_task[task_id]):
            out[pos % shards].append(
                {
                    "task_id": task_id,
                    "orig_init_state_idx": pos,
                    "subset_init_state_idx": pos,
                }
            )
    return out


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--apool-dir", required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    apool = pathlib.Path(args.apool_dir)
    order = load_task_order(args.suite)
    files: dict[int, pathlib.Path] = {}
    for task_id, name in order.items():
        pruned = apool / f"{name}.pruned_init"
        full = apool / f"{name}.init"
        path = pruned if pruned.exists() else full
        if not path.exists():
            raise SystemExit(f"no pool file for task {task_id} ({name}) under {apool}")
        files[task_id] = path
    n_per_task = {t: pool_length(p) for t, p in files.items()}
    entries = shard_entries(n_per_task, args.shards)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s, rows in enumerate(entries):
        (out / f"shard_{s}.json").write_text(
            json.dumps(rows, indent=1), encoding="utf-8"
        )
    manifest = {
        "suite": args.suite,
        "apool_dir": str(apool),
        "shards": args.shards,
        "n_per_task": {str(t): n for t, n in n_per_task.items()},
        "total": sum(n_per_task.values()),
        "pool_sha256": {str(t): _sha(p) for t, p in files.items()},
        "task_order": {str(t): n for t, n in order.items()},
    }
    (out / "shards_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    print(
        f"{args.suite}: {manifest['total']} episodes over {args.shards} shards -> {out}"
    )


if __name__ == "__main__":
    main()
