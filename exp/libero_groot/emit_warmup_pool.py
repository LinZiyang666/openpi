"""Derive a warmup init pool that is disjoint from both the test set and the library.

Why this exists
---------------
The gate theta and the 16 judge thresholds are quantile cuts of a score
distribution, so whatever pool that distribution is measured on becomes part of
the calibration. The pi0.5 line measured it on the first ten A-pool inits per
task and recorded the resulting test-set peek as a limitation. This line does
not have to: the collection ran over the B pool, and which B-pool episodes ended
up inside the S3 library is exactly recoverable.

Two disjointness properties follow, and both matter:

*   **vs. the evaluation set** -- the warmup never touches the A pool, so no
    threshold is fitted on an episode that is later scored.
*   **vs. the library** -- a warmup episode whose own trajectory sits in the
    library would retrieve itself, and self-retrieval scores are not drawn from
    the same distribution as the real thing. Fitting a quantile on them biases
    theta upward, which shifts every operating point on the frontier.

Recovery is deterministic: the collector names each episode
``episode_<global_id>_...`` and ``global_id = task_id * trials + init_idx``, so
``build_size_libraries.episode_identity`` inverts it. A reader can recompute the
selection from the same pkl and compare it against the emitted provenance file.

Public interface: ``library_inits``, ``warmup_inits``, ``shard_entries``,
``emit_shards``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle

from exp.libero_groot import gate_pareto_bindings as gpb
from exp.libero_groot.build_size_libraries import episode_identity


def library_inits(pkl_path: str | pathlib.Path, trials: int) -> dict[int, set[int]]:
    """Init indices whose trajectory is present in the library, per task."""
    with open(pkl_path, "rb") as handle:
        artifact = pickle.load(handle)
    out: dict[int, set[int]] = {}
    for trajectory_id in {entry.trajectory_id for entry in artifact["entries"]}:
        task_id, init_idx = episode_identity(trajectory_id, trials)
        out.setdefault(task_id, set()).add(init_idx)
    return out


def warmup_inits(
    in_library: dict[int, set[int]],
    *,
    tasks: int,
    trials: int,
    per_task: int,
) -> dict[int, list[int]]:
    """The lowest ``per_task`` init indices per task that are *not* in the library.

    Ascending order rather than a sample: the selection has to be reproducible
    from the pkl alone, and a seeded shuffle would add a second thing a reader
    must trust. Raises when a task cannot supply ``per_task`` -- silently
    taking fewer would make one task's scores under-represented in a
    distribution whose quantiles are the whole point.
    """
    out: dict[int, list[int]] = {}
    for task_id in range(tasks):
        used = in_library.get(task_id, set())
        free = [idx for idx in range(trials) if idx not in used]
        if len(free) < per_task:
            raise SystemExit(
                f"task {task_id}: only {len(free)} of {trials} inits are outside "
                f"the library, need {per_task}. Extend the collection or lower "
                "the warmup size deliberately -- do not let one task contribute "
                "fewer scores than the others."
            )
        out[task_id] = free[:per_task]
    return out


def shard_entries(inits: dict[int, list[int]]) -> list[dict]:
    """Flatten the selection into ``main.py --episode-filter`` rows.

    The schema matches ``make_shards.py`` exactly, including the no-remap rule:
    ``subset_init_state_idx`` and ``orig_init_state_idx`` are both the init
    index, because the client only *skips* filtered episodes and never
    re-indexes them.
    """
    return [
        {
            "task_id": task_id,
            "subset_init_state_idx": init_idx,
            "orig_init_state_idx": init_idx,
        }
        for task_id in sorted(inits)
        for init_idx in inits[task_id]
    ]


def emit_shards(
    entries: list[dict], lanes: int, out_dir: pathlib.Path, prefix: str
) -> list[pathlib.Path]:
    """Cut the selection into ``lanes`` contiguous shards, one per worker."""
    if lanes < 1:
        raise ValueError(f"lanes must be >= 1, got {lanes}")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(entries)
    paths = []
    for i in range(lanes):
        shard = entries[total * i // lanes : total * (i + 1) // lanes]
        path = out_dir / f"{prefix}_lane{i}.json"
        path.write_text(json.dumps(shard), encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Emit the gate-Pareto warmup init pool")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--lanes", type=int, required=True, help="sim workers per slot")
    ap.add_argument("--out-dir", required=True, help="where the shard json files go")
    ap.add_argument("--per-task", type=int, default=gpb.WARMUP_PER_TASK)
    args = ap.parse_args(argv)

    binding = gpb.for_suite(args.suite)
    in_library = library_inits(binding.library, gpb.APOOL_TRIALS)
    selected = warmup_inits(
        in_library,
        tasks=gpb.NUM_TASKS,
        trials=gpb.APOOL_TRIALS,
        per_task=args.per_task,
    )
    entries = shard_entries(selected)
    paths = emit_shards(
        entries, args.lanes, pathlib.Path(args.out_dir), f"gpw_{binding.tag}"
    )

    provenance_dir = binding.data_root / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "suite": binding.suite,
        "library": binding.library,
        "trials": gpb.APOOL_TRIALS,
        "per_task": args.per_task,
        "in_library_inits": {str(k): sorted(v) for k, v in sorted(in_library.items())},
        "warmup_inits": {str(k): v for k, v in sorted(selected.items())},
        "episodes": len(entries),
        "shards": [str(p) for p in paths],
    }
    out = provenance_dir / "warmup_pool_provenance.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    overlap = {
        task_id
        for task_id, picked in selected.items()
        if set(picked) & in_library.get(task_id, set())
    }
    if overlap:  # unreachable by construction; a loud tripwire, not a check
        raise SystemExit(f"warmup pool overlaps the library on tasks {sorted(overlap)}")

    print(f"{binding.suite}: {len(entries)} warmup episodes over {args.lanes} lanes")
    print(f"provenance -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
