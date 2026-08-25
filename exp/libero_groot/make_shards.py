"""Split a LIBERO collection run into N independent client lanes.

``examples/libero/main.py`` iterates ``range(num_trials_per_task)`` and merely
*skips* episodes missing from ``--episode-filter`` — it never re-indexes.  So
``initial_states[episode_idx]`` and ``global_episode_id = task_id *
num_trials + episode_idx`` stay identical no matter how the (task, episode)
pairs are partitioned.  That makes the filter a collision-free sharding knob:
any partition reproduces exactly the same 500 episodes as one unsharded run,
and the server-side HDF5 names (``episode_%04d``) stay unique across lanes.

Chunks are cut contiguously in (task_id, episode_idx) order so a lane touches
as few LIBERO tasks as possible — each task costs one ``OffScreenRenderEnv``
construction, and stacking many GL contexts in one process is what exhausted
the framebuffer on the RoboCasa side.

``--done-dir`` drops pairs whose HDF5 already exists, which turns the same tool
into the mop-up pass for a partially finished run (the collector stamps
``success`` into every episode file, so killing a lane never loses labels).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

_EPISODE_RE = re.compile(r"episode_(\d+)_")


def done_episode_ids(done_dir: pathlib.Path) -> set[int]:
    """Global episode ids already flushed to HDF5 under ``done_dir``."""
    ids = set()
    for path in done_dir.rglob("episode_*.h5"):
        match = _EPISODE_RE.match(path.name)
        if match:
            ids.add(int(match.group(1)))
    return ids


def build_shards(
    *, tasks: list[int], trials: int, lanes: int, done: set[int]
) -> list[list[tuple[int, int]]]:
    pairs = [
        (task_id, episode_idx)
        for task_id in tasks
        for episode_idx in range(trials)
        if task_id * trials + episode_idx not in done
    ]
    shards: list[list[tuple[int, int]]] = [[] for _ in range(lanes)]
    if not pairs:
        return shards
    # Contiguous cut: lane i takes pairs [i*n//lanes, (i+1)*n//lanes).
    total = len(pairs)
    for i in range(lanes):
        shards[i] = pairs[total * i // lanes : total * (i + 1) // lanes]
    return shards


def remaining_of(shard: pathlib.Path, done: set[int], trials: int) -> list[dict]:
    """Entries of ``shard`` whose episode has no HDF5 yet."""
    return [
        e
        for e in json.loads(shard.read_text())
        if e["task_id"] * trials + e["subset_init_state_idx"] not in done
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-tasks", type=int, default=10)
    ap.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=None,
        help="Explicit task ids to shard; overrides --num-tasks. Use when other "
        "lanes still own the remaining tasks and must not be double-run.",
    )
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument(
        "--from-shard",
        type=pathlib.Path,
        default=None,
        help="Resume mode: filter this one shard against --done-dir, write the "
        "remainder to --out, and print how many are left.",
    )
    ap.add_argument("--out", type=pathlib.Path, default=None, help="Resume mode output path.")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("."))
    ap.add_argument("--prefix", default="shard")
    ap.add_argument(
        "--done-dir",
        type=pathlib.Path,
        default=None,
        help="Skip (task, episode) pairs whose HDF5 already exists here.",
    )
    args = ap.parse_args()

    done = done_episode_ids(args.done_dir) if args.done_dir else set()

    if args.from_shard is not None:
        # Single-shard resume: what is left of one lane's ownership. Used by the
        # watchdog, which must restart a crashed lane from the remainder --
        # replaying a finished episode would write a second HDF5 under the same
        # global episode id and trip the suite-handoff duplicate check.
        todo = remaining_of(args.from_shard, done, args.trials)
        if args.out is not None:
            args.out.write_text(json.dumps(todo))
        print(len(todo))
        return

    tasks = args.tasks if args.tasks is not None else list(range(args.num_tasks))
    shards = build_shards(tasks=tasks, trials=args.trials, lanes=args.lanes, done=done)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, shard in enumerate(shards):
        entries = [
            {
                "task_id": task_id,
                "subset_init_state_idx": episode_idx,
                # No remap: the unsharded run uses episode_idx as the original
                # index, and the shards must stay byte-identical to it.
                "orig_init_state_idx": episode_idx,
            }
            for task_id, episode_idx in shard
        ]
        out = args.out_dir / f"{args.prefix}_lane{i}.json"
        out.write_text(json.dumps(entries))
        task_ids = sorted({task_id for task_id, _ in shard})
        span = f"{shard[0]}..{shard[-1]}" if shard else "(empty)"
        print(f"lane{i}: {len(entries):4d} eps  tasks={task_ids}  {span}  -> {out}")

    print(f"total={sum(len(s) for s in shards)}  skipped_done={len(done)}")


if __name__ == "__main__":
    main()
