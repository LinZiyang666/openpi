"""Per-task success rates and coverage for a LIBERO collection run.

The authoritative record is the HDF5 itself: the collector stamps ``success``,
``episode_id`` and ``task`` onto every episode file, so a run survives lanes
being killed and re-sharded mid-flight — which the ``results_lane*.json``
files do not, since ``main.py`` only writes them after its loop finishes.

``episode_id = task_id * trials + episode_idx`` (``collect_util``), so task id
and init-state index are recoverable from the filename alone; the HDF5 is
opened only for ``success`` and step count.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

import h5py

_EPISODE_RE = re.compile(r"episode_(\d+)_")


def scan(root: pathlib.Path, trials: int) -> tuple[dict, list[tuple[int, int]]]:
    """Return per-task stats and the list of duplicate (episode_id, count) pairs."""
    seen: dict[int, list[pathlib.Path]] = collections.defaultdict(list)
    for path in sorted(root.rglob("episode_*.h5")):
        match = _EPISODE_RE.match(path.name)
        if match:
            seen[int(match.group(1))].append(path)

    stats: dict[int, dict] = collections.defaultdict(
        lambda: {"n": 0, "ok": 0, "steps": 0, "task": "", "idx": set()}
    )
    for episode_id, paths in seen.items():
        # Duplicates can only come from a re-run; the newest file is the one
        # whose episode actually completed last, so prefer it.
        path = sorted(paths)[-1]
        try:
            with h5py.File(path, "r") as f:
                success = bool(f.attrs.get("success", False))
                steps = int(f.attrs.get("num_steps", len(f.keys())))
                task_name = str(f.attrs.get("task", ""))
        except OSError as exc:  # truncated file from a kill mid-write
            print(f"  ! unreadable {path.name}: {exc}")
            continue
        task_id = episode_id // trials
        entry = stats[task_id]
        entry["n"] += 1
        entry["ok"] += int(success)
        entry["steps"] += steps
        entry["idx"].add(episode_id % trials)
        entry["task"] = entry["task"] or task_name

    dups = [(eid, len(p)) for eid, p in seen.items() if len(p) > 1]
    return stats, sorted(dups)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--num-tasks", type=int, default=10)
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    ap.add_argument(
        "--prune-duplicates",
        action="store_true",
        help="Delete all but the newest file for each duplicated episode id. "
        "Duplicates mean an episode was replayed (a lane restarted from its "
        "original shard instead of the remainder); both files are complete "
        "runs of the same init state, so keeping the newest is arbitrary but "
        "consistent with how this report already reads them.",
    )
    args = ap.parse_args()

    if args.prune_duplicates:
        seen: dict[int, list[pathlib.Path]] = collections.defaultdict(list)
        for path in sorted(args.root.rglob("episode_*.h5")):
            match = _EPISODE_RE.match(path.name)
            if match:
                seen[int(match.group(1))].append(path)
        removed = 0
        for episode_id, paths in sorted(seen.items()):
            if len(paths) < 2:
                continue
            for stale in sorted(paths)[:-1]:
                stale.unlink()
                removed += 1
            print(f"  pruned episode {episode_id}: kept {sorted(paths)[-1].name}")
        print(f"pruned {removed} duplicate file(s)")

    stats, dups = scan(args.root, args.trials)

    total_n = total_ok = 0
    rows = []
    print(f"{'task':>4}  {'n':>4}  {'ok':>4}  {'SR':>6}  {'steps/ep':>8}  description")
    for task_id in range(args.num_tasks):
        entry = stats.get(task_id)
        if entry is None:
            print(f"{task_id:>4}  {0:>4}  {0:>4}  {'--':>6}  {'--':>8}  (no episodes)")
            rows.append({"task_id": task_id, "n": 0, "ok": 0, "missing": list(range(args.trials))})
            continue
        n, ok = entry["n"], entry["ok"]
        total_n += n
        total_ok += ok
        missing = sorted(set(range(args.trials)) - entry["idx"])
        print(
            f"{task_id:>4}  {n:>4}  {ok:>4}  {ok / n:>6.3f}  {entry['steps'] / n:>8.1f}  "
            f"{entry['task'][:60]}"
        )
        if missing:
            print(f"       missing init-state idx ({len(missing)}): {missing[:20]}")
        rows.append(
            {
                "task_id": task_id,
                "n": n,
                "ok": ok,
                "sr": ok / n,
                "task": entry["task"],
                "missing": missing,
            }
        )

    expected = args.num_tasks * args.trials
    print(
        f"\nTOTAL {total_ok}/{total_n} = {total_ok / total_n:.3f} macro-flat"
        if total_n
        else "\nTOTAL 0 episodes"
    )
    print(f"coverage {total_n}/{expected}   duplicate episode ids: {len(dups)}")
    if dups:
        print(f"  dups (episode_id, files): {dups[:20]}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {"root": str(args.root), "tasks": rows, "n": total_n, "ok": total_ok, "dups": dups},
                indent=2,
            )
        )
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
