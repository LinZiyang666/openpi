"""Slice a RoboCasa365 full library into nested size tiers.

Same design as ``exp/libero_groot/build_size_libraries.py`` (X9b cache-size
axis): the unit is a whole episode, the axis is successful trajectories per
task, tier k keeps ``min(k, n_t)`` per task, the per-task order is a seed-0
deterministic shuffle of the sorted trajectory ids, and every tier is verified
(unique ids, no dangling prev/next links, superset of the previous tier).

What differs is only how a trajectory maps to its task: LIBERO recovers the
task from the global episode number, while a RoboCasa library built with
``--trajectory-id-mode relpath`` carries it in the id
(``<teacher>/<Task>/episode_NNNN_aAA``). The full artifact is built beforehand
by ``exp/common/build_in_memory_cache_artifact.py`` from the merged audit
manifest; this script never rebuilds it.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import pickle
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from exp.libero_groot.build_size_libraries import DEFAULT_TIERS, SHUFFLE_SEED, verify_tier  # noqa: E402


def task_of(trajectory_id: str) -> str:
    parts = trajectory_id.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"trajectory id {trajectory_id!r} is not <teacher>/<Task>/<episode>; "
            "build the full library with --trajectory-id-mode relpath"
        )
    return parts[-2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", required=True, type=pathlib.Path, help="full artifact .pkl")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tiers", type=int, nargs="+", default=list(DEFAULT_TIERS))
    ap.add_argument("--seed", type=int, default=SHUFFLE_SEED)
    ap.add_argument("--denoise-schedule", required=True, help="expected artifact schedule id")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.full.open("rb") as f:
        artifact = pickle.load(f)
    entries = artifact["entries"]
    if artifact.get("schedule_id") != args.denoise_schedule:
        raise SystemExit(
            f"full artifact schedule {artifact.get('schedule_id')!r} != requested {args.denoise_schedule!r}"
        )
    by_traj: dict[str, list] = collections.defaultdict(list)
    for e in entries:
        by_traj[e.trajectory_id].append(e)
    by_task: dict[str, list[str]] = collections.defaultdict(list)
    for traj_id in by_traj:
        by_task[task_of(traj_id)].append(traj_id)
    rng = random.Random(args.seed)
    for task in sorted(by_task):
        by_task[task].sort()
        rng.shuffle(by_task[task])
    n_per_task = {t: len(v) for t, v in sorted(by_task.items())}
    print(f"full artifact: {len(entries)} entries over {len(by_traj)} trajectories")
    print(f"successful trajectories per task: {n_per_task}")

    manifest = {
        "prefix": args.prefix,
        "builder_type": artifact["key_builder_type"],
        "vector_dims": artifact["vector_dims"],
        "schedule_id": args.denoise_schedule,
        "pin_id": artifact.get("pin_id"),
        "seed": args.seed,
        "full": str(args.full),
        "trajectories_per_task": n_per_task,
        "tiers": [],
    }
    prev_kept: set[str] | None = None
    for i, k in enumerate(args.tiers, start=1):
        tier = f"S{i}"
        kept = {tid for ids in by_task.values() for tid in ids[: min(k, len(ids))]}
        tier_entries = [e for e in entries if e.trajectory_id in kept]
        problems = verify_tier(tier_entries, kept)
        if prev_kept is not None and not prev_kept <= kept:
            problems.append(f"{tier} is not a superset of the previous tier")
        if problems:
            raise SystemExit(f"{tier} failed verification: {problems}")
        out = args.out_dir / f"{args.prefix}_{tier}.pkl"
        with out.open("wb") as f:
            pickle.dump({**artifact, "entries": tier_entries}, f, protocol=4)
        realized = {t: min(k, len(v)) for t, v in sorted(by_task.items())}
        mean = sum(realized.values()) / len(realized)
        size_mb = out.stat().st_size // (1024 * 1024)
        print(
            f"{tier}: nominal k={k:<3} realized mean={mean:5.1f}/task  "
            f"trajectories={len(kept):<5} entries={len(tier_entries):<7} {size_mb} MB  -> {out.name}"
        )
        manifest["tiers"].append(
            {
                "tier": tier,
                "nominal_k": k,
                "realized_per_task": realized,
                "realized_mean": mean,
                "trajectories": len(kept),
                "entries": len(tier_entries),
                "size_mb": size_mb,
                "path": str(out),
            }
        )
        prev_kept = kept
    manifest_path = args.out_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
