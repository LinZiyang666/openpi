"""Build one full cache library per suite, then slice it into nested size tiers.

Following the X9b cache-size design (``exp/ablation_study/cache_size``) with one
change: that experiment ran a separate build per tier off an explicit episode
list, which re-reads the corpus once per tier. Here the corpus is 89 GB of HDF5,
so the expensive pass runs **once** and the tiers are cut out of the resulting
artifact. Nesting is then true by construction rather than by verification.

Two invariants the slicing rests on:

*   **The slice unit is a whole episode.** ``prev_ids``/``next_ids`` link entries
    within one episode; cutting at step granularity would leave dangling
    references that no consumer checks.
*   **The axis is successful trajectories per task, not episodes sampled.**
    Retrieval is task-scoped (``search_strategy`` builds a
    ``QueryFilter(task_key=...)``), so a task with zero entries in a tier returns
    no candidates and silently falls back to the teacher for every one of its
    evaluation episodes. Low-success tasks are exactly the ones at risk, and
    coverage improves with size, so an episode-sampled axis would bend the curve
    in the direction that flatters the expected conclusion.

Per-task count in tier k is ``min(k, n_t)`` (X9b R1), so a task that tops out
early stops growing while the others keep going; the reported x-axis is the
realized mean, never the nominal k (R4).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import pickle
import random
import re
import subprocess
import sys

_EPISODE_RE = re.compile(r"episode_(\d+)_")
DEFAULT_TIERS: tuple[int, ...] = (1, 2, 5, 10, 20, 50)
SHUFFLE_SEED = 0
BUILDER = "exp/common/build_in_memory_cache_artifact.py"


def episode_identity(trajectory_id: str, trials: int) -> tuple[int, int]:
    """(task_id, init_state_idx) from a collected episode's trajectory id.

    The collector drops the client's ``extra_metadata``, so the HDF5 carries no
    ``orig_init_state_idx``; the global episode id in the filename is the only
    surviving coordinate. ``episode_id = task_id * trials + init_idx`` is the
    single source of that formula (``examples/libero/collect_util.py``).
    """
    match = _EPISODE_RE.search(trajectory_id)
    if match is None:
        raise ValueError(f"trajectory id does not carry an episode number: {trajectory_id!r}")
    episode_id = int(match.group(1))
    return episode_id // trials, episode_id % trials


def build_full(*, data_dir: str, builder_type: str, out: pathlib.Path, workers: int) -> None:
    cmd = [
        sys.executable, BUILDER,
        "--data-dir", data_dir,
        "--builder-type", builder_type,
        "--output", str(out),
        "--outcome-filter", "success",
        "--workers", str(workers),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def order_episodes(
    trajectories: dict[str, list], trials: int, seed: int
) -> dict[int, list[str]]:
    """Per task, the trajectory ids in the order tiers consume them.

    Deterministic shuffle rather than index order: the B-pool init indices carry
    no meaning of their own, and taking them in order would tie tier membership
    to whatever ordering the pool happened to be written in.
    """
    by_task: dict[int, list[str]] = collections.defaultdict(list)
    for traj_id in trajectories:
        task_id, _ = episode_identity(traj_id, trials)
        by_task[task_id].append(traj_id)
    rng = random.Random(seed)
    for task_id in by_task:
        by_task[task_id].sort()  # stable input before the shuffle
        rng.shuffle(by_task[task_id])
    return dict(by_task)


def verify_tier(entries: list, kept: set[str]) -> list[str]:
    """Structural checks on one sliced tier. Returns human-readable violations."""
    problems = []
    ids = {e.id for e in entries}
    if len(ids) != len(entries):
        problems.append(f"duplicate entry ids: {len(entries)} entries, {len(ids)} unique")
    for entry in entries:
        if entry.trajectory_id not in kept:
            problems.append(f"entry {entry.id} belongs to an unselected trajectory")
            break
    # prev/next stay inside the tier because whole episodes are kept; a dangling
    # reference means the slice cut inside an episode.
    for entry in entries:
        for ref in (entry.prev_ids or []) + (entry.next_ids or []):
            if ref not in ids:
                problems.append(f"dangling link {entry.id} -> {ref}")
                break
        else:
            continue
        break
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="Directory of collected .h5 episodes")
    ap.add_argument("--builder-type", required=True)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--prefix", required=True, help="Artifact name stem, e.g. libero_spatial")
    ap.add_argument("--tiers", type=int, nargs="+", default=list(DEFAULT_TIERS))
    ap.add_argument("--trials", type=int, default=50, help="num_trials_per_task at collection")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SHUFFLE_SEED)
    ap.add_argument("--skip-build", action="store_true", help="Reuse an existing _full.pkl")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.out_dir / f"{args.prefix}_full.pkl"
    if not args.skip_build or not full_path.exists():
        build_full(
            data_dir=args.data_dir, builder_type=args.builder_type,
            out=full_path, workers=args.workers,
        )

    print(f"loading {full_path}", flush=True)
    with full_path.open("rb") as f:
        artifact = pickle.load(f)
    entries = artifact["entries"]

    by_traj: dict[str, list] = collections.defaultdict(list)
    for entry in entries:
        by_traj[entry.trajectory_id].append(entry)
    print(f"full artifact: {len(entries)} entries over {len(by_traj)} trajectories")

    ordered = order_episodes(by_traj, args.trials, args.seed)
    n_per_task = {t: len(v) for t, v in sorted(ordered.items())}
    print(f"successful trajectories per task: {n_per_task}")
    empty = [t for t, n in n_per_task.items() if n == 0]
    if empty:
        # X9b R2: a task with no entries makes retrieval fall back to the teacher
        # for that whole task, which is a silent hole in every tier.
        raise SystemExit(f"tasks with zero successful trajectories: {empty}")

    manifest = {
        "prefix": args.prefix,
        "builder_type": artifact["key_builder_type"],
        "vector_dims": artifact["vector_dims"],
        "seed": args.seed,
        "trajectories_per_task": n_per_task,
        "tiers": [],
    }
    prev_kept: set[str] | None = None
    for i, k in enumerate(args.tiers, start=1):
        tier = f"S{i}"
        kept = {
            traj_id
            for task_id, traj_ids in ordered.items()
            for traj_id in traj_ids[: min(k, len(traj_ids))]
        }
        tier_entries = [e for e in entries if e.trajectory_id in kept]
        problems = verify_tier(tier_entries, kept)
        if prev_kept is not None and not prev_kept <= kept:
            problems.append(f"{tier} is not a superset of the previous tier")
        if problems:
            raise SystemExit(f"{tier} failed verification: {problems}")

        out = args.out_dir / f"{args.prefix}_{tier}.pkl"
        with out.open("wb") as f:
            pickle.dump({**artifact, "entries": tier_entries}, f, protocol=4)
        realized = {t: min(k, len(v)) for t, v in ordered.items()}
        mean = sum(realized.values()) / len(realized)
        size_mb = out.stat().st_size // (1024 * 1024)
        print(
            f"{tier}: nominal k={k:<3} realized mean={mean:5.1f}/task  "
            f"trajectories={len(kept):<5} entries={len(tier_entries):<7} {size_mb} MB  -> {out.name}"
        )
        manifest["tiers"].append({
            "tier": tier, "nominal_k": k, "realized_per_task": realized,
            "realized_mean": mean, "trajectories": len(kept),
            "entries": len(tier_entries), "size_mb": size_mb, "path": str(out),
        })
        prev_kept = kept

    manifest_path = args.out_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
