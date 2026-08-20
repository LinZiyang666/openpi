"""Drive the per-tier library builds and record the realized entry counts.

Twelve builds (6 tiers x 2 suites), each fed an explicit episode list so the
nested subsets share one collected corpus instead of twelve copies of it.

Two flags are non-negotiable here:

*   ``--trajectory-id-mode relpath`` -- the collected layout is
    ``task_N/episode_M.h5``, whose stems repeat across tasks. Under the default
    ``stem`` mode all ten tasks' ``episode_0`` collapse onto one id and the
    backend silently keeps only the last one.
*   ``--workers`` pinned low -- the corpus lives on a spinning disk and the
    default (all CPUs) turns the read into many interleaved seek streams.

After each build the artifact is loaded back through ``InMemoryBackend`` and the
entry count is compared against what was built. That equality is the general
probe for id collisions: both the build log and the load log report pre-dedup
counts, so a collision is invisible in either one alone.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import subprocess
import sys

BUILDER = "exp/common/build_in_memory_cache_artifact.py"
TIERS = ("S1", "S2", "S3", "S4", "S5", "S6")


def build_one(
    *,
    data_dir: str,
    episode_list: str,
    output: str,
    builder_type: str,
    workers: int,
    outcome_filter: str,
    python: str = sys.executable,
) -> None:
    cmd = [
        python, BUILDER,
        "--data-dir", data_dir,
        "--builder-type", builder_type,
        "--output", output,
        "--episode-list", episode_list,
        "--trajectory-id-mode", "relpath",
        "--outcome-filter", outcome_filter,
        "--workers", str(workers),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def verify_nesting(out_dir: str, prefix: str) -> list[str]:
    """Every tier's entries must be a subset of the next tier's. Returns violations.

    Nesting is what makes the size axis a *within-library* comparison: S4 is S3
    plus more, not a differently-sampled library of its own. Without it, each
    adjacent-tier test would confound "bigger" with "different trajectories", and
    the whole nested design collapses into six unrelated draws.

    A topped-out tier may equal its successor (``<=``, not ``<``) -- that is R1
    doing its job, not a defect.
    """
    ids: dict[str, set[str]] = {}
    for tier in TIERS:
        with open(pathlib.Path(out_dir) / f"{prefix}_{tier}.pkl", "rb") as f:
            ids[tier] = {e.id for e in pickle.load(f)["entries"]}
    violations = []
    for lo, hi in zip(TIERS, TIERS[1:]):
        orphans = ids[lo] - ids[hi]
        if orphans:
            violations.append(
                f"{lo} is not a subset of {hi}: {len(orphans)} entries present in "
                f"{lo} but absent from {hi} (e.g. {sorted(orphans)[:3]})"
            )
    return violations


def verify_list_coverage(pkl_path: str, episode_list: str) -> tuple[int, int]:
    """Every listed episode must appear in the artifact. Returns (listed, present).

    This is the guard that does not care *why* an episode went missing. The
    builder has its own ``--outcome-filter``, and the episode list is produced by
    a grid that has one too; when they disagree the builder quietly drops the
    episodes the list deliberately included, and the library ends up smaller than
    the tier it claims with no error raised anywhere. Same protection against a
    listed-but-unreadable file, or a path that normalized to something else.
    """
    with open(pkl_path, "rb") as f:
        artifact = pickle.load(f)
    present = {e.trajectory_id for e in artifact["entries"]}
    listed = {line.strip().removesuffix(".h5")
              for line in pathlib.Path(episode_list).read_text().splitlines()
              if line.strip()}
    return len(listed), len(listed & present)


def verify_no_entry_loss(pkl_path: str) -> tuple[int, int]:
    """Return (built, loaded); they must be equal or ids collided.

    ``vector_dims`` is read back from the artifact rather than hardcoded, so a
    non-default ``--builder-type`` does not fail with a confusing dimension
    mismatch raised from inside the backend.
    """
    from openpi.cache.backends.in_memory_backend import InMemoryBackend

    with open(pkl_path, "rb") as f:
        artifact = pickle.load(f)
    built = len(artifact["entries"])
    vector_dims = artifact["vector_dims"]

    backend = InMemoryBackend(vector_dims=vector_dims)
    backend.load_artifact(pkl_path)
    loaded = len(backend._entries)
    return built, loaded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--data-dir", required=True,
                    help="<collect_dir>/<suite> -- must include the suite level")
    ap.add_argument("--list-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--builder-type", default="cp1_spatial_pool_16")
    ap.add_argument("--workers", type=int, default=4,
                    help="kept small on purpose: the corpus is on a spinning disk")
    ap.add_argument("--outcome-filter", default="success", choices=["success", "failure", "all"],
                    help="MUST match the --outcome-filter the size grid was emitted "
                         "with; a mismatch silently shrinks every library below the "
                         "tier it claims. Checked by verify_list_coverage.")
    ap.add_argument("--pkl-prefix", default=None,
                    help="artifact basename prefix; defaults to cache_size_<suite>. "
                         "Use it to keep two outcome-filter groups side by side.")
    ap.add_argument("--report", default=None, help="where to write the entry-count table")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tier in TIERS:
        prefix = args.pkl_prefix or f"cache_size_{args.suite}"
        listing = pathlib.Path(args.list_dir) / f"episodes_{args.suite}_{tier}.txt"
        out = out_dir / f"{prefix}_{tier}.pkl"
        build_one(
            data_dir=args.data_dir,
            episode_list=str(listing),
            output=str(out),
            builder_type=args.builder_type,
            workers=args.workers,
            outcome_filter=args.outcome_filter,
        )
        built, loaded = verify_no_entry_loss(str(out))
        if built != loaded:
            raise SystemExit(
                f"{tier}: built {built} entries but backend loaded {loaded} -- "
                "trajectory ids collided, the library is silently truncated"
            )
        listed, present = verify_list_coverage(str(out), str(listing))
        if listed != present:
            raise SystemExit(
                f"{tier}: {listed} episodes listed but only {present} made it into the "
                f"artifact ({listed - present} dropped). The usual cause is an "
                f"--outcome-filter mismatch: this build used {args.outcome_filter!r} "
                "while the grid that wrote the list used something else."
            )
        rows.append({"tier": tier, "episodes": listed, "entries": built,
                     "pkl": str(out)})
        print(f"  {tier}: {built} entries from {listed} episodes OK", flush=True)

    prefix = args.pkl_prefix or f"cache_size_{args.suite}"
    violations = verify_nesting(str(out_dir), prefix)
    if violations:
        raise SystemExit(
            "nested-library gate failed; the size axis would confound 'bigger' with "
            "'different trajectories':\n  " + "\n  ".join(violations)
        )
    print(f"  nesting OK across {len(TIERS)} tiers", flush=True)

    table = {"suite": args.suite, "outcome_filter": args.outcome_filter,
             "nested": True, "tiers": rows}
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
