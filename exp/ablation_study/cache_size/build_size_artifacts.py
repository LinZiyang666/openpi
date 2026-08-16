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
    python: str = sys.executable,
) -> None:
    cmd = [
        python, BUILDER,
        "--data-dir", data_dir,
        "--builder-type", builder_type,
        "--output", output,
        "--episode-list", episode_list,
        "--trajectory-id-mode", "relpath",
        "--workers", str(workers),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


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
    ap.add_argument("--report", default=None, help="where to write the entry-count table")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tier in TIERS:
        listing = pathlib.Path(args.list_dir) / f"episodes_{args.suite}_{tier}.txt"
        out = out_dir / f"cache_size_{args.suite}_{tier}.pkl"
        build_one(
            data_dir=args.data_dir,
            episode_list=str(listing),
            output=str(out),
            builder_type=args.builder_type,
            workers=args.workers,
        )
        built, loaded = verify_no_entry_loss(str(out))
        if built != loaded:
            raise SystemExit(
                f"{tier}: built {built} entries but backend loaded {loaded} -- "
                "trajectory ids collided, the library is silently truncated"
            )
        rows.append({"tier": tier, "episodes": len(listing.read_text().split()),
                     "entries": built, "pkl": str(out)})
        print(f"  {tier}: {built} entries OK", flush=True)

    table = {"suite": args.suite, "tiers": rows}
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
