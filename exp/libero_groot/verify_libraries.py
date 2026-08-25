"""Post-hoc gate on a built size-tier library set.

``build_size_libraries.py`` checks structure in-process, which cannot catch the
one failure that matters at load time: the backend keys entries by id, so any
id collision silently drops entries and the library is quietly smaller than the
manifest says. Both the build log and the load log report pre-dedup counts, so
the collision is invisible in either one alone -- only the comparison exposes it
(X9b's general probe, kept here for the same reason).

Also re-checks nesting across the tier files as they landed on disk, rather than
trusting the in-memory sets the slicer held.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle


def load_ids(path: pathlib.Path) -> tuple[set[str], set[str], dict]:
    """(entry ids, trajectory ids, artifact metadata) for one tier file."""
    with path.open("rb") as f:
        artifact = pickle.load(f)
    entries = artifact["entries"]
    meta = {k: v for k, v in artifact.items() if k != "entries"}
    return {e.id for e in entries}, {e.trajectory_id for e in entries}, meta | {"n": len(entries)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=pathlib.Path)
    ap.add_argument("--skip-backend", action="store_true",
                    help="Skip the InMemoryBackend round trip (structure checks only).")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    print(f"{manifest['prefix']}  builder={manifest['builder_type']}")
    print(f"vector_dims={manifest['vector_dims']}")

    failures: list[str] = []
    prev_traj: set[str] | None = None
    prev_tier = ""
    for record in manifest["tiers"]:
        path = pathlib.Path(record["path"])
        tier = record["tier"]
        if not path.exists():
            failures.append(f"{tier}: missing file {path}")
            continue
        ids, trajs, meta = load_ids(path)

        if len(ids) != record["entries"]:
            failures.append(
                f"{tier}: {record['entries']} entries built but only {len(ids)} unique ids "
                "-- an id collision would drop the difference at load time"
            )
        if meta["key_builder_type"] != manifest["builder_type"]:
            failures.append(f"{tier}: stamped {meta['key_builder_type']!r}, manifest says "
                            f"{manifest['builder_type']!r}")
        if meta["vector_dims"] != manifest["vector_dims"]:
            failures.append(f"{tier}: vector_dims drift {meta['vector_dims']}")
        if len(trajs) != record["trajectories"]:
            failures.append(f"{tier}: {record['trajectories']} trajectories expected, {len(trajs)} present")
        if prev_traj is not None and not prev_traj <= trajs:
            failures.append(f"{tier}: not a superset of {prev_tier} "
                            f"({len(prev_traj - trajs)} trajectories lost)")

        loaded = ""
        if not args.skip_backend:
            from openpi.cache.backends.in_memory_backend import InMemoryBackend

            backend = InMemoryBackend(vector_dims=meta["vector_dims"])
            backend.load_artifact(str(path))
            count = backend.count()
            loaded = f" backend={count}"
            if count != record["entries"]:
                failures.append(
                    f"{tier}: backend loaded {count} of {record['entries']} entries "
                    "-- entries were silently deduped away"
                )

        print(f"  {tier}: entries={len(ids)} trajectories={len(trajs)}{loaded}  ok")
        prev_traj, prev_tier = trajs, tier

    if failures:
        print("\nFAILURES:")
        for line in failures:
            print(f"  - {line}")
        raise SystemExit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
