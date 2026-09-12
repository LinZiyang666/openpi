"""Bind evaluation inits to original collection sets using actual state bytes.

This module freezes seen/unseen membership before outcomes exist. Small-source
identity uses its complete collection map; large-source identity uses the
pre-success S6/all list. Success filtering and pruning never change membership.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import re

import h5py
import numpy as np

from exp.ablation_study.cache_size.verify_apool import load_init_states

from .common import (
    SUITES,
    VERSION,
    check_identity,
    check_seal,
    file_identity,
    read_json,
    require,
    seal,
    task_prompt,
    write_json,
)


def state_ids(path: str | Path) -> list[str]:
    """Hash each init's canonical float64 bytes while retaining its pool index."""
    states = np.asarray(load_init_states(Path(path)))
    require(
        states.ndim == 2 and states.shape[0] > 0 and states.shape[1] > 0,
        "invalid init array",
    )
    require(bool(np.isfinite(states).all()), "nonfinite init state")
    return [
        hashlib.sha256(
            np.ascontiguousarray(row, dtype=np.float64).tobytes()
        ).hexdigest()
        for row in states
    ]


def checked_apool(path: str | Path, suite: str) -> dict:
    """Rehash the exact worker init directory and reject .pruned_init overrides."""
    from exp.ablation_study.cache_size.run_size_eval import load_apool_digest

    record = load_apool_digest(str(path), required=True, verify_contents=True)
    require(record["suite"] == suite, "A-pool suite mismatch")
    root = Path(record["apool_dir"])
    require(
        not list(root.glob("*.pruned_init")),
        "A-pool has higher-priority .pruned_init overrides",
    )
    for name in record["per_task_digests"]:
        ids = state_ids(root / f"{name}.init")
        require(
            len(ids) == 50 and len(set(ids)) == 50,
            "A-pool needs 50 distinct states per task",
        )
    return record


def prepare_membership(
    suite: str,
    apool_record: str | Path,
    source_manifests: list[dict],
    *,
    repository: str | Path,
    large_collection: str | Path,
    difference_pool: str | Path,
) -> dict:
    """Freeze all 500 identities against the two original source collection sets."""
    require(suite in SUITES, "unsupported suite")
    require(
        len(source_manifests) == 2, "membership requires exactly two source regimes"
    )
    sources = {}
    for source in source_manifests:
        check_seal(source)
        require(
            source["suite"] == suite and source["regime"] not in sources,
            "source suite/regime mismatch",
        )
        sources[source["regime"]] = source
    require(set(sources) == {"rit50", "cs500_success"}, "missing source regime")
    root, collection, difference_pool = (
        Path(repository).resolve(),
        Path(large_collection),
        Path(difference_pool),
    )
    apool = checked_apool(apool_record, suite)
    roster_path = root / f"exp/rit_pareto/config/task_order_{suite}.json"
    roster = read_json(roster_path)
    tasks = {int(k): v["task_name"] for k, v in roster["assignment"].items()}
    require(
        set(tasks) == set(range(10))
        and set(tasks.values()) == set(apool["per_task_digests"]),
        "membership task roster mismatch",
    )
    evidence = [file_identity(apool_record), file_identity(roster_path)]
    cached_ids = {}

    def ids_at(path):
        path = Path(path).resolve(strict=True)
        if path not in cached_ids:
            cached_ids[path] = state_ids(path)
            evidence.append(file_identity(path))
        return cached_ids[path]

    original_sets = {regime: {task: set() for task in tasks} for regime in sources}
    collection_rows = {regime: [] for regime in sources}
    init_map = Path(sources["rit50"]["expected_census"]["init_map"])
    records = read_json(init_map)
    evidence.append(file_identity(init_map))
    require(bool(records), "empty small collection map")
    compact = all("trajectory_id" not in record for record in records)
    require(
        compact or all("trajectory_id" in record for record in records),
        "mixed small collection map schemas",
    )
    small_ids = set()
    collection_directories = set()
    mapped_paths = set()
    if compact:
        # Owner accepts the declared per-task collection set, with exact pool
        # bytes and H5 task census. Never infer individual init IDs from H5 order.
        directory = root / f"exp/common/data/db/libero_cache/{suite}"
        collection_directories.add(directory)
        h5_tasks = Counter()
        for h5_path in sorted(directory.glob("*.h5")):
            mapped_paths.add(h5_path.resolve(strict=True))
            small_ids.add(h5_path.stem)
            with h5py.File(h5_path, "r") as handle:
                require(
                    str(handle.attrs["experiment_name"]) == suite,
                    "small collection suite mismatch",
                )
                h5_tasks[str(handle.attrs["task"])] += 1
            evidence.append(file_identity(h5_path))
        require(
            h5_tasks == Counter(record["prompt"] for record in records),
            "compact collection/HDF5 task census mismatch",
        )
        require(
            len({(r["task_id"], r["subset_idx"]) for r in records}) == len(records),
            "duplicate compact init identity",
        )
        records = [
            dict(
                record,
                suite=suite,
                subset_init_state_idx=record["subset_idx"],
                init_path=f"exp/common/data/db_init/libero_cache/{suite}/{record['task_name']}.init",
                full_init_path=f"exp/common/data/db_init/libero/{suite}/{record['task_name']}.init",
            )
            for record in records
        ]
    else:
        require(
            len(records) == len({r["trajectory_id"] for r in records}),
            "duplicate small collection identity",
        )
    for record in records:
        task = record["task_id"]
        require(
            task in tasks
            and record["suite"] == suite
            and record["task_name"] == tasks[task]
            and record["prompt"] == task_prompt(tasks[task]),
            "small init-map task mismatch",
        )
        if not compact:
            h5_path = (root / record["h5_path"]).resolve(strict=True)
            collection_directories.add(h5_path.parent)
            mapped_paths.add(h5_path)
            with h5py.File(h5_path, "r") as handle:
                require(
                    str(handle.attrs["task"]) == record["prompt"],
                    "small collection task mismatch",
                )
            evidence.append(file_identity(h5_path))
            small_ids.add(record["trajectory_id"])
        subset_path, full_path = (
            root / record["init_path"],
            root / record["full_init_path"],
        )
        subset, full = ids_at(subset_path), ids_at(full_path)
        i, j = record["subset_init_state_idx"], record["orig_init_state_idx"]
        require(
            type(i) is int
            and type(j) is int
            and 0 <= i < len(subset)
            and 0 <= j < len(full),
            "small init index outside actual pool",
        )
        require(
            subset[i] == full[j],
            "small source init map disagrees with actual state bytes",
        )
        original_sets["rit50"][task].add(subset[i])
        collection_rows["rit50"].append(
            {
                **(
                    {"identity_basis": "declared_collection_init_set"}
                    if compact
                    else {"trajectory_id": record["trajectory_id"]}
                ),
                "task_id": task,
                "state_sha256": subset[i],
                "pool_index": i,
                "original_index": j,
                "pool": str(subset_path.resolve()),
            }
        )
    # Extra collection files cannot silently become unknown/unseen identities.
    actual_paths = {
        p.resolve()
        for directory in collection_directories
        for p in directory.glob("*.h5")
    }
    require(
        actual_paths == mapped_paths, "small collection has unmapped HDF5 trajectories"
    )
    if compact:
        for task, name in tasks.items():
            indices = sorted(
                r["pool_index"]
                for r in collection_rows["rit50"]
                if r["task_id"] == task
            )
            require(
                indices
                == list(
                    range(
                        len(
                            ids_at(
                                root
                                / f"exp/common/data/db_init/libero_cache/{suite}/{name}.init"
                            )
                        )
                    )
                ),
                "compact map does not cover the complete declared subset",
            )
    require(
        {r["trajectory_id"] for r in sources["rit50"]["rows"]} <= small_ids,
        "small source absent from original collection",
    )
    all_list = (
        root / f"exp/ablation_study/cache_size/config/lists_all/episodes_{suite}_S6.txt"
    )
    lines = all_list.read_text().splitlines()
    require(
        len(lines) == 450 and len(set(lines)) == 450,
        "S6/all collection identity must cover 450 distinct B-train inits",
    )
    evidence.append(file_identity(all_list))
    large_ids = set()
    for line in lines:
        match = re.fullmatch(r"task_(\d+)/episode_(\d+)\.h5", line)
        require(match is not None, "invalid S6/all relative path")
        task, index = map(int, match.groups())
        require(task in tasks, "unexpected large-source task")
        pool_path = difference_pool / f"{tasks[task]}.init"
        require(
            not pool_path.with_suffix(".pruned_init").exists(),
            "difference pool has an override",
        )
        pool = ids_at(pool_path)
        require(
            len(pool) == 50 and 0 <= index < 50,
            "large init index outside 50-state difference pool",
        )
        h5_path = collection / line
        with h5py.File(h5_path, "r") as handle:
            require(
                str(handle.attrs["task"]) == task_prompt(tasks[task]),
                "large collection task mismatch",
            )
            require(
                "step_0000" in handle
                or any(name.startswith("step_") for name in handle),
                "large collection has no cached steps",
            )
        evidence.append(file_identity(h5_path))
        original_sets["cs500_success"][task].add(pool[index])
        large_ids.add(line.removesuffix(".h5"))
        collection_rows["cs500_success"].append(
            {
                "trajectory_id": line.removesuffix(".h5"),
                "task_id": task,
                "state_sha256": pool[index],
                "pool_index": index,
                "pool": str(pool_path.resolve()),
            }
        )
    require(
        Counter(r["task_id"] for r in collection_rows["cs500_success"])
        == Counter({i: 45 for i in tasks}),
        "S6/all task census mismatch",
    )
    require(
        {r["trajectory_id"] for r in sources["cs500_success"]["rows"]} <= large_ids,
        "large success source absent from original S6/all collection",
    )
    rows = []
    for task, name in tasks.items():
        require(
            all(original_sets[regime][task] for regime in sources),
            "missing source init identity for a task",
        )
        a_ids = ids_at(Path(apool["apool_dir"]) / f"{name}.init")
        b_ids = ids_at(difference_pool / f"{name}.init")
        require(not set(a_ids) & set(b_ids), "A-pool overlaps the difference pool")
        for index, state_id in enumerate(a_ids):
            seen = {
                regime: state_id in original_sets[regime][task] for regime in sources
            }
            rows.append(
                {
                    "suite": suite,
                    "task_id": task,
                    "task_key": task_prompt(name),
                    "orig_init_state_idx": index,
                    "state_sha256": state_id,
                    "seen": seen,
                    "common_unseen": not any(seen.values()),
                }
            )
        require(
            any(r["common_unseen"] for r in rows if r["task_id"] == task),
            "task has no common-unseen inits",
        )
    for identity in evidence:
        check_identity(identity)
    return seal(
        {
            "version": VERSION,
            "kind": "membership",
            "suite": suite,
            "source_digests": {
                regime: source["digest"] for regime, source in sources.items()
            },
            "apool": apool,
            "apool_record": file_identity(apool_record),
            "evidence": evidence,
            "original_collections": collection_rows,
            "rows": rows,
            "common_unseen_count": sum(r["common_unseen"] for r in rows),
            "complete": True,
        }
    )


def validate_membership(manifest: dict, *, rehash: bool = True) -> None:
    """Check exact evaluation coverage and reproducible frozen subset membership."""
    check_seal(manifest)
    require(
        manifest["kind"] == "membership" and manifest["complete"] is True,
        "incomplete membership",
    )
    rows = manifest["rows"]
    require(
        len(rows) == 500
        and {(r["task_id"], r["orig_init_state_idx"]) for r in rows}
        == {(t, i) for t in range(10) for i in range(50)},
        "membership grid mismatch",
    )
    sets = {
        regime: {(r["task_id"], r["state_sha256"]) for r in values}
        for regime, values in manifest["original_collections"].items()
    }
    require(
        set(sets) == {"rit50", "cs500_success"},
        "membership lacks original collection regimes",
    )
    for row in rows:
        seen = {
            regime: (row["task_id"], row["state_sha256"]) in values
            for regime, values in sets.items()
        }
        require(
            row["seen"] == seen and row["common_unseen"] is (not any(seen.values())),
            "membership differs from source states",
        )
    require(
        manifest["common_unseen_count"] == sum(r["common_unseen"] for r in rows),
        "membership count mismatch",
    )
    require(
        all(
            any(r["task_id"] == t and r["common_unseen"] for r in rows)
            for t in range(10)
        ),
        "empty common-unseen task",
    )
    if rehash:
        for identity in manifest["evidence"]:
            check_identity(identity)
        check_identity(manifest["apool_record"])
        require(
            checked_apool(manifest["apool_record"]["path"], manifest["suite"])
            == manifest["apool"],
            "A-pool changed since membership freeze",
        )


def main() -> None:
    """Prepare one suite's membership from both audited source manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=SUITES)
    parser.add_argument("--sources", type=Path, nargs=2, required=True)
    for name in ("apool-record", "large-collection", "difference-pool", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = prepare_membership(
        args.suite,
        args.apool_record,
        [read_json(p) for p in args.sources],
        repository=args.repository,
        large_collection=args.large_collection,
        difference_pool=args.difference_pool,
    )
    validate_membership(result)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
