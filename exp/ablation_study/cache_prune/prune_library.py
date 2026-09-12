"""Audit successful source libraries, score tasks, and export pruned CP1 PKLs.

The CLI prepares one source at a time. Source provenance is checked against
frozen collection records before production retrieval computes task score blocks.
Exports reload raw pickle objects, preserving everything except deleted edges.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
from pathlib import Path
import pickle
import time

import h5py
import numpy as np
import torch

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.types import CheckpointID

from .common import (
    CENSUS,
    REGIMES,
    SUITES,
    VERSION,
    PruneSelection,
    SourceRow,
    check_identity,
    check_seal,
    context,
    digest,
    environment,
    file_identity,
    make_strategy,
    new_directory,
    read_json,
    require,
    retrieval_digest,
    seal,
    sha256,
    template,
    task_prompt,
    write_json,
)


def load_raw(path: str | Path) -> dict:
    """Read a trusted project pickle without the backend's compatibility edits."""
    with Path(path).open("rb") as handle:
        result = pickle.load(handle)
    require(
        isinstance(result, dict) and isinstance(result.get("entries"), list),
        "invalid artifact",
    )
    return result


def source_rows(entries: list) -> list[dict]:
    """Validate intact source chains and capture immutable original coordinates."""
    require(bool(entries), "empty source")
    require(len({e.id for e in entries}) == len(entries), "duplicate entry id")
    trajectories = defaultdict(list)
    for ordinal, entry in enumerate(entries):
        require(isinstance(entry.id, str) and bool(entry.id), "invalid entry id")
        require(entry.checkpoint_id == CheckpointID.CP1, "source is not CP1")
        require(
            isinstance(entry.trajectory_id, str) and bool(entry.trajectory_id),
            "missing trajectory",
        )
        require(
            type(entry.step_idx) is int and entry.step_idx >= 0,
            "missing/noninteger step",
        )
        require(bool(entry.payload.task_key), "missing task key")
        require(
            getattr(entry, "outcome", None) in (None, 1),
            "failure outcome in success source",
        )
        trajectories[entry.trajectory_id].append((ordinal, entry))
    result = [None] * len(entries)
    for trajectory, members in trajectories.items():
        members.sort(key=lambda item: item[1].step_idx)
        require(
            len({e.payload.task_key for _, e in members}) == 1,
            "trajectory crosses tasks",
        )
        require(
            [e.step_idx for _, e in members] == list(range(len(members))),
            "source steps are not contiguous",
        )
        for step, (ordinal, entry) in enumerate(members):
            previous = [] if step == 0 else [members[step - 1][1].id]
            following = [] if step == len(members) - 1 else [members[step + 1][1].id]
            require(
                entry.prev_ids == previous and entry.next_ids == following,
                "broken or branching source chain",
            )
            result[ordinal] = asdict(
                SourceRow(
                    entry.id,
                    ordinal,
                    trajectory,
                    entry.payload.task_key,
                    step,
                    len(members),
                    len(members) - 1 - step,
                    list(previous),
                    list(following),
                )
            )
    return result


def expected_source(suite: str, regime: str, repository: str | Path = ".") -> dict:
    """Resolve the approved census and source provenance from repository records."""
    root = Path(repository).resolve()
    trajectories, entries = CENSUS[(suite, regime)]
    roster_path = root / f"exp/rit_pareto/config/task_order_{suite}.json"
    roster = read_json(roster_path)
    require(
        roster["suite"] == suite
        and set(roster["assignment"]) == {str(i) for i in range(10)},
        "task roster mismatch",
    )
    tasks = {
        i: task_prompt(item["task_name"]) for i, item in roster["assignment"].items()
    }
    value = {
        "suite": suite,
        "regime": regime,
        "trajectories": trajectories,
        "entries": entries,
        "tasks": tasks,
        "roster": file_identity(roster_path),
    }
    if regime == "rit50":
        registry_path = (
            root
            / f"exp/data_authority/records/weighted_sum__{suite}__cp1_spatial_pool_16.json"
        )
        registry = read_json(registry_path)
        value.update(
            kind="hdf5",
            init_map=str(
                root / f"exp/common/data/db/libero_cache/{suite}_init_map.json"
            ),
            repository=str(root),
            registry=file_identity(registry_path),
            allowed_sha256=[
                registry["integrity"]["sha256"],
                *[r["sha256"] for r in registry["replicas"]],
            ],
        )
    else:
        base = root / "exp/ablation_study/cache_size/config"
        value.update(
            kind="success_list",
            success_list=str(base / f"lists_success/episodes_{suite}_S6.txt"),
            census_record=str(base / f"entries_{suite}_success.json"),
        )
    return value


def _success_provenance(rows: list[dict], expected: dict) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["trajectory_id"]].append(row)
    evidence = [expected["roster"]] if "roster" in expected else []
    if expected["kind"] == "success_list":
        record = read_json(expected["census_record"])
        require(
            record["suite"] == expected["suite"]
            and record["outcome_filter"] == "success",
            "wrong success census",
        )
        tiers = [r for r in record["tiers"] if r["tier"] == "S6"]
        require(
            len(tiers) == 1
            and tiers[0]["entries"] == len(rows)
            and tiers[0]["episodes"] == len(grouped),
            "S6 census mismatch",
        )
        lines = Path(expected["success_list"]).read_text().splitlines()
        ids = [line.removesuffix(".h5") for line in lines]
        require(
            len(ids) == len(set(ids)) and set(ids) == set(grouped),
            "success membership mismatch",
        )
        for trajectory, members in grouped.items():
            task = trajectory.split("/")[0].removeprefix("task_")
            require(
                task in expected["tasks"]
                and members[0]["task_key"] == expected["tasks"][task],
                "success task mismatch",
            )
        evidence += [
            file_identity(expected["census_record"]),
            file_identity(expected["success_list"]),
        ]
    elif expected["kind"] == "hdf5":
        records = read_json(expected["init_map"])
        require(bool(records), "empty collection map")
        if all("trajectory_id" not in record for record in records):
            # The historical L10 map identifies a declared init pool, not H5
            # trajectories. Verify success from each exact H5 stem without
            # inventing a trajectory-to-init mapping from repeated episode IDs.
            pool = defaultdict(list)
            for record in records:
                task = str(record["task_id"])
                require(
                    task in expected["tasks"]
                    and record["prompt"] == expected["tasks"][task]
                    and task_prompt(record["task_name"]) == record["prompt"]
                    and type(record["subset_idx"]) is int,
                    "compact collection map task mismatch",
                )
                pool[record["prompt"]].append(record["subset_idx"])
            require(
                set(pool) == set(expected["tasks"].values())
                and all(
                    sorted(indices) == list(range(len(indices)))
                    for indices in pool.values()
                ),
                "compact collection map is incomplete or duplicated",
            )
            directory = (
                Path(expected["repository"])
                / f"exp/common/data/db/libero_cache/{expected['suite']}"
            )
            observed = []
            for path in sorted(directory.glob("*.h5")):
                with h5py.File(path, "r") as handle:
                    require(
                        str(handle.attrs["experiment_name"]) == expected["suite"],
                        "HDF5 suite mismatch",
                    )
                    observed.append(
                        {
                            "trajectory_id": path.stem,
                            "h5_path": str(path.resolve()),
                            "suite": expected["suite"],
                            "prompt": str(handle.attrs["task"]),
                            "entry_count": int(handle.attrs["num_steps"]),
                        }
                    )
            require(
                Counter(record["prompt"] for record in observed)
                == Counter({prompt: len(indices) for prompt, indices in pool.items()}),
                "compact collection map/HDF5 task census mismatch",
            )
            records = observed
        require(
            all("trajectory_id" in record for record in records),
            "mixed collection map schemas",
        )
        ids = [r["trajectory_id"] for r in records]
        require(len(ids) == len(set(ids)), "duplicate init-map trajectory")
        records = {r["trajectory_id"]: r for r in records}
        require(set(grouped) <= set(records), "source has no collection provenance")
        evidence.append(file_identity(expected["init_map"]))
        if "registry" in expected:
            evidence.append(expected["registry"])
        for trajectory, members in grouped.items():
            record = records[trajectory]
            path = Path(expected["repository"]) / record["h5_path"]
            require(record["suite"] == expected["suite"], "collection suite mismatch")
            require(
                record["prompt"] == members[0]["task_key"], "collection task mismatch"
            )
            require(record["entry_count"] == len(members), "collection length mismatch")
            with h5py.File(path, "r") as handle:
                success = handle.attrs.get("success")
                require(
                    isinstance(success, (bool, np.bool_)) and bool(success),
                    "HDF5 does not prove success",
                )
                require(
                    str(handle.attrs["task"]) == members[0]["task_key"],
                    "HDF5 task mismatch",
                )
                require(
                    int(handle.attrs["num_steps"]) == len(members),
                    "HDF5 length mismatch",
                )
            evidence.append(file_identity(path))
    else:
        raise ValueError("unsupported success provenance")
    return evidence


def audit_source(path: str | Path, expected_census: dict) -> dict:
    """Reject wrong census, failed provenance, malformed chains or changed files."""
    identity = file_identity(path)
    if "allowed_sha256" in expected_census:
        require(
            identity["sha256"] in expected_census["allowed_sha256"],
            "source is not a registered replica",
        )
    artifact = load_raw(path)
    require(
        "cp1_d1_pure_cache_only" not in artifact, "a pruned artifact cannot be a source"
    )
    rows = source_rows(artifact["entries"])
    require(len(rows) == expected_census["entries"], "entry census mismatch")
    require(
        len({r["trajectory_id"] for r in rows}) == expected_census["trajectories"],
        "trajectory census mismatch",
    )
    require(
        set(r["task_key"] for r in rows) == set(expected_census["tasks"].values()),
        "task census mismatch",
    )
    evidence = _success_provenance(rows, expected_census)
    for item in evidence:
        check_identity(item)
    check_identity(identity)
    return seal(
        {
            "version": VERSION,
            "kind": "source",
            "suite": expected_census["suite"],
            "regime": expected_census["regime"],
            "file": identity,
            "rows": rows,
            "rows_digest": digest(rows),
            "task_counts": dict(Counter(r["task_key"] for r in rows)),
            "trajectories": expected_census["trajectories"],
            "entries": len(rows),
            "success_provenance": evidence,
            "expected_census": expected_census,
            "environment": environment(),
        }
    )


def compute_task_scores(
    entries: list, retrieval_config: dict, out_dir: str | Path, *, source_manifest: dict
) -> dict:
    """Enumerate real production scores into separate complete float32 task blocks."""
    rows = source_rows(entries)
    check_seal(source_manifest)
    check_identity(source_manifest["file"])
    require(rows == source_manifest["rows"], "scoring source metadata mismatch")
    tasks = defaultdict(list)
    for row in rows:
        tasks[row["task_key"]].append(row["source_ordinal"])
    started = time.perf_counter()
    strategy = make_strategy(
        entries, retrieval_config, top_k=max(map(len, tasks.values()))
    )
    blocks = []
    with new_directory(out_dir) as temporary:
        for task_index, (task, ordinals) in enumerate(sorted(tasks.items())):
            block_start = time.perf_counter()
            positions = {rows[ordinal]["id"]: i for i, ordinal in enumerate(ordinals)}
            matrix = np.full((len(ordinals), len(ordinals)), np.nan, dtype=np.float32)
            for i, ordinal in enumerate(ordinals):
                hits = strategy.search(context(entries[ordinal]))
                require(
                    len(hits) == len(ordinals)
                    and {h.id for h in hits} == set(positions),
                    "incomplete task search",
                )
                for hit in hits:
                    matrix[i, positions[hit.id]] = hit.score
            require(bool(np.isfinite(matrix).all()), "nonfinite production scores")
            name = f"task_{task_index:02d}.npy"
            np.save(temporary / name, matrix, allow_pickle=False)
            blocks.append(
                {
                    "task_key": task,
                    "ordinals": ordinals,
                    "file": name,
                    "sha256": sha256(temporary / name),
                    "shape": list(matrix.shape),
                    "seconds": time.perf_counter() - block_start,
                }
            )
            print(
                f"scored task {task_index + 1}/{len(tasks)}: {len(ordinals)} entries",
                flush=True,
            )
        check_identity(source_manifest["file"])
        manifest = seal(
            {
                "version": VERSION,
                "kind": "scores",
                "rows_digest": digest(rows),
                "source_digest": source_manifest["digest"],
                "source_file": source_manifest["file"],
                "retrieval_digest": retrieval_digest(retrieval_config),
                "blocks": blocks,
                "directory": str(Path(out_dir).resolve()),
                "environment": environment(),
                "seconds": time.perf_counter() - started,
                "complete": True,
            }
        )
        write_json(temporary / "scores.json", manifest)
    return manifest


def load_score_blocks(
    manifest: dict, rows: list[dict]
) -> dict[str, tuple[list[int], np.ndarray]]:
    """Rehash and memory-map every frozen task block, requiring exact row coverage."""
    check_seal(manifest)
    require(
        manifest["complete"] is True and manifest["rows_digest"] == digest(rows),
        "score/source mismatch",
    )
    blocks, covered = {}, []
    for block in manifest["blocks"]:
        path = Path(manifest["directory"]) / block["file"]
        require(sha256(path) == block["sha256"], "score block changed")
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        ordinals = block["ordinals"]
        require(
            matrix.dtype == np.float32
            and matrix.shape == (len(ordinals), len(ordinals)),
            "score block shape/type",
        )
        require(bool(np.isfinite(matrix).all()), "nonfinite score block")
        task = block["task_key"]
        require(
            task not in blocks and ordinals == sorted(set(ordinals)),
            "duplicate task or ordinal",
        )
        require(
            all(0 <= i < len(rows) and rows[i]["task_key"] == task for i in ordinals),
            "cross-task score block",
        )
        blocks[task] = ordinals, matrix
        covered.extend(ordinals)
    require(sorted(covered) == list(range(len(rows))), "score coverage mismatch")
    return blocks


def select_retained(
    source_rows: list[dict], task_scores: dict, threshold: float | None
) -> PruneSelection:
    """Apply strict-shorter greedy pruning with final-kept, cross-trajectory witnesses."""
    rows = source_rows
    require(
        [r["source_ordinal"] for r in rows] == list(range(len(rows))),
        "source ordinal mismatch",
    )
    if threshold is None:
        return PruneSelection(None, digest(rows), [r["id"] for r in rows], [])
    threshold = float(np.float32(threshold))
    require(np.isfinite(threshold), "nonfinite threshold")
    kept_ids, witnesses = set(), []
    covered = []
    for task, (ordinals, scores) in task_scores.items():
        covered.extend(ordinals)
        require(
            scores.shape == (len(ordinals), len(ordinals))
            and scores.dtype == np.float32,
            "invalid score matrix",
        )
        require(all(rows[i]["task_key"] == task for i in ordinals), "cross-task matrix")
        order = sorted(
            range(len(ordinals)),
            key=lambda i: (rows[ordinals[i]]["remaining"], ordinals[i]),
        )
        remaining = np.array([rows[i]["remaining"] for i in ordinals])
        trajectories = np.array([rows[i]["trajectory_id"] for i in ordinals])
        kept = np.empty(len(ordinals), dtype=np.int64)
        n_kept = 0
        for i in order:
            active = kept[:n_kept]
            candidates = active[
                (remaining[active] < remaining[i])
                & (trajectories[active] != trajectories[i])
                & (scores[i, active] >= threshold)
            ]
            if candidates.size:
                j = min(
                    candidates,
                    key=lambda j: (
                        -float(scores[i, j]),
                        int(remaining[j]),
                        ordinals[j],
                    ),
                )
                witnesses.append(
                    {
                        "removed_id": rows[ordinals[i]]["id"],
                        "retained_id": rows[ordinals[j]]["id"],
                        "score": float(scores[i, j]),
                        "r_removed": int(remaining[i]),
                        "r_retained": int(remaining[j]),
                    }
                )
            else:
                kept[n_kept] = i
                n_kept += 1
                kept_ids.add(rows[ordinals[i]]["id"])
    require(
        sorted(covered) == list(range(len(rows))),
        "incomplete or duplicate task score coverage",
    )
    return PruneSelection(
        threshold,
        digest(rows),
        [r["id"] for r in rows if r["id"] in kept_ids],
        witnesses,
    )


def export_pruned(
    source_path: str | Path,
    selection: PruneSelection,
    out_dir: str | Path,
    *,
    source_manifest: dict,
) -> dict:
    """Publish an independent raw-pickle subset without reconnecting deleted links."""
    check_seal(source_manifest)
    require(
        file_identity(source_path) == source_manifest["file"],
        "export source differs from audited P00 parent",
    )
    require(
        selection.source_rows_digest == source_manifest["rows_digest"],
        "selection/source mismatch",
    )
    rows = source_manifest["rows"]
    kept = set(selection.kept_ids)
    require(
        selection.kept_ids == [r["id"] for r in rows if r["id"] in kept],
        "keep ids are unknown, duplicated or reordered",
    )
    require(
        all(r["id"] in kept for r in rows if r["remaining"] == 0), "terminal removed"
    )
    removed = {r["id"] for r in rows} - kept
    require(
        len(selection.witnesses) == len(removed)
        and {w["removed_id"] for w in selection.witnesses} == removed,
        "missing/duplicate deletion witness",
    )
    require(
        all(w["retained_id"] in kept for w in selection.witnesses),
        "witness is not retained",
    )
    if selection.threshold is None:
        require(not removed, "P00 cannot delete entries")
        artifact_file = source_manifest["file"]
        removed_edges = 0
        with new_directory(out_dir) as temporary:
            manifest = _artifact_manifest(
                source_manifest, selection, artifact_file, removed_edges
            )
            write_json(temporary / "artifact.json", manifest)
        return manifest
    raw = load_raw(source_path)
    require(source_rows(raw["entries"]) == rows, "source metadata changed")
    # Reloaded objects have never passed through load_artifact or a strategy.
    entries, removed_edges = [], 0
    for entry in raw["entries"]:
        if entry.id in kept:
            for field in ("prev_ids", "next_ids"):
                edges = getattr(entry, field)
                filtered = [edge for edge in edges if edge in kept]
                removed_edges += len(edges) - len(filtered)
                setattr(entry, field, filtered)
            entries.append(entry)
    output = dict(raw)
    output["entries"] = entries
    # Statistics read a separate lightweight copy; snapshot/key data stay raw.
    stats_entries = []
    for entry in entries:
        replica = copy.copy(entry)
        replica.payload = copy.copy(entry.payload)
        replica.payload.action_chunk = np.array(entry.payload.action_chunk, copy=True)
        replica.query_keys = {
            "robot_state": np.array(entry.query_keys["robot_state"], copy=True)
        }
        stats_entries.append(replica)
    output["library_stats"] = LibraryStats.compute_from_entries(stats_entries)
    output["cp1_d1_pure_cache_only"] = True
    with new_directory(out_dir) as temporary:
        path = temporary / "library.pkl"
        with path.open("xb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        artifact_file = file_identity(path)
        artifact_file["path"] = str(Path(out_dir).resolve() / "library.pkl")
        manifest = _artifact_manifest(
            source_manifest, selection, artifact_file, removed_edges
        )
        check_identity(source_manifest["file"])
        write_json(temporary / "artifact.json", manifest)
    return manifest


def _artifact_manifest(source, selection, identity, removed_edges):
    kept = set(selection.kept_ids)
    return seal(
        {
            "version": VERSION,
            "kind": "artifact",
            "source_digest": source["digest"],
            "parent_file": source["file"],
            "file": identity,
            "selection": selection.to_dict(),
            "entries": len(kept),
            "removed_edges": removed_edges,
            "cp1_d1_pure_cache_only": selection.threshold is not None,
            "task_counts": dict(
                Counter(r["task_key"] for r in source["rows"] if r["id"] in kept)
            ),
            "trajectory_counts": dict(
                Counter(r["trajectory_id"] for r in source["rows"] if r["id"] in kept)
            ),
        }
    )


def main() -> None:
    """Audit one approved source and produce its complete production score blocks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = audit_source(
        args.source, expected_source(args.suite, args.regime, args.repository)
    )
    # Leave an audit record even if scoring is interrupted; there is no freeze
    # until every block and its completion manifest have been atomically saved.
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "source.json", source)
    raw = load_raw(args.source)
    scores = compute_task_scores(
        raw["entries"],
        template(args.suite),
        args.out / "scores",
        source_manifest=source,
    )
    check_identity(source["file"])
    require(
        scores["rows_digest"] == source["rows_digest"], "source changed during scoring"
    )


if __name__ == "__main__":
    main()
