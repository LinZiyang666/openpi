"""Verify pruning witnesses, raw serialization and actual backend retrieval.

Structural checks cover every source and retained entry. Frozen production
float32 blocks are the equivalence reference. A dimension/normalizer dependent
roundoff budget and measured top-1 regret gate reloaded retrieval; a separate
NumPy float64 oracle supplies diagnostics only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from enum import Enum
import hashlib
from pathlib import Path
import struct

import numpy as np
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.factors.base import LibraryStats

from .common import (
    FIELDS,
    VERSION,
    check_identity,
    check_seal,
    context,
    digest,
    read_json,
    require,
    retrieval_digest,
    seal,
    strategy_for_storage,
    template,
    write_json,
)
from .prune_library import load_raw, load_score_blocks, select_retained, source_rows


def value_digest(value) -> str:
    """Hash nested values including concrete array types and missing object fields."""
    state = hashlib.sha256()

    def visit(item):
        typename = f"{type(item).__module__}.{type(item).__qualname__}"
        state.update(typename.encode() + b"\0")
        if isinstance(item, np.ndarray):
            require(
                item.dtype.kind != "O",
                "object arrays are not supported artifact values",
            )
            state.update(str((item.dtype.str, item.shape)).encode())
            state.update(item.tobytes(order="C"))
        elif isinstance(item, torch.Tensor):
            state.update(
                str((item.dtype, tuple(item.shape), str(item.device))).encode()
            )
            state.update(
                item.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
        elif isinstance(item, dict):
            # Dictionary insertion order is retained and is part of the raw contract.
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple)):
            state.update(str(len(item)).encode())
            for child in item:
                visit(child)
        elif isinstance(item, float):
            state.update(struct.pack("!d", item))
        elif isinstance(item, Enum):
            visit(item.value)
        elif isinstance(item, np.generic):
            state.update(item.tobytes())
        elif hasattr(item, "__dict__"):
            visit(vars(item))
        elif item is None or isinstance(item, (str, int, bool, bytes)):
            state.update(repr(item).encode())
        else:
            raise ValueError(f"unsupported artifact value: {typename}")
        state.update(b"\xff")

    visit(value)
    return state.hexdigest()


def _oracle_matrices(candidates):
    matrices = {
        field: np.stack(
            [np.asarray(e.query_keys[field], dtype=np.float64) for e in candidates]
        )
        for field in FIELDS
    }
    require(
        all(np.isfinite(matrix).all() for matrix in matrices.values()),
        "nonfinite candidate keys",
    )
    norms = {
        field: np.linalg.norm(matrix, axis=1)
        for field, matrix in matrices.items()
        if field != "robot_state"
    }
    return matrices, norms


def oracle_scores(
    query, candidates: list, config: dict, *, prepared=None
) -> np.ndarray:
    """Independently compute fixed zscore-tanh fusion in float64 raw score space."""
    result = np.zeros(len(candidates), dtype=np.float64)
    spec = config["checkpoints"]["cp1"]["search_strategy"]
    matrices, norms = _oracle_matrices(candidates) if prepared is None else prepared
    for field in FIELDS:
        q = np.asarray(query.query_keys[field], dtype=np.float64)
        matrix = matrices[field]
        if field == "robot_state":
            raw = -np.linalg.norm(matrix - q, axis=1)
        else:
            raw = (matrix @ q) / np.maximum(norms[field] * np.linalg.norm(q), 1e-8)
        params = spec["score_normalization"]["fields"][field]["params"]
        normalized = 0.5 * (np.tanh((raw - params["mu"]) / params["sigma"]) + 1)
        result += config["keys"][field]["weight"] * normalized
    return result


def float32_error_budget(query, config: dict, *, prepared) -> np.ndarray:
    """Bound two float32 evaluations using serial-reduction gamma bounds.

    u=2**-24 and gamma(n)=n*u/(1-n*u). For a D-term dot/norm reduction,
    n=D+8 includes products, sqrt and division. Two cosine evaluations differ
    by at most 4*gamma/(1-gamma); two L2 evaluations by at most
    2*gamma/(1-gamma)*distance. Propagate through the 1/(2*sigma) Lipschitz
    constant of zscore-tanh, plus scalar arithmetic/constant rounding.
    This conservative forward-error envelope assumes finite normal float32
    arithmetic (and <=4u absolute tanh error), not a fitted tolerance. Actual
    measured error, rather than this worst-case bound, gates top-1 regret.
    """
    matrices, _ = prepared
    u = np.finfo(np.float32).eps / 2
    spec = config["checkpoints"]["cp1"]["search_strategy"]
    budget = np.zeros(len(matrices[FIELDS[0]]), dtype=np.float64)
    for field in FIELDS:
        matrix = matrices[field]
        q = np.asarray(query.query_keys[field], dtype=np.float64)
        require(np.isfinite(q).all(), "nonfinite verification query")
        n = matrix.shape[1] + 8
        require(n * u < 0.5, "dimension outside float32 error model")
        gamma = n * u / (1 - n * u)
        params = spec["score_normalization"]["fields"][field]["params"]
        sigma = params["sigma"] if params["sigma"] > 1e-12 else 1.0
        if field == "robot_state":
            magnitude = np.linalg.norm(matrix - q, axis=1)
            raw_gap = 2 * gamma / (1 - gamma) * magnitude
        else:
            magnitude = 1.0
            raw_gap = 4 * gamma / (1 - gamma)
        # Sixteen elementary rounding terms cover both normalization paths,
        # scalar mu/sigma rounding, tanh and final weighted accumulation.
        scalar_gap = 16 * u * (1 + (magnitude + abs(params["mu"])) / sigma)
        budget += abs(config["keys"][field]["weight"]) * np.minimum(
            1 + 16 * u, raw_gap / (2 * sigma) + scalar_gap
        )
    return budget + 8 * u


def verify_pruned(
    source_manifest: dict,
    artifact_manifest: dict,
    retrieval_config: dict,
    *,
    score_manifest: dict,
) -> dict:
    """Verify full coverage, unchanged retained values, valid witnesses and top-1."""
    for manifest in (source_manifest, artifact_manifest, score_manifest):
        check_seal(manifest)
    source, output = source_manifest, artifact_manifest
    require(
        score_manifest["retrieval_digest"] == retrieval_digest(retrieval_config),
        "verification retrieval differs from frozen scores",
    )
    require(
        score_manifest["source_digest"] == source["digest"],
        "scores belong to another source",
    )
    require(
        output["source_digest"] == source["digest"]
        and output["parent_file"] == source["file"],
        "wrong parent",
    )
    check_identity(source["file"])
    check_identity(output["file"])
    raw_source = load_raw(source["file"]["path"])
    rows = source_rows(raw_source["entries"])
    require(
        rows == source["rows"] and digest(rows) == source["rows_digest"],
        "source row mismatch",
    )
    selection = output["selection"]
    require(
        selection["source_rows_digest"] == digest(rows), "selection coordinates changed"
    )
    blocks = load_score_blocks(score_manifest, rows)
    near_threshold = 0
    # Threshold comparisons are exact against frozen float32 values. This
    # diagnostic counts one representable float32 step, never widens pruning.
    threshold_band = (
        float(abs(np.spacing(np.float32(selection["threshold"]))))
        if selection["threshold"] is not None
        else 0.0
    )
    if selection["threshold"] is not None:
        for ordinals, matrix in blocks.values():
            remaining = np.array([rows[i]["remaining"] for i in ordinals])
            trajectories = np.array([rows[i]["trajectory_id"] for i in ordinals])
            for i in range(len(ordinals)):
                eligible = (remaining < remaining[i]) & (
                    trajectories != trajectories[i]
                )
                near_threshold += int(
                    np.count_nonzero(
                        eligible
                        & (
                            np.abs(
                                matrix[i].astype(np.float64) - selection["threshold"]
                            )
                            <= threshold_band
                        )
                    )
                )
    recomputed = select_retained(rows, blocks, selection["threshold"]).to_dict()
    require(
        selection == recomputed, "selection differs from frozen-source pruning rule"
    )
    kept = set(selection["kept_ids"])
    lookup = {r["id"]: r for r in rows}
    for witness in selection["witnesses"]:
        old, new = lookup[witness["removed_id"]], lookup[witness["retained_id"]]
        require(
            new["id"] in kept and old["id"] not in kept,
            "witness is deleted or deletion retained",
        )
        require(
            old["task_key"] == new["task_key"]
            and old["trajectory_id"] != new["trajectory_id"],
            "illegal witness identity",
        )
        require(
            new["remaining"] < old["remaining"]
            and witness["score"] >= selection["threshold"],
            "illegal witness progress/score",
        )
    require(
        all(r["id"] in kept for r in rows if r["remaining"] == 0),
        "source terminal deleted",
    )
    task_counts = dict(Counter(r["task_key"] for r in rows if r["id"] in kept))
    require(
        set(task_counts) == set(source["task_counts"])
        and output["task_counts"] == task_counts,
        "task coverage changed",
    )
    require(output["entries"] == len(kept), "output census mismatch")
    require(
        output["trajectory_counts"]
        == dict(Counter(r["trajectory_id"] for r in rows if r["id"] in kept)),
        "trajectory census mismatch",
    )
    raw_output = load_raw(output["file"]["path"])
    is_baseline = selection["threshold"] is None
    require(
        output["cp1_d1_pure_cache_only"] is (not is_baseline),
        "artifact restriction mismatch",
    )
    if is_baseline:
        require(
            output["file"] == source["file"],
            "P00 must reference the exact original bytes and path",
        )
        require(selection["kept_ids"] == [r["id"] for r in rows], "P00 removed entries")
    else:
        require(
            set(raw_output)
            == set(raw_source) | {"cp1_d1_pure_cache_only", "library_stats"},
            "top-level keys changed",
        )
        require(
            raw_output["cp1_d1_pure_cache_only"] is True,
            "missing pure-cache restriction",
        )
        for key in set(raw_source) - {"entries", "library_stats"}:
            require(
                value_digest(raw_output[key]) == value_digest(raw_source[key]),
                f"top-level value changed: {key}",
            )
        require(
            value_digest(raw_output["library_stats"])
            == value_digest(LibraryStats.compute_from_entries(raw_output["entries"])),
            "incorrect library statistics",
        )
    require(
        [e.id for e in raw_output["entries"]] == selection["kept_ids"],
        "retained order/content mismatch",
    )
    removed_edges = 0
    originals = {e.id: e for e in raw_source["entries"]}
    for entry in raw_output["entries"]:
        original = originals[entry.id]
        before = {
            k: v for k, v in vars(original).items() if k not in ("prev_ids", "next_ids")
        }
        after = {
            k: v for k, v in vars(entry).items() if k not in ("prev_ids", "next_ids")
        }
        require(
            value_digest(before) == value_digest(after),
            f"retained entry mutated: {entry.id}",
        )
        for field in ("prev_ids", "next_ids"):
            old_edges = getattr(original, field)
            new_edges = [edge for edge in old_edges if edge in kept]
            require(
                getattr(entry, field) == new_edges,
                "bridged, dangling or reordered edge",
            )
            removed_edges += len(old_edges) - len(new_edges)
    require(output["removed_edges"] == removed_edges, "deleted edge census mismatch")
    backend = InMemoryBackend(retrieval_config["backend"]["vector_dims"])
    backend.load_artifact(output["file"]["path"])
    backend.freeze()
    storage = CacheStorage(backend)
    top1 = strategy_for_storage(storage, retrieval_config, 1)
    all_scores = strategy_for_storage(storage, retrieval_config, len(kept))
    tasks = {
        task: [e for e in raw_output["entries"] if e.payload.task_key == task]
        for task in task_counts
    }
    maximum_error, maximum_bound, maximum_regret, near_ties, queries = (
        0.0,
        0.0,
        0.0,
        0,
        0,
    )
    oracle_errors = []
    block_positions = {
        task: {rows[ordinal]["id"]: i for i, ordinal in enumerate(ordinals)}
        for task, (ordinals, _) in blocks.items()
    }
    previous_task, prepared = None, None
    for query in sorted(raw_source["entries"], key=lambda e: e.payload.task_key):
        candidates = tasks[query.payload.task_key]
        if previous_task != query.payload.task_key:
            prepared = None
            prepared = _oracle_matrices(candidates)
            previous_task = query.payload.task_key
        diagnostic = oracle_scores(
            query, candidates, retrieval_config, prepared=prepared
        )
        positions = block_positions[query.payload.task_key]
        matrix = blocks[query.payload.task_key][1]
        reference = np.asarray(
            matrix[positions[query.id], [positions[e.id] for e in candidates]],
            dtype=np.float64,
        )
        results = all_scores.search(context(query))
        actual = {r.id: r.score for r in results}
        require(
            set(actual) == {e.id for e in candidates},
            "loaded search candidate coverage mismatch",
        )
        actual_values = np.array([actual[e.id] for e in candidates])
        require(np.isfinite(actual_values).all(), "nonfinite loaded scores")
        errors = np.abs(actual_values - reference)
        error = float(errors.max())
        maximum_error = max(maximum_error, error)
        bounds = float32_error_budget(query, retrieval_config, prepared=prepared)
        maximum_bound = max(maximum_bound, float(bounds.max()))
        oracle_errors.append(float(np.abs(actual_values - diagnostic).max()))
        require(
            bool(np.all(errors <= bounds)),
            f"retrieval scores exceed float32 propagation bound: {error}",
        )
        winner = top1.search(context(query))
        require(len(winner) == 1 and winner[0].id in actual, "invalid loaded top-1")
        require(
            winner[0].score == max(actual.values()),
            "top-1 differs from actual score maximum",
        )
        index = next(
            i for i, entry in enumerate(candidates) if entry.id == winner[0].id
        )
        regret = float(reference.max() - reference[index])
        require(
            regret <= np.nextafter(2 * error, np.inf),
            "top-1 lies outside measured numerical ambiguity",
        )
        maximum_regret = max(maximum_regret, regret)
        margin = (
            float(np.sort(reference)[-1] - np.sort(reference)[-2])
            if len(reference) > 1
            else 1.0
        )
        near_ties += int(margin <= 2 * error)
        queries += 1
    check_identity(source["file"])
    check_identity(output["file"])
    return seal(
        {
            "version": VERSION,
            "kind": "verification",
            "passed": True,
            "source_digest": source["digest"],
            "artifact_digest": output["digest"],
            "score_digest": score_manifest["digest"],
            "queries": queries,
            "max_score_error": maximum_error,
            "score_reference": "frozen_float32_v1",
            "float32_error_model": "serial_gamma_D_plus_8_zscore_lipschitz_v1",
            "float32_error_model_role": "sanity_envelope",
            "max_float32_bound": maximum_bound,
            "max_top1_regret": maximum_regret,
            "float64_diagnostic": {
                "max_score_error": max(oracle_errors),
                "p99_query_max_score_error": float(np.quantile(oracle_errors, 0.99)),
                "hard_gate": False,
            },
            "near_tie_queries": near_ties,
            "near_threshold_legal_pairs": near_threshold,
            "near_threshold_deleted_pairs": sum(
                abs(w["score"] - selection["threshold"]) <= threshold_band
                for w in selection["witnesses"]
            ),
            "near_threshold_band": threshold_band,
            "retrieval_digest": retrieval_digest(retrieval_config),
            "retained_entries": len(kept),
            "removed_edges": removed_edges,
        }
    )


def main() -> None:
    """Run complete structural and loaded-retrieval verification for one artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-manifest", "artifact-manifest", "score-manifest", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    source = read_json(args.source_manifest)
    result = verify_pruned(
        source,
        read_json(args.artifact_manifest),
        template(source["suite"]),
        score_manifest=read_json(args.score_manifest),
    )
    write_json(args.out, result)


if __name__ == "__main__":
    main()
