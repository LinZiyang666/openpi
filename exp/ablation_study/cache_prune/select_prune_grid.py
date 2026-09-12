"""Freeze nine distinct pruning levels using outcome-blind source-only sweeps.

Candidate thresholds come from legal-pair maxima and two deterministic midpoint
rounds. Exact dynamic programming matches measured deletion rates to the fixed
coverage targets; it never assumes that greedy pruning is monotone in theta.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import time

import numpy as np
import yaml

from .common import (
    ROOT,
    VERSION,
    check_identity,
    check_seal,
    digest,
    environment,
    new_directory,
    read_json,
    require,
    seal,
    write_json,
)
from .prune_library import load_score_blocks, select_retained


def grid_spec() -> dict:
    """Load the checked-in, preregistered coverage and candidate-generation rules."""
    return yaml.safe_load((ROOT / "config/grid.yaml").read_text())


def candidate_thresholds(rows: list[dict], blocks: dict, spec: dict) -> list[float]:
    """Generate float32 quantiles, anchors and midpoint refinements without outcomes."""
    maxima = []
    for ordinals, matrix in blocks.values():
        remaining = np.array([rows[i]["remaining"] for i in ordinals])
        trajectories = np.array([rows[i]["trajectory_id"] for i in ordinals])
        for i in range(len(ordinals)):
            legal = (remaining < remaining[i]) & (trajectories != trajectories[i])
            if legal.any():
                maxima.append(float(matrix[i, legal].max()))
    if not maxima:
        return []
    quantiles = np.quantile(
        maxima, np.linspace(0, 1, spec["quantiles"]), method="linear"
    )
    values = set(
        map(float, np.asarray([*quantiles, *spec["anchors"]], dtype=np.float32))
    )
    for _ in range(spec["midpoint_rounds"]):
        ordered = sorted(values)
        values.update(
            float(np.float32((a + b) / 2)) for a, b in zip(ordered, ordered[1:])
        )
    return sorted(values)


def match_targets(
    candidates: list[dict],
    n_entries: int,
    targets: list[float],
    max_gap: float,
    first_max: float,
) -> list[dict]:
    """Find the exact minimum-error increasing-rate subsequence with specified ties."""
    by_keep = {}
    for candidate in candidates:
        if candidate["removed_count"] == 0:
            continue
        old = by_keep.get(candidate["keep_digest"])
        if old is None or candidate["threshold"] > old["threshold"]:
            by_keep[candidate["keep_digest"]] = candidate
    by_count = {}
    for candidate in by_keep.values():
        old = by_count.get(candidate["removed_count"])
        if old is None or candidate["threshold"] > old["threshold"]:
            by_count[candidate["removed_count"]] = candidate
    ordered = sorted(by_count.values(), key=lambda c: c["removed_count"])
    rates = [Fraction(c["removed_count"], n_entries) for c in ordered]
    gap, first = Fraction(str(max_gap)), Fraction(str(first_max))
    states = {}
    for point, target in enumerate(map(lambda x: Fraction(str(x)), targets)):
        next_states = {}
        for i, candidate in enumerate(ordered):
            cost = abs(rates[i] - target)
            if cost > gap or (point == 0 and not 0 < rates[i] <= first):
                continue
            previous = (
                [(Fraction(0), (), (), ())]
                if point == 0
                else [v for j, v in states.items() if j < i]
            )
            if previous:
                best = min(previous, key=lambda v: v[:3])
                next_states[i] = (
                    best[0] + cost,
                    (*best[1], rates[i]),
                    (*best[2], -candidate["threshold"]),
                    (*best[3], i),
                )
        states = next_states
    require(
        bool(states),
        "grid_not_representable: nine distinct rates cannot meet preregistered coverage",
    )
    best = min(states.values(), key=lambda v: v[:3])
    return [ordered[i] for i in best[3]]


def select_grid(source_manifest: dict, score_manifest: dict, spec: dict) -> dict:
    """Sweep a complete source and freeze a mechanically selected nine-point grid."""
    check_seal(source_manifest)
    check_seal(score_manifest)
    check_identity(source_manifest["file"])
    require(
        score_manifest["source_digest"] == source_manifest["digest"]
        and score_manifest["source_file"] == source_manifest["file"],
        "scores belong to another source",
    )
    require(spec == grid_spec(), "grid protocol differs from preregistered rules")
    rows = source_manifest["rows"]
    blocks = load_score_blocks(score_manifest, rows)
    started = time.perf_counter()
    candidates = []
    for threshold in candidate_thresholds(rows, blocks, spec):
        candidate_start = time.perf_counter()
        selection = select_retained(rows, blocks, threshold)
        kept = set(selection.kept_ids)
        candidates.append(
            {
                "threshold": threshold,
                "threshold_bits": int(np.float32(threshold).view(np.uint32)),
                "removed_count": len(rows) - len(kept),
                "retained_count": len(kept),
                "actual_rate": (len(rows) - len(kept)) / len(rows),
                "keep_digest": digest(selection.kept_ids),
                "task_counts": {
                    task: sum(rows[i]["id"] in kept for i in ordinals)
                    for task, (ordinals, _) in blocks.items()
                },
                "witness_digest": digest(selection.witnesses),
                "seconds": time.perf_counter() - candidate_start,
                "complete": True,
            }
        )
    try:
        selected = match_targets(
            candidates, len(rows), spec["targets"], spec["max_gap"], spec["first_max"]
        )
        for candidate in selected:
            require(
                all(n > 0 for n in candidate["task_counts"].values()),
                "empty task after pruning",
            )
    except ValueError as exc:
        return seal(
            {
                "version": VERSION,
                "kind": "grid",
                "status": "grid_not_representable",
                "source_digest": source_manifest["digest"],
                "score_digest": score_manifest["digest"],
                "candidates": candidates,
                "candidate_digest": digest(candidates),
                "seconds": time.perf_counter() - started,
                "complete": True,
                "reason": str(exc),
            }
        )
    points = [
        {
            "point": "P00",
            "threshold": None,
            "actual_rate": 0.0,
            "target_rate": 0.0,
            "removed_count": 0,
            "retained_count": len(rows),
            "keep_digest": digest([r["id"] for r in rows]),
        }
    ]
    points.extend(
        {**c, "point": f"P{i:02d}", "target_rate": target}
        for i, (c, target) in enumerate(zip(selected, spec["targets"]), 1)
    )
    check_identity(source_manifest["file"])
    return seal(
        {
            "version": VERSION,
            "kind": "grid",
            "status": "frozen",
            "complete": True,
            "suite": source_manifest["suite"],
            "regime": source_manifest["regime"],
            "source_digest": source_manifest["digest"],
            "source_file": source_manifest["file"],
            "score_digest": score_manifest["digest"],
            "retrieval_digest": score_manifest["retrieval_digest"],
            "spec": spec,
            "spec_digest": digest(spec),
            "candidates": candidates,
            "candidate_digest": digest(candidates),
            "candidate_count": len(candidates),
            "points": points,
            "seconds": time.perf_counter() - started,
            "environment": environment(),
        }
    )


def main() -> None:
    """Write a completed sweep and freeze, or a nonrepresentable-grid report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--score-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = select_grid(
        read_json(args.source_manifest), read_json(args.score_manifest), grid_spec()
    )
    with new_directory(args.out) as temporary:
        write_json(temporary / "grid_freeze.json", result)
    if result["status"] != "frozen":
        raise SystemExit(result["reason"])


if __name__ == "__main__":
    main()
