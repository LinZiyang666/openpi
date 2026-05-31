"""Aggregate a per-step latency CSV into a summary JSON.

For each CP1 segment column and each hit_type (plus an ``ALL`` bucket) computes
count / median / p50 / p95 of the recorded millisecond latencies, alongside
run-level hit counts and the cumulative action-history-approximation tally.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json

import numpy as np

# Latency columns produced by ``replay.CSV_HEADER``.
_SEGMENT_COLUMNS = [
    "cp1_collect_ms",
    "cp1_gate_ms",
    "cp1_build_ms",
    "cp1_search_ms",
    "cp1_judge_ms",
    "cp1_fetch_ms",
    "cp1_total_ms",
]


def summarize(csv_path: str) -> dict:
    """Read a per-step CSV and return the aggregated summary dict."""
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    by_segment: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    hit_counts: dict[str, int] = defaultdict(int)
    approx_active_rows = 0

    for row in rows:
        hit = row["hit_type"]
        hit_counts[hit] += 1
        if row.get("action_history_approx_active") == "1":
            approx_active_rows += 1
        for col in _SEGMENT_COLUMNS:
            raw = row.get(col, "")
            if raw != "":
                value = float(raw)
                by_segment[col][hit].append(value)
                by_segment[col]["ALL"].append(value)

    probes = {
        col: {hit: _stats(values) for hit, values in buckets.items()}
        for col, buckets in by_segment.items()
    }
    n_steps = len(rows)
    return {
        "n_steps": n_steps,
        "hit_counts": dict(hit_counts),
        "hit_rate": (hit_counts.get("FULL_HIT", 0) / n_steps) if n_steps else 0.0,
        "action_history_approx_active_rows": approx_active_rows,
        "probes": probes,
    }


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="per-step latency CSV")
    parser.add_argument("--out", required=True, help="output summary JSON path")
    args = parser.parse_args()
    summary = summarize(args.csv)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[summarize] {summary['n_steps']} steps -> {args.out}")


if __name__ == "__main__":
    main()
