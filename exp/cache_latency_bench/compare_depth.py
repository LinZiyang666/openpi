"""Aggregate multiple depth-study runs into a cross-depth latency comparison.

Each input run is one ``trajectory_depth`` — the only independent variable in
the study. For every run this reuses :func:`summarize.summarize` for the
full-run (``ALL``) view and recomputes a steady-state view restricted to
``step_idx >= STEADY_MIN_STEP``. The steady-state view isolates the regime where
the episode trajectory history is long enough that the configured depth is
actually reached: episodes ramp effective depth over their first ``depth-1``
steps (plan T2), so the early steps of a deep run understate its cost.

Output is a six-segment x depth markdown table (ALL + steady-state) plus
per-depth hit counts, written to stdout, and a machine-readable ``comparison.json``.

Public interface: ``parse_runs``, ``compare``, ``render_markdown``, ``main``.
Depends on ``exp.cache_latency_bench.summarize`` for the shared segment columns,
the per-bucket ``_stats`` helper, and the ALL-bucket ``summarize`` aggregator —
so ALL and steady-state are computed by identical statistics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os

from exp.cache_latency_bench.summarize import _SEGMENT_COLUMNS
from exp.cache_latency_bench.summarize import _stats
from exp.cache_latency_bench.summarize import summarize

# Steady-state cutoff: episodes ramp effective depth over the first depth-1 steps
# (plan T2). With max studied depth 5 the ramp ends at step_idx 4, so one shared
# cutoff of 4 gives every depth the same steady-state window — keeping the
# cross-depth comparison apples-to-apples.
STEADY_MIN_STEP = 4

# Fixed hit-type column order for the rendered hit-count table.
_HIT_ORDER = ["FULL_HIT", "WARM_START", "MISS"]


# ------------------------------------------------------------------
# Run parsing
# ------------------------------------------------------------------


def parse_runs(run_args: list[str]) -> dict[str, str]:
    """Parse ``label=path`` CLI entries into an ordered ``{label: path}`` map.

    Fails fast on a missing ``=``, an empty or duplicate label, or a
    non-existent path, so a malformed invocation never silently drops a depth
    from the comparison.
    """
    runs: dict[str, str] = {}
    for item in run_args:
        if "=" not in item:
            raise ValueError(f"--runs entry must be label=path, got: {item!r}")
        label, _, path = item.partition("=")
        if not label:
            raise ValueError(f"--runs entry has empty label: {item!r}")
        if label in runs:
            raise ValueError(f"--runs duplicate label: {label!r}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"per_step.csv not found for run {label!r}: {path}")
        runs[label] = path
    return runs


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------


def _steady_summary(csv_path: str) -> dict:
    """Aggregate only steady-state rows (``step_idx >= STEADY_MIN_STEP``).

    Mirrors :func:`summarize.summarize` field-for-field (same ``_SEGMENT_COLUMNS``
    and ``_stats``) but on the filtered row set, so the ALL and steady-state
    views are produced by identical statistics. An empty value for a segment
    (e.g. ``cp1_fetch_ms`` on a MISS row, where fetch never runs) is skipped, so
    a segment with no samples simply has no bucket.
    """
    with open(csv_path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["step_idx"]) >= STEADY_MIN_STEP]

    by_segment: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    hit_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        hit = row["hit_type"]
        hit_counts[hit] += 1
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
    return {"n_steps": len(rows), "hit_counts": dict(hit_counts), "probes": probes}


def _cell(stat: dict | None) -> dict | None:
    """Project one ``_stats`` dict to a comparison cell, or ``None`` if absent.

    Absent happens for a segment with no samples in a run — notably
    ``cp1_fetch_ms`` in a MISS-only run, where the fetch segment never executes.
    """
    if not stat:
        return None
    return {"median": stat["median"], "p95": stat["p95"], "count": stat["count"]}


def compare(runs: dict[str, str]) -> dict:
    """Build the cross-depth comparison structure from ``{label: csv_path}``."""
    out: dict = {
        "runs": {},
        "steady_min_step": STEADY_MIN_STEP,
        "all_table": {},
        "steady_table": {},
    }
    for label, path in runs.items():
        all_s = summarize(path)
        steady_s = _steady_summary(path)
        out["runs"][label] = {
            "all": {
                "n_steps": all_s["n_steps"],
                "hit_counts": all_s["hit_counts"],
                "hit_rate": all_s["hit_rate"],
            },
            "steady": {
                "n_steps": steady_s["n_steps"],
                "hit_counts": steady_s["hit_counts"],
            },
        }
        for col in _SEGMENT_COLUMNS:
            out["all_table"].setdefault(col, {})[label] = _cell(
                all_s["probes"].get(col, {}).get("ALL")
            )
            out["steady_table"].setdefault(col, {})[label] = _cell(
                steady_s["probes"].get(col, {}).get("ALL")
            )
    return out


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------


def _render_table(title: str, table: dict, labels: list[str]) -> str:
    lines = [
        f"### {title}",
        "",
        "| segment | " + " | ".join(labels) + " |",
        "|" + "---|" * (len(labels) + 1),
    ]
    for col in _SEGMENT_COLUMNS:
        cells = []
        for label in labels:
            c = table.get(col, {}).get(label)
            cells.append(f"{c['median']:.3f} ({c['p95']:.3f})" if c else "—")
        lines.append(f"| {col} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(out: dict) -> str:
    """Render the comparison dict as a markdown report (ALL + steady + hit counts)."""
    labels = list(out["runs"].keys())
    parts = [
        _render_table("ALL bucket — median ms (p95)", out["all_table"], labels),
        "",
        _render_table(
            f"Steady-state (step_idx >= {out['steady_min_step']}) — median ms (p95)",
            out["steady_table"],
            labels,
        ),
        "",
        "### Hit counts (ALL)",
        "",
        "| run | n_steps | " + " | ".join(_HIT_ORDER) + " |",
        "|" + "---|" * (len(_HIT_ORDER) + 2),
    ]
    for label in labels:
        hc = out["runs"][label]["all"]["hit_counts"]
        n = out["runs"][label]["all"]["n_steps"]
        parts.append(
            f"| {label} | {n} | "
            + " | ".join(str(hc.get(k, 0)) for k in _HIT_ORDER)
            + " |"
        )
    return "\n".join(parts)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="one or more label=per_step.csv entries (e.g. d1=.../per_step.csv)",
    )
    parser.add_argument("--out", required=True, help="output comparison.json path")
    args = parser.parse_args()

    runs = parse_runs(args.runs)
    out = compare(runs)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(render_markdown(out))
    print(f"\n[compare_depth] {len(runs)} runs -> {args.out}")


if __name__ == "__main__":
    main()
