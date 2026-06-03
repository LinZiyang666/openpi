"""Unit tests for the depth-study cross-depth comparison aggregator.

All synthetic, CPU-only, no model and no real cache: each test writes a tiny
``per_step.csv`` matching ``replay.CSV_HEADER`` and exercises ``compare_depth``.
Covers (plan §4): ALL vs steady-state aggregation, hit-count propagation,
missing no-fetch buckets, and ``--runs label=path`` fail-fast parsing.
"""

from __future__ import annotations

import csv

import pytest

from exp.cache_latency_bench import compare_depth
from exp.cache_latency_bench.replay import CSV_HEADER


# ------------------------------------------------------------------
# Synthetic CSV builders
# ------------------------------------------------------------------


def _write_csv(path, rows) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(step_idx, hit_type, *, search=1.0, fetch=None):
    """One per_step row; ``fetch=None`` leaves cp1_fetch_ms empty (MISS path)."""
    row = {col: "" for col in CSV_HEADER}
    row.update(
        {
            "repeat": 0,
            "episode_id": "ep0",
            "step_idx": step_idx,
            "task": "taskA",
            "hit_type": hit_type,
            "top_score": 0.9,
            "cp1_collect_ms": 0.10,
            "cp1_gate_ms": 0.10,
            "cp1_build_ms": 0.10,
            "cp1_search_ms": search,
            "cp1_judge_ms": 0.10,
            "cp1_total_ms": 1.0,
        }
    )
    if fetch is not None:
        row["cp1_fetch_ms"] = fetch
    return row


# ------------------------------------------------------------------
# ALL vs steady-state aggregation
# ------------------------------------------------------------------


def test_all_vs_steady_aggregation(tmp_path):
    csv_path = tmp_path / "per_step.csv"
    # search latency 1..6 over step_idx 0..5.
    _write_csv(
        csv_path,
        [_row(i, "FULL_HIT", search=float(i + 1), fetch=0.05) for i in range(6)],
    )

    out = compare_depth.compare({"d5": str(csv_path)})
    all_cell = out["all_table"]["cp1_search_ms"]["d5"]
    steady_cell = out["steady_table"]["cp1_search_ms"]["d5"]

    assert all_cell["count"] == 6
    assert all_cell["median"] == 3.5  # median of 1..6
    # steady = step_idx >= 4 -> {5, 6}
    assert steady_cell["count"] == 2
    assert steady_cell["median"] == 5.5
    assert out["runs"]["d5"]["steady"]["n_steps"] == 2
    assert out["steady_min_step"] == compare_depth.STEADY_MIN_STEP


def test_steady_empty_when_all_early(tmp_path):
    csv_path = tmp_path / "per_step.csv"
    # only step_idx 0,1,2 -> nothing in the steady window.
    _write_csv(
        csv_path, [_row(i, "FULL_HIT", search=1.0, fetch=0.05) for i in range(3)]
    )

    out = compare_depth.compare({"d5": str(csv_path)})
    assert out["runs"]["d5"]["steady"]["n_steps"] == 0
    assert out["steady_table"]["cp1_search_ms"]["d5"] is None
    assert out["all_table"]["cp1_search_ms"]["d5"]["count"] == 3


# ------------------------------------------------------------------
# Hit-count propagation
# ------------------------------------------------------------------


def test_hit_counts_propagation(tmp_path):
    csv_path = tmp_path / "per_step.csv"
    _write_csv(
        csv_path,
        [
            _row(0, "FULL_HIT", fetch=0.05),
            _row(1, "MISS"),
            _row(2, "FULL_HIT", fetch=0.05),
            _row(3, "WARM_START", fetch=0.05),
        ],
    )

    out = compare_depth.compare({"d3": str(csv_path)})
    hc = out["runs"]["d3"]["all"]["hit_counts"]
    assert hc["FULL_HIT"] == 2
    assert hc["MISS"] == 1
    assert hc["WARM_START"] == 1
    assert out["runs"]["d3"]["all"]["n_steps"] == 4


def test_multi_run_table_labels_ordered(tmp_path):
    paths = {}
    for label in ["d1", "d3", "d4"]:
        p = tmp_path / f"{label}.csv"
        _write_csv(p, [_row(i, "FULL_HIT", search=1.0, fetch=0.05) for i in range(5)])
        paths[label] = str(p)

    out = compare_depth.compare(paths)
    assert list(out["runs"].keys()) == ["d1", "d3", "d4"]
    md = compare_depth.render_markdown(out)
    # header keeps the run order
    assert "| segment | d1 | d3 | d4 |" in md


# ------------------------------------------------------------------
# Missing no-fetch bucket (MISS-only run)
# ------------------------------------------------------------------


def test_miss_only_no_fetch_bucket(tmp_path):
    csv_path = tmp_path / "per_step.csv"
    # all MISS -> cp1_fetch_ms never recorded.
    _write_csv(csv_path, [_row(i, "MISS", search=1.0) for i in range(5)])

    out = compare_depth.compare({"d1": str(csv_path)})
    # fetch segment has no samples -> None cell, no crash.
    assert out["all_table"]["cp1_fetch_ms"]["d1"] is None
    assert out["steady_table"]["cp1_fetch_ms"]["d1"] is None
    # other segments still aggregate.
    assert out["all_table"]["cp1_search_ms"]["d1"] is not None
    # render tolerates the gap and shows an em-dash.
    md = compare_depth.render_markdown(out)
    assert "cp1_fetch_ms" in md
    assert "—" in md


# ------------------------------------------------------------------
# parse_runs fail-fast
# ------------------------------------------------------------------


def test_parse_runs_ok(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x")
    assert compare_depth.parse_runs([f"d1={p}"]) == {"d1": str(p)}


def test_parse_runs_missing_eq(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x")
    with pytest.raises(ValueError, match="label=path"):
        compare_depth.parse_runs([str(p)])


def test_parse_runs_empty_label(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x")
    with pytest.raises(ValueError, match="empty label"):
        compare_depth.parse_runs([f"={p}"])


def test_parse_runs_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        compare_depth.parse_runs([f"d1={tmp_path / 'nope.csv'}"])


def test_parse_runs_duplicate(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x")
    with pytest.raises(ValueError, match="duplicate"):
        compare_depth.parse_runs([f"d1={p}", f"d1={p}"])
