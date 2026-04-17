"""Unit tests for ``exp/trajectory_deviation/analysis/analyze_deviation_results.py`` (plan §12).

The plotting wrappers can't be exercised headlessly in this environment
(matplotlib may not be installed in CI), so these tests focus on the pure
data-transform helpers. If those are correct, the plots are a thin
matplotlib layer we can smoke-test manually.

Coverage:
- ``load_spawn_rows`` CSV typing (bool/int coercion).
- ``flatten_deviate_scores`` concatenation.
- ``compute_topk_coverage`` ordering + recall math, with a hand-built
  failure set.
- ``compute_success_rate_grid`` per-(n, k_idx) mean success.
- ``best_cell`` tie-breaker ordering.
- ``compute_strategy_comparison`` filtering.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import exp.trajectory_deviation.analysis.analyze_deviation_results as ana


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def test_load_spawn_rows_coerces_types(tmp_path: Path) -> None:
    p = tmp_path / "spawn_aggregate.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "config", "strategy", "episode", "s", "n", "k_idx",
            "success", "env_steps_executed",
        ])
        w.writeheader()
        w.writerow({
            "config": "clip_w7_d4", "strategy": "top_k",
            "episode": "task_0/episode_0", "s": "5", "n": "3", "k_idx": "0",
            "success": "True", "env_steps_executed": "42",
        })
        w.writerow({
            "config": "clip_w7_d4", "strategy": "random",
            "episode": "task_0/episode_0", "s": "1", "n": "5", "k_idx": "2",
            "success": "False", "env_steps_executed": "10",
        })
    rows = ana.load_spawn_rows(p)
    assert rows[0]["success"] is True and rows[0]["s"] == 5
    assert rows[1]["success"] is False and rows[1]["k_idx"] == 2


# ---------------------------------------------------------------------------
# Analysis 1
# ---------------------------------------------------------------------------


def test_flatten_deviate_scores_concatenates_episodes() -> None:
    scores = {
        "ep_a": {"deviate_score": [0.1, 0.9, 1.5]},
        "ep_b": {"deviate_score": [2.0]},
    }
    out = ana.flatten_deviate_scores(scores)
    assert np.array_equal(np.sort(out), np.array([0.1, 0.9, 1.5, 2.0]))


# ---------------------------------------------------------------------------
# Analysis 2 — Top-k coverage
# ---------------------------------------------------------------------------


def test_compute_topk_coverage_captures_failure_cycles_first() -> None:
    """Hand-built scenario: 5 cycles per episode, 1 "true failure" cycle
    has the highest score. Top-k fraction = 1/5 → recall must reach 1.0
    immediately. This locks (a) descending-sort, (b) failure-set
    construction, (c) recall denominator = total failures."""
    scores = {
        "ep_a": {"deviate_score": [0.1, 0.2, 0.3, 0.9, 0.4]},  # cycle 3 is top
    }
    # Spawn rows: cycle 3 with n=1 failed; all other n=1 spawns succeeded.
    rows = [
        {"config": "cfgA", "strategy": "top_k", "episode": "ep_a",
         "s": 3, "n": 1, "k_idx": 0, "success": False, "env_steps_executed": 10},
        {"config": "cfgA", "strategy": "top_k", "episode": "ep_a",
         "s": 0, "n": 1, "k_idx": 0, "success": True, "env_steps_executed": 10},
    ]
    curve = ana.compute_topk_coverage(
        scores, rows, config="cfgA", n_threshold=1, resolution=5,
    )
    # First cutoff covers top 1/5 cycles → that's cycle 3 → recall = 1.0.
    assert curve[0].top_k_fraction == pytest.approx(0.2)
    assert curve[0].recall == pytest.approx(1.0)


def test_compute_topk_coverage_empty_when_no_failures(caplog) -> None:
    """Degenerate case — no ground-truth failures → return empty (and
    log a warning). Prevents the plot from drawing a meaningless line."""
    scores = {"ep_a": {"deviate_score": [0.1, 0.2]}}
    rows = [
        {"config": "cfgA", "strategy": "top_k", "episode": "ep_a",
         "s": 0, "n": 1, "k_idx": 0, "success": True, "env_steps_executed": 5},
    ]
    with caplog.at_level("WARNING"):
        curve = ana.compute_topk_coverage(scores, rows, config="cfgA", n_threshold=1)
    assert curve == []
    assert "0 failure cycles" in caplog.text


def test_compute_topk_coverage_ignores_large_n_spawns() -> None:
    """A spawn failure at large n (teleport far before the crisis) is
    weak evidence — we only want close-in failures for ground truth.
    Asserting ``n > n_threshold`` rows get dropped locks that contract."""
    scores = {"ep_a": {"deviate_score": [0.9]}}
    rows = [
        {"config": "cfgA", "strategy": "top_k", "episode": "ep_a",
         "s": 0, "n": 20, "k_idx": 0, "success": False, "env_steps_executed": 0},
    ]
    curve = ana.compute_topk_coverage(
        scores, rows, config="cfgA", n_threshold=3, resolution=1,
    )
    # n=20 > threshold=3 → no ground-truth failure → empty curve.
    assert curve == []


# ---------------------------------------------------------------------------
# Analysis 3 — success-rate grid
# ---------------------------------------------------------------------------


def test_compute_success_rate_grid_averages_per_cell() -> None:
    rows = [
        {"config": "c", "strategy": "top_k", "episode": "e", "s": 0, "n": 1, "k_idx": 0,
         "success": True, "env_steps_executed": 1},
        {"config": "c", "strategy": "top_k", "episode": "e", "s": 5, "n": 1, "k_idx": 0,
         "success": False, "env_steps_executed": 1},
        {"config": "c", "strategy": "top_k", "episode": "e", "s": 0, "n": 3, "k_idx": 0,
         "success": True, "env_steps_executed": 1},
        {"config": "other", "strategy": "top_k", "episode": "e", "s": 0, "n": 1, "k_idx": 0,
         "success": False, "env_steps_executed": 1},
    ]
    grid = ana.compute_success_rate_grid(rows, config="c", strategy="top_k")
    # Two (n=1, k=0) rows at 0.5; one (n=3, k=0) row at 1.0. Config-filter
    # must drop the "other" row entirely.
    assert grid[(1, 0)] == pytest.approx(0.5)
    assert grid[(3, 0)] == pytest.approx(1.0)


def test_best_cell_prefers_smaller_n_on_tie() -> None:
    # Two cells tied at 1.0; smaller n wins.
    grid = {(1, 0): 1.0, (5, 0): 1.0, (1, 1): 1.0}
    # Smallest n, then smallest k_idx on tie → (1, 0).
    assert ana.best_cell(grid) == (1, 0)


def test_best_cell_empty() -> None:
    assert ana.best_cell({}) is None


# ---------------------------------------------------------------------------
# Analysis 4 — strategy comparison
# ---------------------------------------------------------------------------


def test_compute_strategy_comparison_filters_and_averages() -> None:
    rows = [
        {"config": "c", "strategy": "top_k", "episode": "e0", "s": 0, "n": 1, "k_idx": 0,
         "success": True, "env_steps_executed": 5},
        {"config": "c", "strategy": "top_k", "episode": "e1", "s": 0, "n": 1, "k_idx": 0,
         "success": False, "env_steps_executed": 5},
        {"config": "c", "strategy": "random", "episode": "e0", "s": 0, "n": 1, "k_idx": 0,
         "success": False, "env_steps_executed": 5},
    ]
    out = ana.compute_strategy_comparison(rows, config="c", n=1, k_idx=0)
    assert out["top_k"] == pytest.approx(0.5)
    assert out["random"] == pytest.approx(0.0)
    # Missing strategy → dropped (not NaN); keeps the bar chart honest.
    assert "equidistant" not in out


# ---------------------------------------------------------------------------
# Driver wiring
# ---------------------------------------------------------------------------


def test_analyze_config_skips_missing_scores(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing deviate_score file must warn + return empty, not crash.
    This is the critical "analyze partial results" path for WIP runs."""
    with caplog.at_level("WARNING"):
        paths = ana.analyze_config(
            "nope",
            deviate_score_dir=tmp_path,
            spawn_rows=[],
            out_dir=tmp_path,
        )
    assert paths == []
    assert "Missing deviate_score JSON" in caplog.text
