"""Tests for exp/verdict_factor_judge/phase3_threshold_solver.py (plan §8.3).

The solver replays warmup factor_raw rows through the actual src/
``CalibrationSamples -> PercentileRollingCalibration(samples, window_size).
bind_keys(keys) -> cal(raw) -> composer._score_only(...)`` chain. Tests
below exercise each step end-to-end on synthetic JSONL fixtures.

Test groups (mirrors plan §8.3):
  Reconstruction interface compatibility   — #1-#2
  Saturated-buffer behaviour               — #3-#4
  Fail-fast on insufficient samples        — #5-#6
  NaN propagation post-saturation          — #7-#9
  Quantile cut math                        — #10-#12
  Output schema                            — #13-#14
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from exp.verdict_factor_judge.phase3_threshold_solver import (
    Recipe,
    derive_thresholds,
    load_per_key_finite_history,
    reconstruct_scores,
    solve_recipe,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _write_jsonl(path: Path, rows: Iterable[dict]) -> Path:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def _row(values: dict[str, float]) -> dict:
    return {"factor_raw": values}


def _safe_recipe(*keys: str) -> Recipe:
    return Recipe(
        recipe_id="test_safe",
        declared_keys=list(keys),
        orientations={k: "safe" for k in keys},
        directions={},
    )


def _risky_recipe(*keys: str) -> Recipe:
    return Recipe(
        recipe_id="test_risky",
        declared_keys=list(keys),
        orientations={k: "risky" for k in keys},
        directions={},
    )


# ----------------------------------------------------------------------
# Reconstruction interface compatibility (#1-#2)
# ----------------------------------------------------------------------


def test_1_end_to_end_runs_returns_finite_scores(tmp_path: Path) -> None:
    """100 rows x 2 declared keys, varied values -> 100 finite scores.

    Hand-computed reference for 1 row to verify orientation flip +
    percentile direction. After bind_keys, buffer for each key is
    preloaded with last 50 of [0.0, 0.01, ..., 0.99]; row #50 (raw a=0.5,
    b=0.5) sees buffer = last 50 from [0..0.49 + (re-evolved)] — the
    exact rank depends on replay order, so we just spot-check finite-ness
    and sane bounds.
    """
    rows = [
        _row({"a": i / 100.0, "b": (99 - i) / 100.0})
        for i in range(100)
    ]
    p = _write_jsonl(tmp_path / "ok.jsonl", rows)
    scores = reconstruct_scores(p, _safe_recipe("a", "b"))
    assert len(scores) == 100
    assert all(not math.isnan(s) for s in scores)
    # All scores in [0, 1] for safe orientation + percentile in [0,1].
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_2_constructor_shape_regression_no_params_kwarg(tmp_path: Path) -> None:
    """Solver MUST use __init__(samples, window_size=...) + bind_keys() +
    cal(raw). A regression to params=/bind_keys= kwargs or .calibrate()
    is caught by monkey-patching the calibration class.
    """
    import openpi.cache.components.factors.calibrations.percentile_rolling as pr_mod

    # Track init/method signatures the solver actually uses.
    seen = {"init_kwargs": None, "called_calibrate": False, "called_call": False}
    real_cls = pr_mod.PercentileRollingCalibration

    class Sentinel(real_cls):
        def __init__(self, samples, *, window_size=200):
            seen["init_kwargs"] = {"samples_type": type(samples).__name__,
                                    "window_size": window_size}
            super().__init__(samples, window_size=window_size)

        def calibrate(self, raw):  # forbidden API
            seen["called_calibrate"] = True
            raise AssertionError("solver used .calibrate(...) which is not the real API")

        def __call__(self, raw):
            seen["called_call"] = True
            return super().__call__(raw)

    rows = [_row({"a": i / 100.0}) for i in range(100)]
    p = _write_jsonl(tmp_path / "ok.jsonl", rows)

    monkey_target = "openpi.cache.components.factors.calibrations.percentile_rolling.PercentileRollingCalibration"
    import sys
    mod = sys.modules["exp.verdict_factor_judge.phase3_threshold_solver"]
    orig = mod.PercentileRollingCalibration
    mod.PercentileRollingCalibration = Sentinel
    try:
        reconstruct_scores(p, _safe_recipe("a"))
    finally:
        mod.PercentileRollingCalibration = orig

    assert seen["init_kwargs"] is not None, "solver did not instantiate the calibration class"
    assert seen["init_kwargs"]["samples_type"] == "CalibrationSamples"
    assert seen["init_kwargs"]["window_size"] == 50
    assert seen["called_call"], "solver must call cal(raw), not cal.calibrate(raw)"
    assert not seen["called_calibrate"], "solver used .calibrate(...) — should be __call__"


# ----------------------------------------------------------------------
# Saturated-buffer behaviour (#3-#4)
# ----------------------------------------------------------------------


def test_3_buffer_pre_saturation_and_maxlen_eviction(tmp_path: Path) -> None:
    """60 rows monotone increasing on 1 key -> buffer saturates at last 50,
    row #0 (raw=1.0) post-append-with-eviction yields percentile = 1/50.

    Walks through:
      - per_key_history = [1.0, 2.0, ..., 60.0]
      - bind_keys preloads last 50 non-NaN: deque([11..60], maxlen=50)
      - replay row #0 (raw=1.0): buf.append(1.0) evicts 11
        -> deque([12..60, 1], maxlen=50), len=50
      - _percentile_rank counts (x <= 1) = 1 (just the 1 itself)
      -> rank = 1/50 = 0.02

    safe orientation -> contrib = 0.02 -> score = 0.02
    risky orientation -> contrib = 1 - 0.02 = 0.98 -> score = 0.98
    """
    rows = [_row({"a": float(i + 1)}) for i in range(60)]
    p = _write_jsonl(tmp_path / "monotone.jsonl", rows)

    safe = reconstruct_scores(p, _safe_recipe("a"))
    risky = reconstruct_scores(p, _risky_recipe("a"))

    assert safe[0] == pytest.approx(1 / 50)        # 0.02
    assert risky[0] == pytest.approx(1 - 1 / 50)   # 0.98 — mirrors g4 (jerk = risky)


def test_4_first_row_finite_score_guarantee(tmp_path: Path) -> None:
    """No cold-start NaN — every row with finite raw input gets a finite score."""
    rows = [_row({"a": float(i + 1)}) for i in range(60)]
    p = _write_jsonl(tmp_path / "finite.jsonl", rows)
    scores = reconstruct_scores(p, _safe_recipe("a"))
    assert not math.isnan(scores[0])


# ----------------------------------------------------------------------
# Fail-fast on insufficient samples (#5-#6)
# ----------------------------------------------------------------------


def test_5_fewer_than_window_size_non_nan_raises(tmp_path: Path) -> None:
    """49 non-NaN rows < window_size=50 -> bind_keys raises ValueError."""
    rows = [_row({"a": float(i)}) for i in range(49)]
    p = _write_jsonl(tmp_path / "sparse.jsonl", rows)
    with pytest.raises(ValueError, match=r"only \d+ non-NaN samples.*window_size=50"):
        reconstruct_scores(p, _safe_recipe("a"))


def test_6_per_key_independent_window_size_enforcement(tmp_path: Path) -> None:
    """Two declared keys: 'a' has 50 finite, 'b' has 50 NaN + 50 finite (total 100 rows)
    where 'a' is sparse if we only feed first half. Test: any single key
    failing window_size triggers the error.
    """
    rows = []
    for i in range(50):
        rows.append(_row({"a": float(i), "b": float("nan")}))    # b NaN here
    # 'a' has 50 non-NaN (just enough); 'b' has 0 non-NaN -> b fails.
    p = _write_jsonl(tmp_path / "mixed_sparse.jsonl", rows)
    with pytest.raises(ValueError, match="non-NaN samples"):
        reconstruct_scores(p, _safe_recipe("a", "b"))


# ----------------------------------------------------------------------
# NaN propagation post-saturation (#7-#9)
# ----------------------------------------------------------------------


def test_7_partial_nan_row_uses_zero_contrib_with_full_denominator(tmp_path: Path) -> None:
    """Some keys NaN, others numeric -> NaN keys contrib 0; denom = full N.

    Build 60-row warmup so saturation succeeds. Then check the first
    row's score directly: a=0.5, b=NaN.
        contrib_a = percentile rank of 0.5 in [last 50 of warmup history for a, then 0.5]
        contrib_b = 0 (NaN)
        score = (contrib_a + 0) / 2
    Rather than chasing the exact rank, assert the structural invariant:
    the partial-NaN score is *at most* contrib_a / 2 (which is at most 1/2).
    """
    rows = [_row({"a": float(i), "b": float(i)}) for i in range(60)]
    p = _write_jsonl(tmp_path / "warmup.jsonl", rows)
    # Replay the same warmup, but make the first row's `b` NaN.
    rows_partial = [
        _row({"a": float(i), "b": (float("nan") if i == 0 else float(i))})
        for i in range(60)
    ]
    p2 = _write_jsonl(tmp_path / "partial.jsonl", rows_partial)
    scores = reconstruct_scores(p2, _safe_recipe("a", "b"))
    # Row 0: contrib_a is some rank in [0, 1]; contrib_b = 0; denom = 2.
    # score = contrib_a / 2 in [0, 0.5].
    assert 0.0 <= scores[0] <= 0.5


def test_8_all_nan_row_mid_replay_returns_zero(tmp_path: Path) -> None:
    """Single all-NaN row mid-replay -> _score_only = 0/N = 0."""
    rows = [_row({"a": float(i)}) for i in range(60)]
    rows[30] = _row({"a": float("nan")})
    p = _write_jsonl(tmp_path / "midnan.jsonl", rows)
    scores = reconstruct_scores(p, _safe_recipe("a"))
    assert scores[30] == pytest.approx(0.0)


def test_9_all_nan_entire_jsonl_fails_at_bind_keys(tmp_path: Path) -> None:
    """Every row NaN -> 0 non-NaN samples -> bind_keys raises before any score."""
    rows = [_row({"a": float("nan")}) for i in range(60)]
    p = _write_jsonl(tmp_path / "allnan.jsonl", rows)
    with pytest.raises(ValueError, match="non-NaN samples"):
        reconstruct_scores(p, _safe_recipe("a"))


# ----------------------------------------------------------------------
# Quantile cut math (#10-#12)
# ----------------------------------------------------------------------


def test_10_uniform_distribution_cuts_match_quantiles() -> None:
    """1000 uniform [0,1] scores, fh=0.2 ws=0.3 -> cuts ~ 0.8 and 0.5."""
    rng = np.random.default_rng(42)
    scores = list(rng.uniform(0, 1, size=1000))
    fh_thr, ws_thr = derive_thresholds(scores, fh_ratio=0.2, ws_ratio=0.3)
    arr = np.asarray(scores)
    n_above_fh = int(np.sum(arr >= fh_thr))
    n_in_ws_band = int(np.sum((arr >= ws_thr) & (arr < fh_thr)))
    assert abs(n_above_fh - 200) <= 5
    assert abs(n_in_ws_band - 300) <= 5


def test_11_full_grid_extreme_zero_miss() -> None:
    """fh=0.5 + ws=0.5 -> sum=1.0 -> WS_thr = min(scores), no MISS in warmup."""
    rng = np.random.default_rng(0)
    scores = list(rng.uniform(0, 1, size=200))
    fh_thr, ws_thr = derive_thresholds(scores, fh_ratio=0.5, ws_ratio=0.5)
    arr = np.asarray(scores)
    assert ws_thr == pytest.approx(min(scores), rel=0, abs=1e-9)
    assert int(np.sum(arr >= ws_thr)) == len(scores)


def test_12_tie_heavy_distribution_no_exception() -> None:
    """Tie-heavy 3-bin distribution: cuts may not exactly partition, but
    must land on one of the bin values without exception.
    """
    scores = [0.0] * 333 + [0.5] * 334 + [1.0] * 333
    fh_thr, ws_thr = derive_thresholds(scores, fh_ratio=0.3, ws_ratio=0.4)
    assert fh_thr in {0.0, 0.5, 1.0}
    assert ws_thr in {0.0, 0.5, 1.0}


# ----------------------------------------------------------------------
# Output schema (#13-#14)
# ----------------------------------------------------------------------


def test_13_solve_recipe_output_schema_lock(tmp_path: Path) -> None:
    """End-to-end run -> output JSON has §5.8 fields, no cold-start vestige."""
    rows = [_row({"a": float(i + 1)}) for i in range(60)]
    p = _write_jsonl(tmp_path / "ok.jsonl", rows)
    grid = [(0.2, 0.2), (0.3, 0.3)]

    out = solve_recipe(p, _safe_recipe("a"), grid, warmup_yaml_id="test__warmup")

    expected_keys = {
        "recipe", "warmup_yaml_id", "warmup_jsonl",
        "n_warmup_rows", "n_finite_scores", "n_dropped_nan_scores",
        "score_stats", "per_key_non_nan_count", "cells",
    }
    assert expected_keys <= set(out.keys())

    # Cold-start vestige fields must NOT appear.
    assert "n_warmup_scores_warm_state" not in out
    assert "score_stats_warm_state" not in out

    # cells dict shape
    assert "fh0.2_ws0.2" in out["cells"]
    assert "fh_thr" in out["cells"]["fh0.2_ws0.2"]
    assert "ws_thr" in out["cells"]["fh0.2_ws0.2"]


def test_14_solve_recipe_fail_fast_marker(tmp_path: Path) -> None:
    """Sparse warmup -> output has 'error' field, no 'cells'."""
    rows = [_row({"a": float(i)}) for i in range(49)]
    p = _write_jsonl(tmp_path / "sparse.jsonl", rows)
    out = solve_recipe(p, _safe_recipe("a"), [(0.2, 0.2)])
    assert "error" in out
    assert "cells" not in out
    assert "non-NaN samples" in out["error"]


# ----------------------------------------------------------------------
# G2 R1 B1: WarmupPool sample parity — runner preload buffer must
# come from the same uncapped JSONL the solver replays, so eval-time
# bind_keys preloads the same last-50 the solver did.
# ----------------------------------------------------------------------


def test_load_per_key_finite_history_drops_nan_keeps_order(tmp_path: Path) -> None:
    rows = [
        _row({"a": 1.0, "b": float("nan")}),
        _row({"a": 2.0, "b": 5.0}),
        _row({"a": float("nan"), "b": 6.0}),
        _row({"a": 3.0, "b": float("nan")}),
    ]
    p = _write_jsonl(tmp_path / "mixed.jsonl", rows)
    out = load_per_key_finite_history(p, ["a", "b"])
    assert out["a"] == [1.0, 2.0, 3.0]
    assert out["b"] == [5.0, 6.0]


def test_load_per_key_finite_history_uncapped(tmp_path: Path) -> None:
    """Helper must not cap — phase3 needs every finite sample so the
    server-side bind_keys takes the true last 50 across the whole dump.
    """
    rows = [_row({"a": float(i)}) for i in range(420)]
    p = _write_jsonl(tmp_path / "big.jsonl", rows)
    out = load_per_key_finite_history(p, ["a"])
    assert len(out["a"]) == 420
    # Order preserved -> last 50 = [370..419]
    assert out["a"][-50:] == [float(i) for i in range(370, 420)]


def test_warmup_pool_buffer_matches_solver_last_50(tmp_path: Path) -> None:
    """G2 R1 B1 lock-in: the buffer the runner sends to
    ``preload_normalizer_buffer`` must yield the same saturated state
    that ``reconstruct_scores`` operated under — concretely, taking the
    last 50 finite values from the runner buffer must equal the last 50
    finite values the solver's CalibrationSamples would have preloaded.

    Catches the regression where the runner caps the buffer at the
    first 400 samples while the solver uses every row: those two
    last-50 windows would diverge for any recipe with > 400 finite
    samples per key.
    """
    rows = [_row({"a": float(i + 1)}) for i in range(420)]
    p = _write_jsonl(tmp_path / "warmup.jsonl", rows)

    # Runner side: load_per_key_finite_history -> server bind_keys -> last 50.
    runner_buffer = load_per_key_finite_history(p, ["a"])
    runner_last_50 = runner_buffer["a"][-50:]

    # Solver side: rebuild the same CalibrationSamples view used by
    # reconstruct_scores and emulate bind_keys' last-50-non-NaN preload.
    per_key_history: dict[str, list[float]] = {"a": []}
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            v = (row.get("factor_raw") or {}).get("a", float("nan"))
            per_key_history["a"].append(float(v))
    solver_non_nan = [v for v in per_key_history["a"] if not math.isnan(v)]
    solver_last_50 = solver_non_nan[-50:]

    assert runner_last_50 == solver_last_50
