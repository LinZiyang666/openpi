"""Unit tests for MarginGateCalibrator (TRACER Phase 5 offline calibration).

Covers the L_cal replay metrics, per-lambda b0 recomputation, snapshot-gated
WARM_START, grid argmin, and the deterministic tie-break.
"""

from __future__ import annotations

import pytest

from openpi.cache.components.margin_gate_calibrator import (
    CalibrationGrid,
    MarginGateCalibrator,
)


def _row(s_pos, *, s_neg=0.0, delta_pos=0.0, u_t=None, snapshot=True, success=True):
    return {
        "s_pos": s_pos, "s_neg": s_neg, "delta_pos": delta_pos, "u_t": u_t,
        "winner_has_warm_snapshot": snapshot, "episode_success": success,
    }


def _params(b0, *, lam=0.0, b2=0.0, b3=0.0, warm=None):
    return {
        "margin_lambda": lam,
        "gate_betas": {"b0": b0, "b1": 1.0, "b2": b2, "b3": b3},
        "threshold": 0.5,
        "warm_tiers": [] if warm is None else [{"threshold": warm, "start_t": 0.5}],
    }


def test_score_metrics_exact():
    # b0=0, threshold on g=0.5 => FULL_HIT iff margin >= 0.
    rows = [
        _row(1.0, success=True),   # FH, safe
        _row(1.0, success=False),  # FH, bad
        _row(-1.0, success=True),  # MISS
        _row(-1.0, success=False), # MISS
    ]
    m = MarginGateCalibrator.evaluate(rows, _params(0.0), c_miss=1.0, c_warm=0.75)
    assert m["n_full_hit"] == 2 and m["n_miss"] == 2 and m["n_warm_start"] == 0
    assert m["BadHitRate"] == pytest.approx(1.0 / (2 + 1e-8))
    assert m["MissRate"] == pytest.approx(0.5)
    assert m["WarmCost"] == pytest.approx(0.0)
    assert m["L_cal"] == pytest.approx(1.0 / (2 + 1e-8) + 1.0 * 0.5)


def test_warm_start_gated_on_snapshot():
    # margin=-0.5 -> g=sigmoid(-0.5)~0.378 in [0.3, 0.5): WS iff snapshot present.
    rows_yes = [_row(-0.5, snapshot=True)]
    rows_no = [_row(-0.5, snapshot=False)]
    p = _params(0.0, warm=0.3)
    assert MarginGateCalibrator.evaluate(rows_yes, p, c_miss=1.0, c_warm=0.75)["n_warm_start"] == 1
    assert MarginGateCalibrator.evaluate(rows_no, p, c_miss=1.0, c_warm=0.75)["n_warm_start"] == 0
    assert MarginGateCalibrator.evaluate(rows_no, p, c_miss=1.0, c_warm=0.75)["n_miss"] == 1


def test_u_t_none_drops_term():
    # u_t None must not contribute even with a large b2 -> same as margin-only.
    rows = [_row(0.2, u_t=None)]
    with_b2 = MarginGateCalibrator.evaluate(rows, _params(0.0, b2=100.0), c_miss=1.0, c_warm=0.0)
    no_b2 = MarginGateCalibrator.evaluate(rows, _params(0.0, b2=0.0), c_miss=1.0, c_warm=0.0)
    assert with_b2["n_full_hit"] == no_b2["n_full_hit"]


def test_calibrate_finds_grid_argmin():
    # All-success rows with high margin: FULL_HIT everywhere is safe (0 bad hits),
    # so the minimizer should accept them (low miss) -> tau_hit low enough.
    rows = [_row(0.5 + 0.01 * i, success=True) for i in range(20)]
    grid = CalibrationGrid(lambdas=(0.0,), b2s=(0.0,), b3s=(0.0,), warm_thresholds=(None,))
    params = MarginGateCalibrator.calibrate(rows, c_miss=1.0, c_warm=0.75, grid=grid)
    # Brute-force the same grid and confirm calibrate returned a global argmin.
    best = min(
        (MarginGateCalibrator.evaluate(rows, _params(-p_pct_neg), c_miss=1.0, c_warm=0.75)["L_cal"]
         for p_pct_neg in [_pctl(rows, 0.0, p) for p in grid.b0_percentiles]),
    )
    got = MarginGateCalibrator.evaluate(rows, params, c_miss=1.0, c_warm=0.75)["L_cal"]
    assert got == pytest.approx(best)


def _pctl(rows, lam, p):
    from openpi.cache.components.margin_gate_calibrator import _percentile
    margins = sorted(r["s_pos"] - lam * r["s_neg"] for r in rows)
    return _percentile(margins, p)


def test_per_lambda_b0_recomputed():
    # Two lambdas produce different margin distributions -> different b0 ladders.
    rows = [_row(1.0, s_neg=0.5), _row(0.5, s_neg=0.1)]
    b0_l0 = sorted(_pctl(rows, 0.0, p) for p in (50, 90))
    b0_l1 = sorted(_pctl(rows, 1.0, p) for p in (50, 90))
    assert b0_l0 != b0_l1


def test_calibrate_deterministic_and_tie_break():
    rows = [_row(0.3, success=True), _row(0.1, success=False)]
    p1 = MarginGateCalibrator.calibrate(rows, c_miss=1.0, c_warm=0.75)
    p2 = MarginGateCalibrator.calibrate(rows, c_miss=1.0, c_warm=0.75)
    assert p1 == p2  # deterministic


def test_calibrate_empty_rows_raises():
    with pytest.raises(ValueError, match="empty rows"):
        MarginGateCalibrator.calibrate([], c_miss=1.0, c_warm=0.75)


def test_calibrate_tie_break_prefers_higher_tau_hit():
    """Two b0 candidates yield identical L_cal (tie) -> higher tau_hit (lower b0) wins."""
    from openpi.cache.components.margin_gate_calibrator import CalibrationGrid
    # margins [0.1, 0.3]; b0 candidates -0.12 (p10) and -0.28 (p90) both give the
    # SAME verdicts (1 MISS, 1 FH) -> equal L_cal -> tie broken by higher tau_hit.
    rows = [_row(0.1, success=True), _row(0.3, success=True)]
    grid = CalibrationGrid(lambdas=(0.0,), b0_percentiles=(10, 90),
                           b2s=(0.0,), b3s=(0.0,), warm_thresholds=(None,))
    params = MarginGateCalibrator.calibrate(rows, c_miss=1.0, c_warm=0.75, grid=grid)
    assert params["gate_betas"]["b0"] == pytest.approx(-0.28)  # the more conservative tau_hit
