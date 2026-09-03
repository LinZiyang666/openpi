"""K-tier RIT-PL: parity with rit_pl at K=2, nesting and costs at K=3, IR inverse."""

from __future__ import annotations

import numpy as np
import pytest

from exp.dispatch_surface import rit_pl
from exp.dispatch_surface.analysis.analytic_cost import STAGE1_MS, STAGE2_MS, STAGE3_MS, unit_cost
from exp.rit_pareto import rit_k


def _table(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.6, 1.0, size=n)
    base = 12.0 * (1.0 - s) + 1.0
    y10 = base + rng.gamma(2.0, 0.6, size=n)
    y7 = 0.7 * base + rng.gamma(2.0, 0.5, size=n)
    y5 = 0.5 * base + rng.gamma(2.0, 0.4, size=n)
    return s, {"full": y10, "warm03": y7, "warm05": y5}


def test_tier_costs_follow_the_stage_formula_and_pin_the_authority():
    assert rit_k.tier_cost("FULL_HIT", None) == unit_cost("FULL_HIT", None)
    assert rit_k.tier_cost("MISS", None) == unit_cost("MISS", None)
    assert rit_k.tier_cost("WARM_START", 0.3) == unit_cost("WARM_START", 0.3)
    assert rit_k.tier_cost("WARM_START", 0.5) == pytest.approx(STAGE1_MS + STAGE2_MS + 0.5 * STAGE3_MS)
    with pytest.raises(ValueError):
        rit_k.tier_cost("WARM_START", 0.33)
    costs = [t.cost_ms for t in rit_k.K3_TIERS]
    assert costs == sorted(costs) and costs[-1] < rit_k.MISS_MS


def test_k2_ladder_reproduces_rit_pl_exactly():
    s, ys = _table()
    knots, n_req = rit_pl.choose_knots(s, rit_pl.KNOT_LADDER)
    ref = rit_pl.fit_pl_quantile(s, ys["warm03"], ys["full"], knots, n_seg_req=n_req,
                                 alpha=0.05, eps_total=rit_pl.EPS_TOTAL)
    fit = rit_k.fit_pl_quantile_k(s, ys, knots, tiers=rit_k.K2_TIERS, n_seg_req=n_req,
                                  alpha=0.05, eps_total=rit_pl.EPS_TOTAL)
    assert np.array_equal(fit.q["full"], ref.q_full)
    assert np.array_equal(fit.q["warm03"], ref.q_warm)
    for delta in (4.0, 6.0, 8.0):
        tf, tw = rit_pl.cuts(ref, delta)
        th = rit_k.cuts(fit, delta)
        assert th["full"] == tf and th["warm03"] == tw
        assert rit_k.predicted_ir(s, [tf, tw], rit_k.K2_TIERS) == pytest.approx(rit_pl.predicted_ir(s, tf, tw))
    a = rit_pl.delta_for_ir(ref, s, 60.0)
    b = rit_k.delta_for_ir(fit, s, 60.0)
    assert a["delta"] == b["delta"] and a["predicted_ir"] == b["predicted_ir"]


def test_k3_fit_is_nested_and_cuts_are_ordered():
    s, ys = _table()
    knots, n_req = rit_k.choose_knots(s, rit_k.KNOT_LADDER)
    fit = rit_k.fit_pl_quantile_k(s, ys, knots, n_seg_req=n_req, alpha=0.05, eps_total=rit_k.EPS_TOTAL)
    assert (fit.q["warm05"] <= fit.q["warm03"] + 1e-9).all()
    assert (fit.q["warm03"] <= fit.q["full"] + 1e-9).all()
    for name in ("full", "warm03", "warm05"):
        assert (np.diff(fit.q[name]) < 0).all()
    lo, hi = rit_k.attainable_range(fit, s)
    assert lo < hi <= 100.0
    for target in (25.0, 50.0, 75.0):
        sol = rit_k.delta_for_ir(fit, s, target)
        th = [sol["thetas"][t.name] for t in fit.tiers]
        assert th[0] >= th[1] >= th[2]
        assert abs(sol["ir_gap"]) <= rit_k.IR_MAX_GAP
        assert sol["predicted_ir"] == pytest.approx(rit_k.predicted_ir(s, th))
    info = rit_k.floor_info(fit, s, sol["delta"])
    assert set(info) == {"full", "warm03", "warm05"}


def test_verdict_takes_the_cheapest_admissible_tier():
    s = np.array([0.99, 0.95, 0.90, 0.80])
    idx = rit_k.verdict_index(s, [0.98, 0.93, 0.85])
    assert idx.tolist() == [0, 1, 2, 3]
    idx = rit_k.verdict_index(s, [float("inf"), 0.93, 0.85])
    assert idx.tolist() == [1, 1, 2, 3]
    ir = rit_k.predicted_ir(s, [0.98, 0.93, 0.85])
    exp_cost = (STAGE1_MS + rit_k.tier_cost("WARM_START", 0.3) + rit_k.tier_cost("WARM_START", 0.5) + rit_k.MISS_MS)
    assert ir == pytest.approx(100.0 * exp_cost / (4 * rit_k.MISS_MS))


def test_record_round_trip():
    s, ys = _table(n=200, seed=3)
    knots, n_req = rit_k.choose_knots(s, rit_k.KNOT_LADDER)
    fit = rit_k.fit_pl_quantile_k(s, ys, knots, n_seg_req=n_req, alpha=0.05, eps_total=rit_k.EPS_TOTAL)
    back = rit_k.fit_from_record(rit_k.fit_record_fields(fit))
    assert rit_k.fit_digests(back) == rit_k.fit_digests(fit)
    assert back.tiers == fit.tiers
