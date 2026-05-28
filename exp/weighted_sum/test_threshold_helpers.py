"""Unit tests for the threshold-pareto helper logic (solve + inf_ratio aggregate).

Covers the edge cases flagged in G1 review: narrow/degenerate distributions
(T_ws >= T_fh -> skip), None/missing cp1_score filtering, and the
inference_ratio aggregation (FULL_HIT 0 / WARM_START@0.5 0.75 / MISS 1).
"""

from __future__ import annotations

import json

from exp.weighted_sum.solve_thresholds import (
    load_cp1_scores,
    solve_quantile,
    solve_zscore,
)
from exp.weighted_sum.summarize_inf_ratio import summarize_inf_ratio


# ---------------------------------------------------------------- inf_ratio
def _write_steps(tmp_path, rows):
    p = tmp_path / "per_step.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_inf_ratio_all_full_hit(tmp_path):
    rows = [{"yaml_id": "y", "hit_type": "FULL_HIT"} for _ in range(10)]
    res = summarize_inf_ratio(_write_steps(tmp_path, rows))
    assert res["y"]["inference_ratio"] == 0.0


def test_inf_ratio_all_miss(tmp_path):
    rows = [{"yaml_id": "y", "hit_type": "MISS"} for _ in range(10)]
    res = summarize_inf_ratio(_write_steps(tmp_path, rows))
    assert res["y"]["inference_ratio"] == 1.0


def test_inf_ratio_all_warm_at_half(tmp_path):
    rows = [{"yaml_id": "y", "hit_type": "WARM_START", "start_t": 0.5} for _ in range(8)]
    res = summarize_inf_ratio(_write_steps(tmp_path, rows))
    assert abs(res["y"]["inference_ratio"] - 0.75) < 1e-9


def test_inf_ratio_mixed_and_enum_repr(tmp_path):
    # 2 FH (0) + 2 WS@0.5 (0.75) + 1 MISS (1) over 5 -> (0+1.5+1)/5 = 0.5
    rows = [
        {"yaml_id": "y", "hit_type": "HitType.FULL_HIT"},
        {"yaml_id": "y", "hit_type": "FULL_HIT"},
        {"yaml_id": "y", "hit_type": "WARM_START", "start_t": 0.5},
        {"yaml_id": "y", "hit_type": "HitType.WARM_START", "start_t": 0.5},
        {"yaml_id": "y", "hit_type": "MISS"},
    ]
    res = summarize_inf_ratio(_write_steps(tmp_path, rows))
    assert abs(res["y"]["inference_ratio"] - 0.5) < 1e-9
    assert res["y"]["n_full_hit"] == 2 and res["y"]["n_warm_start"] == 2 and res["y"]["n_miss"] == 1


# ---------------------------------------------------------------- solver
def test_load_cp1_scores_filters_none_and_nan(tmp_path):
    rows = [
        {"cp1_score": 0.9}, {"cp1_score": None}, {"cp1_score": 0.5},
        {"cp1_score": float("nan")}, {"winner_id": "x"},  # missing key
    ]
    scores = load_cp1_scores(_write_steps(tmp_path, rows))
    assert sorted(scores) == [0.5, 0.9]


def test_solve_quantile_spread_orders():
    scores = [i / 100.0 for i in range(101)]  # 0.00..1.00 uniform
    res = solve_quantile(scores, fh_ratio=0.2, ws_ratio=0.3)
    assert res is not None
    t_fh, t_ws = res
    assert t_fh > t_ws  # strict, loadable


def test_solve_quantile_degenerate_constant_returns_none():
    scores = [0.7] * 200  # zero spread -> tiers collapse
    assert solve_quantile(scores, fh_ratio=0.2, ws_ratio=0.3) is None


def test_solve_zscore_spread_and_degenerate():
    spread = [i / 100.0 for i in range(101)]
    res = solve_zscore(spread, k_fh=0.5, k_ws=-0.5)
    assert res is not None and res[0] > res[1]
    assert solve_zscore([0.7] * 50, k_fh=0.5, k_ws=-0.5) is None  # sigma~0
