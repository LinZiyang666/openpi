"""Tests for exp/verdict_factor_judge/phase3_spec.py (plan §8.4).

Validates the 11-recipe x 16-cell spec module:
  - RECIPES table contains 11 entries with correct shape
  - GRID has 16 cells covering the {0.2, 0.3, 0.4, 0.5}^2 grid
  - Each recipe's declared_keys, orientations, directions are consistent
  - build_warmup_yaml_for_recipe emits AlwaysWarmStartJudge+DumpingJudge shape
  - build_eval_yaml_for_cell emits a weighted_sum_zero_nan composer
    with the requested thresholds + warm_start_t=0.5
  - recipe_to_solver_recipe produces a solver Recipe with the same keys

Note (G2): plan §6.3 envisioned a "two-pass" emit (placeholder yamls
written to disk, then patched). Current implementation skips the
placeholder step and lets the runner call build_eval_yaml_for_cell
directly with the solver's threshold output — see G2 deviation note in
the plan log §10.4.
"""

from __future__ import annotations

import pytest

from exp.verdict_factor_judge.phase3_spec import (
    GRID,
    RECIPES,
    build_eval_yaml_for_cell,
    build_warmup_yaml_for_recipe,
    recipe_to_solver_recipe,
)
from exp.verdict_factor_judge.phase3_threshold_solver import Recipe


# ----------------------------------------------------------------------
# RECIPES table shape
# ----------------------------------------------------------------------


def test_recipes_count_is_eleven() -> None:
    assert len(RECIPES) == 11


def test_recipes_ids_match_plan() -> None:
    expected = {
        "g1_f1b_t_w_fut_d_all",
        "g2_f1b_t_w_long_risk_d_jerk",
        "g3_f1b_t_w_long_risk_d_all",
        "g4_f1b_t_w_short_d_jerk",
        "g5_f1a_t_d_jerk_dir_pair",
        "g6_f1a_a_d_jerk_curv_pair",
        "g7_f1b_a_w_long_risk_d_jerk",
        "g8_f1a_t_d_curv_only",
        "g9_f1b_t_w_sym_s_d_all",
        "g10_f1b_a_w_fut_d_all",
        "g11_f1a_a_d_curv_only",
    }
    assert set(RECIPES.keys()) == expected


@pytest.mark.parametrize("rid", list(RECIPES.keys()))
def test_recipe_entry_has_required_fields(rid: str) -> None:
    r = RECIPES[rid]
    assert r["recipe_id"] == rid
    assert isinstance(r["factors"], list) and len(r["factors"]) >= 1
    assert isinstance(r["declared_keys"], list) and len(r["declared_keys"]) >= 1
    # orientations covers every declared key
    assert set(r["orientations"].keys()) == set(r["declared_keys"])
    # orientations are valid
    for ori in r["orientations"].values():
        assert ori in {"safe", "risky", "non_monotonic"}
    # directions only present for non_monotonic descriptors
    for k, dirn in r["directions"].items():
        assert r["orientations"][k] == "non_monotonic"
        assert dirn in {"high", "low"} or dirn.startswith("range:")


def test_g1_recipe_factors_match_plan() -> None:
    """Spot-check g1: 4 desc x offline x state x [(0,3),(0,5)] = 8 keys."""
    r = RECIPES["g1_f1b_t_w_fut_d_all"]
    assert len(r["factors"]) == 4
    assert len(r["declared_keys"]) == 8
    expected_keys = {
        f"{desc}_offline_state__p0_f{f}"
        for desc in ("jerk", "direction", "dispersion", "path_length")
        for f in (3, 5)
    }
    assert set(r["declared_keys"]) == expected_keys


def test_g4_recipe_factors_match_plan() -> None:
    """Spot-check g4: jerk x offline x state x [(0,3),(1,1),(3,0)] = 3 keys."""
    r = RECIPES["g4_f1b_t_w_short_d_jerk"]
    assert len(r["factors"]) == 1
    assert set(r["declared_keys"]) == {
        "jerk_offline_state__p0_f3",
        "jerk_offline_state__p1_f1",
        "jerk_offline_state__p3_f0",
    }
    # All risky (jerk).
    assert all(o == "risky" for o in r["orientations"].values())
    assert r["directions"] == {}


def test_g8_dispersion_directions_range() -> None:
    """dispersion descriptor -> range:[0.3, 0.7] direction."""
    r = RECIPES["g8_f1a_t_d_curv_only"]
    assert r["declared_keys"] == ["dispersion_online_state__p3_f3"]
    assert r["orientations"] == {"dispersion_online_state__p3_f3": "non_monotonic"}
    assert r["directions"]["dispersion_online_state__p3_f3"].startswith("range:")


def test_g1_path_length_directions_high() -> None:
    """path_length descriptor -> high direction."""
    r = RECIPES["g1_f1b_t_w_fut_d_all"]
    pl_keys = [k for k in r["declared_keys"] if k.startswith("path_length_")]
    assert len(pl_keys) == 2
    for k in pl_keys:
        assert r["directions"][k] == "high"


# ----------------------------------------------------------------------
# GRID
# ----------------------------------------------------------------------


def test_grid_has_16_cells() -> None:
    assert len(GRID) == 16


def test_grid_covers_expected_pairs() -> None:
    assert set(GRID) == {
        (fh, ws) for fh in (0.2, 0.3, 0.4, 0.5) for ws in (0.2, 0.3, 0.4, 0.5)
    }


# ----------------------------------------------------------------------
# Warmup yaml shape
# ----------------------------------------------------------------------


def test_warmup_yaml_uses_always_warm_start_and_dump() -> None:
    y = build_warmup_yaml_for_recipe("spatial16_w8_d4", "g4_f1b_t_w_short_d_jerk")
    judge = y["checkpoints"]["cp1"]["judge"]
    assert judge["type"] == "always_warm_start"
    assert judge["start_t"] == 0.7
    assert "dump" in judge
    assert judge["dump"]["deferred"] is True
    assert "factors" in judge["dump"]


def test_warmup_yaml_dump_factor_list_covers_recipe_factor_keys() -> None:
    """Plan §6.4 / v2_spec _build_dump_factor_superset: warmup must dump
    every key the recipe will need at eval time."""
    rid = "g3_f1b_t_w_long_risk_d_all"
    y = build_warmup_yaml_for_recipe("spatial16_w8_d4", rid)
    dump_factors = y["checkpoints"]["cp1"]["judge"]["dump"]["factors"]
    # Build the union of keys those dump factors will produce.
    dumped_keys: set[str] = set()
    for f in dump_factors:
        if f["type"] == "topk_action_variance":
            dumped_keys.add("topk_action_variance")
            continue
        for w in f["params"]["windows"]:
            dumped_keys.add(f"{f['type']}__p{w['past']}_f{w['future']}")
    # Every recipe key must be in the dump set.
    assert set(RECIPES[rid]["declared_keys"]) <= dumped_keys


def test_warmup_yaml_preload_path_canonical() -> None:
    """G1 R1 B3 fix: path locked to canonical exp/common/data/.../*.pkl."""
    y = build_warmup_yaml_for_recipe("spatial16_w8_d4", "g1_f1b_t_w_fut_d_all")
    pp = y["backend"]["in_memory"]["preload_path"]
    assert pp == "exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl"


# ----------------------------------------------------------------------
# Eval yaml shape
# ----------------------------------------------------------------------


def test_eval_yaml_composer_is_zero_nan_with_thresholds() -> None:
    y = build_eval_yaml_for_cell(
        "spatial16_w8_d4", "g4_f1b_t_w_short_d_jerk",
        fh_thr=0.74, ws_thr=0.55,
        fh_ratio=0.2, ws_ratio=0.2,
    )
    composer = y["checkpoints"]["cp1"]["judge"]["composer"]
    assert composer["type"] == "weighted_sum_zero_nan"
    assert composer["tier_thresholds"] == {"full_hit": 0.74, "warm_start": 0.55}
    assert composer["warm_start_t"] == 0.5


def test_eval_yaml_weights_match_recipe_declared_keys() -> None:
    y = build_eval_yaml_for_cell(
        "spatial16_w8_d4", "g1_f1b_t_w_fut_d_all",
        fh_thr=0.5, ws_thr=0.2, fh_ratio=0.2, ws_ratio=0.2,
    )
    weights = y["checkpoints"]["cp1"]["judge"]["composer"]["weights"]
    expected = set(RECIPES["g1_f1b_t_w_fut_d_all"]["declared_keys"])
    assert set(weights.keys()) == expected
    assert all(w == 1.0 for w in weights.values())


def test_eval_yaml_directions_propagated_for_non_monotonic() -> None:
    y = build_eval_yaml_for_cell(
        "spatial16_w8_d4", "g9_f1b_t_w_sym_s_d_all",
        fh_thr=0.5, ws_thr=0.2, fh_ratio=0.2, ws_ratio=0.2,
    )
    composer = y["checkpoints"]["cp1"]["judge"]["composer"]
    assert "directions" in composer
    expected = RECIPES["g9_f1b_t_w_sym_s_d_all"]["directions"]
    assert composer["directions"] == expected


def test_eval_yaml_calibration_uses_warmup_source() -> None:
    """Plan §6: eval yaml's Layer 3 reads from WarmupPool[eval_yaml_id]."""
    y = build_eval_yaml_for_cell(
        "spatial16_w8_d4", "g4_f1b_t_w_short_d_jerk",
        fh_thr=0.5, ws_thr=0.2, fh_ratio=0.2, ws_ratio=0.2,
    )
    cal = y["checkpoints"]["cp1"]["judge"]["calibration"]
    assert cal["type"] == "percentile_rolling"
    assert cal["params"] == {"window_size": 50}
    assert cal["samples_source"] == {"type": "warmup"}


# ----------------------------------------------------------------------
# Recipe -> solver Recipe shim
# ----------------------------------------------------------------------


def test_recipe_to_solver_recipe_round_trip() -> None:
    sr = recipe_to_solver_recipe("g6_f1a_a_d_jerk_curv_pair")
    assert isinstance(sr, Recipe)
    assert sr.recipe_id == "g6_f1a_a_d_jerk_curv_pair"
    assert set(sr.declared_keys) == set(RECIPES["g6_f1a_a_d_jerk_curv_pair"]["declared_keys"])
    assert sr.orientations == RECIPES["g6_f1a_a_d_jerk_curv_pair"]["orientations"]
    assert sr.directions == RECIPES["g6_f1a_a_d_jerk_curv_pair"]["directions"]
