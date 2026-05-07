"""Verify the v2_spec yaml generator produces well-formed 4-layer composite
yamls + sibling warmup yamls; exercise the new `factor_keys` / `factor`
helpers + the assembler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.verdict_factor_judge.v2_spec import (
    ALL_DESC,
    CFG_SPECS,
    GENERATION,
    RECIPES,
    W_LR,
    build_eval_yaml,
    build_warmup_yaml,
    composer_warm_fallback,
    composer_weighted_sum,
    factor,
    factor_keys,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def test_factor_block_has_type_and_params() -> None:
    blk = factor("jerk_online_state", windows=[{"past": 1, "future": 1}])
    assert blk["type"] == "jerk_online_state"
    assert blk["params"]["windows"] == [{"past": 1, "future": 1}]


def test_factor_keys_topk_is_single() -> None:
    assert factor_keys("topk_action_variance") == ["topk_action_variance"]


def test_factor_keys_window_template() -> None:
    keys = factor_keys("dispersion_offline_state", windows=[{"past": 5, "future": 5}, {"past": 7, "future": 7}])
    assert keys == [
        "dispersion_offline_state__p5_f5",
        "dispersion_offline_state__p7_f7",
    ]


def test_factor_keys_requires_windows_for_window_factors() -> None:
    with pytest.raises(ValueError, match="requires windows"):
        factor_keys("jerk_online_action")


# ----------------------------------------------------------------------
# Composer helpers
# ----------------------------------------------------------------------


def test_composer_weighted_sum_basic() -> None:
    c = composer_weighted_sum({"k": 1.0}, full_hit_threshold=0.5)
    assert c["type"] == "weighted_sum"
    assert c["weights"] == {"k": 1.0}
    assert c["tier_thresholds"] == {"full_hit": 0.5}


def test_composer_warm_fallback_includes_fallback_t() -> None:
    c = composer_warm_fallback({"k": 1.0}, full_hit_threshold=0.3, warm_fallback_start_t=0.5)
    assert c["type"] == "weighted_sum_with_warm_fallback"
    assert c["warm_fallback_start_t"] == 0.5


# ----------------------------------------------------------------------
# Eval yaml structure
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cfg_id", list(CFG_SPECS.keys()))
def test_build_eval_yaml_has_4_layers(cfg_id: str) -> None:
    factors_blk, composer_blk = next(iter(RECIPES.values()))()
    y = build_eval_yaml(cfg_id, factors_blk, composer_blk)
    judge = y["checkpoints"]["cp1"]["judge"]
    assert judge["type"] == "composite"
    assert "normalization" in judge
    assert "factors" in judge
    assert "calibration" in judge
    assert "composer" in judge


def test_build_eval_yaml_export_factor_outputs_default_true() -> None:
    factors_blk, composer_blk = next(iter(RECIPES.values()))()
    y = build_eval_yaml("spatial16_w8_d4", factors_blk, composer_blk)
    assert y["checkpoints"]["cp1"]["judge"]["export_factor_outputs"] is True


def test_build_eval_yaml_normalization_offline_only() -> None:
    factors_blk, composer_blk = next(iter(RECIPES.values()))()
    y = build_eval_yaml("spatial16_w8_d4", factors_blk, composer_blk)
    assert y["checkpoints"]["cp1"]["judge"]["normalization"]["stats_source"]["type"] == "offline"


def test_build_eval_yaml_calibration_warmup_default() -> None:
    factors_blk, composer_blk = next(iter(RECIPES.values()))()
    y = build_eval_yaml("spatial16_w8_d4", factors_blk, composer_blk)
    assert y["checkpoints"]["cp1"]["judge"]["calibration"]["samples_source"]["type"] == "warmup"


# ----------------------------------------------------------------------
# Warmup yaml structure
# ----------------------------------------------------------------------


def test_build_warmup_yaml_dump_covers_17_factors() -> None:
    y = build_warmup_yaml("spatial16_w8_d4", "spatial16_w8_d4_v2_jerk_online_state_lr")
    dump = y["checkpoints"]["cp1"]["judge"]["dump"]
    assert dump["deferred"] is True
    assert dump["config_id"].endswith("__warmup")
    factor_types = sorted({f["type"] for f in dump["factors"]})
    expected = sorted({
        f"{desc}_{source}_{channel}"
        for desc in ALL_DESC for source in ("online", "offline") for channel in ("action", "state")
    } | {"topk_action_variance"})
    assert factor_types == expected


def test_build_warmup_yaml_inner_judge_is_always_warm_start() -> None:
    y = build_warmup_yaml("spatial16_w8_d4", "spatial16_w8_d4_v2_demo")
    judge = y["checkpoints"]["cp1"]["judge"]
    assert judge["type"] == "always_warm_start"
    assert judge["start_t"] == 0.7


def test_build_warmup_yaml_top_k_widened_to_5() -> None:
    """topk_action_variance dump factor needs ≥ 5 candidates."""
    y = build_warmup_yaml("spatial16_w8_d4", "spatial16_w8_d4_v2_demo")
    assert y["checkpoints"]["cp1"]["search_strategy"]["top_k"] == 5


# ----------------------------------------------------------------------
# Generation list invariants
# ----------------------------------------------------------------------


def test_generation_stems_are_unique() -> None:
    stems = [g[1] for g in GENERATION]
    assert len(stems) == len(set(stems))


def test_generation_each_recipe_yields_valid_yaml_pair() -> None:
    for cfg_id, stem, factors_blk, composer_blk in GENERATION:
        ev = build_eval_yaml(cfg_id, factors_blk, composer_blk)
        wm = build_warmup_yaml(cfg_id, stem)
        assert ev["checkpoints"]["cp1"]["judge"]["type"] == "composite"
        assert wm["checkpoints"]["cp1"]["judge"]["dump"]["config_id"] == f"{stem}__warmup"


# ----------------------------------------------------------------------
# Window banks
# ----------------------------------------------------------------------


def test_w_lr_is_long_risk_pair() -> None:
    assert W_LR == [{"past": 5, "future": 5}, {"past": 7, "future": 7}]
