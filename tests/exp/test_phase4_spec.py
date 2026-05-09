"""Tests for exp/verdict_factor_judge/phase4_spec.py.

Covers plan §4.1.8 invariants (INV-1 through INV-10) plus phase3 RECIPES
non-mutation, weight-generator numeric sensitivity, cell id stability,
and yaml builder integration with v2_spec.
"""

from __future__ import annotations

import math

import pytest

from exp.verdict_factor_judge import phase3_spec
from exp.verdict_factor_judge.phase4_spec import (
    ANCHOR_SR,
    LOCKED_CELLS,
    P1_OFFLINE_KEYS,
    P2_OFFLINE_KEYS,
    R1_ALPHAS,
    R2_OFFLINE_PATTERNS,
    R3_ONLINE_PATTERNS,
    R4_WINDOW_PATTERNS,
    RECIPES_PHASE4,
    SHARED_ONLINE_KEYS,
    build_phase4_eval_yaml,
    build_phase4_warmup_yaml,
    cell_id_r1,
    cell_id_r2,
    cell_id_r3,
    cell_id_r4,
    generate_r1_weights,
    generate_r2_weights,
    generate_r3_weights,
    generate_r4_weights,
    recipe_to_solver_recipe,
    warmup_eval_yaml_id,
    warmup_yaml_id,
)


# ----------------------------------------------------------------------
# Group A — Recipe shape (5 tests)
# ----------------------------------------------------------------------


def test_recipes_have_two_entries() -> None:
    assert set(RECIPES_PHASE4) == {
        "p1_state_fut_online_act",
        "p2_action_fut_online_act",
    }


def test_p1_declared_keys_count_10() -> None:
    keys = RECIPES_PHASE4["p1_state_fut_online_act"]["declared_keys"]
    assert len(keys) == 10
    assert len(set(keys)) == 10                                         # no duplicates


def test_p2_declared_keys_count_10() -> None:
    keys = RECIPES_PHASE4["p2_action_fut_online_act"]["declared_keys"]
    assert len(keys) == 10
    assert len(set(keys)) == 10


def test_p1_p2_share_online_keys() -> None:
    p1_online = RECIPES_PHASE4["p1_state_fut_online_act"]["online_keys"]
    p2_online = RECIPES_PHASE4["p2_action_fut_online_act"]["online_keys"]
    assert p1_online == p2_online == SHARED_ONLINE_KEYS


def test_offline_keys_use_correct_channel() -> None:
    """p1 -> all offline keys are channel=state; p2 -> channel=action."""
    for k in P1_OFFLINE_KEYS:
        assert "_offline_state__" in k
    for k in P2_OFFLINE_KEYS:
        assert "_offline_action__" in k


# ----------------------------------------------------------------------
# Group B — Weight invariants (INV-1..INV-7), parametrized
# ----------------------------------------------------------------------


@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
@pytest.mark.parametrize("alpha", R1_ALPHAS)
def test_r1_weights_sum_to_one(rid: str, alpha: float) -> None:
    """INV-2: weights sum to 1.0 (FP epsilon)."""
    w = generate_r1_weights(rid, alpha)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)


@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
@pytest.mark.parametrize("alpha", R1_ALPHAS)
def test_r1_weights_have_exactly_declared_keys(rid: str, alpha: float) -> None:
    """INV-1: weight dict keys equal declared_keys (10 entries)."""
    w = generate_r1_weights(rid, alpha)
    expected = set(RECIPES_PHASE4[rid]["declared_keys"])
    assert set(w) == expected
    assert len(w) == 10


@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
def test_r1_alpha0_zeros_offline(rid: str) -> None:
    """INV-3: alpha=0 -> all offline keys = 0; online uniform = 0.5 each."""
    w = generate_r1_weights(rid, 0.0)
    for k in RECIPES_PHASE4[rid]["offline_keys"]:
        assert w[k] == 0.0
    for k in RECIPES_PHASE4[rid]["online_keys"]:
        assert math.isclose(w[k], 0.5, abs_tol=1e-9)


@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
def test_r1_alpha1_zeros_online(rid: str) -> None:
    """INV-4: alpha=1 -> all online keys = 0; offline uniform = 1/8 each."""
    w = generate_r1_weights(rid, 1.0)
    for k in RECIPES_PHASE4[rid]["online_keys"]:
        assert w[k] == 0.0
    for k in RECIPES_PHASE4[rid]["offline_keys"]:
        assert math.isclose(w[k], 1.0 / 8, abs_tol=1e-9)


@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
@pytest.mark.parametrize("pat_name,pat", list(R2_OFFLINE_PATTERNS.items()))
def test_r2_weights_sum_to_one(
    rid: str, pat_name: str, pat: tuple[int, int, int, int],
) -> None:
    w = generate_r2_weights(rid, alpha_star=0.5, offline_pattern=pat)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9), (
        f"pattern {pat_name} on {rid}: sum {sum(w.values())} != 1.0"
    )


@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
def test_r2_jerk_only_zeros_other_desc(rid: str) -> None:
    """INV-5: pattern (1,0,0,0) zeroes direction/dispersion/path_length offline keys."""
    w = generate_r2_weights(rid, alpha_star=0.5, offline_pattern=(1, 0, 0, 0))
    for k in RECIPES_PHASE4[rid]["offline_keys"]:
        if k.startswith("jerk"):
            assert w[k] > 0.0, f"jerk key {k} unexpectedly zeroed"
        else:
            assert w[k] == 0.0, f"non-jerk key {k} should be 0 under jerk-only"


@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
@pytest.mark.parametrize("on_pat_name,on_pat", list(R3_ONLINE_PATTERNS.items()))
def test_r3_weights_sum_to_one(
    rid: str, on_pat_name: str, on_pat: tuple[int, int],
) -> None:
    w = generate_r3_weights(
        rid, alpha_star=0.4, offline_pattern=(1, 1, 1, 1), online_pattern=on_pat,
    )
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9), (
        f"online pattern {on_pat_name} on {rid}: sum {sum(w.values())} != 1.0"
    )


@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
def test_r3_disp_only_zeros_jerk_online(rid: str) -> None:
    """INV-6: online pattern (0,1) zeroes jerk_online_action; only dispersion_online_action."""
    w = generate_r3_weights(
        rid, alpha_star=0.5, offline_pattern=(1, 1, 1, 1), online_pattern=(0, 1),
    )
    jerk_online = "jerk_online_action__p3_f3"
    disp_online = "dispersion_online_action__p3_f3"
    assert w[jerk_online] == 0.0
    assert math.isclose(w[disp_online], 0.5, abs_tol=1e-9)


@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
@pytest.mark.parametrize("win_pat_name,win_pat", list(R4_WINDOW_PATTERNS.items()))
def test_r4_weights_sum_to_one(
    rid: str, win_pat_name: str, win_pat: tuple[int, int],
) -> None:
    w = generate_r4_weights(
        rid, alpha_star=0.5, offline_pattern=(1, 1, 1, 1),
        online_pattern=(1, 1), window_pattern=win_pat,
    )
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9), (
        f"window pattern {win_pat_name}: sum {sum(w.values())} != 1.0"
    )


@pytest.mark.parametrize("rid", list(RECIPES_PHASE4))
def test_r4_short_only_zeros_long_window(rid: str) -> None:
    """INV-7: window pattern (1,0) zeroes p0_f5 keys; only p0_f3 weighted."""
    w = generate_r4_weights(
        rid, alpha_star=0.5, offline_pattern=(1, 1, 1, 1),
        online_pattern=(1, 1), window_pattern=(1, 0),
    )
    for k in RECIPES_PHASE4[rid]["offline_keys"]:
        if k.endswith("p0_f5"):
            assert w[k] == 0.0, f"long window key {k} should be 0 under short-only"
        elif k.endswith("p0_f3"):
            assert w[k] > 0.0, f"short window key {k} should be > 0 under short-only"


# ----------------------------------------------------------------------
# Group C — Cell ID + warmup id stability (5 tests)
# ----------------------------------------------------------------------


def test_cell_id_r1_format() -> None:
    cid = cell_id_r1("spatial16_w8_d4", "p1_state_fut_online_act", 0.4)
    assert cid == "spatial16_w8_d4_phase4_p1_state_fut_online_act__r1_a0.4"


def test_cell_id_r2_includes_pattern_name() -> None:
    cid = cell_id_r2(
        "spatial16_w8_d4", "p1_state_fut_online_act", 0.4, "jerk-heavy",
    )
    assert "r2_a0.4" in cid
    assert "off-jerk-heavy" in cid


def test_cell_id_uniqueness_across_rounds() -> None:
    """The same (rid, alpha) yields different cell_ids across rounds."""
    rid = "p1_state_fut_online_act"
    a = 0.5
    ids = {
        cell_id_r1("c", rid, a),
        cell_id_r2("c", rid, a, "uniform"),
        cell_id_r3("c", rid, a, "uniform", "uniform"),
        cell_id_r4("c", rid, a, "uniform", "uniform", "uniform"),
    }
    assert len(ids) == 4


def test_warmup_yaml_id_distinct_from_eval_anchor() -> None:
    """warmup_yaml_id ends in __warmup; warmup_eval_yaml_id is the anchor stem."""
    w = warmup_yaml_id("c", "p1_state_fut_online_act")
    e = warmup_eval_yaml_id("c", "p1_state_fut_online_act")
    assert w == e + "__warmup"
    assert e == "c_phase4_p1_state_fut_online_act"


def test_warmup_eval_yaml_id_matches_phase3_format() -> None:
    """Phase 3 used <cfg>_phase3_<recipe> as warmup_eval anchor; phase4 uses
    <cfg>_phase4_<recipe>. The pattern must remain phase-prefixed for the
    server's WarmupPool keying."""
    e = warmup_eval_yaml_id("spatial16_w8_d4", "p2_action_fut_online_act")
    assert e == "spatial16_w8_d4_phase4_p2_action_fut_online_act"


# ----------------------------------------------------------------------
# Group D — Yaml builder integration (5 tests)
# ----------------------------------------------------------------------


def test_build_warmup_yaml_for_recipe_has_correct_factor_count() -> None:
    """Phase4 warmup must dump factors covering all 10 declared keys
    (the v2_spec builder may dump a superset for shared windows)."""
    y = build_phase4_warmup_yaml("spatial16_w8_d4", "p1_state_fut_online_act")
    judge = y["checkpoints"]["cp1"]["judge"]
    assert judge["type"] == "always_warm_start"
    assert "dump" in judge
    dumped = judge["dump"]["factors"]
    assert isinstance(dumped, list) and len(dumped) > 0


def test_build_eval_yaml_uses_provided_weights() -> None:
    """Eval yaml's composer.weights MUST match the dict passed in."""
    rid = "p1_state_fut_online_act"
    weights = generate_r1_weights(rid, 0.4)
    y = build_phase4_eval_yaml(
        cfg_id="spatial16_w8_d4",
        recipe_id=rid,
        fh_thr=0.6,
        ws_thr=0.4,
        fh_ratio=0.5, ws_ratio=0.5,
        composer_weights=weights,
    )
    composer = y["checkpoints"]["cp1"]["judge"]["composer"]
    assert composer["weights"] == weights


def test_build_eval_yaml_locks_warm_start_t_at_0_5() -> None:
    rid = "p1_state_fut_online_act"
    weights = generate_r1_weights(rid, 0.5)
    y = build_phase4_eval_yaml(
        cfg_id="spatial16_w8_d4", recipe_id=rid,
        fh_thr=0.6, ws_thr=0.4, fh_ratio=0.5, ws_ratio=0.5,
        composer_weights=weights,
    )
    composer = y["checkpoints"]["cp1"]["judge"]["composer"]
    assert composer["warm_start_t"] == 0.5


def test_build_eval_yaml_passes_through_thr() -> None:
    rid = "p2_action_fut_online_act"
    weights = generate_r2_weights(rid, 0.4, R2_OFFLINE_PATTERNS["uniform"])
    y = build_phase4_eval_yaml(
        cfg_id="spatial16_w8_d4", recipe_id=rid,
        fh_thr=0.7321, ws_thr=0.4567,
        fh_ratio=0.5, ws_ratio=0.4,
        composer_weights=weights,
    )
    composer = y["checkpoints"]["cp1"]["judge"]["composer"]
    assert composer["tier_thresholds"]["full_hit"] == 0.7321
    assert composer["tier_thresholds"]["warm_start"] == 0.4567


def test_build_eval_yaml_directions_match_recipe() -> None:
    rid = "p1_state_fut_online_act"
    weights = generate_r1_weights(rid, 0.5)
    y = build_phase4_eval_yaml(
        cfg_id="spatial16_w8_d4", recipe_id=rid,
        fh_thr=0.5, ws_thr=0.3, fh_ratio=0.5, ws_ratio=0.5,
        composer_weights=weights,
    )
    composer = y["checkpoints"]["cp1"]["judge"]["composer"]
    expected_directions = RECIPES_PHASE4[rid]["directions"]
    assert composer.get("directions") == expected_directions


def test_build_eval_yaml_rejects_wrong_weight_shape() -> None:
    """Weight dict missing a declared key fails fast."""
    rid = "p1_state_fut_online_act"
    bad_weights = {"not_a_real_key": 1.0}
    with pytest.raises(AssertionError):
        build_phase4_eval_yaml(
            cfg_id="spatial16_w8_d4", recipe_id=rid,
            fh_thr=0.5, ws_thr=0.3, fh_ratio=0.5, ws_ratio=0.5,
            composer_weights=bad_weights,
        )


# ----------------------------------------------------------------------
# Group E — phase3 RECIPES non-mutation (G1 R2 Blocking 2)
# ----------------------------------------------------------------------


def test_phase4_import_does_not_mutate_phase3_recipes() -> None:
    """Importing phase4_spec must NOT add p1/p2 to phase3_spec.RECIPES.

    This guard catches the regression where phase4 mutates the phase3
    global with `phase3_spec.RECIPES.update(...)`. After phase4 is
    loaded, phase3_spec.RECIPES must still contain exactly the 11 g*
    recipes from phase 3.
    """
    keys = set(phase3_spec.RECIPES.keys())
    assert len(keys) == 11
    assert all(k.startswith("g") for k in keys)
    assert "p1_state_fut_online_act" not in keys
    assert "p2_action_fut_online_act" not in keys


def test_phase4_recipes_phase4_dict_is_independent_object() -> None:
    """RECIPES_PHASE4 is a separate dict object from phase3_spec.RECIPES."""
    assert RECIPES_PHASE4 is not phase3_spec.RECIPES
    assert id(RECIPES_PHASE4) != id(phase3_spec.RECIPES)


# ----------------------------------------------------------------------
# Group F — Anchor / locked cell metadata
# ----------------------------------------------------------------------


def test_locked_cells_p1_p2_match_plan() -> None:
    assert LOCKED_CELLS["p1_state_fut_online_act"] == (0.5, 0.5)
    assert LOCKED_CELLS["p2_action_fut_online_act"] == (0.5, 0.4)


def test_anchor_sr_p1_p2_match_phase3_data() -> None:
    """Per plan §0.2: p1 anchor SR=0.95, p2 anchor SR=0.96."""
    assert ANCHOR_SR["p1_state_fut_online_act"] == 0.95
    assert ANCHOR_SR["p2_action_fut_online_act"] == 0.96


def test_recipe_to_solver_recipe_carries_orientations_and_directions() -> None:
    sr = recipe_to_solver_recipe("p1_state_fut_online_act")
    expected = RECIPES_PHASE4["p1_state_fut_online_act"]
    assert sr.recipe_id == "p1_state_fut_online_act"
    assert set(sr.declared_keys) == set(expected["declared_keys"])
    assert sr.orientations == expected["orientations"]
    assert sr.directions == expected["directions"]
