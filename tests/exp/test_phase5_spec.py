"""Unit tests for exp.verdict_factor_judge.phase5.spec.

Coverage maps to plan §5.1 INV-1..INV-11. Highlights:

- INV-2  240 cell + unique yaml_id
- INV-7  4:4:4:3:3:3 alloc → 48/48/45/33/33/33
- INV-9  G3 weight formula: offline 0.5 + online 0.5, degenerate (1,0) gives disp_w=0
- INV-10 G1/G3/G4 base offline keys carry the channel of base_recipe
- INV-11 reconstruct_scores(..., composer_weights=...) with two distinct weight
  vectors yields different (FH_thr, WS_thr) — the regression that catches a
  silent fall-back to solve_recipe / equal weights.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from exp.verdict_factor_judge.phase5.spec import (
    BASE_RECIPE_TO_CHANNEL,
    Cell,
    G1_BASE_RECIPES,
    G1_CHANNELS,
    G1_DESCS,
    G1_WINDOWS,
    G2_BASE_RECIPES,
    G2_CHANNELS,
    G2_DESCS,
    G2_MULTI_COMBOS,
    G3_BASE_RECIPES,
    G3_CHANNELS,
    G3_WEIGHT_PATTERNS,
    G4_BASE_RECIPES,
    G4_CHANNELS,
    G4_FACTOR_SUBSETS,
    G5_RECIPES,
    G5_THRESHOLD_GRID,
    LOCKED_FH,
    LOCKED_WS,
    allocate_to_servers,
    build_eval_yaml_for_cell,
    build_warmup_yaml_for_cell,
    cell_to_solver_recipe,
    generate_all_cells,
    generate_g1_cells,
    generate_g2_cells,
    generate_g3_cells,
    generate_g4_cells,
    generate_g5_cells,
)


# ----------------------------------------------------------------------
# INV-1 / INV-2 — cell counts + yaml_id uniqueness
# ----------------------------------------------------------------------


def test_inv1_g1_g2_g3_g4_g5_each_48_cells() -> None:
    assert len(generate_g1_cells()) == 48
    assert len(generate_g2_cells()) == 48
    assert len(generate_g3_cells()) == 48
    assert len(generate_g4_cells()) == 48
    assert len(generate_g5_cells()) == 48


def test_inv2_total_240_cells_unique_yaml_id() -> None:
    cells = generate_all_cells()
    assert len(cells) == 240
    ids = [c.yaml_id for c in cells]
    assert len(set(ids)) == 240, f"duplicate yaml_id: {len(ids) - len(set(ids))} dups"


def test_inv2_yaml_id_sorted_dictionary_order() -> None:
    cells = generate_all_cells()
    ids = [c.yaml_id for c in cells]
    assert ids == sorted(ids)


# ----------------------------------------------------------------------
# INV-3 — axis coverage per group
# ----------------------------------------------------------------------


def test_inv3_g1_axis_full_cartesian() -> None:
    cells = generate_g1_cells()
    seen = {(c.base_recipe, c.axis_tag.split("_")[1], c.axis_tag.split("_")[2].split("__")[0],
             c.axis_tag.split("__win-")[1]) for c in cells}
    expected = {
        (b, ch, d, f"{p}-{f}")
        for b in G1_BASE_RECIPES for ch in G1_CHANNELS
        for d in G1_DESCS for (p, f) in G1_WINDOWS
    }
    assert seen == expected


def test_inv3_g2_axis_full_cartesian() -> None:
    cells = generate_g2_cells()
    expected_count = len(G2_MULTI_COMBOS) * len(G2_CHANNELS) * len(G2_DESCS) * len(G2_BASE_RECIPES)
    assert len(cells) == expected_count


def test_inv3_g3_axis_full_cartesian() -> None:
    cells = generate_g3_cells()
    assert len(cells) == len(G3_WEIGHT_PATTERNS) * len(G3_CHANNELS) * len(G3_BASE_RECIPES)


def test_inv3_g4_axis_full_cartesian() -> None:
    cells = generate_g4_cells()
    assert len(cells) == len(G4_FACTOR_SUBSETS) * len(G4_CHANNELS) * len(G4_BASE_RECIPES)


def test_inv3_g5_axis_full_cartesian() -> None:
    cells = generate_g5_cells()
    assert len(cells) == len(G5_THRESHOLD_GRID) * len(G5_RECIPES)


# ----------------------------------------------------------------------
# INV-4 — declared_keys ⊆ factor list keys
# ----------------------------------------------------------------------


def _all_factor_keys(cell: Cell) -> set[str]:
    """Replicate v2_spec.factor_keys logic: <type>__p<P>_f<F> per window."""
    out: set[str] = set()
    for fb in cell.factors:
        t = fb["type"]
        params = fb.get("params") or {}
        wins = params.get("windows") or []
        if not wins:
            out.add(t)
            continue
        for w in wins:
            out.add(f"{t}__p{w['past']}_f{w['future']}")
    return out


def test_inv4_declared_keys_subset_of_factor_keys() -> None:
    for cell in generate_all_cells():
        factor_set = _all_factor_keys(cell)
        declared = set(cell.declared_keys)
        assert declared <= factor_set, (
            f"cell {cell.yaml_id}: declared - factor = {declared - factor_set}"
        )


def test_inv4_weights_keys_match_declared_keys() -> None:
    for cell in generate_all_cells():
        assert set(cell.weights) == set(cell.declared_keys), (
            f"cell {cell.yaml_id}: weights/declared_keys mismatch"
        )


# ----------------------------------------------------------------------
# INV-5 — G5 cell axes
# ----------------------------------------------------------------------


def test_inv5_g5_cells_cover_grid_x_recipes() -> None:
    cells = generate_g5_cells()
    seen = {(c.base_recipe, c.fh_ratio, c.ws_ratio) for c in cells}
    expected = {
        (short, fh, ws)
        for (fh, ws) in G5_THRESHOLD_GRID
        for short in ("p1", "p2", "g6")
    }
    assert seen == expected


# ----------------------------------------------------------------------
# INV-6 — yaml builders do NOT mutate phase3.RECIPES / phase4 RECIPES_PHASE4
# ----------------------------------------------------------------------


def test_inv6_yaml_builders_do_not_mutate_external_recipes() -> None:
    from exp.verdict_factor_judge.phase3.spec import RECIPES as PHASE3_RECIPES
    from exp.verdict_factor_judge.phase4.spec import RECIPES_PHASE4

    p3_keys_before = set(PHASE3_RECIPES.keys())
    p4_keys_before = set(RECIPES_PHASE4.keys())
    p3_g1_factors_before = list(PHASE3_RECIPES["g1_f1b_t_w_fut_d_all"]["factors"])

    # Build yamls for a sample of cells
    for cell in generate_all_cells()[:10]:
        build_warmup_yaml_for_cell("spatial16_w8_d4", cell)
        build_eval_yaml_for_cell("spatial16_w8_d4", cell, fh_thr=0.3, ws_thr=0.1)

    assert set(PHASE3_RECIPES.keys()) == p3_keys_before
    assert set(RECIPES_PHASE4.keys()) == p4_keys_before
    assert PHASE3_RECIPES["g1_f1b_t_w_fut_d_all"]["factors"] == p3_g1_factors_before


# ----------------------------------------------------------------------
# INV-7 — server allocation: 4:4:4:3:3:3 → 48/48/45/33/33/33
# ----------------------------------------------------------------------


def test_inv7_server_allocation_443333() -> None:
    cells = generate_all_cells()
    alloc = allocate_to_servers(cells)
    assert [len(alloc[s]) for s in ("S1", "S2", "S3", "S4", "S5", "S6")] == [
        48, 48, 45, 33, 33, 33,
    ]
    # All 240 cells distributed exactly once.
    flat = [c.yaml_id for s in alloc.values() for c in s]
    assert sorted(flat) == sorted(c.yaml_id for c in cells)


def test_inv7_server_allocation_custom_ratio() -> None:
    cells = generate_all_cells()
    alloc = allocate_to_servers(cells, ratio=(1, 1, 1, 1, 1, 1))
    counts = sorted(len(v) for v in alloc.values())
    # 240 / 6 = 40 each
    assert counts == [40, 40, 40, 40, 40, 40]


# ----------------------------------------------------------------------
# INV-8 — G5 does NOT inject base offline (factor list = recipe-native)
# ----------------------------------------------------------------------


def test_inv8_g5_does_not_inject_base_offline() -> None:
    """G5 cells use the recipe's existing factor list, not a synthesized
    base offline + sweep. Assert factor count matches the source recipe."""
    from exp.verdict_factor_judge.phase3.spec import RECIPES as PHASE3_RECIPES
    from exp.verdict_factor_judge.phase4.spec import RECIPES_PHASE4

    g5 = generate_g5_cells()
    p1 = next(c for c in g5 if c.base_recipe == "p1")
    p2 = next(c for c in g5 if c.base_recipe == "p2")
    g6 = next(c for c in g5 if c.base_recipe == "g6")

    assert len(p1.factors) == len(RECIPES_PHASE4["p1_state_fut_online_act"]["factors"])
    assert len(p2.factors) == len(RECIPES_PHASE4["p2_action_fut_online_act"]["factors"])
    assert len(g6.factors) == len(PHASE3_RECIPES["g6_f1a_a_d_jerk_curv_pair"]["factors"])


# ----------------------------------------------------------------------
# INV-9 — G3 weight formula
# ----------------------------------------------------------------------


def test_inv9_g3_offline_total_05_online_total_05() -> None:
    cells = generate_g3_cells()
    for c in cells:
        offline_sum = sum(v for k, v in c.weights.items() if "_offline_" in k)
        online_sum = sum(v for k, v in c.weights.items() if "_online_" in k)
        assert math.isclose(offline_sum, 0.5, abs_tol=1e-9), (
            f"{c.yaml_id} offline_sum={offline_sum}"
        )
        assert math.isclose(online_sum, 0.5, abs_tol=1e-9), (
            f"{c.yaml_id} online_sum={online_sum}"
        )


def test_inv9_g3_jerk_only_disp_weight_zero() -> None:
    cells = generate_g3_cells()
    for c in cells:
        if "pat-1-0" not in c.yaml_id:
            continue
        for k, w in c.weights.items():
            if "dispersion_online_" in k:
                assert w == 0.0, f"{c.yaml_id} disp_w={w} (expected 0 for jerk-only)"


def test_inv9_g3_disp_only_jerk_weight_zero() -> None:
    cells = generate_g3_cells()
    for c in cells:
        if "pat-0-1" not in c.yaml_id:
            continue
        for k, w in c.weights.items():
            if "jerk_online_" in k:
                assert w == 0.0, f"{c.yaml_id} jerk_w={w} (expected 0 for disp-only)"


# ----------------------------------------------------------------------
# INV-10 — base offline channel binding (G1/G3/G4)
# ----------------------------------------------------------------------


def test_inv10_g1_base_offline_keys_carry_base_recipe_channel() -> None:
    cells = generate_g1_cells()
    for c in cells:
        expected_channel = BASE_RECIPE_TO_CHANNEL[c.base_recipe]
        # Find offline keys (must be 8 of them, matching expected_channel).
        offline_keys = [k for k in c.declared_keys if "_offline_" in k]
        assert len(offline_keys) == 8, f"{c.yaml_id} has {len(offline_keys)} offline keys"
        for k in offline_keys:
            assert f"_offline_{expected_channel}__" in k, (
                f"{c.yaml_id} offline key {k} does not match channel {expected_channel}"
            )


def test_inv10_g3_base_offline_keys_carry_base_recipe_channel() -> None:
    cells = generate_g3_cells()
    for c in cells:
        expected_channel = BASE_RECIPE_TO_CHANNEL[c.base_recipe]
        offline_keys = [k for k in c.declared_keys if "_offline_" in k]
        assert len(offline_keys) == 8
        for k in offline_keys:
            assert f"_offline_{expected_channel}__" in k


def test_inv10_g4_base_offline_keys_carry_base_recipe_channel() -> None:
    cells = generate_g4_cells()
    for c in cells:
        expected_channel = BASE_RECIPE_TO_CHANNEL[c.base_recipe]
        # G4 may have 0-4 offline desc; all present must use the base channel.
        offline_keys = [k for k in c.declared_keys if "_offline_" in k]
        for k in offline_keys:
            assert f"_offline_{expected_channel}__" in k, (
                f"{c.yaml_id} offline key {k} ≠ channel {expected_channel}"
            )


# ----------------------------------------------------------------------
# INV-11 — composer_weights passthrough produces different thresholds
# ----------------------------------------------------------------------


def _write_warmup_jsonl(path: Path, declared: list[str], n: int = 80) -> None:
    """Synth warmup factor_raw jsonl with monotone ramps per key (so
    weights actually shift the score distribution)."""
    rows = []
    for i in range(n):
        raw = {}
        for j, k in enumerate(declared):
            # Spread keys across [0, 1) with different phase + noise.
            raw[k] = ((i + j * 7) % n) / n
        rows.append({"factor_raw": raw})
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_inv11_composer_weights_shift_thresholds(tmp_path: Path) -> None:
    """Different composer_weights → different (FH_thr, WS_thr).

    This is the regression guard: if phase5 silently calls
    ``solve_recipe`` (which hardcodes equal weights), this test will see
    identical thresholds for the two weight vectors and fail.
    """
    from exp.verdict_factor_judge.phase3.threshold_solver import (
        derive_thresholds, reconstruct_scores,
    )
    cell = generate_g3_cells()[0]   # 10 declared keys
    recipe = cell_to_solver_recipe(cell)

    jsonl = tmp_path / "raw.jsonl"
    _write_warmup_jsonl(jsonl, list(cell.declared_keys), n=80)

    # Vector 1: uniform
    w1 = {k: 1.0 for k in cell.declared_keys}
    scores1 = reconstruct_scores(jsonl, recipe, composer_weights=w1)
    fh1, ws1 = derive_thresholds(scores1, fh_ratio=0.5, ws_ratio=0.5)

    # Vector 2: heavy skew (only first key has weight)
    w2 = {k: 0.0 for k in cell.declared_keys}
    w2[cell.declared_keys[0]] = 1.0
    scores2 = reconstruct_scores(jsonl, recipe, composer_weights=w2)
    fh2, ws2 = derive_thresholds(scores2, fh_ratio=0.5, ws_ratio=0.5)

    assert (fh1, ws1) != (fh2, ws2), (
        f"composer_weights passthrough failed: uniform={fh1, ws1} skew={fh2, ws2}"
    )


# ----------------------------------------------------------------------
# Cell schema sanity
# ----------------------------------------------------------------------


def test_cells_locked_to_fh05_ws05_except_g5() -> None:
    for c in generate_all_cells():
        if c.group == "g5":
            assert (c.fh_ratio, c.ws_ratio) in G5_THRESHOLD_GRID
        else:
            assert c.fh_ratio == LOCKED_FH and c.ws_ratio == LOCKED_WS


def test_warmup_yaml_id_uniqueness_per_group() -> None:
    """G1/G2/G4: 1 warmup per cell; G3: shared per (base, channel); G5: shared per recipe."""
    cells = generate_all_cells()
    g1_warmups = {c.warmup_yaml_id for c in cells if c.group == "g1"}
    g2_warmups = {c.warmup_yaml_id for c in cells if c.group == "g2"}
    g3_warmups = {c.warmup_yaml_id for c in cells if c.group == "g3"}
    g4_warmups = {c.warmup_yaml_id for c in cells if c.group == "g4"}
    g5_warmups = {c.warmup_yaml_id for c in cells if c.group == "g5"}

    assert len(g1_warmups) == 48
    assert len(g2_warmups) == 48
    assert len(g3_warmups) == 4   # 2 base × 2 channel
    assert len(g4_warmups) == 48
    assert len(g5_warmups) == 3   # 3 recipes


def test_warmup_yaml_for_cell_returns_valid_yaml_dict() -> None:
    cell = generate_g1_cells()[0]
    y = build_warmup_yaml_for_cell("spatial16_w8_d4", cell)
    assert "checkpoints" in y and "cp1" in y["checkpoints"]
    judge = y["checkpoints"]["cp1"]["judge"]
    assert judge["type"] == "always_warm_start"


def test_eval_yaml_for_cell_returns_valid_composer_block() -> None:
    cell = generate_g1_cells()[0]
    y = build_eval_yaml_for_cell("spatial16_w8_d4", cell, fh_thr=0.4, ws_thr=0.1)
    composer = y["checkpoints"]["cp1"]["judge"]["composer"]
    assert composer["type"] == "weighted_sum_zero_nan"
    assert composer["tier_thresholds"] == {"full_hit": 0.4, "warm_start": 0.1}
    assert set(composer["weights"]) == set(cell.declared_keys)


# ----------------------------------------------------------------------
# G2 R1 plan-contract tests (S1)
# ----------------------------------------------------------------------


def test_g4_subset_tags_match_plan_contract() -> None:
    """G4 12 subset tags must match plan §1.5 list verbatim. Catches
    silent replacement of one subset (G2 R1 B2 regression guard).
    """
    expected_tags = (
        "full", "drop-off-path", "drop-off-disp", "drop-off-dir",
        "drop-off-jerk", "off-only", "on-only", "jerk-pair", "disp-pair",
        "off-1win-on-full", "off-jerk-on-full", "jerk-full-stack",
    )
    actual_tags = tuple(tag for (_, _, _, _, tag) in G4_FACTOR_SUBSETS)
    assert actual_tags == expected_tags, (
        f"G4 subset tags drifted from plan: expected {expected_tags}, "
        f"got {actual_tags}"
    )


def test_g4_jerk_full_stack_subset_shape() -> None:
    """plan §1.5 last entry: offline jerk W-FUT+W-K3 + online jerk W-K3 → 4 keys."""
    cells = generate_g4_cells()
    sub = [c for c in cells if "jerk-full-stack" in c.yaml_id]
    # 2 channel × 2 base = 4 cells with this tag
    assert len(sub) == 4
    for c in sub:
        # All declared keys must be jerk_*; offline has 3 windows, online has 1.
        assert all("jerk" in k for k in c.declared_keys), (
            f"{c.yaml_id} declared_keys not all jerk: {c.declared_keys}"
        )
        offline_keys = [k for k in c.declared_keys if "_offline_" in k]
        online_keys = [k for k in c.declared_keys if "_online_" in k]
        assert len(offline_keys) == 3, f"{c.yaml_id} expected 3 offline jerk keys, got {len(offline_keys)}"
        assert len(online_keys) == 1, f"{c.yaml_id} expected 1 online jerk key, got {len(online_keys)}"


def test_g5_p1_p2_weights_match_phase4_r2_alpha1_uniform() -> None:
    """plan §1.6 / G2 R1 B1: G5 p1/p2 weights ≡ phase4 generate_r2_weights(rid, 1.0, (1,1,1,1)).
    Online keys must have weight 0 (online_total = 1 - alpha = 0).
    """
    from exp.verdict_factor_judge.phase4.spec import generate_r2_weights

    g5 = generate_g5_cells()
    for short, recipe_id in [
        ("p1", "p1_state_fut_online_act"),
        ("p2", "p2_action_fut_online_act"),
    ]:
        sample = next(c for c in g5 if c.base_recipe == short)
        expected = generate_r2_weights(recipe_id, alpha_star=1.0, offline_pattern=(1, 1, 1, 1))
        assert sample.weights == expected, (
            f"G5 {short} weights diverge from phase4 R2 alpha=1.0 uniform"
        )
        # Online keys should have weight 0
        for k, w in sample.weights.items():
            if "_online_" in k:
                assert w == 0.0, f"G5 {short} online key {k} weight={w} (expected 0)"


def test_g5_g6_weights_match_phase3_baseline() -> None:
    """plan §1.6: g6 uses phase3 baseline weights (1.0 per declared key)."""
    g5 = generate_g5_cells()
    sample = next(c for c in g5 if c.base_recipe == "g6")
    assert all(w == 1.0 for w in sample.weights.values()), (
        f"G5 g6 weights are not all 1.0: {sample.weights}"
    )
