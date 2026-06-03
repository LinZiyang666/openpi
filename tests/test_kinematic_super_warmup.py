"""Unit tests for exp.weighted_sum.kinematic.

Covers the contracts that G1 R1/R2/R3 reviewers explicitly mandated:

  - 237 cells generated with the exact group breakdown 48/48/48/48/45
    (B1: G5 grid filter arithmetic — exactly 15 pairs after fh+ws<=0.9).
  - All cells share the single ``ws_d1_kin_super_warmup`` warmup id.
  - ``super_warmup_declared_keys()`` actually covers every cell's
    ``declared_keys`` (B2: G5 declared ⊆ super union must hold).
  - Calling ``kinematic.spec.generate_all_cells()`` does NOT mutate
    ``phase5.spec.G5_THRESHOLD_GRID`` (B3: order-independence regression).
  - ``CFG_SPECS["spatial16_ws_d1_best"]`` is present and shape-correct.

The yaml-validator tests touch ``cp1_spatial_pool_16.pkl`` (~630 MB) and
need a live workspace, so they are marked ``@pytest.mark.manual`` (B4:
no fake ``CacheConfig.from_dict`` API — use the real
``load_cache_config(path)``).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest


# ----------------------------------------------------------------------
# Spec contract
# ----------------------------------------------------------------------


def test_237_cells_generated():
    """Total = G1+G2+G3+G4 + G5 = 48*4 + 45 = 237 (G1 R1 B1 arithmetic)."""
    from exp.weighted_sum.kinematic.spec import generate_all_cells

    cells = generate_all_cells()
    groups = Counter(c.group for c in cells)
    assert groups == {"g1": 48, "g2": 48, "g3": 48, "g4": 48, "g5": 45}, groups
    assert len(cells) == 237


def test_warmup_yaml_id_unified():
    """Every kinematic cell must share the single super warmup id."""
    from exp.weighted_sum.kinematic.spec import (
        SUPER_WARMUP_ID,
        generate_all_cells,
    )

    for c in generate_all_cells():
        assert c.warmup_yaml_id == SUPER_WARMUP_ID, (
            f"{c.yaml_id} has warmup_yaml_id={c.warmup_yaml_id!r}, "
            f"expected {SUPER_WARMUP_ID!r}"
        )


def test_yaml_id_prefix_renamed():
    """Yaml ids must use the kinematic prefix (no phase5 leakage)."""
    from exp.weighted_sum.kinematic.spec import YAML_PREFIX, generate_all_cells

    for c in generate_all_cells():
        assert c.yaml_id.startswith(YAML_PREFIX + "_"), (
            f"{c.yaml_id!r} missing {YAML_PREFIX} prefix"
        )
        # phase5 cfg_id substring should be gone after rename
        assert "spatial16_ws_d1_best_phase5" not in c.yaml_id, c.yaml_id


# ----------------------------------------------------------------------
# G5 grid arithmetic (G1 R1 B1 verified)
# ----------------------------------------------------------------------


def test_g5_grid_filtered():
    """G5 grid: fh+ws<=0.9 keeps 15 of 16 pairs, drops exactly (0.5, 0.5)."""
    from exp.weighted_sum.kinematic.spec import _g5_grid_filtered

    grid = _g5_grid_filtered()
    assert len(grid) == 15, f"expected 15 pairs, got {len(grid)}"

    for fh, ws in grid:
        assert fh + ws <= 0.9 + 1e-9, f"({fh}, {ws}) violates fh+ws<=0.9"

    # Boundary pairs kept
    assert (0.4, 0.5) in grid
    assert (0.5, 0.4) in grid
    # Only (0.5, 0.5) excluded
    assert (0.5, 0.5) not in grid


def test_g5_45_cells_with_3_recipes():
    """G5 = 15 (FH, WS) pairs × 3 recipes (p1 / p2 / g6) = 45 cells."""
    from exp.weighted_sum.kinematic.spec import G5_RECIPES, generate_all_cells

    g5_cells = [c for c in generate_all_cells() if c.group == "g5"]
    assert len(g5_cells) == 45
    assert len(G5_RECIPES) == 3
    # Each recipe contributes 15 cells
    per_recipe = Counter(c.base_recipe for c in g5_cells)
    assert all(v == 15 for v in per_recipe.values()), per_recipe


# ----------------------------------------------------------------------
# Super warmup keys cover all declared keys (G1 R1 B2 mandate)
# ----------------------------------------------------------------------


def test_super_factor_superset_covers_all_declared_keys():
    """Every cell.declared_keys must be ⊆ super_warmup_declared_keys()."""
    from exp.weighted_sum.kinematic.spec import (
        generate_all_cells,
        super_warmup_declared_keys,
    )

    sup = super_warmup_declared_keys()
    miss: set[str] = set()
    for c in generate_all_cells():
        miss |= set(c.declared_keys) - sup
    assert not miss, f"{len(miss)} keys not covered: {sorted(miss)[:10]}"


def test_super_factor_count_matches_expected_50():
    """Super union has 50 keys (verified by R1A agent + spec.py sanity).

    If this changes, the super warmup yaml's dump factor count changes too —
    treat as a deliberate spec change, not a regression.
    """
    from exp.weighted_sum.kinematic.spec import super_warmup_declared_keys

    sup = super_warmup_declared_keys()
    assert len(sup) == 50, f"super union size changed: {len(sup)} (expected 50)"


# ----------------------------------------------------------------------
# Phase5 module NON-mutation (G1 R1 B3 mandate, order-dependence test)
# ----------------------------------------------------------------------


def test_phase5_orig_spec_still_works_REGARDLESS_OF_ORDER():
    """Calling kinematic.generate_all_cells() FIRST must not mutate phase5.

    R3 v2's monkey-patch design (``p5.G5_THRESHOLD_GRID = ...``) would
    silently break this — phase5's native ``generate_all_cells()`` would
    see the filtered 15-pair grid and only return 237 cells. R4 uses a
    local generator so the global is untouched. This test guards the
    invariant explicitly.
    """
    from exp.verdict_factor_judge.phase5 import spec as p5_mod
    from exp.weighted_sum.kinematic.spec import generate_all_cells as kin_gen

    # Trigger kinematic generation (would mutate phase5 in the rejected design)
    _ = kin_gen()

    # phase5 module state must be untouched
    assert len(p5_mod.G5_THRESHOLD_GRID) == 16, (
        f"kinematic.spec mutated phase5.G5_THRESHOLD_GRID "
        f"(got {len(p5_mod.G5_THRESHOLD_GRID)} pairs, expected 16) — "
        f"this is the B3 regression that R4 explicitly forbids"
    )

    # phase5 native generator must still produce its 240 cells
    cells = p5_mod.generate_all_cells()
    assert len(cells) == 240
    g5_count = sum(1 for c in cells if c.group == "g5")
    assert g5_count == 48, f"phase5 G5 count drifted: {g5_count} (expected 48)"


# ----------------------------------------------------------------------
# v2_spec.CFG_SPECS injection (G1 R1 R3 §1.1 contract)
# ----------------------------------------------------------------------


def test_cfg_specs_has_spatial16_ws_d1_best():
    """v2_spec.CFG_SPECS must expose the kinematic d1-best cfg."""
    from exp.verdict_factor_judge.common.v2_spec import CFG_SPECS

    assert "spatial16_ws_d1_best" in CFG_SPECS
    cfg = CFG_SPECS["spatial16_ws_d1_best"]
    # Required shape (v2_spec.build_eval_yaml depends on these keys)
    for key in (
        "key_builder_type",
        "vector_dims",
        "keys",
        "preload_pkl",
        "search_strategy",
    ):
        assert key in cfg, f"CFG_SPECS['spatial16_ws_d1_best'] missing {key!r}"

    # Search strategy must be weighted_score_sum_knn (not weighted_rrf_knn)
    assert cfg["search_strategy"]["type"] == "weighted_score_sum_knn"
    # d1 = no trajectory_depth field (defaults to 1)
    assert "trajectory_depth" not in cfg["search_strategy"]
    assert "trajectory_weights" not in cfg["search_strategy"]
    # Keys verbatim from wsweep d1 best yaml
    assert cfg["keys"]["vision_0"]["weight"] == 0.0625
    assert cfg["keys"]["vision_1"]["weight"] == 0.5
    assert cfg["keys"]["robot_state"]["weight"] == 0.4375


def test_phase5_orig_cfg_untouched():
    """spatial16_w8_d4 (phase5 native cfg) must remain intact."""
    from exp.verdict_factor_judge.common.v2_spec import CFG_SPECS

    assert "spatial16_w8_d4" in CFG_SPECS
    cfg = CFG_SPECS["spatial16_w8_d4"]
    # phase5 uses weighted_rrf_knn d4
    assert cfg["search_strategy"]["type"] == "weighted_rrf_knn"
    assert cfg["search_strategy"]["trajectory_depth"] == 4


# ----------------------------------------------------------------------
# Super warmup yaml structure
# ----------------------------------------------------------------------


def test_super_warmup_yaml_structure():
    """Built super warmup yaml must be schema-correct (without server load)."""
    from exp.weighted_sum.kinematic.spec import CFG_ID_DEFAULT
    from exp.weighted_sum.kinematic.super_warmup import build_super_warmup_yaml

    y = build_super_warmup_yaml(cfg_id=CFG_ID_DEFAULT)
    # Required top-level shape (cache.config schema)
    for k in (
        "enabled",
        "keys",
        "key_builder",
        "checkpoints",
        "backend",
        "write_policy",
    ):
        assert k in y, f"missing top-level {k!r}"

    cp1 = y["checkpoints"]["cp1"]
    assert cp1["judge"]["type"] == "always_warm_start"
    assert cp1["judge"]["dump"]["deferred"] is True
    assert cp1["judge"]["dump"]["config_id"] == "ws_d1_kin_super_warmup"
    # Dump factors must be a non-empty list
    factors = cp1["judge"]["dump"]["factors"]
    assert isinstance(factors, list) and len(factors) > 0
    # All factor types must be the descriptor_source_channel form
    for f in factors:
        assert "_" in f["type"], f"unexpected factor type {f['type']!r}"

    # Search must be weighted_score_sum_knn (d1 weighted_sum retrieval)
    ss = cp1["search_strategy"]
    assert ss["type"] == "weighted_score_sum_knn"
    assert ss["top_k"] == 5  # raised from eval default 1 (super warmup convention)


def test_super_warmup_factors_match_declared_union():
    """Super warmup dump factors must cover every key in super union.

    Reconstructs ``{f"{type}__p{P}_f{F}"}`` from emitted factor blocks and
    confirms equality with ``super_warmup_declared_keys()``.
    """
    from exp.weighted_sum.kinematic.spec import super_warmup_declared_keys
    from exp.weighted_sum.kinematic.spec import CFG_ID_DEFAULT
    from exp.weighted_sum.kinematic.super_warmup import build_super_warmup_yaml

    y = build_super_warmup_yaml(cfg_id=CFG_ID_DEFAULT)
    factors = y["checkpoints"]["cp1"]["judge"]["dump"]["factors"]

    emitted_keys: set[str] = set()
    for fblock in factors:
        type_name = fblock["type"]
        for w in fblock["params"]["windows"]:
            emitted_keys.add(f"{type_name}__p{w['past']}_f{w['future']}")

    needed = super_warmup_declared_keys()
    missing = needed - emitted_keys
    extras = emitted_keys - needed
    assert not missing, (
        f"super warmup missing {len(missing)} keys: {sorted(missing)[:5]}"
    )
    # Extras are forbidden — every emitted key should correspond to a real cell
    # declared key (otherwise we're wasting extraction time per verdict).
    assert not extras, (
        f"super warmup has {len(extras)} extra keys: {sorted(extras)[:5]}"
    )


# ----------------------------------------------------------------------
# Eval yaml structure (offline calibration source — G1 R1 R5 path)
# ----------------------------------------------------------------------


def test_eval_yaml_uses_offline_calibration_source():
    """Eval yaml must use samples_source.type=offline, not warmup."""
    from exp.weighted_sum.kinematic.spec import (
        build_eval_yaml_for_cell,
        generate_all_cells,
    )

    cell = next(c for c in generate_all_cells() if c.group == "g5")
    y = build_eval_yaml_for_cell(cell, fh_thr=0.5, ws_thr=0.1)

    cal = y["checkpoints"]["cp1"]["judge"]["calibration"]
    assert cal["type"] == "percentile_rolling"
    assert cal["params"]["window_size"] == 50
    src = cal["samples_source"]
    assert src["type"] == "offline", (
        f"samples_source.type={src['type']!r} — must be 'offline' to bypass WarmupPool "
        f"LRU eviction (G1 R1 R1A B5 fix)"
    )
    assert src["offline"]["format"] == "jsonl"
    # Path is server-relative; runner pushes super_warmup_raw.jsonl to that
    # location on both ziyang10 and xuanle.
    assert "kinematic_phase5" in src["offline"]["path"]
    assert src["offline"]["path"].endswith("super_warmup_raw.jsonl")


def test_eval_yaml_search_strategy_matches_d1_best():
    """Eval yaml retrieval is d1 weighted_score_sum_knn (no trajectory)."""
    from exp.weighted_sum.kinematic.spec import (
        build_eval_yaml_for_cell,
        generate_all_cells,
    )

    cell = next(c for c in generate_all_cells())
    y = build_eval_yaml_for_cell(cell, fh_thr=0.5, ws_thr=0.1)
    ss = y["checkpoints"]["cp1"]["search_strategy"]
    assert ss["type"] == "weighted_score_sum_knn"
    assert ss["top_k"] == 1  # eval default (super warmup raises to 5)
    assert "trajectory_depth" not in ss


# ----------------------------------------------------------------------
# Runner argv adapter (G2 R1 B2 + B3: task_ids must parse to tuple[int, ...])
# ----------------------------------------------------------------------


def test_runner_parse_task_ids_csv():
    """CSV string must parse into ``tuple[int, ...]``.

    G2 R1 B2: passing raw ``"0,1,2"`` to run_phase._build_libero_argv
    yields ``--task-ids 0 , 1 , 2 , ...`` (per-character iteration over
    the string). The fix in ``runner._parse_task_ids`` converts to int
    tuple so the argv emit loop produces ``--task-ids 0 1 2 ...``.
    """
    from exp.weighted_sum.kinematic.runner import _parse_task_ids

    # Default CLI value
    assert _parse_task_ids("0,1,2,3,4,5,6,7,8,9") == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    # Empty → empty tuple
    assert _parse_task_ids("") == ()
    # Whitespace tolerant
    assert _parse_task_ids("0, 1, 2") == (0, 1, 2)
    # All elements must be int
    out = _parse_task_ids("5,7,9")
    assert all(isinstance(t, int) for t in out)


def test_runner_libero_args_emits_well_formed_task_ids():
    """End-to-end: argparse parse → adapter → namespace yields int tuple.

    Reproduces the G2 R1 B2 failure mode by going through the actual
    ``runner._parse`` argparse path.
    """
    from exp.weighted_sum.kinematic.runner import (
        _build_libero_args_from_cli,
        _parse,
    )

    # parse CLI defaults
    args = _parse(["--mode", "run-warmup"])
    assert isinstance(args.task_ids, str), "argparse stores --task-ids as string"

    adapted = _build_libero_args_from_cli(args)["args"]
    # The adapted namespace must hand the LIBERO argv builder a tuple of ints
    assert isinstance(adapted.task_ids, tuple)
    assert all(isinstance(t, int) for t in adapted.task_ids), adapted.task_ids
    assert adapted.task_ids == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    # Simulate the run_phase argv emit loop (run_phase.py:243-244) to
    # confirm we no longer produce comma tokens.
    if adapted.task_ids:
        tokens = ["--task-ids"] + [str(t) for t in adapted.task_ids]
    else:
        tokens = []
    assert tokens == [
        "--task-ids",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ], tokens
    # Negative: confirm passing a raw string WOULD break (regression sentinel).
    bad_tokens = ["--task-ids"] + [str(t) for t in "0,1,2,3,4,5,6,7,8,9"]
    assert "," in bad_tokens, "argv loop must emit comma tokens when iterating string"


# ----------------------------------------------------------------------
# Manual tests — require live workspace (G1 R1 B4: real load_cache_config)
# ----------------------------------------------------------------------


@pytest.mark.manual
def test_super_warmup_yaml_validates_against_config(tmp_path: Path):
    """Real cache.config validator on emitted super warmup yaml.

    Requires ``exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl``
    (630MB) on disk because ``load_cache_config`` validates preload_path.
    Run locally: ``pytest -m manual tests/test_kinematic_super_warmup.py``.
    """
    import yaml as _yaml
    from openpi.cache.config import load_cache_config
    from exp.weighted_sum.kinematic.spec import CFG_ID_DEFAULT
    from exp.weighted_sum.kinematic.super_warmup import build_super_warmup_yaml

    yaml_dict = build_super_warmup_yaml(cfg_id=CFG_ID_DEFAULT)
    yaml_path = tmp_path / "super_warmup.yaml"
    yaml_path.write_text(_yaml.safe_dump(yaml_dict))
    cfg = load_cache_config(yaml_path)
    assert cfg.checkpoints["cp1"].judge.type == "always_warm_start"


@pytest.mark.manual
def test_eval_yaml_validates_against_config(tmp_path: Path):
    """Real cache.config validator on emitted eval yaml."""
    import yaml as _yaml
    from openpi.cache.config import load_cache_config
    from exp.weighted_sum.kinematic.spec import (
        build_eval_yaml_for_cell,
        generate_all_cells,
    )

    cell = next(c for c in generate_all_cells() if c.group == "g5")
    yaml_dict = build_eval_yaml_for_cell(
        cell,
        fh_thr=0.5,
        ws_thr=0.1,
    )
    yaml_path = tmp_path / "eval.yaml"
    yaml_path.write_text(_yaml.safe_dump(yaml_dict))
    cfg = load_cache_config(yaml_path)
    assert cfg.checkpoints["cp1"].judge.calibration.samples_source.type == "offline"
