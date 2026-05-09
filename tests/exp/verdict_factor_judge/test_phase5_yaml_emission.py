"""Integration smoke tests for phase5 yaml emission.

Verifies the yaml builder pipeline end-to-end:
  - 240 cells emit unique yaml_ids
  - factor list ↔ weight dict ↔ declared_keys all aligned
  - warmup yaml dedup expectations (per §3.4)
"""

from __future__ import annotations

from pathlib import Path

import yaml

from exp.verdict_factor_judge.phase5.spec import (
    build_eval_yaml_for_cell,
    build_warmup_yaml_for_cell,
    generate_all_cells,
    generate_g1_cells,
    generate_g2_cells,
    generate_g3_cells,
    generate_g4_cells,
    generate_g5_cells,
)


def test_240_cell_emit_yaml_set_size_240(tmp_path: Path) -> None:
    cells = generate_all_cells()
    seen_yaml_ids: set[str] = set()
    for cell in cells:
        y = build_eval_yaml_for_cell("spatial16_w8_d4", cell, fh_thr=0.4, ws_thr=0.1)
        # Round-trip through yaml.safe_dump → safe_load to verify dump-ability
        s = yaml.safe_dump(y)
        re = yaml.safe_load(s)
        assert isinstance(re, dict)
        seen_yaml_ids.add(cell.yaml_id)
    assert len(seen_yaml_ids) == 240


def test_warmup_dedup_count_per_group() -> None:
    """Per plan §3.4: G1 48, G2 48, G3 4, G4 48, G5 0 (reuse historical)."""
    cells = generate_all_cells()
    g1 = {c.warmup_yaml_id for c in cells if c.group == "g1"}
    g2 = {c.warmup_yaml_id for c in cells if c.group == "g2"}
    g3 = {c.warmup_yaml_id for c in cells if c.group == "g3"}
    g4 = {c.warmup_yaml_id for c in cells if c.group == "g4"}
    g5 = {c.warmup_yaml_id for c in cells if c.group == "g5"}

    # G5 warmups are historical phase3/phase4 ids — count should be 3 (one per recipe).
    assert len(g1) == 48
    assert len(g2) == 48
    assert len(g3) == 4
    assert len(g4) == 48
    assert len(g5) == 3   # p1/p2/g6 historical


def test_eval_yaml_has_thresholds_and_weights() -> None:
    cell = generate_g1_cells()[0]
    y = build_eval_yaml_for_cell("spatial16_w8_d4", cell, fh_thr=0.45, ws_thr=0.05)
    composer = y["checkpoints"]["cp1"]["judge"]["composer"]
    assert composer["tier_thresholds"]["full_hit"] == 0.45
    assert composer["tier_thresholds"]["warm_start"] == 0.05
    assert set(composer["weights"]) == set(cell.declared_keys)


def test_warmup_yaml_factor_list_includes_cell_factors() -> None:
    """Warmup factor list must contain factor types matching the eval cell
    (so factor_raw covers all declared keys after replay).
    """
    for cell in (generate_g1_cells()[0], generate_g3_cells()[0], generate_g4_cells()[0]):
        wy = build_warmup_yaml_for_cell("spatial16_w8_d4", cell)
        dump_factors = wy["checkpoints"]["cp1"]["judge"]["dump"]["factors"]
        dump_types = {f["type"] for f in dump_factors}
        cell_types = {f["type"] for f in cell.factors}
        assert cell_types <= dump_types, (
            f"{cell.yaml_id}: cell factor types {cell_types - dump_types} missing in warmup"
        )


def test_g5_yaml_id_recipe_short_correct() -> None:
    g5 = generate_g5_cells()
    for c in g5:
        assert c.base_recipe in ("p1", "p2", "g6")
        assert c.base_recipe in c.yaml_id
