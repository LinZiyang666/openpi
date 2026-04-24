"""Tests for ``exp/random_periodic_gate/generate_batches.py``.

Plan: ``logs/random_periodic_gate_plan.log.md`` §6.4.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from exp.random_periodic_gate import generate_batches as gb


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def test_grid_has_38_entries_per_cfg():
    grid = gb.build_grid_for_cfg("clip_w7_d4")
    assert len(grid) == 38


def test_grid_slug_counts_match_spec():
    grid = gb.build_grid_for_cfg("clip_w7_d4")
    periodic = [g for g in grid if g.gate_type == "periodic"]
    random = [g for g in grid if g.gate_type == "random"]
    # 4 cache_len x 5 inference_len = 20 periodic;
    # 6 p_inference x 3 seed = 18 random.
    assert len(periodic) == 20
    assert len(random) == 18


def test_grid_slugs_are_unique_and_sorted():
    grid = gb.build_grid_for_cfg("clip_w7_d4")
    slugs = [g.slug for g in grid]
    assert len(set(slugs)) == len(slugs), "slugs must be unique"
    assert slugs == sorted(slugs), "grid must be sorted by slug"


def test_batch_index_for_cfg_mapping():
    # Plan §4 binds cfg 0 -> batch1 (clip), cfg 1 -> batch2 (spatial16),
    # cfg 2 -> batch3 (max_pool).
    assert gb.batch_index_for_cfg(0) == 1
    assert gb.batch_index_for_cfg(1) == 2
    assert gb.batch_index_for_cfg(2) == 3


def test_periodic_slug_format():
    gp = gb.GridPoint(
        cfg="clip_w7_d4", gate_type="periodic",
        params={"cache_len": 5, "inference_len": 2},
    )
    assert gp.slug == "periodic_k5_n2"


def test_random_slug_format():
    gp = gb.GridPoint(
        cfg="clip_w7_d4", gate_type="random",
        params={"p_inference": 0.20, "seed": 1},
    )
    assert gp.slug == "random_p0p20_s1"


def test_random_slug_edge_cases():
    gp_low = gb.GridPoint(
        cfg="x", gate_type="random", params={"p_inference": 0.05, "seed": 0},
    )
    gp_high = gb.GridPoint(
        cfg="x", gate_type="random", params={"p_inference": 0.70, "seed": 2},
    )
    assert gp_low.slug == "random_p0p05_s0"
    assert gp_high.slug == "random_p0p70_s2"




# ---------------------------------------------------------------------------
# Render contract: dict-level equality outside the gate subtree.
# ---------------------------------------------------------------------------


def _base_fixture() -> dict:
    return {
        "enabled": True,
        "keys": {"robot_state": {"enabled": True, "weight": 1.0}},
        "key_builder": {"type": "placeholder"},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},   # placeholder
                "judge": {"type": "always_hit"},
                "search_strategy": {"type": "qdrant_weighted_rrf_knn"},
            },
        },
        "backend": {"type": "qdrant", "vector_dims": {"robot_state": 32}},
        "write_policy": {"type": "never"},
    }


def test_render_dict_preserves_fields_outside_gate():
    base = _base_fixture()
    point = gb.GridPoint(
        cfg="clip_w7_d4", gate_type="random",
        params={"p_inference": 0.3, "seed": 0},
    )
    rendered = gb.render_dict(base, point)

    expected = copy.deepcopy(base)
    expected["checkpoints"]["cp1"]["gate"] = {
        "type": "random", "p_inference": 0.3, "seed": 0,
    }
    assert rendered == expected


def test_render_dict_does_not_mutate_input():
    base = _base_fixture()
    original = copy.deepcopy(base)
    gb.render_dict(
        base,
        gb.GridPoint(
            cfg="x", gate_type="periodic",
            params={"cache_len": 1, "inference_len": 1},
        ),
    )
    assert base == original


def test_render_dict_periodic_sets_k_and_n():
    base = _base_fixture()
    point = gb.GridPoint(
        cfg="clip_w7_d4", gate_type="periodic",
        params={"cache_len": 5, "inference_len": 2},
    )
    rendered = gb.render_dict(base, point)
    assert rendered["checkpoints"]["cp1"]["gate"] == {
        "type": "periodic", "cache_len": 5, "inference_len": 2,
    }


# ---------------------------------------------------------------------------
# Real-filesystem layout + idempotency + validate flag.
# ---------------------------------------------------------------------------


def test_generate_all_writes_114_files_in_three_batches(monkeypatch, tmp_path):
    """Redirect output roots into tmp_path and run the full pipeline."""
    # Base YAMLs still live under the real repo root; only the batch
    # directories get redirected so we can inspect them in isolation.
    real_root = gb._project_root()
    fake_root = tmp_path

    # Copy base templates into the fake tree so the generator reads them
    # alongside the redirected batch dirs.
    (fake_root / "exp" / "random_periodic_gate" / "config").mkdir(parents=True)
    for cfg in gb.CFG_NAMES:
        src = real_root / "exp" / "random_periodic_gate" / "config" / f"base_{cfg}.yaml"
        dst = fake_root / "exp" / "random_periodic_gate" / "config" / f"base_{cfg}.yaml"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(gb, "_project_root", lambda: fake_root)

    written = gb.generate_all(validate=False)
    assert len(written) == 114

    per_batch = {}
    for p in written:
        per_batch.setdefault(p.parent.name, 0)
        per_batch[p.parent.name] += 1
    assert per_batch == {f"batch{i}": 38 for i in range(1, 4)}


def test_generate_all_is_byte_identical_on_second_run(monkeypatch, tmp_path):
    real_root = gb._project_root()
    fake_root = tmp_path
    (fake_root / "exp" / "random_periodic_gate" / "config").mkdir(parents=True)
    for cfg in gb.CFG_NAMES:
        (fake_root / "exp" / "random_periodic_gate" / "config" / f"base_{cfg}.yaml").write_text(
            (real_root / "exp" / "random_periodic_gate" / "config" / f"base_{cfg}.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    monkeypatch.setattr(gb, "_project_root", lambda: fake_root)

    first = gb.generate_all()
    first_bytes = {p.relative_to(fake_root): p.read_bytes() for p in first}
    second = gb.generate_all()
    second_bytes = {p.relative_to(fake_root): p.read_bytes() for p in second}
    assert first_bytes == second_bytes, "generator must be idempotent"


def test_rendered_yaml_roundtrip_matches_base_except_gate():
    """End-to-end: load base, load rendered, diff restricted to gate subtree."""
    real_root = gb._project_root()
    base_path = real_root / "exp" / "random_periodic_gate" / "config" / "base_clip_w7_d4.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    # batch1 holds the full 38 clip_w7_d4 slugs (both periodic and random).
    # Pick one of each to exercise the render contract.
    if not (real_root / "exp" / "random_periodic_gate" / "config" / "batch1").exists():
        pytest.skip("real batch files not present; run generate_batches first")

    sample = (
        real_root
        / "exp"
        / "random_periodic_gate"
        / "config"
        / "batch1"
        / "periodic_k10_n1.yaml"
    )
    rendered = yaml.safe_load(sample.read_text(encoding="utf-8"))
    base_mod = copy.deepcopy(base)
    base_mod["checkpoints"]["cp1"]["gate"] = rendered["checkpoints"]["cp1"]["gate"]
    assert rendered == base_mod

    assert rendered["checkpoints"]["cp1"]["gate"] == {
        "type": "periodic", "cache_len": 10, "inference_len": 1,
    }

    sample_random = (
        real_root
        / "exp"
        / "random_periodic_gate"
        / "config"
        / "batch1"
        / "random_p0p30_s0.yaml"
    )
    rendered_r = yaml.safe_load(sample_random.read_text(encoding="utf-8"))
    base_mod_r = copy.deepcopy(base)
    base_mod_r["checkpoints"]["cp1"]["gate"] = rendered_r["checkpoints"]["cp1"]["gate"]
    assert rendered_r == base_mod_r
    assert rendered_r["checkpoints"]["cp1"]["gate"] == {
        "type": "random", "p_inference": 0.3, "seed": 0,
    }
