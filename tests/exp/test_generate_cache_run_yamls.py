from __future__ import annotations

import yaml
import pytest

from exp.cache_experiment.generate_cache_run_yamls import COMBOS, _generate_fine_grid, render_yaml


def _calibration() -> dict:
    fields = {
        "vision_0": {"p5": 0.1, "p95": 0.9},
        "vision_1": {"p5": 0.2, "p95": 0.95},
        "prompt_emb": {"p5": 0.3, "p95": 0.96},
        "robot_state": {"p5": 0.05, "p95": 0.8},
    }
    return {
        "cp1_mean_pool": fields,
        "cp1_spatial_pool_16": fields,
        "cp1_spatial_pool_64": fields,
        "cp1_max_pool": fields,
    }


def test_render_yaml_for_score_sum_includes_percentile_normalization():
    combo = next(c for c in COMBOS if c.builder_type == "cp1_mean_pool" and c.strategy_abbrev == "sum")
    text = render_yaml(
        combo,
        {"vision_0": 0.5, "vision_1": 0.25, "robot_state": 0.25},
        "w4",
        "artifacts",
        _calibration(),
    )
    data = yaml.safe_load(text)

    ss = data["checkpoints"]["cp1"]["search_strategy"]
    assert ss["type"] == "weighted_score_sum_knn"
    assert ss["score_normalization"]["type"] == "percentile"
    assert "robot_state" in ss["score_normalization"]["fields"]
    assert data["backend"]["in_memory"]["preload_path"] == "artifacts/cp1_mean_pool.pkl"


def test_render_yaml_for_rrf_omits_score_normalization():
    combo = next(c for c in COMBOS if c.builder_type == "cp1_mean_pool" and c.strategy_abbrev == "rrf")
    text = render_yaml(
        combo,
        {"vision_0": 0.75, "vision_1": 0.25, "robot_state": 0.0},
        "w2",
        "artifacts",
        _calibration(),
    )
    data = yaml.safe_load(text)

    ss = data["checkpoints"]["cp1"]["search_strategy"]
    assert ss["type"] == "weighted_rrf_knn"
    assert ss["rrf_k"] == 60
    assert "score_normalization" not in ss


def test_render_yaml_fails_when_calibration_missing_for_positive_weight_field():
    combo = next(c for c in COMBOS if c.builder_type == "cp1_mean_pool" and c.strategy_abbrev == "sum")
    bad_cal = {"cp1_mean_pool": {"vision_0": {"p5": 0.1, "p95": 0.9}}}

    with pytest.raises(ValueError, match="missing stats"):
        render_yaml(
            combo,
            {"vision_0": 0.5, "vision_1": 0.25, "robot_state": 0.25},
            "w4",
            "artifacts",
            bad_cal,
        )


def test_generate_fine_grid_produces_unique_normalized_weights():
    center = {"vision_0": 0.5, "vision_1": 0.25, "robot_state": 0.25}
    grid = _generate_fine_grid(center, step=0.1, radius=0.1)

    assert grid
    assert len(grid) == len({tuple(sorted(w.items())) for w in grid})
    for weights in grid:
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in weights.values())
