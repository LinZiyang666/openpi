"""Tests for experiment config parsing, validation, and factory."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from openpi.cache.config import (
    ConfigValidationError,
    FieldSimilarityConfig,
    InMemoryConfig,
    ScoreNormalizationConfig,
    load_cache_config,
    validate_cache_config,
    _dict_to_dataclass,
    CacheConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "cache.yaml"
    p.write_text(textwrap.dedent(content))
    return p


_RRF_YAML = """\
enabled: true
key_builder:
  type: cp1_mean_pool
keys:
  vision_0: {enabled: true, weight: 0.75}
  vision_1: {enabled: true, weight: 0.25}
  vision_2: {enabled: false, weight: 0.0}
  prompt_emb: {enabled: true, weight: 0.0}
  robot_state: {enabled: true, weight: 0.25}
checkpoints:
  cp1:
    enabled: true
    gate: {type: always_search}
    judge: {type: always_hit}
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
      rrf_k: 60
      field_similarity:
        vision_0: {type: cosine}
        vision_1: {type: cosine}
        prompt_emb: {type: cosine}
        robot_state:
          type: l2
          to_similarity: {type: exp, tau: 0.334717}
backend:
  type: in_memory
  vector_dims:
    vision_0: 2048
    vision_1: 2048
    prompt_emb: 2048
    robot_state: 32
  in_memory:
    preload_path: /tmp/test_artifact.pkl
"""

_SUM_YAML = """\
enabled: true
key_builder:
  type: cp1_mean_pool
keys:
  vision_0: {enabled: true, weight: 0.5}
  vision_1: {enabled: true, weight: 0.25}
  vision_2: {enabled: false, weight: 0.0}
  prompt_emb: {enabled: true, weight: 0.0}
  robot_state: {enabled: true, weight: 0.25}
checkpoints:
  cp1:
    enabled: true
    gate: {type: always_search}
    judge: {type: always_hit}
    search_strategy:
      type: weighted_score_sum_knn
      top_k: 1
      field_similarity:
        vision_0: {type: cosine}
        vision_1: {type: cosine}
        prompt_emb: {type: cosine}
        robot_state:
          type: l2
          to_similarity: {type: exp, tau: 0.334717}
      score_normalization:
        type: percentile
        fields:
          vision_0: {p5: 0.82, p95: 0.99}
          vision_1: {p5: 0.80, p95: 0.98}
          prompt_emb: {p5: 0.85, p95: 0.99}
          robot_state: {p5: 0.15, p95: 0.88}
backend:
  type: in_memory
  vector_dims:
    vision_0: 2048
    vision_1: 2048
    prompt_emb: 2048
    robot_state: 32
  in_memory:
    preload_path: /tmp/test_artifact.pkl
"""


# ---------------------------------------------------------------------------
# TestConfigParsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_rrf_yaml_parses(self, tmp_path):
        path = _write_yaml(tmp_path, _RRF_YAML)
        config = load_cache_config(path)
        assert config.key_builder.type == "cp1_mean_pool"
        assert config.backend.type == "in_memory"
        assert config.backend.in_memory.preload_path == "/tmp/test_artifact.pkl"
        cp1 = config.checkpoints["cp1"]
        assert cp1.search_strategy.type == "weighted_rrf_knn"
        assert cp1.search_strategy.rrf_k == 60
        # field_similarity parsed
        fs = cp1.search_strategy.field_similarity
        assert fs is not None
        assert isinstance(fs["vision_0"], FieldSimilarityConfig)
        assert fs["vision_0"].type == "cosine"
        assert fs["robot_state"].type == "l2"
        assert fs["robot_state"].to_similarity == {"type": "exp", "tau": 0.334717}

    def test_sum_yaml_parses(self, tmp_path):
        path = _write_yaml(tmp_path, _SUM_YAML)
        config = load_cache_config(path)
        cp1 = config.checkpoints["cp1"]
        assert cp1.search_strategy.type == "weighted_score_sum_knn"
        sn = cp1.search_strategy.score_normalization
        assert sn is not None
        assert isinstance(sn, ScoreNormalizationConfig)
        assert sn.type == "percentile"
        assert sn.fields["vision_0"]["p5"] == 0.82

    def test_in_memory_config_parsed(self, tmp_path):
        path = _write_yaml(tmp_path, _RRF_YAML)
        config = load_cache_config(path)
        assert isinstance(config.backend.in_memory, InMemoryConfig)
        assert config.backend.in_memory.index_type == "brute_force"


# ---------------------------------------------------------------------------
# TestConfigValidation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_score_sum_requires_normalization(self, tmp_path):
        """weighted_score_sum_knn without score_normalization should fail."""
        yaml_content = _SUM_YAML.replace(
            "      score_normalization:\n"
            "        type: percentile\n"
            "        fields:\n"
            "          vision_0: {p5: 0.82, p95: 0.99}\n"
            "          vision_1: {p5: 0.80, p95: 0.98}\n"
            "          prompt_emb: {p5: 0.85, p95: 0.99}\n"
            "          robot_state: {p5: 0.15, p95: 0.88}",
            ""
        )
        path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError, match="score_normalization"):
            load_cache_config(path)

    def test_score_sum_incomplete_calibration_fails(self, tmp_path):
        """percentile normalization missing stats for a weighted field should fail."""
        # Remove robot_state stats (weight=0.25, so it's required)
        yaml_content = _SUM_YAML.replace(
            "          robot_state: {p5: 0.15, p95: 0.88}\n",
            "",
        )
        path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError, match="robot_state"):
            load_cache_config(path)

    def test_score_sum_zero_weight_field_not_required(self, tmp_path):
        """percentile normalization can omit stats for weight=0 fields."""
        # prompt_emb has weight=0.0 in _SUM_YAML, so removing its stats is fine
        yaml_content = _SUM_YAML.replace(
            "          prompt_emb: {p5: 0.85, p95: 0.99}\n",
            "",
        )
        path = _write_yaml(tmp_path, yaml_content)
        config = load_cache_config(path)
        assert config.checkpoints["cp1"].search_strategy.type == "weighted_score_sum_knn"

    def test_cp1_builder_requires_vision_0(self, tmp_path):
        """cp1_* builder with vision_0 disabled should fail."""
        yaml_content = _RRF_YAML.replace(
            "vision_0: {enabled: true, weight: 0.75}",
            "vision_0: {enabled: false, weight: 0.0}",
        )
        path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError, match="vision_0"):
            load_cache_config(path)

    def test_in_memory_rrf_requires_in_memory_backend(self, tmp_path):
        """weighted_rrf_knn with qdrant backend should fail."""
        yaml_content = _RRF_YAML.replace(
            "  type: in_memory",
            "  type: qdrant",
        )
        path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError, match="in_memory"):
            load_cache_config(path)

    def test_cp1_builder_requires_preload_path(self, tmp_path):
        """cp1_* builder with in_memory backend needs preload_path."""
        yaml_content = _RRF_YAML.replace(
            "    preload_path: /tmp/test_artifact.pkl",
            "    preload_path: null",
        )
        path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError, match="preload_path"):
            load_cache_config(path)

    def test_unknown_key_builder_type(self, tmp_path):
        yaml_content = _RRF_YAML.replace("cp1_mean_pool", "unknown_type")
        path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError, match="key_builder"):
            load_cache_config(path)

    def test_valid_key_builder_types(self, tmp_path):
        """All 4 cp1_* types should pass validation."""
        for bt in ("cp1_mean_pool", "cp1_spatial_pool_16", "cp1_spatial_pool_64", "cp1_max_pool"):
            yaml_content = _RRF_YAML.replace("cp1_mean_pool", bt)
            if bt in ("cp1_spatial_pool_16",):
                yaml_content = yaml_content.replace("vision_0: 2048", "vision_0: 32768")
                yaml_content = yaml_content.replace("vision_1: 2048", "vision_1: 32768")
            elif bt in ("cp1_spatial_pool_64",):
                yaml_content = yaml_content.replace("vision_0: 2048", "vision_0: 8192")
                yaml_content = yaml_content.replace("vision_1: 2048", "vision_1: 8192")
            path = _write_yaml(tmp_path, yaml_content)
            config = load_cache_config(path)
            assert config.key_builder.type == bt
