"""Tests for verdict-factor config dataclasses + B0 fail-fast composite
rejection.

Covers:
  - FactorConfig / ComposerConfig / NormalizerConfig dataclass parsing
  - JudgeConfig.factors deserializes as list[FactorConfig], not list[dict]
  - _list_inner_dataclass helper
  - validate_cache_config rejects type=composite at config load (B0 gate)
  - _build_composer / _build_normalizer error paths
  - _build_judge composite branch (B0 ships the wiring; validator gates it
    in the normal flow but it is reachable via direct call).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from openpi.cache.components.judge import CompositeJudge
from openpi.cache.config import (
    ComposerConfig,
    ConfigValidationError,
    FactorConfig,
    JudgeConfig,
    NormalizerConfig,
    _build_composer,
    _build_judge,
    _build_normalizer,
    _dict_to_dataclass,
    _JUDGE_TYPES,
    _list_inner_dataclass,
    load_cache_config,
)


# ------------------------------------------------------------------
# Dataclass round-trip
# ------------------------------------------------------------------


def test_judge_config_factors_parses_as_list_of_dataclass():
    data = {
        "type": "composite",
        "factors": [
            {"type": "f2", "params": {"K": 5}},
            {"type": "f1a_a", "params": {"window_k": 3, "descriptors": ["jerk"]}},
        ],
        "composer": {"type": "weighted_sum", "weights": {"f2_var": 1.0}, "tier_thresholds": {"full_hit": 0.5}},
        "normalizer": {"type": "percentile_rolling"},
    }
    cfg = _dict_to_dataclass(JudgeConfig, data)
    assert isinstance(cfg.factors, list)
    assert all(isinstance(f, FactorConfig) for f in cfg.factors)
    assert cfg.factors[0].type == "f2"
    assert cfg.factors[0].params == {"K": 5}
    assert isinstance(cfg.composer, ComposerConfig)
    assert isinstance(cfg.normalizer, NormalizerConfig)


def test_judge_config_factors_optional_default_none():
    cfg = _dict_to_dataclass(JudgeConfig, {"type": "threshold", "threshold": 0.9})
    assert cfg.factors is None
    assert cfg.composer is None
    assert cfg.normalizer is None


# ------------------------------------------------------------------
# _list_inner_dataclass helper
# ------------------------------------------------------------------


def test_list_inner_dataclass_recognizes_optional_form():
    assert _list_inner_dataclass("Optional[list[FactorConfig]]") is FactorConfig


def test_list_inner_dataclass_recognizes_pep604_form():
    assert _list_inner_dataclass("list[FactorConfig] | None") is FactorConfig


def test_list_inner_dataclass_recognizes_bare_form():
    assert _list_inner_dataclass("list[FactorConfig]") is FactorConfig


def test_list_inner_dataclass_returns_none_for_non_dataclass_inner():
    assert _list_inner_dataclass("list[float]") is None
    assert _list_inner_dataclass("list[dict[str, float]]") is None


def test_list_inner_dataclass_returns_none_for_unknown_name():
    assert _list_inner_dataclass("list[CompletelyMadeUp]") is None


# ------------------------------------------------------------------
# B0 fail-fast: composite YAML rejected at validation
# ------------------------------------------------------------------


def test_b0_judge_types_excludes_composite():
    assert "composite" not in _JUDGE_TYPES


def _write(tmp: Path, body: str) -> Path:
    p = tmp / "cache.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_composite_yaml_rejected_at_load_time(tmp_path):
    yaml_body = """
        backend: {type: in_memory}
        keys:
          robot_state: {enabled: true, weight: 1.0}
        key_builder: {type: placeholder}
        checkpoints:
          cp1:
            enabled: true
            judge:
              type: composite
              factors: [{type: f2, params: {K: 3}}]
              composer:
                type: weighted_sum
                weights: {f2_var: 1.0}
                tier_thresholds: {full_hit: 0.5}
            search_strategy: {type: weighted_rrf_knn}
        """
    p = _write(tmp_path, yaml_body)
    with pytest.raises(ConfigValidationError) as exc:
        load_cache_config(str(p))
    msg = str(exc.value)
    assert "composite" in msg
    # B1+ message helps users know when it becomes available.
    assert "B1" in msg


# ------------------------------------------------------------------
# _build_composer / _build_normalizer
# ------------------------------------------------------------------


def test_build_composer_weighted_sum_requires_weights():
    cfg = ComposerConfig(type="weighted_sum")
    with pytest.raises(ConfigValidationError, match="weights"):
        _build_composer(cfg)


def test_build_composer_weighted_sum_requires_tier_full_hit():
    cfg = ComposerConfig(type="weighted_sum", weights={"a": 1.0})
    with pytest.raises(ConfigValidationError, match="full_hit"):
        _build_composer(cfg)


def test_build_composer_and_requires_per_factor_thresholds():
    cfg = ComposerConfig(type="and")
    with pytest.raises(ConfigValidationError, match="per_factor_thresholds"):
        _build_composer(cfg)


def test_build_composer_or_requires_per_factor_thresholds():
    cfg = ComposerConfig(type="or")
    with pytest.raises(ConfigValidationError, match="per_factor_thresholds"):
        _build_composer(cfg)


def test_build_composer_unknown_type():
    cfg = ComposerConfig(type="nonexistent")
    with pytest.raises(ConfigValidationError, match="Unknown composer.type"):
        _build_composer(cfg)


def test_build_composer_weighted_sum_succeeds():
    cfg = ComposerConfig(
        type="weighted_sum",
        weights={"f2_var": 1.0},
        tier_thresholds={"full_hit": 0.5, "warm_start": 0.3},
        warm_start_t=0.5,
    )
    c = _build_composer(cfg)
    # Stub still raises on compose; we only verify it constructs cleanly.
    assert c._full_hit_threshold == 0.5
    assert c._warm_start_threshold == 0.3
    assert c._warm_start_t == 0.5


def test_build_normalizer_percentile_rolling():
    cfg = NormalizerConfig(type="percentile_rolling", window_size=50)
    n = _build_normalizer(cfg)
    assert n._window_size == 50


def test_build_normalizer_unknown_type():
    cfg = NormalizerConfig(type="nope")
    with pytest.raises(ConfigValidationError, match="Unknown normalizer.type"):
        _build_normalizer(cfg)


# ------------------------------------------------------------------
# _build_judge composite branch (direct call bypassing validator)
# ------------------------------------------------------------------


def test_build_judge_composite_with_f2_only():
    # F2 has requires_library_stats=False so library_stats=None is fine.
    cfg = JudgeConfig(
        type="composite",
        factors=[FactorConfig(type="f2", params={"K": 4})],
        composer=ComposerConfig(
            type="weighted_sum",
            weights={"f2_var": 1.0},
            tier_thresholds={"full_hit": 0.5},
        ),
        normalizer=NormalizerConfig(type="percentile_rolling"),
    )
    judge = _build_judge(cfg, library_stats=None)
    assert isinstance(judge, CompositeJudge)
    assert judge.min_required_top_k == 4


def test_build_judge_composite_missing_factors_raises():
    cfg = JudgeConfig(type="composite", composer=ComposerConfig(
        type="weighted_sum", weights={"a": 1.0}, tier_thresholds={"full_hit": 0.5},
    ))
    with pytest.raises(ConfigValidationError, match="at least one factor"):
        _build_judge(cfg, library_stats=None)


def test_build_judge_composite_missing_composer_raises():
    cfg = JudgeConfig(
        type="composite",
        factors=[FactorConfig(type="f2", params={"K": 1})],
    )
    with pytest.raises(ConfigValidationError, match="composer"):
        _build_judge(cfg, library_stats=None)
