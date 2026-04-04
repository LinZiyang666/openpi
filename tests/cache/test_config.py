"""Tests for cache config system (load, validate, build, assemble)."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import torch

from openpi.cache.config import (
    BackendConfig,
    CacheConfig,
    CheckpointConfig,
    ConfigValidationError,
    GateConfig,
    JudgeConfig,
    KeyBuilderConfig,
    KeyFieldConfig,
    KeysConfig,
    SearchStrategyConfig,
    TimerConfig,
    _substitute_env_vars,
    build_cache_components,
    load_cache_config,
    validate_cache_config,
)
from openpi.cache.types import CheckpointID


# ---------------------------------------------------------------------------
# Helper: write YAML to tmp file
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test_cache.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# test_load_default_config: load cache.yaml from project root
# ---------------------------------------------------------------------------


def test_load_default_config():
    """Load the project's default cache.yaml and verify structure."""
    project_root = Path(__file__).resolve().parents[2]
    yaml_path = project_root / "cache.yaml"
    if not yaml_path.exists():
        pytest.skip("cache.yaml not found at project root")

    config = load_cache_config(yaml_path)
    assert isinstance(config, CacheConfig)
    assert config.enabled is True
    assert "cp1" in config.checkpoints
    assert "cp3" in config.checkpoints
    assert config.backend.type == "in_memory"
    assert config.key_builder.type == "placeholder"


# ---------------------------------------------------------------------------
# test_env_var_substitution
# ---------------------------------------------------------------------------


def test_env_var_substitution_with_default():
    text = "url: ${NONEXISTENT_VAR:-http://fallback:1234}"
    result = _substitute_env_vars(text)
    assert result == "url: http://fallback:1234"


def test_env_var_substitution_with_env(monkeypatch):
    monkeypatch.setenv("TEST_CACHE_URL", "http://real:5678")
    text = "url: ${TEST_CACHE_URL:-http://fallback:1234}"
    result = _substitute_env_vars(text)
    assert result == "url: http://real:5678"


def test_env_var_substitution_no_default():
    text = "url: ${DEFINITELY_NONEXISTENT_12345}"
    result = _substitute_env_vars(text)
    assert result == "url: ${DEFINITELY_NONEXISTENT_12345}"


# ---------------------------------------------------------------------------
# test_validation_keys_vs_vector_dims
# ---------------------------------------------------------------------------


def test_validation_keys_vs_vector_dims():
    """Enabled key not in vector_dims should raise."""
    config = CacheConfig(
        keys=KeysConfig(vision_0=KeyFieldConfig(enabled=True)),
        backend=BackendConfig(vector_dims={"robot_state": 32}),
    )
    with pytest.raises(ConfigValidationError, match="vision_0 is enabled but not found"):
        validate_cache_config(config)


# ---------------------------------------------------------------------------
# test_validation_key_builder_vs_enabled_keys
# ---------------------------------------------------------------------------


def test_validation_key_builder_vs_enabled_keys():
    """Placeholder key_builder + vision_0 enabled should raise."""
    config = CacheConfig(
        keys=KeysConfig(vision_0=KeyFieldConfig(enabled=True)),
        backend=BackendConfig(vector_dims={"robot_state": 32, "vision_0": 1024}),
        key_builder=KeyBuilderConfig(type="placeholder"),
    )
    with pytest.raises(ConfigValidationError, match="placeholder.*only supports"):
        validate_cache_config(config)


# ---------------------------------------------------------------------------
# test_validation_invalid_checkpoint
# ---------------------------------------------------------------------------


def test_validation_invalid_checkpoint():
    config = CacheConfig(
        checkpoints={"cp1": CheckpointConfig(), "cp99": CheckpointConfig()},
    )
    with pytest.raises(ConfigValidationError, match="cp99"):
        validate_cache_config(config)


# ---------------------------------------------------------------------------
# test_validation_invalid_step_filter
# ---------------------------------------------------------------------------


def test_validation_invalid_step_filter():
    config = CacheConfig(
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(step_filter="invalid")
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match="invalid"):
        validate_cache_config(config)


# ---------------------------------------------------------------------------
# test_build_cache_components
# ---------------------------------------------------------------------------


def test_build_cache_components():
    """Factory produces correct component types."""
    config = CacheConfig(
        enabled=True,
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(judge=JudgeConfig(threshold=0.98)),
            "cp3": CheckpointConfig(judge=JudgeConfig(threshold=0.95)),
        },
    )
    validate_cache_config(config)
    components = build_cache_components(config)

    assert "timer" in components
    assert "storage" in components
    assert "key_builder" in components
    assert "gates" in components
    assert "judges" in components
    assert "search_strategies" in components

    assert CheckpointID.CP1 in components["gates"]
    assert CheckpointID.CP3 in components["gates"]
    assert CheckpointID.CP1 in components["judges"]
    assert CheckpointID.CP3 in components["judges"]
    assert CheckpointID.CP1 in components["search_strategies"]
    assert CheckpointID.CP3 in components["search_strategies"]


# ---------------------------------------------------------------------------
# test_build_and_assemble: full assembly into Orchestrator
# ---------------------------------------------------------------------------


def test_build_and_assemble():
    """From config -> build -> Orchestrator construction should not raise."""
    from openpi.cache.orchestrator import CacheOrchestrator

    config = CacheConfig(
        enabled=True,
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(judge=JudgeConfig(threshold=0.98)),
            "cp3": CheckpointConfig(judge=JudgeConfig(threshold=0.95)),
        },
    )
    validate_cache_config(config)
    components = build_cache_components(config)

    orch = CacheOrchestrator(
        storage=components["storage"],
        key_builder=components["key_builder"],
        gates=components["gates"],
        judges=components["judges"],
        search_strategies=components["search_strategies"],
        timer=components["timer"],
    )
    assert orch is not None


# ---------------------------------------------------------------------------
# test_yaml_anchor_merge: CP1/CP3 have independent judge thresholds after merge
# ---------------------------------------------------------------------------


def test_yaml_anchor_merge(tmp_path):
    yaml_content = """\
    enabled: true
    keys:
      robot_state: { enabled: true, weight: 1.0 }
    key_builder:
      type: placeholder
    checkpoints:
      _defaults: &cp_defaults
        gate:
          type: always_search
        search_strategy:
          type: simple_knn
      cp1:
        <<: *cp_defaults
        judge:
          type: threshold
          threshold: 0.98
      cp3:
        <<: *cp_defaults
        judge:
          type: threshold
          threshold: 0.95
    backend:
      type: in_memory
      vector_dims:
        robot_state: 32
    """
    p = _write_yaml(tmp_path, yaml_content)
    config = load_cache_config(p)

    assert config.checkpoints["cp1"].judge.threshold == 0.98
    assert config.checkpoints["cp3"].judge.threshold == 0.95


# ---------------------------------------------------------------------------
# test_fusion_weights_from_keys
# ---------------------------------------------------------------------------


def test_fusion_weights_from_keys():
    """Keys config weights should be dispatched to SearchStrategy."""
    from openpi.cache.components.search_strategy import SimpleKnnStrategy

    config = CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=2.5)),
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(judge=JudgeConfig(threshold=0.98)),
        },
    )
    validate_cache_config(config)
    components = build_cache_components(config)

    strategy = components["search_strategies"][CheckpointID.CP1]
    assert isinstance(strategy, SimpleKnnStrategy)
    assert strategy._fusion_weights == {"robot_state": 2.5}


# ---------------------------------------------------------------------------
# test_valid_config_passes: default config should not raise
# ---------------------------------------------------------------------------


def test_valid_default_config_passes():
    config = CacheConfig()
    validate_cache_config(config)


# ---------------------------------------------------------------------------
# test_validation_qdrant_step_filter: qdrant + step_filter!=all should raise
# ---------------------------------------------------------------------------


def test_validation_qdrant_step_filter_exact():
    config = CacheConfig(
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(step_filter="exact")
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match="step_filter.*not supported.*qdrant"):
        validate_cache_config(config)


def test_validation_qdrant_step_filter_window():
    config = CacheConfig(
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(step_filter="window")
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match="step_filter.*not supported.*qdrant"):
        validate_cache_config(config)


def test_validation_qdrant_step_filter_all_passes():
    """qdrant + step_filter=all should be valid."""
    config = CacheConfig(
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(step_filter="all")
            ),
        },
    )
    # Should not raise.
    validate_cache_config(config)


# ---------------------------------------------------------------------------
# test_validation_placeholder_robot_state_disabled: must be enabled
# ---------------------------------------------------------------------------


def test_validation_placeholder_robot_state_disabled():
    config = CacheConfig(
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=False)),
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        key_builder=KeyBuilderConfig(type="placeholder"),
    )
    with pytest.raises(ConfigValidationError, match="robot_state.enabled=true"):
        validate_cache_config(config)


# ---------------------------------------------------------------------------
# test_checkpoint_enabled_false: disabled CP excluded from components
# ---------------------------------------------------------------------------


def test_checkpoint_enabled_false_excluded():
    """Disabled checkpoint should not appear in built components."""
    config = CacheConfig(
        enabled=True,
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(enabled=True, judge=JudgeConfig(threshold=0.98)),
            "cp3": CheckpointConfig(enabled=False),
        },
    )
    validate_cache_config(config)
    components = build_cache_components(config)

    assert CheckpointID.CP1 in components["gates"]
    assert CheckpointID.CP3 not in components["gates"]
    assert CheckpointID.CP1 in components["judges"]
    assert CheckpointID.CP3 not in components["judges"]
    assert CheckpointID.CP1 in components["search_strategies"]
    assert CheckpointID.CP3 not in components["search_strategies"]


def test_disabled_checkpoint_orchestrator_returns_miss():
    """Orchestrator.check() on a disabled CP should return MISS without error."""
    from openpi.cache.orchestrator import CacheOrchestrator

    config = CacheConfig(
        enabled=True,
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(enabled=True, judge=JudgeConfig(threshold=0.98)),
            "cp3": CheckpointConfig(enabled=False),
        },
    )
    validate_cache_config(config)
    components = build_cache_components(config)

    orch = CacheOrchestrator(
        storage=components["storage"],
        key_builder=components["key_builder"],
        gates=components["gates"],
        judges=components["judges"],
        search_strategies=components["search_strategies"],
        timer=components["timer"],
    )

    from openpi.cache.components.judge import HitType
    from tests.cache.conftest import make_stage1

    state = torch.randn(1, 32)
    result = orch.check(CheckpointID.CP3, stage1=make_stage1(state))
    assert result.hit_type == HitType.MISS
