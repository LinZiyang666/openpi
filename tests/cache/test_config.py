"""Tests for cache config system (load, validate, build, assemble)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import torch

from openpi.cache.config import (
    BackendConfig,
    CacheConfig,
    CheckpointConfig,
    ConfigValidationError,
    FieldSimilarityConfig,
    GateConfig,
    JudgeConfig,
    KeyBuilderConfig,
    KeyFieldConfig,
    KeysConfig,
    ScoreNormalizationConfig,
    SearchStrategyConfig,
    WritePolicyConfig,
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
    assert config.key_builder.type == "cp1_mean_pool"


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
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
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
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
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
          type: qdrant_weighted_rrf_knn
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
      type: qdrant
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
    from openpi.cache.components.search_strategy import QdrantWeightedRrfKnnStrategy

    config = CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=2.5)),
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(judge=JudgeConfig(threshold=0.98)),
        },
    )
    validate_cache_config(config)
    components = build_cache_components(config)

    strategy = components["search_strategies"][CheckpointID.CP1]
    assert isinstance(strategy, QdrantWeightedRrfKnnStrategy)
    assert strategy._fusion_weights == {"robot_state": 2.5}


# ---------------------------------------------------------------------------
# test_valid_config_passes: default config should not raise
# ---------------------------------------------------------------------------


def test_valid_default_config_passes():
    config = CacheConfig()
    validate_cache_config(config)


# ---------------------------------------------------------------------------
# test_validation_qdrant_step_filter: qdrant supports exact/window/all
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
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
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
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
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
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
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


# ---------------------------------------------------------------------------
# Trajectory config validation tests
# ---------------------------------------------------------------------------


def test_trajectory_depth_zero_rejected():
    config = CacheConfig(
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(trajectory_depth=0)
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match="trajectory_depth must be >= 1"):
        validate_cache_config(config)


def test_search_top_k_zero_rejected():
    config = CacheConfig(
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(top_k=0)
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match="top_k must be >= 1"):
        validate_cache_config(config)


def test_field_similarity_invalid_type_rejected():
    from openpi.cache.config import FieldSimilarityConfig

    config = CacheConfig(
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(
                    field_similarity={"robot_state": FieldSimilarityConfig(type="cosin")},
                )
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match=r"field_similarity.*invalid"):
        validate_cache_config(config)


def test_trajectory_weights_required_when_depth_gt1():
    config = CacheConfig(
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(
                    type="weighted_rrf_knn",
                    trajectory_depth=3,
                    trajectory_weights=None,
                )
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match="trajectory_weights required"):
        validate_cache_config(config)


def test_trajectory_weights_length_mismatch():
    config = CacheConfig(
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(
                    type="weighted_rrf_knn",
                    trajectory_depth=3,
                    trajectory_weights=[0.5, 0.5],  # length=2, depth=3
                )
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match="trajectory_weights length"):
        validate_cache_config(config)


def test_trajectory_negative_weights_rejected():
    config = CacheConfig(
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(
                    type="weighted_rrf_knn",
                    trajectory_depth=2,
                    trajectory_weights=[0.5, -0.1],
                )
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match="non-negative"):
        validate_cache_config(config)


def test_trajectory_zero_sum_weights_rejected():
    config = CacheConfig(
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(
                    type="weighted_rrf_knn",
                    trajectory_depth=2,
                    trajectory_weights=[0.0, 0.0],
                )
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match="sum must be > 0"):
        validate_cache_config(config)


def test_qdrant_trajectory_depth_gt1_rejected():
    config = CacheConfig(
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(
                    trajectory_depth=3,
                    trajectory_weights=[0.5, 0.3, 0.2],
                )
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match="not supported with Qdrant"):
        validate_cache_config(config)


def test_trajectory_depth_1_no_weights_ok():
    """depth=1 with no trajectory_weights should be valid."""
    config = CacheConfig(
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(trajectory_depth=1)
            ),
        },
    )
    # Should not raise
    validate_cache_config(config)


def test_in_memory_trajectory_depth_gt1_valid():
    """in_memory + trajectory_depth > 1 with matching weights should pass."""
    config = CacheConfig(
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(
                    type="weighted_rrf_knn",
                    trajectory_depth=3,
                    trajectory_weights=[0.5, 0.3, 0.2],
                )
            ),
        },
    )
    # Should not raise
    validate_cache_config(config)


# ---------------------------------------------------------------------------
# Write policy config validation tests
# ---------------------------------------------------------------------------


def test_write_policy_valid_types():
    for wtype in ("on_any_miss", "always", "never"):
        config = CacheConfig(
            write_policy=WritePolicyConfig(type=wtype),
        )
        validate_cache_config(config)  # Should not raise


def test_write_policy_invalid_type():
    config = CacheConfig(
        write_policy=WritePolicyConfig(type="invalid_policy"),
    )
    with pytest.raises(ConfigValidationError, match="write_policy.type"):
        validate_cache_config(config)


# ---------------------------------------------------------------------------
# warm_tiers validation
# ---------------------------------------------------------------------------


def _config_with_warm_tiers(warm_tiers, cp_name="cp1", judge_type="threshold", threshold=0.98):
    return CacheConfig(
        enabled=True,
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
        checkpoints={
            cp_name: CheckpointConfig(
                judge=JudgeConfig(type=judge_type, threshold=threshold, warm_tiers=warm_tiers),
            ),
        },
    )


def test_warm_tiers_valid():
    tiers = [
        {"threshold": 0.95, "start_t": 0.3},
        {"threshold": 0.90, "start_t": 0.5},
    ]
    config = _config_with_warm_tiers(tiers)
    validate_cache_config(config)


def test_warm_tiers_threshold_not_decreasing():
    tiers = [
        {"threshold": 0.90, "start_t": 0.3},
        {"threshold": 0.95, "start_t": 0.5},
    ]
    config = _config_with_warm_tiers(tiers)
    with pytest.raises(ConfigValidationError, match="strictly decreasing"):
        validate_cache_config(config)


def test_warm_tiers_invalid_start_t():
    tiers = [{"threshold": 0.95, "start_t": 0.35}]
    config = _config_with_warm_tiers(tiers)
    with pytest.raises(ConfigValidationError, match="not a valid timestep"):
        validate_cache_config(config)


def test_warm_tiers_cp3_rejected():
    tiers = [{"threshold": 0.90, "start_t": 0.3}]
    config = _config_with_warm_tiers(tiers, cp_name="cp3")
    with pytest.raises(ConfigValidationError, match="only supported on CP1"):
        validate_cache_config(config)


def test_warm_tiers_always_hit_rejected():
    tiers = [{"threshold": 0.90, "start_t": 0.3}]
    config = _config_with_warm_tiers(tiers, judge_type="always_hit")
    with pytest.raises(ConfigValidationError, match="requires judge.type='threshold'"):
        validate_cache_config(config)


def test_warm_tiers_first_tier_not_below_threshold():
    tiers = [{"threshold": 0.99, "start_t": 0.3}]
    config = _config_with_warm_tiers(tiers, threshold=0.98)
    with pytest.raises(ConfigValidationError, match="strictly decreasing"):
        validate_cache_config(config)


def test_warm_tiers_none_passes():
    config = _config_with_warm_tiers(None)
    validate_cache_config(config)


def test_warm_tiers_empty_list_passes():
    config = _config_with_warm_tiers([])
    validate_cache_config(config)


def test_warm_tiers_start_t_canonicalized():
    tiers = [{"threshold": 0.95, "start_t": 0.30000000000000004}]
    config = _config_with_warm_tiers(tiers)
    validate_cache_config(config)
    assert config.checkpoints["cp1"].judge.warm_tiers[0]["start_t"] == 0.3


# ---------------------------------------------------------------------------
# always_warm_start validation + _build_judge branch
# ---------------------------------------------------------------------------


def _config_with_always_warm_start(
    start_t: float | None = 0.5,
    cp_name: str = "cp1",
    warm_tiers: list | None = None,
):
    return CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            cp_name: CheckpointConfig(
                gate=GateConfig(type="always_search"),
                judge=JudgeConfig(
                    type="always_warm_start",
                    start_t=start_t,
                    warm_tiers=warm_tiers,
                ),
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"),
            ),
        },
    )


def test_always_warm_start_valid():
    config = _config_with_always_warm_start(start_t=0.5)
    validate_cache_config(config)


def test_always_warm_start_missing_start_t_rejected():
    config = _config_with_always_warm_start(start_t=None)
    with pytest.raises(ConfigValidationError, match="requires 'start_t'"):
        validate_cache_config(config)


def test_always_warm_start_non_canonical_start_t_rejected():
    config = _config_with_always_warm_start(start_t=0.35)
    with pytest.raises(ConfigValidationError, match="not a valid timestep"):
        validate_cache_config(config)


def test_always_warm_start_cp3_rejected():
    config = _config_with_always_warm_start(cp_name="cp3")
    with pytest.raises(ConfigValidationError, match="only supported on CP1"):
        validate_cache_config(config)


def test_always_warm_start_with_warm_tiers_rejected():
    tiers = [{"threshold": 0.9, "start_t": 0.3}]
    config = _config_with_always_warm_start(start_t=0.5, warm_tiers=tiers)
    with pytest.raises(ConfigValidationError, match="cannot be combined with warm_tiers"):
        validate_cache_config(config)


def test_always_warm_start_start_t_canonicalized():
    config = _config_with_always_warm_start(start_t=0.30000000000000004)
    validate_cache_config(config)
    assert config.checkpoints["cp1"].judge.start_t == 0.3


def test_always_warm_start_build_per_connection_returns_judge():
    from openpi.cache.components.judge import AlwaysWarmStartJudge
    from openpi.cache.config import build_per_connection_components, build_shared_storage

    config = _config_with_always_warm_start(start_t=0.5)
    validate_cache_config(config)
    shared = build_shared_storage(config)
    components = build_per_connection_components(config, shared, quiet=True)

    judge = components["judges"][CheckpointID.CP1]
    assert isinstance(judge, AlwaysWarmStartJudge)
    assert judge._start_t == 0.5


# ---------------------------------------------------------------------------
# AlwaysSkipGate: validation whitelist + _build_gate branch
# ---------------------------------------------------------------------------


def test_gate_always_skip_passes_validation():
    """Whitelist in validate_cache_config must accept gate.type='always_skip'."""
    config = CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_skip"),
                judge=JudgeConfig(threshold=0.98),
            ),
        },
    )
    # Should not raise.
    validate_cache_config(config)


def test_gate_unknown_type_rejected():
    config = CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="totally_bogus"),
                judge=JudgeConfig(threshold=0.98),
            ),
        },
    )
    with pytest.raises(ConfigValidationError, match="gate.type 'totally_bogus'"):
        validate_cache_config(config)


def test_build_gate_constructs_always_skip():
    """build_cache_components should instantiate AlwaysSkipGate for gate.type='always_skip'."""
    from openpi.cache.components.gate import AlwaysSkipGate

    config = CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_skip"),
                judge=JudgeConfig(threshold=0.98),
            ),
        },
    )
    validate_cache_config(config)
    components = build_cache_components(config)
    assert isinstance(components["gates"][CheckpointID.CP1], AlwaysSkipGate)


# ---------------------------------------------------------------------------
# Per-connection CacheStorage facade isolation
# ---------------------------------------------------------------------------


def test_per_connection_storage_is_fresh_facade_sharing_backend():
    """Each call to build_per_connection_components produces an independent facade.

    Prefill state (enter_prefill_mode) set on one connection's facade must not
    bleed into another's. See trajectory-deviation implementation plan §5.
    """
    from openpi.cache.config import build_per_connection_components, build_shared_storage

    config = CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_search"),
                judge=JudgeConfig(threshold=0.98),
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"),
            ),
        },
    )
    validate_cache_config(config)
    shared = build_shared_storage(config)

    conn_a = build_per_connection_components(config, shared, quiet=True)
    conn_b = build_per_connection_components(config, shared, quiet=True)

    # Two distinct facade instances …
    assert conn_a["storage"] is not conn_b["storage"]
    assert conn_a["storage"] is not shared
    # … but all three share the same backend.
    assert conn_a["storage"]._backend is shared._backend
    assert conn_b["storage"]._backend is shared._backend


def test_per_connection_prefill_state_is_isolated():
    """Entering prefill mode on facade A must not affect facade B."""
    from openpi.cache.config import build_per_connection_components, build_shared_storage
    from openpi.cache.storage_types import CachePayload

    config = CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_search"),
                judge=JudgeConfig(threshold=0.98),
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"),
            ),
        },
    )
    validate_cache_config(config)
    shared = build_shared_storage(config)

    conn_a = build_per_connection_components(config, shared, quiet=True)
    conn_b = build_per_connection_components(config, shared, quiet=True)

    payload = CachePayload(action_chunk=torch.zeros(50, 32))
    conn_a["storage"].enter_prefill_mode(payload)

    assert conn_a["storage"]._prefill_mode is True
    assert conn_b["storage"]._prefill_mode is False
    assert shared._prefill_mode is False


def test_per_connection_search_strategy_uses_owning_facade():
    """Each search_strategy must hold the *same* per-connection facade that
    was returned under ``"storage"`` — otherwise prefill mode would enter on
    facade A but the strategy would still search through ``shared``, bypassing
    the synthetic hit. Directly asserts on ``_storage`` identity, the pathway
    through which prefill short-circuits propagate.
    """
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.config import build_per_connection_components, build_shared_storage

    config = CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_search"),
                judge=JudgeConfig(threshold=0.98),
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"),
            ),
        },
    )
    validate_cache_config(config)
    shared = build_shared_storage(config)

    conn_a = build_per_connection_components(config, shared, quiet=True)
    conn_b = build_per_connection_components(config, shared, quiet=True)

    cp1_id = next(iter(conn_a["search_strategies"].keys()))
    strat_a = conn_a["search_strategies"][cp1_id]
    strat_b = conn_b["search_strategies"][cp1_id]

    # Strategy's storage MUST be its own connection's facade, not the shared one.
    assert isinstance(strat_a._storage, CacheStorage)
    assert strat_a._storage is conn_a["storage"]
    assert strat_b._storage is conn_b["storage"]
    assert strat_a._storage is not shared
    assert strat_a._storage is not strat_b._storage


# ---------------------------------------------------------------------------
# cp1_llm_layer_extract — KeyBuilderConfig + PrefixReducerConfig + factory
# ---------------------------------------------------------------------------


def _llm_layer_extract_config(
    *,
    extract_layer: int = 0,
    prefix_reducer_type: str = "prefix_mean_pool",
    enabled: dict[str, bool] | None = None,
    vector_dims: dict[str, int] | None = None,
) -> CacheConfig:
    """Helper: minimal valid cp1_llm_layer_extract config for in_memory backend."""
    from openpi.cache.config import PrefixReducerConfig

    enabled = enabled or {"vision_0": True, "robot_state": True}
    vector_dims = vector_dims or {"vision_0": 2048, "robot_state": 32}
    return CacheConfig(
        enabled=True,
        keys=KeysConfig(
            vision_0=KeyFieldConfig(enabled=enabled.get("vision_0", False)),
            vision_1=KeyFieldConfig(enabled=enabled.get("vision_1", False)),
            vision_2=KeyFieldConfig(enabled=enabled.get("vision_2", False)),
            prompt_emb=KeyFieldConfig(enabled=enabled.get("prompt_emb", False)),
            robot_state=KeyFieldConfig(enabled=enabled.get("robot_state", True)),
        ),
        backend=BackendConfig(
            type="in_memory",
            vector_dims=vector_dims,
            in_memory=__import__("openpi.cache.config", fromlist=["InMemoryConfig"])
                .InMemoryConfig(preload_path="dummy.pkl"),
        ),
        key_builder=KeyBuilderConfig(
            type="cp1_llm_layer_extract",
            extract_layer=extract_layer,
            prefix_reducer=PrefixReducerConfig(type=prefix_reducer_type),
        ),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_search"),
                judge=JudgeConfig(threshold=0.98),
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"),
            ),
        },
    )


def test_llm_layer_extract_valid_prefix_mean_pool():
    config = _llm_layer_extract_config()
    validate_cache_config(config)


def test_llm_layer_extract_valid_per_modality_mean_pool():
    config = _llm_layer_extract_config(
        prefix_reducer_type="per_modality_mean_pool",
        enabled={"vision_0": True, "vision_1": True, "vision_2": True,
                 "prompt_emb": True, "robot_state": True},
        vector_dims={"vision_0": 2048, "vision_1": 2048, "vision_2": 2048,
                     "prompt_emb": 2048, "robot_state": 32},
    )
    validate_cache_config(config)


def test_llm_layer_extract_layer_out_of_range():
    config = _llm_layer_extract_config(extract_layer=18)
    with pytest.raises(ConfigValidationError, match="extract_layer=18 out of range"):
        validate_cache_config(config)


def test_llm_layer_extract_layer_negative():
    config = _llm_layer_extract_config(extract_layer=-1)
    with pytest.raises(ConfigValidationError, match="extract_layer=-1 out of range"):
        validate_cache_config(config)


def test_llm_layer_extract_unknown_prefix_reducer_type():
    config = _llm_layer_extract_config(prefix_reducer_type="bogus")
    with pytest.raises(ConfigValidationError, match="prefix_reducer.type 'bogus' unknown"):
        validate_cache_config(config)


def test_llm_layer_extract_prefix_mean_pool_rejects_extra_vision_field():
    """prefix_mean_pool only emits vision_0; enabling vision_1 must fail loudly."""
    config = _llm_layer_extract_config(
        prefix_reducer_type="prefix_mean_pool",
        enabled={"vision_0": True, "vision_1": True, "robot_state": True},
        vector_dims={"vision_0": 2048, "vision_1": 2048, "robot_state": 32},
    )
    with pytest.raises(ConfigValidationError, match="would never be"):
        validate_cache_config(config)


def test_llm_layer_extract_prefix_mean_pool_rejects_prompt_emb():
    config = _llm_layer_extract_config(
        prefix_reducer_type="prefix_mean_pool",
        enabled={"vision_0": True, "prompt_emb": True, "robot_state": True},
        vector_dims={"vision_0": 2048, "prompt_emb": 2048, "robot_state": 32},
    )
    with pytest.raises(ConfigValidationError, match="would never be"):
        validate_cache_config(config)


def test_llm_layer_extract_dim_mismatch_rejected():
    """vector_dims.vision_0 must equal 2048 (gemma_2b width)."""
    config = _llm_layer_extract_config(
        vector_dims={"vision_0": 999, "robot_state": 32},
    )
    with pytest.raises(ConfigValidationError, match="does not match prefix_reducer output dim 2048"):
        validate_cache_config(config)


def test_llm_layer_extract_factory_returns_correct_class():
    """build_cache_components produces CP1LLMLayerExtractKeyBuilder."""
    from openpi.cache.components.llm_layer_key_builder import (
        CP1LLMLayerExtractKeyBuilder,
    )

    config = _llm_layer_extract_config()
    # Override backend to qdrant so we don't need a real artifact file for build.
    config.backend = BackendConfig(
        type="qdrant", vector_dims={"vision_0": 2048, "robot_state": 32},
    )
    # qdrant requires its own search strategy.
    config.checkpoints["cp1"].search_strategy = SearchStrategyConfig(
        type="qdrant_weighted_rrf_knn",
    )
    validate_cache_config(config)
    components = build_cache_components(config)
    assert isinstance(components["key_builder"], CP1LLMLayerExtractKeyBuilder)


# ---- New per_modality reducers: max_pool / spatial_pool_16 / spatial_pool_4 ----

def test_llm_layer_extract_valid_per_modality_max_pool():
    config = _llm_layer_extract_config(
        prefix_reducer_type="per_modality_max_pool",
        enabled={"vision_0": True, "vision_1": True, "vision_2": True,
                 "prompt_emb": True, "robot_state": True},
        vector_dims={"vision_0": 2048, "vision_1": 2048, "vision_2": 2048,
                     "prompt_emb": 2048, "robot_state": 32},
    )
    validate_cache_config(config)  # should not raise


def test_llm_layer_extract_valid_per_modality_spatial_pool_16():
    config = _llm_layer_extract_config(
        prefix_reducer_type="per_modality_spatial_pool_16",
        enabled={"vision_0": True, "vision_1": True, "vision_2": True,
                 "prompt_emb": True, "robot_state": True},
        vector_dims={"vision_0": 32768, "vision_1": 32768, "vision_2": 32768,
                     "prompt_emb": 2048, "robot_state": 32},
    )
    validate_cache_config(config)


def test_llm_layer_extract_valid_per_modality_spatial_pool_4():
    config = _llm_layer_extract_config(
        prefix_reducer_type="per_modality_spatial_pool_4",
        enabled={"vision_0": True, "vision_1": True, "vision_2": True,
                 "prompt_emb": True, "robot_state": True},
        vector_dims={"vision_0": 8192, "vision_1": 8192, "vision_2": 8192,
                     "prompt_emb": 2048, "robot_state": 32},
    )
    validate_cache_config(config)


def test_llm_layer_extract_spatial_pool_16_rejects_wrong_vision_dim():
    """vision dim must be 32768 for spatial_pool_16 (16 * 2048); reject 2048."""
    config = _llm_layer_extract_config(
        prefix_reducer_type="per_modality_spatial_pool_16",
        enabled={"vision_0": True, "prompt_emb": True, "robot_state": True},
        vector_dims={"vision_0": 2048, "prompt_emb": 2048, "robot_state": 32},
    )
    with pytest.raises(ConfigValidationError, match="does not match prefix_reducer output dim 32768"):
        validate_cache_config(config)


def test_llm_layer_extract_spatial_pool_4_rejects_wrong_vision_dim():
    config = _llm_layer_extract_config(
        prefix_reducer_type="per_modality_spatial_pool_4",
        enabled={"vision_0": True, "prompt_emb": True, "robot_state": True},
        vector_dims={"vision_0": 32768, "prompt_emb": 2048, "robot_state": 32},
    )
    with pytest.raises(ConfigValidationError, match="does not match prefix_reducer output dim 8192"):
        validate_cache_config(config)


def test_llm_layer_extract_spatial_pool_16_rejects_wrong_prompt_dim():
    """prompt_emb falls back to masked mean (2048); reject 32768."""
    config = _llm_layer_extract_config(
        prefix_reducer_type="per_modality_spatial_pool_16",
        enabled={"vision_0": True, "prompt_emb": True, "robot_state": True},
        vector_dims={"vision_0": 32768, "prompt_emb": 32768, "robot_state": 32},
    )
    with pytest.raises(ConfigValidationError, match="does not match prefix_reducer output dim 2048"):
        validate_cache_config(config)


def test_llm_layer_extract_factory_max_pool_returns_correct_reducer():
    from openpi.cache.components.prefix_reducer import PerModalityMaxPoolReducer
    from openpi.cache.config import PrefixReducerConfig, _build_prefix_reducer

    reducer = _build_prefix_reducer(PrefixReducerConfig(type="per_modality_max_pool"))
    assert isinstance(reducer, PerModalityMaxPoolReducer)


def test_llm_layer_extract_factory_spatial_pool_16_returns_correct_reducer():
    from openpi.cache.components.prefix_reducer import PerModalitySpatialPoolReducer
    from openpi.cache.config import PrefixReducerConfig, _build_prefix_reducer

    reducer = _build_prefix_reducer(
        PrefixReducerConfig(type="per_modality_spatial_pool_16"),
    )
    assert isinstance(reducer, PerModalitySpatialPoolReducer)
    assert reducer.output_dims["vision_0"] == 32768
    assert reducer.output_dims["prompt_emb"] == 2048


def test_llm_layer_extract_factory_spatial_pool_4_returns_correct_reducer():
    from openpi.cache.components.prefix_reducer import PerModalitySpatialPoolReducer
    from openpi.cache.config import PrefixReducerConfig, _build_prefix_reducer

    reducer = _build_prefix_reducer(
        PrefixReducerConfig(type="per_modality_spatial_pool_4"),
    )
    assert isinstance(reducer, PerModalitySpatialPoolReducer)
    assert reducer.output_dims["vision_0"] == 8192


# ---------------------------------------------------------------------------
# cp1_spatial_pool_4 canonical name / cp1_spatial_pool_64 backward-compat alias
# ---------------------------------------------------------------------------


def test_cp1_spatial_pool_4_and_64_resolve_to_same_class():
    """The legacy `cp1_spatial_pool_64` (64x compression ratio naming) and the
    canonical `cp1_spatial_pool_4` (4 output tokens) must instantiate the same
    key builder class — the rename only affects the public type string."""
    from openpi.cache.components.key_builder import (
        CP1SpatialPool4KeyBuilder,
        CP1SpatialPool64KeyBuilder,
    )
    from openpi.cache.config import _build_key_builder

    assert CP1SpatialPool4KeyBuilder is CP1SpatialPool64KeyBuilder
    enabled_fields = ["vision_0", "robot_state"]
    vector_dims = {"vision_0": 8192, "robot_state": 32}
    a = _build_key_builder(
        KeyBuilderConfig(type="cp1_spatial_pool_4"), enabled_fields, vector_dims,
    )
    b = _build_key_builder(
        KeyBuilderConfig(type="cp1_spatial_pool_64"), enabled_fields, vector_dims,
    )
    assert type(a) is type(b) is CP1SpatialPool4KeyBuilder


# ---------------------------------------------------------------------------
# RandomGate / PeriodicGate dispatch + validation
# Plan: logs/random_periodic_gate_plan.log.md §5.2 / §5.4
# ---------------------------------------------------------------------------


def _minimal_cache_config_with_gate(gate_cfg: GateConfig) -> CacheConfig:
    """Build a minimal CacheConfig that only exercises gate validation.

    keys.robot_state enabled + placeholder key_builder + in_memory backend
    with matching vector_dims is the smallest passing skeleton; only the
    gate block is varied across tests.
    """
    return CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        key_builder=KeyBuilderConfig(type="placeholder"),
        checkpoints={
            "cp1": CheckpointConfig(
                enabled=True,
                gate=gate_cfg,
                judge=JudgeConfig(type="always_hit"),
                search_strategy=SearchStrategyConfig(type="qdrant_weighted_rrf_knn"),
            ),
        },
        backend=BackendConfig(type="qdrant", vector_dims={"robot_state": 32}),
    )


def test_random_gate_valid_config_passes():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="random", p_inference=0.3, seed=0)
    )
    validate_cache_config(cfg)  # must not raise


def test_random_gate_missing_p_inference_rejected():
    cfg = _minimal_cache_config_with_gate(GateConfig(type="random", seed=0))
    with pytest.raises(ConfigValidationError, match="requires 'p_inference'"):
        validate_cache_config(cfg)


def test_random_gate_missing_seed_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="random", p_inference=0.3)
    )
    with pytest.raises(ConfigValidationError, match="requires 'seed'"):
        validate_cache_config(cfg)


def test_random_gate_out_of_range_p_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="random", p_inference=1.5, seed=0)
    )
    with pytest.raises(ConfigValidationError, match="p_inference=1.5"):
        validate_cache_config(cfg)


def test_random_gate_negative_seed_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="random", p_inference=0.3, seed=-1)
    )
    with pytest.raises(ConfigValidationError, match="seed=-1"):
        validate_cache_config(cfg)


def test_random_gate_rejects_periodic_fields():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="random", p_inference=0.3, seed=0, cache_len=5)
    )
    with pytest.raises(ConfigValidationError, match="cannot set.*cache_len"):
        validate_cache_config(cfg)


def test_periodic_gate_valid_config_passes():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="periodic", cache_len=5, inference_len=2)
    )
    validate_cache_config(cfg)


def test_periodic_gate_missing_cache_len_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="periodic", inference_len=2)
    )
    with pytest.raises(ConfigValidationError, match="requires 'cache_len'"):
        validate_cache_config(cfg)


def test_periodic_gate_missing_inference_len_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="periodic", cache_len=5)
    )
    with pytest.raises(ConfigValidationError, match="requires 'inference_len'"):
        validate_cache_config(cfg)


def test_periodic_gate_zero_cache_len_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="periodic", cache_len=0, inference_len=2)
    )
    with pytest.raises(ConfigValidationError, match="cache_len=0"):
        validate_cache_config(cfg)


def test_periodic_gate_zero_inference_len_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="periodic", cache_len=5, inference_len=0)
    )
    with pytest.raises(ConfigValidationError, match="inference_len=0"):
        validate_cache_config(cfg)


def test_periodic_gate_rejects_random_fields():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="periodic", cache_len=5, inference_len=2, p_inference=0.3)
    )
    with pytest.raises(ConfigValidationError, match="cannot set.*p_inference"):
        validate_cache_config(cfg)


def test_legacy_always_search_gate_rejects_new_fields():
    # Regression: existing 3 gate types must not silently accept new param fields.
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="always_search", p_inference=0.3)
    )
    with pytest.raises(ConfigValidationError, match="type='always_search'.*cannot set"):
        validate_cache_config(cfg)


def test_legacy_always_search_gate_without_new_fields_passes():
    cfg = _minimal_cache_config_with_gate(GateConfig(type="always_search"))
    validate_cache_config(cfg)


def test_legacy_always_skip_gate_without_new_fields_passes():
    cfg = _minimal_cache_config_with_gate(GateConfig(type="always_skip"))
    validate_cache_config(cfg)


def test_legacy_client_controlled_gate_without_new_fields_passes():
    cfg = _minimal_cache_config_with_gate(GateConfig(type="client_controlled"))
    validate_cache_config(cfg)


def test_build_gate_constructs_random_gate():
    from openpi.cache.components.gate import RandomGate
    from openpi.cache.config import _build_gate

    g = _build_gate(GateConfig(type="random", p_inference=0.25, seed=1))
    assert isinstance(g, RandomGate)


def test_build_gate_constructs_periodic_gate():
    from openpi.cache.components.gate import PeriodicGate
    from openpi.cache.config import _build_gate

    g = _build_gate(GateConfig(type="periodic", cache_len=3, inference_len=2))
    assert isinstance(g, PeriodicGate)


# ---------------------------------------------------------------------------
# Integer-typed gate field regressions (G2 Round 1 Blocking #2).
# ---------------------------------------------------------------------------


def test_random_gate_rejects_fractional_seed():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="random", p_inference=0.3, seed=0.5)
    )
    with pytest.raises(ConfigValidationError, match="seed must be a non-negative int"):
        validate_cache_config(cfg)


def test_random_gate_rejects_bool_seed():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="random", p_inference=0.3, seed=True)
    )
    with pytest.raises(ConfigValidationError, match="seed must be a non-negative int"):
        validate_cache_config(cfg)


def test_random_gate_rejects_non_numeric_p_inference():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="random", p_inference="0.3", seed=0)
    )
    with pytest.raises(ConfigValidationError, match="p_inference must be a real number"):
        validate_cache_config(cfg)


def test_periodic_gate_rejects_fractional_cache_len():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="periodic", cache_len=1.5, inference_len=2)
    )
    with pytest.raises(ConfigValidationError, match="cache_len must be an int"):
        validate_cache_config(cfg)


def test_periodic_gate_rejects_fractional_inference_len():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="periodic", cache_len=1, inference_len=2.5)
    )
    with pytest.raises(ConfigValidationError, match="inference_len must be an int"):
        validate_cache_config(cfg)


def test_periodic_gate_rejects_bool_cache_len():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="periodic", cache_len=True, inference_len=1)
    )
    with pytest.raises(ConfigValidationError, match="cache_len must be an int"):
        validate_cache_config(cfg)


# ---------------------------------------------------------------------------
# ScoreHysteresisGate (server-side N1) dispatch + validation.
# Plan: logs/n1_serverside_gate_stage1c.log.md §4.4.
# ---------------------------------------------------------------------------


def test_score_hysteresis_gate_valid_config_passes():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(
            type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2,
            probe_interval=3,
        )
    )
    validate_cache_config(cfg)  # must not raise


def test_score_hysteresis_gate_probe_interval_omitted_passes():
    # probe_interval is optional (None -> never probe).
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2)
    )
    validate_cache_config(cfg)


def test_score_hysteresis_gate_missing_theta_low_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="score_hysteresis", theta_high=0.95, j=2)
    )
    with pytest.raises(ConfigValidationError, match="requires 'theta_low'"):
        validate_cache_config(cfg)


def test_score_hysteresis_gate_missing_j_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.95)
    )
    with pytest.raises(ConfigValidationError, match="requires 'j'"):
        validate_cache_config(cfg)


def test_score_hysteresis_gate_theta_high_below_low_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.5, j=2)
    )
    with pytest.raises(ConfigValidationError, match="theta_high >= theta_low"):
        validate_cache_config(cfg)


def test_score_hysteresis_gate_non_finite_theta_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(
            type="score_hysteresis", theta_low=float("nan"), theta_high=0.95, j=2
        )
    )
    with pytest.raises(ConfigValidationError, match="must be finite"):
        validate_cache_config(cfg)


def test_score_hysteresis_gate_zero_j_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=0)
    )
    with pytest.raises(ConfigValidationError, match="j=0 must be >= 1"):
        validate_cache_config(cfg)


def test_score_hysteresis_gate_zero_probe_interval_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(
            type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2,
            probe_interval=0,
        )
    )
    with pytest.raises(ConfigValidationError, match="probe_interval=0 must be >= 1"):
        validate_cache_config(cfg)


def test_score_hysteresis_gate_rejects_bool_j():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=True)
    )
    with pytest.raises(ConfigValidationError, match="j must be an int"):
        validate_cache_config(cfg)


def test_score_hysteresis_gate_rejects_bool_theta():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="score_hysteresis", theta_low=True, theta_high=0.95, j=2)
    )
    with pytest.raises(ConfigValidationError, match="theta_low must be a real number"):
        validate_cache_config(cfg)


def test_score_hysteresis_gate_rejects_stray_fields():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(
            type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2,
            p_inference=0.3,
        )
    )
    with pytest.raises(ConfigValidationError, match="cannot set.*p_inference"):
        validate_cache_config(cfg)


# --- Stage 3b: score_hysteresis L (V2 injection threshold) ---


def test_score_hysteresis_gate_valid_L_passes():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(
            type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2,
            probe_interval=3, L=6,
        )
    )
    validate_cache_config(cfg)  # must not raise


def test_score_hysteresis_gate_L_omitted_passes():
    # L is optional (None -> pure N1). Omitting it must not raise.
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2)
    )
    validate_cache_config(cfg)


def test_score_hysteresis_gate_zero_L_rejected():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2, L=0)
    )
    with pytest.raises(ConfigValidationError, match="L=0 must be >= 1"):
        validate_cache_config(cfg)


def test_score_hysteresis_gate_rejects_bool_L():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2, L=True)
    )
    with pytest.raises(ConfigValidationError, match="L must be None or an int"):
        validate_cache_config(cfg)


def test_score_hysteresis_gate_rejects_float_L():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2, L=1.5)
    )
    with pytest.raises(ConfigValidationError, match="L must be None or an int"):
        validate_cache_config(cfg)


def test_build_gate_score_hysteresis_carries_L():
    from openpi.cache.components.gate import ScoreHysteresisGate
    from openpi.cache.config import _build_gate

    g = _build_gate(
        GateConfig(
            type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2,
            probe_interval=3, L=6,
        )
    )
    assert isinstance(g, ScoreHysteresisGate) and g._L == 6


@pytest.mark.parametrize("gate", [
    GateConfig(type="always_search"),
    GateConfig(type="always_skip"),
    GateConfig(type="client_controlled"),
    GateConfig(type="random", p_inference=0.3, seed=0),
    GateConfig(type="periodic", cache_len=3, inference_len=1),
])
def test_legacy_gates_not_stray_from_L_default(gate):
    # Adding `L` (default None) to GateConfig must NOT make legacy gates appear to
    # set an alien field. Every legacy config must still validate cleanly.
    validate_cache_config(_minimal_cache_config_with_gate(gate))


def test_legacy_gate_rejects_score_hysteresis_fields():
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="always_search", theta_low=0.9)
    )
    with pytest.raises(ConfigValidationError, match="type='always_search'.*cannot set"):
        validate_cache_config(cfg)


def test_build_gate_constructs_score_hysteresis_gate():
    from openpi.cache.components.gate import ScoreHysteresisGate
    from openpi.cache.config import _build_gate

    g = _build_gate(
        GateConfig(
            type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2,
            probe_interval=3,
        )
    )
    assert isinstance(g, ScoreHysteresisGate)


def test_build_gate_score_hysteresis_probe_interval_none():
    from openpi.cache.components.gate import ScoreHysteresisGate
    from openpi.cache.config import _build_gate

    g = _build_gate(
        GateConfig(type="score_hysteresis", theta_low=0.9, theta_high=0.95, j=2)
    )
    assert isinstance(g, ScoreHysteresisGate)


# ---------------------------------------------------------------------------
# weighted_score_sum_knn per_field score_normalization validation
# ---------------------------------------------------------------------------


def _score_sum_config(score_normalization, field_similarity=None):
    return CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                search_strategy=SearchStrategyConfig(
                    type="weighted_score_sum_knn",
                    field_similarity=field_similarity,
                    score_normalization=score_normalization,
                ),
            ),
        },
    )


def test_score_sum_per_field_valid():
    cfg = _score_sum_config(
        ScoreNormalizationConfig(
            type="per_field",
            fields={"robot_state": {"method": "exp_l2", "params": {"tau": 0.33}}},
        ),
        field_similarity={
            "robot_state": FieldSimilarityConfig(type="l2", to_similarity={"tau": 0.33})
        },
    )
    validate_cache_config(cfg)  # must not raise


def test_score_sum_rejects_type_none():
    cfg = _score_sum_config(ScoreNormalizationConfig(type="none"))
    with pytest.raises(ConfigValidationError, match="requires score_normalization"):
        validate_cache_config(cfg)


def test_score_sum_rejects_unknown_type():
    # A typo'd type must fail fast, not silently fall back to no-normalization.
    cfg = _score_sum_config(
        ScoreNormalizationConfig(
            type="typo",
            fields={"robot_state": {"method": "exp_l2", "params": {"tau": 0.33}}},
        ),
        field_similarity={
            "robot_state": FieldSimilarityConfig(type="l2", to_similarity={"tau": 0.33})
        },
    )
    with pytest.raises(ConfigValidationError, match="unknown score_normalization.type"):
        validate_cache_config(cfg)


def test_score_sum_per_field_unknown_method():
    cfg = _score_sum_config(
        ScoreNormalizationConfig(
            type="per_field",
            fields={"robot_state": {"method": "bogus", "params": {}}},
        ),
        field_similarity={
            "robot_state": FieldSimilarityConfig(type="l2", to_similarity={"tau": 0.33})
        },
    )
    with pytest.raises(ConfigValidationError, match="unknown or ineligible"):
        validate_cache_config(cfg)


def test_score_sum_per_field_sim_type_incompatible():
    # logit is cosine-only; robot_state is l2 -> must be rejected.
    cfg = _score_sum_config(
        ScoreNormalizationConfig(
            type="per_field",
            fields={"robot_state": {"method": "logit", "params": {"lo": 0.0, "hi": 1.0}}},
        ),
        field_similarity={
            "robot_state": FieldSimilarityConfig(type="l2", to_similarity={"tau": 0.33})
        },
    )
    with pytest.raises(ConfigValidationError, match="incompatible with sim_type"):
        validate_cache_config(cfg)


def test_score_sum_per_field_rejects_back_compat_method():
    # legacy_percentile / direction_unify are reachable only via type:percentile
    # / type:none, never as a per_field method.
    cfg = _score_sum_config(
        ScoreNormalizationConfig(
            type="per_field",
            fields={"robot_state": {"method": "direction_unify", "params": {}}},
        ),
        field_similarity={
            "robot_state": FieldSimilarityConfig(type="l2", to_similarity={"tau": 0.33})
        },
    )
    with pytest.raises(ConfigValidationError, match="ineligible"):
        validate_cache_config(cfg)


def test_score_sum_per_field_missing_coverage():
    # robot_state is weighted but absent from fields -> coverage error.
    cfg = _score_sum_config(
        ScoreNormalizationConfig(
            type="per_field",
            fields={"vision_0": {"method": "affine_clip", "params": {"lo": 0.0, "hi": 1.0}}},
        ),
    )
    with pytest.raises(ConfigValidationError, match="missing"):
        validate_cache_config(cfg)


# ---------------------------------------------------------------------------
# follow_winner gate (Stage 4a N2) validation
# ---------------------------------------------------------------------------


def _follow_winner_cfg(gate_cfg: GateConfig) -> CacheConfig:
    """follow_winner needs an in_memory backend; the shared helper defaults to
    qdrant (with a qdrant-only search strategy), so swap in an in_memory backend
    and an in_memory-compatible strategy for the valid / gate-param tests."""
    cfg = _minimal_cache_config_with_gate(gate_cfg)
    cfg.backend = BackendConfig(type="in_memory", vector_dims={"robot_state": 32})
    # weighted_rrf_knn is the minimal in_memory-compatible strategy (unlike
    # weighted_score_sum_knn it needs no score_normalization block), so the only
    # thing under test here is the gate validation.
    cfg.checkpoints["cp1"].search_strategy = SearchStrategyConfig(type="weighted_rrf_knn")
    return cfg


def test_follow_winner_gate_valid_config_passes():
    cfg = _follow_winner_cfg(GateConfig(type="follow_winner", lock_streak=3, budget=5))
    validate_cache_config(cfg)  # must not raise


def test_follow_winner_gate_missing_lock_streak_rejected():
    cfg = _follow_winner_cfg(GateConfig(type="follow_winner", budget=5))
    with pytest.raises(ConfigValidationError, match="requires 'lock_streak'"):
        validate_cache_config(cfg)


def test_follow_winner_gate_missing_budget_rejected():
    cfg = _follow_winner_cfg(GateConfig(type="follow_winner", lock_streak=3))
    with pytest.raises(ConfigValidationError, match="requires 'budget'"):
        validate_cache_config(cfg)


def test_follow_winner_gate_zero_lock_streak_rejected():
    cfg = _follow_winner_cfg(GateConfig(type="follow_winner", lock_streak=0, budget=5))
    with pytest.raises(ConfigValidationError, match="lock_streak=0"):
        validate_cache_config(cfg)


def test_follow_winner_gate_stray_field_rejected():
    cfg = _follow_winner_cfg(
        GateConfig(type="follow_winner", lock_streak=3, budget=5, cache_len=7)
    )
    with pytest.raises(ConfigValidationError, match="cannot set"):
        validate_cache_config(cfg)


def test_follow_winner_gate_requires_in_memory_backend():
    # The shared helper backend is qdrant; blind replay needs in_memory -> reject.
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="follow_winner", lock_streak=3, budget=5)
    )
    with pytest.raises(ConfigValidationError, match="requires backend"):
        validate_cache_config(cfg)


def test_legacy_gate_rejects_follow_winner_fields():
    # A legacy gate carrying lock_streak / budget is rejected as stray fields
    # (they are now part of _gate_all_param_fields).
    cfg = _minimal_cache_config_with_gate(
        GateConfig(type="always_search", lock_streak=3, budget=5)
    )
    with pytest.raises(ConfigValidationError, match="cannot set"):
        validate_cache_config(cfg)


# ---------------------------------------------------------------------------
# Projection key builder (M1 Phase 2)
# ---------------------------------------------------------------------------


def _robot_state_only_keys():
    return KeysConfig(
        vision_0=KeyFieldConfig(enabled=False),
        vision_1=KeyFieldConfig(enabled=False),
        vision_2=KeyFieldConfig(enabled=False),
        prompt_emb=KeyFieldConfig(enabled=False),
        robot_state=KeyFieldConfig(enabled=True),
    )


def _vision0_robot_state_keys():
    return KeysConfig(
        vision_0=KeyFieldConfig(enabled=True),
        vision_1=KeyFieldConfig(enabled=False),
        vision_2=KeyFieldConfig(enabled=False),
        prompt_emb=KeyFieldConfig(enabled=False),
        robot_state=KeyFieldConfig(enabled=True),
    )


def test_projection_factory_builds_wrapper_and_identity_matches_pool():
    from types import SimpleNamespace

    from openpi.cache.components.projection_key_builder import ProjectionKeyBuilder
    from openpi.cache.config import ProjectionKeyBuilderConfig, _build_key_builder
    from openpi.cache.types import PROMPT_EMB, ROBOT_STATE, VISION_0, VISION_1

    enabled = [VISION_0, VISION_1, PROMPT_EMB, ROBOT_STATE]
    dims = {VISION_0: 2048, VISION_1: 2048, PROMPT_EMB: 2048, ROBOT_STATE: 32}
    prefix_len = 256 * 3 + 20
    stage1 = SimpleNamespace(
        prefix_embs=torch.randn(1, prefix_len, 2048), state=torch.randn(1, 32)
    )

    kb = _build_key_builder(
        KeyBuilderConfig(
            type="projection",
            projection=ProjectionKeyBuilderConfig(inner_type="cp1_mean_pool"),
        ),
        enabled,
        dims,
    )
    assert isinstance(kb, ProjectionKeyBuilder)

    pool = _build_key_builder(KeyBuilderConfig(type="cp1_mean_pool"), enabled, dims)
    kb.collect(CheckpointID.CP1, stage1=stage1)
    pool.collect(CheckpointID.CP1, stage1=stage1)
    got = kb.build(CheckpointID.CP1)
    expected = pool.build(CheckpointID.CP1)

    assert set(got) == set(expected)
    for field in expected:
        assert torch.equal(got[field], expected[field]), field


def test_projection_factory_rejects_bad_inner_type():
    from openpi.cache.config import ProjectionKeyBuilderConfig, _build_key_builder
    from openpi.cache.types import ROBOT_STATE

    cfg = KeyBuilderConfig(
        type="projection",
        projection=ProjectionKeyBuilderConfig(inner_type="bogus"),
    )
    with pytest.raises(ConfigValidationError, match="stateless pool"):
        _build_key_builder(cfg, [ROBOT_STATE], {ROBOT_STATE: 32})


def test_projection_factory_rejects_recursive_inner():
    from openpi.cache.config import ProjectionKeyBuilderConfig, _build_key_builder
    from openpi.cache.types import ROBOT_STATE

    cfg = KeyBuilderConfig(
        type="projection",
        projection=ProjectionKeyBuilderConfig(inner_type="projection"),
    )
    with pytest.raises(ConfigValidationError, match="stateless pool"):
        _build_key_builder(cfg, [ROBOT_STATE], {ROBOT_STATE: 32})


def test_projection_validator_accepts_good_inner():
    from openpi.cache.config import ProjectionKeyBuilderConfig

    # A projection over cp1_mean_pool inherits the cp1 key contract: it needs
    # vision_0 + robot_state (qdrant backend => no preload requirement).
    config = CacheConfig(
        keys=_vision0_robot_state_keys(),
        backend=BackendConfig(vector_dims={"vision_0": 2048, "robot_state": 32}),
        key_builder=KeyBuilderConfig(
            type="projection",
            projection=ProjectionKeyBuilderConfig(inner_type="cp1_mean_pool"),
        ),
    )
    validate_cache_config(config)  # no raise


def test_projection_inherits_cp1_key_enablement_contract():
    # G2 finding 1: projection(cp1_mean_pool) must inherit the cp1 requirement
    # for vision_0; a robot-state-only config is rejected just like direct cp1_*.
    from openpi.cache.config import ProjectionKeyBuilderConfig

    config = CacheConfig(
        keys=_robot_state_only_keys(),
        backend=BackendConfig(vector_dims={"robot_state": 32}),
        key_builder=KeyBuilderConfig(
            type="projection",
            projection=ProjectionKeyBuilderConfig(inner_type="cp1_mean_pool"),
        ),
    )
    with pytest.raises(ConfigValidationError, match="requires keys.vision_0"):
        validate_cache_config(config)


def test_projection_in_memory_requires_preload():
    # G2 finding 2: in_memory + projection(cp1_*) inherits the cp1 preload
    # requirement, so a missing preload_path is rejected.
    from openpi.cache.config import ProjectionKeyBuilderConfig

    config = CacheConfig(
        keys=_vision0_robot_state_keys(),
        backend=BackendConfig(
            type="in_memory", vector_dims={"vision_0": 2048, "robot_state": 32}
        ),
        key_builder=KeyBuilderConfig(
            type="projection",
            projection=ProjectionKeyBuilderConfig(inner_type="cp1_mean_pool"),
        ),
    )
    with pytest.raises(ConfigValidationError, match="preload_path"):
        validate_cache_config(config)


def test_projection_in_memory_with_preload_accepts():
    from openpi.cache.config import InMemoryConfig, ProjectionKeyBuilderConfig

    # in_memory requires an in_memory-compatible search strategy (the default
    # qdrant_weighted_rrf_knn is qdrant-only); use weighted_rrf_knn.
    in_mem_cp = {
        "cp1": CheckpointConfig(
            search_strategy=SearchStrategyConfig(type="weighted_rrf_knn")
        ),
        "cp3": CheckpointConfig(
            search_strategy=SearchStrategyConfig(type="weighted_rrf_knn")
        ),
    }
    config = CacheConfig(
        keys=_vision0_robot_state_keys(),
        backend=BackendConfig(
            type="in_memory",
            vector_dims={"vision_0": 2048, "robot_state": 32},
            in_memory=InMemoryConfig(preload_path="artifacts/proj.pkl"),
        ),
        key_builder=KeyBuilderConfig(
            type="projection",
            projection=ProjectionKeyBuilderConfig(inner_type="cp1_mean_pool"),
        ),
        checkpoints=in_mem_cp,
    )
    validate_cache_config(config)  # no raise


def test_projection_validator_rejects_bad_inner():
    from openpi.cache.config import ProjectionKeyBuilderConfig

    config = CacheConfig(
        keys=_robot_state_only_keys(),
        backend=BackendConfig(vector_dims={"robot_state": 32}),
        key_builder=KeyBuilderConfig(
            type="projection",
            projection=ProjectionKeyBuilderConfig(inner_type="bogus"),
        ),
    )
    with pytest.raises(ConfigValidationError, match="stateless pool"):
        validate_cache_config(config)
