"""Schedule identity through config validation, judge construction and storage binding."""

from __future__ import annotations

import types

import pytest

from openpi.cache.components.judge import AlwaysWarmStartJudge
from openpi.cache.config import (
    BackendConfig,
    CacheConfig,
    CheckpointConfig,
    ComposerConfig,
    ConfigValidationError,
    GateConfig,
    InMemoryConfig,
    JudgeConfig,
    KeyFieldConfig,
    KeysConfig,
    SearchStrategyConfig,
    _build_inner_judge,
    _check_denoise_schedule_binding,
    _check_warm_library_completeness,
    effective_denoise_schedule,
    required_warm_timesteps,
    validate_cache_config,
)
from openpi.cache.types import PI05_V1, groot_n15_schedule


def _config(
    judge: JudgeConfig, *, schedule: str | None = None, preload: str | None = "lib.pkl"
):
    return CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(
            type="in_memory",
            vector_dims={"robot_state": 32},
            in_memory=InMemoryConfig(preload_path=preload),
        ),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_search"),
                judge=judge,
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"),
            ),
        },
        denoise_schedule=schedule,
    )


def _storage(meta):
    return types.SimpleNamespace(artifact_meta=meta)


# ---------------------------------------------------------------------------
# validate_cache_config
# ---------------------------------------------------------------------------


def test_unknown_schedule_id_is_a_config_error():
    with pytest.raises(ConfigValidationError, match="unknown denoise schedule id"):
        validate_cache_config(
            _config(JudgeConfig(type="always_hit"), schedule="groot_n15_v1")
        )


def test_legacy_reading_keeps_pi05_timesteps_when_no_schedule_is_named():
    validate_cache_config(_config(JudgeConfig(type="always_warm_start", start_t=0.5)))
    with pytest.raises(ConfigValidationError, match="not a valid timestep of pi05_v1"):
        validate_cache_config(
            _config(JudgeConfig(type="always_warm_start", start_t=0.25))
        )


def test_groot_schedule_makes_its_own_points_legal_and_pi05_points_illegal():
    ok = _config(
        JudgeConfig(type="always_warm_start", start_t=0.25), schedule="groot_n15_k4_v1"
    )
    validate_cache_config(ok)
    bad = _config(
        JudgeConfig(type="always_warm_start", start_t=0.3), schedule="groot_n15_k4_v1"
    )
    with pytest.raises(
        ConfigValidationError, match="not a valid timestep of groot_n15_k4_v1"
    ):
        validate_cache_config(bad)


def test_warm_tiers_are_validated_against_the_named_schedule():
    tiers = [{"threshold": 0.9, "start_t": 0.125}]
    validate_cache_config(
        _config(
            JudgeConfig(type="threshold", warm_tiers=tiers), schedule="groot_n15_k8_v1"
        )
    )
    with pytest.raises(ConfigValidationError, match="groot_n15_k4_v1"):
        validate_cache_config(
            _config(
                JudgeConfig(type="threshold", warm_tiers=tiers),
                schedule="groot_n15_k4_v1",
            )
        )


def test_effective_schedule_resolves_or_raises():
    assert (
        effective_denoise_schedule(_config(JudgeConfig(type="always_hit"))) is PI05_V1
    )
    assert effective_denoise_schedule(
        _config(JudgeConfig(type="always_hit"), schedule="groot_n15_k8_v1")
    ) == groot_n15_schedule(8)
    with pytest.raises(ConfigValidationError):
        effective_denoise_schedule(
            _config(JudgeConfig(type="always_hit"), schedule="nope")
        )


# ---------------------------------------------------------------------------
# judge construction
# ---------------------------------------------------------------------------


def test_always_warm_start_judge_takes_its_valid_set_from_the_schedule():
    AlwaysWarmStartJudge(0.25, schedule=groot_n15_schedule(4))
    with pytest.raises(ValueError, match="pi05_v1"):
        AlwaysWarmStartJudge(0.25)
    built = _build_inner_judge(
        JudgeConfig(type="always_warm_start", start_t=0.75),
        schedule=groot_n15_schedule(4),
    )
    assert isinstance(built, AlwaysWarmStartJudge)


# ---------------------------------------------------------------------------
# required_warm_timesteps
# ---------------------------------------------------------------------------


def test_required_timesteps_enumerate_every_explicit_warm_exit():
    assert required_warm_timesteps(
        _config(JudgeConfig(type="always_warm_start", start_t=0.25))
    ) == {0.25}
    tiers = [{"threshold": 0.9, "start_t": 0.5}, {"threshold": 0.8, "start_t": 0.25}]
    assert required_warm_timesteps(
        _config(JudgeConfig(type="threshold", warm_tiers=tiers))
    ) == {0.5, 0.25}
    composite = JudgeConfig(
        type="composite",
        composer=ComposerConfig(
            type="weighted_sum_with_warm_fallback",
            warm_start_t=0.5,
            warm_fallback_start_t=0.25,
        ),
    )
    assert required_warm_timesteps(_config(composite)) == {0.5, 0.25}
    assert (
        required_warm_timesteps(_config(JudgeConfig(type="always_hit"))) == frozenset()
    )


def test_unenumerable_warm_judge_is_refused_only_under_a_named_schedule():
    router = JudgeConfig(type="mlp_router")
    assert required_warm_timesteps(_config(router)) == frozenset()  # legacy: tolerated
    with pytest.raises(ConfigValidationError, match="cannot be enumerated"):
        required_warm_timesteps(_config(router, schedule="groot_n15_k4_v1"))


# ---------------------------------------------------------------------------
# storage binding
# ---------------------------------------------------------------------------


def test_binding_is_skipped_for_full_hit_only_recipes():
    _check_denoise_schedule_binding(
        _storage({"schedule_id": "groot_n15_k8_v1"}),
        _config(JudgeConfig(type="always_hit")),
    )


def test_binding_refuses_an_unstamped_library_under_a_groot_recipe():
    cfg = _config(
        JudgeConfig(type="always_warm_start", start_t=0.25), schedule="groot_n15_k4_v1"
    )
    with pytest.raises(ConfigValidationError, match="records no denoise schedule"):
        _check_denoise_schedule_binding(_storage({"schedule_id": None}), cfg)


def test_binding_accepts_an_unstamped_library_under_the_legacy_reading():
    cfg = _config(JudgeConfig(type="always_warm_start", start_t=0.5))
    _check_denoise_schedule_binding(_storage({"schedule_id": None}), cfg)


def test_binding_refuses_a_library_from_another_loop():
    """Same geometry, k=4 library, k=8 recipe: the only place this is caught."""
    cfg = _config(
        JudgeConfig(type="always_warm_start", start_t=0.5), schedule="groot_n15_k8_v1"
    )
    with pytest.raises(ConfigValidationError, match="does not match"):
        _check_denoise_schedule_binding(
            _storage({"schedule_id": "groot_n15_k4_v1"}), cfg
        )


def test_binding_refuses_entries_whose_step_count_contradicts_the_stamp():
    cfg = _config(
        JudgeConfig(type="always_warm_start", start_t=0.5), schedule="groot_n15_k4_v1"
    )
    with pytest.raises(ConfigValidationError, match="denoising_num_steps=8"):
        _check_denoise_schedule_binding(
            _storage({"schedule_id": "groot_n15_k4_v1", "denoising_num_steps": 8}), cfg
        )
    _check_denoise_schedule_binding(
        _storage({"schedule_id": "groot_n15_k4_v1", "denoising_num_steps": 4}), cfg
    )


def test_binding_refuses_a_missing_metadata_surface():
    cfg = _config(
        JudgeConfig(type="always_warm_start", start_t=0.5), schedule="groot_n15_k4_v1"
    )
    with pytest.raises(ConfigValidationError, match="no artifact identity"):
        _check_denoise_schedule_binding(_storage(None), cfg)


# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------


def _meta(completeness, *, entries=10, consensus=10):
    return {
        "schedule_id": "groot_n15_k4_v1",
        "entry_count": entries,
        "schema_consensus_count": consensus,
        "intermediates_completeness": completeness,
    }


def test_completeness_requires_every_required_t_on_every_entry():
    cfg = _config(
        JudgeConfig(type="always_warm_start", start_t=0.5), schedule="groot_n15_k4_v1"
    )
    _check_warm_library_completeness(_storage(_meta({"0.5000": 1.0})), cfg)
    with pytest.raises(ConfigValidationError, match="90.0000%"):
        _check_warm_library_completeness(_storage(_meta({"0.5000": 0.9})), cfg)
    with pytest.raises(ConfigValidationError, match="present on 0.0000%"):
        _check_warm_library_completeness(_storage(_meta({"0.2500": 1.0})), cfg)


def test_completeness_refuses_heterogeneous_schema_and_foreign_timesteps():
    cfg = _config(
        JudgeConfig(type="always_warm_start", start_t=0.5), schedule="groot_n15_k4_v1"
    )
    with pytest.raises(ConfigValidationError, match="heterogeneous"):
        _check_warm_library_completeness(
            _storage(_meta({"0.5000": 1.0}, consensus=9)), cfg
        )
    foreign = _config(
        JudgeConfig(type="always_warm_start", start_t=0.3), schedule="groot_n15_k4_v1"
    )
    with pytest.raises(ConfigValidationError, match="not recoverable"):
        _check_warm_library_completeness(_storage(_meta({"0.3000": 1.0})), foreign)


def test_completeness_is_legacy_neutral_and_exempts_full_hit_recipes():
    _check_warm_library_completeness(
        _storage(_meta({})), _config(JudgeConfig(type="always_warm_start", start_t=0.5))
    )
    _check_warm_library_completeness(
        _storage(_meta({})),
        _config(JudgeConfig(type="always_hit"), schedule="groot_n15_k4_v1"),
    )
