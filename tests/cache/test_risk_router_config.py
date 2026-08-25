"""X15 U5 — yaml-load gates for the ``risk_router`` judge.

The judge depends on capabilities only one backend/strategy pair provides, and
on parameters that have no safe default. Both classes of error are cheap to
catch at load and expensive to catch at hour three of a run, so they are
validation failures rather than runtime surprises.

Key dependency: ``openpi.cache.config.validate_cache_config``.
"""

from __future__ import annotations

import pytest

from openpi.cache.config import (
    BackendConfig,
    CacheConfig,
    CheckpointConfig,
    ConfigValidationError,
    JudgeConfig,
    KeyBuilderConfig,
    ScoreNormalizationConfig,
    SearchStrategyConfig,
    validate_cache_config,
)


def _strategy(**overrides) -> SearchStrategyConfig:
    base = dict(
        type="weighted_score_sum_knn",
        top_k=5,
        score_normalization=ScoreNormalizationConfig(
            type="per_field",
            fields={"robot_state": {"method": "zscore", "params": {"mu": 0.0, "sigma": 1.0}}},
        ),
    )
    base.update(overrides)
    return SearchStrategyConfig(**base)


def _judge(**overrides) -> JudgeConfig:
    base = dict(
        type="risk_router",
        risk_model_path="/tmp/risk.pt",
        tau=0.5,
        dwell=1,
        task_index=0,
        replan_steps=5,
        library_replan_steps=5,
    )
    base.update(overrides)
    return JudgeConfig(**base)


def _config(*, judge: JudgeConfig | None = None, strategy=None, backend=None) -> CacheConfig:
    return CacheConfig(
        backend=backend
        or BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        key_builder=KeyBuilderConfig(type="placeholder"),
        checkpoints={
            "cp1": CheckpointConfig(
                judge=judge or _judge(),
                search_strategy=strategy or _strategy(),
            )
        },
    )


def test_valid_config_passes() -> None:
    validate_cache_config(_config())


@pytest.mark.parametrize(
    "missing",
    ["risk_model_path", "tau", "task_index", "replan_steps", "library_replan_steps"],
)
def test_required_fields_have_no_silent_default(missing) -> None:
    with pytest.raises(ConfigValidationError, match=missing):
        validate_cache_config(_config(judge=_judge(**{missing: None})))


def test_top_k_below_the_feature_width_is_refused() -> None:
    """The top-k score features cannot be built from a top-1 search."""
    with pytest.raises(ConfigValidationError, match="top_k"):
        validate_cache_config(
            _config(strategy=_strategy(top_k=1))
        )


def test_non_weighted_strategy_is_refused() -> None:
    """Only weighted-score-sum fusion emits the per-field diagnostics."""
    with pytest.raises(ConfigValidationError, match="weighted_score_sum_knn"):
        validate_cache_config(
            _config(strategy=_strategy(type="weighted_rrf_knn"))
        )


def test_dwell_below_one_is_refused_at_load() -> None:
    with pytest.raises(ConfigValidationError, match="dwell"):
        validate_cache_config(_config(judge=_judge(dwell=0)))


def test_risk_router_is_not_a_routing_judge() -> None:
    """teacher/cache needs no sidecar routing, so the type stays out of the
    routing allowlist rather than widening the config surface."""
    from openpi.cache.config import _JUDGE_TYPES, _ROUTING_JUDGE_TYPES

    assert "risk_router" in _JUDGE_TYPES
    assert "risk_router" not in _ROUTING_JUDGE_TYPES
