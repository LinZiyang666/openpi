"""Tests for the WarmupPool → per-connection normalizer preload path
(verdict_factor_judge B2.3).

The integration these tests pin:

  ``WebsocketPolicyServer`` writes ``CurrentCacheBundle.yaml_id`` from the
  ``load_cache_config`` ctrl payload; ``serve_policy.py`` then forwards it
  into ``build_per_connection_components(..., yaml_id=...)`` whenever a new
  worker connection wraps a bundle. ``_preload_normalizer_from_warmup_pool``
  sees the matching ``WarmupPool`` entry, drills past any ``DumpingJudge``
  wrapper, and pre-fills each composite judge's ``PercentileRollingNormalizer``
  so eval-side verdicts skip the cold-start sentinel.

A per-connection bundle without a ``yaml_id`` (legacy path, default-None
field) MUST NOT preload anything — that is the backward-compat invariant
for non-warmup yaml runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openpi.cache import warmup_pool as _warmup_pool_mod
from openpi.cache.components.factors.normalizers import PercentileRollingNormalizer
from openpi.cache.components.judge import CompositeJudge, DumpingJudge
from openpi.cache.config import (
    BackendConfig,
    CacheConfig,
    CheckpointConfig,
    ComposerConfig,
    DumpConfig,
    FactorConfig,
    GateConfig,
    JudgeConfig,
    KeyFieldConfig,
    KeysConfig,
    NormalizerConfig,
    SearchStrategyConfig,
    _preload_normalizer_from_warmup_pool,
    build_per_connection_components,
    build_shared_storage,
    validate_cache_config,
)
from openpi.cache.types import CheckpointID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_pool(monkeypatch):
    """Each test runs against a fresh WarmupPool so order cannot bleed."""
    fresh = _warmup_pool_mod.WarmupPool()
    monkeypatch.setattr(_warmup_pool_mod, "_GLOBAL", fresh)
    yield fresh


def _composite_config(*, with_dump: bool = False) -> CacheConfig:
    """Minimal in-memory composite judge config.

    F2 is used because ``requires_library_stats=False`` so the per-connection
    builder does not need a preloaded artifact for the test to construct
    the judge.
    """
    judge = JudgeConfig(
        type="composite",
        factors=[FactorConfig(type="f2", params={"K": 4})],
        composer=ComposerConfig(
            type="weighted_sum",
            weights={"f2_var": 1.0},
            tier_thresholds={"full_hit": 0.5},
        ),
        normalizer=NormalizerConfig(type="percentile_rolling", window_size=20),
    )
    if with_dump:
        judge.dump = DumpConfig(
            path="/tmp/__test_dump.jsonl",
            config_id="test_dump",
            factors=[FactorConfig(type="f2", params={"K": 4})],
        )
    return CacheConfig(
        enabled=True,
        keys=KeysConfig(robot_state=KeyFieldConfig(enabled=True, weight=1.0)),
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_search"),
                judge=judge,
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn"),
            ),
        },
    )


def _make_judge_with_normalizer(buffers_keys: list[str]) -> SimpleNamespace:
    """Tiny stand-in that mimics CompositeJudge.{_normalizer} surface."""
    norm = PercentileRollingNormalizer(window_size=10)
    norm.bind_keys(buffers_keys)
    judge = SimpleNamespace(_normalizer=norm)
    return judge


# ---------------------------------------------------------------------------
# Helper-level (faster, isolates the WarmupPool wiring)
# ---------------------------------------------------------------------------


def test_preload_helper_pre_fills_naked_composite_judge() -> None:
    judge = _make_judge_with_normalizer(["f2_var"])
    _warmup_pool_mod.get_global_pool().set("eval_yaml_x", {"f2_var": [0.1, 0.2, 0.3]})

    n_preloaded = _preload_normalizer_from_warmup_pool({CheckpointID.CP1: judge}, "eval_yaml_x")
    assert n_preloaded == 1
    assert list(judge._normalizer._buffers["f2_var"]) == [0.1, 0.2, 0.3]


def test_preload_helper_drills_past_dumping_judge_wrapper() -> None:
    """If the Composite judge is wrapped by a DumpingJudge (warmup yaml),
    the helper must follow ``_inner`` to find the normalizer."""
    inner = _make_judge_with_normalizer(["f2_var"])
    wrapper = SimpleNamespace(_inner=inner)
    _warmup_pool_mod.get_global_pool().set("eval_yaml_x", {"f2_var": [0.5]})

    n_preloaded = _preload_normalizer_from_warmup_pool({CheckpointID.CP1: wrapper}, "eval_yaml_x")
    assert n_preloaded == 1
    assert list(inner._normalizer._buffers["f2_var"]) == [0.5]


def test_preload_helper_filters_unbound_keys() -> None:
    """A key the normalizer never bound (schema drift between yamls) MUST
    be silently dropped — preloading would raise KeyError otherwise."""
    judge = _make_judge_with_normalizer(["f2_var"])
    _warmup_pool_mod.get_global_pool().set(
        "eval_yaml_x", {"f2_var": [0.1], "stale_key_from_old_yaml": [99.0]},
    )
    _preload_normalizer_from_warmup_pool({CheckpointID.CP1: judge}, "eval_yaml_x")
    assert list(judge._normalizer._buffers["f2_var"]) == [0.1]
    assert "stale_key_from_old_yaml" not in judge._normalizer._buffers


def test_preload_helper_noop_when_pool_has_no_entry() -> None:
    judge = _make_judge_with_normalizer(["f2_var"])
    n_preloaded = _preload_normalizer_from_warmup_pool({CheckpointID.CP1: judge}, "missing_yaml")
    assert n_preloaded == 0
    assert list(judge._normalizer._buffers["f2_var"]) == []


def test_preload_helper_noop_when_judge_lacks_normalizer() -> None:
    """ThresholdJudge / AlwaysHitJudge etc. have no _normalizer attribute —
    the helper must skip them silently (so legacy yamls keep working)."""
    judge = MagicMock(spec=[])  # nothing exposed
    _warmup_pool_mod.get_global_pool().set("eval_yaml_x", {"f2_var": [0.1]})
    n_preloaded = _preload_normalizer_from_warmup_pool({CheckpointID.CP1: judge}, "eval_yaml_x")
    assert n_preloaded == 0


# ---------------------------------------------------------------------------
# End-to-end through build_per_connection_components
# ---------------------------------------------------------------------------


def test_build_per_connection_components_preloads_composite_normalizer() -> None:
    config = _composite_config()
    validate_cache_config(config)
    shared = build_shared_storage(config)

    pool_buf = {"f2_var": [round(0.05 * i, 4) for i in range(15)]}
    _warmup_pool_mod.get_global_pool().set("eval_x", pool_buf)

    components = build_per_connection_components(
        config, shared, yaml_id="eval_x", quiet=True,
    )
    judge = components["judges"][CheckpointID.CP1]
    assert isinstance(judge, CompositeJudge)
    assert list(judge._normalizer._buffers["f2_var"]) == pool_buf["f2_var"]


def test_build_per_connection_components_no_yaml_id_skips_preload() -> None:
    """Backward compat: legacy callers that do not pass yaml_id keep the
    cold-start sentinel behaviour."""
    config = _composite_config()
    validate_cache_config(config)
    shared = build_shared_storage(config)

    _warmup_pool_mod.get_global_pool().set("eval_x", {"f2_var": [0.1, 0.2]})

    components = build_per_connection_components(config, shared, quiet=True)
    judge = components["judges"][CheckpointID.CP1]
    # Pool entry exists but yaml_id is None → no preload.
    assert list(judge._normalizer._buffers["f2_var"]) == []


def test_build_per_connection_components_unknown_yaml_id_is_noop() -> None:
    """yaml_id passed but no matching pool entry → no error, no preload."""
    config = _composite_config()
    validate_cache_config(config)
    shared = build_shared_storage(config)

    components = build_per_connection_components(
        config, shared, yaml_id="never_loaded", quiet=True,
    )
    judge = components["judges"][CheckpointID.CP1]
    assert list(judge._normalizer._buffers["f2_var"]) == []
