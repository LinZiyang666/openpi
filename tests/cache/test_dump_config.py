"""Unit tests for `JudgeConfig.dump` schema, builder, and validator.

Covers:
  - Schema parse: dump block round-trips through `_dict_to_dataclass`
  - Builder: `_build_judge` wraps inner judge in DumpingJudge when dump set
  - Builder: dump factor list constructed via same capability-flag injection
    as composite judge factors (library_stats forwarded to F1a/F1b)
  - Validator: dump.path parent must exist
  - Validator: dump.path required (non-empty)
  - Validator: dump.config_id required (non-empty)
  - Validator: dump.factors[i].type must be registered
  - Validator: capability vs backend compatibility (requires_library_stats /
    requires_chain_walk -> backend.type == 'in_memory')
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openpi.cache.config import (
    BackendConfig,
    CacheConfig,
    DumpConfig,
    FactorConfig,
    JudgeConfig,
    _build_judge,
    _dict_to_dataclass,
    _validate_dump_static,
)
from openpi.cache.components.judge import AlwaysHitJudge, DumpingJudge


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _backend_cfg(backend_type: str = "in_memory") -> CacheConfig:
    cfg = CacheConfig()
    cfg.backend.type = backend_type
    return cfg


def _run_dump_validator(dump: DumpConfig, backend_type: str = "in_memory") -> list[str]:
    """Run only the dump-specific static checks; return collected error messages."""
    errors: list[str] = []
    cfg = _backend_cfg(backend_type)
    _validate_dump_static("checkpoints.cp1", dump, cfg, errors)
    return errors


# ------------------------------------------------------------------
# Schema parse round-trip
# ------------------------------------------------------------------


def test_dump_schema_parses_from_dict():
    raw = {
        "type": "always_hit",
        "dump": {
            "path": "/tmp/foo.jsonl",
            "config_id": "my_cfg",
            "factors": [
                {"type": "f2", "params": {"K": 5}},
            ],
        },
    }
    cfg = _dict_to_dataclass(JudgeConfig, raw)
    assert cfg.dump is not None
    assert cfg.dump.path == "/tmp/foo.jsonl"
    assert cfg.dump.config_id == "my_cfg"
    assert len(cfg.dump.factors) == 1
    assert cfg.dump.factors[0].type == "f2"
    assert cfg.dump.factors[0].params == {"K": 5}


def test_dump_schema_optional_when_absent():
    cfg = _dict_to_dataclass(JudgeConfig, {"type": "always_hit"})
    assert cfg.dump is None


# ------------------------------------------------------------------
# Builder wraps inner judge
# ------------------------------------------------------------------


def test_build_judge_no_dump_returns_naked_inner(tmp_path):
    cfg = JudgeConfig(type="always_hit")
    judge = _build_judge(cfg)
    assert isinstance(judge, AlwaysHitJudge)
    assert not isinstance(judge, DumpingJudge)


def test_build_judge_with_dump_wraps_in_dumping_judge(tmp_path):
    cfg = JudgeConfig(
        type="always_hit",
        dump=DumpConfig(
            path=str(tmp_path / "d.jsonl"),
            config_id="cfg_a",
            factors=[FactorConfig(type="f2", params={"K": 5})],
        ),
    )
    judge = _build_judge(cfg)
    assert isinstance(judge, DumpingJudge)
    # inner is AlwaysHit, accessible via __getattr__ fallback or _inner
    assert isinstance(judge._inner, AlwaysHitJudge)


def test_build_judge_dump_extractor_capability_injection(tmp_path):
    """F1b factor (requires_library_stats=True) must receive library_stats in dump path."""
    from openpi.cache.components.factors.base import LibraryStats
    import torch

    library_stats = LibraryStats(
        action_sigma=torch.ones(32),
        action_active_mask=torch.ones(32, dtype=torch.bool),
        state_sigma=torch.ones(32),
        state_active_mask=torch.ones(32, dtype=torch.bool),
    )
    cfg = JudgeConfig(
        type="always_hit",
        dump=DumpConfig(
            path=str(tmp_path / "d.jsonl"),
            config_id="cfg_a",
            factors=[
                FactorConfig(
                    type="f1b_a",
                    params={
                        "windows": [{"past": 0, "future": 3}],
                        "descriptors": ["jerk", "dir"],
                        "active_eps": 0.01,
                    },
                ),
            ],
        ),
    )
    judge = _build_judge(cfg, library_stats=library_stats)
    assert isinstance(judge, DumpingJudge)
    # The F1b extractor should have library_stats stashed via constructor
    f1b = judge._dump_extractors[0]
    assert f1b._library_stats is library_stats


# ------------------------------------------------------------------
# Validator
# ------------------------------------------------------------------


def test_validator_dump_path_parent_must_exist():
    dump = DumpConfig(path="/no/such/dir/d.jsonl", config_id="cfg_a", factors=[])
    errors = _run_dump_validator(dump)
    assert any("parent directory does not exist" in e for e in errors)


def test_validator_dump_path_required():
    dump = DumpConfig(path="", config_id="cfg_a", factors=[])
    errors = _run_dump_validator(dump)
    assert any("dump.path is required" in e for e in errors)


def test_validator_dump_config_id_required(tmp_path):
    dump = DumpConfig(path=str(tmp_path / "d.jsonl"), config_id="", factors=[])
    errors = _run_dump_validator(dump)
    assert any("dump.config_id is required" in e for e in errors)


def test_validator_dump_factor_must_be_registered(tmp_path):
    dump = DumpConfig(
        path=str(tmp_path / "d.jsonl"),
        config_id="cfg_a",
        factors=[FactorConfig(type="nonexistent_factor", params={})],
    )
    errors = _run_dump_validator(dump)
    assert any("not registered" in e for e in errors)


def test_validator_dump_factor_library_stats_requires_in_memory(tmp_path):
    """F1a/F1b dump factor + non-in_memory backend should raise."""
    dump = DumpConfig(
        path=str(tmp_path / "d.jsonl"),
        config_id="cfg_a",
        factors=[
            FactorConfig(
                type="f1b_a",
                params={
                    "windows": [{"past": 0, "future": 3}],
                    "descriptors": ["jerk"],
                    "active_eps": 0.01,
                },
            ),
        ],
    )
    errors = _run_dump_validator(dump, backend_type="qdrant")
    assert any("requires library_stats" in e for e in errors)


def test_validator_passes_for_well_formed_dump(tmp_path):
    dump = DumpConfig(
        path=str(tmp_path / "d.jsonl"),
        config_id="cfg_a",
        factors=[FactorConfig(type="f2", params={"K": 5})],
    )
    errors = _run_dump_validator(dump)
    assert errors == []


# ------------------------------------------------------------------
# Integration: dump-side top-k widening through build_cache_components
# Locks Phase 0 calibration: AlwaysHit + dump.factors=[f2 K=5] must
# widen the search strategy effective_top_k to 5, not silently 1.
# ------------------------------------------------------------------


def test_build_cache_components_widens_top_k_for_dump_factors(tmp_path):
    from openpi.cache.config import (
        BackendConfig,
        CacheConfig,
        CheckpointConfig,
        SearchStrategyConfig,
        build_cache_components,
        validate_cache_config,
    )
    from openpi.cache.types import CheckpointID

    judge = JudgeConfig(
        type="always_hit",
        dump=DumpConfig(
            path=str(tmp_path / "d.jsonl"),
            config_id="cfg_a",
            factors=[FactorConfig(type="f2", params={"K": 5})],
        ),
    )
    cfg = CacheConfig(
        enabled=True,
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                judge=judge,
                # YAML asks for top_k=1 (the verdict's natural ask); F2 dump
                # extractor must widen this to 5 via the wrapper's hint.
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn", top_k=1),
            ),
        },
    )
    validate_cache_config(cfg)
    components = build_cache_components(cfg)

    # DumpingJudge wrapper exposes min_required_top_k = max(inner=0, F2.K=5) = 5
    judge_built = components["judges"][CheckpointID.CP1]
    assert judge_built.min_required_top_k == 5

    # The strategy itself was built with effective_top_k = max(yaml=1, hint=5) = 5
    strat = components["search_strategies"][CheckpointID.CP1]
    # Implementations stash the effective top_k as `_top_k` (see
    # search_strategy.py); pull it out so we lock the behaviour, not the
    # YAML field, against future regressions where the wrapper still
    # reports K=5 but the strategy silently keeps top_k=1.
    assert strat._top_k == 5


def test_build_cache_components_no_dump_keeps_yaml_top_k(tmp_path):
    """Control: without dump, AlwaysHit (no min_required_top_k attr) → strategy keeps yaml top_k=1."""
    from openpi.cache.config import (
        BackendConfig,
        CacheConfig,
        CheckpointConfig,
        SearchStrategyConfig,
        build_cache_components,
        validate_cache_config,
    )
    from openpi.cache.types import CheckpointID

    cfg = CacheConfig(
        enabled=True,
        backend=BackendConfig(type="in_memory", vector_dims={"robot_state": 32}),
        checkpoints={
            "cp1": CheckpointConfig(
                judge=JudgeConfig(type="always_hit"),
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn", top_k=1),
            ),
        },
    )
    validate_cache_config(cfg)
    components = build_cache_components(cfg)
    strat = components["search_strategies"][CheckpointID.CP1]
    assert strat._top_k == 1
