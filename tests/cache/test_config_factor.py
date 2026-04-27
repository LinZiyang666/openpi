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
# Composite enabled in B1+ (formerly fail-fast in B0)
# ------------------------------------------------------------------


def test_b1_judge_types_includes_composite():
    assert "composite" in _JUDGE_TYPES


def _write(tmp: Path, body: str) -> Path:
    p = tmp / "cache.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_composite_yaml_loads_in_b1(tmp_path):
    # F2-only composite — no library_stats requirement, no chain walk.
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
    # Should not raise
    cfg = load_cache_config(str(p))
    assert cfg.checkpoints["cp1"].judge.type == "composite"


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


# ------------------------------------------------------------------
# B1 composite-specific validator (7 checks)
# ------------------------------------------------------------------


def _minimal_yaml_with_judge(judge_block: str) -> str:
    return textwrap.dedent(f"""
        backend: {{type: in_memory}}
        keys:
          robot_state: {{enabled: true, weight: 1.0}}
        key_builder: {{type: placeholder}}
        checkpoints:
          cp1:
            enabled: true
            judge:
              {judge_block}
            search_strategy: {{type: weighted_rrf_knn}}
        """)


def test_validator_rejects_unknown_factor_type(tmp_path):
    body = _minimal_yaml_with_judge("""type: composite
              factors: [{type: not_a_real_factor, params: {}}]
              composer: {type: weighted_sum, weights: {}, tier_thresholds: {full_hit: 0.5}}
        """)
    p = _write(tmp_path, body)
    with pytest.raises(ConfigValidationError, match="not a registered factor"):
        load_cache_config(str(p))


def test_validator_rejects_library_stats_factor_with_qdrant(tmp_path):
    body = textwrap.dedent("""
        backend:
          type: qdrant
          qdrant: {url: "http://localhost:6333", collection_name: "x"}
          vector_dims: {robot_state: 32}
        keys:
          robot_state: {enabled: true, weight: 1.0}
        key_builder: {type: placeholder}
        checkpoints:
          cp1:
            enabled: true
            judge:
              type: composite
              factors:
                - type: f1a_a
                  params: {window_k: 3, descriptors: [jerk, dir]}
              composer:
                type: weighted_sum
                weights: {f1a_a_jerk: 1.0, f1a_a_dir: 1.0}
                tier_thresholds: {full_hit: 0.5}
            search_strategy: {type: qdrant_weighted_rrf_knn}
        """)
    p = _write(tmp_path, body)
    with pytest.raises(ConfigValidationError, match="requires backend.type='in_memory'"):
        load_cache_config(str(p))


def test_validator_rejects_chain_walk_factor_with_qdrant(tmp_path):
    body = textwrap.dedent("""
        backend:
          type: qdrant
          qdrant: {url: "http://localhost:6333", collection_name: "x"}
          vector_dims: {robot_state: 32}
        keys:
          robot_state: {enabled: true, weight: 1.0}
        key_builder: {type: placeholder}
        checkpoints:
          cp1:
            enabled: true
            judge:
              type: composite
              factors:
                - type: f1a_t
                  params: {window_k: 3, descriptors: [jerk]}
              composer:
                type: weighted_sum
                weights: {f1a_t_jerk: 1.0}
                tier_thresholds: {full_hit: 0.5}
            search_strategy: {type: qdrant_weighted_rrf_knn}
        """)
    p = _write(tmp_path, body)
    with pytest.raises(ConfigValidationError, match="requires backend.type='in_memory'"):
        load_cache_config(str(p))


def test_validator_warm_start_below_full_hit(tmp_path):
    body = _minimal_yaml_with_judge("""type: composite
              factors: [{type: f2, params: {K: 3}}]
              composer:
                type: weighted_sum
                weights: {f2_var: 1.0}
                tier_thresholds: {full_hit: 0.5, warm_start: 0.7}
        """)
    p = _write(tmp_path, body)
    with pytest.raises(ConfigValidationError, match="strictly less than"):
        load_cache_config(str(p))


def test_validator_non_monotonic_directions_required(tmp_path):
    # f1b_a with curv_radius (non_monotonic) under non-zero weight → must
    # appear in directions or validator rejects the YAML.
    body = _minimal_yaml_with_judge("""type: composite
              factors:
                - type: f1b_a
                  params:
                    windows: [{past: 0, future: 5}]
                    descriptors: [curv_radius]
                    active_eps: 0.01
              composer:
                type: weighted_sum
                weights: {f1b_a_curv_radius__p0_f5: 1.0}
                tier_thresholds: {full_hit: 0.5}
        """)
    p = _write(tmp_path, body)
    with pytest.raises(ConfigValidationError, match="missing a direction"):
        load_cache_config(str(p))


def test_validator_full_hit_threshold_required(tmp_path):
    body = _minimal_yaml_with_judge("""type: composite
              factors: [{type: f2, params: {K: 3}}]
              composer:
                type: weighted_sum
                weights: {f2_var: 1.0}
                tier_thresholds: {warm_start: 0.3}
        """)
    p = _write(tmp_path, body)
    with pytest.raises(ConfigValidationError, match="full_hit"):
        load_cache_config(str(p))


# ------------------------------------------------------------------
# B2 collect_offline_writers_from_judges
# ------------------------------------------------------------------


def test_collect_offline_writers_picks_writer_capable_extractors():
    """collect_offline_writers walks per-CP composite judges, pulling
    extractors that expose `compute_for_episode` (the OfflineWriter
    capability). F1a/F2 (online-only) are skipped; F1b is included."""
    from openpi.cache.components.factors.consensus import TopKActionConsensus
    from openpi.cache.components.factors.runtime_continuity import (
        RuntimeContinuityAction,
    )
    from openpi.cache.components.factors.source_window import (
        SourceWindowSmoothnessAction,
    )
    from openpi.cache.components.judge import CompositeJudge
    from openpi.cache.components.factors.composers import WeightedSumComposer
    from openpi.cache.config import _collect_offline_writers_from_judges
    from openpi.cache.types import CheckpointID
    import torch as torch_

    ls = type(
        "FakeLS", (),
        {
            "action_sigma": torch_.ones(2),
            "action_active_mask": torch_.ones(2, dtype=torch_.bool),
            "state_sigma": torch_.ones(2),
            "state_active_mask": torch_.ones(2, dtype=torch_.bool),
        },
    )()

    f1a = RuntimeContinuityAction(
        window_k=2, descriptors=["jerk"], library_stats=ls,
    )
    f1b = SourceWindowSmoothnessAction(
        windows=[(0, 2)], descriptors=["jerk"], active_eps=0.01, library_stats=ls,
    )
    f2 = TopKActionConsensus(K=3)

    composer = WeightedSumComposer(
        weights={"f1a_a_jerk": 1.0, "f1b_a_jerk__p0_f2": 1.0, "f2_var": 1.0},
        full_hit_threshold=0.5,
    )
    judge = CompositeJudge(extractors=[f1a, f1b, f2], composer=composer)

    out = _collect_offline_writers_from_judges({CheckpointID.CP1: judge})

    # Only f1b has compute_for_episode
    assert out == [f1b]


def test_collect_offline_writers_dedups_across_checkpoints():
    """Same writer instance referenced from CP1 and CP3 should appear
    only once (id() de-dup)."""
    from openpi.cache.components.factors.composers import WeightedSumComposer
    from openpi.cache.components.factors.source_window import (
        SourceWindowSmoothnessAction,
    )
    from openpi.cache.components.judge import CompositeJudge
    from openpi.cache.config import _collect_offline_writers_from_judges
    from openpi.cache.types import CheckpointID
    import torch as torch_

    ls = type(
        "FakeLS", (),
        {
            "action_sigma": torch_.ones(2),
            "action_active_mask": torch_.ones(2, dtype=torch_.bool),
            "state_sigma": torch_.ones(2),
            "state_active_mask": torch_.ones(2, dtype=torch_.bool),
        },
    )()
    shared = SourceWindowSmoothnessAction(
        windows=[(0, 2)], descriptors=["jerk"], active_eps=0.01, library_stats=ls,
    )
    composer = WeightedSumComposer(
        weights={"f1b_a_jerk__p0_f2": 1.0}, full_hit_threshold=0.5,
    )
    j1 = CompositeJudge(extractors=[shared], composer=composer)
    composer2 = WeightedSumComposer(
        weights={"f1b_a_jerk__p0_f2": 1.0}, full_hit_threshold=0.5,
    )
    j2 = CompositeJudge(extractors=[shared], composer=composer2)

    out = _collect_offline_writers_from_judges({
        CheckpointID.CP1: j1, CheckpointID.CP3: j2,
    })

    assert len(out) == 1
    assert out[0] is shared


def test_collect_offline_writers_empty_judges_returns_empty():
    from openpi.cache.config import _collect_offline_writers_from_judges
    assert _collect_offline_writers_from_judges({}) == []


def test_collect_offline_writers_skips_non_composite_judges():
    """Threshold / always_hit judges have no `_extractors` attribute —
    helper must skip them gracefully without raising."""
    from openpi.cache.components.judge import ThresholdJudge
    from openpi.cache.config import _collect_offline_writers_from_judges
    from openpi.cache.types import CheckpointID

    out = _collect_offline_writers_from_judges({
        CheckpointID.CP1: ThresholdJudge(cp1_threshold=0.5),
    })
    assert out == []


# ------------------------------------------------------------------
# Composite warm-start validator (5a-5d)
# ------------------------------------------------------------------


def _composite_warm_start_yaml(
    *,
    cp: str = "cp1",
    tier_warm: float | None = 0.4,
    warm_start_t: float | None = 0.3,
) -> str:
    tier_block = "{full_hit: 0.8"
    if tier_warm is not None:
        tier_block += f", warm_start: {tier_warm}"
    tier_block += "}"
    warm_t_line = f"\n                warm_start_t: {warm_start_t}" if warm_start_t is not None else ""
    return textwrap.dedent(f"""
        backend: {{type: in_memory}}
        keys:
          robot_state: {{enabled: true, weight: 1.0}}
        key_builder: {{type: placeholder}}
        checkpoints:
          {cp}:
            enabled: true
            judge:
              type: composite
              factors: [{{type: f2, params: {{K: 3}}}}]
              composer:
                type: weighted_sum
                weights: {{f2_var: 1.0}}
                tier_thresholds: {tier_block}{warm_t_line}
            search_strategy: {{type: weighted_rrf_knn}}
        """)


def test_validator_5a_warm_start_t_must_be_canonical(tmp_path):
    body = _composite_warm_start_yaml(tier_warm=0.4, warm_start_t=0.55)
    p = _write(tmp_path, body)
    with pytest.raises(ConfigValidationError, match="canonical denoise timestep"):
        load_cache_config(str(p))


def test_validator_5a_warm_start_t_canonical_value_accepted(tmp_path):
    body = _composite_warm_start_yaml(tier_warm=0.4, warm_start_t=0.3)
    p = _write(tmp_path, body)
    cfg = load_cache_config(str(p))                    # no raise
    # Normalized writeback (no float drift)
    assert cfg.checkpoints["cp1"].judge.composer.warm_start_t == 0.3


def test_validator_5b_pairwise_tier_warm_without_warm_start_t(tmp_path):
    body = _composite_warm_start_yaml(tier_warm=0.4, warm_start_t=None)
    p = _write(tmp_path, body)
    with pytest.raises(ConfigValidationError, match="warm_start_t is missing"):
        load_cache_config(str(p))


def test_validator_5b_pairwise_warm_start_t_without_tier_warm(tmp_path):
    body = _composite_warm_start_yaml(tier_warm=None, warm_start_t=0.3)
    p = _write(tmp_path, body)
    with pytest.raises(ConfigValidationError, match="tier_thresholds.warm_start is missing"):
        load_cache_config(str(p))


def test_validator_5c_warm_start_only_on_cp1(tmp_path):
    body = _composite_warm_start_yaml(cp="cp3", tier_warm=0.4, warm_start_t=0.3)
    p = _write(tmp_path, body)
    with pytest.raises(ConfigValidationError, match="only supported on CP1"):
        load_cache_config(str(p))


# ------------------------------------------------------------------
# B2 production assembly: build_per_connection_components feeds
# offline_writers + library_stats through to CacheOrchestrator
# ------------------------------------------------------------------


def test_production_assembly_passes_offline_writers_and_library_stats(tmp_path):
    """Smoke check that build_per_connection_components dict carries the
    new B2 keys (`offline_writers`, `library_stats`) so production
    assembly (scripts/serve_policy.py) can forward them to the
    Orchestrator. Without this, episode-end factor enrichment would
    silently no-op even though composite F1b is configured."""
    import pickle
    import torch as torch_

    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.components.factors.base import LibraryStats
    from openpi.cache.config import build_per_connection_components, load_cache_config
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID as CP

    # Build a tiny artifact pkl so the in_memory backend can preload it
    # (so library_stats is populated).
    entries = [
        CacheEntry(
            id=f"e{t}", checkpoint_id=CP.CP1,
            query_keys={"robot_state": torch_.tensor([float(t), 0.0])},
            payload=CachePayload(action_chunk=torch_.tensor([[float(t), 0.0]])),
            step_idx=t, trajectory_id="traj-0",
        )
        for t in range(3)
    ]
    pre_built = LibraryStats(
        action_sigma=torch_.ones(2),
        action_active_mask=torch_.ones(2, dtype=torch_.bool),
        state_sigma=torch_.ones(2),
        state_active_mask=torch_.ones(2, dtype=torch_.bool),
    )
    artifact = {
        "key_builder_type": "placeholder",
        "checkpoint_id": "CP1",
        "vector_dims": {"robot_state": 2},
        "entries": entries,
        "library_stats": pre_built,
    }
    artifact_path = tmp_path / "art.pkl"
    artifact_path.write_bytes(pickle.dumps(artifact))

    yaml_body = textwrap.dedent(f"""
        backend:
          type: in_memory
          vector_dims: {{robot_state: 2}}
          in_memory: {{preload_path: {artifact_path}}}
        keys:
          robot_state: {{enabled: true, weight: 1.0}}
        key_builder: {{type: placeholder}}
        checkpoints:
          cp1:
            enabled: true
            judge:
              type: composite
              factors:
                - type: f1b_a
                  params:
                    windows: [{{past: 1, future: 1}}]
                    descriptors: [jerk]
                    active_eps: 0.01
              composer:
                type: weighted_sum
                weights: {{f1b_a_jerk__p1_f1: 1.0}}
                tier_thresholds: {{full_hit: 0.5}}
            search_strategy: {{type: weighted_rrf_knn}}
        """)
    yml = tmp_path / "cache.yaml"
    yml.write_text(yaml_body)

    cfg = load_cache_config(str(yml))
    backend = InMemoryBackend({"robot_state": 2})
    backend.load_artifact(str(artifact_path))
    storage = CacheStorage(backend)
    components = build_per_connection_components(cfg, storage)

    # The B2 keys must be present and populated.
    assert "offline_writers" in components
    assert "library_stats" in components
    # F1b-A is OfflineWriter-capable and must show up
    assert len(components["offline_writers"]) == 1
    assert hasattr(components["offline_writers"][0], "compute_for_episode")
    # library_stats came from the in-memory backend's loaded artifact
    assert components["library_stats"] is not None
