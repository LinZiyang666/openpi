"""Tests for SearchStrategy (SearchContext, QdrantWeightedRrfKnnStrategy, TrajectoryMixin)."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import (
    ConstantDepthPolicy,
    DepthFeatures,
    DynamicDepthKnnStrategy,
    HeuristicDepthPolicy,
    QdrantWeightedRrfKnnStrategy,
    SearchContext,
    SearchStrategy,
    TrajectoryMixin,
    WeightedRrfKnnStrategy,
    WeightedScoreSumKnnStrategy,
)
from openpi.cache.storage_types import CachePayload, QuerySpec, SearchResultLite
from openpi.cache.types import ROBOT_STATE, CheckpointID
from tests.cache.conftest import insert_entry


# ---------------------------------------------------------------------------
# test_qdrant_weighted_rrf_knn_basic_search: delegates to storage.search()
# ---------------------------------------------------------------------------


def test_qdrant_weighted_rrf_knn_basic_search():
    storage = MagicMock(spec=CacheStorage)
    storage.search.return_value = [
        SearchResultLite(id="abc", score=0.123, checkpoint_id=CheckpointID.CP1)
    ]

    strategy = QdrantWeightedRrfKnnStrategy(storage, top_k=1, step_filter="all")
    ctx = SearchContext(
        query_keys={ROBOT_STATE: torch.randn(32)},
        checkpoint_id=CheckpointID.CP1,
        current_step=0,
    )
    results = strategy.search(ctx)
    assert len(results) == 1
    assert isinstance(results[0], SearchResultLite)
    storage.search.assert_called_once()


# ---------------------------------------------------------------------------
# test_query_spec_fusion_params: fusion_weights + backend_hints in QuerySpec
# ---------------------------------------------------------------------------


def test_query_spec_fusion_params():
    """Verify fusion_weights and backend_hints are correctly passed to storage.search()."""
    storage = MagicMock(spec=CacheStorage)
    storage.search.return_value = []

    weights = {"robot_state": 1.0, "vision_0": 0.5}
    strategy = QdrantWeightedRrfKnnStrategy(
        storage,
        top_k=3,
        rrf_k=42,
        fusion_weights=weights,
        candidate_multiplier=10,
    )
    ctx = SearchContext(
        query_keys={ROBOT_STATE: torch.randn(32)},
        checkpoint_id=CheckpointID.CP1,
    )
    strategy.search(ctx)

    storage.search.assert_called_once()
    spec: QuerySpec = storage.search.call_args[0][0]
    assert spec.top_k == 3
    assert spec.fusion_weights == weights
    assert spec.backend_hints["rrf_k"] == 42
    assert spec.backend_hints["candidate_multiplier"] == 10


# ---------------------------------------------------------------------------
# test_step_filter_all: no filter added
# ---------------------------------------------------------------------------


def test_step_filter_all():
    storage = MagicMock(spec=CacheStorage)
    storage.search.return_value = []

    strategy = QdrantWeightedRrfKnnStrategy(storage, step_filter="all")
    ctx = SearchContext(
        query_keys={ROBOT_STATE: torch.randn(32)},
        checkpoint_id=CheckpointID.CP1,
        current_step=5,
    )
    strategy.search(ctx)

    spec: QuerySpec = storage.search.call_args[0][0]
    assert spec.filters is None


# ---------------------------------------------------------------------------
# test_step_filter_exact: mock storage, assert step_range==(step, step)
# ---------------------------------------------------------------------------


def test_step_filter_exact():
    storage = MagicMock(spec=CacheStorage)
    storage.search.return_value = []

    strategy = QdrantWeightedRrfKnnStrategy(storage, step_filter="exact")
    ctx = SearchContext(
        query_keys={ROBOT_STATE: torch.randn(32)},
        checkpoint_id=CheckpointID.CP1,
        current_step=7,
    )
    strategy.search(ctx)

    spec: QuerySpec = storage.search.call_args[0][0]
    assert spec.filters is not None
    assert spec.filters.step_range == (7, 7)


# ---------------------------------------------------------------------------
# test_step_filter_window: mock storage, assert step_range correct
# ---------------------------------------------------------------------------


def test_step_filter_window():
    storage = MagicMock(spec=CacheStorage)
    storage.search.return_value = []

    strategy = QdrantWeightedRrfKnnStrategy(storage, step_filter="window", step_window=3)
    ctx = SearchContext(
        query_keys={ROBOT_STATE: torch.randn(32)},
        checkpoint_id=CheckpointID.CP1,
        current_step=10,
    )
    strategy.search(ctx)

    spec: QuerySpec = storage.search.call_args[0][0]
    assert spec.filters is not None
    assert spec.filters.step_range == (7, 13)


def test_step_filter_window_clamps_lower_bound():
    """Window mode clamps lower bound to 0."""
    storage = MagicMock(spec=CacheStorage)
    storage.search.return_value = []

    strategy = QdrantWeightedRrfKnnStrategy(storage, step_filter="window", step_window=5)
    ctx = SearchContext(
        query_keys={ROBOT_STATE: torch.randn(32)},
        checkpoint_id=CheckpointID.CP1,
        current_step=2,
    )
    strategy.search(ctx)

    spec: QuerySpec = storage.search.call_args[0][0]
    assert spec.filters.step_range == (0, 7)


# ---------------------------------------------------------------------------
# test_search_context_fields: all fields correctly set
# ---------------------------------------------------------------------------


def test_search_context_fields():
    keys = {ROBOT_STATE: torch.randn(32)}
    ctx = SearchContext(
        query_keys=keys,
        checkpoint_id=CheckpointID.CP3,
        current_step=42,
        task_key="my_task",
    )
    assert ctx.query_keys is keys
    assert ctx.checkpoint_id == CheckpointID.CP3
    assert ctx.current_step == 42
    assert ctx.task_key == "my_task"


# ---------------------------------------------------------------------------
# test_protocol_compliance: QdrantWeightedRrfKnnStrategy satisfies SearchStrategy Protocol
# ---------------------------------------------------------------------------


def test_protocol_compliance():
    storage = MagicMock(spec=CacheStorage)
    strategy = QdrantWeightedRrfKnnStrategy(storage)
    assert isinstance(strategy, SearchStrategy)


# ---------------------------------------------------------------------------
# TrajectoryMixin tests
# ---------------------------------------------------------------------------


class TestTrajectoryMixin:
    """Tests for the TrajectoryMixin shared history buffer logic."""

    def test_record_query_keys_builds_history(self):
        storage = MagicMock(spec=CacheStorage)
        storage.search.return_value = []
        strategy = QdrantWeightedRrfKnnStrategy(
            storage, trajectory_depth=3, trajectory_weights=[0.5, 0.3, 0.2],
        )

        keys1 = {ROBOT_STATE: torch.randn(32)}
        keys2 = {ROBOT_STATE: torch.randn(32)}
        strategy.record_query_keys(keys1)
        strategy.record_query_keys(keys2)

        assert len(strategy._query_history) == 2
        assert strategy._query_history[0] is keys1
        assert strategy._query_history[1] is keys2

    def test_on_episode_start_clears_history(self):
        storage = MagicMock(spec=CacheStorage)
        storage.search.return_value = []
        strategy = QdrantWeightedRrfKnnStrategy(
            storage, trajectory_depth=3, trajectory_weights=[0.5, 0.3, 0.2],
        )

        strategy.record_query_keys({ROBOT_STATE: torch.randn(32)})
        strategy.record_action(torch.randn(50, 32))
        assert len(strategy._query_history) == 1
        assert len(strategy._action_history) == 1

        strategy.on_episode_start()
        assert len(strategy._query_history) == 0
        assert len(strategy._action_history) == 0

    def test_build_trajectory_fields_depth_1_returns_empty(self):
        storage = MagicMock(spec=CacheStorage)
        strategy = QdrantWeightedRrfKnnStrategy(
            storage, trajectory_depth=1,
        )
        strategy.record_query_keys({ROBOT_STATE: torch.randn(32)})
        assert strategy._build_trajectory_fields() == {}

    def test_build_trajectory_fields_insufficient_history(self):
        storage = MagicMock(spec=CacheStorage)
        strategy = QdrantWeightedRrfKnnStrategy(
            storage, trajectory_depth=3, trajectory_weights=[0.5, 0.3, 0.2],
        )
        # Only 1 entry in history, need at least 2 for trajectory
        strategy.record_query_keys({ROBOT_STATE: torch.randn(32)})
        assert strategy._build_trajectory_fields() == {}

    def test_build_trajectory_fields_sufficient_history(self):
        storage = MagicMock(spec=CacheStorage)
        strategy = QdrantWeightedRrfKnnStrategy(
            storage, trajectory_depth=3, trajectory_weights=[0.5, 0.3, 0.2],
        )
        for _ in range(3):
            strategy.record_query_keys({ROBOT_STATE: torch.randn(32)})

        fields = strategy._build_trajectory_fields()
        assert "trajectory_history" in fields
        assert "trajectory_weights" in fields
        assert len(fields["trajectory_history"]) == 3
        assert len(fields["trajectory_weights"]) == 3
        # Newest first
        assert fields["trajectory_weights"] == [0.5, 0.3, 0.2]

    def test_build_trajectory_fields_partial_history(self):
        """depth=3 but only 2 history entries → returns 2-level fields."""
        storage = MagicMock(spec=CacheStorage)
        strategy = QdrantWeightedRrfKnnStrategy(
            storage, trajectory_depth=3, trajectory_weights=[0.5, 0.3, 0.2],
        )
        for _ in range(2):
            strategy.record_query_keys({ROBOT_STATE: torch.randn(32)})

        fields = strategy._build_trajectory_fields()
        assert len(fields["trajectory_history"]) == 2
        assert len(fields["trajectory_weights"]) == 2
        assert fields["trajectory_weights"] == [0.5, 0.3]

    def test_search_records_query_keys_automatically(self):
        """search() should call record_query_keys internally."""
        storage = MagicMock(spec=CacheStorage)
        storage.search.return_value = []
        strategy = QdrantWeightedRrfKnnStrategy(
            storage, trajectory_depth=2, trajectory_weights=[0.6, 0.4],
        )

        ctx = SearchContext(
            query_keys={ROBOT_STATE: torch.randn(32)},
            checkpoint_id=CheckpointID.CP1,
        )
        strategy.search(ctx)
        assert len(strategy._query_history) == 1

    def test_trajectory_fields_in_query_spec(self):
        """When history is sufficient, QuerySpec should contain trajectory fields."""
        storage = MagicMock(spec=CacheStorage)
        storage.search.return_value = []
        strategy = QdrantWeightedRrfKnnStrategy(
            storage, trajectory_depth=2, trajectory_weights=[0.6, 0.4],
        )

        # First search: only 1 entry, no trajectory
        ctx1 = SearchContext(
            query_keys={ROBOT_STATE: torch.randn(32)},
            checkpoint_id=CheckpointID.CP1,
        )
        strategy.search(ctx1)
        spec1: QuerySpec = storage.search.call_args[0][0]
        assert spec1.trajectory_history is None

        # Second search: 2 entries, trajectory should be present
        ctx2 = SearchContext(
            query_keys={ROBOT_STATE: torch.randn(32)},
            checkpoint_id=CheckpointID.CP1,
        )
        strategy.search(ctx2)
        spec2: QuerySpec = storage.search.call_args[0][0]
        assert spec2.trajectory_history is not None
        assert len(spec2.trajectory_history) == 2
        assert spec2.trajectory_weights == [0.6, 0.4]


# ---------------------------------------------------------------------------
# Dynamic chain-depth strategy (TRACER Phase 1 / M3)
# ---------------------------------------------------------------------------

_DIM = 32


class TestHeuristicDepthPolicy:
    """HeuristicDepthPolicy deterministic smoothness bucketing."""

    def _pol(self) -> HeuristicDepthPolicy:
        return HeuristicDepthPolicy(
            allowed_depths=[1, 3, 5], smoothness_thresholds=[0.5, 1.5], fallback_depth=1
        )

    def test_none_smoothness_returns_fallback(self):
        pol = self._pol()
        assert pol.select(DepthFeatures(current_step=0, history_len=0, action_smoothness=None)) == 1

    def test_smooth_selects_deepest(self):
        # s = 0.2 < 0.5 -> bucket 0 -> deepest depth 5
        assert self._pol().select(DepthFeatures(0, 5, 0.2)) == 5

    def test_mid_selects_middle(self):
        # s = 1.0 in [0.5, 1.5) -> bucket 1 -> depth 3
        assert self._pol().select(DepthFeatures(0, 5, 1.0)) == 3

    def test_abrupt_selects_shallowest(self):
        # s = 2.0 >= 1.5 -> bucket 2 -> depth 1
        assert self._pol().select(DepthFeatures(0, 5, 2.0)) == 1

    def test_tie_ge_boundary_is_half_open(self):
        pol = self._pol()
        # >= resolves ties: s == threshold falls into the upper (shallower) bucket
        assert pol.select(DepthFeatures(0, 5, 0.5)) == 3
        assert pol.select(DepthFeatures(0, 5, 1.5)) == 1

    def test_output_always_in_allowed(self):
        pol = self._pol()
        for s in (None, 0.0, 0.5, 1.0, 1.5, 100.0):
            assert pol.select(DepthFeatures(0, 5, s)) in {1, 3, 5}

    def test_construct_bad_threshold_len(self):
        with pytest.raises(ValueError, match="length"):
            HeuristicDepthPolicy([1, 3, 5], [0.5], 1)  # needs 2 thresholds

    def test_construct_non_ascending_thresholds(self):
        with pytest.raises(ValueError, match="ascending"):
            HeuristicDepthPolicy([1, 3, 5], [1.5, 0.5], 1)

    def test_construct_fallback_not_in_allowed(self):
        with pytest.raises(ValueError, match="fallback_depth"):
            HeuristicDepthPolicy([1, 3, 5], [0.5, 1.5], 2)


# ---- Non-regression parity: constant policy == fixed-depth strategy ----


def _build_chain_library(storage: CacheStorage, *, n_chains: int = 3, chain_len: int = 6) -> None:
    """Insert n_chains linked trajectory chains of chain_len entries each."""
    torch.manual_seed(1)
    payload = CachePayload(action_chunk=torch.randn(50, _DIM))
    for c in range(n_chains):
        traj = f"t{c}"
        ids = [f"{traj}:{i}" for i in range(chain_len)]
        for i in range(chain_len):
            insert_entry(
                storage,
                CheckpointID.CP1,
                torch.randn(1, _DIM),
                payload,
                entry_id=ids[i],
                step_idx=i,
                prev_ids=[ids[i - 1]] if i > 0 else [],
                next_ids=[ids[i + 1]] if i < chain_len - 1 else [],
                trajectory_id=traj,
            )


@pytest.mark.parametrize("base_fusion", ["weighted_rrf", "weighted_score_sum"])
@pytest.mark.parametrize("depth", [1, 3, 5])
def test_constant_depth_parity_partial_history(base_fusion, depth):
    """DynamicDepthKnnStrategy(constant@D) is value-identical to the fixed-depth
    strategy at EVERY history length, including step-0-through-partial-history.
    """
    backend = InMemoryBackend({ROBOT_STATE: _DIM})
    storage = CacheStorage(backend)
    _build_chain_library(storage, chain_len=6)

    weights = None if depth == 1 else [round(1.0 / (k + 1), 4) for k in range(depth)]
    field_sim = {ROBOT_STATE: {"type": "cosine"}}
    score_norm = {
        "type": "per_field",
        "fields": {ROBOT_STATE: {"method": "affine_clip", "params": {"lo": -1.0, "hi": 1.0}}},
    }
    common = dict(
        top_k=3,
        step_filter="all",
        fusion_weights={ROBOT_STATE: 1.0},
        trajectory_depth=depth,
        trajectory_weights=weights,
    )

    if base_fusion == "weighted_rrf":
        fixed = WeightedRrfKnnStrategy(storage, rrf_k=60, field_similarity=field_sim, **common)
        dyn = DynamicDepthKnnStrategy(
            storage,
            base_fusion="weighted_rrf",
            depth_policy=ConstantDepthPolicy(depth),
            allowed_depths=[depth],
            rrf_k=60,
            field_similarity=field_sim,
            **common,
        )
    else:
        fixed = WeightedScoreSumKnnStrategy(
            storage, field_similarity=field_sim, score_normalization=score_norm, **common
        )
        dyn = DynamicDepthKnnStrategy(
            storage,
            base_fusion="weighted_score_sum",
            depth_policy=ConstantDepthPolicy(depth),
            allowed_depths=[depth],
            field_similarity=field_sim,
            score_normalization=score_norm,
            **common,
        )

    torch.manual_seed(7)
    for step in range(6):
        ctx = SearchContext(
            query_keys={ROBOT_STATE: torch.randn(_DIM)},
            checkpoint_id=CheckpointID.CP1,
            current_step=step,
        )
        r_fixed = fixed.search(ctx)
        r_dyn = dyn.search(ctx)
        assert [r.id for r in r_dyn] == [r.id for r in r_fixed], f"ids diverge at step {step}"
        assert [r.score for r in r_dyn] == pytest.approx(
            [r.score for r in r_fixed]
        ), f"scores diverge at step {step}"


# ---- Runtime behaviour: fail-loud, history clamp, session/memo, smoothness ----


class _BadPolicy:
    """DepthPolicy that returns a depth outside allowed_depths."""

    def select(self, features: DepthFeatures) -> int:
        return 99


def _dyn(storage, **overrides) -> DynamicDepthKnnStrategy:
    kwargs = dict(
        base_fusion="weighted_rrf",
        depth_policy=ConstantDepthPolicy(3),
        allowed_depths=[3],
        top_k=1,
        fusion_weights={ROBOT_STATE: 1.0},
        rrf_k=60,
        trajectory_depth=3,
        trajectory_weights=[0.5, 0.3, 0.2],
    )
    kwargs.update(overrides)
    return DynamicDepthKnnStrategy(storage, **kwargs)


def test_out_of_range_policy_output_raises():
    storage = MagicMock(spec=CacheStorage)
    strat = _dyn(storage, depth_policy=_BadPolicy(), allowed_depths=[1, 3])
    ctx = SearchContext(query_keys={ROBOT_STATE: torch.randn(_DIM)}, checkpoint_id=CheckpointID.CP1)
    with pytest.raises(ValueError, match="outside allowed_depths"):
        strat.search(ctx)


def test_history_clamp_truncates_depth_unrenormalized():
    storage = MagicMock(spec=CacheStorage)
    storage.search.return_value = []
    strat = _dyn(
        storage,
        depth_policy=ConstantDepthPolicy(5),
        allowed_depths=[5],
        trajectory_depth=5,
        trajectory_weights=[0.5, 0.2, 0.15, 0.1, 0.05],
    )
    for _ in range(2):  # only 2 history entries -> effective_depth clamps to 2
        ctx = SearchContext(query_keys={ROBOT_STATE: torch.randn(_DIM)}, checkpoint_id=CheckpointID.CP1)
        strat.search(ctx)
    spec: QuerySpec = storage.search.call_args[0][0]
    assert len(spec.trajectory_history) == 2
    # prefix weights, NOT renormalized (mirrors TrajectoryMixin semantics)
    assert spec.trajectory_weights == [0.5, 0.2]


def test_dynamic_session_fields_in_query_spec():
    storage = MagicMock(spec=CacheStorage)
    storage.search.return_value = []
    strat = _dyn(storage)
    strat.on_episode_start()  # mints a per-episode search session id
    for _ in range(3):
        ctx = SearchContext(query_keys={ROBOT_STATE: torch.randn(_DIM)}, checkpoint_id=CheckpointID.CP1)
        strat.search(ctx)
    spec: QuerySpec = storage.search.call_args[0][0]
    assert spec.search_session_id is not None
    assert spec.trajectory_query_ids is not None
    assert len(spec.trajectory_query_ids) == 3


def test_action_smoothness_calculation():
    storage = MagicMock(spec=CacheStorage)
    strat = _dyn(storage, depth_policy=ConstantDepthPolicy(1), allowed_depths=[1], trajectory_depth=1, trajectory_weights=None)
    assert strat._action_smoothness() is None  # 0 actions
    strat.record_action(torch.ones(50, _DIM))
    assert strat._action_smoothness() is None  # only 1 action
    strat.record_action(torch.ones(50, _DIM) * 3.0)
    # chunk[0] diff = ||3*ones - 1*ones|| = ||2*ones(_DIM)|| = 2*sqrt(_DIM)
    assert strat._action_smoothness() == pytest.approx(2.0 * math.sqrt(_DIM))


def test_base_fusion_invalid_raises():
    storage = MagicMock(spec=CacheStorage)
    with pytest.raises(ValueError, match="base_fusion"):
        _dyn(storage, base_fusion="bogus")


def test_protocol_compliance_dynamic():
    storage = MagicMock(spec=CacheStorage)
    assert isinstance(_dyn(storage), SearchStrategy)


# ---- Config parse round-trip + validation ----

_VALID_DYN_HEURISTIC_YAML = """enabled: true
keys:
  robot_state: {enabled: true, weight: 1.0}
key_builder:
  type: placeholder
checkpoints:
  cp1:
    enabled: true
    gate: {type: always_search}
    judge: {type: always_hit}
    search_strategy:
      type: dynamic_depth_knn
      base_fusion: weighted_rrf
      top_k: 1
      trajectory_depth: 5
      trajectory_weights: [0.4, 0.3, 0.15, 0.1, 0.05]
      allowed_depths: [1, 3, 5]
      depth_policy:
        type: heuristic
        smoothness_thresholds: [0.5, 1.5]
        fallback_depth: 1
backend:
  type: in_memory
  vector_dims: {robot_state: 32}
"""

_VALID_DYN_CONSTANT_YAML = """enabled: true
keys:
  robot_state: {enabled: true, weight: 1.0}
key_builder:
  type: placeholder
checkpoints:
  cp1:
    enabled: true
    gate: {type: always_search}
    judge: {type: always_hit}
    search_strategy:
      type: dynamic_depth_knn
      base_fusion: weighted_rrf
      top_k: 1
      trajectory_depth: 3
      trajectory_weights: [0.5, 0.3, 0.2]
      allowed_depths: [1, 3]
      depth_policy:
        type: constant
        depth: 3
backend:
  type: in_memory
  vector_dims: {robot_state: 32}
"""


def _load(tmp_path, text):
    from openpi.cache.config import load_cache_config

    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return load_cache_config(str(p))


def test_depth_policy_parse_roundtrip_heuristic(tmp_path):
    from openpi.cache.config import DepthPolicyConfig

    cfg = _load(tmp_path, _VALID_DYN_HEURISTIC_YAML)
    ss = cfg.checkpoints["cp1"].search_strategy
    assert ss.type == "dynamic_depth_knn"
    assert ss.base_fusion == "weighted_rrf"
    assert ss.allowed_depths == [1, 3, 5]
    # depth_policy must be materialized as a dataclass, not left a raw dict.
    assert isinstance(ss.depth_policy, DepthPolicyConfig)
    assert ss.depth_policy.type == "heuristic"
    assert ss.depth_policy.smoothness_thresholds == [0.5, 1.5]
    assert ss.depth_policy.fallback_depth == 1


def test_depth_policy_parse_roundtrip_constant(tmp_path):
    from openpi.cache.config import DepthPolicyConfig

    cfg = _load(tmp_path, _VALID_DYN_CONSTANT_YAML)
    ss = cfg.checkpoints["cp1"].search_strategy
    assert isinstance(ss.depth_policy, DepthPolicyConfig)
    assert ss.depth_policy.type == "constant"
    assert ss.depth_policy.depth == 3


def test_build_dynamic_strategy_constant():
    from openpi.cache.config import DepthPolicyConfig, SearchStrategyConfig, _build_search_strategy

    storage = MagicMock(spec=CacheStorage)
    cfg = SearchStrategyConfig(
        type="dynamic_depth_knn",
        base_fusion="weighted_rrf",
        trajectory_depth=3,
        trajectory_weights=[0.5, 0.3, 0.2],
        allowed_depths=[1, 3],
        depth_policy=DepthPolicyConfig(type="constant", depth=3),
    )
    strat = _build_search_strategy(cfg, storage, {ROBOT_STATE: 1.0})
    assert isinstance(strat, DynamicDepthKnnStrategy)


def test_build_dynamic_strategy_heuristic():
    from openpi.cache.config import DepthPolicyConfig, SearchStrategyConfig, _build_search_strategy

    storage = MagicMock(spec=CacheStorage)
    cfg = SearchStrategyConfig(
        type="dynamic_depth_knn",
        base_fusion="weighted_rrf",
        trajectory_depth=5,
        trajectory_weights=[0.4, 0.3, 0.15, 0.1, 0.05],
        allowed_depths=[1, 3, 5],
        depth_policy=DepthPolicyConfig(type="heuristic", smoothness_thresholds=[0.5, 1.5], fallback_depth=1),
    )
    strat = _build_search_strategy(cfg, storage, {ROBOT_STATE: 1.0})
    assert isinstance(strat, DynamicDepthKnnStrategy)


def test_valid_dynamic_yaml_loads(tmp_path):
    cfg = _load(tmp_path, _VALID_DYN_HEURISTIC_YAML)
    assert cfg.checkpoints["cp1"].search_strategy.type == "dynamic_depth_knn"


def test_dynamic_requires_in_memory(tmp_path):
    from openpi.cache.config import ConfigValidationError

    bad = _VALID_DYN_HEURISTIC_YAML.replace("type: in_memory", "type: qdrant")
    with pytest.raises(ConfigValidationError, match="in_memory"):
        _load(tmp_path, bad)


def test_dynamic_base_fusion_invalid(tmp_path):
    from openpi.cache.config import ConfigValidationError

    bad = _VALID_DYN_HEURISTIC_YAML.replace("base_fusion: weighted_rrf", "base_fusion: bogus_fusion")
    with pytest.raises(ConfigValidationError, match="base_fusion"):
        _load(tmp_path, bad)


def test_dynamic_allowed_depth_out_of_range(tmp_path):
    from openpi.cache.config import ConfigValidationError

    bad = _VALID_DYN_HEURISTIC_YAML.replace("allowed_depths: [1, 3, 5]", "allowed_depths: [1, 3, 9]")
    with pytest.raises(ConfigValidationError, match="trajectory_depth"):
        _load(tmp_path, bad)


def test_dynamic_constant_depth_not_in_allowed(tmp_path):
    from openpi.cache.config import ConfigValidationError

    bad = _VALID_DYN_CONSTANT_YAML.replace("        depth: 3", "        depth: 2")
    with pytest.raises(ConfigValidationError, match="allowed_depths"):
        _load(tmp_path, bad)


def test_dynamic_heuristic_threshold_len_wrong(tmp_path):
    from openpi.cache.config import ConfigValidationError

    bad = _VALID_DYN_HEURISTIC_YAML.replace("smoothness_thresholds: [0.5, 1.5]", "smoothness_thresholds: [0.5]")
    with pytest.raises(ConfigValidationError, match="length must equal"):
        _load(tmp_path, bad)


def test_dynamic_allowed_depths_empty_rejected(tmp_path):
    """Explicit empty allowed_depths must error, NOT default to [trajectory_depth]."""
    from openpi.cache.config import ConfigValidationError

    bad = _VALID_DYN_CONSTANT_YAML.replace("allowed_depths: [1, 3]", "allowed_depths: []")
    with pytest.raises(ConfigValidationError, match="non-empty"):
        _load(tmp_path, bad)


def test_dynamic_allowed_depths_non_ascending(tmp_path):
    from openpi.cache.config import ConfigValidationError

    bad = _VALID_DYN_HEURISTIC_YAML.replace("allowed_depths: [1, 3, 5]", "allowed_depths: [5, 3, 1]")
    with pytest.raises(ConfigValidationError, match="ascending"):
        _load(tmp_path, bad)


def test_dynamic_heuristic_fallback_not_in_allowed(tmp_path):
    from openpi.cache.config import ConfigValidationError

    bad = _VALID_DYN_HEURISTIC_YAML.replace("fallback_depth: 1", "fallback_depth: 2")
    with pytest.raises(ConfigValidationError, match="fallback_depth"):
        _load(tmp_path, bad)
