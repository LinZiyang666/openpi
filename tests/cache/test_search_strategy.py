"""Tests for SearchStrategy (SearchContext, QdrantWeightedRrfKnnStrategy, TrajectoryMixin)."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import (
    QdrantWeightedRrfKnnStrategy,
    SearchContext,
    SearchStrategy,
    TrajectoryMixin,
    WeightedRrfKnnStrategy,
)
from openpi.cache.storage_types import QuerySpec, SearchResultLite
from openpi.cache.types import ROBOT_STATE, CheckpointID


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
