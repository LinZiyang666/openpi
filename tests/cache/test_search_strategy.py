"""Tests for SearchStrategy (SearchContext, QdrantWeightedRrfKnnStrategy)."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import (
    QdrantWeightedRrfKnnStrategy,
    SearchContext,
    SearchStrategy,
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
