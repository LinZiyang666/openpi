"""Tests for SearchStrategy (SearchContext, SimpleKnnStrategy)."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import (
    SearchContext,
    SearchStrategy,
    SimpleKnnStrategy,
)
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    QuerySpec,
    SearchResultLite,
)
from openpi.cache.types import ROBOT_STATE, CheckpointID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage() -> tuple[CacheStorage, InMemoryBackend]:
    dims = {"robot_state": 32}
    backend = InMemoryBackend(dims)
    storage = CacheStorage(backend)
    return storage, backend


def _insert_entry(storage: CacheStorage, state: torch.Tensor, cp: CheckpointID = CheckpointID.CP1) -> None:
    """Insert a single entry with given state vector."""
    import torch.nn.functional as F

    key = F.normalize(state, dim=0)
    entry = CacheEntry(
        id=f"test_{key[0].item():.4f}",
        checkpoint_id=cp,
        query_keys={ROBOT_STATE: key.cpu().float().contiguous()},
        payload=CachePayload(action_chunk=torch.randn(50, 32)),
    )
    storage.insert(entry)


# ---------------------------------------------------------------------------
# test_simple_knn_basic_search: step_filter="all", full path through InMemoryBackend
# ---------------------------------------------------------------------------


def test_simple_knn_basic_search():
    storage, backend = _make_storage()
    state = torch.zeros(32)
    state[0] = 1.0
    _insert_entry(storage, state)

    strategy = SimpleKnnStrategy(storage, top_k=1, step_filter="all")
    ctx = SearchContext(
        query_keys={ROBOT_STATE: state},
        checkpoint_id=CheckpointID.CP1,
        current_step=0,
    )
    results = strategy.search(ctx)
    assert len(results) >= 1
    assert isinstance(results[0], SearchResultLite)
    assert results[0].score > 0.9


# ---------------------------------------------------------------------------
# test_query_spec_fusion_params: fusion_weights + backend_hints in QuerySpec
# ---------------------------------------------------------------------------


def test_query_spec_fusion_params():
    """Verify fusion_weights and backend_hints are correctly passed to storage.search()."""
    storage = MagicMock(spec=CacheStorage)
    storage.search.return_value = []

    weights = {"robot_state": 1.0, "vision_0": 0.5}
    strategy = SimpleKnnStrategy(
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

    strategy = SimpleKnnStrategy(storage, step_filter="all")
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

    strategy = SimpleKnnStrategy(storage, step_filter="exact")
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

    strategy = SimpleKnnStrategy(storage, step_filter="window", step_window=3)
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

    strategy = SimpleKnnStrategy(storage, step_filter="window", step_window=5)
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
# test_protocol_compliance: SimpleKnnStrategy satisfies SearchStrategy Protocol
# ---------------------------------------------------------------------------


def test_protocol_compliance():
    storage, _ = _make_storage()
    strategy = SimpleKnnStrategy(storage)
    assert isinstance(strategy, SearchStrategy)
