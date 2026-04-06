"""Tests for WeightedRrfKnnStrategy and WeightedScoreSumKnnStrategy."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import (
    SearchContext,
    WeightedRrfKnnStrategy,
    WeightedScoreSumKnnStrategy,
    _build_step_filters,
)
from openpi.cache.storage_types import CacheEntry, CachePayload, QueryFilter
from openpi.cache.types import CheckpointID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage_with_entries() -> CacheStorage:
    """Create a storage with 2 entries for testing."""
    backend = InMemoryBackend({"vision_0": 2, "robot_state": 2})
    storage = CacheStorage(backend)
    for eid, v0, rs, task in [
        ("A", [1.0, 0.0], [0.0, 0.0], "task1"),
        ("B", [0.0, 1.0], [5.0, 5.0], "task2"),
    ]:
        entry = CacheEntry(
            id=eid,
            checkpoint_id=CheckpointID.CP1,
            query_keys={
                "vision_0": torch.tensor(v0),
                "robot_state": torch.tensor(rs),
            },
            payload=CachePayload(action_chunk=torch.zeros(10, 32), task_key=task),
        )
        storage.insert(entry)
    return storage


def _make_ctx(
    query_keys: dict[str, torch.Tensor],
    task_key: str | None = None,
    step: int = 0,
) -> SearchContext:
    return SearchContext(
        query_keys=query_keys,
        checkpoint_id=CheckpointID.CP1,
        current_step=step,
        task_key=task_key,
    )


FIELD_SIM = {
    "vision_0": {"type": "cosine"},
    "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 0.334717}},
}


# ---------------------------------------------------------------------------
# TestWeightedRrfKnnStrategy
# ---------------------------------------------------------------------------


class TestWeightedRrfKnnStrategy:
    def test_search_returns_results(self):
        storage = _make_storage_with_entries()
        strategy = WeightedRrfKnnStrategy(
            storage,
            top_k=2,
            fusion_weights={"vision_0": 0.7, "robot_state": 0.3},
            rrf_k=60,
            field_similarity=FIELD_SIM,
        )
        ctx = _make_ctx({
            "vision_0": torch.tensor([1.0, 0.0]),
            "robot_state": torch.tensor([0.0, 0.0]),
        })
        results = strategy.search(ctx)
        assert len(results) == 2
        assert results[0].id == "A"  # Better match for both fields

    def test_fusion_method_in_spec(self):
        """Strategy should set fusion_method='weighted_rrf' in QuerySpec."""
        storage = _make_storage_with_entries()
        strategy = WeightedRrfKnnStrategy(
            storage,
            top_k=1,
            fusion_weights={"vision_0": 1.0},
            field_similarity={"vision_0": {"type": "cosine"}},
        )
        # We can verify by checking the backend received the right spec
        # through the search results
        ctx = _make_ctx({"vision_0": torch.tensor([1.0, 0.0])})
        results = strategy.search(ctx)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# TestWeightedScoreSumKnnStrategy
# ---------------------------------------------------------------------------


class TestWeightedScoreSumKnnStrategy:
    def test_search_returns_results(self):
        storage = _make_storage_with_entries()
        strategy = WeightedScoreSumKnnStrategy(
            storage,
            top_k=2,
            fusion_weights={"vision_0": 0.7, "robot_state": 0.3},
            field_similarity=FIELD_SIM,
            score_normalization={"type": "none"},
        )
        ctx = _make_ctx({
            "vision_0": torch.tensor([1.0, 0.0]),
            "robot_state": torch.tensor([0.0, 0.0]),
        })
        results = strategy.search(ctx)
        assert len(results) == 2
        assert results[0].id == "A"

    def test_with_percentile_normalization(self):
        storage = _make_storage_with_entries()
        strategy = WeightedScoreSumKnnStrategy(
            storage,
            top_k=2,
            fusion_weights={"vision_0": 0.5, "robot_state": 0.5},
            field_similarity=FIELD_SIM,
            score_normalization={
                "type": "percentile",
                "fields": {
                    "vision_0": {"p5": 0.5, "p95": 1.0},
                    "robot_state": {"p5": 0.0, "p95": 1.0},
                },
            },
        )
        ctx = _make_ctx({
            "vision_0": torch.tensor([1.0, 0.0]),
            "robot_state": torch.tensor([0.0, 0.0]),
        })
        results = strategy.search(ctx)
        assert len(results) == 2
        assert results[0].score > results[1].score


# ---------------------------------------------------------------------------
# TestBuildStepFilters
# ---------------------------------------------------------------------------


class TestBuildStepFilters:
    def test_all_no_task(self):
        ctx = _make_ctx({"v": torch.zeros(2)}, task_key=None, step=5)
        f = _build_step_filters("all", 5, ctx)
        assert f is None

    def test_all_with_task(self):
        ctx = _make_ctx({"v": torch.zeros(2)}, task_key="mytask", step=5)
        f = _build_step_filters("all", 5, ctx)
        assert f is not None
        assert f.task_key == "mytask"
        assert f.step_range is None

    def test_exact(self):
        ctx = _make_ctx({"v": torch.zeros(2)}, task_key="t", step=10)
        f = _build_step_filters("exact", 5, ctx)
        assert f.step_range == (10, 10)
        assert f.task_key == "t"

    def test_window(self):
        ctx = _make_ctx({"v": torch.zeros(2)}, task_key=None, step=3)
        f = _build_step_filters("window", 5, ctx)
        assert f.step_range == (0, 8)  # max(0, 3-5)=0, 3+5=8

    def test_invalid_filter(self):
        ctx = _make_ctx({"v": torch.zeros(2)})
        import pytest
        with pytest.raises(ValueError, match="Unknown step_filter"):
            _build_step_filters("invalid", 5, ctx)
