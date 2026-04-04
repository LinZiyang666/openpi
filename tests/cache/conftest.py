"""Shared fixtures and helpers for cache tests.

InMemoryBackend: minimal VectorStoreBackend for unit tests.
Computes cosine similarity in [-1, 1], matching Qdrant cosine semantics.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from openpi.cache.backend_base import VectorStoreBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import ThresholdJudge
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    QuerySpec,
    SearchResultLite,
)
from openpi.cache.timing import SystemTimer


# ---------------------------------------------------------------------------
# InMemoryBackend
# ---------------------------------------------------------------------------


class InMemoryBackend(VectorStoreBackend):
    """Minimal in-memory backend for unit tests.

    Stores entries in a dict. search() computes cosine similarity
    against stored vectors for the first matching field.
    Score range: [-1, 1] (same as Qdrant cosine).
    """

    def __init__(self, vector_dims: dict[str, int]) -> None:
        self._dims = vector_dims
        self._entries: dict[str, CacheEntry] = {}
        # Counters for call tracking in tests.
        self.search_call_count: int = 0
        self.fetch_payload_call_count: int = 0

    @property
    def vector_dims(self) -> dict[str, int]:
        return self._dims

    def supported_filters(self) -> frozenset[str]:
        return frozenset({"checkpoint_id"})

    def insert(self, entry: CacheEntry) -> None:
        self._entries[entry.id] = entry

    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        self.search_call_count += 1
        if not self._entries:
            return []
        results: list[SearchResultLite] = []
        for eid, entry in self._entries.items():
            # Filter by checkpoint_id if specified.
            if spec.checkpoint_id is not None and entry.checkpoint_id != spec.checkpoint_id:
                continue
            # Cosine similarity on first matching field.
            score = 0.0
            for field in spec.query_keys:
                if field in entry.query_keys:
                    q = spec.query_keys[field].float()
                    e = entry.query_keys[field].float()
                    score = float(F.cosine_similarity(q.unsqueeze(0), e.unsqueeze(0)))
                    break
            results.append(
                SearchResultLite(id=eid, score=score, checkpoint_id=entry.checkpoint_id)
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: spec.top_k]

    def fetch_payload(self, id: str) -> CachePayload:
        self.fetch_payload_call_count += 1
        if id not in self._entries:
            raise KeyError(id)
        return self._entries[id].payload

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._entries.pop(i, None)

    def count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# CountingStorage — wraps CacheStorage to track fetch_payload calls
# ---------------------------------------------------------------------------


class CountingStorage(CacheStorage):
    """CacheStorage subclass that counts fetch_payload() calls."""

    def __init__(self, backend: VectorStoreBackend) -> None:
        super().__init__(backend)
        self.fetch_payload_call_count: int = 0

    def fetch_payload(self, id: str) -> CachePayload:
        self.fetch_payload_call_count += 1
        return super().fetch_payload(id)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_stage1(state: torch.Tensor) -> SimpleNamespace:
    """Create a mock Stage1Output with given state tensor."""
    return SimpleNamespace(state=state)


def make_stage3(action_chunk: torch.Tensor) -> SimpleNamespace:
    """Create a mock Stage3Output with given action_chunk tensor."""
    return SimpleNamespace(action_chunk=action_chunk)


def make_orchestrator(
    vector_dims: Optional[dict[str, int]] = None,
    gate=None,
    judge=None,
) -> tuple[CacheOrchestrator, InMemoryBackend, CacheStorage]:
    """Create orchestrator with InMemoryBackend + default components."""
    dims = vector_dims or {"robot_state": 32}
    backend = InMemoryBackend(dims)
    storage = CacheStorage(backend)
    kb = PlaceholderKeyBuilder()
    g = gate if gate is not None else AlwaysSearchGate()
    j = judge if judge is not None else ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)
    timer = SystemTimer(enabled=False)
    orch = CacheOrchestrator(storage, kb, g, j, timer)
    return orch, backend, storage


def make_counting_orchestrator(
    vector_dims: Optional[dict[str, int]] = None,
    judge=None,
) -> tuple[CacheOrchestrator, InMemoryBackend, CountingStorage]:
    """Create orchestrator with CountingStorage for fetch_payload tracking."""
    dims = vector_dims or {"robot_state": 32}
    backend = InMemoryBackend(dims)
    storage = CountingStorage(backend)
    kb = PlaceholderKeyBuilder()
    g = AlwaysSearchGate()
    j = judge if judge is not None else ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)
    timer = SystemTimer(enabled=False)
    orch = CacheOrchestrator(storage, kb, g, j, timer)
    return orch, backend, storage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_payload() -> CachePayload:
    """Valid CP1 CachePayload with deterministic action_chunk."""
    torch.manual_seed(42)
    return CachePayload(action_chunk=torch.randn(50, 32))
