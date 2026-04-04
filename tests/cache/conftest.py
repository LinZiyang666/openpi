"""Shared fixtures and helpers for cache tests.

InMemoryBackend is now in src/openpi/cache/backends/in_memory_backend.py.
This module re-exports it and provides factory helpers that wire up
Orchestrator with the new per-checkpoint dict signatures.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import ThresholdJudge
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.components.search_strategy import SimpleKnnStrategy
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID


# ---------------------------------------------------------------------------
# CountingStorage — wraps CacheStorage to track fetch_payload calls
# ---------------------------------------------------------------------------


class CountingStorage(CacheStorage):
    """CacheStorage subclass that counts fetch_payload() calls."""

    def __init__(self, backend: InMemoryBackend) -> None:
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


def _wrap_per_checkpoint(component):
    """Wrap a single component into a dict for all checkpoints."""
    return {CheckpointID.CP1: component, CheckpointID.CP3: component}


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
    strategy = SimpleKnnStrategy(storage, top_k=1)
    orch = CacheOrchestrator(
        storage,
        kb,
        gates=_wrap_per_checkpoint(g),
        judges=_wrap_per_checkpoint(j),
        search_strategies=_wrap_per_checkpoint(strategy),
        timer=timer,
    )
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
    strategy = SimpleKnnStrategy(storage, top_k=1)
    orch = CacheOrchestrator(
        storage,
        kb,
        gates=_wrap_per_checkpoint(g),
        judges=_wrap_per_checkpoint(j),
        search_strategies=_wrap_per_checkpoint(strategy),
        timer=timer,
    )
    return orch, backend, storage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_payload() -> CachePayload:
    """Valid CP1 CachePayload with deterministic action_chunk."""
    torch.manual_seed(42)
    return CachePayload(action_chunk=torch.randn(50, 32))
