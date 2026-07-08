"""Shared fixtures and helpers for cache tests.

InMemoryBackend is now in src/openpi/cache/backends/in_memory_backend.py.
This module re-exports it and provides factory helpers that wire up
Orchestrator with the new per-checkpoint dict signatures.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import ThresholdJudge
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import CacheEntry, CachePayload, QuerySpec
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
# Direct insertion helpers
# ---------------------------------------------------------------------------


def stable_hash(checkpoint_id: CheckpointID, query_keys: dict[str, torch.Tensor]) -> str:
    """Deterministic id from checkpoint + query key bytes (test helper)."""
    h = hashlib.sha256()
    h.update(checkpoint_id.name.encode())
    for name in sorted(query_keys.keys()):
        h.update(name.encode())
        h.update(query_keys[name].numpy().tobytes())
    return h.hexdigest()[:32]


def insert_entry(
    storage: CacheStorage,
    checkpoint_id: CheckpointID,
    state: torch.Tensor,
    payload: CachePayload,
    *,
    entry_id: Optional[str] = None,
    step_idx: Optional[int] = None,
    prev_ids: Optional[list[str]] = None,
    next_ids: Optional[list[str]] = None,
    trajectory_id: Optional[str] = None,
    outcome: Optional[int] = None,
) -> CacheEntry:
    """Insert a CacheEntry directly into storage. Returns the entry.

    PlaceholderKeyBuilder produces {"robot_state": state.squeeze(0)},
    so this helper mimics that key extraction for test setup.
    """
    query_keys = {"robot_state": state.squeeze(0).cpu().float().contiguous()}
    eid = entry_id or stable_hash(checkpoint_id, query_keys)
    entry = CacheEntry(
        id=eid,
        checkpoint_id=checkpoint_id,
        query_keys=query_keys,
        payload=payload,
        step_idx=step_idx,
        prev_ids=prev_ids or [],
        next_ids=next_ids or [],
        trajectory_id=trajectory_id,
        outcome=outcome,
    )
    storage.insert(entry)
    return entry


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


class TestStorageSearchStrategy:
    """Test-only strategy that forwards directly to CacheStorage.search().

    Used by in-memory orchestrator tests so production Qdrant-specific search
    strategy types are not exercised against the wrong backend.
    """

    def __init__(self, storage: CacheStorage, *, top_k: int = 1) -> None:
        self._storage = storage
        self._top_k = top_k

    def search(self, ctx):
        return self._storage.search(
            QuerySpec(
                query_keys=ctx.query_keys,
                top_k=self._top_k,
                checkpoint_id=ctx.checkpoint_id,
            )
        )


def make_orchestrator(
    vector_dims: Optional[dict[str, int]] = None,
    gate=None,
    judge=None,
    write_policy=None,
) -> tuple[CacheOrchestrator, InMemoryBackend, CacheStorage]:
    """Create orchestrator with InMemoryBackend + default components."""
    dims = vector_dims or {"robot_state": 32}
    backend = InMemoryBackend(dims)
    storage = CacheStorage(backend)
    kb = PlaceholderKeyBuilder()
    g = gate if gate is not None else AlwaysSearchGate()
    j = judge if judge is not None else ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)
    timer = SystemTimer(enabled=False)
    strategy = TestStorageSearchStrategy(storage, top_k=1)
    orch = CacheOrchestrator(
        storage,
        kb,
        gates=_wrap_per_checkpoint(g),
        judges=_wrap_per_checkpoint(j),
        search_strategies=_wrap_per_checkpoint(strategy),
        timer=timer,
        write_policy=write_policy,
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
    strategy = TestStorageSearchStrategy(storage, top_k=1)
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
