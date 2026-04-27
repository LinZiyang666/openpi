"""Tests for the duck-typed verdict-factor facade methods on CacheStorage:
fetch_entry + library_stats property. Both forward to backend capability
when present (InMemoryBackend) and degrade gracefully otherwise.
"""

from __future__ import annotations

import pytest
import torch

from openpi.cache.backend_base import VectorStoreBackend
from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.storage_types import (
    BatchInsertResult,
    CacheEntry,
    CachePayload,
    QuerySpec,
    SearchResultLite,
)
from openpi.cache.types import CheckpointID


def _entry(eid: str = "a") -> CacheEntry:
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.zeros(8)},
        payload=CachePayload(action_chunk=torch.zeros(10, 32)),
    )


# ------------------------------------------------------------------
# library_stats facade
# ------------------------------------------------------------------


def test_library_stats_default_none():
    storage = CacheStorage(InMemoryBackend({"robot_state": 8}))
    assert storage.library_stats is None


def test_library_stats_reads_backend_attribute():
    backend = InMemoryBackend({"robot_state": 8})
    sentinel = object()
    backend.library_stats = sentinel
    storage = CacheStorage(backend)
    assert storage.library_stats is sentinel


# ------------------------------------------------------------------
# fetch_entry facade
# ------------------------------------------------------------------


def test_fetch_entry_returns_entry():
    backend = InMemoryBackend({"robot_state": 8})
    backend._entries["a"] = _entry("a")
    storage = CacheStorage(backend)
    e = storage.fetch_entry("a")
    assert isinstance(e, CacheEntry)
    assert e.id == "a"


def test_fetch_entry_propagates_keyerror():
    storage = CacheStorage(InMemoryBackend({"robot_state": 8}))
    with pytest.raises(KeyError):
        storage.fetch_entry("missing")


# ------------------------------------------------------------------
# Backend without fetch_entry capability
# ------------------------------------------------------------------


class _NoFetchEntryBackend(VectorStoreBackend):
    """Minimal backend without fetch_entry — exercises the facade
    NotImplementedError path."""

    @property
    def vector_dims(self):
        return {"robot_state": 8}

    def supported_filters(self):
        return frozenset()

    def insert(self, entry):
        return None

    def fetch_payload(self, id):
        return CachePayload(action_chunk=torch.zeros(10, 32))

    def delete(self, ids):
        return None

    def count(self):
        return 0

    def batch_insert(self, entries):
        return BatchInsertResult(inserted=0, failed_ids=[])

    def search(self, spec):
        return []

    def close(self):
        pass


def test_fetch_entry_unsupported_backend_raises_not_implemented():
    storage = CacheStorage(_NoFetchEntryBackend())
    with pytest.raises(NotImplementedError, match="fetch_entry"):
        storage.fetch_entry("anything")


def test_library_stats_unsupported_backend_returns_none():
    storage = CacheStorage(_NoFetchEntryBackend())
    assert storage.library_stats is None
