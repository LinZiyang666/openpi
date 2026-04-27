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


# ------------------------------------------------------------------
# B2: load_artifact populates library_stats (with fallback recompute)
# ------------------------------------------------------------------


def test_load_artifact_with_library_stats_field(tmp_path):
    """New (B2) artifact format includes `library_stats` — backend uses
    that field directly, no fallback recompute."""
    import pickle

    import torch as torch_

    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.components.factors.base import LibraryStats

    backend = InMemoryBackend({"robot_state": 4})
    storage = CacheStorage(backend)

    pre_built = LibraryStats(
        action_sigma=torch_.tensor([0.5, 0.5, 0.5, 0.5]),
        action_active_mask=torch_.ones(4, dtype=torch_.bool),
        state_sigma=torch_.tensor([0.3, 0.3, 0.3, 0.3]),
        state_active_mask=torch_.ones(4, dtype=torch_.bool),
    )
    artifact = {
        "key_builder_type": "placeholder",
        "checkpoint_id": "CP1",
        "vector_dims": {"robot_state": 4},
        "entries": [],
        "library_stats": pre_built,
    }
    p = tmp_path / "art.pkl"
    p.write_bytes(pickle.dumps(artifact))

    backend.load_artifact(str(p))

    # Backend now exposes the stats it loaded; facade duck-types through.
    # `is` would be wrong — pickle round-trip creates a new instance.
    loaded = storage.library_stats
    assert loaded is not None
    assert torch_.equal(loaded.action_sigma, pre_built.action_sigma)
    assert torch_.equal(loaded.action_active_mask, pre_built.action_active_mask)
    assert torch_.equal(loaded.state_sigma, pre_built.state_sigma)
    assert torch_.equal(loaded.state_active_mask, pre_built.state_active_mask)


def test_load_artifact_missing_library_stats_falls_back_to_compute(tmp_path):
    """Legacy artifact (no `library_stats` key) → backend recomputes from
    loaded entries and exposes the result via the facade."""
    import pickle

    import torch as torch_

    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID as CP

    backend = InMemoryBackend({"robot_state": 4})
    storage = CacheStorage(backend)

    entries = [
        CacheEntry(
            id=f"e{t}", checkpoint_id=CP.CP1,
            query_keys={"robot_state": torch_.tensor([float(t), 0.0, 0.0, 0.0])},
            payload=CachePayload(action_chunk=torch_.tensor([[float(t), 0.0, 0.0, 0.0]])),
        )
        for t in range(3)
    ]
    artifact = {
        "key_builder_type": "placeholder",
        "checkpoint_id": "CP1",
        "vector_dims": {"robot_state": 4},
        "entries": entries,
        # Note: no `library_stats` key — this is the legacy shape
    }
    p = tmp_path / "legacy.pkl"
    p.write_bytes(pickle.dumps(artifact))

    backend.load_artifact(str(p))

    # Fallback ran — stats are non-None and have the right shapes
    ls = storage.library_stats
    assert ls is not None
    assert ls.action_sigma.shape == (4,)
    assert ls.state_sigma.shape == (4,)


def test_load_artifact_explicitly_none_library_stats_falls_back(tmp_path):
    """Pipeline that pickled `library_stats=None` (e.g., build skipped
    OfflineWriters) must also trigger fallback recompute, not leave the
    backend exposing None."""
    import pickle

    import torch as torch_

    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID as CP

    backend = InMemoryBackend({"robot_state": 2})
    storage = CacheStorage(backend)

    entries = [
        CacheEntry(
            id="e0", checkpoint_id=CP.CP1,
            query_keys={"robot_state": torch_.tensor([1.0, 0.0])},
            payload=CachePayload(action_chunk=torch_.tensor([[1.0, 0.0]])),
        ),
        CacheEntry(
            id="e1", checkpoint_id=CP.CP1,
            query_keys={"robot_state": torch_.tensor([2.0, 0.0])},
            payload=CachePayload(action_chunk=torch_.tensor([[2.0, 0.0]])),
        ),
    ]
    artifact = {
        "key_builder_type": "placeholder",
        "checkpoint_id": "CP1",
        "vector_dims": {"robot_state": 2},
        "entries": entries,
        "library_stats": None,
    }
    p = tmp_path / "none_stats.pkl"
    p.write_bytes(pickle.dumps(artifact))

    backend.load_artifact(str(p))
    assert storage.library_stats is not None
