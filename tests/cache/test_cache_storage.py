"""Tests for CacheStorage._check_entry_dims intersection fix (T5.1–T5.4)."""

import pytest
import torch

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

from tests.cache.conftest import InMemoryBackend


def _entry(query_keys: dict[str, torch.Tensor], cp: CheckpointID = CheckpointID.CP1) -> CacheEntry:
    """Helper: build a CacheEntry with given query_keys."""
    return CacheEntry(
        id="test_entry",
        checkpoint_id=cp,
        query_keys=query_keys,
        payload=CachePayload(action_chunk=torch.randn(50, 32)),
    )


# T5.1: entry has subset of backend fields -> accepted
def test_check_entry_dims_subset_accepted():
    backend = InMemoryBackend({"robot_state": 32, "vision_0": 1024})
    storage = CacheStorage(backend)
    entry = _entry({"robot_state": torch.randn(32)})
    storage.insert(entry)  # should not raise
    assert backend.count() == 1


# T5.2: no overlap -> raises ValueError
def test_check_entry_dims_no_overlap_raises():
    backend = InMemoryBackend({"vision_0": 1024})
    storage = CacheStorage(backend)
    entry = _entry({"robot_state": torch.randn(32)})
    with pytest.raises(ValueError, match="no fields matching"):
        storage.insert(entry)


# T5.3: overlapping field, wrong dim -> raises ValueError
def test_check_entry_dims_wrong_dim_raises():
    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    entry = _entry({"robot_state": torch.randn(64)})
    with pytest.raises(ValueError, match="shape mismatch"):
        storage.insert(entry)


# T5.4: extra fields in entry not in backend -> silently ignored
def test_check_entry_dims_extra_fields_ignored():
    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    entry = _entry({
        "robot_state": torch.randn(32),
        "unknown_field": torch.randn(10),
    })
    storage.insert(entry)  # should not raise
    assert backend.count() == 1


# ---------------------------------------------------------------------------
# Prefill mode (Step 3 spawn runner trajectory-history replay)
# ---------------------------------------------------------------------------


def _query_spec(dim: int = 32) -> "QuerySpec":  # noqa: F821 — forward ref only used in tests
    from openpi.cache.storage_types import QuerySpec

    return QuerySpec(
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.randn(dim)},
        top_k=1,
    )


def test_prefill_mode_search_returns_synthetic_hit():
    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    payload = CachePayload(action_chunk=torch.arange(50 * 32, dtype=torch.float32).reshape(50, 32))
    storage.enter_prefill_mode(payload)

    results = storage.search(_query_spec())

    assert len(results) == 1
    assert results[0].id == "__prefill__"
    assert results[0].score == 1.0
    assert results[0].checkpoint_id == CheckpointID.CP1


def test_prefill_mode_fetch_payload_returns_stored_payload():
    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    payload = CachePayload(action_chunk=torch.arange(50 * 32, dtype=torch.float32).reshape(50, 32))
    storage.enter_prefill_mode(payload)

    # fetch_payload should return the stored payload regardless of id.
    out = storage.fetch_payload("any-id-whatsoever")
    assert out is payload


def test_prefill_mode_does_not_touch_backend_on_search():
    class _ExplodingBackend(InMemoryBackend):
        def search(self, spec):  # noqa: D401 — test helper
            raise AssertionError("backend.search should not be called in prefill mode")

        def fetch_payload(self, id):
            raise AssertionError("backend.fetch_payload should not be called in prefill mode")

    backend = _ExplodingBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    storage.enter_prefill_mode(CachePayload(action_chunk=torch.zeros(50, 32)))
    storage.search(_query_spec())
    storage.fetch_payload("whatever")  # would assert if backend got called


def test_exit_prefill_mode_restores_backend_path():
    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    storage.enter_prefill_mode(CachePayload(action_chunk=torch.zeros(50, 32)))
    storage.exit_prefill_mode()

    # After exit: synthetic hit is gone; backend returns empty (nothing inserted).
    results = storage.search(_query_spec())
    assert results == []


def test_prefill_mode_replacing_payload_is_idempotent():
    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    p1 = CachePayload(action_chunk=torch.zeros(50, 32))
    p2 = CachePayload(action_chunk=torch.ones(50, 32))
    storage.enter_prefill_mode(p1)
    storage.enter_prefill_mode(p2)  # replaces p1

    assert storage.fetch_payload("x") is p2


def test_prefill_mode_validation_still_runs():
    """Shape validation must still fire in prefill mode so misconfigured callers get a clear error."""
    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    storage.enter_prefill_mode(CachePayload(action_chunk=torch.zeros(50, 32)))

    bad_spec = _query_spec(dim=64)  # backend expects 32
    with pytest.raises(ValueError, match="shape mismatch"):
        storage.search(bad_spec)
