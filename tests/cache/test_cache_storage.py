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
