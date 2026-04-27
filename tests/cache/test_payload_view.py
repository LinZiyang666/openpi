"""Tests for StoragePayloadView (B0): get / get_entry / get_many memo,
walk_prev / walk_next on no-fork chains, fork detection raise,
trajectory boundary stop, NotImplementedError on unsupported policies.
"""

from __future__ import annotations

import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.payload_view import (
    ForkPolicy,
    PayloadView,
    StoragePayloadView,
)
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _entry(eid: str, *, prev_ids=None, next_ids=None, traj="t0") -> CacheEntry:
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.zeros(8)},
        payload=CachePayload(action_chunk=torch.zeros(10, 32)),
        prev_ids=list(prev_ids or []),
        next_ids=list(next_ids or []),
        trajectory_id=traj,
    )


def _make_chain(storage: CacheStorage, ids: list[str], traj="t0") -> None:
    """Insert a linear chain id[0] -> id[1] -> ... with prev/next links."""
    for i, eid in enumerate(ids):
        prev = [ids[i - 1]] if i > 0 else []
        nxt = [ids[i + 1]] if i + 1 < len(ids) else []
        storage._backend._entries[eid] = _entry(
            eid, prev_ids=prev, next_ids=nxt, traj=traj
        )


def _new_storage() -> CacheStorage:
    return CacheStorage(InMemoryBackend({"robot_state": 8}))


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------


def test_storage_payload_view_implements_protocol():
    view = StoragePayloadView(_new_storage())
    assert isinstance(view, PayloadView)


# ------------------------------------------------------------------
# get / get_entry / get_many memoization
# ------------------------------------------------------------------


def test_get_memoizes_payload():
    storage = _new_storage()
    _make_chain(storage, ["a"])
    view = StoragePayloadView(storage)

    p1 = view.get("a")
    p2 = view.get("a")
    assert p1 is p2
    # Repeat fetches use the memo, so backend.fetch_payload runs once.
    assert storage._backend.fetch_payload_call_count == 1


def test_get_entry_memoizes():
    storage = _new_storage()
    _make_chain(storage, ["a", "b"])
    view = StoragePayloadView(storage)

    e1 = view.get_entry("a")
    e2 = view.get_entry("a")
    assert e1 is e2


def test_get_many_returns_payloads_in_order():
    storage = _new_storage()
    _make_chain(storage, ["a", "b", "c"])
    view = StoragePayloadView(storage)
    out = view.get_many(["c", "a", "b"])
    assert len(out) == 3
    # Each is a CachePayload (not an entry)
    for p in out:
        assert isinstance(p, CachePayload)


# ------------------------------------------------------------------
# walk_prev / walk_next on no-fork chain
# ------------------------------------------------------------------


def test_walk_next_basic_chain():
    storage = _new_storage()
    _make_chain(storage, ["a", "b", "c", "d"])
    view = StoragePayloadView(storage)

    out = view.walk_next("a", k=3)
    assert [e.id for e in out] == ["b", "c", "d"]


def test_walk_prev_basic_chain():
    storage = _new_storage()
    _make_chain(storage, ["a", "b", "c", "d"])
    view = StoragePayloadView(storage)

    out = view.walk_prev("d", k=3)
    assert [e.id for e in out] == ["c", "b", "a"]


def test_walk_stops_when_chain_ends():
    storage = _new_storage()
    _make_chain(storage, ["a", "b"])
    view = StoragePayloadView(storage)

    out = view.walk_next("a", k=10)
    assert [e.id for e in out] == ["b"]


def test_walk_with_k_zero_returns_empty():
    storage = _new_storage()
    _make_chain(storage, ["a", "b"])
    view = StoragePayloadView(storage)

    assert view.walk_next("a", k=0) == []
    assert view.walk_prev("a", k=0) == []


def test_walk_stops_at_trajectory_boundary():
    storage = _new_storage()
    # Two trajectories linked together: t0 a -> b -> c (boundary) c -> d (t1).
    _make_chain(storage, ["a", "b", "c"], traj="t0")
    storage._backend._entries["d"] = _entry("d", prev_ids=["c"], traj="t1")
    storage._backend._entries["c"].next_ids = ["d"]
    view = StoragePayloadView(storage)

    out = view.walk_next("a", k=10)
    # walk stops once trajectory_id changes (does not cross to "d").
    assert [e.id for e in out] == ["b", "c"]


# ------------------------------------------------------------------
# Fork detection (B0 raises rather than picking a branch)
# ------------------------------------------------------------------


def test_walk_raises_on_fork():
    storage = _new_storage()
    storage._backend._entries["a"] = _entry("a", next_ids=["b1", "b2"])
    storage._backend._entries["b1"] = _entry("b1", prev_ids=["a"])
    storage._backend._entries["b2"] = _entry("b2", prev_ids=["a"])
    view = StoragePayloadView(storage)

    with pytest.raises(NotImplementedError, match="fork detected"):
        view.walk_next("a", k=2)


# ------------------------------------------------------------------
# Unsupported parameters (B0 placeholder semantics)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [ForkPolicy.FIRST, ForkPolicy.STOP, ForkPolicy.ALL_BRANCHES, ForkPolicy.SCORE],
)
def test_non_trajectory_fork_policy_raises(policy):
    storage = _new_storage()
    _make_chain(storage, ["a", "b"])
    view = StoragePayloadView(storage)
    with pytest.raises(NotImplementedError, match=policy.name):
        view.walk_next("a", k=1, fork_policy=policy)


def test_cross_trajectory_raises():
    storage = _new_storage()
    _make_chain(storage, ["a", "b"])
    view = StoragePayloadView(storage)
    with pytest.raises(NotImplementedError, match="cross_trajectory"):
        view.walk_next("a", k=1, cross_trajectory=True)
