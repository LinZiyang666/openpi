"""Tests for F1a-A / F1a-T `extract` algorithms.

Algorithm contract is `docs/cache/verdict_factor_judge.md` §3.3:
- F1a-A splice length = K+1 (last K history actions + winner's action_chunk[0])
- F1a-T splice length = 2K+1 (K history states + winner state + K walk_next states)

Tests cover:
- nominal numeric path produces non-NaN under sweep / shake regimes
- history-too-short → all-NaN (episode early boundary)
- F1a-T trajectory boundary (walk_next short) → all-NaN
- F1a-T fork (walk_next raises NotImplementedError) → all-NaN
- state fail-safe layers (b) and (c) — empty mask / per-entry missing
- key contract: extract output keys == descriptor_orientations.keys()
"""

from __future__ import annotations

import math

import pytest
import torch

from openpi.cache.components.factors.base import HistoryView, LibraryStats
from openpi.cache.components.factors.runtime_continuity import (
    RuntimeContinuityAction,
    RuntimeContinuityState,
)
from openpi.cache.storage_types import CacheEntry, CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID


# ------------------------------------------------------------------
# Lightweight PayloadView mock — duck-types the real Protocol surface
# without dragging the whole CacheStorage through.
# ------------------------------------------------------------------


class _MockView:
    def __init__(
        self,
        entries: dict[str, CacheEntry],
        next_chain: dict[str, list[str]] | None = None,
        fork_ids: set[str] | None = None,
    ) -> None:
        self._entries = entries
        self._next_chain = next_chain or {}
        self._fork_ids = fork_ids or set()

    def get(self, entry_id: str) -> CachePayload:
        return self._entries[entry_id].payload

    def get_entry(self, entry_id: str) -> CacheEntry:
        return self._entries[entry_id]

    def get_many(self, entry_ids):
        return [self.get(eid) for eid in entry_ids]

    def walk_prev(self, entry_id, k, **_):
        raise NotImplementedError("test mock: walk_prev unused")

    def walk_next(self, entry_id, k, **_):
        if entry_id in self._fork_ids:
            raise NotImplementedError("fork policy hit")
        ids = self._next_chain.get(entry_id, [])
        return [self._entries[i] for i in ids[:k]]


# ------------------------------------------------------------------
# Fixture helpers
# ------------------------------------------------------------------


def _make_library_stats(action_dim: int = 4, state_dim: int = 4) -> LibraryStats:
    return LibraryStats(
        action_sigma=torch.ones(action_dim),
        action_active_mask=torch.ones(action_dim, dtype=torch.bool),
        state_sigma=torch.ones(state_dim),
        state_active_mask=torch.ones(state_dim, dtype=torch.bool),
    )


def _make_entry(eid: str, action_chunk: torch.Tensor, robot_state: torch.Tensor | None = None) -> CacheEntry:
    qk: dict[str, torch.Tensor] = {}
    if robot_state is not None:
        qk["robot_state"] = robot_state
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys=qk,
        payload=CachePayload(action_chunk=action_chunk),
    )


def _result(eid: str) -> SearchResultLite:
    return SearchResultLite(id=eid, score=1.0, checkpoint_id=CheckpointID.CP1)


# ------------------------------------------------------------------
# F1a-A nominal path
# ------------------------------------------------------------------


def test_f1a_a_nominal_produces_finite_descriptors():
    K = 3
    library_stats = _make_library_stats(action_dim=2)
    extractor = RuntimeContinuityAction(
        window_k=K,
        descriptors=["jerk", "dir", "curv_radius", "cum_disp"],
        library_stats=library_stats,
    )
    # Linear motion: history actions move along [1, 0]
    history_actions = [torch.tensor([float(t), 0.0]) for t in range(K)]
    winner_action = torch.tensor([float(K), 0.0])
    winner_chunk = winner_action.unsqueeze(0)               # [1, 2]
    entries = {"w": _make_entry("w", winner_chunk)}
    view = _MockView(entries)
    history = HistoryView(actions=history_actions, states=[])

    out = extractor.extract([_result("w")], view, history, {})

    assert set(out.keys()) == set(extractor.descriptor_orientations.keys())
    # Linear sweep → jerk == 0 (perfectly smooth)
    assert out["f1a_a_jerk"] == pytest.approx(0.0, abs=1e-6)
    # All velocities identical → dir == 1
    assert out["f1a_a_dir"] == pytest.approx(1.0, abs=1e-6)


def test_f1a_a_history_too_short_returns_all_nan():
    K = 5
    extractor = RuntimeContinuityAction(
        window_k=K,
        descriptors=["jerk", "dir"],
        library_stats=_make_library_stats(action_dim=2),
    )
    history_actions = [torch.zeros(2)] * (K - 1)            # one short
    entries = {"w": _make_entry("w", torch.zeros(1, 2))}
    view = _MockView(entries)
    history = HistoryView(actions=history_actions, states=[])

    out = extractor.extract([_result("w")], view, history, {})

    for v in out.values():
        assert math.isnan(v)


def test_f1a_a_empty_results_returns_all_nan():
    extractor = RuntimeContinuityAction(
        window_k=2,
        descriptors=["jerk", "dir"],
        library_stats=_make_library_stats(action_dim=2),
    )
    history = HistoryView(actions=[torch.zeros(2)] * 2, states=[])
    out = extractor.extract([], _MockView({}), history, {})
    for v in out.values():
        assert math.isnan(v)


# ------------------------------------------------------------------
# F1a-T nominal path
# ------------------------------------------------------------------


def test_f1a_t_nominal_uses_walk_next_for_forward_states():
    K = 2
    library_stats = _make_library_stats(state_dim=2)
    extractor = RuntimeContinuityState(
        window_k=K,
        descriptors=["jerk", "dir", "curv_radius", "cum_disp"],
        library_stats=library_stats,
    )
    # Build a chain w → f1 → f2 with linear state motion
    state = lambda t: torch.tensor([float(t), 0.0])
    entries = {
        "w":  _make_entry("w",  torch.zeros(1, 2), robot_state=state(2)),
        "f1": _make_entry("f1", torch.zeros(1, 2), robot_state=state(3)),
        "f2": _make_entry("f2", torch.zeros(1, 2), robot_state=state(4)),
    }
    next_chain = {"w": ["f1", "f2"]}
    view = _MockView(entries, next_chain=next_chain)

    history = HistoryView(actions=[], states=[state(0), state(1)])

    out = extractor.extract([_result("w")], view, history, {})

    assert set(out.keys()) == set(extractor.descriptor_orientations.keys())
    # 5 collinear z-scored points → jerk == 0, dir == 1
    assert out["f1a_t_jerk"] == pytest.approx(0.0, abs=1e-6)
    assert out["f1a_t_dir"] == pytest.approx(1.0, abs=1e-6)


def test_f1a_t_walk_next_short_returns_all_nan():
    K = 3
    extractor = RuntimeContinuityState(
        window_k=K,
        descriptors=["jerk", "dir"],
        library_stats=_make_library_stats(state_dim=2),
    )
    entries = {
        "w":  _make_entry("w",  torch.zeros(1, 2), robot_state=torch.zeros(2)),
        "f1": _make_entry("f1", torch.zeros(1, 2), robot_state=torch.zeros(2)),
    }
    next_chain = {"w": ["f1"]}                              # only 1 forward, need 3
    view = _MockView(entries, next_chain=next_chain)
    history = HistoryView(actions=[], states=[torch.zeros(2)] * K)

    out = extractor.extract([_result("w")], view, history, {})

    for v in out.values():
        assert math.isnan(v)


def test_f1a_t_fork_raises_caught_returns_all_nan():
    K = 2
    extractor = RuntimeContinuityState(
        window_k=K,
        descriptors=["jerk", "dir"],
        library_stats=_make_library_stats(state_dim=2),
    )
    entries = {"w": _make_entry("w", torch.zeros(1, 2), robot_state=torch.zeros(2))}
    view = _MockView(entries, fork_ids={"w"})               # walk_next will raise
    history = HistoryView(actions=[], states=[torch.zeros(2)] * K)

    out = extractor.extract([_result("w")], view, history, {})

    for v in out.values():
        assert math.isnan(v)


# ------------------------------------------------------------------
# State fail-safe layer (b): empty library / zero mask
# ------------------------------------------------------------------


def test_f1a_t_empty_state_library_returns_all_nan():
    K = 2
    library_stats = LibraryStats(
        action_sigma=torch.ones(2),
        action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=torch.zeros(0),                          # empty state library
        state_active_mask=torch.zeros(0, dtype=torch.bool),
    )
    extractor = RuntimeContinuityState(
        window_k=K, descriptors=["jerk", "dir"], library_stats=library_stats,
    )
    entries = {"w": _make_entry("w", torch.zeros(1, 2), robot_state=torch.zeros(2))}
    view = _MockView(entries)
    history = HistoryView(actions=[], states=[torch.zeros(2)] * K)

    out = extractor.extract([_result("w")], view, history, {})

    for v in out.values():
        assert math.isnan(v)


def test_f1a_t_zero_active_mask_returns_all_nan():
    K = 2
    library_stats = LibraryStats(
        action_sigma=torch.ones(2),
        action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=torch.ones(2),
        state_active_mask=torch.zeros(2, dtype=torch.bool),  # zero active mask
    )
    extractor = RuntimeContinuityState(
        window_k=K, descriptors=["jerk", "dir"], library_stats=library_stats,
    )
    entries = {
        "w":  _make_entry("w",  torch.zeros(1, 2), robot_state=torch.zeros(2)),
        "f1": _make_entry("f1", torch.zeros(1, 2), robot_state=torch.zeros(2)),
        "f2": _make_entry("f2", torch.zeros(1, 2), robot_state=torch.zeros(2)),
    }
    view = _MockView(entries, next_chain={"w": ["f1", "f2"]})
    history = HistoryView(actions=[], states=[torch.zeros(2)] * K)

    out = extractor.extract([_result("w")], view, history, {})

    for v in out.values():
        assert math.isnan(v)


# ------------------------------------------------------------------
# State fail-safe layer (c): per-entry robot_state missing
# ------------------------------------------------------------------


def test_f1a_t_winner_missing_state_returns_all_nan():
    K = 2
    extractor = RuntimeContinuityState(
        window_k=K, descriptors=["jerk", "dir"],
        library_stats=_make_library_stats(state_dim=2),
    )
    # Winner has no robot_state in query_keys
    entries = {"w": _make_entry("w", torch.zeros(1, 2), robot_state=None)}
    view = _MockView(entries, next_chain={"w": []})
    history = HistoryView(actions=[], states=[torch.zeros(2)] * K)

    out = extractor.extract([_result("w")], view, history, {})

    for v in out.values():
        assert math.isnan(v)


def test_f1a_t_forward_entry_missing_state_returns_all_nan():
    K = 2
    extractor = RuntimeContinuityState(
        window_k=K, descriptors=["jerk", "dir"],
        library_stats=_make_library_stats(state_dim=2),
    )
    entries = {
        "w":  _make_entry("w",  torch.zeros(1, 2), robot_state=torch.zeros(2)),
        "f1": _make_entry("f1", torch.zeros(1, 2), robot_state=torch.zeros(2)),
        "f2": _make_entry("f2", torch.zeros(1, 2), robot_state=None),       # missing
    }
    view = _MockView(entries, next_chain={"w": ["f1", "f2"]})
    history = HistoryView(actions=[], states=[torch.zeros(2)] * K)

    out = extractor.extract([_result("w")], view, history, {})

    for v in out.values():
        assert math.isnan(v)


# ------------------------------------------------------------------
# describe / orientations consistency
# ------------------------------------------------------------------


def test_f1a_a_describe_matches_extract_keys():
    extractor = RuntimeContinuityAction(
        window_k=2, descriptors=["jerk", "dir"],
        library_stats=_make_library_stats(action_dim=2),
    )
    declared = set(extractor.descriptor_orientations.keys())
    history = HistoryView(actions=[torch.zeros(2)] * 2, states=[])
    entries = {"w": _make_entry("w", torch.zeros(1, 2))}
    view = _MockView(entries)
    actual = set(extractor.extract([_result("w")], view, history, {}).keys())
    assert actual == declared


def test_f1a_t_describe_matches_extract_keys():
    K = 1
    library_stats = _make_library_stats(state_dim=2)
    extractor = RuntimeContinuityState(
        window_k=K, descriptors=["jerk", "dir", "curv_radius"],
        library_stats=library_stats,
    )
    declared = set(extractor.descriptor_orientations.keys())
    entries = {
        "w":  _make_entry("w",  torch.zeros(1, 2), robot_state=torch.zeros(2)),
        "f1": _make_entry("f1", torch.zeros(1, 2), robot_state=torch.zeros(2)),
    }
    view = _MockView(entries, next_chain={"w": ["f1"]})
    history = HistoryView(actions=[], states=[torch.zeros(2)] * K)
    actual = set(extractor.extract([_result("w")], view, history, {}).keys())
    assert actual == declared
