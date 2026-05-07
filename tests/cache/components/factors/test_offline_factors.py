"""Unit tests for the 8 offline factors (B2 refactor)."""

from __future__ import annotations

import math

import pytest
import torch

from openpi.cache.components.factors.base import (
    Factor,
    FactorContext,
    HistoryView,
    LibraryStats,
)
from openpi.cache.components.factors.offline import (
    DirectionOfflineAction,
    DirectionOfflineState,
    DispersionOfflineAction,
    DispersionOfflineState,
    JerkOfflineAction,
    JerkOfflineState,
    PathLengthOfflineAction,
    PathLengthOfflineState,
)
from openpi.cache.components.factors.registry import get_class
from openpi.cache.storage_types import CacheEntry, CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_entry(
    eid: str,
    *,
    action_chunk_first: list[float] = (0.0, 0.0),
    state: list[float] | None = None,
    factors: dict[str, float] | None = None,
) -> CacheEntry:
    payload = CachePayload(
        action_chunk=torch.tensor([list(action_chunk_first)], dtype=torch.float32),
        factors=factors,
    )
    qk = {"robot_state": torch.tensor(state, dtype=torch.float32)} if state else {}
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys=qk,
        payload=payload,
        trajectory_id="traj-1",
    )


def _make_library_stats(action_dim: int = 2, state_dim: int = 2) -> LibraryStats:
    a = torch.ones(action_dim, dtype=torch.float32)
    s = torch.ones(state_dim, dtype=torch.float32)
    return LibraryStats(
        action_sigma=a, action_active_mask=torch.ones_like(a, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones_like(s, dtype=torch.bool),
    )


class _StubView:
    def __init__(self, entries: list[CacheEntry]) -> None:
        self._entries = {e.id: e for e in entries}

    def get(self, eid: str) -> CachePayload:
        return self._entries[eid].payload


def _make_ctx_for_online(entries: list[CacheEntry]) -> FactorContext:
    """Build a ctx where the first entry is the winner (used by extract())."""
    return FactorContext(
        results=[
            SearchResultLite(id=entries[0].id, score=1.0, checkpoint_id=CheckpointID.CP1)
        ],
        view=_StubView(entries),
        history=HistoryView(actions=[], states=[]),
        normalization=None,   # offline factors do not consult normalization at extract time
    )


# ----------------------------------------------------------------------
# Registry / protocol
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "jerk_offline_action", "jerk_offline_state",
        "direction_offline_action", "direction_offline_state",
        "dispersion_offline_action", "dispersion_offline_state",
        "path_length_offline_action", "path_length_offline_state",
    ],
)
def test_each_offline_factor_registered_and_protocol_conformant(name: str) -> None:
    cls = get_class(name)
    inst = cls(windows=[(1, 1)])
    assert isinstance(inst, Factor)
    assert inst.requires_chain_walk is False
    assert inst.required_top_k == 0


# ----------------------------------------------------------------------
# describe(params) classmethod
# ----------------------------------------------------------------------


def test_describe_emits_one_key_per_window_with_correct_orientation() -> None:
    out = JerkOfflineAction.describe({"windows": [(2, 3), (5, 5)]})
    assert out == {
        "jerk_offline_action__p2_f3": "risky",
        "jerk_offline_action__p5_f5": "risky",
    }


def test_describe_path_length_state_orientation() -> None:
    out = PathLengthOfflineState.describe({"windows": [(0, 5)]})
    assert out == {"path_length_offline_state__p0_f5": "non_monotonic"}


# ----------------------------------------------------------------------
# Online surface — read from winner.payload.factors
# ----------------------------------------------------------------------


def test_offline_extract_reads_factors_dict_for_keys() -> None:
    factors = {
        "jerk_offline_action__p1_f1": 0.42,
        "jerk_offline_action__p2_f2": 0.84,
    }
    e0 = _make_entry("e0", factors=factors)
    ctx = _make_ctx_for_online([e0])
    f = JerkOfflineAction(windows=[(1, 1), (2, 2)])
    out = f.extract(ctx)
    assert out == {
        "jerk_offline_action__p1_f1": 0.42,
        "jerk_offline_action__p2_f2": 0.84,
    }


def test_offline_extract_missing_factors_dict_emits_all_nan() -> None:
    e0 = _make_entry("e0", factors=None)
    ctx = _make_ctx_for_online([e0])
    f = JerkOfflineAction(windows=[(1, 1)])
    out = f.extract(ctx)
    assert math.isnan(out["jerk_offline_action__p1_f1"])


def test_offline_extract_missing_specific_key_emits_nan() -> None:
    e0 = _make_entry("e0", factors={"jerk_offline_action__p1_f1": 0.5})
    ctx = _make_ctx_for_online([e0])
    f = JerkOfflineAction(windows=[(1, 1), (3, 3)])
    out = f.extract(ctx)
    assert out["jerk_offline_action__p1_f1"] == 0.5
    assert math.isnan(out["jerk_offline_action__p3_f3"])


def test_offline_extract_empty_results_emits_all_nan() -> None:
    e0 = _make_entry("e0", factors={"jerk_offline_action__p1_f1": 0.5})
    ctx = FactorContext(
        results=[],
        view=_StubView([e0]),
        history=HistoryView(actions=[], states=[]),
        normalization=None,
    )
    f = JerkOfflineAction(windows=[(1, 1)])
    out = f.extract(ctx)
    assert math.isnan(out["jerk_offline_action__p1_f1"])


# ----------------------------------------------------------------------
# OfflineWriter surface — compute_for_episode
# ----------------------------------------------------------------------


def test_compute_for_episode_action_channel_emits_per_entry_dict() -> None:
    """6-entry chain, window (1, 1) — entries 1..4 should be finite (window
    fits), entries 0 and 5 are boundary → NaN."""
    entries = [
        _make_entry(f"e{i}", action_chunk_first=[float(i), float(i * 2)])
        for i in range(6)
    ]
    f = JerkOfflineAction(windows=[(1, 1)])
    rows = f.compute_for_episode(entries, _make_library_stats())
    assert len(rows) == 6
    # Boundary
    assert math.isnan(rows[0]["jerk_offline_action__p1_f1"])
    assert math.isnan(rows[5]["jerk_offline_action__p1_f1"])
    # Interior — finite (could be 0 since this chain is linear)
    for i in range(1, 5):
        val = rows[i]["jerk_offline_action__p1_f1"]
        assert math.isfinite(val)


def test_compute_for_episode_state_channel_bails_on_missing_state() -> None:
    entries = [
        _make_entry("e0", state=[0.0, 0.0]),
        _make_entry("e1"),                    # missing robot_state
        _make_entry("e2", state=[2.0, 4.0]),
    ]
    f = JerkOfflineState(windows=[(1, 1)])
    rows = f.compute_for_episode(entries, _make_library_stats())
    assert len(rows) == 3
    for row in rows:
        for k, v in row.items():
            assert math.isnan(v), f"{k} expected NaN but got {v}"


def test_compute_for_episode_empty_entries_returns_empty_list() -> None:
    f = JerkOfflineAction(windows=[(1, 1)])
    assert f.compute_for_episode([], _make_library_stats()) == []


def test_compute_for_episode_empty_state_library_emits_all_nan() -> None:
    entries = [
        _make_entry(f"e{i}", state=[float(i), float(i * 2)])
        for i in range(4)
    ]
    # state library stats are zero-length (e.g. backend without state)
    a = torch.ones(2, dtype=torch.float32)
    library_stats = LibraryStats(
        action_sigma=a,
        action_active_mask=torch.ones_like(a, dtype=torch.bool),
        state_sigma=torch.zeros(0, dtype=torch.float32),
        state_active_mask=torch.zeros(0, dtype=torch.bool),
    )
    f = JerkOfflineState(windows=[(1, 1)])
    rows = f.compute_for_episode(entries, library_stats)
    for row in rows:
        for v in row.values():
            assert math.isnan(v)


def test_compute_for_episode_zero_active_mask_emits_all_nan() -> None:
    entries = [
        _make_entry(f"e{i}", action_chunk_first=[float(i), float(i * 2)])
        for i in range(4)
    ]
    a = torch.ones(2, dtype=torch.float32)
    library_stats = LibraryStats(
        action_sigma=a, action_active_mask=torch.zeros_like(a, dtype=torch.bool),
        state_sigma=torch.ones(2), state_active_mask=torch.ones(2, dtype=torch.bool),
    )
    f = JerkOfflineAction(windows=[(1, 1)])
    rows = f.compute_for_episode(entries, library_stats)
    for row in rows:
        for v in row.values():
            assert math.isnan(v)


def test_compute_for_episode_multi_window() -> None:
    entries = [
        _make_entry(f"e{i}", action_chunk_first=[float(i), 0.0])
        for i in range(8)
    ]
    f = JerkOfflineAction(windows=[(1, 1), (3, 3)])
    rows = f.compute_for_episode(entries, _make_library_stats())
    assert len(rows) == 8
    for i, row in enumerate(rows):
        assert set(row.keys()) == {
            "jerk_offline_action__p1_f1",
            "jerk_offline_action__p3_f3",
        }


def test_required_payload_fields_default_empty() -> None:
    assert JerkOfflineAction(windows=[(1, 1)]).required_payload_fields() == set()
    assert PathLengthOfflineState(windows=[(1, 1)]).required_payload_fields() == set()
