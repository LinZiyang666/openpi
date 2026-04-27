"""Tests for `LibraryStats.compute_from_entries` (B2).

Covers:
- per-DOF sigma values match the np.std baseline
- active mask threshold cuts at active_eps
- state-missing entry pool yields zero-length placeholder sigma + mask
- numpy and torch input both flow through (`torch.as_tensor` bridge)
- empty entries → all zero-length placeholders
- HistoryView dataclass field semantics (lightweight sanity)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openpi.cache.components.factors.base import HistoryView, LibraryStats
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


# ------------------------------------------------------------------
# Fixture helpers
# ------------------------------------------------------------------


def _entry(eid, action_chunk, state=None) -> CacheEntry:
    qk = {}
    if state is not None:
        qk["robot_state"] = state
    return CacheEntry(
        id=eid, checkpoint_id=CheckpointID.CP1,
        query_keys=qk, payload=CachePayload(action_chunk=action_chunk),
    )


# ------------------------------------------------------------------
# Numerical correctness — action side
# ------------------------------------------------------------------


def test_action_sigma_matches_population_std():
    # 3 entries, action_chunk[0] = [a, b]; per-DOF std (unbiased=False) is
    # sqrt(mean((x - mean)^2)).
    entries = [
        _entry("e0", torch.tensor([[1.0, 2.0]])),
        _entry("e1", torch.tensor([[2.0, 4.0]])),
        _entry("e2", torch.tensor([[3.0, 6.0]])),
    ]
    ls = LibraryStats.compute_from_entries(entries)
    expected = torch.tensor([1.0, 2.0]).std(unbiased=False)
    # dim 0: std of [1,2,3] = sqrt(2/3); dim 1: std of [2,4,6] = sqrt(8/3)
    assert ls.action_sigma[0] == pytest.approx((2.0 / 3.0) ** 0.5, rel=1e-6)
    assert ls.action_sigma[1] == pytest.approx((8.0 / 3.0) ** 0.5, rel=1e-6)


def test_action_active_mask_threshold():
    # dim 0 has stddev ~0.816 (above eps 0.01), dim 1 is constant 0
    entries = [
        _entry("e0", torch.tensor([[1.0, 0.0]])),
        _entry("e1", torch.tensor([[2.0, 0.0]])),
        _entry("e2", torch.tensor([[3.0, 0.0]])),
    ]
    ls = LibraryStats.compute_from_entries(entries, active_eps_action=0.01)
    assert ls.action_active_mask[0].item() is True
    assert ls.action_active_mask[1].item() is False


def test_action_active_mask_eps_tunable():
    entries = [
        _entry("e0", torch.tensor([[0.5]])),
        _entry("e1", torch.tensor([[0.51]])),
    ]
    # std of [0.5, 0.51] = 0.005 → below default eps 0.01 → inactive
    ls_default = LibraryStats.compute_from_entries(entries)
    assert ls_default.action_active_mask[0].item() is False
    # Lower eps → still active
    ls_lenient = LibraryStats.compute_from_entries(entries, active_eps_action=1e-4)
    assert ls_lenient.action_active_mask[0].item() is True


# ------------------------------------------------------------------
# State side: present and absent
# ------------------------------------------------------------------


def test_state_sigma_when_robot_state_present():
    entries = [
        _entry("e0", torch.zeros(1, 2), state=torch.tensor([1.0, 0.0])),
        _entry("e1", torch.zeros(1, 2), state=torch.tensor([2.0, 0.0])),
    ]
    ls = LibraryStats.compute_from_entries(entries)
    # dim 0: std of [1,2] = 0.5; dim 1: 0
    assert ls.state_sigma.shape == (2,)
    assert ls.state_active_mask[0].item() is True
    assert ls.state_active_mask[1].item() is False


def test_state_missing_returns_zero_length_placeholder():
    # No entry carries robot_state — zero-length placeholders so down-stream
    # state-side factors see `numel() == 0` and bail out via the guard.
    entries = [
        _entry("e0", torch.zeros(1, 2)),
        _entry("e1", torch.zeros(1, 2)),
    ]
    ls = LibraryStats.compute_from_entries(entries)
    assert ls.state_sigma.numel() == 0
    assert ls.state_active_mask.numel() == 0
    assert ls.state_sigma.dtype == torch.float32
    assert ls.state_active_mask.dtype == torch.bool


def test_state_partial_present_uses_only_present_entries():
    # 2 of 3 entries carry state — sigma is computed over the 2 present.
    entries = [
        _entry("e0", torch.zeros(1, 1), state=torch.tensor([1.0])),
        _entry("e1", torch.zeros(1, 1)),
        _entry("e2", torch.zeros(1, 1), state=torch.tensor([3.0])),
    ]
    ls = LibraryStats.compute_from_entries(entries)
    # std of [1, 3] population = 1.0
    assert ls.state_sigma[0].item() == pytest.approx(1.0, rel=1e-6)


# ------------------------------------------------------------------
# numpy input bridge (post-detach lifecycle, B2 plan §4.5)
# ------------------------------------------------------------------


def test_numpy_action_chunk_is_accepted():
    # Post-detach entries (built by build_in_memory_cache_artifact subprocess
    # path) carry numpy arrays. compute_from_entries must tolerate both.
    entries = [
        _entry("e0", np.array([[1.0, 2.0]], dtype=np.float32)),
        _entry("e1", np.array([[3.0, 4.0]], dtype=np.float32)),
    ]
    ls = LibraryStats.compute_from_entries(entries)
    assert ls.action_sigma.shape == (2,)
    assert ls.action_sigma[0].item() == pytest.approx(1.0, rel=1e-6)


def test_numpy_robot_state_is_accepted():
    entries = [
        _entry("e0", np.array([[0.0]], dtype=np.float32),
               state=np.array([1.0, 0.0], dtype=np.float32)),
        _entry("e1", np.array([[0.0]], dtype=np.float32),
               state=np.array([3.0, 0.0], dtype=np.float32)),
    ]
    ls = LibraryStats.compute_from_entries(entries)
    assert ls.state_sigma[0].item() == pytest.approx(1.0, rel=1e-6)


# ------------------------------------------------------------------
# Empty entries
# ------------------------------------------------------------------


def test_empty_entries_returns_all_zero_placeholders():
    ls = LibraryStats.compute_from_entries([])
    assert ls.action_sigma.numel() == 0
    assert ls.action_active_mask.numel() == 0
    assert ls.state_sigma.numel() == 0
    assert ls.state_active_mask.numel() == 0


# ------------------------------------------------------------------
# HistoryView field semantics (lightweight sanity)
# ------------------------------------------------------------------


def test_history_view_holds_action_and_state_lists():
    h = HistoryView(actions=[torch.zeros(2)], states=[torch.zeros(3)])
    assert len(h.actions) == 1
    assert len(h.states) == 1


def test_history_view_can_be_empty():
    h = HistoryView(actions=[], states=[])
    assert h.actions == []
    assert h.states == []
