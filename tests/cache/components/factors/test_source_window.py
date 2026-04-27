"""Tests for F1b-A / F1b-T `extract` (online read-only) +
`compute_for_episode` (offline writer).

Algorithms per `docs/cache/verdict_factor_judge.md` §3.4:
- compute_for_episode runs the kernel over per-window slices of the
  whole episode chain
- boundary windows (out of episode) → NaN per docs §3.4 boundary rule
- F1b-T state fail-safe: empty library / zero mask / per-entry missing
  state → entire episode emits all-NaN factor dicts
- extract (OnlineExtractor) is read-only against payload.factors
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.components.factors.source_window import (
    SourceWindowSmoothnessAction,
    SourceWindowSmoothnessState,
)
from openpi.cache.storage_types import CacheEntry, CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID


# ------------------------------------------------------------------
# Mock view + fixture builders
# ------------------------------------------------------------------


class _MockView:
    def __init__(self, entries: dict[str, CacheEntry]) -> None:
        self._entries = entries

    def get(self, eid: str) -> CachePayload:
        return self._entries[eid].payload

    def get_entry(self, eid: str) -> CacheEntry:
        return self._entries[eid]

    def get_many(self, eids):
        return [self.get(e) for e in eids]


def _entry(eid, action_chunk, state=None, factors=None) -> CacheEntry:
    qk = {}
    if state is not None:
        qk["robot_state"] = state
    pl = CachePayload(action_chunk=action_chunk)
    if factors is not None:
        pl.factors = dict(factors)
    return CacheEntry(
        id=eid, checkpoint_id=CheckpointID.CP1,
        query_keys=qk, payload=pl,
    )


def _result(eid: str) -> SearchResultLite:
    return SearchResultLite(id=eid, score=1.0, checkpoint_id=CheckpointID.CP1)


def _library_stats(action_dim=2, state_dim=2) -> LibraryStats:
    return LibraryStats(
        action_sigma=torch.ones(action_dim),
        action_active_mask=torch.ones(action_dim, dtype=torch.bool),
        state_sigma=torch.ones(state_dim),
        state_active_mask=torch.ones(state_dim, dtype=torch.bool),
    )


# ------------------------------------------------------------------
# OfflineWriter — F1b-A nominal path
# ------------------------------------------------------------------


def test_compute_for_episode_returns_one_dict_per_entry():
    extractor = SourceWindowSmoothnessAction(
        windows=[(1, 1)],
        descriptors=["jerk", "dir"],
        active_eps=0.01,
    )
    # 5-entry chain, action_chunk[0] = [t, 0]
    entries = [_entry(f"e{t}", torch.tensor([[float(t), 0.0]])) for t in range(5)]
    out = extractor.compute_for_episode(entries, _library_stats())
    assert len(out) == 5
    # Each dict has 2 descriptors × 1 window = 2 keys
    for d in out:
        assert set(d.keys()) == set(extractor.descriptor_orientations.keys())


def test_compute_for_episode_nominal_descriptors_finite_at_interior():
    extractor = SourceWindowSmoothnessAction(
        windows=[(1, 1)],
        descriptors=["jerk", "dir", "curv_radius", "cum_disp"],
        active_eps=0.01,
    )
    entries = [_entry(f"e{t}", torch.tensor([[float(t), 0.0]])) for t in range(5)]
    out = extractor.compute_for_episode(entries, _library_stats())
    # Interior entry t=2 has full window; descriptors should be finite
    interior = out[2]
    assert not math.isnan(interior["f1b_a_jerk__p1_f1"])
    assert not math.isnan(interior["f1b_a_dir__p1_f1"])
    # Linear motion → jerk == 0
    assert interior["f1b_a_jerk__p1_f1"] == pytest.approx(0.0, abs=1e-6)


def test_compute_for_episode_boundary_entries_emit_nan():
    extractor = SourceWindowSmoothnessAction(
        windows=[(1, 1)],
        descriptors=["jerk"],
        active_eps=0.01,
    )
    entries = [_entry(f"e{t}", torch.tensor([[float(t), 0.0]])) for t in range(5)]
    out = extractor.compute_for_episode(entries, _library_stats())
    # First entry: window [-1, 1] pokes past start → NaN
    assert math.isnan(out[0]["f1b_a_jerk__p1_f1"])
    # Last entry: window [3, 5] pokes past end (T=5, hi=5 invalid) → NaN
    assert math.isnan(out[4]["f1b_a_jerk__p1_f1"])


def test_compute_for_episode_multiple_windows_emit_per_window_keys():
    extractor = SourceWindowSmoothnessAction(
        windows=[(0, 2), (1, 1)],
        descriptors=["dir"],
        active_eps=0.01,
    )
    entries = [_entry(f"e{t}", torch.tensor([[float(t), 0.0]])) for t in range(6)]
    out = extractor.compute_for_episode(entries, _library_stats())
    interior = out[2]
    assert "f1b_a_dir__p0_f2" in interior
    assert "f1b_a_dir__p1_f1" in interior


# ------------------------------------------------------------------
# F1b-T state fail-safe
# ------------------------------------------------------------------


def test_state_compute_for_episode_empty_library_emits_all_nan():
    extractor = SourceWindowSmoothnessState(
        windows=[(1, 1)],
        descriptors=["jerk", "dir"],
        active_eps=0.01,
    )
    # Empty state library → state_sigma.numel() == 0
    ls = LibraryStats(
        action_sigma=torch.ones(2),
        action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=torch.zeros(0),
        state_active_mask=torch.zeros(0, dtype=torch.bool),
    )
    entries = [
        _entry(f"e{t}", torch.zeros(1, 2), state=torch.tensor([float(t), 0.0]))
        for t in range(5)
    ]
    out = extractor.compute_for_episode(entries, ls)
    assert len(out) == 5
    for d in out:
        for v in d.values():
            assert math.isnan(v)


def test_state_compute_for_episode_zero_mask_emits_all_nan():
    extractor = SourceWindowSmoothnessState(
        windows=[(1, 1)],
        descriptors=["jerk"],
        active_eps=0.01,
    )
    ls = LibraryStats(
        action_sigma=torch.ones(2),
        action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=torch.ones(2),
        state_active_mask=torch.zeros(2, dtype=torch.bool),
    )
    entries = [
        _entry(f"e{t}", torch.zeros(1, 2), state=torch.tensor([float(t), 0.0]))
        for t in range(5)
    ]
    out = extractor.compute_for_episode(entries, ls)
    for d in out:
        for v in d.values():
            assert math.isnan(v)


def test_state_compute_for_episode_per_entry_missing_state_emits_all_nan():
    extractor = SourceWindowSmoothnessState(
        windows=[(1, 1)],
        descriptors=["jerk"],
        active_eps=0.01,
    )
    ls = _library_stats(state_dim=2)
    entries = [
        _entry("e0", torch.zeros(1, 2), state=torch.tensor([0.0, 0.0])),
        _entry("e1", torch.zeros(1, 2), state=torch.tensor([1.0, 0.0])),
        _entry("e2", torch.zeros(1, 2), state=None),                    # missing
        _entry("e3", torch.zeros(1, 2), state=torch.tensor([3.0, 0.0])),
    ]
    out = extractor.compute_for_episode(entries, ls)
    assert len(out) == 4
    for d in out:
        for v in d.values():
            assert math.isnan(v)


# ------------------------------------------------------------------
# numpy input bridge
# ------------------------------------------------------------------


def test_compute_for_episode_accepts_numpy_action_chunks():
    extractor = SourceWindowSmoothnessAction(
        windows=[(1, 1)],
        descriptors=["jerk"],
        active_eps=0.01,
    )
    entries = [
        _entry(f"e{t}", np.array([[float(t), 0.0]], dtype=np.float32))
        for t in range(4)
    ]
    out = extractor.compute_for_episode(entries, _library_stats())
    # Interior entry should produce finite jerk (= 0 for linear motion)
    assert out[1]["f1b_a_jerk__p1_f1"] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------
# OnlineExtractor — read from payload.factors
# ------------------------------------------------------------------


def test_extract_reads_payload_factors():
    extractor = SourceWindowSmoothnessAction(
        windows=[(1, 1)],
        descriptors=["jerk"],
        active_eps=0.01,
    )
    factors = {"f1b_a_jerk__p1_f1": 0.42}
    entries = {"w": _entry("w", torch.zeros(1, 2), factors=factors)}
    view = _MockView(entries)
    out = extractor.extract([_result("w")], view, None, {})
    assert out == {"f1b_a_jerk__p1_f1": 0.42}


def test_extract_payload_factors_none_returns_all_nan():
    extractor = SourceWindowSmoothnessAction(
        windows=[(1, 1)],
        descriptors=["jerk"],
        active_eps=0.01,
    )
    entries = {"w": _entry("w", torch.zeros(1, 2))}                     # factors=None
    view = _MockView(entries)
    out = extractor.extract([_result("w")], view, None, {})
    assert math.isnan(out["f1b_a_jerk__p1_f1"])


def test_extract_missing_single_key_returns_nan_for_that_key():
    extractor = SourceWindowSmoothnessAction(
        windows=[(1, 1)],
        descriptors=["jerk", "dir"],
        active_eps=0.01,
    )
    # Only jerk pre-computed; dir missing
    factors = {"f1b_a_jerk__p1_f1": 0.42}
    entries = {"w": _entry("w", torch.zeros(1, 2), factors=factors)}
    view = _MockView(entries)
    out = extractor.extract([_result("w")], view, None, {})
    assert out["f1b_a_jerk__p1_f1"] == 0.42
    assert math.isnan(out["f1b_a_dir__p1_f1"])


def test_extract_empty_results_returns_all_nan():
    extractor = SourceWindowSmoothnessAction(
        windows=[(1, 1)],
        descriptors=["jerk"],
        active_eps=0.01,
    )
    view = _MockView({})
    out = extractor.extract([], view, None, {})
    assert math.isnan(out["f1b_a_jerk__p1_f1"])


# ------------------------------------------------------------------
# Constructor: library_stats is now Optional (B2 plan §s2)
# ------------------------------------------------------------------


def test_constructor_accepts_optional_library_stats():
    # Offline build path constructs writers without library_stats; that
    # field is supplied per-call to compute_for_episode.
    extractor = SourceWindowSmoothnessAction(
        windows=[(1, 1)],
        descriptors=["jerk"],
        active_eps=0.01,
        library_stats=None,
    )
    assert extractor._library_stats is None
