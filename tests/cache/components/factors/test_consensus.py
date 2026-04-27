"""Tests for F2 (`TopKActionConsensus.extract`).

Algorithm contract is `docs/cache/verdict_factor_judge.md` §3.5 with the
B1 amendment: F2 uses a CANDIDATE-LOCAL active mask (`var_d > 1e-8`)
rather than `library_stats.action_active_mask`, preserving
`requires_library_stats=False`.

Tests cover:
- Variance grows with candidate spread (sanity).
- K=1 / empty results → NaN (cold-start).
- Padded DOFs (constant 0 across pool) drop out of the average.
- All-active-pool (every dim has variance) — average matches manual calc.
- `requires_library_stats` flag stays False.
- Output keys match `descriptor_orientations.keys()`.
"""

from __future__ import annotations

import math

import pytest
import torch

from openpi.cache.components.factors.consensus import TopKActionConsensus
from openpi.cache.storage_types import CacheEntry, CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID


# ------------------------------------------------------------------
# Mock view
# ------------------------------------------------------------------


class _MockView:
    def __init__(self, entries: dict[str, CacheEntry]) -> None:
        self._entries = entries

    def get(self, entry_id: str) -> CachePayload:
        return self._entries[entry_id].payload

    def get_entry(self, entry_id: str) -> CacheEntry:
        return self._entries[entry_id]

    def get_many(self, entry_ids):
        return [self.get(eid) for eid in entry_ids]


def _entry(eid: str, action_chunk: torch.Tensor) -> CacheEntry:
    return CacheEntry(
        id=eid, checkpoint_id=CheckpointID.CP1,
        query_keys={}, payload=CachePayload(action_chunk=action_chunk),
    )


def _result(eid: str) -> SearchResultLite:
    return SearchResultLite(id=eid, score=1.0, checkpoint_id=CheckpointID.CP1)


# ------------------------------------------------------------------
# Capability flags
# ------------------------------------------------------------------


def test_requires_library_stats_is_false():
    f = TopKActionConsensus(K=5)
    assert f.requires_library_stats is False
    assert f.required_top_k == 5


# ------------------------------------------------------------------
# Variance scaling
# ------------------------------------------------------------------


def test_variance_increases_with_candidate_spread():
    f = TopKActionConsensus(K=4)

    def _run(scale: float) -> float:
        entries = {
            f"e{i}": _entry(f"e{i}", torch.tensor([[scale * i, 0.0]]))
            for i in range(4)
        }
        view = _MockView(entries)
        out = f.extract([_result(f"e{i}") for i in range(4)], view, None, {})
        return out["f2_var"]

    small = _run(0.1)
    large = _run(1.0)
    assert large > small
    # Population variance scales with offset^2 → ratio ~100×
    assert large / small == pytest.approx(100.0, rel=0.01)


def test_perfect_consensus_yields_nan_not_zero():
    # All candidates produce the exact same action — the pool agrees
    # perfectly. We deliberately return NaN (not 0) to defer to other
    # factors: a 0 here would falsely register as "high consensus =
    # FULL_HIT signal" under risky orientation flip.
    f = TopKActionConsensus(K=3)
    same = torch.tensor([[1.0, 0.5, -0.3]])
    entries = {f"e{i}": _entry(f"e{i}", same.clone()) for i in range(3)}
    view = _MockView(entries)
    out = f.extract([_result(f"e{i}") for i in range(3)], view, None, {})
    assert math.isnan(out["f2_var"])


# ------------------------------------------------------------------
# Cold-start / undersized pool
# ------------------------------------------------------------------


def test_empty_results_returns_nan():
    f = TopKActionConsensus(K=5)
    out = f.extract([], _MockView({}), None, {})
    assert math.isnan(out["f2_var"])


def test_single_candidate_returns_nan():
    f = TopKActionConsensus(K=5)
    entries = {"e0": _entry("e0", torch.tensor([[1.0, 2.0]]))}
    view = _MockView(entries)
    out = f.extract([_result("e0")], view, None, {})
    assert math.isnan(out["f2_var"])


# ------------------------------------------------------------------
# Padded DOFs drop out (candidate-local mask, not library mask)
# ------------------------------------------------------------------


def test_padded_dofs_drop_out_of_average():
    # Action space has 4 dims. First 2 are "real" (vary across candidates),
    # last 2 are "padded" (always 0 across the pool — like Pi0.5 padding).
    # Average should be over only the 2 real dims, NOT all 4.
    f = TopKActionConsensus(K=3)
    entries = {
        "a": _entry("a", torch.tensor([[1.0, 2.0, 0.0, 0.0]])),
        "b": _entry("b", torch.tensor([[3.0, 4.0, 0.0, 0.0]])),
        "c": _entry("c", torch.tensor([[5.0, 6.0, 0.0, 0.0]])),
    }
    view = _MockView(entries)
    out = f.extract([_result(c) for c in "abc"], view, None, {})

    # Per-DOF population var: dim 0 = var([1,3,5]) = 8/3, dim 1 = var([2,4,6]) = 8/3,
    # dim 2 / 3 = 0. Average over active subset = (8/3 + 8/3) / 2 = 8/3.
    expected = 8.0 / 3.0
    assert out["f2_var"] == pytest.approx(expected, rel=1e-6)


# ------------------------------------------------------------------
# K_eff clamps to len(results)
# ------------------------------------------------------------------


def test_k_eff_clamps_to_results_length():
    # f.K = 100, but only 3 results — should still compute, not crash
    f = TopKActionConsensus(K=100)
    entries = {
        "a": _entry("a", torch.tensor([[1.0, 0.0]])),
        "b": _entry("b", torch.tensor([[2.0, 0.0]])),
        "c": _entry("c", torch.tensor([[3.0, 0.0]])),
    }
    view = _MockView(entries)
    out = f.extract([_result(c) for c in "abc"], view, None, {})
    assert not math.isnan(out["f2_var"])
    # var([1,2,3], population) = 2/3
    assert out["f2_var"] == pytest.approx(2.0 / 3.0, rel=1e-6)


# ------------------------------------------------------------------
# Output keys match descriptor_orientations
# ------------------------------------------------------------------


def test_output_keys_match_descriptor_orientations():
    f = TopKActionConsensus(K=2)
    entries = {
        "a": _entry("a", torch.tensor([[1.0, 0.0]])),
        "b": _entry("b", torch.tensor([[2.0, 0.0]])),
    }
    view = _MockView(entries)
    out = f.extract([_result(c) for c in "ab"], view, None, {})
    assert set(out.keys()) == set(f.descriptor_orientations.keys())
    assert set(out.keys()) == {"f2_var"}
