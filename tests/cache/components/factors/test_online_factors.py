"""Unit tests for the 8 online factors (B2 refactor)."""

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
from openpi.cache.components.factors.normalization import ZScoreNormalization
from openpi.cache.components.factors.online import (
    DirectionOnlineAction,
    DirectionOnlineState,
    DispersionOnlineAction,
    DispersionOnlineState,
    JerkOnlineAction,
    JerkOnlineState,
    PathLengthOnlineAction,
    PathLengthOnlineState,
    _normalize_windows,
)
from openpi.cache.components.factors.registry import get_class
from openpi.cache.storage_types import CacheEntry, CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID


# ----------------------------------------------------------------------
# Stub PayloadView for tests
# ----------------------------------------------------------------------


class _StubView:
    """Minimal PayloadView stub: holds a chain (id -> entry) and supports
    walk_next within the chain. ``forks`` lets a test toggle a fake fork
    error to exercise the bail-out path."""

    def __init__(self, entries: list[CacheEntry], *, fork_at: str | None = None) -> None:
        self._entries = {e.id: e for e in entries}
        self._chain = [e.id for e in entries]
        self._fork_at = fork_at

    def get(self, entry_id: str) -> CachePayload:
        return self._entries[entry_id].payload

    def get_entry(self, entry_id: str) -> CacheEntry:
        return self._entries[entry_id]

    def get_many(self, entry_ids: list[str]) -> list[CachePayload]:
        return [self.get(i) for i in entry_ids]

    def walk_next(self, entry_id: str, k: int) -> list[CacheEntry]:
        if self._fork_at == entry_id and k > 0:
            raise NotImplementedError("test fork")
        idx = self._chain.index(entry_id)
        return [self._entries[i] for i in self._chain[idx + 1 : idx + 1 + k]]


def _make_entry(eid: str, action_chunk_first: list[float], state: list[float] | None = None) -> CacheEntry:
    payload = CachePayload(
        action_chunk=torch.tensor([action_chunk_first], dtype=torch.float32),
    )
    qk = {"robot_state": torch.tensor(state, dtype=torch.float32)} if state else {}
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys=qk,
        payload=payload,
        trajectory_id="traj-1",
        prev_ids=[],
        next_ids=[],
    )


def _make_lib_stats(action_dim: int = 2, state_dim: int = 2) -> LibraryStats:
    a = torch.ones(action_dim, dtype=torch.float32)
    s = torch.ones(state_dim, dtype=torch.float32)
    return LibraryStats(
        action_sigma=a, action_active_mask=torch.ones_like(a, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones_like(s, dtype=torch.bool),
    )


def _make_ctx(
    *,
    chain: list[CacheEntry],
    history_actions: list[list[float]] | None = None,
    history_states: list[list[float]] | None = None,
    fork_at: str | None = None,
    library_stats: LibraryStats | None = None,
) -> FactorContext:
    if library_stats is None:
        library_stats = _make_lib_stats()
    norm = ZScoreNormalization(library_stats)
    history = HistoryView(
        actions=[torch.tensor(a, dtype=torch.float32) for a in (history_actions or [])],
        states=[torch.tensor(s, dtype=torch.float32) for s in (history_states or [])],
    )
    view = _StubView(chain, fork_at=fork_at)
    results = [SearchResultLite(id=chain[0].id, score=1.0, checkpoint_id=CheckpointID.CP1)]
    return FactorContext(results=results, view=view, history=history, normalization=norm)


# ----------------------------------------------------------------------
# Window normalization helper
# ----------------------------------------------------------------------


def test_normalize_windows_dict_form() -> None:
    out = _normalize_windows([{"past": 2, "future": 3}])
    assert out == [(2, 3)]


def test_normalize_windows_tuple_form() -> None:
    out = _normalize_windows([(1, 1), (3, 5)])
    assert out == [(1, 1), (3, 5)]


# ----------------------------------------------------------------------
# Factor protocol conformance + registry presence
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "jerk_online_action", "jerk_online_state",
        "direction_online_action", "direction_online_state",
        "dispersion_online_action", "dispersion_online_state",
        "path_length_online_action", "path_length_online_state",
    ],
)
def test_each_online_factor_registered_and_protocol_conformant(name: str) -> None:
    cls = get_class(name)
    inst = cls(windows=[(1, 1)])
    assert isinstance(inst, Factor)
    assert inst.requires_chain_walk is True
    assert inst.required_top_k == 0


# ----------------------------------------------------------------------
# describe(params) classmethod
# ----------------------------------------------------------------------


def test_describe_emits_one_key_per_window_with_correct_orientation() -> None:
    out = JerkOnlineAction.describe({"windows": [(2, 3), (5, 5)]})
    assert out == {
        "jerk_online_action__p2_f3": "risky",
        "jerk_online_action__p5_f5": "risky",
    }


def test_describe_direction_state_orientation_safe() -> None:
    out = DirectionOnlineState.describe({"windows": [(0, 3)]})
    assert out == {"direction_online_state__p0_f3": "safe"}


def test_describe_dispersion_orientation_non_monotonic() -> None:
    out = DispersionOnlineAction.describe({"windows": [(1, 1)]})
    assert out == {"dispersion_online_action__p1_f1": "non_monotonic"}


def test_describe_path_length_orientation_non_monotonic() -> None:
    out = PathLengthOnlineState.describe({"windows": [(2, 2)]})
    assert out == {"path_length_online_state__p2_f2": "non_monotonic"}


def test_describe_matches_instance_orientations() -> None:
    inst = JerkOnlineAction(windows=[(1, 2), (3, 4)])
    assert inst.descriptor_orientations == JerkOnlineAction.describe(
        {"windows": [(1, 2), (3, 4)]}
    )


# ----------------------------------------------------------------------
# Action channel extract — happy path
# ----------------------------------------------------------------------


def test_jerk_online_action_happy_path_emits_finite_value() -> None:
    """3-step splice with non-zero jerk: P=2, F=2 → splice length 5."""
    chain = [
        _make_entry("e0", [0.0, 0.0]),  # winner
        _make_entry("e1", [1.0, 1.0]),  # walk_next 1
        _make_entry("e2", [2.0, 4.0]),  # walk_next 2
    ]
    ctx = _make_ctx(
        chain=chain,
        history_actions=[[0.0, 0.0], [0.0, 0.0]],   # P=2 history (zero motion)
    )
    f = JerkOnlineAction(windows=[(2, 2)])
    out = f.extract(ctx)
    assert "jerk_online_action__p2_f2" in out
    val = out["jerk_online_action__p2_f2"]
    assert math.isfinite(val) and val >= 0


def test_direction_online_action_emits_finite() -> None:
    chain = [
        _make_entry("e0", [0.0, 0.0]),
        _make_entry("e1", [1.0, 0.0]),
        _make_entry("e2", [2.0, 0.0]),
    ]
    ctx = _make_ctx(
        chain=chain,
        history_actions=[[-2.0, 0.0], [-1.0, 0.0]],
    )
    f = DirectionOnlineAction(windows=[(2, 2)])
    out = f.extract(ctx)
    val = out["direction_online_action__p2_f2"]
    # All velocities point along +x → cosines all 1.
    assert math.isclose(val, 1.0, abs_tol=1e-5)


# ----------------------------------------------------------------------
# State channel extract — happy path
# ----------------------------------------------------------------------


def test_jerk_online_state_happy_path() -> None:
    chain = [
        _make_entry("e0", [0.0, 0.0], state=[0.0, 0.0]),  # winner
        _make_entry("e1", [0.0, 0.0], state=[1.0, 1.0]),
        _make_entry("e2", [0.0, 0.0], state=[2.0, 4.0]),
    ]
    ctx = _make_ctx(
        chain=chain,
        history_states=[[0.0, 0.0], [0.0, 0.0]],
    )
    f = JerkOnlineState(windows=[(2, 2)])
    out = f.extract(ctx)
    val = out["jerk_online_state__p2_f2"]
    assert math.isfinite(val) and val >= 0


# ----------------------------------------------------------------------
# Boundary handling
# ----------------------------------------------------------------------


def test_history_too_short_emits_nan() -> None:
    chain = [_make_entry("e0", [0.0, 0.0]), _make_entry("e1", [1.0, 1.0])]
    ctx = _make_ctx(
        chain=chain,
        history_actions=[[0.0, 0.0]],   # only 1, but P=3
    )
    f = JerkOnlineAction(windows=[(3, 1)])
    out = f.extract(ctx)
    assert math.isnan(out["jerk_online_action__p3_f1"])


def test_walk_next_runs_out_emits_nan() -> None:
    chain = [_make_entry("e0", [0.0, 0.0])]   # only winner, no downstream
    ctx = _make_ctx(
        chain=chain,
        history_actions=[[0.0, 0.0], [0.0, 0.0]],
    )
    f = JerkOnlineAction(windows=[(2, 2)])
    out = f.extract(ctx)
    assert math.isnan(out["jerk_online_action__p2_f2"])


def test_fork_emits_nan() -> None:
    chain = [
        _make_entry("e0", [0.0, 0.0]),
        _make_entry("e1", [1.0, 1.0]),
        _make_entry("e2", [2.0, 4.0]),
    ]
    ctx = _make_ctx(
        chain=chain,
        history_actions=[[0.0, 0.0], [0.0, 0.0]],
        fork_at="e0",
    )
    f = JerkOnlineAction(windows=[(2, 2)])
    out = f.extract(ctx)
    assert math.isnan(out["jerk_online_action__p2_f2"])


def test_state_missing_robot_state_emits_nan() -> None:
    # Winner has no robot_state → bail.
    chain = [
        _make_entry("e0", [0.0, 0.0]),                                # no state
        _make_entry("e1", [0.0, 0.0], state=[1.0, 1.0]),
        _make_entry("e2", [0.0, 0.0], state=[2.0, 4.0]),
    ]
    ctx = _make_ctx(
        chain=chain,
        history_states=[[0.0, 0.0], [0.0, 0.0]],
    )
    f = JerkOnlineState(windows=[(2, 2)])
    out = f.extract(ctx)
    assert math.isnan(out["jerk_online_state__p2_f2"])


def test_empty_results_emits_all_nan() -> None:
    chain = [_make_entry("e0", [0.0, 0.0])]
    ctx = _make_ctx(chain=chain, history_actions=[[0.0, 0.0], [0.0, 0.0]])
    ctx = FactorContext(
        results=[],
        view=ctx.view,
        history=ctx.history,
        normalization=ctx.normalization,
    )
    f = JerkOnlineAction(windows=[(1, 1), (2, 2)])
    out = f.extract(ctx)
    assert all(math.isnan(v) for v in out.values())
    assert set(out.keys()) == {
        "jerk_online_action__p1_f1",
        "jerk_online_action__p2_f2",
    }


# ----------------------------------------------------------------------
# Multi-window
# ----------------------------------------------------------------------


def test_multi_window_emits_one_key_per_window() -> None:
    chain = [_make_entry(f"e{i}", [float(i), float(i * 2)]) for i in range(6)]
    ctx = _make_ctx(
        chain=chain,
        history_actions=[[0.0, 0.0]] * 5,
    )
    f = JerkOnlineAction(windows=[(1, 1), (2, 2), (3, 1)])
    out = f.extract(ctx)
    assert set(out.keys()) == {
        "jerk_online_action__p1_f1",
        "jerk_online_action__p2_f2",
        "jerk_online_action__p3_f1",
    }


# ----------------------------------------------------------------------
# Construction validation
# ----------------------------------------------------------------------


def test_empty_windows_rejected() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        JerkOnlineAction(windows=[])
