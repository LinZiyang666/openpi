"""Tests for B1 Orchestrator history wiring + on_task_end leak fix.

Covers:
- anchor checkpoint selection (CP1-only / CP3-only / CP1+CP3 → CP1 wins)
- state append happens once per inference cycle on the anchor CP only
- state append fires on every path: gate-skip / FULL_HIT / WARM_START / MISS
- _action_history grows on broadcast_action
- _reset_episode_buffer clears both histories
- on_task_end clears buffers (was a leak before B1)
"""

from __future__ import annotations

import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import AlwaysSearchGate, AlwaysSkipGate
from openpi.cache.components.judge import AlwaysHitJudge, ThresholdJudge
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID

from .conftest import TestStorageSearchStrategy, insert_entry, make_stage1


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_orch(checkpoint_ids: list[CheckpointID], judge=None) -> CacheOrchestrator:
    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    kb = PlaceholderKeyBuilder()
    g = AlwaysSearchGate()
    j = judge if judge is not None else ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)
    strat = TestStorageSearchStrategy(storage, top_k=1)
    gates = {cp: g for cp in checkpoint_ids}
    judges = {cp: j for cp in checkpoint_ids}
    strats = {cp: strat for cp in checkpoint_ids}
    return CacheOrchestrator(
        storage, kb,
        gates=gates, judges=judges, search_strategies=strats,
        timer=SystemTimer(enabled=False),
    )


def _state(seed: float) -> torch.Tensor:
    return torch.full((1, 32), seed)


# ------------------------------------------------------------------
# Anchor checkpoint selection
# ------------------------------------------------------------------


def test_anchor_cp_is_cp1_when_cp1_enabled():
    o = _make_orch([CheckpointID.CP1, CheckpointID.CP3])
    assert o._state_history_anchor_cp == CheckpointID.CP1


def test_anchor_cp_is_cp3_when_only_cp3_enabled():
    o = _make_orch([CheckpointID.CP3])
    assert o._state_history_anchor_cp == CheckpointID.CP3


def test_anchor_cp_is_cp1_when_only_cp1_enabled():
    o = _make_orch([CheckpointID.CP1])
    assert o._state_history_anchor_cp == CheckpointID.CP1


# ------------------------------------------------------------------
# State history append rules
# ------------------------------------------------------------------


def test_state_appended_once_per_cycle_on_anchor():
    o = _make_orch([CheckpointID.CP1, CheckpointID.CP3])
    state = _state(0.5)
    # Call CP1 (anchor) and CP3 in the same inference cycle
    o.check(CheckpointID.CP1, stage1=make_stage1(state))
    o.check(CheckpointID.CP3, stage1=make_stage1(state))
    # Anchor (CP1) appended once; CP3 did NOT append
    assert len(o._state_history) == 1
    o.clear()


def test_state_appended_on_gate_skip_path():
    # AlwaysSkipGate → orchestrator returns MISS without searching.
    # State must STILL be appended on the anchor CP for gap-free history.
    o = _make_orch(
        [CheckpointID.CP1],
        judge=ThresholdJudge(cp1_threshold=0.5, cp3_threshold=0.5),
    )
    o._gates[CheckpointID.CP1] = AlwaysSkipGate()
    o.check(CheckpointID.CP1, stage1=make_stage1(_state(0.1)))
    o.check(CheckpointID.CP1, stage1=make_stage1(_state(0.2)))
    assert len(o._state_history) == 2


def test_state_appended_on_full_hit_path():
    o = _make_orch([CheckpointID.CP1], judge=AlwaysHitJudge())
    # Pre-insert one entry so search returns it
    from openpi.cache.storage_types import CachePayload
    payload = CachePayload(action_chunk=torch.zeros(50, 32))
    insert_entry(o._storage, CheckpointID.CP1, _state(0.1), payload)

    o.on_task_begin()
    o.check(CheckpointID.CP1, stage1=make_stage1(_state(0.1)))
    assert len(o._state_history) == 1


# ------------------------------------------------------------------
# Action history append
# ------------------------------------------------------------------


def test_broadcast_action_appends_to_action_history():
    o = _make_orch([CheckpointID.CP1])
    a1 = torch.zeros(50, 32)
    a2 = torch.ones(50, 32)
    o.broadcast_action(a1)
    o.broadcast_action(a2)
    assert len(o._action_history) == 2
    # Tensors are detached + on cpu (no autograd state retained)
    assert o._action_history[0].device.type == "cpu"


def test_broadcast_action_reduces_chunk_to_first_action():
    """F1a-A's `_build_action_splice` and the unit-test contract in
    test_runtime_continuity.py both expect history.actions[-K:] to be a
    list of single-action [A]-shaped tensors (mirroring _state_history's
    per-step robot_state). Pi05 broadcasts the full [chunk_len, A] chunk
    per inference, so the orchestrator must reduce to action_chunk[0]
    before appending. Regression for the bug where the entire chunk was
    appended verbatim, causing torch.stack in _build_action_splice to
    raise on shape mismatch and DumpingJudge to swallow the error into
    100% NaN for all f1a_a_* keys.
    """
    o = _make_orch([CheckpointID.CP1])
    o.broadcast_action(torch.zeros(50, 32))
    o.broadcast_action(torch.ones(50, 32))
    assert o._action_history[0].shape == (32,)
    assert o._action_history[1].shape == (32,)
    # Verify contents — first row of each chunk, not the whole chunk.
    assert torch.equal(o._action_history[0], torch.zeros(32))
    assert torch.equal(o._action_history[1], torch.ones(32))


def test_broadcast_action_passes_through_already_1d_action():
    """Defensive: if a caller (legacy or test) hands a 1-D [A] tensor in,
    the orchestrator should not index into a non-existent leading dim."""
    o = _make_orch([CheckpointID.CP1])
    o.broadcast_action(torch.full((32,), 7.0))
    assert o._action_history[0].shape == (32,)
    assert torch.equal(o._action_history[0], torch.full((32,), 7.0))


# ------------------------------------------------------------------
# Lifecycle: reset clears histories
# ------------------------------------------------------------------


def test_reset_episode_buffer_clears_histories():
    o = _make_orch([CheckpointID.CP1])
    o.broadcast_action(torch.zeros(50, 32))
    o.check(CheckpointID.CP1, stage1=make_stage1(_state(0.1)))
    assert o._action_history and o._state_history

    o._reset_episode_buffer()

    assert not o._action_history
    assert not o._state_history


def test_on_episode_start_clears_histories():
    o = _make_orch([CheckpointID.CP1])
    o.broadcast_action(torch.zeros(50, 32))
    o.check(CheckpointID.CP1, stage1=make_stage1(_state(0.1)))

    o.on_episode_start()

    assert not o._action_history
    assert not o._state_history


# ------------------------------------------------------------------
# on_task_end leak fix
# ------------------------------------------------------------------


def test_on_task_end_clears_history_buffers():
    # Pre-B1 behavior: on_task_end only closed search sessions but left
    # episode_steps / miss_by_checkpoint live, leaking across reconnects.
    # B1 fix: also call _reset_episode_buffer so action / state history
    # don't bleed into the next connection.
    o = _make_orch([CheckpointID.CP1])
    o.broadcast_action(torch.zeros(50, 32))
    o.check(CheckpointID.CP1, stage1=make_stage1(_state(0.1)))
    assert o._action_history and o._state_history

    o.on_task_end()

    assert not o._action_history
    assert not o._state_history


# ------------------------------------------------------------------
# History injection into Judge
# ------------------------------------------------------------------


def test_judge_receives_view_and_history_kwargs():
    captured = {}

    class _CapturingJudge:
        def __call__(self, results, checkpoint_id, cached_data, *, view=None, history=None):
            captured["view"] = view
            captured["history"] = history
            from openpi.cache.components.judge import HitType, JudgeResult
            return JudgeResult(HitType.MISS)
        def on_episode_start(self):
            pass

    o = _make_orch([CheckpointID.CP1], judge=_CapturingJudge())
    o.broadcast_action(torch.zeros(50, 32))
    o.check(CheckpointID.CP1, stage1=make_stage1(_state(0.1)))

    assert captured["view"] is not None
    assert captured["history"] is not None
    # History is snapshot of the current action / state buffers
    assert len(captured["history"].actions) == 1
    assert len(captured["history"].states) == 1


def test_old_judge_without_view_kwargs_still_works():
    # AlwaysHitJudge / ThresholdJudge accept **kwargs (B0 ship). Confirm
    # check() still drives them when view+history are injected.
    o = _make_orch([CheckpointID.CP1], judge=AlwaysHitJudge())
    from openpi.cache.storage_types import CachePayload
    payload = CachePayload(action_chunk=torch.zeros(50, 32))
    insert_entry(o._storage, CheckpointID.CP1, _state(0.1), payload)
    o.on_task_begin()
    res = o.check(CheckpointID.CP1, stage1=make_stage1(_state(0.1)))
    assert res.hit_type.name == "FULL_HIT"
