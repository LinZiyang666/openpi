"""Unit tests for episode_start.extra_metadata propagation chain.

Covers:
  - Wire-level: WebsocketClientPolicy.episode_start serializes `__extra__`
    when extra_metadata is non-empty; omits it when None / empty (legacy
    backward compat).
  - Server-side: orchestrator stashes extra on `_current_episode_extra` and
    clears it on episode_end.
  - Lifecycle filtered dispatch: existing kwargs-free judges (AlwaysHit /
    AlwaysWarmStart / Threshold / CompositeJudge) silently swallow
    `extra_metadata` via `_safe_call_lifecycle` inspect.signature filter;
    DumpingJudge actually consumes it.
  - End-to-end: extra_metadata fed at episode_start reaches DumpingJudge
    `_current_extra` after broadcast.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openpi.cache.components.judge import (
    AlwaysHitJudge,
    AlwaysWarmStartJudge,
    DumpingJudge,
    ThresholdJudge,
)
from openpi.cache.orchestrator import CacheOrchestrator


# ------------------------------------------------------------------
# Wire-level: extra_metadata serialized as __extra__ when present
# ------------------------------------------------------------------


def _build_mock_ws_client(recorded: list):
    """Construct a WebsocketClientPolicy with mocked ws + packer for inspection."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    # Stand in for the websocket connection: capture sent payloads, return
    # a packed empty ack on recv.
    fake_ws = MagicMock()
    import msgpack_numpy

    def _send(packed):
        recorded.append(msgpack_numpy.unpackb(packed))

    fake_ws.send.side_effect = _send
    fake_ws.recv.return_value = msgpack_numpy.packb({"__ack__": "episode_start"})
    client._ws = fake_ws
    client._packer = msgpack_numpy.Packer()
    return client


def test_wire_omits_extra_when_none():
    sent = []
    client = _build_mock_ws_client(sent)
    client.episode_start(experiment="exp", task="t", episode_id=1, episode_name="")
    assert "__extra__" not in sent[0]


def test_wire_omits_extra_when_empty_dict():
    sent = []
    client = _build_mock_ws_client(sent)
    client.episode_start(
        experiment="exp", task="t", episode_id=1, episode_name="",
        extra_metadata={},
    )
    assert "__extra__" not in sent[0]


def test_wire_includes_extra_when_set():
    sent = []
    client = _build_mock_ws_client(sent)
    client.episode_start(
        experiment="exp", task="t", episode_id=1, episode_name="",
        extra_metadata={"task_id": 3, "orig_init_state_idx": 7},
    )
    assert sent[0]["__extra__"] == {"task_id": 3, "orig_init_state_idx": 7}


# ------------------------------------------------------------------
# _safe_call_lifecycle filtering: existing judges accept extra kwarg silently
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "judge",
    [
        AlwaysHitJudge(),
        AlwaysWarmStartJudge(start_t=0.5),
        ThresholdJudge(),
    ],
)
def test_safe_call_lifecycle_swallows_extra_for_kwargs_free_judges(judge):
    """`on_episode_start(self)` judges must not raise when extra_metadata kwarg passed."""
    # Should be no-op (signature filter drops the kwarg).
    CacheOrchestrator._safe_call_lifecycle(
        judge, "on_episode_start",
        extra_metadata={"task_id": 1, "orig_init_state_idx": 0},
    )


def test_safe_call_lifecycle_passes_extra_to_dumping_judge(tmp_path):
    """DumpingJudge declares extra_metadata kwarg, so filter must pass it through."""
    wrapper = DumpingJudge(AlwaysHitJudge(), [], tmp_path / "d.jsonl", "cfg_a")
    CacheOrchestrator._safe_call_lifecycle(
        wrapper, "on_episode_start",
        extra_metadata={"task_id": 3, "orig_init_state_idx": 7},
    )
    assert wrapper._current_extra == {"task_id": 3, "orig_init_state_idx": 7}


# ------------------------------------------------------------------
# End-to-end: Orchestrator.on_episode_start -> DumpingJudge stash
# ------------------------------------------------------------------


def _minimal_orchestrator(judge):
    """Construct a bare-minimum CacheOrchestrator with one CP1 judge.

    Components other than the judge are mocked — this test focuses on
    Orchestrator's own lifecycle state machine, not component wiring.
    """
    from openpi.cache.types import CheckpointID

    storage = MagicMock()
    storage.open_search_session = MagicMock()
    storage.close_search_session = MagicMock()

    # Bare key_builder / strategy / gate — they only need on_episode_start
    # callable. `get_search_session_id` returning None skips the backend
    # session-register loop.
    def _make_lifecycle_stub():
        s = MagicMock()
        s.on_episode_start = MagicMock()
        s.get_search_session_id = MagicMock(return_value=None)
        return s

    return CacheOrchestrator(
        storage=storage,
        key_builder=_make_lifecycle_stub(),
        gates={CheckpointID.CP1: _make_lifecycle_stub()},
        judges={CheckpointID.CP1: judge},
        search_strategies={CheckpointID.CP1: _make_lifecycle_stub()},
    )


def test_orchestrator_stashes_extra_on_episode_start():
    judge = AlwaysHitJudge()
    orch = _minimal_orchestrator(judge)
    orch.on_episode_start(
        task_key="t", episode_id="ep1",
        extra_metadata={"task_id": 3, "orig_init_state_idx": 7},
    )
    assert orch._current_episode_extra == {"task_id": 3, "orig_init_state_idx": 7}


def test_orchestrator_clears_extra_on_episode_end():
    judge = AlwaysHitJudge()
    orch = _minimal_orchestrator(judge)
    orch.on_episode_start(
        task_key="t", episode_id="ep1",
        extra_metadata={"task_id": 3, "orig_init_state_idx": 7},
    )
    orch.on_episode_end()
    assert orch._current_episode_extra == {}


def test_orchestrator_resets_extra_on_next_episode_start_no_extra():
    judge = AlwaysHitJudge()
    orch = _minimal_orchestrator(judge)
    orch.on_episode_start(
        task_key="t", episode_id="ep1",
        extra_metadata={"task_id": 3, "orig_init_state_idx": 7},
    )
    orch.on_episode_start(task_key="t", episode_id="ep2")  # no extra
    assert orch._current_episode_extra == {}


def test_orchestrator_broadcast_reaches_dumping_judge_only(tmp_path):
    """DumpingJudge stashes; AlwaysHit (also a judge) does not raise."""
    dump_judge = DumpingJudge(AlwaysHitJudge(), [], tmp_path / "d.jsonl", "cfg_a")
    orch = _minimal_orchestrator(dump_judge)
    orch.on_episode_start(
        task_key="t", episode_id="ep1",
        extra_metadata={"task_id": 3, "orig_init_state_idx": 7},
    )
    assert dump_judge._current_extra == {"task_id": 3, "orig_init_state_idx": 7}


def test_orchestrator_legacy_no_kwarg_call_still_works():
    """Caller that doesn't pass extra_metadata at all -> stashed as empty."""
    judge = AlwaysHitJudge()
    orch = _minimal_orchestrator(judge)
    orch.on_episode_start(task_key="t", episode_id="ep1")
    assert orch._current_episode_extra == {}
