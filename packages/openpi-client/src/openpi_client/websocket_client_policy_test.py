"""Unit tests for ``WebsocketClientPolicy``.

All tests fake ``websockets.sync.client.connect`` so no real server is needed.
The fake captures everything sent on the wire and lets each test queue the
server-side replies that ``recv()`` should return.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy as wcp


# ------------------------------------------------------------------
# Fake websocket
# ------------------------------------------------------------------


class _FakeWS:
    """Captures ``send`` calls and returns pre-queued responses from ``recv``."""

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self.sent: list[bytes] = []
        self.close_calls = 0
        self.close_should_raise: bool = False

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self) -> Any:
        # msgpack bytes for normal messages; raw str for server-side errors.
        return self._replies.pop(0)

    def close(self) -> None:
        self.close_calls += 1
        if self.close_should_raise:
            raise RuntimeError("simulated close failure")


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    recv_after_metadata: list[Any] | None = None,
) -> tuple[wcp.WebsocketClientPolicy, _FakeWS]:
    """Patch ``websockets.sync.client.connect`` so ``__init__`` returns instantly.

    The fake websocket's first ``recv`` returns the server metadata that
    ``_wait_for_server`` consumes; any remaining replies are left in the queue
    for the test body to exercise.
    """
    metadata = msgpack_numpy.packb({"server": "fake"})
    replies: list[Any] = [metadata] + list(recv_after_metadata or [])
    ws = _FakeWS(replies=replies)

    def _fake_connect(*_args, **_kwargs):
        return ws

    monkeypatch.setattr(wcp.websockets.sync.client, "connect", _fake_connect)
    client = wcp.WebsocketClientPolicy(host="ws://fake")
    return client, ws


# ------------------------------------------------------------------
# episode_start: backward compat + forwarding
# ------------------------------------------------------------------


def test_episode_start_defaults_episode_name_to_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy callers (no ``episode_name`` kwarg) must produce an empty string
    on the wire — the server treats empty as "no override" (plan §3.2), so a
    legacy client talking to a new server keeps pre-Layer-B-1 behaviour.
    """
    client, ws = _make_client(
        monkeypatch, recv_after_metadata=[msgpack_numpy.packb({"__ack__": "episode_start"})]
    )

    client.episode_start(experiment="exp", task="task", episode_id=1)

    assert len(ws.sent) == 1
    sent = msgpack_numpy.unpackb(ws.sent[0])
    assert sent["__ctrl__"] == "episode_start"
    assert sent["__episode_name__"] == ""
    assert sent["__experiment__"] == "exp"
    assert sent["__task__"] == "task"
    assert sent["__episode_id__"] == 1


def test_episode_start_forwards_episode_name(monkeypatch: pytest.MonkeyPatch) -> None:
    client, ws = _make_client(
        monkeypatch, recv_after_metadata=[msgpack_numpy.packb({"__ack__": "episode_start"})]
    )

    client.episode_start(
        experiment="exp",
        task="task",
        episode_id=1,
        episode_name="task_3/episode_2",
    )

    sent = msgpack_numpy.unpackb(ws.sent[0])
    assert sent["__episode_name__"] == "task_3/episode_2"


# ------------------------------------------------------------------
# prefill_trajectory: wire format + error path
# ------------------------------------------------------------------


def test_prefill_trajectory_sends_expected_ctrl_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wire-level smoke: the ctrl message carries observations/actions verbatim
    plus the two flags. Actions are numpy arrays so we also assert they
    round-trip through ``msgpack_numpy`` without lossy conversion.
    """
    client, ws = _make_client(
        monkeypatch, recv_after_metadata=[msgpack_numpy.packb({"__ack__": "prefill_trajectory"})]
    )

    obs = [{"state": np.zeros(7, dtype=np.float32)} for _ in range(3)]
    actions = [np.arange(50 * 7, dtype=np.float32).reshape(50, 7) for _ in range(3)]

    ack = client.prefill_trajectory(obs, actions, record=False, on_miss="error")

    assert ack == {"__ack__": "prefill_trajectory"}
    assert len(ws.sent) == 1
    sent = msgpack_numpy.unpackb(ws.sent[0])
    assert sent["__ctrl__"] == "prefill_trajectory"
    assert sent["record"] is False
    assert sent["on_miss"] == "error"
    assert len(sent["observations"]) == 3
    assert len(sent["actions"]) == 3
    np.testing.assert_array_equal(sent["actions"][0], actions[0])
    np.testing.assert_array_equal(sent["observations"][0]["state"], obs[0]["state"])


def test_prefill_trajectory_raises_on_string_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors ``infer``'s contract: a string response signals a server-side
    error, so the client must surface it as ``RuntimeError`` rather than
    attempting to unpack the string as msgpack bytes.
    """
    client, _ = _make_client(monkeypatch, recv_after_metadata=["server blew up"])

    with pytest.raises(RuntimeError, match="server blew up"):
        client.prefill_trajectory([{}], [np.zeros(1, dtype=np.float32)])


# ------------------------------------------------------------------
# Lifecycle: context manager + close()
# ------------------------------------------------------------------


def test_context_manager_closes_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    client, ws = _make_client(monkeypatch)

    with client as c:
        assert c is client
    assert ws.close_calls == 1


def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling ``close()`` twice must not raise — runners may defensively
    close in a ``finally`` branch even when ``__exit__`` has already run.
    """
    client, ws = _make_client(monkeypatch)

    client.close()
    client.close()
    assert ws.close_calls == 2


def test_close_swallows_close_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A half-closed server-side connection can make ``ws.close()`` raise;
    the client must not let that mask the real exception from the ``with``
    body (plan §19.B4 rationale for the swallowing ``try/except``).
    """
    client, ws = _make_client(monkeypatch)
    ws.close_should_raise = True

    # Must not propagate.
    client.close()
    assert ws.close_calls == 1
