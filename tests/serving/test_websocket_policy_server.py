"""Unit tests for ``WebsocketPolicyServer`` ctrl-message dispatcher.

We drive ``_handler`` end-to-end with a fake ``ServerConnection``: the test
queues msgpack-encoded frames that ``recv()`` yields in order, and captures
everything ``send()`` emits so dispatch behaviour can be asserted without
running a real websocket stack. This is the cheapest way to exercise plan
§22.4's focus areas (keyword dispatch for episode_start/end, same-connection
prefill_trajectory, and the regression that ``prefill_begin`` / ``prefill_end``
stay unimplemented).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from openpi_client import msgpack_numpy

from openpi.serving import websocket_policy_server as wps


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


class _FakeConnection:
    """Minimal ``ServerConnection`` stub.

    ``recv()`` pops pre-queued bytes and, when exhausted, raises
    ``ConnectionClosed`` so ``_handler`` exits its loop naturally (mirrors
    the real ``websockets`` library contract). ``send()`` records every
    frame for later assertion; a frame may be either bytes (msgpack ack)
    or str (server-side error, per the existing ``infer`` protocol).
    """

    remote_address = ("test", 0)

    def __init__(self, incoming: list[bytes]) -> None:
        self._incoming = list(incoming)
        self.sent: list[Any] = []
        self.closed = False

    async def recv(self) -> bytes:
        if not self._incoming:
            # websockets raises ConnectionClosed on EOF — _handler catches it
            # and runs its shutdown path. We use the real exception class so
            # the dispatcher's ``except websockets.ConnectionClosed`` matches.
            import websockets

            raise websockets.ConnectionClosed(rcvd=None, sent=None)
        return self._incoming.pop(0)

    async def send(self, payload: Any) -> None:
        self.sent.append(payload)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True


def _pack(obj: Any) -> bytes:
    return msgpack_numpy.Packer().pack(obj)


def _run_handler(policy: Any, incoming: list[bytes]) -> _FakeConnection:
    """Build a single-connection server and drain one connection worth of frames.

    Returns the fake connection so the caller can inspect ``sent``. Runs the
    handler under ``asyncio.run`` so the test body stays synchronous.
    """
    server = wps.WebsocketPolicyServer(policy=policy, metadata={"server": "test"})
    conn = _FakeConnection(incoming)
    asyncio.run(server._handler(conn))
    return conn


def _ack_payloads(conn: _FakeConnection) -> list[dict]:
    """Skip the metadata frame and decode every remaining msgpack ack."""
    # First frame is always the server metadata handshake.
    return [msgpack_numpy.unpackb(p) for p in conn.sent[1:] if isinstance(p, (bytes, bytearray))]


# ---------------------------------------------------------------------------
# Mock policies
# ---------------------------------------------------------------------------


def _make_policy_with_lifecycle():
    """MagicMock exposing ``on_episode_start`` / ``on_episode_end`` / ``infer``."""
    policy = MagicMock(spec=["on_episode_start", "on_episode_end", "infer"])
    return policy


def _make_cache_policy():
    """Policy that also advertises ``prefill_trajectory`` (cache-wrapped)."""
    policy = MagicMock(
        spec=["on_episode_start", "on_episode_end", "prefill_trajectory", "infer"]
    )
    return policy


# ---------------------------------------------------------------------------
# T1 / T13: episode_start keyword dispatch
# ---------------------------------------------------------------------------


def test_episode_start_dispatched_with_keyword_args() -> None:
    policy = _make_policy_with_lifecycle()
    _run_handler(
        policy,
        incoming=[
            _pack(
                {
                    "__ctrl__": "episode_start",
                    "__experiment__": "exp",
                    "__task__": "task",
                    "__episode_id__": 7,
                    "__episode_name__": "task_3/episode_2",
                }
            )
        ],
    )

    policy.on_episode_start.assert_called_once_with(
        experiment="exp",
        task="task",
        episode_id=7,
        episode_name="task_3/episode_2",
    )


def test_episode_start_defaults_episode_name_to_empty_string() -> None:
    """Legacy clients (no ``__episode_name__`` field) must produce ``""``."""
    policy = _make_policy_with_lifecycle()
    _run_handler(
        policy,
        incoming=[
            _pack(
                {
                    "__ctrl__": "episode_start",
                    "__experiment__": "exp",
                    "__task__": "task",
                    "__episode_id__": 7,
                }
            )
        ],
    )

    policy.on_episode_start.assert_called_once_with(
        experiment="exp",
        task="task",
        episode_id=7,
        episode_name="",
    )


# ---------------------------------------------------------------------------
# T14: episode_end keyword dispatch
# ---------------------------------------------------------------------------


def test_episode_end_dispatched_with_success_kwarg() -> None:
    policy = _make_policy_with_lifecycle()
    _run_handler(
        policy,
        incoming=[_pack({"__ctrl__": "episode_end", "__success__": True})],
    )

    policy.on_episode_end.assert_called_once_with(success=True)


# ---------------------------------------------------------------------------
# T3: prefill_trajectory dispatch
# ---------------------------------------------------------------------------


def test_prefill_trajectory_dispatches_with_keyword_args() -> None:
    policy = _make_cache_policy()
    obs_list = [{"state": [1.0, 2.0]}]
    act_list = [[0.1, 0.2, 0.3]]

    conn = _run_handler(
        policy,
        incoming=[
            _pack(
                {
                    "__ctrl__": "prefill_trajectory",
                    "observations": obs_list,
                    "actions": act_list,
                    "record": False,
                    "on_miss": "error",
                }
            )
        ],
    )

    policy.prefill_trajectory.assert_called_once_with(
        obs_list,
        act_list,
        record=False,
        on_miss="error",
    )
    acks = _ack_payloads(conn)
    assert acks == [{"__ack__": "prefill_trajectory"}]


# ---------------------------------------------------------------------------
# T15: prefill_trajectory on non-cache policy -> msgpack error ack
# ---------------------------------------------------------------------------


def test_prefill_trajectory_on_non_cache_policy_sends_error_ack() -> None:
    """cleanup/10: the server now unifies the prefill error path onto the
    msgpack ``{"__ack__": "error", "msg": ...}`` shape every other ctrl uses.
    The client raises ``RuntimeError`` on either this dict OR the legacy bare
    string — the dict is the new default because it round-trips through
    ``msgpack_numpy.unpackb`` without tripping the worker's ``isinstance(
    response, str)`` compat branch unnecessarily."""
    policy = _make_policy_with_lifecycle()  # no ``prefill_trajectory`` attr
    conn = _run_handler(
        policy,
        incoming=[
            _pack(
                {
                    "__ctrl__": "prefill_trajectory",
                    "observations": [{}],
                    "actions": [[0.0]],
                }
            )
        ],
    )

    acks = _ack_payloads(conn)
    assert len(acks) == 1
    assert acks[0]["__ack__"] == "error"
    assert "prefill_trajectory requires a cache-wrapped policy" in acks[0]["msg"]


def test_prefill_trajectory_worker_exception_surfaces_as_error_ack() -> None:
    """cleanup/10 invariant: if the server-side ``prefill_trajectory`` raises,
    the exception must NOT crash the connection. It must be wrapped into an
    ``{"__ack__": "error", "msg": str(exc)}`` frame so the client sees the
    failure as a regular protocol error."""
    policy = _make_cache_policy()
    policy.prefill_trajectory.side_effect = RuntimeError("synthetic prefill boom")
    conn = _run_handler(
        policy,
        incoming=[
            _pack(
                {
                    "__ctrl__": "prefill_trajectory",
                    "observations": [{"x": 0}],
                    "actions": [[1.0]],
                }
            )
        ],
    )

    acks = _ack_payloads(conn)
    assert len(acks) == 1
    assert acks[0]["__ack__"] == "error"
    assert "synthetic prefill boom" in acks[0]["msg"]


# ---------------------------------------------------------------------------
# T4: prefill_begin / prefill_end stay unimplemented (regression guard)
# ---------------------------------------------------------------------------


def test_prefill_begin_falls_through_to_ignored_ack() -> None:
    """Plan §19.B7 forbids re-introducing prefill_begin / prefill_end as
    wire-level control messages. This test freezes that decision: if a
    future change adds a branch for them it must explicitly update this
    test, which forces the change to surface in review."""
    policy = _make_cache_policy()
    conn = _run_handler(
        policy,
        incoming=[_pack({"__ctrl__": "prefill_begin", "payload_b64": "unused"})],
    )

    acks = _ack_payloads(conn)
    assert acks == [{"__ack__": "ignored"}]
    # And the policy must not have received any prefill call.
    policy.prefill_trajectory.assert_not_called()


def test_prefill_end_falls_through_to_ignored_ack() -> None:
    policy = _make_cache_policy()
    conn = _run_handler(
        policy,
        incoming=[_pack({"__ctrl__": "prefill_end"})],
    )

    acks = _ack_payloads(conn)
    assert acks == [{"__ack__": "ignored"}]


# ---------------------------------------------------------------------------
# Backwards-compat: policies without lifecycle methods must not crash
# ---------------------------------------------------------------------------


def test_episode_start_skipped_when_policy_has_no_hook() -> None:
    """A plain ``Policy`` has no ``on_episode_start``. Server must still ack."""
    bare_policy = MagicMock(spec=["infer"])
    conn = _run_handler(
        bare_policy,
        incoming=[
            _pack({"__ctrl__": "episode_start", "__experiment__": "e", "__task__": "t"})
        ],
    )

    acks = _ack_payloads(conn)
    assert acks == [{"__ack__": "episode_start"}]


# ---------------------------------------------------------------------------
# Pytest "no asyncio plugin" shim: tests above invoke ``asyncio.run`` directly
# rather than using ``@pytest.mark.asyncio`` so this module has no external
# plugin dependency. The dummy below silences pytest's unused-import warning.
# ---------------------------------------------------------------------------


def _unused_pytest_import() -> None:
    _ = pytest  # noqa: F841
