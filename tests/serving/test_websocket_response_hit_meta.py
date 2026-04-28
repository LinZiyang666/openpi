"""Backward-compat smoke for the ``__hit_meta__`` response field.

The verdict_factor_judge B1 wire change is additive: ``InferenceInterceptor``
attaches a ``__hit_meta__`` dict to every ``infer`` response. Existing
clients (older SDKs, non-cache experiments) must continue to work whether
that field is present or absent. The tests below pin both directions:

  1. ``WebsocketPolicyServer._handler`` does not strip extra response keys —
     a payload carrying ``__hit_meta__`` reaches the wire intact alongside
     the server-injected ``server_timing``.
  2. ``msgpack_numpy`` round-trips ``__hit_meta__`` so the
     ``WebsocketClientPolicy.infer`` decode path (which is just
     ``msgpack_numpy.unpackb``) preserves the field for callers.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from openpi_client import msgpack_numpy

from openpi.serving import websocket_policy_server as wps


class _FakeConnection:
    """Minimal ``ServerConnection`` stub — same shape as the dispatcher tests
    in ``test_websocket_policy_server.py``."""

    remote_address = ("test", 0)

    def __init__(self, incoming: list[bytes]) -> None:
        self._incoming = list(incoming)
        self.sent: list[Any] = []
        self.closed = False

    async def recv(self) -> bytes:
        if not self._incoming:
            import websockets

            raise websockets.ConnectionClosed(rcvd=None, sent=None)
        return self._incoming.pop(0)

    async def send(self, payload: Any) -> None:
        self.sent.append(payload)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        del code, reason  # signature parity with websockets.ServerConnection
        self.closed = True


def _pack(obj: Any) -> bytes:
    return msgpack_numpy.Packer().pack(obj)


# ---------------------------------------------------------------------------
# Server pass-through
# ---------------------------------------------------------------------------


def test_server_propagates_hit_meta_in_infer_response() -> None:
    """A policy whose infer() result includes __hit_meta__ must reach the
    wire with that field intact — the server adds ``server_timing`` but does
    not delete or rename anything else."""
    expected_meta = {
        "hit_type": "WARM_START",
        "start_t": 0.7,
        "winner_id": "episode_0019:42",
        "cp1_score": 0.96,
    }
    infer_result = {
        "actions": [[1.0, 2.0]],
        "state": [0.1, 0.2],
        "__hit_meta__": expected_meta,
    }
    policy = MagicMock(spec=["infer"])
    policy.infer.return_value = dict(infer_result)

    server = wps.WebsocketPolicyServer(policy=policy, metadata={"server": "test"})
    conn = _FakeConnection(incoming=[_pack({"obs": True})])
    asyncio.run(server._handler(conn))

    # First frame is metadata handshake; second is the infer response.
    assert len(conn.sent) >= 2
    decoded = msgpack_numpy.unpackb(conn.sent[1])
    assert "__hit_meta__" in decoded
    assert decoded["__hit_meta__"] == expected_meta
    # server adds wall-clock timing without touching __hit_meta__.
    assert "server_timing" in decoded


def test_server_works_without_hit_meta_field() -> None:
    """A policy that omits __hit_meta__ (legacy / non-cache path) must still
    round-trip through the server unchanged — no KeyError, no failure to
    reply."""
    infer_result = {
        "actions": [[1.0, 2.0]],
        "state": [0.1, 0.2],
    }
    policy = MagicMock(spec=["infer"])
    policy.infer.return_value = dict(infer_result)

    server = wps.WebsocketPolicyServer(policy=policy, metadata={"server": "test"})
    conn = _FakeConnection(incoming=[_pack({"obs": True})])
    asyncio.run(server._handler(conn))

    decoded = msgpack_numpy.unpackb(conn.sent[1])
    assert "__hit_meta__" not in decoded
    assert "server_timing" in decoded
    assert decoded["actions"] == [[1.0, 2.0]]


# ---------------------------------------------------------------------------
# msgpack round-trip (client decode path)
# ---------------------------------------------------------------------------


def test_msgpack_roundtrip_preserves_hit_meta_with_nones() -> None:
    """The MISS placeholder uses ``None`` for start_t / winner_id / cp1_score;
    msgpack must preserve None across encode/decode so the client can detect
    "no winner" without misreading it as 0.0 / empty string."""
    payload = {
        "actions": [[0.0]],
        "__hit_meta__": {
            "hit_type": "MISS",
            "start_t": None,
            "winner_id": None,
            "cp1_score": None,
        },
    }
    decoded = msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(payload))
    assert decoded["__hit_meta__"] == payload["__hit_meta__"]
    # Sanity: explicit None preserved (msgpack 'nil' decodes to Python None).
    assert decoded["__hit_meta__"]["start_t"] is None
