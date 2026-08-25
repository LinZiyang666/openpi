"""Wire protocol round-trip / framing tests (M1)."""

from __future__ import annotations

import socket
import struct

import msgpack
import pytest

from openpi.conductor import protocol as P  # noqa: N812
from openpi.conductor import task as T  # noqa: N812


def test_encode_decode_roundtrip():
    frame = P.encode(P.MSG_PULL, {"worker_id": "w0", "server_host": "h"})
    # strip the 4-byte length prefix
    msg_type, payload = P.decode(frame[4:])
    assert msg_type == P.MSG_PULL
    assert payload == {"worker_id": "w0", "server_host": "h"}


def test_version_mismatch_raises(monkeypatch):
    frame = P.encode(P.MSG_PULL, {})
    monkeypatch.setattr(P, "PROTOCOL_VERSION", P.PROTOCOL_VERSION + 1)
    with pytest.raises(P.ProtocolError, match="version mismatch"):
        P.decode(frame[4:])


def test_task_wire_roundtrip():
    t = T.EpisodeTask(
        task_uid="y:eval:0:0",
        yaml_id="y",
        phase="eval",
        experiment="libero_spatial",
        task_id=0,
        episode_idx=0,
        orig_init_state_idx=0,
        server_host="h",
        server_port=8000,
        bundle_id="b",
        extra={"k": 1},
    )
    back = P.task_from_wire(P.task_to_wire(t))
    assert back == t


def test_result_wire_roundtrip():
    r = T.EpisodeResult(
        task_uid="u", success=True, n_steps=42, error=None, per_step_rows=[{"step": 0, "hit_type": "MISS"}]
    )
    back = P.result_from_wire(P.result_to_wire(r))
    assert back == r


def test_framed_send_recv_over_socketpair():
    a, b = socket.socketpair()
    try:
        P.send_message(a, P.MSG_ASSIGN, {"task": {"x": 1}})
        msg_type, payload = P.recv_message(b)
        assert msg_type == P.MSG_ASSIGN
        assert payload == {"task": {"x": 1}}
    finally:
        a.close()
        b.close()


def test_recv_exact_raises_on_short_close():
    a, b = socket.socketpair()
    a.close()  # peer closed immediately
    try:
        with pytest.raises(ConnectionError):
            P.recv_message(b)
    finally:
        b.close()


def test_recv_message_rejects_oversized_frame():
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack(">I", P.MAX_FRAME + 1))
        with pytest.raises(P.ProtocolError, match="too large"):
            P.recv_message(b)
    finally:
        a.close()
        b.close()


def test_decode_rejects_malformed_message():
    frame = msgpack.packb([1, 2, 3], use_bin_type=True)  # a list, not a {type:...} dict
    with pytest.raises(P.ProtocolError, match="malformed"):
        P.decode(frame)


def test_result_from_wire_ignores_fields_this_build_does_not_know():
    """Worker and driver live on different machines and skew in practice.

    Without this, a worker one version ahead raises TypeError on *every*
    episode, which looks like a dead fleet rather than a version mismatch.
    """
    from openpi.conductor.protocol import result_from_wire

    r = result_from_wire(
        {"task_uid": "u", "success": True, "n_steps": 3, "invented_later": 42}
    )
    assert r.task_uid == "u"
    assert r.n_steps == 3
