"""Wire protocol for worker <-> agent <-> driver messaging.

Transport (plan §5.3, G1R1 Item 5): msgpack-over-TCP with a 4-byte big-endian
length prefix per frame. Every message carries ``protocol_version`` so a version
skew is detected loudly rather than mis-decoded. ``pull`` / ``report`` are
idempotent on ``task_uid`` so a duplicate (e.g. after a reconnect) is safe.

Coupling map:
  DEPENDS ON:  task.py (EpisodeTask / EpisodeResult), msgpack, standard library
  CONSUMED BY: worker.py, agent.py, driver.py
  IF CHANGED:  bump PROTOCOL_VERSION; all three layers must agree
  NOTE:        no numpy on the wire (EpisodeTask is scalar, per_step_rows are
               plain dicts) so plain msgpack suffices.
"""

from __future__ import annotations

import dataclasses
import socket
import struct
from typing import Any

import msgpack

from openpi.conductor import task as _task

PROTOCOL_VERSION = 1

# Message type constants (worker/agent/driver wire vocabulary).
MSG_PULL = "pull"
MSG_ASSIGN = "assign"
MSG_REPORT_PROGRESS = "report.progress"
MSG_REPORT_RESULT = "report.result"
MSG_SHUTDOWN = "shutdown"

_LEN = struct.Struct(">I")  # 4-byte big-endian unsigned length prefix
MAX_FRAME = 64 * 1024 * 1024  # 64 MB ceiling; per_step_rows are small, this is slack


class ProtocolError(RuntimeError):
    """Raised on a framing / version / decode failure."""


# ----------------------------------------------------------------------
# (de)serialization of domain objects
# ----------------------------------------------------------------------


def task_to_wire(t: _task.EpisodeTask) -> dict[str, Any]:
    return dataclasses.asdict(t)


def task_from_wire(d: dict[str, Any]) -> _task.EpisodeTask:
    return _task.EpisodeTask(**d)


def result_to_wire(r: _task.EpisodeResult) -> dict[str, Any]:
    return dataclasses.asdict(r)


def result_from_wire(d: dict[str, Any]) -> _task.EpisodeResult:
    return _task.EpisodeResult(**d)


# ----------------------------------------------------------------------
# message encode / decode
# ----------------------------------------------------------------------


def encode(msg_type: str, payload: dict[str, Any] | None = None) -> bytes:
    """Encode one message to a length-prefixed msgpack frame."""
    body = {"v": PROTOCOL_VERSION, "type": msg_type, "payload": payload or {}}
    packed = msgpack.packb(body, use_bin_type=True)
    return _LEN.pack(len(packed)) + packed


def decode(frame: bytes) -> tuple[str, dict[str, Any]]:
    """Decode a msgpack frame body (without the length prefix).

    Returns ``(msg_type, payload)``; raises ``ProtocolError`` on version skew.
    """
    body = msgpack.unpackb(frame, raw=False)
    if not isinstance(body, dict) or "type" not in body:
        raise ProtocolError(f"malformed message: {body!r}")
    version = body.get("v")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version mismatch: peer={version} self={PROTOCOL_VERSION}")
    return body["type"], body.get("payload", {})


# ----------------------------------------------------------------------
# framed socket I/O
# ----------------------------------------------------------------------


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise ``ConnectionError`` on a short close."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed connection mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, msg_type: str, payload: dict[str, Any] | None = None) -> None:
    """Send one framed message over a blocking TCP socket."""
    sock.sendall(encode(msg_type, payload))


def recv_message(sock: socket.socket) -> tuple[str, dict[str, Any]]:
    """Receive one framed message. Raises ConnectionError if the peer closed."""
    header = _recv_exact(sock, _LEN.size)
    (length,) = _LEN.unpack(header)
    if length > MAX_FRAME:
        # Guard against a misaligned/garbage length prefix requesting a huge buffer.
        raise ProtocolError(f"frame too large: {length} > {MAX_FRAME}")
    frame = _recv_exact(sock, length)
    return decode(frame)
