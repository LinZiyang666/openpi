"""Tests for the verdict_factor_judge B2 warmup ctrl handlers.

Three new ctrl messages and one extended message form the wire protocol:

  - ``fetch_dump{warmup_yaml_id}`` — read a warmup dump file back.
  - ``preload_normalizer_buffer{eval_yaml_id, buffer}`` — stash raw factor
    values in the WarmupPool.
  - ``unload_warmup_buffer{eval_yaml_id}`` — clear the pool entry AND delete
    the corresponding warmup dump file.

The strict double-yaml_id naming (``warmup_yaml_id`` vs ``eval_yaml_id``)
is enforced both at field-name level and via the server's "derive warmup
name from eval id" rule. Path traversal — including symlink-based escape —
must be rejected by the ``.resolve()`` allowlist.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from openpi_client import msgpack_numpy

from openpi.cache import warmup_pool as _warmup_pool_mod
from openpi.serving import websocket_policy_server as wps


# ---------------------------------------------------------------------------
# Fixture: install a tmp warmup dump root + reset module state per test
# ---------------------------------------------------------------------------


@pytest.fixture
def warmup_root(tmp_path, monkeypatch):
    """Point the WS module at a fresh warmup root + reset the pool singleton."""
    root = tmp_path / "warmup"
    root.mkdir(mode=0o700)
    wps.set_warmup_dump_root(root)
    # Replace the WarmupPool singleton with a clean instance so test ordering
    # cannot bleed state across cases.
    fresh = _warmup_pool_mod.WarmupPool()
    monkeypatch.setattr(_warmup_pool_mod, "_GLOBAL", fresh)
    yield root.resolve()
    wps.set_warmup_dump_root(None)


# ---------------------------------------------------------------------------
# fetch_dump: success + error paths
# ---------------------------------------------------------------------------


def test_fetch_dump_returns_file_bytes(warmup_root) -> None:
    target = warmup_root / "phase1_idx0__warmup.jsonl"
    target.write_text('{"k": 1}\n{"k": 2}\n')
    ack = wps._handle_fetch_dump("phase1_idx0__warmup")
    assert ack["__ack__"] == "fetch_dump"
    assert ack["warmup_yaml_id"] == "phase1_idx0__warmup"
    assert ack["content"] == target.read_bytes()


def test_fetch_dump_rejects_traversal_with_dotdot(warmup_root) -> None:
    ack = wps._handle_fetch_dump("../etc/passwd")
    assert ack["__ack__"] == "error"
    assert "invalid" in ack["msg"]


def test_fetch_dump_rejects_traversal_with_slash(warmup_root) -> None:
    ack = wps._handle_fetch_dump("subdir/leak")
    assert ack["__ack__"] == "error"
    assert "invalid" in ack["msg"]


def test_fetch_dump_rejects_when_root_missing(monkeypatch) -> None:
    """Server started without --warmup-dump-root must refuse fetch_dump
    rather than silently defaulting to a writable location."""
    monkeypatch.setattr(wps, "_warmup_dump_root", None)
    ack = wps._handle_fetch_dump("anything")
    assert ack["__ack__"] == "error"
    assert "not configured" in ack["msg"]


def test_fetch_dump_returns_not_found_when_file_absent(warmup_root) -> None:
    ack = wps._handle_fetch_dump("not_present")
    assert ack["__ack__"] == "error"
    assert "not found" in ack["msg"]


def test_fetch_dump_blocks_symlink_traversal(warmup_root, tmp_path) -> None:
    """Symlink that points outside the root must be rejected after .resolve()
    follows the link — the candidate's resolved parent escapes the root."""
    secret = tmp_path / "secret.txt"
    secret.write_text("not for clients")
    link = warmup_root / "leak__warmup.jsonl"
    os.symlink(secret, link)
    ack = wps._handle_fetch_dump("leak__warmup")
    assert ack["__ack__"] == "error"
    assert "invalid" in ack["msg"]


# ---------------------------------------------------------------------------
# preload_normalizer_buffer
# ---------------------------------------------------------------------------


def test_preload_buffer_stashes_in_pool(warmup_root) -> None:
    buf = {"f1a_a_jerk": [0.1, 0.2, 0.3], "f2_var": [0.5, 0.6]}
    ack = wps._handle_preload_normalizer_buffer("eval_yaml_x", buf)
    assert ack["__ack__"] == "preload_normalizer_buffer"
    assert ack["eval_yaml_id"] == "eval_yaml_x"
    assert ack["n_keys"] == 2
    pool = _warmup_pool_mod.get_global_pool()
    assert "eval_yaml_x" in pool
    stored = pool.get("eval_yaml_x")
    assert stored == buf


def test_preload_buffer_rejects_invalid_eval_yaml_id(warmup_root) -> None:
    ack = wps._handle_preload_normalizer_buffer("../sneaky", {"k": [1.0]})
    assert ack["__ack__"] == "error"
    ack = wps._handle_preload_normalizer_buffer("a/b", {"k": [1.0]})
    assert ack["__ack__"] == "error"
    ack = wps._handle_preload_normalizer_buffer("", {"k": [1.0]})
    assert ack["__ack__"] == "error"


def test_preload_buffer_rejects_non_dict_buffer(warmup_root) -> None:
    ack = wps._handle_preload_normalizer_buffer("ev", [1, 2, 3])
    assert ack["__ack__"] == "error"


def test_preload_buffer_coerces_numeric_strings(warmup_root) -> None:
    """msgpack delivers Python int/float; coercion via float() handles both."""
    ack = wps._handle_preload_normalizer_buffer(
        "ev", {"k": [1, 2.0, 3]}
    )
    assert ack["__ack__"] == "preload_normalizer_buffer"
    pool_buf = _warmup_pool_mod.get_global_pool().get("ev")
    assert pool_buf == {"k": [1.0, 2.0, 3.0]}


# ---------------------------------------------------------------------------
# unload_warmup_buffer + double yaml_id naming integrity (R1-G1R4)
# ---------------------------------------------------------------------------


def test_unload_buffer_clears_pool_and_deletes_dump_file(warmup_root) -> None:
    """End-to-end double-yaml_id contract: client passes ``eval_yaml_id``;
    server derives ``<eval>__warmup`` and deletes only that dump file."""
    eval_id = "phase1_idx0"
    warmup_id = f"{eval_id}__warmup"
    pool = _warmup_pool_mod.get_global_pool()
    pool.set(eval_id, {"f1a_a_jerk": [0.1]})

    warmup_dump = warmup_root / f"{warmup_id}.jsonl"
    warmup_dump.write_text('{"x": 1}\n')

    # An UNRELATED file with the eval id (not the warmup id) MUST survive —
    # a regression here would mean the server is using the wrong name to
    # derive the path to delete (the original G1R3 bug).
    unrelated = warmup_root / f"{eval_id}.jsonl"
    unrelated.write_text("untouched\n")

    ack = wps._handle_unload_warmup_buffer(eval_id)
    assert ack["__ack__"] == "unload_warmup_buffer"
    assert ack["eval_yaml_id"] == eval_id
    assert ack["deleted_dump_file"] is True

    assert eval_id not in pool
    assert not warmup_dump.exists()
    assert unrelated.exists(), "server deleted the wrong file — name conflation regression"


def test_unload_buffer_idempotent_when_pool_empty(warmup_root) -> None:
    ack = wps._handle_unload_warmup_buffer("never_loaded")
    assert ack["__ack__"] == "unload_warmup_buffer"
    assert ack["deleted_dump_file"] is False


def test_unload_buffer_rejects_invalid_eval_yaml_id(warmup_root) -> None:
    ack = wps._handle_unload_warmup_buffer("../bad")
    assert ack["__ack__"] == "error"
    ack = wps._handle_unload_warmup_buffer("a/b")
    assert ack["__ack__"] == "error"
    ack = wps._handle_unload_warmup_buffer("")
    assert ack["__ack__"] == "error"


# ---------------------------------------------------------------------------
# Dispatcher integration: ctrl messages reach the new handlers
# ---------------------------------------------------------------------------


class _FakeConnection:
    """Same pattern as test_websocket_policy_server.py."""

    remote_address = ("test", 0)

    def __init__(self, incoming: list[bytes]) -> None:
        self._incoming = list(incoming)
        self.sent: list[Any] = []

    async def recv(self) -> bytes:
        if not self._incoming:
            import websockets
            raise websockets.ConnectionClosed(rcvd=None, sent=None)
        return self._incoming.pop(0)

    async def send(self, payload: Any) -> None:
        self.sent.append(payload)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        del code, reason  # signature parity with websockets.ServerConnection


def _pack(obj: Any) -> bytes:
    return msgpack_numpy.Packer().pack(obj)


def _ack_decoded(conn: _FakeConnection) -> list[dict]:
    return [msgpack_numpy.unpackb(p) for p in conn.sent[1:] if isinstance(p, (bytes, bytearray))]


def test_dispatcher_routes_three_new_ctrls(warmup_root) -> None:
    """Drive all three ctrls through ``_handler`` end-to-end so the wire
    field names are pinned (renaming would silently break the runner)."""
    target = warmup_root / "ev0__warmup.jsonl"
    target.write_text('{"k": 1}\n')
    # Snapshot bytes BEFORE the dispatcher runs — the unload ctrl below
    # deletes the file, so reading it after the run would fail.
    expected_bytes = target.read_bytes()

    policy = MagicMock(spec=["infer"])
    server = wps.WebsocketPolicyServer(policy=policy, metadata={"server": "test"})
    incoming = [
        _pack({"__ctrl__": "fetch_dump", "warmup_yaml_id": "ev0__warmup"}),
        _pack({
            "__ctrl__": "preload_normalizer_buffer",
            "eval_yaml_id": "ev0",
            "buffer": {"f1a_a_jerk": [0.1, 0.2]},
        }),
        _pack({"__ctrl__": "unload_warmup_buffer", "eval_yaml_id": "ev0"}),
    ]
    conn = _FakeConnection(incoming)
    asyncio.run(server._handler(conn))

    acks = _ack_decoded(conn)
    assert acks[0]["__ack__"] == "fetch_dump"
    assert acks[0]["content"] == expected_bytes
    assert acks[1]["__ack__"] == "preload_normalizer_buffer"
    assert acks[1]["n_keys"] == 1
    assert acks[2]["__ack__"] == "unload_warmup_buffer"
    assert acks[2]["deleted_dump_file"] is True
    # File deletion was the last act — verify it actually happened on disk.
    assert not target.exists()


def test_dispatcher_rejects_field_swap_between_eval_and_warmup_ids(warmup_root) -> None:
    """Sending ``yaml_id`` instead of ``warmup_yaml_id`` MUST fail — the
    handler reads the strict field, treats absence as empty string, and
    rejects empty as invalid."""
    target = warmup_root / "ev0__warmup.jsonl"
    target.write_text('{"k": 1}\n')

    policy = MagicMock(spec=["infer"])
    server = wps.WebsocketPolicyServer(policy=policy, metadata={"server": "test"})
    incoming = [
        # Wrong field name — server treats this as missing warmup_yaml_id.
        _pack({"__ctrl__": "fetch_dump", "yaml_id": "ev0__warmup"}),
    ]
    conn = _FakeConnection(incoming)
    asyncio.run(server._handler(conn))
    acks = _ack_decoded(conn)
    assert acks[0]["__ack__"] == "error"
    assert "invalid" in acks[0]["msg"]
