"""Unit tests for ``exp/_cache_config_rpc.py``.

Covers ``send_load_cache_config`` with a mock ``websockets.connect`` so no
real server is required. Per plan §19.B7, ``prefill_begin``/``prefill_end``
RPCs are intentionally not provided — prefill must travel on the same
connection as the subsequent ``infer`` calls, which is only achievable
through ``websocket_client_policy.prefill_trajectory``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import msgpack
import pytest

from exp import _cache_config_rpc

# ------------------------------------------------------------------
# Fake websocket
# ------------------------------------------------------------------


class _FakeWebsocket:
    """Minimal websocket stub. ``recv`` returns the queued messages in order."""

    def __init__(self, replies: list[bytes]) -> None:
        self._replies = list(replies)
        self.sent: list[bytes] = []
        # Populated by the fake ``connect`` wrapper so tests can assert the
        # URI the helper tried to reach.
        self.connected_uri: str | None = None
        self.recv = AsyncMock(side_effect=self._replies)
        self.send = AsyncMock(side_effect=self._record_send)

    async def _record_send(self, payload: bytes) -> None:
        self.sent.append(payload)


def _install_fake_connect(monkeypatch: pytest.MonkeyPatch, ws: _FakeWebsocket) -> None:
    @asynccontextmanager
    async def _connect(uri: str):
        # ``_send_ctrl`` calls ``websockets.connect(server_url)`` with no extra
        # kwargs, so a positional-only signature matches reality.
        ws.connected_uri = uri
        yield ws

    monkeypatch.setattr(_cache_config_rpc.websockets, "connect", _connect)


# ------------------------------------------------------------------
# send_load_cache_config
# ------------------------------------------------------------------


def test_send_load_cache_config_sends_yaml_and_returns_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text("enabled: true\n")
    # metadata packet (discarded) + ack packet.
    ws = _FakeWebsocket(
        replies=[
            msgpack.packb({"metadata": "hello"}),
            msgpack.packb({"__ack__": "load_cache_config", "version": 7}),
        ]
    )
    _install_fake_connect(monkeypatch, ws)

    version = _cache_config_rpc.send_load_cache_config("ws://fake", yaml_path)

    assert version == 7
    assert ws.connected_uri == "ws://fake"
    assert len(ws.sent) == 1
    sent_msg = msgpack.unpackb(ws.sent[0])
    assert sent_msg == {"__ctrl__": "load_cache_config", "yaml_content": "enabled: true\n"}


def test_send_load_cache_config_raises_on_ack_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text("")
    ws = _FakeWebsocket(
        replies=[
            msgpack.packb({"metadata": "x"}),
            msgpack.packb({"__ack__": "error", "msg": "bad yaml"}),
        ]
    )
    _install_fake_connect(monkeypatch, ws)

    with pytest.raises(RuntimeError, match="Control message failed"):
        _cache_config_rpc.send_load_cache_config("ws://fake", yaml_path)


def test_send_load_cache_config_returns_minus_one_when_version_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards the ``.get("version", -1)`` fallback: a buggy server that acks
    without the version field should surface as ``-1`` rather than KeyError."""
    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text("")
    ws = _FakeWebsocket(
        replies=[
            msgpack.packb({"metadata": "x"}),
            msgpack.packb({"__ack__": "load_cache_config"}),
        ]
    )
    _install_fake_connect(monkeypatch, ws)

    assert _cache_config_rpc.send_load_cache_config("ws://fake", yaml_path) == -1
