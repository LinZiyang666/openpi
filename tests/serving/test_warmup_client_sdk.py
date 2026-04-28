"""Tests for ``WebsocketClientPolicy`` warmup ctrl helpers (B2.4).

These pin the wire schema produced by the SDK side: every ctrl method must
emit exactly the field names the server's dispatcher reads, and every ack
type must be validated by ``_send_ctrl`` so a silent server bug surfaces as
a Python exception (not a misinterpreted return value).

Tests use a fake websocket so no real server is required; the SDK's
``__init__`` is bypassed via ``object.__new__`` to avoid the connect /
metadata-handshake roundtrip.
"""

from __future__ import annotations

from typing import Any

import pytest
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class _FakeWS:
    """Captures ``send`` payloads + replays canned ``recv`` responses."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.sent: list[Any] = []

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self) -> bytes | str:
        if not self._responses:
            raise RuntimeError("test ran out of canned responses")
        return self._responses.pop(0)


def _make_client(responses: list[Any]) -> WebsocketClientPolicy:
    """Build a client with a stub ws + packer, no real connect."""
    client = object.__new__(WebsocketClientPolicy)
    client._packer = msgpack_numpy.Packer()
    client._server_metadata = {}
    client._ws = _FakeWS(responses)
    client._uri = "ws://test/"
    client._api_key = None
    return client


def _last_sent_msg(client: WebsocketClientPolicy) -> dict:
    return msgpack_numpy.unpackb(client._ws.sent[-1])


# ---------------------------------------------------------------------------
# load_cache_config
# ---------------------------------------------------------------------------


def test_load_cache_config_emits_yaml_id_when_provided() -> None:
    client = _make_client(
        [msgpack_numpy.Packer().pack({"__ack__": "load_cache_config", "version": 7})]
    )
    ack = client.load_cache_config(yaml_content="keys: {}\n", yaml_id="phase1_idx0__warmup")
    assert ack["version"] == 7
    sent = _last_sent_msg(client)
    assert sent["__ctrl__"] == "load_cache_config"
    assert sent["yaml_content"] == "keys: {}\n"
    assert sent["yaml_id"] == "phase1_idx0__warmup"


def test_load_cache_config_omits_yaml_id_when_absent() -> None:
    """Backward compat: legacy callers omit yaml_id; the field must NOT
    appear on the wire (server treats absent as None)."""
    client = _make_client(
        [msgpack_numpy.Packer().pack({"__ack__": "load_cache_config", "version": 1})]
    )
    client.load_cache_config(yaml_path="/some.yaml")
    sent = _last_sent_msg(client)
    assert "yaml_id" not in sent
    assert sent["yaml_path"] == "/some.yaml"


def test_load_cache_config_requires_path_or_content() -> None:
    client = _make_client([])
    with pytest.raises(ValueError, match="yaml_path or yaml_content"):
        client.load_cache_config()


# ---------------------------------------------------------------------------
# fetch_dump
# ---------------------------------------------------------------------------


def test_fetch_dump_returns_content_bytes() -> None:
    payload = b'{"k":1}\n{"k":2}\n'
    client = _make_client(
        [msgpack_numpy.Packer().pack({
            "__ack__": "fetch_dump",
            "warmup_yaml_id": "ev0__warmup",
            "content": payload,
        })]
    )
    out = client.fetch_dump("ev0__warmup")
    assert out == payload
    sent = _last_sent_msg(client)
    assert sent == {"__ctrl__": "fetch_dump", "warmup_yaml_id": "ev0__warmup"}


def test_fetch_dump_raises_on_error_ack() -> None:
    client = _make_client(
        [msgpack_numpy.Packer().pack({"__ack__": "error", "msg": "dump not found"})]
    )
    with pytest.raises(RuntimeError, match="dump not found"):
        client.fetch_dump("missing")


# ---------------------------------------------------------------------------
# preload_normalizer_buffer
# ---------------------------------------------------------------------------


def test_preload_normalizer_buffer_round_trip() -> None:
    client = _make_client(
        [msgpack_numpy.Packer().pack({
            "__ack__": "preload_normalizer_buffer",
            "eval_yaml_id": "ev0",
            "n_keys": 2,
        })]
    )
    ack = client.preload_normalizer_buffer("ev0", {"f1a_a_jerk": [0.1, 0.2], "f2_var": [0.5]})
    assert ack["n_keys"] == 2
    sent = _last_sent_msg(client)
    assert sent["__ctrl__"] == "preload_normalizer_buffer"
    assert sent["eval_yaml_id"] == "ev0"
    assert sent["buffer"]["f1a_a_jerk"] == [0.1, 0.2]


def test_preload_normalizer_buffer_raises_on_error_ack() -> None:
    client = _make_client(
        [msgpack_numpy.Packer().pack({"__ack__": "error", "msg": "buffer must be dict"})]
    )
    with pytest.raises(RuntimeError, match="buffer must be dict"):
        client.preload_normalizer_buffer("ev0", {})


# ---------------------------------------------------------------------------
# unload_warmup_buffer
# ---------------------------------------------------------------------------


def test_unload_warmup_buffer_round_trip() -> None:
    client = _make_client(
        [msgpack_numpy.Packer().pack({
            "__ack__": "unload_warmup_buffer",
            "eval_yaml_id": "ev0",
            "deleted_dump_file": True,
        })]
    )
    ack = client.unload_warmup_buffer("ev0")
    assert ack["deleted_dump_file"] is True
    sent = _last_sent_msg(client)
    assert sent == {"__ctrl__": "unload_warmup_buffer", "eval_yaml_id": "ev0"}


def test_unload_warmup_buffer_does_not_send_warmup_id_directly() -> None:
    """The wire MUST NOT carry the warmup file name — the server derives
    ``<eval>__warmup`` from the eval id. Pinning this prevents a future
    refactor from re-introducing the G1R3 conflation."""
    client = _make_client(
        [msgpack_numpy.Packer().pack({
            "__ack__": "unload_warmup_buffer", "eval_yaml_id": "ev0", "deleted_dump_file": False,
        })]
    )
    client.unload_warmup_buffer("ev0")
    sent = _last_sent_msg(client)
    assert "warmup_yaml_id" not in sent


# ---------------------------------------------------------------------------
# Generic ack-mismatch detection
# ---------------------------------------------------------------------------


def test_send_ctrl_raises_on_unexpected_ack_type() -> None:
    """An ack with the wrong ``__ack__`` value (e.g. 'ignored' from an old
    server that does not implement the ctrl) MUST raise so callers don't
    silently treat the no-op as success."""
    client = _make_client(
        [msgpack_numpy.Packer().pack({"__ack__": "ignored"})]
    )
    with pytest.raises(RuntimeError, match="Unexpected server ack"):
        client.fetch_dump("anything")
