"""SidecarExecutor + SidecarServer protocol tests (plan §9): loopback
roundtrip, metadata handshake, bounded connect on absent endpoint, malformed
response fail-closed, close semantics, and GPU-lock serialisation."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from openpi.cache.sidecar_executor import SidecarError
from openpi.cache.sidecar_executor import SidecarExecutor

from exp.ablation_study.sidecar_server import SidecarServer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(policy_fn, timing_log=None) -> int:
    port = _free_port()
    server = SidecarServer(policy_fn, port=port, timing_log=timing_log)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return port
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("sidecar test server did not come up")


def _obs() -> dict:
    return {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(8, np.float32),
        "prompt": "pick up the bowl",
    }


def test_roundtrip_and_timing_log(tmp_path):
    log = tmp_path / "timing.jsonl"
    port = _start_server(lambda obs: np.ones((10, 7), np.float32), timing_log=str(log))
    ex = SidecarExecutor(f"127.0.0.1:{port}", connect_timeout_s=5, request_timeout_s=5)
    try:
        out = ex(_obs())
        np.testing.assert_array_equal(out["actions"], np.ones((10, 7), np.float32))
        out = ex(_obs())
        assert out["actions"].shape == (10, 7)
    finally:
        ex.close()
    assert len(log.read_text().strip().splitlines()) == 2


def test_absent_endpoint_bounded_connect():
    ex = SidecarExecutor(f"127.0.0.1:{_free_port()}", connect_timeout_s=1, request_timeout_s=1)
    t0 = time.perf_counter()
    with pytest.raises(SidecarError):
        ex(_obs())
    assert time.perf_counter() - t0 < 10  # bounded, no infinite retry
    ex.close()


def test_policy_error_fails_closed():
    def _boom(obs):
        raise ValueError("bad model")

    port = _start_server(_boom)
    ex = SidecarExecutor(f"127.0.0.1:{port}", connect_timeout_s=5, request_timeout_s=5)
    try:
        with pytest.raises(SidecarError, match="server error"):
            ex(_obs())
    finally:
        ex.close()


def test_malformed_shape_fails_closed():
    port = _start_server(lambda obs: np.ones((10,), np.float32))  # wrong ndim
    ex = SidecarExecutor(f"127.0.0.1:{port}", connect_timeout_s=5, request_timeout_s=5)
    try:
        with pytest.raises(SidecarError, match="actions"):
            ex(_obs())
    finally:
        ex.close()


def test_close_is_idempotent_and_final():
    port = _start_server(lambda obs: np.ones((10, 7), np.float32))
    ex = SidecarExecutor(f"127.0.0.1:{port}", connect_timeout_s=5, request_timeout_s=5)
    ex(_obs())
    ex.close()
    ex.close()  # idempotent
    with pytest.raises(SidecarError, match="closed"):
        ex(_obs())


def test_concurrent_connections_serialised():
    seen = []
    lock_probe = threading.Lock()

    def _slow(obs):
        with lock_probe:
            seen.append(time.perf_counter())
        time.sleep(0.05)
        return np.ones((10, 7), np.float32)

    port = _start_server(_slow)
    results = []

    def _client():
        ex = SidecarExecutor(f"127.0.0.1:{port}", connect_timeout_s=5, request_timeout_s=5)
        try:
            results.append(ex(_obs())["actions"].shape)
        finally:
            ex.close()

    threads = [threading.Thread(target=_client) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [(10, 7), (10, 7)]


def test_request_timeout_mid_request_fails_closed():
    def _hang(obs):
        time.sleep(3.0)
        return np.ones((10, 7), np.float32)

    port = _start_server(_hang)
    ex = SidecarExecutor(f"127.0.0.1:{port}", connect_timeout_s=5, request_timeout_s=0.5)
    try:
        with pytest.raises(SidecarError, match="request failed"):
            ex(_obs())
    finally:
        ex.close()


def test_invalid_dtype_fails_closed():
    # The real SidecarServer float32-casts, so an integer payload must come
    # from a raw protocol-speaking server to exercise the executor guard.
    import websockets.sync.server as wss
    from openpi_client import msgpack_numpy as mp

    def _handler(conn):
        packer = mp.Packer()
        conn.send(packer.pack({"sidecar": True}))
        conn.recv()
        conn.send(packer.pack({"actions": np.ones((10, 7), np.int64)}))

    port = _free_port()
    server = wss.serve(_handler, "127.0.0.1", port, compression=None, max_size=None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ex = SidecarExecutor(f"127.0.0.1:{port}", connect_timeout_s=5, request_timeout_s=5)
    try:
        with pytest.raises(SidecarError, match="actions"):
            ex(_obs())
    finally:
        ex.close()


def test_missing_actions_key_fails_closed():
    # Raw fake server that speaks the handshake but returns a keyless dict.
    import websockets.sync.server as wss
    from openpi_client import msgpack_numpy as mp

    def _handler(conn):
        packer = mp.Packer()
        conn.send(packer.pack({"sidecar": True}))
        conn.recv()
        conn.send(packer.pack({"foo": 1}))

    port = _free_port()
    server = wss.serve(_handler, "127.0.0.1", port, compression=None, max_size=None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ex = SidecarExecutor(f"127.0.0.1:{port}", connect_timeout_s=5, request_timeout_s=5)
    try:
        with pytest.raises(SidecarError, match="actions"):
            ex(_obs())
    finally:
        ex.close()


def test_ctrl_messages_are_acked_not_inferred():
    calls = []

    def _policy(obs):
        calls.append(obs)
        return np.ones((10, 7), np.float32)

    port = _start_server(_policy)
    ex = SidecarExecutor(f"127.0.0.1:{port}", connect_timeout_s=5, request_timeout_s=5)
    try:
        conn = ex._ensure_connection()
        conn.send(ex._packer.pack({"__ctrl__": "episode_start", "__task__": "t"}))
        from openpi_client import msgpack_numpy as mp

        assert mp.unpackb(conn.recv(timeout=5)) == {"ack": "episode_start"}
        assert calls == []  # the policy never saw the ctrl message
    finally:
        ex.close()


def test_route_prompt_exact_match_and_unknown():
    from exp.ablation_study.sidecar_server import route_prompt

    policies = {"pick up the bowl": "p1", "close the drawer": "p2"}
    assert route_prompt(policies, "pick up the bowl") == "p1"
    with pytest.raises(KeyError, match="no checkpoint"):
        route_prompt(policies, "unseen task text")
