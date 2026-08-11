"""Bounded, fail-closed websocket executor for ablation routing.

``SidecarExecutor`` is the callable injected into ``InferenceInterceptor`` as
``hit_executor`` / ``miss_executor`` (ablation_study plan §6.1). It forwards
the untransformed client observation dict to a sidecar policy server speaking
the openpi websocket msgpack protocol and returns the sidecar's output dict
(client-space ``{"actions": ...}``) verbatim.

Failure semantics are deliberately fail-closed: connect timeout, request
timeout, connection loss, or a malformed response all raise ``SidecarError``
so the episode fails and the conductor's episode-level retry takes over. The
executor never falls back to the wrapped Pi0.5 policy — a silent fallback
would corrupt the routed arm's semantics.

The connection is established lazily on first call with a hard deadline
(``connect_timeout_s``), by a direct bounded ``websockets.sync.client.connect``
plus the protocol's metadata handshake. It deliberately does NOT reuse
``WebsocketClientPolicy._wait_for_server`` (an unbounded retry loop; wrapping
it in an outer timeout would leak a live thread and socket attempt).

Coupling map:
  DEPENDS ON:  websockets.sync.client, openpi_client.msgpack_numpy
  CONSUMED BY: scripts/serve_policy.py _wrap_policy (construction),
               InferenceInterceptor (call + close via on_task_end)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)


class SidecarError(RuntimeError):
    """Raised on any sidecar connect/request/response failure (fail-closed)."""


def probe_endpoint(endpoint: str, timeout_s: float = 5.0) -> None:
    """One-shot reachability probe: bounded connect + metadata handshake.

    Raises ``SidecarError`` when the endpoint is absent or does not speak the
    protocol. Used by ``_wrap_policy`` at wrapper-construction time so a
    missing sidecar is rejected before the first inference.
    """
    ex = SidecarExecutor(endpoint, connect_timeout_s=timeout_s, request_timeout_s=timeout_s)
    try:
        ex._ensure_connection()
    finally:
        ex.close()


class SidecarExecutor:
    """Callable ``obs dict -> outputs dict`` over a bounded websocket session."""

    def __init__(
        self,
        endpoint: str,
        *,
        connect_timeout_s: float = 10.0,
        request_timeout_s: float = 30.0,
        label: str = "sidecar",
    ) -> None:
        host, _, port = endpoint.rpartition(":")
        if not host or not port:
            raise SidecarError(f"sidecar endpoint {endpoint!r} is not 'host:port'")
        self._uri = f"ws://{host}:{port}"
        self._connect_timeout_s = float(connect_timeout_s)
        self._request_timeout_s = float(request_timeout_s)
        self._label = label
        self._packer = msgpack_numpy.Packer()
        self._conn: websockets.sync.client.ClientConnection | None = None
        self._metadata: dict | None = None
        # One executor instance serves one server connection, but close() may
        # race an in-flight call on teardown; a lock keeps socket use serial.
        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _ensure_connection(self) -> websockets.sync.client.ClientConnection:
        if self._closed:
            raise SidecarError(f"{self._label}: executor already closed")
        if self._conn is not None:
            return self._conn
        conn = None
        try:
            conn = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                open_timeout=self._connect_timeout_s,
            )
            # Protocol handshake: server sends one metadata dict on accept.
            self._metadata = msgpack_numpy.unpackb(
                conn.recv(timeout=self._connect_timeout_s)
            )
        except Exception as e:
            # A handshake failure must not orphan the freshly opened socket.
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            raise SidecarError(
                f"{self._label}: cannot establish bounded connection to "
                f"{self._uri} within {self._connect_timeout_s}s: {e}"
            ) from e
        self._conn = conn
        logger.info("%s: connected to %s", self._label, self._uri)
        return conn

    def close(self) -> None:
        """Idempotent, deterministic teardown (called from on_task_end)."""
        with self._lock:
            self._closed = True
            conn, self._conn = self._conn, None
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001 — teardown must not raise
                    logger.warning("%s: error closing connection", self._label, exc_info=True)

    # ------------------------------------------------------------------
    # Executor call
    # ------------------------------------------------------------------

    def __call__(self, obs: dict) -> dict:
        with self._lock:
            conn = self._ensure_connection()
            try:
                conn.send(self._packer.pack(obs))
                response = conn.recv(timeout=self._request_timeout_s)
            except Exception as e:
                self._drop_connection()
                raise SidecarError(
                    f"{self._label}: request failed within {self._request_timeout_s}s: {e}"
                ) from e
        if isinstance(response, str):
            raise SidecarError(f"{self._label}: server error: {response}")
        outputs = msgpack_numpy.unpackb(response)
        self._validate_outputs(outputs)
        return outputs

    def _drop_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def _validate_outputs(self, outputs: Any) -> None:
        """Reject malformed sidecar responses before they reach the client."""
        if not isinstance(outputs, dict) or "actions" not in outputs:
            raise SidecarError(
                f"{self._label}: response is not a dict with an 'actions' key: "
                f"{type(outputs).__name__}"
            )
        actions = np.asarray(outputs["actions"])
        if actions.ndim != 2 or not np.issubdtype(actions.dtype, np.floating):
            raise SidecarError(
                f"{self._label}: 'actions' must be a float [horizon, dim] array, "
                f"got shape={actions.shape} dtype={actions.dtype}"
            )
