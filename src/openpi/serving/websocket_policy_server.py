"""WebSocket server that exposes a policy for remote inference.

Overview
--------
``WebsocketPolicyServer`` serves ``infer`` requests over a msgpack-encoded
WebSocket protocol.  See ``openpi_client.websocket_client_policy`` for the
matching client.

Concurrency modes
-----------------
* **Single-connection** (default, ``concurrent=False``): accepts one client at
  a time.  Additional connections are rejected with close code 1013.
* **Concurrent** (``concurrent=True``): accepts multiple simultaneous clients.
  Each connection gets its own policy wrapper stack (created via the
  ``connection_policy_factory``), sharing the same base policy (GPU model).
  ``asyncio.to_thread`` is used for ``policy.infer()`` so the event loop is
  never blocked.

Timing integration
------------------
When the wrapped policy implements the ``TaskLifecycle`` protocol (i.e. it
has ``on_task_begin`` and ``on_task_end`` methods — as ``InferenceInterceptor``
does), the server calls:

* ``on_task_begin()`` when a client connection **opens**.
* ``on_task_end()`` when a client connection **closes** (or errors out).

``on_task_end()`` triggers ``SystemTimer.on_task_end()``, which prints a
per-probe timing summary to the terminal and writes a CSV (if configured).

When the policy does *not* implement ``TaskLifecycle`` (plain ``Policy``
without ``--cache``), the ``hasattr`` checks are simply skipped — no
behaviour change for the non-cache path.

Note on ``stage_timing`` removal
---------------------------------
The Step 1 design collected per-inference ``stage_timing`` dicts from the
action output and aggregated them manually at connection close.  That logic
has been removed in Step 2.  Timing aggregation is now entirely the
responsibility of ``SystemTimer``, which is called via the ``TaskLifecycle``
hooks above.  The ``server_timing`` key (wall-clock infer time measured by
this server) is still present in every action response.

Currently only the ``load`` and ``infer`` methods of the client protocol
are implemented.
"""

import asyncio
import http
import logging
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames


# ---------------------------------------------------------------------------
# Global cache bundle (for dynamic YAML switching in experiments)
# ---------------------------------------------------------------------------


@dataclass
class CurrentCacheBundle:
    """Snapshot of the current cache config for experiment runs.

    Atomically replaced by load_cache_config control messages.
    Same-run worker connections share the same shared_storage.
    """

    config_path: str
    cache_config: object        # CacheConfig
    shared_storage: object      # CacheStorage
    version: int


_bundle_lock = threading.Lock()
_current_bundle: Optional[CurrentCacheBundle] = None
_bundle_version: int = 0


def get_current_cache_bundle() -> Optional[CurrentCacheBundle]:
    """Return current cache bundle snapshot (if any). Thread-safe."""
    with _bundle_lock:
        return _current_bundle

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the WebSocket protocol.

    Args:
        policy: Any object implementing ``BasePolicy.infer()``.  In
                single-connection mode this is the fully-wrapped policy.  In
                concurrent mode this is the *base* (unwrapped) policy shared
                across all connections.
                If the policy also implements the ``TaskLifecycle`` protocol
                (``on_task_begin`` / ``on_task_end``), the server calls those
                methods at connection open / close.
        host: Bind address (default ``"0.0.0.0"`` — all interfaces).
        port: TCP port.  ``None`` lets the OS choose a free port.
        metadata: Arbitrary dict sent to the client immediately after the
                  WebSocket handshake.  Typically includes robot type, action
                  shape, etc.  Defaults to ``{}``.
        concurrent: When ``True``, allow multiple simultaneous client
                    connections with per-connection wrapper stacks.
        connection_policy_factory: Called once per new connection in
                    concurrent mode.  Receives the base ``policy`` and returns
                    a per-connection wrapper stack.  Required when
                    ``concurrent=True``.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        concurrent: bool = False,
        connection_policy_factory: Optional[Callable[[_base_policy.BasePolicy], _base_policy.BasePolicy]] = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._concurrent = concurrent
        self._connection_policy_factory = connection_policy_factory
        self._has_active_connection = False  # for single-connection mode
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        # ------------------------------------------------------------------
        # Determine per-connection policy
        # ------------------------------------------------------------------
        if self._concurrent:
            # Concurrent mode: create a fresh wrapper stack for this connection.
            conn_policy = self._connection_policy_factory(self._policy)
        else:
            # Single-connection mode: reject if another connection is active.
            if self._has_active_connection:
                logger.warning(
                    "Rejected connection from %s — server is in single-connection mode.",
                    websocket.remote_address,
                )
                await websocket.close(
                    code=1013,
                    reason="Server is in single-connection mode. Only one client at a time.",
                )
                return
            self._has_active_connection = True
            conn_policy = self._policy

        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Notify the policy that a new task (connection) is starting.
        # InferenceInterceptor implements on_task_begin(); plain Policy does not.
        # The hasattr check keeps this server decoupled from cache internals.
        if hasattr(conn_policy, "on_task_begin"):
            conn_policy.on_task_begin()

        await websocket.send(packer.pack(self._metadata))

        # prev_total_time tracks total round-trip time of the *previous*
        # request (infer + send) so it can be reported in the *next* response.
        # This gives the client visibility into end-to-end cycle time.
        prev_total_time = None

        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if "__ctrl__" in obs:
                    ctrl = obs["__ctrl__"]
                    if ctrl == "episode_start":
                        if hasattr(conn_policy, "on_episode_start"):
                            conn_policy.on_episode_start(
                                obs.get("__experiment__", "unknown"),
                                obs.get("__task__", ""),
                                obs.get("__episode_id__", -1),
                            )
                        await websocket.send(packer.pack({"__ack__": "episode_start"}))
                    elif ctrl == "episode_end":
                        if hasattr(conn_policy, "on_episode_end"):
                            conn_policy.on_episode_end(obs.get("__success__", False))
                        await websocket.send(packer.pack({"__ack__": "episode_end"}))
                    elif ctrl == "load_cache_config":
                        if not self._concurrent:
                            msg = (
                                "load_cache_config requires --concurrent mode. "
                                "In single-connection mode the policy is wrapped once at startup "
                                "and cannot be dynamically switched."
                            )
                            logger.error(msg)
                            await websocket.send(packer.pack({"__ack__": "error", "msg": msg}))
                            continue
                        yaml_path = obs.get("yaml_path", "")
                        if not yaml_path:
                            await websocket.send(packer.pack({"__ack__": "error", "msg": "missing yaml_path"}))
                        else:
                            try:
                                from openpi.cache.config import build_shared_storage, load_cache_config

                                cache_config = load_cache_config(yaml_path)
                                shared_storage = build_shared_storage(cache_config)
                                global _current_bundle, _bundle_version
                                with _bundle_lock:
                                    _bundle_version += 1
                                    _current_bundle = CurrentCacheBundle(
                                        config_path=yaml_path,
                                        cache_config=cache_config,
                                        shared_storage=shared_storage,
                                        version=_bundle_version,
                                    )
                                version = _bundle_version
                                logger.info("Cache bundle updated to v%d: %s", version, yaml_path)
                                await websocket.send(packer.pack({
                                    "__ack__": "load_cache_config",
                                    "yaml_path": yaml_path,
                                    "version": version,
                                }))
                            except Exception as e:
                                logger.error("Failed to load cache config %s: %s", yaml_path, e)
                                await websocket.send(packer.pack({"__ack__": "error", "msg": str(e)}))
                    else:
                        await websocket.send(packer.pack({"__ack__": "ignored"}))
                    continue

                infer_time = time.monotonic()
                # Run blocking inference in a thread so the asyncio event
                # loop stays responsive for other connections and health checks.
                action = await asyncio.to_thread(conn_policy.infer, obs)
                infer_time = time.monotonic() - infer_time

                # server_timing: wall-clock time spent in policy.infer(),
                # as measured by this server.  Does not include network IO.
                # For per-stage GPU timing, see SystemTimer output at task end.
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # Round-trip time of the previous cycle (infer + network send).
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                # Notify the policy that the task has ended.
                # SystemTimer.on_task_end() will print the per-probe timing
                # summary and (if configured) write a CSV file.
                if hasattr(conn_policy, "on_task_end"):
                    conn_policy.on_task_end()
                if not self._concurrent:
                    self._has_active_connection = False
                break

            except Exception:
                # Send the traceback to the client before closing so remote
                # debugging is possible without SSH access to the server.
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                # Ensure task-end hooks run even on error so timing records
                # are not silently lost (e.g. CSV is still flushed).
                if hasattr(conn_policy, "on_task_end"):
                    conn_policy.on_task_end()
                if not self._concurrent:
                    self._has_active_connection = False
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    """Respond to ``GET /healthz`` with 200 OK; pass other requests through."""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal WebSocket handshake for all other paths.
    return None
