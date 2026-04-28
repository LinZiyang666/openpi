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
import os
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
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

    ``yaml_id`` records the eval-yaml stem that the verdict_factor_judge
    runner shim sent in the ``load_cache_config`` payload. Persists into
    ``build_per_connection_components(..., yaml_id=...)`` so the WarmupPool
    preload path can locate the matching buffer; ``None`` means the legacy
    code path runs unchanged.
    """

    config_path: str
    cache_config: object        # CacheConfig
    shared_storage: object      # CacheStorage
    version: int
    yaml_id: Optional[str] = None


_bundle_lock = threading.Lock()
_current_bundle: Optional[CurrentCacheBundle] = None
_bundle_version: int = 0


def get_current_cache_bundle() -> Optional[CurrentCacheBundle]:
    """Return current cache bundle snapshot (if any). Thread-safe."""
    with _bundle_lock:
        return _current_bundle


# ---------------------------------------------------------------------------
# Warmup dump root (verdict_factor_judge B2)
#
# Set once at server startup by ``scripts/serve_policy.py`` after creating
# the directory with mode 0o700 + ownership/mode self-check. The 3 new
# warmup ctrl handlers (`fetch_dump`, `unload_warmup_buffer`) refuse to run
# when this is None — that is the only safe failure mode for an experiment
# accidentally started without the flag.
# ---------------------------------------------------------------------------


_warmup_dump_root: Optional[Path] = None


def set_warmup_dump_root(path: Optional[Path]) -> None:
    """Install the resolved warmup dump root for the running server.

    Pass ``None`` to disable (the default state). The path MUST already be a
    real directory; this setter does not create it. Called once at startup.
    """
    global _warmup_dump_root
    _warmup_dump_root = Path(path).resolve() if path is not None else None


def get_warmup_dump_root() -> Optional[Path]:
    """Return the configured warmup dump root, or ``None`` if disabled."""
    return _warmup_dump_root


def _safe_resolve_under_root(name: str, root: Path) -> Optional[Path]:
    """Resolve ``<root>/<name>.jsonl`` and confirm it stays under ``root``.

    Returns ``None`` when traversal is detected (so the caller can reject
    with a uniform error ack). ``root`` MUST already be ``.resolve()``-d so
    symlink-following on the candidate side cannot escape it.
    """
    if not name or "/" in name or ".." in name:
        return None
    candidate = (root / f"{name}.jsonl").resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# verdict_factor_judge warmup ctrl helpers
#
# Sit at module scope so the dispatcher in ``_handler`` reads as a small
# router. Each returns the dict the dispatcher should pack and send.
# ---------------------------------------------------------------------------


def _fill_deferred_dump_paths(cache_config, yaml_id: Optional[str]) -> None:
    """Resolve every ``dump.deferred=True`` path under the configured root.

    Raises ``ValueError`` when a deferred dump exists but ``yaml_id`` is
    missing, or when no warmup dump root is configured. Mutates
    ``cache_config`` in place; non-deferred dumps are untouched.
    """
    has_deferred = any(
        cp.judge.dump is not None and getattr(cp.judge.dump, "deferred", False)
        for cp in cache_config.checkpoints.values()
    )
    if not has_deferred:
        return
    if not yaml_id:
        raise ValueError(
            "load_cache_config: yaml_id is required when the yaml carries a "
            "dump.deferred=True checkpoint"
        )
    root = get_warmup_dump_root()
    if root is None:
        raise ValueError(
            "load_cache_config: server was started without --warmup-dump-root, "
            "deferred dumps cannot be resolved"
        )
    candidate = _safe_resolve_under_root(yaml_id, root)
    if candidate is None:
        raise ValueError(
            f"load_cache_config: yaml_id {yaml_id!r} fails the "
            "warmup_dump_root allowlist (contains '/', '..', or escapes via symlink)"
        )
    for cp in cache_config.checkpoints.values():
        dump = cp.judge.dump
        if dump is not None and getattr(dump, "deferred", False):
            dump.path = str(candidate)
            dump.deferred = False  # downstream builders see a normal dump


def _handle_fetch_dump(warmup_yaml_id: str) -> dict:
    """Read a warmup dump file and return its bytes wrapped in an ack."""
    if not isinstance(warmup_yaml_id, str):
        return {"__ack__": "error", "msg": "warmup_yaml_id must be str"}
    root = get_warmup_dump_root()
    if root is None:
        return {"__ack__": "error", "msg": "warmup_dump_root not configured"}
    candidate = _safe_resolve_under_root(warmup_yaml_id, root)
    if candidate is None:
        return {"__ack__": "error", "msg": "invalid warmup_yaml_id"}
    if not candidate.exists():
        return {"__ack__": "error", "msg": "dump not found"}
    return {
        "__ack__": "fetch_dump",
        "warmup_yaml_id": warmup_yaml_id,
        "content": candidate.read_bytes(),
    }


def _handle_preload_normalizer_buffer(eval_yaml_id: str, buffer) -> dict:
    """Stash a per-key raw factor buffer in the WarmupPool keyed by eval yaml."""
    if not isinstance(eval_yaml_id, str) or not eval_yaml_id:
        return {"__ack__": "error", "msg": "eval_yaml_id must be a non-empty str"}
    if "/" in eval_yaml_id or ".." in eval_yaml_id:
        return {"__ack__": "error", "msg": "invalid eval_yaml_id"}
    if not isinstance(buffer, dict):
        return {"__ack__": "error", "msg": "buffer must be a dict"}
    # msgpack may decode list values as ndarray; coerce to plain Python lists
    # of floats so the WarmupPool's deep-copy semantics produce the same
    # representation regardless of what the wire delivered.
    coerced: dict[str, list[float]] = {}
    for k, v in buffer.items():
        if not isinstance(k, str):
            return {"__ack__": "error", "msg": "buffer keys must be str"}
        try:
            coerced[k] = [float(x) for x in v]
        except (TypeError, ValueError) as e:
            return {"__ack__": "error", "msg": f"buffer[{k!r}] not numeric: {e}"}
    from openpi.cache.warmup_pool import get_global_pool

    get_global_pool().set(eval_yaml_id, coerced)
    return {
        "__ack__": "preload_normalizer_buffer",
        "eval_yaml_id": eval_yaml_id,
        "n_keys": len(coerced),
    }


def _handle_unload_warmup_buffer(eval_yaml_id: str) -> dict:
    """Drop the WarmupPool entry AND delete the warmup dump file on disk."""
    if not isinstance(eval_yaml_id, str) or not eval_yaml_id:
        return {"__ack__": "error", "msg": "eval_yaml_id must be a non-empty str"}
    if "/" in eval_yaml_id or ".." in eval_yaml_id:
        return {"__ack__": "error", "msg": "invalid eval_yaml_id"}
    from openpi.cache.warmup_pool import get_global_pool

    get_global_pool().pop(eval_yaml_id)
    root = get_warmup_dump_root()
    deleted = False
    if root is not None:
        # Server derives the warmup name; the wire never carries it. This
        # closes the G1R3 name-conflation hole where a single yaml_id field
        # could be either the eval id or the warmup id.
        warmup_yaml_id = f"{eval_yaml_id}__warmup"
        candidate = _safe_resolve_under_root(warmup_yaml_id, root)
        if candidate is not None and candidate.exists():
            try:
                candidate.unlink()
                deleted = True
            except OSError as e:
                logger.warning("unload_warmup_buffer: failed to delete %s: %s", candidate, e)
    return {
        "__ack__": "unload_warmup_buffer",
        "eval_yaml_id": eval_yaml_id,
        "deleted_dump_file": deleted,
    }


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
            try:
                conn_policy = self._connection_policy_factory(self._policy)
            except Exception:
                logger.exception(
                    "connection_policy_factory failed for %s",
                    websocket.remote_address,
                )
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Failed to create per-connection policy.",
                )
                return
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
                        # Keyword dispatch (plan §21.S1.3): wrappers such as
                        # ``CollectionPolicy`` forward these kwargs through the
                        # chain; positional calls would break any wrapper that
                        # inserts an extra parameter (e.g. ``episode_name``).
                        if hasattr(conn_policy, "on_episode_start"):
                            conn_policy.on_episode_start(
                                experiment=obs.get("__experiment__", "unknown"),
                                task=obs.get("__task__", ""),
                                episode_id=obs.get("__episode_id__", -1),
                                episode_name=obs.get("__episode_name__", ""),
                                extra_metadata=obs.get("__extra__", {}),
                            )
                        await websocket.send(packer.pack({"__ack__": "episode_start"}))
                    elif ctrl == "episode_end":
                        # Keyword dispatch symmetric with ``episode_start``; see
                        # plan §22.4 must-fix MF-2.
                        if hasattr(conn_policy, "on_episode_end"):
                            conn_policy.on_episode_end(success=obs.get("__success__", False))
                        await websocket.send(packer.pack({"__ack__": "episode_end"}))
                    elif ctrl == "prefill_trajectory":
                        # Only exposed when this connection's policy is cache-
                        # wrapped (``InferenceInterceptor``). Cleanup/10 unifies
                        # the error path onto the msgpack ``{"__ack__": "error",
                        # "msg": ...}`` shape used everywhere else in this
                        # server — the pre-cleanup bare-string send broke the
                        # client's unpack path and silently lost context for
                        # any prefill exception raised on the worker thread.
                        if not hasattr(conn_policy, "prefill_trajectory"):
                            await websocket.send(packer.pack({
                                "__ack__": "error",
                                "msg": (
                                    "prefill_trajectory requires a cache-wrapped policy "
                                    "(InferenceInterceptor); this connection's policy "
                                    "does not expose it."
                                ),
                            }))
                            continue
                        # Run in a thread: the interceptor drives stage1 for
                        # every prefill step, so synchronous execution would
                        # block the asyncio event loop (health checks, other
                        # connections). Matches the dispatch of ``infer``.
                        try:
                            await asyncio.to_thread(
                                conn_policy.prefill_trajectory,
                                obs["observations"],
                                obs["actions"],
                                record=obs.get("record", False),
                                on_miss=obs.get("on_miss", "error"),
                            )
                        except Exception as exc:  # noqa: BLE001 — surface ANY error to client
                            logger.exception("prefill_trajectory failed on worker thread")
                            await websocket.send(packer.pack({
                                "__ack__": "error", "msg": str(exc),
                            }))
                            continue
                        await websocket.send(packer.pack({"__ack__": "prefill_trajectory"}))
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
                        yaml_content = obs.get("yaml_content", "")
                        # verdict_factor_judge B2: optional yaml_id is the
                        # eval/warmup yaml stem the runner registers so that
                        # (a) deferred dumps can be filled and (b) the
                        # WarmupPool preload finds its entry. Empty string
                        # preserves the legacy bundle.yaml_id=None semantic.
                        msg_yaml_id = obs.get("yaml_id", "") or None
                        if not yaml_path and not yaml_content:
                            await websocket.send(packer.pack({"__ack__": "error", "msg": "missing yaml_path or yaml_content"}))
                        else:
                            try:
                                from openpi.cache.config import build_shared_storage, load_cache_config

                                if yaml_content:
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                                        tmp.write(yaml_content)
                                        yaml_path = tmp.name
                                cache_config = load_cache_config(yaml_path)
                                # Fill deferred dump.path slots from the
                                # configured warmup dump root. The validator
                                # already passed (deferred=True bypasses path
                                # checks); now make the path concrete so the
                                # DumpingJudge built later sees a real file
                                # location.
                                _fill_deferred_dump_paths(cache_config, msg_yaml_id)
                                shared_storage = build_shared_storage(cache_config)
                                global _current_bundle, _bundle_version
                                with _bundle_lock:
                                    _bundle_version += 1
                                    _current_bundle = CurrentCacheBundle(
                                        config_path=yaml_path,
                                        cache_config=cache_config,
                                        shared_storage=shared_storage,
                                        version=_bundle_version,
                                        yaml_id=msg_yaml_id,
                                    )
                                version = _bundle_version
                                logger.info("Cache bundle updated to v%d: %s (yaml_id=%s)", version, yaml_path, msg_yaml_id)
                                await websocket.send(packer.pack({
                                    "__ack__": "load_cache_config",
                                    "yaml_path": yaml_path,
                                    "version": version,
                                }))
                            except Exception as e:
                                logger.error("Failed to load cache config %s: %s", yaml_path, e)
                                await websocket.send(packer.pack({"__ack__": "error", "msg": str(e)}))
                    elif ctrl == "fetch_dump":
                        # verdict_factor_judge B2: serve a previously-emitted
                        # warmup dump file back to the runner so it can derive
                        # per-key buffers for `preload_normalizer_buffer`. Path
                        # MUST stay under the configured root after .resolve()
                        # to defeat symlink-based traversal.
                        await websocket.send(packer.pack(
                            _handle_fetch_dump(obs.get("warmup_yaml_id", ""))
                        ))
                    elif ctrl == "preload_normalizer_buffer":
                        # verdict_factor_judge B2: stash the runner-derived
                        # raw factor buffers in the WarmupPool so that the
                        # next eval-yaml `build_per_connection_components`
                        # call preloads each composite normalizer.
                        await websocket.send(packer.pack(
                            _handle_preload_normalizer_buffer(
                                obs.get("eval_yaml_id", ""),
                                obs.get("buffer", {}),
                            )
                        ))
                    elif ctrl == "unload_warmup_buffer":
                        # verdict_factor_judge B2: clear the WarmupPool entry
                        # for this eval yaml AND delete the corresponding
                        # warmup dump file. Server derives the warmup name
                        # from the eval yaml id so the wire never carries the
                        # filesystem name (defense-in-depth against the
                        # name-conflation bug from G1R3).
                        await websocket.send(packer.pack(
                            _handle_unload_warmup_buffer(obs.get("eval_yaml_id", ""))
                        ))
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
