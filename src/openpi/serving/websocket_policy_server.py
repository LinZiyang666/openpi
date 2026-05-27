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
from collections.abc import Callable
from dataclasses import dataclass
import http
import logging
from pathlib import Path
import threading
import time
import traceback

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
    yaml_id: str | None = None


_bundle_lock = threading.Lock()
_current_bundle: CurrentCacheBundle | None = None
_bundle_version: int = 0

# M2 multi-bundle support (Phase 3): single-server hosts multiple yamls
# simultaneously, addressed by bundle_id. The legacy single-latest
# ``_current_bundle`` is preserved as the default-bundle slot and as the
# "current" pointer for callers that do not declare a bundle_id (backward
# compat for old clients).
_bundles: dict[str, "CurrentCacheBundle"] = {}
_DEFAULT_BUNDLE_ID = "default"


def get_current_cache_bundle(bundle_id: str | None = None) -> CurrentCacheBundle | None:
    """Return a cache bundle snapshot, looked up by bundle_id when given.

    Lookup order:
      - explicit ``bundle_id``: returns ``_bundles[bundle_id]`` or ``None``;
      - ``bundle_id is None``: returns the legacy single-latest bundle
        (``_current_bundle``) — backward compat for callers that don't yet
        understand bundle ids (old client code paths and ``_wrap_policy``).
    """
    with _bundle_lock:
        if bundle_id is not None:
            return _bundles.get(bundle_id)
        return _current_bundle


def get_cache_bundle_ids() -> list[str]:
    """Return the list of currently-loaded bundle ids (snapshot)."""
    with _bundle_lock:
        return list(_bundles.keys())


# ---------------------------------------------------------------------------
# Warmup dump root (verdict_factor_judge B2)
#
# Set once at server startup by ``scripts/serve_policy.py`` after creating
# the directory with mode 0o700 + ownership/mode self-check. The 3 new
# warmup ctrl handlers (`fetch_dump`, `unload_warmup_buffer`) refuse to run
# when this is None — that is the only safe failure mode for an experiment
# accidentally started without the flag.
# ---------------------------------------------------------------------------


_warmup_dump_root: Path | None = None


def set_warmup_dump_root(path: Path | None) -> None:
    """Install the resolved warmup dump root for the running server.

    Pass ``None`` to disable (the default state). The path MUST already be a
    real directory; this setter does not create it. Called once at startup.
    """
    global _warmup_dump_root
    _warmup_dump_root = Path(path).resolve() if path is not None else None


def get_warmup_dump_root() -> Path | None:
    """Return the configured warmup dump root, or ``None`` if disabled."""
    return _warmup_dump_root


def _safe_resolve_under_root(name: str, root: Path) -> Path | None:
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


def _fill_deferred_dump_paths(cache_config, yaml_id: str | None) -> None:
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


# ---------------------------------------------------------------------------
# Memory-monitoring ctrl helpers (Phase 7: GPU mem attribution)
#
# All four handlers gate on ``OPENPI_MONITOR_LEVEL`` (resolved at server
# startup) so a single env var is the master switch. The recorder lives in
# ``openpi.serving.monitor`` and holds everything in-memory until a
# ``dump_metrics`` ctrl pulls the whole timeline back to the client.
# ---------------------------------------------------------------------------


def _handle_dump_metrics(payload: dict) -> dict:
    """Return the in-memory metrics buffers (util / batch / timing / mem).

    The ``mem_snapshots`` entries each carry a ``snapshot`` Python object
    that msgpack cannot serialise directly (segment dicts contain str
    keys but no tensors, so normally fine — we still pickle each snapshot
    per-entry as ``snapshot_pickle`` bytes to defeat future schema drift).

    Optional payload keys (all bool, all default True):
        include_util, include_batch, include_mem, include_timing
        clear: if True, ALSO clear the buffers after dumping (so the next
               sweep iteration starts fresh).
    """
    from openpi.serving import monitor as _monitor

    level = _monitor.get_monitor_level()
    if level <= _monitor.MonitorLevel.OFF:
        return {
            "__ack__": "error",
            "msg": f"dump_metrics requires OPENPI_MONITOR_LEVEL >= BASIC (current: {level.name})",
        }
    rec = _monitor.get_recorder()
    inc_util = payload.get("include_util", True)
    inc_batch = payload.get("include_batch", True)
    inc_mem = payload.get("include_mem", True)
    inc_timing = payload.get("include_timing", True)
    do_clear = payload.get("clear", False)
    out = rec.dump_all(
        include_util=inc_util,
        include_batch=inc_batch,
        include_mem=inc_mem,
        include_timing=inc_timing,
    )
    # Pickle each snapshot tree separately so a partial decode failure on
    # the client side never corrupts the rest of the payload.
    if "mem_snapshots" in out:
        import pickle
        repacked = []
        for entry in out["mem_snapshots"]:
            e = dict(entry)
            snap = e.pop("snapshot", None)
            try:
                e["snapshot_pickle"] = pickle.dumps(snap, protocol=4)
            except Exception as exc:
                logger.warning("dump_metrics: snapshot pickle failed: %s", exc)
                e["snapshot_pickle"] = b""
            repacked.append(e)
        out["mem_snapshots"] = repacked
    out["stats"] = rec.stats()
    out["level"] = level.name
    if do_clear:
        out["cleared"] = rec.clear(
            util=inc_util, batch=inc_batch, mem=inc_mem, timing=inc_timing,
        )
    return {"__ack__": "dump_metrics", **out}


def _handle_snapshot_mem(payload: dict) -> dict:
    """Capture ``torch.cuda.memory_snapshot()`` + ``memory_summary()`` at
    the current moment and stash in the recorder under ``label``.

    Payload: {"label": str}. ``label`` defaults to a timestamp-based slug.
    """
    from openpi.serving import monitor as _monitor

    level = _monitor.get_monitor_level()
    if level < _monitor.MonitorLevel.SNAPSHOT:
        return {
            "__ack__": "error",
            "msg": (
                f"snapshot_mem requires OPENPI_MONITOR_LEVEL >= SNAPSHOT "
                f"(current: {level.name})"
            ),
        }
    label = payload.get("label") or f"snap_{time.strftime('%Y%m%d_%H%M%S')}"
    result = _monitor.take_memory_snapshot(label)
    if not result.get("ok"):
        return {"__ack__": "error", "msg": result.get("reason", "unknown")}
    return {"__ack__": "snapshot_mem", **result}


def _handle_mem_history_start(payload: dict) -> dict:
    """Turn on ``_record_memory_history``. Requires level=HISTORY."""
    from openpi.serving import monitor as _monitor

    max_entries = int(payload.get("max_entries", 100_000))
    context = payload.get("context", "python")
    stacks = payload.get("stacks", "python")
    result = _monitor.enable_memory_history(
        max_entries=max_entries, context=context, stacks=stacks,
    )
    if not result.get("ok"):
        return {"__ack__": "error", "msg": result.get("reason", "unknown"), **result}
    return {"__ack__": "mem_history_start", **result}


def _handle_mem_history_stop(payload: dict) -> dict:
    """Turn off ``_record_memory_history``."""
    from openpi.serving import monitor as _monitor

    result = _monitor.disable_memory_history()
    if not result.get("ok"):
        return {"__ack__": "error", "msg": result.get("reason", "unknown"), **result}
    return {"__ack__": "mem_history_stop", **result}


def _handle_set_batch_params(payload: dict) -> dict:
    """Hot-mutate the coordinator's max_batch_size / max_wait_ms without
    restarting the server. Useful for sweeping param grids in benchmarks.
    """
    from openpi.serving.batching_coordinator import get_active_coordinator
    coord = get_active_coordinator()
    if coord is None:
        return {"__ack__": "error", "msg": "no active coordinator (single-connection mode?)"}
    res = coord.update_batch_params(
        max_batch_size=payload.get("max_batch_size"),
        max_wait_ms=payload.get("max_wait_ms"),
    )
    return {"__ack__": "set_batch_params", **res}


def _handle_throughput_summary(payload: dict) -> dict:
    """Compute throughput / batching / GPU util stats from the current
    in-memory buffers. Optionally clear buffers after (``clear=True``).
    """
    from openpi.serving import monitor as _monitor

    level = _monitor.get_monitor_level()
    if level <= _monitor.MonitorLevel.OFF:
        return {"__ack__": "error", "msg": "OPENPI_MONITOR_LEVEL=off"}
    rec = _monitor.get_recorder()
    summary = rec.throughput_summary()
    if payload.get("clear", False):
        cleared = rec.clear(util=True, batch=True, timing=False, mem=False)
        summary["cleared"] = cleared
    return {"__ack__": "throughput_summary", **summary}


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
        connection_policy_factory: Callable[..., _base_policy.BasePolicy] | None = None,
        ready_callback: Callable[[], None] | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._concurrent = concurrent
        self._connection_policy_factory = connection_policy_factory
        self._has_active_connection = False  # for single-connection mode
        # Fired exactly once, after the listen socket is bound — lets a
        # supervisor (replica scale-out) wait for "actually serving" rather
        # than racing the bind.
        self._ready_callback = ready_callback
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            # 256 MB cap (was None): bound inbound frame size so an
            # unauthenticated client on the public ingress cannot OOM the server
            # with a multi-GB frame. Far above any legitimate obs / prefill
            # payload (a few MB at most).
            max_size=256 * 1024 * 1024,
            process_request=_health_check,
        ) as server:
            if self._ready_callback is not None:
                self._ready_callback()
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        # ------------------------------------------------------------------
        # Determine per-connection policy
        # ------------------------------------------------------------------
        # M2 (Phase 3, G2 R1 Item 2) — lazy lifecycle for concurrent mode:
        # the per-connection wrapper stack is no longer created at handler
        # entry. We wait for the first ``__ctrl__`` ``select_bundle`` or
        # ``episode_start{bundle_id}`` so the factory receives the correct
        # ``bundle_id`` (M2). Old clients that omit ``bundle_id`` fall back
        # to ``_DEFAULT_BUNDLE_ID`` so the LIBERO main.py worker that has not
        # been upgraded still works end-to-end.
        conn_policy = None
        bound_bundle_id: str | None = None
        if not self._concurrent:
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
        # Event-driven monitor hook: on connection open, bump the
        # process-wide active-connection counter AND ask the monitor to
        # take a snapshot labelled with the new count. At level<SNAPSHOT
        # this is a no-op so basic mode is unaffected. The snapshot
        # captures the moment ALL prior connections are still alive
        # alongside this brand-new one — peak per-conn observation.
        from openpi.serving import monitor as _server_monitor
        _conn_count_after_open = _server_monitor.increment_connection()
        _server_monitor.fire_event(
            "conn_open", payload={"conn_count": _conn_count_after_open},
        )
        packer = msgpack_numpy.Packer()

        def _bind_bundle(bundle_id: str) -> None:
            """Lazy factory call: create wrapper stack bound to ``bundle_id``
            and fire ``on_task_begin``. Idempotent — first call binds, later
            calls with the same id are no-ops; a different id swaps the
            wrapper (on_task_end on the old stack, then re-bind).
            """
            nonlocal conn_policy, bound_bundle_id
            if conn_policy is not None and bound_bundle_id == bundle_id:
                return
            # Build the new stack first so a factory failure leaves the old
            # binding intact rather than half torn down.
            new_policy = self._connection_policy_factory(self._policy, bundle_id)
            if conn_policy is not None and hasattr(conn_policy, "on_task_end"):
                try:
                    conn_policy.on_task_end()
                except Exception:
                    logger.exception("on_task_end failed during bundle switch")
            # Commit the swap BEFORE on_task_begin: if on_task_begin raises,
            # conn_policy must reference the new (current) stack, never the
            # old one that was just on_task_end'd — otherwise later infer/close
            # would run against an already-ended policy.
            conn_policy = new_policy
            bound_bundle_id = bundle_id
            if hasattr(new_policy, "on_task_begin"):
                new_policy.on_task_begin()
            logger.info("connection %s bound to bundle %r",
                        websocket.remote_address, bundle_id)

        # Non-concurrent mode keeps its on_task_begin behaviour (C1 preserved).
        # on_task_begin and the metadata send both run BEFORE the request
        # loop's try/except, so a failure here (client dropped mid-handshake,
        # or on_task_begin raised) would otherwise escape _handler without
        # decrementing the active-connection counter — leaking the count and,
        # in single-connection mode, permanently rejecting all future clients.
        try:
            if not self._concurrent and hasattr(conn_policy, "on_task_begin"):
                conn_policy.on_task_begin()
            await websocket.send(packer.pack(self._metadata))
        except Exception:
            # Mirror the in-loop teardown: close a half-run on_task_begin (flush
            # timing CSV etc.) rather than leaking a half-begun policy.
            if hasattr(conn_policy, "on_task_end"):
                try:
                    conn_policy.on_task_end()
                except Exception:
                    logger.exception("on_task_end failed during handshake teardown")
            if not self._concurrent:
                self._has_active_connection = False
            from openpi.serving import monitor as _server_monitor
            _server_monitor.decrement_connection()
            raise

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
                        # M2 (lazy lifecycle, G2 R1 Item 2): the very first
                        # ``episode_start`` from a concurrent connection binds
                        # the wrapper stack if no ``select_bundle`` came in
                        # first. Old clients that never send ``select_bundle``
                        # land here with whatever ``bundle_id`` they declare
                        # (or the ``"default"`` slot).
                        if self._concurrent and conn_policy is None:
                            bundle_id = obs.get("bundle_id", _DEFAULT_BUNDLE_ID)
                            try:
                                _bind_bundle(bundle_id)
                            except Exception as exc:
                                logger.exception("episode_start bundle bind failed")
                                await websocket.send(packer.pack({
                                    "__ack__": "error",
                                    "msg": f"episode_start bundle bind failed: {exc}",
                                }))
                                continue
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
                        # Event-driven snapshot + autoflush at episode boundary.
                        # Cooldown (15s in monitor.py) prevents flush storms when
                        # episodes are short. No-op at level<SNAPSHOT.
                        from openpi.serving import monitor as _server_monitor
                        _ep_n = _server_monitor.current_connection_count()
                        _server_monitor.fire_event(
                            "episode_end",
                            payload={"conn_count": _ep_n, "success": obs.get("__success__", False)},
                        )
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
                        except Exception as exc:
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
                        # M2 (Phase 3): optional bundle_id field; falls back to
                        # yaml_id, then to "default" for legacy clients that do
                        # not yet declare a bundle_id.
                        msg_bundle_id = obs.get("bundle_id", "") or msg_yaml_id or _DEFAULT_BUNDLE_ID
                        if not yaml_path and not yaml_content:
                            await websocket.send(packer.pack({"__ack__": "error", "msg": "missing yaml_path or yaml_content"}))
                        else:
                            try:
                                from openpi.cache.config import build_shared_storage
                                from openpi.cache.config import load_cache_config

                                if yaml_content:
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                                        tmp.write(yaml_content)
                                        yaml_path = tmp.name
                                cache_config = load_cache_config(yaml_path)
                                # Runtime write-frozen contract (C2, M4.5):
                                # fail fast on any write-enabled config arriving
                                # via ``load_cache_config`` ctrl msg, matching the
                                # same enforcement applied at server start. The
                                # raised error is surfaced to the client as an
                                # error ack by the enclosing ``except`` below.
                                try:
                                    from scripts.serve_policy import _enforce_runtime_write_policy

                                    cache_config = _enforce_runtime_write_policy(cache_config)
                                except ImportError:
                                    # Fallback: inline enforcement when scripts/ isn't on the
                                    # import path (e.g. embedded test harness). Keeps the
                                    # contract enforced regardless of entry point.
                                    from openpi.cache.config import ConfigValidationError
                                    if cache_config.write_policy.type != "never":
                                        raise ConfigValidationError(
                                            "load_cache_config: write_policy "
                                            f"{cache_config.write_policy.type!r} is not allowed at "
                                            "server runtime (C2 write-frozen). Set write_policy: "
                                            "{type: never} and build cache artifacts offline."
                                        ) from None
                                # Fill deferred dump.path slots from the
                                # configured warmup dump root. The validator
                                # already passed (deferred=True bypasses path
                                # checks); now make the path concrete so the
                                # DumpingJudge built later sees a real file
                                # location.
                                _fill_deferred_dump_paths(cache_config, msg_yaml_id)
                                # build_shared_storage performs the artifact
                                # pickle.load (tens of MB); run it off the event
                                # loop so a large/slow load does not block every
                                # other connection (and /healthz) meanwhile.
                                shared_storage = await asyncio.to_thread(
                                    build_shared_storage, cache_config
                                )
                                global _current_bundle, _bundle_version
                                with _bundle_lock:
                                    _bundle_version += 1
                                    new_bundle = CurrentCacheBundle(
                                        config_path=yaml_path,
                                        cache_config=cache_config,
                                        shared_storage=shared_storage,
                                        version=_bundle_version,
                                        yaml_id=msg_yaml_id,
                                    )
                                    # M2 (Phase 3): index by bundle_id AND keep
                                    # the legacy single-latest pointer so old
                                    # consumers (``_wrap_policy`` via
                                    # ``get_current_cache_bundle()`` without args)
                                    # still see the most-recent bundle.
                                    _bundles[msg_bundle_id] = new_bundle
                                    _current_bundle = new_bundle
                                version = _bundle_version
                                logger.info(
                                    "Cache bundle updated to v%d: %s (yaml_id=%s, bundle_id=%s)",
                                    version, yaml_path, msg_yaml_id, msg_bundle_id,
                                )
                                await websocket.send(packer.pack({
                                    "__ack__": "load_cache_config",
                                    "yaml_path": yaml_path,
                                    "version": version,
                                    "bundle_id": msg_bundle_id,
                                }))
                            except Exception as e:
                                logger.error("Failed to load cache config %s: %s", yaml_path, e)
                                await websocket.send(packer.pack({"__ack__": "error", "msg": str(e)}))
                    elif ctrl == "select_bundle":
                        # M2 (Phase 3, G2 R1 Item 2): client declares which
                        # bundle this connection binds to. Lazy lifecycle:
                        # the per-connection wrapper stack is created here
                        # (or replaced if the connection re-selects later).
                        sb_bundle_id = obs.get("bundle_id", _DEFAULT_BUNDLE_ID)
                        with _bundle_lock:
                            known = sb_bundle_id in _bundles
                        if not known and sb_bundle_id != _DEFAULT_BUNDLE_ID:
                            await websocket.send(packer.pack({
                                "__ack__": "error",
                                "msg": (
                                    f"select_bundle: unknown bundle_id {sb_bundle_id!r}. "
                                    f"Load it with __ctrl__=load_cache_config first."
                                ),
                            }))
                            continue
                        if self._concurrent:
                            try:
                                _bind_bundle(sb_bundle_id)
                            except Exception as exc:
                                logger.exception("select_bundle bind failed")
                                await websocket.send(packer.pack({
                                    "__ack__": "error",
                                    "msg": f"select_bundle bind failed: {exc}",
                                }))
                                continue
                        await websocket.send(packer.pack({
                            "__ack__": "select_bundle", "bundle_id": sb_bundle_id,
                        }))
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
                    elif ctrl == "dump_metrics":
                        # Phase 7 mem-attribution: returns the in-memory
                        # ring buffers (util / batch / timing / mem
                        # snapshots) so the operator can pull the whole
                        # timeline back without grepping log files.
                        await websocket.send(packer.pack(
                            _handle_dump_metrics(obs)
                        ))
                    elif ctrl == "snapshot_mem":
                        # On-demand torch.cuda.memory_snapshot() capture.
                        # Requires OPENPI_MONITOR_LEVEL >= SNAPSHOT.
                        await websocket.send(packer.pack(
                            _handle_snapshot_mem(obs)
                        ))
                    elif ctrl == "mem_history_start":
                        # Enable torch.cuda.memory._record_memory_history;
                        # requires OPENPI_MONITOR_LEVEL = HISTORY.
                        await websocket.send(packer.pack(
                            _handle_mem_history_start(obs)
                        ))
                    elif ctrl == "mem_history_stop":
                        await websocket.send(packer.pack(
                            _handle_mem_history_stop(obs)
                        ))
                    elif ctrl == "set_batch_params":
                        # Hot-mutate coordinator batch params. Payload:
                        # {"max_batch_size": int, "max_wait_ms": float}
                        await websocket.send(packer.pack(
                            _handle_set_batch_params(obs)
                        ))
                    elif ctrl == "throughput_summary":
                        # Compute + return throughput / batching / GPU util
                        # rollup from in-memory buffers. Optional clear=True
                        # to wipe util/batch buffers for next-ramp baseline.
                        await websocket.send(packer.pack(
                            _handle_throughput_summary(obs)
                        ))
                    else:
                        await websocket.send(packer.pack({"__ack__": "ignored"}))
                    continue

                # M2 (lazy lifecycle, G2 R1 Item 2): concurrent connections
                # must declare their bundle (via ``select_bundle`` or the
                # ``bundle_id`` field on the first ``episode_start``) before
                # any inference. If the wrapper stack is not yet bound, fall
                # back to the ``"default"`` slot for backward-compat with
                # clients that never send either ctrl.
                if self._concurrent and conn_policy is None:
                    try:
                        _bind_bundle(_DEFAULT_BUNDLE_ID)
                    except Exception as exc:
                        logger.exception("default bundle bind failed")
                        await websocket.send(packer.pack({
                            "__ack__": "error",
                            "msg": (
                                f"select_bundle or episode_start{{bundle_id}} required "
                                f"before infer; default-bind failed: {exc}"
                            ),
                        }))
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
                # Drop per-connection wrapper state. In the serving entrypoint
                # torch.cuda.empty_cache() is normally patched to a no-op so
                # freed CUDA blocks stay reserved by this process from the
                # system's perspective while remaining reusable internally.
                conn_policy = None
                try:
                    import gc as _gc
                    _gc.collect()
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
                # Event-driven monitor: decrement + snapshot AFTER cleanup
                # finishes. The snapshot then captures the genuine "K-1
                # connections retained" state, including what cleanup did
                # not release. This is the per-conn-leak detector.
                from openpi.serving import monitor as _server_monitor
                _n_after_close = _server_monitor.decrement_connection()
                _server_monitor.fire_event(
                    "conn_close", payload={"conn_count": _n_after_close},
                )
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
                # Mirror the ConnectionClosed cleanup so leaked wrapper state
                # / KV cache from an abnormal exit (e.g. worker SIGKILL) is
                # released internally. The serving entrypoint normally keeps
                # CUDA cached blocks reserved by suppressing empty_cache().
                conn_policy = None
                try:
                    import gc as _gc
                    _gc.collect()
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
                # Same monitor hook as the normal close path so abnormal
                # exits also produce a labelled snapshot for diagnostics.
                from openpi.serving import monitor as _server_monitor
                _n_after_err = _server_monitor.decrement_connection()
                _server_monitor.fire_event(
                    "conn_close",
                    payload={"conn_count": _n_after_err, "abnormal": True},
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    """Respond to ``GET /healthz`` with 200 OK; pass other requests through."""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal WebSocket handshake for all other paths.
    return None
