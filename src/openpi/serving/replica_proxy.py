"""In-process WebSocket router fronting N serving replicas behind one port.

``serve_policy.py --replicas R`` spawns R child serving processes (each the
ordinary concurrent server, on an internal loopback port) and runs this router
in the supervisor process on the single public port. Multiple processes are
required because the single-process serving ceiling is the GIL-serialized
Python/CUDA kernel-launch path; each child has its own interpreter, so launch
parallelism scales with R. The router just multiplexes one public port onto the
children.

Routing is **per connection, never per request** — this is a correctness
contract, not an optimization. Each backend keeps per-connection state
(``CacheOrchestrator._state_history`` / ``_step_counter``, search sessions, and
the stage2 -> stage3 ``DynamicCache`` KV lifetime). A per-request balancer would
send a later request in the same episode to a backend that has no history of it
and no KV cache, corrupting WARM_START / FULL_HIT decisions. So an ``infer``
connection is pinned to one backend for its whole life.

Three frame classes (decided from the connection's first client frame):

* **sticky** — ``infer`` and per-connection ctrls (``episode_start`` /
  ``episode_end`` / ``select_bundle`` / ``prefill_trajectory`` / ``reset``).
  Pinned to one least-connections backend; frames are then piped raw.
* **broadcast** — ctrls that load shared bundle/runtime state on *every* replica
  (``load_cache_config`` / ``preload_normalizer_buffer`` /
  ``unload_warmup_buffer`` / ``set_batch_params`` / ``mem_history_start`` /
  ``mem_history_stop``). Fanned out to all backends; the first ack is returned.
  This is why ``preload_phase5.py`` works unchanged through the public port: the
  router replicates the bundle to all backends automatically.
* **aggregate** — metrics queries (``dump_metrics`` / ``throughput_summary`` /
  ``snapshot_mem``) **and** ``fetch_dump``. Sent to all backends and merged
  (metrics via :func:`merge_metrics`; the warmup dump via
  :func:`merge_dump_replies`, which concatenates each replica's slice) so a
  single query returns whole-host data.

The server sends its metadata frame first and the client receives it before
sending anything (``WebsocketClientPolicy._wait_for_server``). The router
therefore greets each client with metadata cached from a backend, *then* reads
the client's first frame to classify it.
"""

from __future__ import annotations

import asyncio
import logging

from openpi_client import msgpack_numpy
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed
from websockets.exceptions import WebSocketException

# Per-replica fan-out (broadcast/aggregate) recv timeout: a wedged backend that
# accepts the connection but never replies would otherwise stall the whole ctrl
# (and the client) forever, since asyncio.gather waits on every recv.
_FANOUT_TIMEOUT_S = 30.0

# Match WebsocketPolicyServer's public ingress cap: large enough for legitimate
# obs / control payloads, finite so the multi-replica router cannot be OOM'd by
# an unauthenticated multi-GB frame before forwarding to a capped backend.
_MAX_WS_FRAME_BYTES = 256 * 1024 * 1024

# Ctrls that mutate shared per-replica state -> must reach every backend.
CTRL_BROADCAST = frozenset({
    "load_cache_config",
    "preload_normalizer_buffer",
    "unload_warmup_buffer",
    "set_batch_params",
    "mem_history_start",
    "mem_history_stop",
})
# Metrics queries -> fan out and merge into whole-host numbers.
CTRL_AGGREGATE = frozenset({
    "dump_metrics",
    "throughput_summary",
    "snapshot_mem",
    # warmup dump is split across replicas (each child's DumpingJudge writes only
    # the episodes routed to it), so fetch_dump must fan out + concatenate — a
    # sticky single-replica fetch returns a partial buffer.
    "fetch_dump",
})
# Everything else (raw infer, episode_start/end, select_bundle, prefill,
# reset) is per-connection and routed sticky.


def classify(frame: bytes | str) -> tuple[str, str | None]:
    """Return ``(kind, ctrl_name)`` for a client frame.

    ``kind`` is one of ``"sticky"`` / ``"broadcast"`` / ``"aggregate"``. A frame
    that does not unpack to a dict (e.g. a raw infer obs) is ``"sticky"``.
    """
    try:
        obj = msgpack_numpy.unpackb(frame)
    except Exception:
        return ("sticky", None)
    if not isinstance(obj, dict):
        return ("sticky", None)
    ctrl = obj.get("__ctrl__")
    if ctrl in CTRL_BROADCAST:
        return ("broadcast", ctrl)
    if ctrl in CTRL_AGGREGATE:
        return ("aggregate", ctrl)
    return ("sticky", ctrl)


# The server's `_handle_fetch_dump` (websocket_policy_server.py) returns an error
# with this EXACT msg only for the benign case: a replica has no dump file for the
# requested warmup id because no warmup episode was connection-routed to that child.
# Every other error (root not configured / invalid id / bad request) is a real
# failure. This is the server<->router contract; a test asserts the two stay in sync.
FETCH_DUMP_NOT_FOUND_MSG = "dump not found"


def merge_dump_replies(replies: list[dict], *, n_expected: int) -> dict:
    """Concatenate the warmup dump bytes fetched from every replica.

    Each child server's ``DumpingJudge`` writes only the warmup episodes that were
    routed (connection-sticky) to it, so the whole-host dump is the byte
    concatenation of all replicas' slices. Two reply shapes are safe to fold into
    that concatenation: a successful slice, and the *benign* "no warmup episode was
    routed here" case (server error whose msg is :data:`FETCH_DUMP_NOT_FOUND_MSG`).

    Anything else makes a success reply UNSAFE, because the client cannot tell a
    whole-host dump from one missing a slice — returning survivors only would be a
    silent partial calibration buffer (the exact bug the owner override targets).
    The merge therefore surfaces an error when it sees a fatal child error (root not
    configured / invalid id / bad request), a malformed reply, or a *dropped*
    backend (an exception/disconnect during fan-out yields fewer decoded replies
    than ``n_expected``, so that replica's slice is lost).
    """
    chunks: list[bytes] = []
    warmup_yaml_id = None
    fatal: list[str] = []
    for r in replies:
        if not isinstance(r, dict):
            fatal.append(f"malformed reply ({type(r).__name__})")
            continue
        if r.get("__ack__") == "error":
            if r.get("msg") == FETCH_DUMP_NOT_FOUND_MSG:
                continue  # benign: this replica had no routed warmup episode
            fatal.append(str(r.get("msg", "unknown error")))
            continue
        content = r.get("content")
        if not isinstance(content, (bytes, bytearray)):
            fatal.append("success reply without bytes content")
            continue
        if r.get("warmup_yaml_id") is not None:
            warmup_yaml_id = r["warmup_yaml_id"]
        chunks.append(bytes(content))
    n_dropped = n_expected - len(replies)
    if n_dropped > 0:
        fatal.append(f"{n_dropped} replica(s) unreachable/dropped during fan-out")
    if fatal:
        return {"__ack__": "error", "msg": "fetch_dump would be a partial dump: " + "; ".join(fatal)}
    if not chunks:
        # every replica was benign-missing -> the warmup genuinely produced no dump
        return {"__ack__": "error", "msg": "fetch_dump: no dump on any replica"}
    return {"__ack__": "fetch_dump", "warmup_yaml_id": warmup_yaml_id, "content": b"".join(chunks)}


def merge_metrics(replies: list[dict]) -> dict:
    """Merge per-replica metrics dicts into one whole-host dict.

    List-valued keys (``batch`` / ``util`` / ``mem`` / ``timing``) are
    concatenated, each record tagged with its source ``replica`` index so
    downstream analysis can split if needed. Concatenation is the right
    aggregation: throughput is total inferences over the shared wall-clock
    window (sum of per-replica counts), and GPU-util samples all read the same
    physical device, so a mean over the concatenated samples is the host util.
    Scalar / non-list keys keep the first replica's value.
    """
    if not replies:
        return {}
    merged: dict = {}
    all_keys: set[str] = set()
    for d in replies:
        all_keys.update(d.keys())
    for key in all_keys:
        present = [(i, d[key]) for i, d in enumerate(replies) if key in d]
        if present and all(isinstance(v, list) for _, v in present):
            out: list = []
            for idx, lst in present:
                for rec in lst:
                    if isinstance(rec, dict):
                        rec = {**rec, "replica": idx}
                    out.append(rec)
            merged[key] = out
        else:
            merged[key] = present[0][1]
    return merged


def merge_throughput_summary(summaries: list[dict], n_expected: int | None = None) -> dict:
    """Aggregate per-replica ``throughput_summary`` dicts, PRESERVING the schema.

    The single-server ``MetricsRecorder.throughput_summary`` response has a fixed
    shape that operator tools (``dump_mem.py summary``) index directly:
    ``elapsed_s``, ``stage1_throughput_inf_per_s``, ``stages`` (per-stage dicts),
    ``averages``, ``peaks``, ``n_util_samples``, ``n_batch_events``. This merge
    returns the SAME keys with whole-host values rather than a new schema:

    * throughput / inferences / counts (additive across replicas)  -> SUM
    * per-stage ``avg_*``                                          -> count-weighted mean
    * ``max_size`` / ``worst_wait_ms`` / all ``peaks``             -> MAX
    * per-process ``averages`` (torch_alloc_mb / queue_depth / cpu_proc_pct) -> SUM
    * whole-host ``averages`` (gpu_util_pct / sys_cpu_pct / sys_ram_used_mb /
      cgroup_ram_used_mb) -> MEAN (every replica reports the same shared value)
    * ``elapsed_s`` (same window) -> MAX

    Partial state is surfaced explicitly (not silently hidden): the result
    carries ``n_replicas`` (expected), ``n_contributing``, and ``partial=True``
    when fewer replicas contributed than expected. When NONE can contribute the
    result is a structured ``insufficient_window`` marker, not a fake aggregate.
    """
    if n_expected is None:
        n_expected = len(summaries)
    contributing = [s for s in summaries if isinstance(s, dict) and "stages" in s]
    n_contrib = len(contributing)
    if n_contrib == 0:
        _d = [s for s in summaries if isinstance(s, dict)]
        # Distinguish "no replica COULD contribute" (backend error / dropped
        # response, e.g. all down or monitor OFF) from "window just too short".
        # Conflating them as insufficient_window is an unsafe control-plane
        # response. `summaries` only holds decoded replies, so a count short of
        # n_expected means some backend reply was an exception/non-decodable.
        errored = [s for s in _d if s.get("__ack__") == "error"]
        dropped = max(0, n_expected - len(summaries))
        if errored or dropped:
            first = next((str(s.get("msg", "error")) for s in errored), "")
            return {"__ack__": "error", "n_replicas": n_expected,
                    "n_contributing": 0, "n_responses": len(summaries),
                    "n_errored": len(errored), "n_unreachable": dropped,
                    "msg": f"throughput_summary unavailable on all {n_expected} replicas "
                           f"({len(errored)} error, {dropped} unreachable)"
                           + (f": {first}" if first else "")}
        # All replies were genuine short-window / empty markers.
        return {"insufficient_window": True, "n_replicas": n_expected,
                "n_contributing": 0, "n_responses": len(summaries),
                "n_util_events": sum(int(s.get("n_util_events", 0)) for s in _d),
                "n_batch_events": sum(int(s.get("n_batch_events", 0)) for s in _d)}

    _PER_PROCESS = {"torch_alloc_mb", "queue_depth", "cpu_proc_pct"}  # sum
    _ADDITIVE_STAGE = {"count", "inferences", "throughput_inf_per_s"}  # sum
    _MAX_STAGE = {"max_size", "worst_wait_ms"}  # max

    # --- per-stage merge (sum additive, max maxes, count-weighted-mean avgs) ---
    stages: dict = {}
    weighted: dict = {}  # stage -> avg_key -> sum(count*avg)
    for s in contributing:
        for sk, d in s.get("stages", {}).items():
            if not isinstance(d, dict) or d.get("count", 0) == 0:
                stages.setdefault(sk, {"count": 0})
                continue
            agg = stages.setdefault(sk, {})
            w = weighted.setdefault(sk, {})
            cnt = int(d.get("count", 0))
            for k, v in d.items():
                if k in _ADDITIVE_STAGE:
                    agg[k] = agg.get(k, 0) + v
                elif k in _MAX_STAGE:
                    agg[k] = max(agg.get(k, 0), v)
                elif k == "avg_size":
                    continue  # recomputed below from totals
                else:  # avg_* -> accumulate count-weighted numerator
                    w[k] = w.get(k, 0.0) + cnt * float(v)
    for sk, agg in stages.items():
        tot = agg.get("count", 0)
        if tot:
            agg["avg_size"] = agg.get("inferences", 0) / tot
            for k, num in weighted.get(sk, {}).items():
                agg[k] = num / tot

    # --- averages: per-process summed, whole-host meaned ---
    avg_keys: set[str] = set()
    for s in contributing:
        avg_keys.update(s.get("averages", {}).keys())
    averages: dict = {}
    for k in avg_keys:
        vals = [float(s.get("averages", {}).get(k, 0.0)) for s in contributing]
        averages[k] = sum(vals) if k in _PER_PROCESS else sum(vals) / len(vals)

    # --- peaks: max across replicas ---
    peak_keys: set[str] = set()
    for s in contributing:
        peak_keys.update(s.get("peaks", {}).keys())
    peaks = {k: max(float(s.get("peaks", {}).get(k, 0.0)) for s in contributing)
             for k in peak_keys}

    result = {
        # Preserve the single-server control-response shape so routed mode is a
        # drop-in for callers (dump_mem.py): same __ack__ + same rate keys.
        "__ack__": "throughput_summary",
        "elapsed_s": max(float(s.get("elapsed_s", 0.0)) for s in contributing),
        "stage1_throughput_inf_per_s":
            sum(float(s.get("stage1_throughput_inf_per_s", 0.0)) for s in contributing),
        "stages": stages,
        "averages": averages,
        "peaks": peaks,
        "n_util_samples": sum(int(s.get("n_util_samples", 0)) for s in contributing),
        "n_batch_events": sum(int(s.get("n_batch_events", 0)) for s in contributing),
        "n_replicas": n_expected,
        "n_contributing": n_contrib,
    }
    # If the request carried clear=true each backend returns a per-buffer
    # ``cleared`` count dict; sum them so the operator sees whole-host cleared
    # totals (and knows the per-replica buffers WERE cleared).
    cleared_dicts = [s["cleared"] for s in contributing
                     if isinstance(s.get("cleared"), dict)]
    if cleared_dicts:
        result["cleared"] = {
            k: sum(int(c.get(k, 0)) for c in cleared_dicts)
            for k in {key for c in cleared_dicts for key in c}
        }
    if n_contrib < n_expected:
        result["partial"] = True
    return result


class ReplicaProxy:
    """Connection-sticky, control-broadcasting WebSocket router."""

    def __init__(self, backend_host: str, backend_ports: list[int]) -> None:
        self._backends = [(backend_host, p) for p in backend_ports]
        self._active = dict.fromkeys(self._backends, 0)
        self._metadata: bytes | None = None
        self._packer = msgpack_numpy.Packer()

    def _uri(self, backend: tuple[str, int]) -> str:
        return f"ws://{backend[0]}:{backend[1]}"

    async def _open_backend(self, backend, *, swallow_metadata: bool):
        """Open a backend connection; optionally discard its greeting frame."""
        ws = await ws_connect(self._uri(backend), max_size=None, compression=None)
        if swallow_metadata:
            await ws.recv()  # discard the backend's metadata greeting
        return ws

    async def prime_metadata(self, *, attempts: int = 90, delay: float = 2.0) -> None:
        """Fetch and cache the metadata greeting from the first backend.

        Retries because backends bind their port only after model load; the
        supervisor starts the proxy as soon as the ready events fire, which may
        race the actual bind by a few milliseconds.
        """
        last: Exception | None = None
        for _ in range(attempts):
            ws = None
            try:
                ws = await ws_connect(self._uri(self._backends[0]), max_size=None, compression=None)
                self._metadata = await ws.recv()
                return
            except (OSError, WebSocketException) as exc:
                # OSError = connection refused during the bind race;
                # WebSocketException = backend accepted then half-closed before
                # the greeting. Both are retryable; never leak the socket.
                last = exc
                await asyncio.sleep(delay)
            finally:
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass
        raise RuntimeError(f"could not fetch metadata from backend: {last}")

    # --- per-connection handling -------------------------------------------

    async def handle(self, client_ws) -> None:
        # Greet first: clients recv metadata before sending anything.
        await client_ws.send(self._metadata)
        try:
            first = await client_ws.recv()
        except ConnectionClosed:
            return
        kind, _ = classify(first)
        if kind == "sticky":
            await self._handle_sticky(client_ws, first)
        else:
            await self._handle_control(client_ws, first, kind)

    async def _handle_sticky(self, client_ws, first_frame) -> None:
        backend = min(self._backends, key=lambda b: self._active[b])
        self._active[backend] += 1
        logging.info("conn -> %s sticky (active=%s)", self._uri(backend),
                     {f"{p}": n for (_, p), n in self._active.items()})
        backend_ws = None
        try:
            backend_ws = await self._open_backend(backend, swallow_metadata=True)
            await backend_ws.send(first_frame)
            c2b = asyncio.create_task(_pump(client_ws, backend_ws))
            b2c = asyncio.create_task(_pump(backend_ws, client_ws))
            _done, pending = await asyncio.wait({c2b, b2c}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if pending:
                # Await the cancellation so the cancelled pump fully unwinds
                # before we close backend_ws (avoids "Task ... pending" warnings
                # and a use-after-close race on the socket).
                await asyncio.gather(*pending, return_exceptions=True)
        except (OSError, WebSocketException) as exc:
            # Backend unreachable / died. The client already received the
            # metadata greeting and is blocked in recv(); send a text-frame
            # error so WebsocketClientPolicy.infer() raises RuntimeError (the
            # server's own infer-error convention) instead of hanging.
            logging.warning("backend %s unavailable: %s", self._uri(backend), exc)
            try:
                await client_ws.send(f"replica backend unavailable: {exc}")
            except (ConnectionClosed, WebSocketException):
                pass
        finally:
            if backend_ws is not None:
                await backend_ws.close()
            self._active[backend] -= 1

    async def _handle_control(self, client_ws, first_frame, kind: str) -> None:
        frame = first_frame
        while True:
            if kind == "aggregate":
                reply = await self._aggregate(frame)
            else:
                reply = await self._broadcast(frame)
            try:
                await client_ws.send(reply)
                frame = await client_ws.recv()
            except ConnectionClosed:
                break
            kind, _ = classify(frame)
            if kind == "sticky":
                # Mixed connection: the client sent control frame(s) and then
                # switched to data on the SAME socket (e.g. load_cache_config
                # then select_bundle / infer via one WebsocketClientPolicy).
                # Hand off to sticky routing — pin a backend, forward this
                # buffered frame, and pipe the rest — rather than dropping it.
                await self._handle_sticky(client_ws, frame)
                return

    async def _broadcast(self, frame) -> bytes:
        async def one(backend):
            ws = await self._open_backend(backend, swallow_metadata=True)
            try:
                await ws.send(frame)
                return await asyncio.wait_for(ws.recv(), timeout=_FANOUT_TIMEOUT_S)
            finally:
                await ws.close()

        acks = await asyncio.gather(*[one(b) for b in self._backends], return_exceptions=True)

        # A broadcast ctrl (load_cache_config / preload / unload / set_batch_params)
        # MUST take effect on EVERY replica. If any backend failed — either raised
        # (unreachable/crashed) or returned an ``{"__ack__":"error"}`` payload —
        # surface an error rather than the first success ack. Otherwise a later
        # sticky infer pinned to the failed replica would run against unloaded
        # bundle state (the cache corruption this router exists to prevent).
        def _failure(a) -> str | None:
            if isinstance(a, BaseException):
                return str(a)
            if isinstance(a, (bytes, bytearray)):
                try:
                    d = msgpack_numpy.unpackb(a)
                except Exception:
                    return None
                if isinstance(d, dict) and d.get("__ack__") == "error":
                    return str(d.get("msg", "error"))
                return None
            return f"unexpected ack type {type(a).__name__}"

        failures = [f for f in (_failure(a) for a in acks) if f is not None]
        if failures:
            logging.error("broadcast ctrl failed on %d/%d replicas: %s",
                          len(failures), len(self._backends), failures[0])
            return self._packer.pack({
                "__ack__": "error",
                "msg": f"broadcast failed on {len(failures)}/{len(self._backends)} "
                       f"replicas: {failures[0]}",
            })
        return acks[0]  # all replicas succeeded; acks are equivalent

    async def _aggregate(self, frame) -> bytes:
        async def one(backend):
            ws = await self._open_backend(backend, swallow_metadata=True)
            try:
                await ws.send(frame)
                return await asyncio.wait_for(ws.recv(), timeout=_FANOUT_TIMEOUT_S)
            finally:
                await ws.close()

        raw = await asyncio.gather(*[one(b) for b in self._backends], return_exceptions=True)
        dicts = []
        for r in raw:
            if isinstance(r, (bytes, bytearray)):
                try:
                    dicts.append(msgpack_numpy.unpackb(r))
                except Exception:
                    pass
        # throughput_summary is mostly scalars/nested dicts, which the generic
        # list-concatenating merge would collapse to replica 0's values only.
        # Use a dedicated summer so the public response reports whole-host
        # throughput rather than one replica's slice.
        _, ctrl = classify(frame)
        if ctrl == "fetch_dump":
            # Warmup dump is split across replicas — concatenate each replica's
            # slice into the whole-host dump. Pass the expected replica count so a
            # dropped/errored backend surfaces as an error instead of a silently
            # partial dump (G2R6).
            merged = merge_dump_replies(dicts, n_expected=len(self._backends))
        elif ctrl == "throughput_summary":
            # Pass the expected replica count so missing/errored replicas (whose
            # responses never reached `dicts`) are surfaced as partial, not
            # silently aggregated over the survivors only.
            merged = merge_throughput_summary(dicts, n_expected=len(self._backends))
        else:
            merged = merge_metrics(dicts)
        return self._packer.pack(merged)

    async def serve(self, public_port: int) -> None:
        await self.prime_metadata()
        async with ws_serve(self.handle, "0.0.0.0", public_port,
                            max_size=_MAX_WS_FRAME_BYTES, compression=None) as server:
            logging.info("replica_proxy listening on 0.0.0.0:%d -> %s",
                         public_port, [p for _, p in self._backends])
            await server.serve_forever()


async def _pump(src, dst) -> None:
    try:
        async for message in src:
            await dst.send(message)
    except ConnectionClosed:
        pass


def run_proxy(public_port: int, backend_ports: list[int],
              backend_host: str = "127.0.0.1") -> None:
    """Blocking entry point used by the serve_policy supervisor."""
    proxy = ReplicaProxy(backend_host, backend_ports)
    asyncio.run(proxy.serve(public_port))
