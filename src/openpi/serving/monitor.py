"""Server-wide monitoring configuration and in-memory event recorder.

This module owns the **single master switch** for all serving-side monitoring
(``OPENPI_MONITOR_LEVEL`` env var) and the **in-memory ring buffers** that
hold structured metric events for later bulk retrieval. Disk writes are
opt-in: the recorder never spills to disk by itself. A WebSocket ``__ctrl__``
endpoint or an explicit ``flush_to_disk()`` call is required to persist.

Monitor levels
--------------
``OFF`` *(default)*
    All monitoring is off. ``SystemTimer.enabled=False`` (zero-overhead
    no-op), ``BatchingCoordinator`` does not spawn the util sampler thread,
    ``[batch_done]`` logs at ``DEBUG`` (default not visible), no timing / CPU /
    GPU / throughput / concurrency data is recorded. Memory snapshot / history
    ctrl endpoints are rejected. This is the default: set
    ``OPENPI_MONITOR_LEVEL`` only when you want to measure.

``BASIC``
    SystemTimer enabled, util thread running, ``[batch_done]`` at ``INFO``,
    events pushed into in-memory ring buffers. Memory snapshot/history
    ctrl endpoints are rejected.

``SNAPSHOT``
    ``BASIC`` + accept ``__ctrl__: snapshot_mem`` (calls
    ``torch.cuda.memory_snapshot()`` + ``memory_summary()`` on demand).
    Cheap (~100 ms / call) — fine to use during sweeps to capture state at
    specific concurrency points.

``HISTORY``
    ``SNAPSHOT`` + ``torch.cuda.memory._record_memory_history(enabled='all',
    context='python', stacks='python')`` is enabled at server start so every
    alloc/free is logged with a Python stack trace. **5–20 % inference
    latency overhead** — use only during dedicated diagnostic runs.

Ring buffer semantics
---------------------
* All events live in-memory in ``collections.deque`` ring buffers.
* Concurrent writes are serialised by ``self._lock`` (uncontested ~1 us).
* Buffer sizes are deliberately small (util / batch O(10k); snapshots O(64))
  so the server cannot leak unbounded memory under sustained load.
* ``dump_all()`` returns a deep-copied snapshot suitable for msgpack /
  pickle serialisation. The original buffers are unaffected.
* ``clear(...)`` is the only way to drop events; never auto-cleared.

Usage
-----
::

    from openpi.serving import monitor

    level = monitor.get_monitor_level()
    if level >= monitor.MonitorLevel.BASIC:
        monitor.get_recorder().record_util(...)

    # On-demand memory snapshot:
    if level >= monitor.MonitorLevel.SNAPSHOT:
        monitor.take_memory_snapshot("after-N4-ramp")

Thread safety
-------------
``MetricsRecorder`` is a process-wide singleton; access via ``get_recorder``.
All record methods take ``self._lock``. ``dump_all`` snapshots under the
lock and then releases before serialising, so a long readout does not
block ongoing writes.
"""

from __future__ import annotations

from collections import deque
import enum
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Level enum
# ---------------------------------------------------------------------------


class MonitorLevel(enum.IntEnum):
    """Server-wide monitoring level.

    ``IntEnum`` so callers can write ``level >= MonitorLevel.SNAPSHOT``
    without unwrapping. Higher numbers = more instrumentation.
    """

    OFF = 0
    BASIC = 1
    SNAPSHOT = 2
    HISTORY = 3

    @classmethod
    def from_str(cls, s: str | None) -> MonitorLevel:
        """Parse a level name (case-insensitive). Unknown / empty → OFF.

        Default is OFF so that the server records no timing / CPU / GPU /
        throughput / concurrency data unless monitoring is explicitly requested
        via ``OPENPI_MONITOR_LEVEL`` — keeping the production serving path at
        zero instrumentation overhead.
        """
        if not s:
            return cls.OFF
        s = s.strip().lower()
        for member in cls:
            if member.name.lower() == s:
                return member
        # Numeric form ("0"/"1"/...).
        try:
            return cls(int(s))
        except (ValueError, KeyError):
            logger.warning(
                "OPENPI_MONITOR_LEVEL=%r unrecognised; falling back to OFF", s
            )
            return cls.OFF


_LEVEL: MonitorLevel | None = None


def get_monitor_level() -> MonitorLevel:
    """Resolve the effective monitor level (cached per process).

    Reads ``OPENPI_MONITOR_LEVEL`` once on first call; subsequent calls
    return the cached value. To change at runtime use ``set_monitor_level``
    (intended for tests only).
    """
    global _LEVEL
    if _LEVEL is None:
        _LEVEL = MonitorLevel.from_str(os.environ.get("OPENPI_MONITOR_LEVEL"))
    return _LEVEL


def set_monitor_level(level: MonitorLevel) -> None:
    """Force the monitor level (tests / runtime override)."""
    global _LEVEL
    _LEVEL = level


# ---------------------------------------------------------------------------
# In-memory event recorder
# ---------------------------------------------------------------------------


class MetricsRecorder:
    """Process-wide in-memory store for structured metric events.

    Holds three independent ring buffers:

    * ``util`` — periodic ``[metrics.util]`` snapshots from the
      ``BatchingCoordinator`` util thread (or any other periodic sampler).
    * ``batch`` — per-batch dispatch events from
      ``BatchingCoordinator._stage_loop``.
    * ``mem_snapshots`` — on-demand ``torch.cuda.memory_snapshot()`` +
      ``memory_summary()`` captures triggered by the snapshot ctrl handler.

    Also stores ``timing`` records pushed in bulk from ``SystemTimer``
    when a client connection closes — avoids per-record disk IO.

    All methods are thread-safe.
    """

    def __init__(
        self,
        *,
        util_capacity: int = 4096,
        batch_capacity: int = 16384,
        snapshot_capacity: int = 64,
        timing_capacity: int = 200_000,
    ) -> None:
        self._lock = threading.Lock()
        self._util: deque[dict] = deque(maxlen=util_capacity)
        self._batch: deque[dict] = deque(maxlen=batch_capacity)
        self._mem_snapshots: deque[dict] = deque(maxlen=snapshot_capacity)
        self._timing: deque[dict] = deque(maxlen=timing_capacity)
        # Dropped counters track overflow so callers can see when their
        # buffer sizing is too tight (the deque silently discards otherwise).
        self._dropped: dict[str, int] = {"util": 0, "batch": 0, "mem": 0, "timing": 0}
        # ``_appended`` is the monotonic total ever pushed, useful for
        # detecting whether new events arrived between two dump_all calls
        # without diffing payloads.
        self._appended: dict[str, int] = {"util": 0, "batch": 0, "mem": 0, "timing": 0}

    # ------------------------------------------------------------------
    # Record APIs (called from instrumentation sites)
    # ------------------------------------------------------------------

    def record_util(self, event: dict) -> None:
        """Push a util-sampler event. ``event`` is the kwargs dict that
        would otherwise have gone into the ``[metrics.util]`` log line.
        ``ts_epoch`` is auto-stamped from ``time.time()``."""
        event = dict(event)
        event.setdefault("ts_epoch", time.time())
        with self._lock:
            if len(self._util) == self._util.maxlen:
                self._dropped["util"] += 1
            self._util.append(event)
            self._appended["util"] += 1

    def record_batch(self, event: dict) -> None:
        """Push a per-batch event from ``BatchingCoordinator._stage_loop``."""
        event = dict(event)
        event.setdefault("ts_epoch", time.time())
        with self._lock:
            if len(self._batch) == self._batch.maxlen:
                self._dropped["batch"] += 1
            self._batch.append(event)
            self._appended["batch"] += 1

    def record_mem_snapshot(
        self,
        label: str,
        *,
        summary_text: str,
        snapshot_obj: Any,
        ts_epoch: float | None = None,
    ) -> None:
        """Stash an on-demand CUDA memory snapshot.

        ``snapshot_obj`` is whatever ``torch.cuda.memory_snapshot()``
        returned (a list of segment dicts). It is NOT serialised here —
        the recorder keeps the Python object until a dump request, when it
        is pickled in bulk. This lets the client see snapshots in their
        original ``frames``/``blocks`` structure.
        """
        entry = {
            "label": label,
            "ts_epoch": ts_epoch if ts_epoch is not None else time.time(),
            "summary_text": summary_text,
            "snapshot": snapshot_obj,
        }
        with self._lock:
            if len(self._mem_snapshots) == self._mem_snapshots.maxlen:
                self._dropped["mem"] += 1
            self._mem_snapshots.append(entry)
            self._appended["mem"] += 1

    def record_timing_bulk(self, rows: list[dict]) -> None:
        """Push a batch of timing records (e.g. from
        ``SystemTimer.on_task_end``) into the timing ring buffer.

        Each row is a flat dict with at least ``timestamp / task_id / name /
        elapsed_ms`` and optionally resource keys (``cpu.proc_pct`` etc).
        """
        with self._lock:
            for row in rows:
                if len(self._timing) == self._timing.maxlen:
                    self._dropped["timing"] += 1
                self._timing.append(row)
                self._appended["timing"] += 1

    # ------------------------------------------------------------------
    # Readout / serialisation
    # ------------------------------------------------------------------

    def dump_all(
        self,
        *,
        include_util: bool = True,
        include_batch: bool = True,
        include_mem: bool = True,
        include_timing: bool = True,
    ) -> dict:
        """Snapshot the requested buffers and return as a plain dict.

        The returned dict is suitable for msgpack/pickle. Memory snapshot
        entries still carry the raw Python ``snapshot`` object — callers
        that need to serialise via msgpack should drop that key or
        pickle-encode it first (see ``websocket_policy_server`` handler).
        """
        with self._lock:
            out: dict[str, Any] = {
                "ts_epoch": time.time(),
                "appended_totals": dict(self._appended),
                "dropped_totals": dict(self._dropped),
            }
            if include_util:
                out["util"] = list(self._util)
            if include_batch:
                out["batch"] = list(self._batch)
            if include_mem:
                out["mem_snapshots"] = list(self._mem_snapshots)
            if include_timing:
                out["timing"] = list(self._timing)
        return out

    def clear(
        self,
        *,
        util: bool = False,
        batch: bool = False,
        mem: bool = False,
        timing: bool = False,
    ) -> dict:
        """Drop the selected buffers; returns counts of cleared entries."""
        cleared: dict[str, int] = {}
        with self._lock:
            if util:
                cleared["util"] = len(self._util)
                self._util.clear()
            if batch:
                cleared["batch"] = len(self._batch)
                self._batch.clear()
            if mem:
                cleared["mem"] = len(self._mem_snapshots)
                self._mem_snapshots.clear()
            if timing:
                cleared["timing"] = len(self._timing)
                self._timing.clear()
        return cleared

    def throughput_summary(self) -> dict:
        """Compute throughput / batching / util stats from the current in-memory
        ring buffers. Cheap: single pass, no payload copy.

        Returns a dict ready for one-shot ctrl response. Time window = first
        util event to last util event (5-sec sampler grid). Inference count =
        sum of stage=1 batch sizes (every ``policy.infer`` request goes through
        stage 1 exactly once). Batch fill ratio = avg(size) / max(size).
        """
        with self._lock:
            batch_events = list(self._batch)
            util_events = list(self._util)
        # Time window: a rate needs >=2 timestamps spanning a real interval.
        # Prefer the 1 Hz util grid; fall back to the batch-event span. A single
        # sample (or events clustered at one instant) cannot yield a meaningful
        # rate — return an insufficient-window marker rather than dividing by a
        # ~0 span clamped to 1e-6, which would report a physically impossible
        # throughput (e.g. 10 inferences / 1e-6 s = 10,000,000 inf/s).
        if len(util_events) >= 2:
            t_start = float(util_events[0].get("ts_epoch", 0.0))
            t_end = float(util_events[-1].get("ts_epoch", 0.0))
        elif len(batch_events) >= 2:
            t_start = float(batch_events[0].get("ts_epoch", 0.0))
            t_end = float(batch_events[-1].get("ts_epoch", 0.0))
        elif util_events or batch_events:
            return {"insufficient_window": True,
                    "n_util_events": len(util_events),
                    "n_batch_events": len(batch_events)}
        else:
            return {"empty": True}
        elapsed_s = t_end - t_start
        if elapsed_s < 1e-3:
            return {"insufficient_window": True, "elapsed_s": elapsed_s,
                    "n_util_events": len(util_events),
                    "n_batch_events": len(batch_events)}

        # Per-stage breakdown
        def _blank():
            return {"count": 0, "inferences": 0, "wait_avg_sum": 0.0,
                    "wait_max_sum": 0.0, "run_sum": 0.0, "spread_sum": 0.0,
                    "assemble_sum": 0.0, "forward_sum": 0.0,
                    "max_size": 0, "max_wait_seen": 0.0}
        per_stage: dict[int, dict[str, float]] = {1: _blank(), 2: _blank(), 3: _blank()}
        for ev in batch_events:
            _s_raw = ev.get("stage", 0)
            if not isinstance(_s_raw, int):
                continue  # e.g. "3bucket" sub-bucket events — analysed separately
            s = _s_raw
            if s not in per_stage:
                continue
            sz = int(ev.get("size", 0))
            d = per_stage[s]
            d["count"] += 1
            d["inferences"] += sz
            d["wait_avg_sum"] += float(ev.get("wait_avg_ms", ev.get("wait_ms", 0.0)))
            d["wait_max_sum"] += float(ev.get("wait_max_ms", ev.get("wait_ms", 0.0)))
            d["run_sum"] += float(ev.get("run_ms", 0.0))
            d["spread_sum"] += float(ev.get("enqueue_spread_ms", 0.0))
            d["assemble_sum"] += float(ev.get("assemble_ms", 0.0))
            d["forward_sum"] += float(ev.get("forward_ms", 0.0))
            d["max_size"] = max(d["max_size"], sz)
            d["max_wait_seen"] = max(d["max_wait_seen"], float(ev.get("wait_max_ms", 0.0)))

        # msgpack-numpy strict_map_key disallows int dict keys; use str.
        stages_report = {}
        for s, d in per_stage.items():
            key = f"stage{s}"
            if d["count"] == 0:
                stages_report[key] = {"count": 0}
                continue
            stages_report[key] = {
                "count": d["count"],
                "inferences": d["inferences"],
                "avg_size": d["inferences"] / d["count"],
                "max_size": d["max_size"],
                "avg_wait_ms": d["wait_avg_sum"] / d["count"],      # avg per-req wait
                "avg_wait_max_ms": d["wait_max_sum"] / d["count"],  # avg head-of-batch wait
                "worst_wait_ms": d["max_wait_seen"],                # single worst wait
                "avg_run_ms": d["run_sum"] / d["count"],
                "avg_assemble_ms": d["assemble_sum"] / d["count"],  # CPU batch stacking
                "avg_forward_ms": d["forward_sum"] / d["count"],    # GPU forward + split
                "avg_spread_ms": d["spread_sum"] / d["count"],
                "throughput_inf_per_s": d["inferences"] / elapsed_s,
            }

        # Peaks AND averages from util events. Avg is the primary signal
        # (sustained utilisation) — peak is single-tick noise.
        peak_gpu = 0
        peak_alloc = 0.0
        peak_q = 0
        peak_cgroup = 0.0
        sum_gpu = sum_alloc = sum_q = sum_cgroup = 0.0
        sum_cpu_proc = sum_cpu_sys = 0.0
        sum_ram_used = 0.0
        for u in util_events:
            g = int(u.get("gpu_util_pct", 0))
            a = float(u.get("torch_alloc_mb", 0.0))
            q = int(u.get("q1", 0)) + int(u.get("q2", 0)) + int(u.get("q3", 0))
            c = float(u.get("cgroup_ram_used_mb", 0.0))
            peak_gpu = max(peak_gpu, g); sum_gpu += g
            peak_alloc = max(peak_alloc, a); sum_alloc += a
            peak_q = max(peak_q, q); sum_q += q
            peak_cgroup = max(peak_cgroup, c); sum_cgroup += c
            sum_cpu_proc += float(u.get("cpu_proc_pct", 0.0))
            sum_cpu_sys += float(u.get("sys_cpu_pct", 0.0))
            sum_ram_used += float(u.get("sys_ram_used_mb", 0.0))
        n_u = max(len(util_events), 1)

        return {
            "elapsed_s": elapsed_s,
            "stage1_throughput_inf_per_s": stages_report.get("stage1", {}).get("throughput_inf_per_s", 0.0),
            "stages": stages_report,
            "averages": {
                "gpu_util_pct": sum_gpu / n_u,
                "torch_alloc_mb": sum_alloc / n_u,
                "queue_depth": sum_q / n_u,
                "cgroup_ram_used_mb": sum_cgroup / n_u,
                "cpu_proc_pct": sum_cpu_proc / n_u,
                "sys_cpu_pct": sum_cpu_sys / n_u,
                "sys_ram_used_mb": sum_ram_used / n_u,
            },
            "peaks": {
                "gpu_util_pct": peak_gpu,
                "torch_alloc_mb": peak_alloc,
                "queue_depth": peak_q,
                "cgroup_ram_used_mb": peak_cgroup,
            },
            "n_util_samples": len(util_events),
            "n_batch_events": len(batch_events),
        }

    def stats(self) -> dict:
        """Cheap status dict (no payload copy)."""
        with self._lock:
            return {
                "lengths": {
                    "util": len(self._util),
                    "batch": len(self._batch),
                    "mem_snapshots": len(self._mem_snapshots),
                    "timing": len(self._timing),
                },
                "appended_totals": dict(self._appended),
                "dropped_totals": dict(self._dropped),
                "capacities": {
                    "util": self._util.maxlen,
                    "batch": self._batch.maxlen,
                    "mem_snapshots": self._mem_snapshots.maxlen,
                    "timing": self._timing.maxlen,
                },
            }


_RECORDER: MetricsRecorder | None = None
_RECORDER_LOCK = threading.Lock()


def get_recorder() -> MetricsRecorder:
    """Process-wide ``MetricsRecorder`` singleton (lazy-initialised)."""
    global _RECORDER
    if _RECORDER is None:
        with _RECORDER_LOCK:
            if _RECORDER is None:
                _RECORDER = MetricsRecorder()
    return _RECORDER


def reset_recorder_for_tests() -> None:
    """Test-only: drop the singleton so each test starts clean."""
    global _RECORDER
    with _RECORDER_LOCK:
        _RECORDER = None


# ---------------------------------------------------------------------------
# Memory history / snapshot helpers (PyTorch native API wrappers)
# ---------------------------------------------------------------------------


_HISTORY_ENABLED = False
_HISTORY_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Event-driven auto-snapshot (server-side)
#
# Server modules (WebsocketPolicyServer, BatchingCoordinator) call into
# ``fire_event`` at key transitions (connection open/close, OOM near-miss,
# explicit milestones from a client ctrl). When the monitor level is
# SNAPSHOT or HISTORY, each event triggers ``take_memory_snapshot`` with
# a label derived from the event payload, so the operator's only job is
# to spawn / kill workers — the monitor decides WHEN to snapshot.
#
# A per-kind cooldown prevents storms (e.g. burst of disconnects).
# ---------------------------------------------------------------------------


_CONN_COUNTER_LOCK = threading.Lock()
_CONN_COUNT = 0

_EVENT_COOLDOWN_S: dict[str, float] = {
    "conn_open": 0.0,    # always snapshot on connect (we want the active peak)
    "conn_close": 0.0,   # and on disconnect (cleanup observed)
    "milestone": 0.0,    # client-driven explicit markers
    "oom_warning": 5.0,  # alloc_retries crossing — debounce
    "episode_end": 15.0, # one snapshot per ~15s of episode wall-time, not every step
}
_LAST_FIRE: dict[str, float] = {}

# Auto-flush configuration: when ``OPENPI_MONITOR_AUTOFLUSH_DIR`` is set on
# server startup, certain events also pickle the recorder contents to that
# directory so the operator never has to issue an explicit ``dump_metrics``
# ctrl. By default we flush on ``conn_close`` (each session boundary) so
# every experiment naturally produces a per-session pkl.
_AUTOFLUSH_DIR = os.environ.get("OPENPI_MONITOR_AUTOFLUSH_DIR", "").strip() or None
_AUTOFLUSH_KINDS = {"conn_close", "episode_end"}  # event kinds that trigger an autoflush
_AUTOFLUSH_LOCK = threading.Lock()
_AUTOFLUSH_COUNTER = 0


def autoflush_dir() -> str | None:
    return _AUTOFLUSH_DIR


def _do_autoflush(label_tag: str) -> dict:
    """Pickle the recorder + the in-flight memory snapshots to disk."""
    if _AUTOFLUSH_DIR is None:
        return {"ok": False, "reason": "OPENPI_MONITOR_AUTOFLUSH_DIR unset"}
    global _AUTOFLUSH_COUNTER
    with _AUTOFLUSH_LOCK:
        _AUTOFLUSH_COUNTER += 1
        seq = _AUTOFLUSH_COUNTER
    try:
        import pickle as _pickle

        os.makedirs(_AUTOFLUSH_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        # Sanitize tag: filename-safe.
        safe_tag = "".join(c if c.isalnum() or c in "_-" else "_" for c in label_tag)
        fname = f"metrics_{seq:04d}_{safe_tag}_{ts}.pkl"
        path = os.path.join(_AUTOFLUSH_DIR, fname)
        data = get_recorder().dump_all()
        # mem_snapshots entries each carry a raw Python ``snapshot`` object
        # (segments + device_traces). Pickle them as-is — receiver side
        # (``memory_diagnose.py``) iterates ``snapshot`` directly.
        with open(path, "wb") as f:
            _pickle.dump(data, f, protocol=4)
        logger.info(
            "[autoflush] -> %s (util=%d batch=%d timing=%d mem_snapshots=%d)",
            path,
            len(data.get("util", [])), len(data.get("batch", [])),
            len(data.get("timing", [])), len(data.get("mem_snapshots", [])),
        )
        return {"ok": True, "path": path, "seq": seq}
    except Exception as exc:
        logger.exception("autoflush failed")
        return {"ok": False, "reason": str(exc)}


def fire_event(kind: str, *, label_hint: str = "", payload: dict | None = None) -> dict:
    """Server-side hook: try to take an auto-snapshot for ``kind``.

    No-ops when monitor level < SNAPSHOT, when cooldown window for this
    kind has not elapsed, or when CUDA is unavailable.

    The snapshot label is ``"<kind>_<count_or_hint>_<seq>"`` so multiple
    fires at the same connection count don't collide.

    Returns the take_memory_snapshot result dict (or ``{"ok": False, ...}``
    on no-op) so callers can log it inline.
    """
    if get_monitor_level() < MonitorLevel.SNAPSHOT:
        return {"ok": False, "reason": "level<SNAPSHOT"}
    now = time.monotonic()
    last = _LAST_FIRE.get(kind, -1e9)
    cd = _EVENT_COOLDOWN_S.get(kind, 0.0)
    if now - last < cd:
        return {"ok": False, "reason": "cooldown", "since_last_s": now - last}
    _LAST_FIRE[kind] = now
    payload = payload or {}
    n = payload.get("conn_count", 0)
    suffix = label_hint or str(n)
    # Append an integer suffix so repeated events at the same conn count
    # don't collide (e.g. open→close→open→close at N=1).
    seq_key = f"{kind}_{suffix}_seq"
    seq = payload.get("seq", _LAST_FIRE.get(seq_key, 0) + 1)
    _LAST_FIRE[seq_key] = seq
    label = f"{kind}_n{n}_#{seq}"
    snap_result = take_memory_snapshot(label)
    # Event-driven autoflush: snapshot is already attached to the recorder;
    # if this event kind triggers autoflush AND a flush dir is configured,
    # also pickle the recorder to disk so the operator never needs an
    # explicit dump_metrics ctrl. Errors are non-fatal — fire_event must
    # not crash the calling _handler.
    if kind in _AUTOFLUSH_KINDS and _AUTOFLUSH_DIR is not None:
        flush_result = _do_autoflush(label)
        snap_result["autoflush"] = flush_result
    return snap_result


def increment_connection() -> int:
    """Atomic ++ of the server-wide active-connection counter.

    Called by WebsocketPolicyServer._handler when a new WS connection
    completes the handshake. Returns the new count.
    """
    global _CONN_COUNT
    with _CONN_COUNTER_LOCK:
        _CONN_COUNT += 1
        return _CONN_COUNT


def decrement_connection() -> int:
    """Atomic -- of the server-wide active-connection counter.

    Returns the new count (>=0). Called on every disconnect path.
    """
    global _CONN_COUNT
    with _CONN_COUNTER_LOCK:
        _CONN_COUNT = max(0, _CONN_COUNT - 1)
        return _CONN_COUNT


def current_connection_count() -> int:
    with _CONN_COUNTER_LOCK:
        return _CONN_COUNT


def reset_connection_count_for_tests() -> None:
    global _CONN_COUNT
    with _CONN_COUNTER_LOCK:
        _CONN_COUNT = 0
    _LAST_FIRE.clear()


def memory_history_enabled() -> bool:
    return _HISTORY_ENABLED


def enable_memory_history(
    *,
    max_entries: int = 100_000,
    context: str = "all",
    stacks: str = "python",
) -> dict:
    """Turn on ``torch.cuda.memory._record_memory_history``.

    Refuses to run when the monitor level is below HISTORY so the
    expensive Python-stack capture cannot be turned on by accident.

    Returns a dict describing what was enabled (or why it was not).
    """
    if get_monitor_level() < MonitorLevel.HISTORY:
        return {
            "ok": False,
            "reason": "OPENPI_MONITOR_LEVEL < HISTORY",
            "level": get_monitor_level().name,
        }
    global _HISTORY_ENABLED
    with _HISTORY_LOCK:
        if _HISTORY_ENABLED:
            return {"ok": True, "already_enabled": True}
        try:
            import torch

            if not torch.cuda.is_available():
                return {"ok": False, "reason": "cuda_unavailable"}
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context=context,
                stacks=stacks,
                max_entries=max_entries,
            )
            _HISTORY_ENABLED = True
            logger.info(
                "memory history recording ENABLED (max_entries=%d context=%s stacks=%s)",
                max_entries, context, stacks,
            )
            return {
                "ok": True,
                "max_entries": max_entries,
                "context": context,
                "stacks": stacks,
            }
        except Exception as exc:
            logger.exception("enable_memory_history failed")
            return {"ok": False, "reason": f"exception: {exc}"}


def disable_memory_history() -> dict:
    """Turn off ``_record_memory_history``. Safe to call when already off."""
    global _HISTORY_ENABLED
    with _HISTORY_LOCK:
        if not _HISTORY_ENABLED:
            return {"ok": True, "already_disabled": True}
        try:
            import torch

            torch.cuda.memory._record_memory_history(enabled=None)
            _HISTORY_ENABLED = False
            logger.info("memory history recording DISABLED")
            return {"ok": True}
        except Exception as exc:
            logger.exception("disable_memory_history failed")
            return {"ok": False, "reason": f"exception: {exc}"}


def take_memory_snapshot(label: str) -> dict:
    """Capture a full PyTorch CUDA memory snapshot under ``label``.

    Internally calls ``torch.cuda.memory._snapshot()`` (private API but
    stable since torch 2.0) which returns ``{"segments": [...],
    "device_traces": [...]}``:

    * ``segments`` — every cached CUDA segment + the blocks it contains.
      Each block has ``size / requested_size / state / address / frames``.
      ``frames`` is populated when ``_record_memory_history`` is enabled.
    * ``device_traces`` — alloc / free / segment-alloc / segment-free
      events with ``time_us``; lets the offline analyzer compute per-block
      age. Empty when history is disabled.

    Also captures ``memory_summary(abbreviated=False)`` (size-bucket
    histogram) for human-readable text. No-ops when the monitor level
    is below SNAPSHOT or CUDA is unavailable.
    """
    if get_monitor_level() < MonitorLevel.SNAPSHOT:
        return {
            "ok": False,
            "reason": "OPENPI_MONITOR_LEVEL < SNAPSHOT",
            "level": get_monitor_level().name,
        }
    try:
        import torch

        if not torch.cuda.is_available():
            return {"ok": False, "reason": "cuda_unavailable"}
        # ``memory_summary`` raises ``KeyError`` when the calling thread's
        # CUDA context has not yet allocated anything (e.g. asyncio main
        # thread when actual inference runs in ``asyncio.to_thread``
        # worker threads). The snapshot itself is what we need; the
        # summary text is best-effort.
        try:
            summary = torch.cuda.memory_summary(device=0, abbreviated=False)
        except (KeyError, RuntimeError) as exc:
            summary = f"<memory_summary unavailable: {exc}>"
        # Prefer the dict-form _snapshot() (segments + device_traces);
        # fall back to the public memory_snapshot() if the private API
        # is missing on this build.
        snap_obj: Any
        try:
            snap_obj = torch.cuda.memory._snapshot()
        except Exception:
            snap_obj = {"segments": torch.cuda.memory_snapshot(), "device_traces": []}
        recorder = get_recorder()
        recorder.record_mem_snapshot(
            label=label, summary_text=summary, snapshot_obj=snap_obj,
        )
        # Compact stats only — full payload stays in the recorder.
        segments = snap_obj.get("segments", []) if isinstance(snap_obj, dict) else list(snap_obj)
        n_segments = len(segments)
        active_bytes = 0
        inactive_bytes = 0
        for seg in segments:
            for block in seg.get("blocks", ()):
                size = int(block.get("size", 0))
                state = block.get("state", "")
                if "active" in state:
                    active_bytes += size
                elif "inactive" in state:
                    inactive_bytes += size
        traces = snap_obj.get("device_traces", []) if isinstance(snap_obj, dict) else []
        return {
            "ok": True,
            "label": label,
            "n_segments": n_segments,
            "active_mb": active_bytes / (1024 * 1024),
            "inactive_mb": inactive_bytes / (1024 * 1024),
            "n_trace_events": sum(len(t) for t in traces) if traces else 0,
            "history_recorded": memory_history_enabled(),
        }
    except Exception as exc:
        logger.exception("take_memory_snapshot failed")
        return {"ok": False, "reason": f"exception: {exc}"}
