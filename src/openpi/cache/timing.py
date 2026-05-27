"""Timing system for the Pi0/Pi0.5 inference cache pipeline.

Overview
--------
This module provides ``SystemTimer``, a pluggable, hardware-aware latency
measurement system.  It is designed to time all components in the inference +
cache pipeline without interfering with inference correctness or performance.

Architecture
------------
Measurement is *probe-based*: each pipeline component (stage1_vision,
cp1_gate, …) registers a named *probe* once at startup, then calls
``timer.measure("probe_name")`` in the hot path.  Two backends handle the
actual timing:

* ``CudaEventBackend`` — for GPU-resident operations (inference stages,
  GPU-side vector search).  Uses ``torch.cuda.Event`` to record GPU-side
  timestamps; ``end_event.synchronize()`` blocks the CPU only until *that*
  specific event is processed, without flushing the entire CUDA pipeline.
  Falls back to ``PerfCounterBackend`` when CUDA is unavailable.

* ``PerfCounterBackend`` — for CPU-only operations (gate decisions, FAISS
  CPU search, write-back threads, etc.).  Uses ``time.perf_counter_ns`` for
  sub-microsecond resolution.

Task lifecycle
--------------
A "task" corresponds to one client WebSocket connection.  Call
``on_task_begin()`` when the connection opens and ``on_task_end()`` when it
closes.  ``on_task_end()`` prints a per-probe summary to the terminal and
(optionally) flushes all records for that task to a CSV file in one batch.

All ``TimingRecord`` objects are held in an in-memory ``deque`` (ring buffer)
during the task.  The CSV write at task end is the *only* disk IO, avoiding
per-record write overhead.

Extensibility
-------------
* **New probes**: call ``register_probe(name, backend="cpu"/"cuda")`` from
  the new component's ``__init__``.  No changes to ``SystemTimer`` needed.
* **Custom CUDA streams**: pass ``stream=your_stream`` to ``register_probe``
  so cache-stream operations are timed on the correct stream (Step 4+).
* **Resource monitors**: the ``ResourceMonitor`` protocol and
  ``add_resource_monitor`` / ``record_resource_snapshot`` interfaces track
  GPU VRAM / CPU RAM. ``CpuMonitor`` / ``GpuMonitor`` are auto-registered at
  construction and sampled on every ``measure()`` block at SNAPSHOT level+.

Thread safety
-------------
``deque.append`` is atomic under CPython's GIL, so concurrent probe writes
from different threads are safe.  However, ``on_task_end`` (which snapshots
the deque) should only be called from the main thread.  When Step 4 adds
background write-back threads, review whether a lock is needed around
``_get_task_records``.

Disabling
---------
Set ``enabled=False`` to make ``measure()`` a zero-overhead no-op.
Recommended in production when latency numbers are not needed.

Usage example::

    timer = SystemTimer(enabled=True, output_csv_dir="/tmp/timing")

    # Register probes once at startup (component __init__):
    timer.register_probe("stage1_vision", backend="cuda")
    timer.register_probe("stage2_llm",    backend="cuda")
    timer.register_probe("stage3_flow",   backend="cuda")
    timer.register_probe("cp1_gate",      backend="cpu")   # Step 4+

    # Task lifecycle (driven by WebSocket server):
    timer.on_task_begin()

    # Hot path (called once per inference):
    with timer.measure("stage1_vision"):
        stage1 = model.run_stage1(obs)
    with timer.measure("stage2_llm"):
        stage2 = model.run_stage2(stage1)

    timer.on_task_end()   # prints summary, writes CSV if configured
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from dataclasses import field
import os
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimingRecord:
    """A single timing measurement captured by one ``timer.measure()`` call.

    ``resources`` holds resource-usage snapshots (GPU VRAM, CPU RAM, etc.)
    attached to the same measurement point. It is ``None`` unless resource
    monitors are active (SNAPSHOT level+), in which case
    ``record_resource_snapshot()`` populates it.
    """

    name: str
    """Probe name, e.g. ``"stage1_vision"`` or ``"cp2_judge"``."""

    elapsed_ms: float
    """Wall / GPU execution time in milliseconds."""

    timestamp: float
    """``time.time()`` at the moment the measurement was *completed*.
    Used as the time axis when exporting to CSV."""

    task_id: int
    """Identifier of the task (connection) this record belongs to."""

    resources: dict[str, float] | None = field(default=None, compare=False)
    """Resource-monitor snapshot. Keys are metric names,
    e.g. ``{"gpu_alloc_mb": 4096.0}``; ``None`` when no monitor is active."""


@dataclass
class TimingStats:
    """Aggregated statistics for one probe over a set of ``TimingRecord`` objects."""

    name: str
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_records(cls, name: str, records: list[TimingRecord]) -> TimingStats:
        """Compute statistics from a list of records for the same probe.

        Args:
            name: Probe name.
            records: All ``TimingRecord`` objects whose ``name`` matches.
                     Must be non-empty.
        """
        values = np.array([r.elapsed_ms for r in records], dtype=np.float64)
        return cls(
            name=name,
            count=len(values),
            mean_ms=float(np.mean(values)),
            p50_ms=float(np.percentile(values, 50)),
            p95_ms=float(np.percentile(values, 95)),
            p99_ms=float(np.percentile(values, 99)),
            min_ms=float(np.min(values)),
            max_ms=float(np.max(values)),
        )


# ---------------------------------------------------------------------------
# Backend protocols and implementations
# ---------------------------------------------------------------------------

class TimingBackend(Protocol):
    """Protocol for interchangeable timing backends.

    Implementors must be reusable across multiple ``measure()`` calls.  Each
    call to ``start()`` returns an opaque *handle* that is passed unchanged to
    the corresponding ``stop()`` call.

    Implementors must **not** hold mutable per-measurement state as instance
    variables; all state must live in the handle returned by ``start()``.
    This allows the same backend instance to be shared across nested
    ``measure()`` calls (which will happen when cache and inference probes are
    both active in Step 4+).
    """

    def start(self) -> Any:
        """Begin timing.  Returns an opaque handle."""
        ...

    def stop(self, handle: Any) -> float:
        """End timing.  Returns elapsed time in milliseconds.

        May block the calling thread briefly (e.g. CUDA event synchronize).
        """
        ...


class CudaEventBackend:
    """GPU timing backend using ``torch.cuda.Event``.

    Records start and end events directly onto the specified CUDA stream.
    ``stop()`` calls ``end_event.synchronize()``, which blocks the CPU thread
    only until the GPU has processed the end event — it does NOT flush the
    entire CUDA pipeline (unlike ``torch.cuda.synchronize()``).

    When CUDA is unavailable (e.g. CPU-only test environments), this backend
    transparently falls back to ``PerfCounterBackend`` behaviour so that code
    running in CI without a GPU still works correctly.

    Args:
        stream: The CUDA stream on which to record events.
                ``None`` (default) records on the *current* stream at the time
                ``measure()`` is entered, which is the default stream during
                normal inference.  Pass an explicit stream (e.g.
                ``cache_stream``) for operations running on a non-default
                stream (Step 4+).

    Note:
        CUDA Event timing measures the interval between two points on the
        *GPU timeline*, not wall-clock time.  This gives the actual GPU
        execution time, excluding Python overhead and inter-stage gaps.
        It is the preferred method for Stage 1 / 2 / 3 timing.
    """

    def __init__(self, stream: torch.cuda.Stream | None = None) -> None:
        self._stream = stream
        self._fallback = PerfCounterBackend()

    def start(self) -> Any:
        if not torch.cuda.is_available():
            # No GPU: fall back to CPU nanosecond timer.
            return ("cpu", self._fallback.start())

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        # record() inserts a marker into the stream.  If self._stream is None,
        # the marker goes onto the *current* CUDA stream (default stream during
        # inference).  Using an explicit stream is required for cache_stream
        # operations in Step 4+.
        start_evt.record(self._stream)
        return ("cuda", start_evt, end_evt)

    def stop(self, handle: Any) -> float:
        if handle[0] == "cpu":
            return self._fallback.stop(handle[1])

        _, start_evt, end_evt = handle
        end_evt.record(self._stream)
        # Block only until *this* end event is processed.  Other streams and
        # kernels queued after this event are unaffected.
        end_evt.synchronize()
        # elapsed_time returns milliseconds as a float.
        return start_evt.elapsed_time(end_evt)


class PerfCounterBackend:
    """CPU timing backend using ``time.perf_counter_ns``.

    Uses the highest-resolution timer available on the platform.  Suitable
    for CPU-only operations: gate decisions, FAISS CPU search, write-back
    threads, similarity judge, etc.

    Resolution is typically sub-microsecond; accuracy depends on OS scheduler
    jitter.  For GPU operations use ``CudaEventBackend`` instead.
    """

    def start(self) -> int:
        """Returns current time in nanoseconds."""
        return time.perf_counter_ns()

    def stop(self, handle: int) -> float:
        """Returns elapsed time in milliseconds."""
        return (time.perf_counter_ns() - handle) / 1_000_000.0


# ---------------------------------------------------------------------------
# Resource monitor protocol + CPU/GPU implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class ResourceMonitor(Protocol):
    """Protocol for pluggable resource-usage monitors.

    Concrete ``CpuMonitor`` / ``GpuMonitor`` implementations live at the bottom
    of this module and are auto-registered by ``SystemTimer`` at SNAPSHOT level
    and above. ``add_resource_monitor`` registers additional monitors;
    ``record_resource_snapshot`` samples every registered monitor and merges the
    result into the current ``TimingRecord.resources``.

    A monitor exposes ``name`` (CSV key prefix) and
    ``sample() -> dict[str, float]``, e.g.::

        class GpuMonitor:
            name = "gpu"
            def sample(self) -> dict[str, float]:
                stats = torch.cuda.memory_stats()
                return {"alloc_mb": stats["allocated_bytes.all.current"] / 1e6}
    """

    name: str
    """Unique identifier for this monitor, used as key prefix in CSV."""

    def sample(self) -> dict[str, float]:
        """Capture current resource metrics.

        Returns:
            Dict mapping metric names to numeric values.
            E.g. ``{"alloc_mb": 4096.0, "peak_mb": 5120.0}``.
        """
        ...


# ---------------------------------------------------------------------------
# TaskLifecycle protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TaskLifecycle(Protocol):
    """Protocol for objects that participate in task (connection) lifecycle.

    ``WebsocketPolicyServer`` calls these methods when a client connects and
    disconnects.  The server uses ``hasattr`` to check for the protocol rather
    than importing ``InferenceInterceptor`` directly, keeping server code
    decoupled from cache internals.

    ``InferenceInterceptor`` implements this protocol by forwarding calls to
    its internal ``SystemTimer``.

    Note:
        If the policy is a plain ``Policy`` (non-cache path), these methods
        are absent and the server skips the calls gracefully.
    """

    def on_task_begin(self) -> None:
        """Called when a client connection opens.

        Resets per-task state so that the summary at task end reflects only
        the current connection's measurements.
        """
        ...

    def on_task_end(self) -> None:
        """Called when a client connection closes.

        Prints a timing summary to the terminal and (if configured) flushes
        all records for this task to a CSV file in one batch write.
        """
        ...


# ---------------------------------------------------------------------------
# SystemTimer
# ---------------------------------------------------------------------------

# Default probe display order in the summary table.  Probes not listed here
# appear after these in alphabetical order.  Update this list when new
# standard probes are added (e.g. cp1_gate in Step 4).
_DISPLAY_ORDER = [
    "stage1_vision",
    "stage2_llm",
    "stage3_flow",
    "stage3_partial_flow",
    "total_inference",
]


class SystemTimer:
    """Unified timing system for the Pi0 inference + cache pipeline.

    See module docstring for an overview and usage example.

    Args:
        enabled: When ``False``, ``measure()`` is a zero-overhead no-op.
                 Use ``False`` in production when timing data is not needed.
                 NOTE: when the server-wide ``OPENPI_MONITOR_LEVEL`` is
                 ``OFF`` the timer is force-disabled regardless of this
                 argument (the env var is the master switch — §monitor.py).
        buffer_size: Maximum number of ``TimingRecord`` objects kept in memory.
                     Older records are dropped (ring buffer) when the limit is
                     reached.  10 000 is ample for a typical robot episode
                     (< 1 000 inferences × 5 probes = 5 000 records).
        output_csv_dir: When non-None and ``auto_flush_csv=True``,
                        ``on_task_end()`` writes a CSV file here per task.
                        With ``auto_flush_csv=False`` (the new default), this
                        is just the target directory used by an *explicit*
                        ``flush_to_disk()`` call — task end keeps the data in
                        memory only.
        quiet: When ``True``, suppress the timing summary printed to stdout
               at task end.  Timing data is still collected and CSV output
               (if configured) is unaffected.  Used in concurrent server
               mode to avoid interleaved output from multiple connections.
        auto_flush_csv: Legacy behaviour switch. ``False`` (default) keeps
                        all timing records in memory + pushes a bulk copy
                        into the process-wide ``MetricsRecorder``; ``True``
                        additionally writes one CSV per task at task end
                        (pre-2026-05-24 behaviour).
    """

    def __init__(
        self,
        enabled: bool = True,
        buffer_size: int = 10_000,
        output_csv_dir: str | None = None,
        quiet: bool = False,
        auto_flush_csv: bool = False,
    ) -> None:
        # Master switch (§monitor.py). OPENPI_MONITOR_LEVEL gates two things
        # independently:
        #   - OFF       → timer is a full no-op (enabled=False).
        #   - BASIC+    → timer RECORDS into the in-memory MetricsRecorder.
        #   - SNAPSHOT+ → per-measure() CUDA-event probes are allowed.
        # CUDA-event ``stop()`` calls ``end_event.synchronize()`` — a GPU sync
        # on EVERY ``measure()`` block (stage1/2/3 + cache probes). Under
        # concurrent serving with N connections that fires ~5×N GPU syncs/s,
        # serialising the device and capping GPU utilisation (Phase-7 throughput
        # finding). So below SNAPSHOT we keep the timer RECORDING but route every
        # probe to the CPU ``PerfCounterBackend`` (no GPU sync) via
        # ``_gpu_probes_enabled``. Nothing is lost for throughput benchmarking;
        # only the GPU-accurate per-stage latency is coarser.
        self._gpu_probes_enabled = True
        try:
            from openpi.serving import monitor as _monitor
            level = _monitor.get_monitor_level()
            if level < _monitor.MonitorLevel.BASIC:
                enabled = False
            self._gpu_probes_enabled = level >= _monitor.MonitorLevel.SNAPSHOT
            self._recorder = (
                _monitor.get_recorder()
                if level >= _monitor.MonitorLevel.BASIC
                else None
            )
        except Exception:
            self._recorder = None

        self._enabled = enabled
        self._output_csv_dir = output_csv_dir
        self._quiet = quiet
        self._auto_flush_csv = auto_flush_csv

        # Ring buffer: holds TimingRecord objects.  deque.append is atomic
        # under CPython's GIL, safe for concurrent probe writes from threads.
        self._records: deque[TimingRecord] = deque(maxlen=buffer_size)

        # Monotonic counter of total records ever appended (never decremented).
        # Used to identify the boundary between tasks without relying on deque
        # length, which can wrap when buffer_size is exceeded.
        self._total_appended: int = 0

        # Set by on_task_begin(); marks where the current task starts.
        self._task_start_total: int = 0

        # Incrementing task ID; used in CSV filenames and summary headers.
        self._task_id: int = 0

        # Maps probe name → backend instance.
        # Populated by register_probe(); read-only in the hot path.
        self._probes: dict[str, TimingBackend] = {}

        # Resource monitors. Auto-registered only at SNAPSHOT+ (gpu-probes
        # level): every measure() block then also samples per-process CPU% /
        # RSS / GPU util% / GPU mem via psutil + NVML. At BASIC we deliberately
        # leave this empty so the hot path records lightweight timing rows only
        # (no per-probe psutil/NVML sampling); whole-process util still comes
        # from the coordinator's 1 Hz sampler. Users can register additional
        # monitors via ``add_resource_monitor``.
        self._resource_monitors: list[ResourceMonitor] = []
        if self._enabled and self._gpu_probes_enabled:
            try:
                self._resource_monitors.append(CpuMonitor())
                self._resource_monitors.append(GpuMonitor())
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Probe registration  (called once at component __init__, not hot path)
    # -----------------------------------------------------------------------

    def register_probe(
        self,
        name: str,
        backend: str = "cuda",
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Register a named timing probe.

        Must be called before the first ``measure(name)`` call.  Can be called
        again to update the backend (e.g. to switch a probe from "cuda" to a
        specific non-default stream in Step 4+).

        Args:
            name: Unique probe identifier.  Use the naming convention from
                  ``docs/architecture/cache_system.md`` §9:
                  ``stage1_vision``, ``cp1_gate``, ``cp2_search``, etc.
            backend: ``"cuda"`` for GPU operations (default), ``"cpu"`` for
                     CPU-only operations.  Passing an invalid string raises
                     ``ValueError``.
            stream: CUDA stream for ``"cuda"`` backend.  ``None`` means the
                    default stream (correct for inference stages).  Pass an
                    explicit ``torch.cuda.Stream`` for cache-stream operations
                    (Step 4+).  Ignored when ``backend="cpu"``.
        """
        if backend == "cuda":
            # Below SNAPSHOT level the per-measure GPU sync is suppressed: a
            # "cuda" probe falls back to CPU timing (no synchronize()), so
            # recording still works at BASIC without the throughput hit.
            if self._gpu_probes_enabled:
                self._probes[name] = CudaEventBackend(stream=stream)
            else:
                self._probes[name] = PerfCounterBackend()
        elif backend == "cpu":
            self._probes[name] = PerfCounterBackend()
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Expected 'cuda' or 'cpu'."
            )

    # -----------------------------------------------------------------------
    # Hot-path measurement context manager
    # -----------------------------------------------------------------------

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Time the body of a ``with`` block and store the result.

        When ``enabled=False``, this is a pure no-op: no objects are created,
        no GPU events are recorded.

        If ``name`` was not registered via ``register_probe()``, a
        ``CudaEventBackend`` is used as a lenient fallback so that probes
        added during development don't require upfront registration.  A
        ``RuntimeWarning`` is emitted to catch the omission during testing.

        Args:
            name: Probe name, must match a ``register_probe`` call.

        Usage::

            with timer.measure("stage2_llm"):
                past_kv = model.run_stage2(stage1)
            # After the with-block, one TimingRecord("stage2_llm", ...) has
            # been appended to the internal ring buffer.
        """
        if not self._enabled:
            yield
            return

        backend = self._probes.get(name)
        if backend is None:
            import warnings
            warnings.warn(
                f"SystemTimer: probe '{name}' was not registered via "
                "register_probe().  Using CudaEventBackend as fallback.  "
                "Call register_probe() in the component __init__ to silence this.",
                RuntimeWarning,
                stacklevel=2,
            )
            backend = (
                CudaEventBackend() if self._gpu_probes_enabled
                else PerfCounterBackend()
            )
            self._probes[name] = backend  # cache for future calls

        handle = backend.start()
        try:
            yield
        finally:
            elapsed_ms = backend.stop(handle)
            record = TimingRecord(
                name=name,
                elapsed_ms=elapsed_ms,
                timestamp=time.time(),
                task_id=self._task_id,
            )
            self._records.append(record)
            self._total_appended += 1
            if self._resource_monitors:
                self.record_resource_snapshot(name)

    # -----------------------------------------------------------------------
    # Task lifecycle  (called by WebsocketPolicyServer via TaskLifecycle)
    # -----------------------------------------------------------------------

    def on_task_begin(self) -> None:
        """Mark the start of a new task (client connection).

        Records the current ``_total_appended`` counter so that
        ``_get_task_records()`` can isolate measurements from this task even
        if the ring buffer contains records from previous tasks.

        Called by ``InferenceInterceptor.on_task_begin()``, which is invoked
        by ``WebsocketPolicyServer._handler()`` when a connection opens.
        """
        self._task_start_total = self._total_appended

    def on_task_end(self) -> None:
        """Finalise the current task.

        Actions taken in order:
        1. Compute per-probe statistics for records collected since
           ``on_task_begin()``.
        2. Print a formatted summary table to the terminal (stdout).
        3. Push all task records into the process-wide ``MetricsRecorder``
           (in-memory only — no disk IO). The records remain available via
           ``dump_metrics`` ctrl until cleared or evicted by the ring buffer.
        4. If both ``output_csv_dir`` and ``auto_flush_csv=True`` were
           passed at construction time, ALSO write a CSV file in one batch
           write. ``auto_flush_csv=False`` (the new default) keeps task end
           free of disk writes; persistent storage is opt-in via
           ``flush_to_disk()`` or the WebSocket ``dump_metrics`` handler.
        5. Increment ``task_id`` for the next connection.

        Note: Records are *not* cleared from the ring buffer; they remain
        accessible (up to ``buffer_size``) for post-hoc inspection.

        Called by ``InferenceInterceptor.on_task_end()``, which is invoked
        by ``WebsocketPolicyServer._handler()`` when a connection closes.
        """
        task_records = self._get_task_records()
        self._print_summary(task_records)

        # Push the just-completed task's rows into the process-wide
        # recorder so a single ``dump_metrics`` ctrl returns timing for
        # every connection that ever ran. The recorder ring-buffer caps
        # the total at ``timing_capacity`` (default 200k rows) to keep
        # long-lived servers from leaking memory.
        if self._enabled and task_records and self._recorder is not None:
            rows = [
                {
                    "timestamp": r.timestamp,
                    "task_id": r.task_id,
                    "name": r.name,
                    "elapsed_ms": r.elapsed_ms,
                    **({} if not r.resources else dict(r.resources)),
                }
                for r in task_records
            ]
            try:
                self._recorder.record_timing_bulk(rows)
            except Exception:
                import warnings
                warnings.warn(
                    "SystemTimer: failed to push records to MetricsRecorder; "
                    "in-memory aggregation disabled for this task.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if (
            self._enabled
            and self._auto_flush_csv
            and self._output_csv_dir is not None
        ):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                self._output_csv_dir,
                f"timing_task_{self._task_id:04d}_{ts}.csv",
            )
            self.export_csv(path, records=task_records)

        self._task_id += 1

    def flush_to_disk(
        self,
        path: str | None = None,
        *,
        task_only: bool = False,
    ) -> str:
        """Explicit bulk CSV flush — call from an operator script when you
        actually want disk persistence.

        Args:
            path: Output file path. Defaults to
                ``<output_csv_dir>/timing_flush_<ts>.csv``. ``output_csv_dir``
                must be set (constructor arg) when ``path`` is None.
            task_only: When True, only flush records from the current /
                latest task (matches ``on_task_end`` scope). When False
                (default), flush every record still in the ring buffer.

        Returns the path that was written.
        """
        if path is None:
            if self._output_csv_dir is None:
                raise ValueError(
                    "flush_to_disk(): no path given and output_csv_dir is None"
                )
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                self._output_csv_dir,
                f"timing_flush_{ts}.csv",
            )
        records = (
            self._get_task_records() if task_only else list(self._records)
        )
        self.export_csv(path, records=records)
        return path

    # -----------------------------------------------------------------------
    # Statistics and export
    # -----------------------------------------------------------------------

    def summary(self, task_only: bool = True) -> dict[str, TimingStats]:
        """Compute per-probe statistics.

        Args:
            task_only: If ``True`` (default), only include records collected
                       since the last ``on_task_begin()`` call.  If ``False``,
                       include all records currently in the ring buffer
                       (spanning multiple tasks).

        Returns:
            Dict mapping probe name → ``TimingStats``.  Probes with no
            records in the selected range are omitted.
        """
        records = self._get_task_records() if task_only else list(self._records)
        return _compute_stats(records)

    def export_csv(
        self,
        path: str,
        records: list[TimingRecord] | None = None,
    ) -> None:
        """Write timing records to a CSV file in a single batch write.

        All rows are first assembled in memory, then written to disk at once.
        This avoids the overhead of opening/closing the file or flushing after
        each row, which is important when exporting thousands of records at
        task end.

        Args:
            path: Absolute or relative path for the output file.  Parent
                  directories must already exist.
            records: Records to export.  Defaults to the current task's
                     records (same as ``summary(task_only=True)``).

        CSV columns:
            ``timestamp``, ``task_id``, ``name``, ``elapsed_ms``
            Future columns for resource data will be appended when
            ``ResourceMonitor`` implementations are added.
        """
        if records is None:
            records = self._get_task_records()

        # Determine resource columns (present only if any record has resources).
        resource_keys: list[str] = []
        for r in records:
            if r.resources:
                resource_keys = sorted(r.resources.keys())
                break

        # Build all rows in memory first, then write once.
        rows: list[list[Any]] = []
        for r in records:
            row: list[Any] = [r.timestamp, r.task_id, r.name, r.elapsed_ms]
            for k in resource_keys:
                row.append(r.resources.get(k, "") if r.resources else "")
            rows.append(row)

        header = ["timestamp", "task_id", "name", "elapsed_ms"] + resource_keys
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)  # single batch write

    # -----------------------------------------------------------------------
    # Resource monitor interface
    # -----------------------------------------------------------------------

    def add_resource_monitor(self, monitor: ResourceMonitor) -> None:
        """Register a resource-usage monitor.

        Registered monitors are sampled by ``record_resource_snapshot`` (called
        automatically after each ``measure()`` block at SNAPSHOT level+).
        Multiple monitors can be registered; each contributes its own keys to
        ``TimingRecord.resources``.

        Args:
            monitor: Any object implementing the ``ResourceMonitor`` protocol.
        """
        self._resource_monitors.append(monitor)

    def record_resource_snapshot(self, name: str) -> None:
        """Sample all registered monitors and attach results to the most
        recent ``TimingRecord`` whose ``name`` matches.

        Iterates over ``self._resource_monitors`` (added via
        ``add_resource_monitor``), calls each ``.sample()``, and merges
        the returned ``{key: float}`` dict into the matching record's
        ``resources`` field. Keys are prefixed with the monitor's
        ``name`` so different monitors do not collide.

        Args:
            name: Probe name identifying which record to annotate.
        """
        if not self._enabled or not self._resource_monitors:
            return
        # Find the most recent record matching `name`. Records are appended in
        # order so scanning the tail of the deque is cheap.
        target: TimingRecord | None = None
        for r in reversed(self._records):
            if r.name == name:
                target = r
                break
        if target is None:
            return
        if target.resources is None:
            target.resources = {}
        for monitor in self._resource_monitors:
            try:
                snap = monitor.sample()
            except Exception:
                continue
            prefix = getattr(monitor, "name", monitor.__class__.__name__)
            for k, v in snap.items():
                target.resources[f"{prefix}.{k}"] = float(v)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_task_records(self) -> list[TimingRecord]:
        """Return records appended since the last ``on_task_begin()`` call.

        Uses the monotonic ``_total_appended`` counter to compute how many
        records belong to the current task, then slices from the tail of the
        deque.  This is robust when the deque wraps (i.e. ``buffer_size``
        records were added within one task); in that case only the most recent
        ``buffer_size`` records are returned with a warning.
        """
        task_count = self._total_appended - self._task_start_total
        if task_count <= 0:
            return []

        all_records = list(self._records)  # snapshot
        if task_count > len(all_records):
            import warnings
            warnings.warn(
                f"SystemTimer: ring buffer overflowed during task "
                f"(task produced {task_count} records, buffer holds "
                f"{len(all_records)}).  Only the most recent "
                f"{len(all_records)} records are available.  "
                "Increase buffer_size if full task history is needed.",
                RuntimeWarning,
                stacklevel=3,
            )
            return all_records

        return all_records[-task_count:]

    def _print_summary(self, task_records: list[TimingRecord]) -> None:
        """Print a formatted timing summary table to stdout.

        The table shows per-probe statistics (count, mean, p50, p95, p99) in
        milliseconds.  Probes are displayed in ``_DISPLAY_ORDER``; unrecognised
        probes appear afterwards in alphabetical order.

        A "total (sum)" row is appended when ``stage1_vision``,
        ``stage2_llm``, and ``stage3_flow`` all have the same call count,
        computed by summing the three stage times for each individual
        inference call (not by summing per-probe means, which would give the
        correct mean but incorrect percentiles).

        When ``quiet=True``, the summary is suppressed (used in concurrent
        mode to avoid interleaved output from multiple connections).
        """
        if self._quiet:
            return

        if not task_records:
            print(
                f"\n=== Inference Timing Summary "
                f"(task #{self._task_id}) — no records ===\n"
            )
            return

        stats = _compute_stats(task_records)
        if not stats:
            return

        # Determine number of inferences = max probe count in this task.
        n_calls = max(s.count for s in stats.values())

        # Build display order: standard probes first, then alphabetical rest.
        ordered_names = [n for n in _DISPLAY_ORDER if n in stats]
        extra = sorted(n for n in stats if n not in _DISPLAY_ORDER)
        ordered_names.extend(extra)

        col_w = 26  # width of probe name column
        header = (
            f"  {'Probe':<{col_w}} {'N':>5} "
            f"{'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}  ms"
        )
        separator = "  " + "-" * (len(header) - 2)

        lines = [
            f"\n=== Inference Timing Summary "
            f"(task #{self._task_id}, {n_calls} inferences) ===",
            header,
            separator,
        ]

        for name in ordered_names:
            s = stats[name]
            lines.append(
                f"  {name:<{col_w}} {s.count:>5} "
                f"{s.mean_ms:>8.1f} {s.p50_ms:>8.1f} "
                f"{s.p95_ms:>8.1f} {s.p99_ms:>8.1f}"
            )

        # Per-call total row: sum stage1 + stage2 + stage3 per inference.
        stage_names = ("stage1_vision", "stage2_llm", "stage3_flow")
        if all(n in stats for n in stage_names):
            counts = [stats[n].count for n in stage_names]
            if len(set(counts)) == 1:  # all three have identical call counts
                # Reconstruct per-call totals from the raw records.
                per_probe: dict[str, list[float]] = {
                    n: [] for n in stage_names
                }
                for r in task_records:
                    if r.name in per_probe:
                        per_probe[r.name].append(r.elapsed_ms)

                # Zip the three lists to get per-call sums.
                min_len = min(len(v) for v in per_probe.values())
                totals = [
                    per_probe["stage1_vision"][i]
                    + per_probe["stage2_llm"][i]
                    + per_probe["stage3_flow"][i]
                    for i in range(min_len)
                ]
                if totals:
                    arr = np.array(totals)
                    lines.append(separator)
                    lines.append(
                        f"  {'total (sum)':<{col_w}} {len(totals):>5} "
                        f"{float(np.mean(arr)):>8.1f} "
                        f"{float(np.percentile(arr, 50)):>8.1f} "
                        f"{float(np.percentile(arr, 95)):>8.1f} "
                        f"{float(np.percentile(arr, 99)):>8.1f}"
                    )

        lines.append("")
        print("\n".join(lines))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _compute_stats(records: list[TimingRecord]) -> dict[str, TimingStats]:
    """Group ``records`` by probe name and compute ``TimingStats`` for each.

    Records with zero elapsed time are included (they indicate fast/cached
    paths and should not be silently dropped).
    """
    grouped: dict[str, list[TimingRecord]] = {}
    for r in records:
        grouped.setdefault(r.name, []).append(r)

    return {
        name: TimingStats.from_records(name, recs)
        for name, recs in grouped.items()
    }


# ---------------------------------------------------------------------------
# ResourceMonitor implementations (concrete classes for the Protocol above)
# ---------------------------------------------------------------------------


class CpuMonitor:
    """Per-process CPU% and RSS (MB) via psutil.

    ``cpu%`` is the proportion of one CPU core used by this process since
    the last sample (psutil semantics). ``rss_mb`` is resident memory in
    megabytes.
    """

    name = "cpu"

    def __init__(self) -> None:
        try:
            import psutil

            self._proc = psutil.Process(os.getpid())
            self._proc.cpu_percent(interval=None)  # prime
        except ImportError:
            self._proc = None

    def sample(self) -> dict[str, float]:
        if self._proc is None:
            return {}
        try:
            return {
                "proc_pct": float(self._proc.cpu_percent(interval=None)),
                "rss_mb": float(self._proc.memory_info().rss) / (1024 * 1024),
            }
        except Exception:
            return {}


class GpuMonitor:
    """Per-GPU utilisation% and memory.used (MB) via pynvml.

    Reads the device pinned by ``CUDA_VISIBLE_DEVICES[0]`` so the numbers
    match what the server process is actually using.
    """

    name = "gpu"

    def __init__(self) -> None:
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            try:
                idx = int(visible) if visible.strip() else 0
            except ValueError:
                idx = 0
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._pynvml = pynvml
        except Exception:
            pass

    def sample(self) -> dict[str, float]:
        if self._handle is None:
            return {}
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {
                "util_pct": float(util.gpu),
                "mem_used_mb": float(mem.used) / (1024 * 1024),
            }
        except Exception:
            return {}
