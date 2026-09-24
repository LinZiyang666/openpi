"""Model-agnostic stage batching core (plan logs/cache_trace_mode_plan.log.md §6.1).

Everything ``BatchingCoordinator`` did that is not Pi0.5-specific lives here:
the three stage queues and worker threads, the pull-then-group and
bucket-first schedulers, the metrics, the submit / enqueue / wait API and the
Stage-3 sub-bucketing. Every model-specific operation — stacking and
splitting stage outputs, the bucket key, the default snapshot set, the actual
forwards — goes through the ``StageBatcher`` adapter, so the GR00T island
(which cannot import jax) reuses this core with its own batcher and the
Pi0.5 coordinator keeps its historical behaviour through
``openpi.serving.batching_coordinator.Pi05StageBatcher``.

Trace additions (all additive): ``Stage3MissPayload.save_timesteps`` (per
request snapshot set; ``None`` = adapter default and the adapter's original
call shape), ``ready_events`` on both Stage-3 payloads (producer-stream
events the worker waits on before it reads the inputs),
``submit_many_to_stage`` (one decision's variants enqueued together so they
group with other connections' same-key requests), and per-bucket fault
isolation in the Stage-3 loop.

This module imports torch and the serving monitor only — never jax, the
models or ``stage_io``.
"""

from __future__ import annotations

import collections
import contextlib
from dataclasses import dataclass
import logging
import queue
import threading
import time
from typing import Any, Hashable, Literal, Protocol, Sequence, Union, runtime_checkable

import torch

from openpi.serving import monitor as _monitor

logger = logging.getLogger(__name__)

# Singleton reference so the websocket ctrl handler can hot-mutate batch
# params without restarting the server (Phase 7 follow-up: avoids the
# restart-+-preload cycle for each param-sweep point in the throughput
# benchmark grid).
_ACTIVE_COORDINATOR: "BatchingCore | None" = None
# Fatal CUDA failures cannot prove input completion. Keep these references
# even if a caller drops/stops the core; process teardown is the release point.
_FATAL_PAYLOADS: list[Any] = []

# Finite backstop for submit_to_stage callers that pass no timeout (the
# interceptor passes None): if a stage worker thread ever dies, callers would
# otherwise wait on reply_event forever. A wedged/dead stage then surfaces as a
# TimeoutError instead of hanging every connection. Far above any real
# batched-inference time (~seconds).
_DEFAULT_SUBMIT_TIMEOUT_S = 300.0


class _FatalStageError(RuntimeError):
    """CUDA completion could not be established; this core cannot be reused."""


def get_active_coordinator() -> "BatchingCore | None":
    """Return the currently-started coordinator, or None when serving is in
    single-connection mode or coordinator hasn't been spawned yet."""
    return _ACTIVE_COORDINATOR


def record_ready_events(device: str | torch.device | None) -> tuple:
    """Event(s) marking every write to a payload's inputs on the producer stream.

    Called by the producing thread right after it finished writing / moving
    ``stage2`` / ``noise`` / ``start_x``; the worker waits on them before it
    reads the tensors (plan §6.2). Empty on CPU.
    """
    if device is None:
        return ()
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return ()
    ev = torch.cuda.Event()
    ev.record(torch.cuda.current_stream(dev))
    return (ev,)


# ------------------------------------------------------------------
# Stage 3 payloads (sum type — G1 R2 Item 3: no shared `noise` field)
# ------------------------------------------------------------------


@dataclass
class Stage3MissPayload:
    """MISS path: ``run_stage3(stage2, noise, return_intermediates=True)``.

    The Interceptor builds this when CP1 returns MISS, sampling a fresh noise
    on the per-request thread (matches the legacy single-request semantics).
    ``save_timesteps=None`` keeps the adapter's default snapshot set and its
    original call shape; a trace build passes the whole schedule.
    ``ready_events`` are producer-stream events (``record_ready_events``).
    """
    stage2_out: Any
    noise: torch.Tensor      # [action_horizon, action_dim] — unbatched
    num_steps: int = 10
    save_timesteps: tuple[float, ...] | None = None
    ready_events: tuple = ()


@dataclass
class Stage3WarmStartPayload:
    """WARM_START path: ``run_stage3_from(stage2, start_x, start_t, num_steps=...)``.

    No ``noise`` field — WARM_START continues from the cached ``start_x``;
    the noise that produced ``start_x`` was consumed when the original MISS
    cache entry was created. ``capture_first_step`` asks the adapter to keep
    the first executed step's input / output (online-RIT feedback).
    """
    stage2_out: Any
    start_x: torch.Tensor    # [action_horizon, action_dim] — unbatched
    start_t: float
    num_steps: int
    capture_first_step: bool = False
    ready_events: tuple = ()


Stage3InitPayload = Union[Stage3MissPayload, Stage3WarmStartPayload]


@dataclass
class Stage3VariantOutput:
    """Model-agnostic Stage-3 result of one request (plan §6.1).

    ``action`` keeps the unit batch axis (``[1, H, D]``) so the same object
    is read by the direct and the coordinator path; ``intermediates`` maps
    snapshot ``t`` to the ``[1, H, D]`` chunk that loop step consumed.
    """

    action: torch.Tensor                      # [1, H, D]
    intermediates: dict | None = None
    first_step_input: torch.Tensor | None = None
    first_step_x: torch.Tensor | None = None


# ------------------------------------------------------------------
# StageRequest envelope
# ------------------------------------------------------------------


@dataclass
class StageRequest:
    """A single per-request submission to one of the three stage queues."""

    request_id: str
    bundle_id: str
    stage_id: Literal[1, 2, 3]
    # Per stage_id:
    #   1: payload = unbatched obs dict (any pytree of np/torch leaves)
    #   2: payload = Stage1Output (B=1 batched, from split_stage1_output)
    #   3: payload = Stage3MissPayload | Stage3WarmStartPayload
    payload: Any
    reply_event: threading.Event
    reply_slot: list | None = None  # list-of-len-1 used as a mutable slot; None until worker writes
    error: BaseException | None = None
    enqueue_t: float = 0.0  # time.monotonic() at submit — for wait_ms instrumentation
    # CP2 capture capability, frozen from the submitting connection's own
    # config snapshot (never from a mutable bundle registry): a stage-2 batch
    # containing any True request runs ``run_stage2_capture`` so that request
    # receives ``Stage2Output.prefix_out``. Default False keeps every legacy
    # caller on ``run_stage2``.
    requires_stage2_capture: bool = False


# ------------------------------------------------------------------
# Model adapter protocol
# ------------------------------------------------------------------


@runtime_checkable
class StageBatcher(Protocol):
    """Everything model-specific the core delegates (plan §6.1).

    ``run_stage3_miss`` receives every request's own ``save_timesteps``
    (``None`` = the adapter's default) and must return one native output per
    request carrying exactly that request's snapshot set; when no request set
    anything the adapter keeps its original call shape.
    """

    def bucket_key(self, payload: Any) -> Hashable: ...

    def run_stage1_batch(self, payloads: list) -> list: ...

    def run_stage2_batch(self, payloads: list, *, capture: bool) -> list: ...

    def run_stage3_miss(
        self,
        payloads: list,
        *,
        num_steps: int,
        save_timesteps_per_request: list,
    ) -> list: ...

    def run_stage3_warm(
        self,
        payloads: list,
        *,
        start_t: float,
        num_steps: int,
        capture_first_step: bool,
    ) -> list: ...


# ------------------------------------------------------------------
# Core
# ------------------------------------------------------------------


class BatchingCore:
    """Three-stage dynamic batching worker pool over a ``StageBatcher``.

    Lifecycle: ``start()`` spawns three daemon threads (one per stage),
    ``stop()`` signals them to drain and exit. Use as a context manager when
    convenient (``with BatchingCore(...) as bc: ...``).

    Parameters
    ----------
    batcher:
        The model adapter (``StageBatcher``) that owns every model-specific
        operation: stacking / splitting, bucket keys, default snapshot set,
        and the three stage forwards.
    device:
        Device the worker streams are created on.
    max_batch_size:
        Max requests merged into one forward. Default 32.
    max_wait_ms:
        Max time the worker waits for additional requests before issuing a
        partial batch. Default 25ms.
    """

    def __init__(
        self,
        batcher: "StageBatcher",
        *,
        device: str | torch.device,
        # Measured sweet spot for LIBERO closed-loop serving, not a guess:
        # under closed loop each window issues few requests, so a batch rarely
        # fills and a long wait is pure latency tax. 10 ms is too aggressive for
        # a batch to form at all; >50 ms only adds latency. 32 is the safe upper
        # bound on batch size. See docs/experiments/conductor_tutorial.md §8.1.
        max_batch_size: int = 32,
        max_wait_ms: float = 25.0,
    ) -> None:
        self._batcher = batcher
        self._device = torch.device(device)
        self._max_batch_size = int(max_batch_size)
        self._max_wait_s = float(max_wait_ms) / 1000.0

        # Per-stage CUDA streams (Phase-7 throughput fix). All three stage
        # worker threads otherwise launch kernels onto the *default* stream,
        # which serialises GPU work even though the stages run on separate
        # threads — the GPU starves between launches (util capped ~33 %).
        # Giving each stage its own stream lets e.g. stage1(req B) overlap
        # stage3(req A) on the GPU. Cross-stage tensor handoff is safe
        # because it round-trips through the interceptor's per-connection
        # CP logic (which reads tensors host-side, forcing a sync) before
        # the next stage submit. Env ``OPENPI_DISABLE_STAGE_STREAMS=1``
        # falls back to the default stream for A/B comparison.
        # Per-stage worker-thread count. stage3 (10-step denoise) is the
        # slowest stage and gates pipeline throughput because a single worker
        # thread processes one batch at a time. Run multiple stage3 worker
        # threads (each pulling independent batches from the shared stage3
        # queue, each on its own CUDA stream) so several denoise batches
        # execute concurrently on the GPU. Batches are independent (per-request
        # KV caches in the payload, no shared state), so correctness holds.
        import os as _os_s
        self._stage_worker_counts = {
            1: int(_os_s.environ.get("BATCHING_STAGE1_WORKERS", "1")),
            2: int(_os_s.environ.get("BATCHING_STAGE2_WORKERS", "1")),
            3: int(_os_s.environ.get("BATCHING_STAGE3_WORKERS", "1")),
        }
        # One CUDA stream per (stage, worker) so concurrent workers don't
        # serialise on the default stream. Keyed by (stage_id, worker_idx).
        self._worker_streams: dict[tuple[int, int], torch.cuda.Stream | None] = {}
        self._streams_enabled = False
        try:
            if (
                torch.cuda.is_available()
                and self._device.type == "cuda"
                and _os_s.environ.get("OPENPI_DISABLE_STAGE_STREAMS", "") != "1"
            ):
                for _sid, _cnt in self._stage_worker_counts.items():
                    for _w in range(_cnt):
                        self._worker_streams[(_sid, _w)] = torch.cuda.Stream(device=self._device)
                self._streams_enabled = True
                logger.info(
                    "BatchingCoordinator: per-worker CUDA streams enabled; "
                    "stage workers = %s", self._stage_worker_counts,
                )
        except Exception:
            self._worker_streams = {}
            self._streams_enabled = False

        self._queues: dict[int, queue.Queue[StageRequest]] = {
            1: queue.Queue(), 2: queue.Queue(), 3: queue.Queue(),
        }
        self._stop = threading.Event()
        self._admission_lock = threading.Lock()
        self.fatal_error: BaseException | None = None
        # Keep unsafe inputs alive until process teardown if CUDA cannot
        # confirm completion. A failed context must never recycle them.
        self._fatal_payloads: list[Any] = []
        self._stage_threads: list[threading.Thread] = []
        # KV-cache leak guard counter — see _stage_loop. Always runs
        # regardless of monitor level; not an instrumentation field.
        self._batches_since_clear = 0
        # Per-batch breakdown timers. Thread-local because stage1/2/3 (and any
        # env-enabled extra per-stage workers) run _run_batch concurrently on
        # different threads — a shared attribute would let one stage's record
        # pick up another stage's assemble/forward split (G2 R1 race fix).
        self._tls = threading.local()
        # Resolve the server-wide monitor level once at coordinator
        # construction. ``OFF`` skips the util sampler thread entirely
        # (no pynvml init, no per-batch metric computation) so the
        # baseline serving path has zero monitoring overhead.
        self._monitor_level = _monitor.get_monitor_level()
        self._recorder = (
            _monitor.get_recorder()
            if self._monitor_level >= _monitor.MonitorLevel.BASIC
            else None
        )
        self._util_proc = None
        self._util_gpu_handle = None
        self._util_pynvml = None
        self._util_thread: threading.Thread | None = None
        if self._monitor_level >= _monitor.MonitorLevel.BASIC:
            # In-process util sampler thread. Logs ``[metrics.util]`` every
            # 5s and pushes the same structured event into the recorder's
            # in-memory ring buffer so a single ``dump_metrics`` ctrl returns
            # the full timeseries without grepping log files.
            try:
                import os as _os

                import psutil as _ps
                self._util_proc = _ps.Process(_os.getpid())
                self._util_proc.cpu_percent(interval=None)
            except Exception:
                pass
            try:
                import os as _os

                import pynvml as _pv
                _pv.nvmlInit()
                visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                try:
                    idx = int(visible) if visible.strip() else 0
                except ValueError:
                    idx = 0
                self._util_gpu_handle = _pv.nvmlDeviceGetHandleByIndex(idx)
                self._util_pynvml = _pv
            except Exception:
                pass
            self._util_thread = threading.Thread(
                target=self._util_loop, name="BatchingCoordinator-util", daemon=True,
            )
            self._util_thread.start()


    def _util_loop(self) -> None:
        while not self._stop.is_set():
            cpu_proc_pct = 0.0
            rss_mb = 0.0
            sys_cpu_pct = 0.0
            sys_ram_used_mb = 0.0
            sys_ram_total_mb = 0.0
            cgroup_ram_used_mb = 0.0
            gpu_util = 0
            gpu_mem_mb = 0
            torch_alloc_mb = 0.0
            torch_reserved_mb = 0.0
            torch_active_mb = 0.0
            torch_alloc_retries = 0
            try:
                if self._util_proc is not None:
                    cpu_proc_pct = float(self._util_proc.cpu_percent(interval=None))
                    rss_mb = float(self._util_proc.memory_info().rss) / (1024 * 1024)
                import psutil as _ps
                sys_cpu_pct = float(_ps.cpu_percent(interval=None))
                vm = _ps.virtual_memory()
                sys_ram_used_mb = float(vm.used) / (1024 * 1024)
                sys_ram_total_mb = float(vm.total) / (1024 * 1024)
            except Exception:
                pass
            try:
                with open("/sys/fs/cgroup/memory.current") as f:
                    cgroup_ram_used_mb = float(f.read().strip()) / (1024 * 1024)
            except Exception:
                pass
            try:
                if self._util_gpu_handle is not None:
                    u = self._util_pynvml.nvmlDeviceGetUtilizationRates(self._util_gpu_handle)
                    m = self._util_pynvml.nvmlDeviceGetMemoryInfo(self._util_gpu_handle)
                    gpu_util = int(u.gpu)
                    gpu_mem_mb = int(m.used / (1024 * 1024))
            except Exception:
                pass
            # Detailed PyTorch CUDA accounting: distinguishes live tensors
            # ("allocated") from PyTorch's cached pool ("reserved"). gpu_mem_mb
            # (nvidia-smi) ≈ reserved + cuda runtime overhead. If allocated <<
            # reserved, leak is fragmentation. If allocated ≈ reserved ≈
            # gpu_mem_mb, leak is live tensors not GC'd.
            try:
                if torch.cuda.is_available():
                    s = torch.cuda.memory_stats()
                    torch_alloc_mb = s.get("allocated_bytes.all.current", 0) / (1024 * 1024)
                    torch_reserved_mb = s.get("reserved_bytes.all.current", 0) / (1024 * 1024)
                    torch_active_mb = s.get("active_bytes.all.current", 0) / (1024 * 1024)
                    torch_alloc_retries = s.get("num_alloc_retries", 0)
            except Exception:
                pass
            # Queue depth across all stages — pending work pinned in coordinator.
            q1 = self._queues[1].qsize()
            q2 = self._queues[2].qsize()
            q3 = self._queues[3].qsize()
            logger.info(
                "[metrics.util] cpu_proc_pct=%.1f rss_mb=%.0f sys_cpu_pct=%.1f "
                "sys_ram_mb=%.0f/%.0f cgroup_ram_mb=%.0f gpu_util_pct=%d gpu_mem_mb=%d "
                "torch_alloc_mb=%.0f torch_reserved_mb=%.0f torch_active_mb=%.0f "
                "torch_retries=%d q1=%d q2=%d q3=%d",
                cpu_proc_pct, rss_mb, sys_cpu_pct,
                sys_ram_used_mb, sys_ram_total_mb, cgroup_ram_used_mb,
                gpu_util, gpu_mem_mb,
                torch_alloc_mb, torch_reserved_mb, torch_active_mb,
                torch_alloc_retries, q1, q2, q3,
            )
            if self._recorder is not None:
                self._recorder.record_util({
                    "cpu_proc_pct": cpu_proc_pct,
                    "rss_mb": rss_mb,
                    "sys_cpu_pct": sys_cpu_pct,
                    "sys_ram_used_mb": sys_ram_used_mb,
                    "sys_ram_total_mb": sys_ram_total_mb,
                    "cgroup_ram_used_mb": cgroup_ram_used_mb,
                    "gpu_util_pct": gpu_util,
                    "gpu_mem_mb": gpu_mem_mb,
                    "torch_alloc_mb": torch_alloc_mb,
                    "torch_reserved_mb": torch_reserved_mb,
                    "torch_active_mb": torch_active_mb,
                    "torch_alloc_retries": torch_alloc_retries,
                    "q1": q1, "q2": q2, "q3": q3,
                })
            # 1 Hz sampling — at 5 Hz the [metrics.util] log line was too
            # coarse to estimate avg GPU utilisation over short ramps
            # (< 30 s ramp → only 6 samples). At 1 s we get 30+ samples
            # which yields a useful arithmetic mean. Sampler overhead is
            # ~1 ms per tick (pynvml + psutil + torch.cuda.memory_stats),
            # negligible vs 100-200 ms inference latency.
            self._stop.wait(1.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn ``self._stage_worker_counts[stage]`` daemon worker threads
        per stage. Multiple workers on the same stage pull independent batches
        from the shared stage queue and run concurrently (each on its own CUDA
        stream), letting the bottleneck stage process several batches at once.
        """
        if self._stage_threads:
            raise RuntimeError("BatchingCoordinator.start: already started")
        if self._stop.is_set():
            raise RuntimeError("BatchingCoordinator.start: core has stopped") from self.fatal_error
        for stage_id in (1, 2, 3):
            for worker_idx in range(self._stage_worker_counts[stage_id]):
                t = threading.Thread(
                    target=self._stage_loop, args=(stage_id, worker_idx),
                    name=f"BatchingCoordinator-stage{stage_id}-w{worker_idx}",
                    daemon=True,
                )
                self._stage_threads.append(t)
                t.start()
        # Publish self as the active coordinator so ctrl handlers can
        # mutate batch params at runtime (see ``get_active_coordinator``).
        global _ACTIVE_COORDINATOR
        _ACTIVE_COORDINATOR = self

    def update_batch_params(
        self,
        *,
        max_batch_size: int | None = None,
        max_wait_ms: float | None = None,
    ) -> dict:
        """Hot-mutate dispatcher parameters from a control thread.

        Eventually-consistent: each attribute write is atomic under the GIL, but
        a stage loop may read ``_max_batch_size`` and ``_max_wait_s`` on
        different iterations, so a single in-flight batch can straddle old and
        new values. Harmless — it self-corrects on the next batch.
        """
        if max_batch_size is not None:
            self._max_batch_size = int(max_batch_size)
        if max_wait_ms is not None:
            self._max_wait_s = float(max_wait_ms) / 1000.0
        logger.info(
            "BatchingCoordinator params updated: max_batch_size=%d max_wait_ms=%.1f",
            self._max_batch_size, self._max_wait_s * 1000.0,
        )
        return {
            "max_batch_size": self._max_batch_size,
            "max_wait_ms": self._max_wait_s * 1000.0,
        }

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal the workers to drain and exit; join with ``timeout`` each."""
        with self._admission_lock:
            self._stop.set()
        for t in self._stage_threads:
            t.join(timeout=timeout)
        self._stage_threads = [t for t in self._stage_threads if t.is_alive()]
        global _ACTIVE_COORDINATOR
        if _ACTIVE_COORDINATOR is self and not self._stage_threads and self.fatal_error is None:
            _ACTIVE_COORDINATOR = None

    def __enter__(self) -> "BatchingCore":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Submit (blocking)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Submit (blocking)
    # ------------------------------------------------------------------

    def enqueue_to_stage(
        self,
        stage_id: Literal[1, 2, 3],
        bundle_id: str,
        payload: Any,
        *,
        request_id: str | None = None,
        requires_stage2_capture: bool = False,
    ) -> StageRequest:
        """Enqueue a request to ``stage_id`` and return its envelope (non-blocking)."""
        if not self._stage_threads:
            raise RuntimeError(
                "BatchingCoordinator.submit_to_stage: coordinator not started"
            )
        if stage_id not in (1, 2, 3):
            raise ValueError(f"Invalid stage_id {stage_id!r}; expected 1, 2, or 3")
        req = StageRequest(
            request_id=request_id or "anon",
            bundle_id=bundle_id,
            stage_id=stage_id,
            payload=payload,
            reply_event=threading.Event(),
            reply_slot=[None],
            enqueue_t=time.monotonic(),
            requires_stage2_capture=bool(requires_stage2_capture),
        )
        with self._admission_lock:
            if self._stop.is_set():
                raise RuntimeError("BatchingCoordinator is stopping") from self.fatal_error
            self._queues[stage_id].put(req)
        return req

    def _default_submit_timeout(self) -> float:
        """Finite backstop used when a caller passes ``timeout=None``."""
        return _DEFAULT_SUBMIT_TIMEOUT_S

    def wait_for(self, req: StageRequest, *, timeout: float | None = None):
        """Block until the worker filled ``req`` (or ``timeout``), return its output.

        A timeout only ends the wait: the worker still owns the request and
        its input tensors until it finishes or fails the batch (plan §6.2).
        """
        # None -> finite backstop so a dead stage worker thread cannot hang
        # the caller forever.
        effective_timeout = self._default_submit_timeout() if timeout is None else timeout
        if not req.reply_event.wait(timeout=effective_timeout):
            raise TimeoutError(
                f"BatchingCoordinator.submit_to_stage(stage_id={req.stage_id}) timed out"
            )
        if req.error is not None:
            raise req.error
        return req.reply_slot[0]

    def submit_to_stage(
        self,
        stage_id: Literal[1, 2, 3],
        bundle_id: str,
        payload: Any,
        *,
        request_id: str | None = None,
        timeout: float | None = None,
        requires_stage2_capture: bool = False,
    ):
        """Submit a request to ``stage_id``; block until the worker fills in
        the per-request output.

        Raises ``TimeoutError`` on timeout (``None`` = finite backstop of
        ``_DEFAULT_SUBMIT_TIMEOUT_S``) and re-raises any exception the stage
        worker recorded for the request.
        """
        req = self.enqueue_to_stage(
            stage_id,
            bundle_id,
            payload,
            request_id=request_id,
            requires_stage2_capture=requires_stage2_capture,
        )
        return self.wait_for(req, timeout=timeout)

    def submit_many_to_stage(
        self,
        stage_id: Literal[1, 2, 3],
        bundle_id: str,
        payloads: Sequence[Any],
        *,
        request_id: str | None = None,
        timeout: float | None = None,
        requires_stage2_capture: bool = False,
    ) -> list:
        """Submit several requests of one decision together (plan §6.1).

        All payloads are enqueued back to back (microseconds apart) so the
        worker's next pull groups them with other connections' same-key
        requests; then each is awaited in order under one shared deadline.
        If any request fails, the remaining enqueued siblings are still
        harvested (the worker owns them until they complete) and the first
        error is raised afterwards. A payload that could not be enqueued at
        all never reaches a worker and its error is raised after the others.
        """
        reqs: list[StageRequest] = []
        enqueue_error: BaseException | None = None
        for payload in payloads:
            try:
                reqs.append(
                    self.enqueue_to_stage(
                        stage_id,
                        bundle_id,
                        payload,
                        request_id=request_id,
                        requires_stage2_capture=requires_stage2_capture,
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - remaining siblings are never enqueued
                enqueue_error = exc
                break
        effective_timeout = self._default_submit_timeout() if timeout is None else timeout
        deadline = time.monotonic() + effective_timeout
        outs: list = []
        first_error: BaseException | None = None
        for req in reqs:
            try:
                remaining = max(0.0, deadline - time.monotonic())
                outs.append(self.wait_for(req, timeout=remaining))
            except BaseException as exc:  # noqa: BLE001 - harvest the rest first
                outs.append(None)
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        if enqueue_error is not None:
            raise enqueue_error
        return outs

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _stage_loop(self, stage_id: int, worker_idx: int = 0) -> None:
        """Pull a batch off the queue, run it, distribute outputs.

        Wraps the entire worker loop in ``torch.no_grad()`` because the
        interceptor's own ``torch.no_grad()`` context is thread-local and
        DOES NOT propagate to this background thread (Phase 7 mem-attribution
        finding: without this every Linear / MLP forward in
        ``run_stage1/2/3`` builds an autograd graph and retains
        activations for a backward pass that never comes — ~6 GB / active
        connection on pi05_libero).

        ``no_grad`` (not ``inference_mode``) is the deliberate choice so
        the output tensors keep regular version counters and can be
        round-tripped through per-session state — e.g. KV cache reused
        from stage2 → stage3 across connections, orchestrator search
        sessions holding tensor handles, etc. ``inference_mode`` would
        produce "inference tensors" that throw if later observed in any
        autograd context and would break session-state isolation.

        ``worker_idx`` selects this thread's CUDA stream so concurrent
        workers on the same stage don't serialise on the default stream.
        """
        q = self._queues[stage_id]
        stream = self._worker_streams.get((stage_id, worker_idx))
        with torch.no_grad():
            self._stage_loop_inner(stage_id, q, stream)

    def _stage3_dispatch_loop(self, q, stream=None) -> None:
        """Bucket-first stage3 scheduler (Phase-7 / external-expert direction).

        Instead of pulling a mixed batch and splitting it into serial
        ``(mode, start_t, num_steps)`` sub-buckets, accumulate requests into
        per-key buckets and dispatch each key independently when it reaches
        ``max_batch_size`` OR its oldest request exceeds ``max_wait_s``. This
        grows the actual per-denoise GPU batch (the measured avg was only 5.9
        with 52 singleton buckets), so each denoise loop is shared across more
        requests — fewer denoise loops + larger GEMMs ⇒ fewer kernel launches
        per inference (the launch-bound limiter) and better GPU efficiency.
        Correctness is unchanged: same per-request KV caches, same denoise math.
        """
        # key -> list[StageRequest]; key -> earliest enqueue monotonic time
        pending: dict[tuple, list] = {}
        pending_t0: dict[tuple, float] = {}
        while self.fatal_error is None and (not self._stop.is_set() or not q.empty() or pending):
            # Block briefly for the first arrival, then drain everything queued.
            try:
                req = q.get(timeout=0.05)
            except queue.Empty:
                req = None
            now = time.monotonic()
            if req is not None:
                items = [req]
                # Drain whatever else is immediately available.
                while True:
                    try:
                        items.append(q.get_nowait())
                    except queue.Empty:
                        break
                for r in items:
                    try:
                        k = self._batcher.bucket_key(r.payload)
                    except BaseException as exc:
                        self._fail_batch([r], exc, stream)
                        continue
                    if k not in pending:
                        pending[k] = []
                        pending_t0[k] = now
                    pending[k].append(r)
            # Decide which keys to dispatch: full bucket or deadline expired.
            ready: list[tuple] = []
            for k, reqs in pending.items():
                if self._stop.is_set() or len(reqs) >= self._max_batch_size or (now - pending_t0[k]) >= self._max_wait_s:
                    ready.append(k)
            for k in ready:
                bucket = pending.pop(k)
                pending_t0.pop(k, None)
                if self.fatal_error is not None:
                    self._fail_batch(bucket, self.fatal_error, stream)
                    continue
                self._dispatch_stage3_bucket(bucket, stream)
        if self.fatal_error is not None:
            for bucket in pending.values():
                self._fail_batch(bucket, self.fatal_error, stream)

    def _dispatch_stage3_bucket(self, bucket: list, stream=None) -> None:
        """Run one homogeneous stage3 bucket (already key-grouped), with the
        same instrumentation + stream handling as the generic loop."""
        if not bucket:
            return
        metrics_on = self._recorder is not None
        t_dispatch = time.monotonic() if metrics_on else 0.0
        try:
            if stream is not None:
                with torch.cuda.stream(stream):
                    self._run_stage3_bucket(bucket)
            else:
                self._run_stage3_bucket(bucket)
        except BaseException as exc:
            logger.exception("BatchingCoordinator stage3 bucket failed")
            self._fail_batch(bucket, exc, stream)
        else:
            if metrics_on:
                try:
                    run_ms = (time.monotonic() - t_dispatch) * 1000.0
                    waits = [(t_dispatch - r.enqueue_t) * 1000.0 for r in bucket]
                    p0 = bucket[0].payload
                    mode = "miss" if isinstance(p0, Stage3MissPayload) else "warm_start"
                    self._recorder.record_batch({
                        "stage": 3,
                        "size": len(bucket),
                        "wait_ms": max(waits),
                        "wait_avg_ms": sum(waits) / len(waits),
                        "wait_max_ms": max(waits),
                        "wait_min_ms": min(waits),
                        "run_ms": run_ms,
                        "assemble_ms": 0.0,
                        "forward_ms": run_ms,
                        "q_depth": 0,
                        "enqueue_spread_ms": 0.0,
                    })
                    self._recorder.record_batch({
                        "stage": "3bucket", "mode": mode,
                        "start_t": getattr(p0, "start_t", -1.0) if getattr(p0, "start_t", None) is not None else -1.0,
                        "num_steps": getattr(p0, "num_steps", -1),
                        "size": len(bucket), "forward_ms": run_ms,
                    })
                except Exception:
                    logger.exception(
                        "BatchingCoordinator stage3 bucket-first metrics/record failed (non-fatal)"
                    )
        # KV-cache leak guard (mirror the generic loop).
        for req in bucket:
            req.payload = None
        self._batches_since_clear += 1
        if self._batches_since_clear >= 32:
            self._batches_since_clear = 0
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


    def _stage_loop_inner(self, stage_id: int, q, stream=None) -> None:
        # stage3 bucket-first scheduler is opt-in via env. It was measured
        # neutral-to-slightly-negative on the phase5 closed-loop workload
        # (bucket size is arrival-rate-limited, not split-limited), so the
        # default is the generic pull-then-group loop which gave the best
        # observed a100 throughput (~12 inf/s). Set
        # OPENPI_STAGE3_BUCKET_FIRST=1 to re-enable.
        import os as _os_bf
        if stage_id == 3 and _os_bf.environ.get("OPENPI_STAGE3_BUCKET_FIRST", "") == "1":
            self._stage3_dispatch_loop(q, stream)
            return
        while self.fatal_error is None and (not self._stop.is_set() or not q.empty()):
            try:
                first = q.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self._max_wait_s
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(q.get(timeout=remaining))
                except queue.Empty:
                    break
            # Only compute instrumentation numbers when at least BASIC level.
            # In OFF mode we skip the time.monotonic() / qsize() calls and the
            # logger.info entirely — keeps the dispatch path free of any
            # measurement-side latency.
            metrics_on = self._recorder is not None
            t_dispatch = time.monotonic() if metrics_on else 0.0
            q_depth_snapshot = q.qsize() if metrics_on else 0
            try:
                if stream is not None:
                    # _run_batch runs its forwards on this worker's stream and
                    # calls ``torch.cuda.current_stream().synchronize()``
                    # internally right before it distributes reply tensors, so
                    # the interceptor never reads a tensor whose producing
                    # kernels are still in flight. The sync blocks only this
                    # worker thread; other workers'/stages' streams keep
                    # running, so GPU overlap is preserved.
                    with torch.cuda.stream(stream):
                        self._run_batch(stage_id, batch)
                else:
                    self._run_batch(stage_id, batch)
            except BaseException as exc:
                logger.exception("BatchingCoordinator stage %d batch failed", stage_id)
                self._fail_batch(batch, exc, stream)
            else:
                if metrics_on:
                    # Instrumentation must never kill the stage worker thread:
                    # a dead worker would hang every future submit_to_stage
                    # caller, since they wait on reply_event with no timeout.
                    try:
                        run_ms = (time.monotonic() - t_dispatch) * 1000.0
                        # Per-request wait: how long EACH request sat in the
                        # queue from its own enqueue to dispatch. wait_first_ms
                        # (the head of the batch) is the worst case; wait_last
                        # is best case; avg is the typical experience. Together
                        # with batch size this gives the wait-vs-batching
                        # tradeoff cleanly.
                        waits = [(t_dispatch - r.enqueue_t) * 1000.0 for r in batch]
                        wait_avg_ms = sum(waits) / len(waits)
                        wait_max_ms = max(waits)
                        wait_min_ms = min(waits)
                        # enqueue_spread_ms: arrival spread of all reqs in batch
                        enq_times = [r.enqueue_t for r in batch]
                        enqueue_spread_ms = (max(enq_times) - min(enq_times)) * 1000.0
                        # arrival_offsets_ms: each req's arrival time relative
                        # to first req in batch, sorted. Lets analysers see
                        # whether requests arrived in a tight burst or scattered.
                        t0 = min(enq_times)
                        arrival_offsets_ms = sorted(
                            (t - t0) * 1000.0 for t in enq_times
                        )
                        logger.info(
                            "[batch_done] stage=%d size=%d wait_avg=%.1f wait_max=%.1f run_ms=%.1f q=%d spread=%.1f",
                            stage_id, len(batch), wait_avg_ms, wait_max_ms,
                            run_ms, q_depth_snapshot, enqueue_spread_ms,
                        )
                        self._recorder.record_batch({
                            "stage": stage_id,
                            "size": len(batch),
                            # Keep wait_ms as alias for wait_first (head) for
                            # backward compat with prior summary code.
                            "wait_ms": wait_max_ms,
                            "wait_avg_ms": wait_avg_ms,
                            "wait_max_ms": wait_max_ms,
                            "wait_min_ms": wait_min_ms,
                            "run_ms": run_ms,
                            # Thread-local: _run_batch ran on THIS stage worker
                            # thread, so these are this batch's own assemble/forward
                            # split — not another concurrently-running stage's.
                            "assemble_ms": getattr(self._tls, "assemble_ms", 0.0),
                            "forward_ms": getattr(self._tls, "forward_ms", 0.0),
                            "q_depth": q_depth_snapshot,
                            "enqueue_spread_ms": enqueue_spread_ms,
                            "arrival_offsets_ms": arrival_offsets_ms,
                        })
                    except Exception:
                        logger.exception(
                            "BatchingCoordinator stage %d: batch metrics/record failed (non-fatal)",
                            stage_id,
                        )
            # KV-cache leak guard: drop per-request payload reference now that
            # the caller has read reply_slot. payload often holds Stage2Output
            # (DynamicCache with per-layer K/V tensors, ~400 MB / batch at
            # paligemma_2b + max_token_len=200) which otherwise lingers until
            # the next batch overwrites ``batch``. Without this the queue
            # backlog effectively pins multiple stage_outputs on GPU.
            for req in batch:
                req.payload = None
            batch.clear()
            # Periodic allocator cleanup hook. In the serving entrypoint
            # torch.cuda.empty_cache() is normally patched to a no-op: request
            # payload tensors are still released and cached CUDA blocks are
            # reusable internally, but the process keeps its VRAM reservation
            # from the system's perspective.
            self._batches_since_clear += 1
            if self._batches_since_clear >= 32:
                self._batches_since_clear = 0
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        _t.cuda.empty_cache()
                except Exception:
                    pass


    def _fail_batch(self, batch: list[StageRequest], exc: BaseException, stream=None) -> bool:
        """Clean the actual worker stream before replying; retain on fatal failure."""
        fatal = self.fatal_error or (exc if isinstance(exc, _FatalStageError) else None)
        if fatal is None:
            try:
                with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                    self._sync_stage_stream()
            except BaseException as sync_error:
                fatal = _FatalStageError(f"failed to synchronize stage after error: {sync_error}")
                fatal.__cause__ = sync_error
        if fatal is not None:
            batch = list(batch)
            with self._admission_lock:
                self.fatal_error = fatal
                self._stop.set()
                for q in self._queues.values():
                    while True:
                        try:
                            batch.append(q.get_nowait())
                        except queue.Empty:
                            break
                self._fatal_payloads.extend(r.payload for r in batch)
                _FATAL_PAYLOADS.extend(r.payload for r in batch)
            exc = fatal
        for req in batch:
            if not req.reply_event.is_set():
                req.error = exc
                req.reply_event.set()
        return fatal is not None

    def _publish_batch(self, batch: list[StageRequest], outputs) -> None:
        """Publish replies atomically with respect to another worker's fatal error."""
        with self._admission_lock:
            if self.fatal_error is not None:
                raise self.fatal_error
            for req, out in zip(batch, outputs):
                req.reply_slot[0] = out
                req.reply_event.set()

    def _sync_stage_stream(self) -> None:
        """Block the calling worker thread until the current CUDA stream's
        queued kernels finish. Called right before reply tensors are handed
        to the interceptor so it never reads in-flight GPU output. No-op when
        the core uses CPU. Only this
        thread blocks — other stage streams keep running (overlap preserved).
        """
        if self._device.type != "cuda" and not self._streams_enabled:
            return
        try:
            torch.cuda.current_stream(self._device).synchronize()
        except Exception as exc:
            raise _FatalStageError(f"stage CUDA synchronization failed: {exc}") from exc


    def _run_batch(self, stage_id: int, batch: list[StageRequest]) -> None:
        """Dispatch a homogeneous (stage1/2) or sub-bucketed (stage3) batch."""
        # Breakdown timers (read by _stage_loop_inner for the batch_done
        # event). assemble = CPU-side stacking; forward = model call + result
        # split + reply distribution.
        self._tls.assemble_ms = 0.0
        self._tls.forward_ms = 0.0
        _t_a = time.monotonic()
        if stage_id == 1:
            _t_f = time.monotonic()
            shards = self._batcher.run_stage1_batch([r.payload for r in batch])
            self._sync_stage_stream()
            self._publish_batch(batch, shards)
            self._tls.forward_ms = (time.monotonic() - _t_f) * 1000.0
            return

        if stage_id == 2:
            _t_f = time.monotonic()
            # Request-aware capture: one CP2 request in the batch is enough to
            # take the capture variant (same forward, prefix output retained);
            # CP1 requests in that batch ignore the extra field.
            shards = self._batcher.run_stage2_batch(
                [r.payload for r in batch],
                capture=any(r.requires_stage2_capture for r in batch),
            )
            self._sync_stage_stream()
            self._publish_batch(batch, shards)
            self._tls.forward_ms = (time.monotonic() - _t_f) * 1000.0
            return

        # stage 3 — sub-bucket by the adapter's key; each bucket gets one
        # batched forward. assemble time = bucketing; forward time = the
        # actual denoise loops (the heavy GPU part).
        buckets = self._group_stage3_requests(batch)
        self._tls.assemble_ms = (time.monotonic() - _t_a) * 1000.0
        _t_f = time.monotonic()
        # Record sub-bucket fragmentation: how the nominal stage3 batch is
        # actually split into groups that run serially. The reported stage3
        # ``avg_size`` hides this — the real per-denoise GPU batch is each
        # bucket's size, not the whole batch.
        for bucket in buckets:
            if self.fatal_error is not None:
                self._fail_batch(bucket, self.fatal_error)
                continue
            _bt = time.monotonic()
            try:
                self._run_stage3_bucket(bucket)
            except BaseException as exc:  # noqa: BLE001 - per-bucket fault isolation
                # A failing bucket fails only its own requests; the other
                # buckets of this batch still run (plan §6.1). The stream is
                # synchronised first so no kernel of the failed forward can
                # still be reading the inputs when the caller releases them.
                logger.exception("BatchingCoordinator stage3 bucket failed")
                self._fail_batch(bucket, exc)
                continue
            if self._recorder is not None and bucket:
                try:
                    p0 = bucket[0].payload
                    mode = "miss" if isinstance(p0, Stage3MissPayload) else "warm_start"
                    start_t = getattr(p0, "start_t", None)
                    self._recorder.record_batch({
                        "stage": "3bucket",
                        "mode": mode,
                        "start_t": start_t if start_t is not None else -1.0,
                        "num_steps": getattr(p0, "num_steps", -1),
                        "size": len(bucket),
                        "forward_ms": (time.monotonic() - _bt) * 1000.0,
                    })
                except Exception:
                    logger.exception(
                        "BatchingCoordinator stage3 sub-bucket metrics/record failed (non-fatal)"
                    )
        self._tls.forward_ms = (time.monotonic() - _t_f) * 1000.0

    def _group_stage3_requests(
        self,
        reqs: list[StageRequest],
    ) -> list[list[StageRequest]]:
        groups: dict[Hashable, list[StageRequest]] = collections.defaultdict(list)
        for req in reqs:
            p = req.payload
            if not isinstance(p, (Stage3MissPayload, Stage3WarmStartPayload)):
                req.error = TypeError(
                    f"Unknown Stage3 payload type: {type(p).__name__}"
                )
                req.reply_event.set()
                continue
            groups[self._batcher.bucket_key(p)].append(req)
        return list(groups.values())

    def _wait_ready(self, bucket: list[StageRequest]) -> None:
        """Make this worker's stream wait for every producer event of the bucket."""
        events = [
            ev for r in bucket for ev in (getattr(r.payload, "ready_events", ()) or ())
        ]
        if not events:
            return
        stream = torch.cuda.current_stream(self._device)
        for ev in events:
            stream.wait_event(ev)

    def _run_stage3_bucket(self, bucket: list[StageRequest]) -> None:
        """Run one homogeneous Stage 3 sub-batch."""
        if not bucket:
            return
        self._wait_ready(bucket)
        p0 = bucket[0].payload
        payloads = [r.payload for r in bucket]
        if isinstance(p0, Stage3MissPayload):
            outs = self._batcher.run_stage3_miss(
                payloads,
                num_steps=p0.num_steps,
                save_timesteps_per_request=[p.save_timesteps for p in payloads],
            )
        else:  # Stage3WarmStartPayload
            outs = self._batcher.run_stage3_warm(
                payloads,
                start_t=p0.start_t,
                num_steps=p0.num_steps,
                capture_first_step=any(p.capture_first_step for p in payloads),
            )
        self._sync_stage_stream()
        self._publish_batch(bucket, outs)
