"""M1 BatchingCoordinator — stage-level dynamic batching for concurrent serving.

Concurrent-mode requests submit per-request payloads to one of three stage
queues; a background worker per stage pulls requests, fills a dynamic batch
(``max_batch_size`` or ``max_wait_ms`` deadline), runs one batched forward,
and distributes results back to the submitting threads via reply events.

Why three queues, not one
-------------------------
Each request transitions through three stages (stage1 → stage2 → stage3),
and stages have different compute characteristics:
  * Stage 1 — SigLIP vision + Gemma embedding (vision-heavy)
  * Stage 2 — Gemma LLM backbone forward (LLM-heavy, fills KV cache)
  * Stage 3 — Action expert flow-matching (action-head heavy, num_steps Euler loops)
A request only reaches stage N+1 after CPU-side cache work (CP1/CP3 check,
key building, history append) on the per-request thread. Per-stage queues
let each stage build independent batches without cross-stage coupling.

Stage 3 sub-bucketing (plan §4.1.3 / G1 R2 Item 3)
--------------------------------------------------
``run_stage3`` (MISS, with optional intermediates) and ``run_stage3_from``
(WARM_START, no noise param) have incompatible signatures, and WARM_START
samples may carry different ``start_t`` / ``num_steps``. Stage 3 collects
requests, groups them by ``(mode, start_t, num_steps)``, and issues one
batched forward per bucket.

Sub-batch split for CP1 FULL_HIT (plan §4.1 A.2)
------------------------------------------------
FULL_HIT after CP1 short-circuits to the cached action — those requests
never enter stage2/stage3 barriers. The Interceptor handles the early
return on its OS thread; the coordinator only sees the remaining requests.

Hard constraint C1 — non-concurrent path bypass
-----------------------------------------------
The coordinator is only instantiated in concurrent mode by the server entry
point. ``Policy.infer`` and ``InferenceInterceptor.infer`` on the
non-concurrent code path retain their direct stage1/2/3 calls + post-stage3
CP3 (the existing line 628-635 timing-probe block) — see ``C1`` in
``logs/concurrent_serving_optimization_plan.log.md`` §2.1.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch

from openpi.models_pytorch.pi0_pytorch import (
    Stage1Output,
    Stage2Output,
    Stage3Output,
)
from openpi.serving import stage_io

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Stage 3 payloads (sum type — G1 R2 Item 3: no shared `noise` field)
# ------------------------------------------------------------------


@dataclass
class Stage3MissPayload:
    """MISS path: ``run_stage3(stage2, noise, return_intermediates=True)``.

    The Interceptor builds this when CP1 returns MISS, sampling a fresh noise
    on the per-request thread (matches the legacy single-request semantics).
    """
    stage2_out: Stage2Output
    noise: torch.Tensor      # [action_horizon, action_dim] — unbatched
    num_steps: int = 10


@dataclass
class Stage3WarmStartPayload:
    """WARM_START path: ``run_stage3_from(stage2, start_x, start_t, num_steps=...)``.

    No ``noise`` field — WARM_START continues from the cached ``start_x``;
    the noise that produced ``start_x`` was consumed when the original MISS
    cache entry was created.
    """
    stage2_out: Stage2Output
    start_x: torch.Tensor    # [action_horizon, action_dim] — unbatched
    start_t: float
    num_steps: int


Stage3InitPayload = Union[Stage3MissPayload, Stage3WarmStartPayload]


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
    reply_slot: list = None  # list-of-len-1 used as a mutable slot; None until worker writes
    error: Optional[BaseException] = None


# ------------------------------------------------------------------
# Coordinator
# ------------------------------------------------------------------


class BatchingCoordinator:
    """Three-stage dynamic batching worker pool.

    Lifecycle: ``start()`` spawns three daemon threads (one per stage),
    ``stop()`` signals them to drain and exit. Use as a context manager when
    convenient (``with BatchingCoordinator(...) as bc: ...``).

    Parameters
    ----------
    model:
        The PI0Pytorch base model. Must support ``run_stage1`` / ``run_stage2``
        / ``run_stage3`` / ``run_stage3_from``.
    device:
        Device on which observations are stacked. Should match ``model``'s
        device; auto-detected from ``model`` parameter device when None.
    max_batch_size:
        Max requests merged into one forward. Default 8 (plan §4.1 A.1).
    max_wait_ms:
        Max time the worker waits for additional requests before issuing a
        partial batch. Default 10ms.
    """

    def __init__(
        self,
        model,
        *,
        device: Optional[str | torch.device] = None,
        max_batch_size: int = 8,
        max_wait_ms: float = 10.0,
    ) -> None:
        self._model = model
        self._device = torch.device(device) if device is not None else next(model.parameters()).device
        self._max_batch_size = int(max_batch_size)
        self._max_wait_s = float(max_wait_ms) / 1000.0

        self._queues: dict[int, queue.Queue[StageRequest]] = {
            1: queue.Queue(), 2: queue.Queue(), 3: queue.Queue(),
        }
        self._stop = threading.Event()
        self._stage_threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn one daemon worker thread per stage."""
        if self._stage_threads:
            raise RuntimeError("BatchingCoordinator.start: already started")
        for stage_id in (1, 2, 3):
            t = threading.Thread(
                target=self._stage_loop, args=(stage_id,),
                name=f"BatchingCoordinator-stage{stage_id}", daemon=True,
            )
            self._stage_threads.append(t)
            t.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal the workers to drain and exit; join with ``timeout`` each."""
        self._stop.set()
        for t in self._stage_threads:
            t.join(timeout=timeout)
        self._stage_threads.clear()

    def __enter__(self) -> "BatchingCoordinator":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Submit (blocking)
    # ------------------------------------------------------------------

    def submit_to_stage(
        self,
        stage_id: Literal[1, 2, 3],
        bundle_id: str,
        payload: Any,
        *,
        request_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Stage1Output | Stage2Output | Stage3Output:
        """Submit a request to ``stage_id``; block until the worker fills in
        the per-request output.

        Raises whatever exception the worker raised on the batched forward
        (per-request, not per-batch — every request in the same batch sees
        the same exception, which is acceptable for an L3 architectural
        layer error).
        """
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
        )
        self._queues[stage_id].put(req)
        # Block until the worker signals completion.
        if not req.reply_event.wait(timeout=timeout):
            raise TimeoutError(
                f"BatchingCoordinator.submit_to_stage(stage_id={stage_id}) timed out"
            )
        if req.error is not None:
            raise req.error
        return req.reply_slot[0]

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _stage_loop(self, stage_id: int) -> None:
        """Pull a batch off the queue, run it, distribute outputs."""
        q = self._queues[stage_id]
        while not self._stop.is_set():
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
            try:
                self._run_batch(stage_id, batch)
            except BaseException as exc:  # noqa: BLE001 — surface to every submitter
                logger.exception("BatchingCoordinator stage %d batch failed", stage_id)
                for req in batch:
                    req.error = exc
                    req.reply_event.set()

    def _run_batch(self, stage_id: int, batch: list[StageRequest]) -> None:
        """Dispatch a homogeneous (stage1/2) or sub-bucketed (stage3) batch."""
        n = len(batch)
        if stage_id == 1:
            obs_list = [r.payload for r in batch]
            batched = stage_io.stack_observation(obs_list, device=self._device)
            # Coordinator path receives unbatched dicts from the interceptor
            # (G2 R2 Item 1) — wrap into ``Observation`` here once, after the
            # single stack pass so the resulting Observation has shape
            # ``[N, ...]``, not ``[N, 1, ...]``. Detection is duck-typed:
            # only dicts that look like full Observation payloads (have an
            # ``image`` key, per ``Observation.from_dict``) are wrapped;
            # other dicts (e.g. test stubs) pass through so existing tests
            # keep working.
            if isinstance(batched, dict) and "image" in batched:
                from openpi.models import model as _model
                batched_obs = _model.Observation.from_dict(batched)
            else:
                batched_obs = batched
            stage1_batched = self._model.run_stage1(batched_obs)
            shards = stage_io.split_stage1_output(stage1_batched, n)
            for req, out in zip(batch, shards):
                req.reply_slot[0] = out
                req.reply_event.set()
            return

        if stage_id == 2:
            stage1_batched = stage_io.stack_stage1_output([r.payload for r in batch])
            stage2_batched = self._model.run_stage2(stage1_batched)
            shards = stage_io.split_stage2_output(stage2_batched, n)
            for req, out in zip(batch, shards):
                req.reply_slot[0] = out
                req.reply_event.set()
            return

        # stage 3 — sub-bucket by (mode, start_t, num_steps); each bucket
        # gets one batched forward.
        buckets = self._group_stage3_requests(batch)
        for bucket in buckets:
            self._run_stage3_bucket(bucket)

    @staticmethod
    def _group_stage3_requests(
        reqs: list[StageRequest],
    ) -> list[list[StageRequest]]:
        groups: dict[tuple, list[StageRequest]] = collections.defaultdict(list)
        for req in reqs:
            p = req.payload
            if isinstance(p, Stage3MissPayload):
                key = ("miss", None, p.num_steps)
            elif isinstance(p, Stage3WarmStartPayload):
                key = ("warm_start", p.start_t, p.num_steps)
            else:
                req.error = TypeError(
                    f"Unknown Stage3 payload type: {type(p).__name__}"
                )
                req.reply_event.set()
                continue
            groups[key].append(req)
        return list(groups.values())

    def _run_stage3_bucket(self, bucket: list[StageRequest]) -> None:
        """Run one homogeneous Stage 3 sub-batch."""
        if not bucket:
            return
        p0 = bucket[0].payload
        stage2_batched = stage_io.stack_stage2_output(
            [r.payload.stage2_out for r in bucket]
        )

        if isinstance(p0, Stage3MissPayload):
            noise_batched = torch.stack(
                [r.payload.noise for r in bucket], dim=0,
            ).to(self._device)
            out = self._model.run_stage3(
                stage2_batched,
                noise=noise_batched,
                num_steps=p0.num_steps,
                return_intermediates=True,
            )
            inter_shards = stage_io.split_stage3_intermediates(
                out.intermediates, len(bucket),
            )
            for i, req in enumerate(bucket):
                req.reply_slot[0] = Stage3Output(
                    action_chunk=out.action_chunk[i:i + 1],
                    intermediates=inter_shards[i],
                )
                req.reply_event.set()
        else:  # Stage3WarmStartPayload
            start_x_batched = torch.stack(
                [r.payload.start_x for r in bucket], dim=0,
            ).to(self._device)
            out = self._model.run_stage3_from(
                stage2_batched,
                start_x=start_x_batched,
                start_t=p0.start_t,
                num_steps=p0.num_steps,
            )
            for i, req in enumerate(bucket):
                req.reply_slot[0] = Stage3Output(
                    action_chunk=out.action_chunk[i:i + 1],
                    intermediates=None,
                )
                req.reply_event.set()
