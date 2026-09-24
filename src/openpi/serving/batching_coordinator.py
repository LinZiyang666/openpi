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

Structure (trace plan §6.1)
---------------------------
The scheduling machinery lives in the jax-free
``openpi.serving.batching_core.BatchingCore``; this module is the Pi0.5
adapter: ``Pi05StageBatcher`` owns the ``stage_io`` stacking / splitting and
the model calls, and ``BatchingCoordinator`` is the core wired to that
adapter with the historical constructor signature. The public names
(``Stage3MissPayload``, ``Stage3WarmStartPayload``, ``StageRequest``,
``get_active_coordinator``) are re-exported from the core.
"""

from __future__ import annotations

import logging
from typing import Any, Hashable

import torch

from openpi.serving import batching_core as _core
from openpi.serving import stage_io
from openpi.serving.batching_core import (  # noqa: F401 - public re-exports
    _DEFAULT_SUBMIT_TIMEOUT_S,
    BatchingCore,
    Stage3InitPayload,
    Stage3MissPayload,
    Stage3VariantOutput,
    Stage3WarmStartPayload,
    StageBatcher,
    StageRequest,
    get_active_coordinator,
    record_ready_events,
)

logger = logging.getLogger(__name__)

# The model's default ``run_stage3(save_timesteps=...)`` set. A MISS bucket
# whose requests all leave ``save_timesteps=None`` calls the model exactly as
# before (no kwarg); otherwise the union of the requested sets is computed
# once and each request gets its own subset back. Guarded by a signature test.
_MODEL_DEFAULT_SAVE_TIMESTEPS: tuple[float, ...] = (0.7, 0.5, 0.3)


class Pi05StageBatcher:
    """``StageBatcher`` for ``PI0Pytorch`` (the historical coordinator body)."""

    def __init__(self, model, device: torch.device) -> None:
        self._model = model
        self._device = torch.device(device)

    # -- bucket key -------------------------------------------------------

    @staticmethod
    def bucket_key(payload: Any) -> Hashable:
        if isinstance(payload, Stage3MissPayload):
            return ("miss", None, payload.num_steps)
        if isinstance(payload, Stage3WarmStartPayload):
            return ("warm_start", payload.start_t, payload.num_steps)
        return ("unknown", None, None)

    @staticmethod
    def default_save_timesteps() -> tuple[float, ...]:
        return _MODEL_DEFAULT_SAVE_TIMESTEPS

    @staticmethod
    def order_timesteps(values) -> tuple[float, ...]:
        # Pi0.5 loops in descending flow time.
        return tuple(sorted(set(float(v) for v in values), reverse=True))

    # -- stage 1 / 2 ---------------------------------------------------------

    def run_stage1_batch(self, payloads: list) -> list:
        batched = stage_io.stack_observation(payloads, device=self._device)
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
        return stage_io.split_stage1_output(stage1_batched, len(payloads))

    def run_stage2_batch(self, payloads: list, *, capture: bool) -> list:
        stage1_batched = stage_io.stack_stage1_output(payloads)
        if capture:
            stage2_batched = self._model.run_stage2_capture(stage1_batched)
        else:
            stage2_batched = self._model.run_stage2(stage1_batched)
        return stage_io.split_stage2_output(stage2_batched, len(payloads))

    # -- stage 3 ---------------------------------------------------------------

    def run_stage3_miss(
        self,
        payloads: list,
        *,
        num_steps: int,
        save_timesteps_per_request: list,
    ) -> list:
        stage2_batched = stage_io.stack_stage2_output([p.stage2_out for p in payloads])
        noise_batched = torch.stack([p.noise for p in payloads], dim=0).to(self._device)
        requested = [s for s in save_timesteps_per_request if s is not None]
        if not requested:
            # Legacy call shape: no ``save_timesteps`` kwarg at all.
            out = self._model.run_stage3(
                stage2_batched,
                noise=noise_batched,
                num_steps=num_steps,
                return_intermediates=True,
            )
            inter_shards = stage_io.split_stage3_intermediates(out.intermediates, len(payloads))
            return [
                _Stage3Reply(action_chunk=out.action_chunk[i:i + 1], intermediates=inter_shards[i])
                for i in range(len(payloads))
            ]
        union = self.order_timesteps(set().union(*[
            set(_MODEL_DEFAULT_SAVE_TIMESTEPS if s is None else s)
            for s in save_timesteps_per_request
        ]))
        out = self._model.run_stage3(
            stage2_batched,
            noise=noise_batched,
            num_steps=num_steps,
            return_intermediates=True,
            save_timesteps=union,
        )
        inter_shards = stage_io.split_stage3_intermediates(out.intermediates, len(payloads))
        replies = []
        for i, wanted in enumerate(save_timesteps_per_request):
            keep = set(wanted) if wanted is not None else set(_MODEL_DEFAULT_SAVE_TIMESTEPS)
            shard = {t: v for t, v in (inter_shards[i] or {}).items() if t in keep}
            replies.append(_Stage3Reply(action_chunk=out.action_chunk[i:i + 1], intermediates=shard))
        return replies

    def run_stage3_warm(
        self,
        payloads: list,
        *,
        start_t: float,
        num_steps: int,
        capture_first_step: bool,
    ) -> list:
        del capture_first_step  # Pi0.5 warm starts expose no first-step capture
        stage2_batched = stage_io.stack_stage2_output([p.stage2_out for p in payloads])
        start_x_batched = torch.stack([p.start_x for p in payloads], dim=0).to(self._device)
        out = self._model.run_stage3_from(
            stage2_batched,
            start_x=start_x_batched,
            start_t=start_t,
            num_steps=num_steps,
        )
        return [
            _Stage3Reply(action_chunk=out.action_chunk[i:i + 1], intermediates=None)
            for i in range(len(payloads))
        ]


def _Stage3Reply(*, action_chunk, intermediates):
    """The Stage-3 reply type legacy callers expect (``Stage3Output``)."""
    from openpi.models_pytorch.pi0_pytorch import Stage3Output

    return Stage3Output(action_chunk=action_chunk, intermediates=intermediates)


class BatchingCoordinator(BatchingCore):
    """Three-stage dynamic batching worker pool for ``PI0Pytorch``.

    Parameters
    ----------
    model:
        The PI0Pytorch base model. Must support ``run_stage1`` / ``run_stage2``
        / ``run_stage3`` / ``run_stage3_from``.
    device:
        Device on which observations are stacked. Should match ``model``'s
        device; auto-detected from ``model`` parameter device when None.
    max_batch_size:
        Max requests merged into one forward. Default 32.
    max_wait_ms:
        Max time the worker waits for additional requests before issuing a
        partial batch. Default 25ms.
    """

    def __init__(
        self,
        model,
        *,
        device: str | torch.device | None = None,
        max_batch_size: int = 32,
        max_wait_ms: float = 25.0,
    ) -> None:
        resolved = torch.device(device) if device is not None else next(model.parameters()).device
        self._model = model
        super().__init__(
            Pi05StageBatcher(model, resolved),
            device=resolved,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
        )

    def _default_submit_timeout(self) -> float:
        # Read this module's constant at call time so the historical
        # ``patch.object(batching_coordinator, "_DEFAULT_SUBMIT_TIMEOUT_S", ...)``
        # keeps working.
        return _DEFAULT_SUBMIT_TIMEOUT_S


__all__ = [
    "BatchingCoordinator",
    "BatchingCore",
    "Pi05StageBatcher",
    "Stage3InitPayload",
    "Stage3MissPayload",
    "Stage3VariantOutput",
    "Stage3WarmStartPayload",
    "StageBatcher",
    "StageRequest",
    "get_active_coordinator",
    "record_ready_events",
]
