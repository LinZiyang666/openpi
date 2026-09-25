"""GR00T N1.5 warm reset executor: guarded continuation / self-start with measured steps.

Plan ``logs/warm_continuation_first_class_plan.log.md`` §4.3.3 / §4.4 / §4.6.
``run_groot_continuation`` and ``groot_self_start`` stand beside
``GrootStagedRunner.run_stage3_from`` and carry that entry's runtime guards
themselves, all before any head call: the autocast session
(``runner._require_session``), the library schedule equal to the live head's
(read on every call), the plan equal to the library schedule, and a
recoverable ``start_t``. ``openpi.cache.groot.staged`` is not modified (its
header pins the G0-C equivalence gate).

Bit-for-bit semantics (B = 1) are those of ``exp/step_diag/groot.py``: the
continuation dispatches on the grid to the same three loops -- upstream's own
``denoise_loop`` for a reset to pure noise, ``grid_denoise_loop`` (the
step_diag mid / shoot loop, one statement sequence) for a reset below noise
and for a shoot -- and the self start is the runner's transcribed K-step loop
from private noise. Step counts are measured: a call-local counter around
``staged.denoise_step`` for the continuation, the loop's ``on_step`` callback
count for the self start. There is no batching coordinator on the non-trace
GR00T path (plan §3.4-F1); connections are serialised by the entry point's
infer lock.

Public interface: ``grid_denoise_loop``, ``run_groot_continuation``,
``groot_self_start``, ``GrootWarmResetExecutor``, ``build_groot_warm_reset``.
Depends on torch, ``openpi.cache.groot.staged`` and the model-agnostic
``warm_reset`` modules; jax-free (imported by the GR00T island).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from openpi.cache.groot import staged as _staged
from openpi.cache.groot.staged import GrootStage3Output
from openpi.cache.types import DIRECTION_ASC, DenoiseSchedule
from openpi.cache.warm_reset.evidence import GrootWarmResetEvidencePolicy
from openpi.cache.warm_reset.runtime import (
    WarmResetParts,
    WarmResetSession,
    _build,
    _CountedStep,
    decision_meta,
)
from openpi.cache.warm_reset.types import (
    POINT_FINAL,
    SelfStartPlan,
    WarmResetPlan,
    WarmResetSpec,
    groot_loop_grid,
    native_grid,
    private_noise,
    resolve_plan,
    resolve_self_plan,
    validate_plan,
    validate_self_plan,
)


def _loop_steps(plan: WarmResetPlan) -> int:
    """Loop length handed to the continuation loop (the fault-injection seam of the tests)."""
    return plan.n_steps


def grid_denoise_loop(
    action_head: Any,
    backbone_output: Any,
    action_input: Any,
    *,
    start: torch.Tensor,
    tau0: float,
    dt: float,
    num_steps: int,
    step_fn: Optional[Callable[..., torch.Tensor]] = None,
) -> torch.Tensor:
    """Upstream's ascending loop on the grid ``tau_i = tau0 + i * dt`` (GR00T native time).

    Same preprocessing, bucket discretisation and ``step_fn`` as
    ``staged.denoise_loop``, with a free start and step: the step_diag
    ``mid_denoise_loop`` / ``shoot_denoise_loop`` statement sequence.
    """
    if not 0.0 <= tau0 < 1.0 or dt <= 0.0 or num_steps < 1:
        raise ValueError(f"tau0={tau0} / dt={dt} / num_steps={num_steps} out of range")
    step_fn = _staged.denoise_step if step_fn is None else step_fn
    processed = action_head.process_backbone_output(backbone_output)
    vl = processed.backbone_features
    embodiment_id = action_input["embodiment_id"]
    state_features = action_head.state_encoder(action_input["state"], embodiment_id)
    batch_size = vl.shape[0]
    actions = start.to(device=vl.device, dtype=vl.dtype)
    for i in range(num_steps):
        t_cont = tau0 + i * dt
        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=int(t_cont * action_head.num_timestep_buckets), device=vl.device
        )
        actions = step_fn(action_head, vl, state_features, embodiment_id, actions, timesteps_tensor, dt).clone()
    return actions


def _require_live(runner: Any, schedule: DenoiseSchedule, *, where: str, start_t: float) -> None:
    """The library schedule must be the loop the head runs right now (read per call)."""
    live = runner.live_schedule()
    if schedule != live:
        raise RuntimeError(
            f"{where}: library schedule {schedule.schedule_id} but the action head is "
            f"running {live.schedule_id}; a snapshot at t={start_t} would be resumed at "
            "the wrong step."
        )
    if schedule.direction != DIRECTION_ASC:
        raise ValueError(f"{where}: {schedule.schedule_id} is not a GR00T (ascending) schedule")


def _check_batch(stage2: Any, x: torch.Tensor, *, where: str) -> torch.Tensor:
    """Normalize a unit-batch start and reject inconsistent inputs before any head call."""
    if x.dim() == 2:
        x = x[None, ...]
    batch = stage2.backbone_features.shape[0]
    if (
        batch < 1 or x.dim() != 3 or x.shape[0] != batch
        or stage2.attention_mask.shape[0] != batch
        or any(stage2.action_inputs[key].shape[0] != batch for key in ("state", "embodiment_id"))
    ):
        raise ValueError(f"{where}: start/noise {tuple(x.shape)} does not match the stage-2 batch {batch}")
    return x


def run_groot_continuation(
    runner: Any, stage2: Any, start_x: torch.Tensor, plan: WarmResetPlan, *, schedule: DenoiseSchedule
) -> GrootStage3Output:
    """Continue ``start_x`` on ``plan``'s grid; guards 1-4 run before any head call.

    Must be called inside ``runner.session()``. ``steps_run`` is the measured
    number of ``denoise_step`` calls.
    """
    runner._require_session("run_groot_continuation")  # noqa: SLF001 - the runner's own guard
    _require_live(runner, schedule, where="run_groot_continuation", start_t=plan.start_t)
    validate_plan(plan, schedule)
    start_x = _check_batch(stage2, start_x, where="run_groot_continuation")
    head = runner._model.action_head  # noqa: SLF001 - same seam as the runner's own resume
    backbone_outputs = runner._head_inputs(stage2)  # noqa: SLF001
    step = _CountedStep(_staged.denoise_step)
    tau0, dt = groot_loop_grid(plan)
    with runner._timer.measure("stage3_warm"):  # noqa: SLF001
        if tau0 is None:
            action_pred = _staged.denoise_loop(
                head, backbone_outputs, stage2.action_inputs, noise=start_x,
                num_steps=_loop_steps(plan), start_index=0, step_fn=step,
            )
        else:
            action_pred = grid_denoise_loop(
                head, backbone_outputs, stage2.action_inputs, start=start_x,
                tau0=tau0, dt=dt, num_steps=_loop_steps(plan), step_fn=step,
            )
    runner._model.validate_data(  # noqa: SLF001
        _staged._batch_feature({"action_pred": action_pred}), backbone_outputs, is_training=False  # noqa: SLF001
    )
    return GrootStage3Output(action_pred=action_pred, start_t=plan.start_t, steps_run=step.calls)


def groot_self_start(
    runner: Any, stage2: Any, noise: torch.Tensor, plan: SelfStartPlan, *, schedule: DenoiseSchedule
) -> tuple[torch.Tensor, int]:
    """The direct K-step inference from ``noise``: ``(snapshot or final action, measured steps)``.

    Guards 1-4 and the capture mapping are checked before the loop; the loop is
    ``runner.run_stage3(noise=, on_step=)`` (upstream's transcription). The
    observer takes ``x_in`` of step ``capture_index`` and counts every step; a
    snapshot request that the loop never reached raises.
    """
    runner._require_session("groot_self_start")  # noqa: SLF001 - the runner's own guard
    _require_live(runner, schedule, where="groot_self_start", start_t=plan.start_t)
    validate_self_plan(plan, schedule)
    noise = _check_batch(stage2, noise, where="groot_self_start")
    captured: dict[str, torch.Tensor] = {}
    calls = [0]

    def observer(step: int, x_in: torch.Tensor, x_out: torch.Tensor) -> None:
        calls[0] += 1
        if plan.capture_index is not None and step == plan.capture_index:
            captured["snapshot"] = x_in.detach().clone()

    out = runner.run_stage3(stage2, noise=noise, on_step=observer)
    if plan.capture_index is None:
        return out.action_pred, calls[0]
    if "snapshot" not in captured:
        raise RuntimeError(
            f"self start: the loop never reached step {plan.capture_index} (start_t={plan.start_t})"
        )
    return captured["snapshot"], calls[0]


# ------------------------------------------------------------------
# Executor
# ------------------------------------------------------------------


class GrootWarmResetExecutor:
    """Runs the WARM_START continuation of ``GrootCacheInterceptor`` under a warm reset spec.

    Called inside the interceptor's ``runner.session()`` with the library
    schedule ``_library_schedule`` resolved; the entries re-check it against
    the live head themselves.
    """

    def __init__(self, spec: WarmResetSpec, session: WarmResetSession) -> None:
        self.spec = spec
        self._session = session
        self._digest = spec.digest()

    def _self_start(self, runner: Any, stage2: Any, noise: torch.Tensor, plan: SelfStartPlan, schedule: DenoiseSchedule) -> torch.Tensor:
        self._session.count_self_start_call()
        x, steps = groot_self_start(runner, stage2, noise, plan, schedule=schedule)
        self._session.add_self_start_steps(steps)
        return x

    def _continue(self, runner: Any, stage2: Any, start_x: torch.Tensor, plan: WarmResetPlan, schedule: DenoiseSchedule) -> GrootStage3Output:
        self._session.count_continuation_call()
        out = run_groot_continuation(runner, stage2, start_x, plan, schedule=schedule)
        self._session.add_continuation_steps(out.steps_run)
        return out

    def run(self, *, runner: Any, stage2: Any, cp_result: Any, schedule: DenoiseSchedule) -> tuple[GrootStage3Output, dict]:
        """Resolve the plan, produce the start, run the continuation; ``(stage3, hit meta)``.

        The start is shaped like the payload's snapshot at the verdict's
        ``start_t`` (``[H, D]`` host tensor), as the step_diag reference does.
        """
        session = self._session
        if session.decision_idx is None:
            raise RuntimeError("warm reset: no open decision (the evidence wrapper is not installed)")
        payload = cp_result.payload
        start_t = cp_result.start_t
        plan = resolve_plan(self.spec, schedule, start_t)
        like = payload.intermediates[start_t]
        seed = None
        if self.spec.self_start:
            self_plan = resolve_self_plan(self.spec, schedule, start_t)
            if (self_plan.schedule_id, self_plan.k) != (plan.schedule_id, plan.k):
                raise ValueError("warm reset: self-start and continuation plans disagree on the schedule")
            seed = session.self_seed()
            noise = private_noise(seed, tuple(like.shape[-2:]))[None, ...].to(device=like.device)
            x = self._self_start(runner, stage2, noise, self_plan, schedule)
            start_x = x.to(device=like.device, dtype=like.dtype).reshape(like.shape)
        elif self.spec.point == POINT_FINAL:
            start_x = payload.action_chunk.to(device=like.device, dtype=like.dtype).reshape(like.shape)
        else:
            start_x = like
        out = self._continue(runner, stage2, start_x, plan, schedule)
        taus, dt = native_grid(plan)
        buckets = runner._model.action_head.num_timestep_buckets  # noqa: SLF001
        extra = {"tau": list(taus), "bucket": [int(tau * buckets) for tau in taus]}
        meta = decision_meta(
            self.spec, self._digest, plan, session, t=plan.flow_times(), dt=dt, self_seed=seed, extra=extra
        )
        return out, meta


def build_groot_warm_reset(config: Any, *, bundle_id: str, yaml_id: Optional[str], yaml_path: Optional[str]) -> Optional[WarmResetParts]:
    """Per-connection executor + evidence wrapper for the GR00T entry points; ``None`` without a block."""
    return _build(
        config,
        family="groot",
        bundle_id=bundle_id,
        yaml_id=yaml_id,
        yaml_path=yaml_path,
        executor_factory=GrootWarmResetExecutor,
        wrapper_factory=GrootWarmResetEvidencePolicy,
    )


__all__ = [
    "GrootWarmResetExecutor",
    "build_groot_warm_reset",
    "grid_denoise_loop",
    "groot_self_start",
    "run_groot_continuation",
]
