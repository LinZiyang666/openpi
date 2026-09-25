"""Pi0.5 warm reset executor: continuation and self-start loops with measured steps.

Plan ``logs/warm_continuation_first_class_plan.log.md`` §4.3.2 / §4.4 / §4.5.
``run_pi05_continuation`` and ``run_pi05_self_start`` are the stage-3 entries
that stand beside ``PI0Pytorch.run_stage3_from`` / ``run_stage3``; the
interceptor reaches them through ``Pi05WarmResetExecutor`` either directly
(``direct_runner``) or through the batching coordinator
(``Stage3WarmResetPayload`` -> ``Pi05StageBatcher.run_stage3_warm_reset``).

Bit-for-bit semantics (B = 1) are those of ``exp/step_diag/pi05.py``: the
continuation builds ``timestep`` / ``dt`` with the same float32 expressions
(a shoot replays the full loop's accumulation to the start like
``run_stage3_from``), and the self start repeats ``_stage3_with_intermediates``
statement for statement. Every Euler step goes through a call-local counter
around ``model.denoise_step``; the returned ``steps_run`` is that counter,
never the budget, and nothing on the model instance is replaced.

Public interface: ``WarmResetStage3Output``, ``Pi05SelfStartOutput``,
``run_pi05_continuation``, ``run_pi05_self_start``,
``Pi05WarmResetExecutor``, ``build_pi05_warm_reset``.
Depends on ``openpi.models_pytorch.pi0_pytorch`` (Pi0.5 only; the GR00T island
never imports this module) and the model-agnostic ``warm_reset`` modules.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Sequence

import torch

from openpi.cache.types import PI05_V1
from openpi.cache.warm_reset.evidence import WarmResetEvidencePolicy
from openpi.cache.warm_reset.runtime import (
    WarmResetParts,
    WarmResetSession,
    _build,
    _CountedStep,
    decision_meta,
)
from openpi.cache.warm_reset.types import (
    KIND_RESET,
    POINT_FINAL,
    POINT_SNAPSHOT,
    SelfStartPlan,
    WarmResetPlan,
    WarmResetSpec,
    private_noise,
    resolve_plan,
    resolve_self_plan,
    validate_plan,
    validate_self_plan,
)
from openpi.models_pytorch.pi0_pytorch import Stage3Output, _warm_start_num_steps


@dataclasses.dataclass
class WarmResetStage3Output(Stage3Output):
    """A continuation's ``Stage3Output`` plus the measured Euler step count."""

    steps_run: int = 0


@dataclasses.dataclass
class Pi05SelfStartOutput:
    """One row of a self-start run: final action, requested snapshot, measured steps.

    ``action_chunk`` / ``snapshot`` keep the unit batch axis (``[1, H, D]``);
    ``snapshot`` is ``None`` for a final-action capture.
    """

    action_chunk: torch.Tensor
    snapshot: Optional[torch.Tensor]
    steps_run: int


def _loop_steps(plan: WarmResetPlan) -> int:
    """Euler iterations of a continuation (the fault-injection seam of the tests)."""
    return plan.n_steps


def _self_loop_steps(plan: SelfStartPlan) -> int:
    """Euler iterations of a self start (the fault-injection seam of the tests)."""
    return plan.k


# ------------------------------------------------------------------
# Stage-3 entries
# ------------------------------------------------------------------


def run_pi05_continuation(model: Any, stage2: Any, start_x: torch.Tensor, plan: WarmResetPlan) -> WarmResetStage3Output:
    """Continue ``start_x`` ([B, H, D]) on ``plan``'s grid with one plan for the whole batch.

    Refuses (``ValueError``) before the first ``denoise_step`` any plan that is
    not a valid ``PI05_V1`` continuation. reset: ``t0 = level``,
    ``dt = -level / N``; shoot: ``t0`` replays ``K - n(start_t)`` grid steps
    from 1.0 in float32, ``dt = -level / N``.
    """
    validate_plan(plan, PI05_V1)
    stage1 = stage2.stage1
    device = stage1.state.device
    bsize = stage1.state.shape[0]
    if start_x.dim() != 3 or bsize < 1 or start_x.shape[0] != bsize:
        raise ValueError(
            f"run_pi05_continuation: start {tuple(start_x.shape)} does not match the stage-2 batch {bsize}"
        )
    dt = torch.tensor(-plan.level / plan.n_steps, dtype=torch.float32, device=device)
    if plan.kind == KIND_RESET:
        timestep = torch.tensor(plan.level, dtype=torch.float32, device=device)
    else:
        grid = torch.tensor(-1.0 / plan.k, dtype=torch.float32, device=device)
        timestep = torch.tensor(1.0, dtype=torch.float32, device=device)
        for _ in range(plan.k - _warm_start_num_steps(plan.start_t, plan.k)):
            timestep = timestep + grid
    step = _CountedStep(model.denoise_step)
    x_t = start_x
    for _ in range(_loop_steps(plan)):
        v_t = step(stage1.state, stage1.prefix_pad_masks, stage2.past_key_values, x_t, timestep.expand(bsize))
        x_t = x_t + dt * v_t
        timestep = timestep + dt
    return WarmResetStage3Output(action_chunk=x_t, steps_run=step.calls)


def run_pi05_self_start(
    model: Any, stage2: Any, noise: torch.Tensor, plans: Sequence[SelfStartPlan]
) -> list[Pi05SelfStartOutput]:
    """The full K-step loop from ``noise`` ([B, H, D]), one plan per batch row.

    Every plan must be a valid ``PI05_V1`` self start of one grid, and the
    stage-2 and noise batch sizes must equal ``len(plans)``; all of it is
    checked before the first ``denoise_step``. Row ``i`` gets its own snapshot
    (the input of step ``plans[i].capture_index``) or ``None`` for a final
    capture; a snapshot that the loop never reached raises. ``steps_run`` is
    the per-row iteration count, not B x K.
    """
    plans = list(plans)
    if not plans:
        raise ValueError("run_pi05_self_start needs at least one plan")
    for plan in plans:
        validate_self_plan(plan, PI05_V1)
    key = plans[0].grid_key()
    if any(plan.grid_key() != key for plan in plans):
        raise ValueError("run_pi05_self_start: plans of one batch must share a grid")
    stage1 = stage2.stage1
    batch = len(plans)
    if noise.dim() != 3 or noise.shape[0] != batch or stage1.state.shape[0] != batch:
        raise ValueError(
            f"run_pi05_self_start: {batch} plans but noise {tuple(noise.shape)} / "
            f"stage-2 batch {stage1.state.shape[0]}"
        )
    device = stage1.state.device
    dt = torch.tensor(-1.0 / plans[0].k, dtype=torch.float32, device=device)
    timestep = torch.tensor(1.0, dtype=torch.float32, device=device)
    step = _CountedStep(model.denoise_step)
    snapshots: list[Optional[torch.Tensor]] = [None] * batch
    x_t = noise
    for step_idx in range(_self_loop_steps(plans[0])):
        for i, plan in enumerate(plans):
            if plan.capture_index == step_idx:
                snapshots[i] = x_t[i : i + 1].clone()
        v_t = step(stage1.state, stage1.prefix_pad_masks, stage2.past_key_values, x_t, timestep.expand(batch))
        x_t = x_t + dt * v_t
        timestep = timestep + dt
    outs = []
    for i, plan in enumerate(plans):
        if plan.capture_index is not None and snapshots[i] is None:
            raise RuntimeError(
                f"self start: the loop never reached step {plan.capture_index} (start_t={plan.start_t})"
            )
        outs.append(Pi05SelfStartOutput(action_chunk=x_t[i : i + 1], snapshot=snapshots[i], steps_run=step.calls))
    return outs


# ------------------------------------------------------------------
# Executor
# ------------------------------------------------------------------


class Pi05WarmResetExecutor:
    """Runs the WARM_START continuation of ``InferenceInterceptor`` under a warm reset spec.

    Injected into the interceptor (``warm_reset=``); the interceptor only hands
    over the verdict, the stage-2 handle, the snapshot it already took and its
    stage-3 binding (``direct_runner`` or the coordinator submission).
    """

    def __init__(self, spec: WarmResetSpec, session: WarmResetSession) -> None:
        self.spec = spec
        self._session = session
        self._digest = spec.digest()

    @property
    def self_start(self) -> bool:
        """Whether decisions run a direct self-start inference first."""
        return self.spec.self_start

    def direct_runner(self, model: Any) -> Callable[[Any, torch.Tensor, Any], Any]:
        """The non-coordinator binding: dispatch a plan to its entry on ``model``."""

        def run(stage2: Any, x: torch.Tensor, plan: Any) -> Any:
            if isinstance(plan, SelfStartPlan):
                return run_pi05_self_start(model, stage2, x, [plan])[0]
            return run_pi05_continuation(model, stage2, x, plan)

        return run

    def _self_start(self, run_stage3: Callable, stage2: Any, noise: torch.Tensor, plan: SelfStartPlan) -> Any:
        self._session.count_self_start_call()
        out = run_stage3(stage2, noise, plan)
        self._session.add_self_start_steps(out.steps_run)
        return out

    def _continue(self, run_stage3: Callable, stage2: Any, start_x: torch.Tensor, plan: WarmResetPlan) -> Any:
        self._session.count_continuation_call()
        out = run_stage3(stage2, start_x, plan)
        self._session.add_continuation_steps(out.steps_run)
        return out

    def run(self, *, stage2: Any, cp_result: Any, snapshot_x: torch.Tensor, run_stage3: Callable, timer: Any) -> tuple[Stage3Output, dict]:
        """Resolve the plan, produce the start, run the continuation; ``(stage3, hit meta)``.

        ``snapshot_x`` is the ``[1, H, D]`` stage-3-device snapshot at the
        verdict's ``start_t``; the final action and the self-start result are
        cast and shaped like it, as the step_diag reference does.
        """
        session = self._session
        if session.decision_idx is None:
            raise RuntimeError("warm reset: no open decision (the evidence wrapper is not installed)")
        payload = cp_result.payload
        plan = resolve_plan(self.spec, PI05_V1, cp_result.start_t)
        if payload.denoising_num_steps != plan.k:
            raise ValueError(
                f"warm reset: payload denoising_num_steps={payload.denoising_num_steps} "
                f"but the plan runs K={plan.k}"
            )
        like = snapshot_x
        seed = None
        if self.spec.self_start:
            self_plan = resolve_self_plan(self.spec, PI05_V1, cp_result.start_t)
            if (self_plan.schedule_id, self_plan.k) != (plan.schedule_id, plan.k):
                raise ValueError("warm reset: self-start and continuation plans disagree on the schedule")
            seed = session.self_seed()
            noise = private_noise(seed, tuple(like.shape[-2:]))[None, ...].to(device=like.device)
            with timer.measure("stage3_self_start"):
                direct = self._self_start(run_stage3, stage2, noise, self_plan)
            chosen = direct.snapshot if self.spec.point == POINT_SNAPSHOT else direct.action_chunk
            start_x = chosen.to(device=like.device, dtype=like.dtype).reshape(like.shape)
        elif self.spec.point == POINT_FINAL:
            start_x = payload.action_chunk.to(device=like.device, dtype=like.dtype).reshape(like.shape)
        else:
            start_x = like
        with timer.measure("stage3_warm"):
            stage3 = self._continue(run_stage3, stage2, start_x, plan)
        dt = float(torch.tensor(-plan.level / plan.n_steps, dtype=torch.float32))
        meta = decision_meta(self.spec, self._digest, plan, session, t=plan.flow_times(), dt=dt, self_seed=seed)
        return stage3, meta


def build_pi05_warm_reset(config: Any, *, bundle_id: str, yaml_id: Optional[str], yaml_path: Optional[str]) -> Optional[WarmResetParts]:
    """Per-connection executor + evidence wrapper for ``_wrap_policy``; ``None`` without a block."""
    return _build(
        config,
        family="pi05",
        bundle_id=bundle_id,
        yaml_id=yaml_id,
        yaml_path=yaml_path,
        executor_factory=Pi05WarmResetExecutor,
        wrapper_factory=WarmResetEvidencePolicy,
    )


__all__ = [
    "Pi05SelfStartOutput",
    "Pi05WarmResetExecutor",
    "WarmResetStage3Output",
    "build_pi05_warm_reset",
    "run_pi05_continuation",
    "run_pi05_self_start",
]
