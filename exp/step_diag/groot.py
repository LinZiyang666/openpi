"""GR00T N1.5 served objects of the step-vs-warm-start line (RoboCasa365 / LIBERO).

Two compositions, both satisfying the ``get_action`` + episode-hook protocol the GR00T servers
hand to ``GrootPolicyAdapter`` (the same contract ``GrootRitShadow`` satisfies):

``GrootDiagPolicy`` (mode ``shadow``)
    Teacher-executed decision: stage 1 -> read-only CP1 check (retrieval winner only) -> stage 2
    -> the upstream full loop (``runner.run_stage3(stage2)``, noise drawn by upstream) is the
    executed action. The recorder then draws private noises and runs, on the *same* stage-2
    handle, ``staged.denoise_loop(head, runner._head_inputs(stage2), ..., num_steps=k)`` for the
    full count and every reduced ``k`` (a fresh head-input mapping per call: the action head
    normalises the backbone features in place), plus ``runner.run_stage3_from`` from the winner's
    cached snapshot for every warm ``t``. The live head's ``num_inference_timesteps`` is never
    changed.

``GrootEvidencePolicy`` (modes ``plain`` / ``full`` / ``warm``)
    Transparent wrapper around the production served object (the plain teacher with the head
    pinned to ``k`` by the k-sweep constructor patch, or ``GrootCacheInterceptor`` with the forced
    warm-start yaml). It wraps the runner's ``run_stage3`` / ``run_stage3_from`` *instance*
    methods to capture the last ``GrootStage3Output`` of the request thread and records one
    evidence row per decision: executed chunk, ``steps_run``, hit type and ``start_t``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
import torch

from openpi.cache.groot import staged as _staged
from openpi.cache.groot.staged import GrootStage3Output
from openpi.cache.groot.interceptor import _is_batched, _unsqueeze_values, _squeeze_values
from openpi.cache.types import CheckpointID

from exp.step_diag.recorder import DiagRecorder, EpisodeIdentity, make_noise

logger = logging.getLogger("exp.step_diag.groot")


def _storage_chunk(chunk: torch.Tensor) -> torch.Tensor:
    out = chunk[0].detach().cpu().float().contiguous()
    if out.is_inference():
        out = out.clone()
    return out


GROOT_SHOOT_VARIANTS = ("overshoot", "mid_shoot", "mid_shoot50")
GROOT_WARM_VARIANTS = ("reset_t", "reset_final", "mid_final", "mid_final50", "mid_snap", "mid_snap50",
                       *GROOT_SHOOT_VARIANTS)
GROOT_FINAL_START_VARIANTS = ("reset_final", "mid_final", "mid_final50")

# Self-start ablation: the evidence wrapper publishes the decision's private-noise seed here before the
# served object runs, and the variant publishes what it did (seed, direct-inference NFE) for the row.
_SELF = threading.local()


def groot_self_start(runner: Any, stage2: Any, like: torch.Tensor, start_t: float, *, schedule: Any, variant: str,
                     seed: int, step_fn: Any = None) -> torch.Tensor:
    """The start a cache arm would have fed, produced by a direct full inference on this decision.

    Runs upstream's K-step loop (``staged.denoise_loop``, the transcription ``run_stage3(noise=)`` uses) from
    private float32 noise (``make_noise(seed)``; the loop casts it to the backbone-feature dtype, and the global RNG
    is not touched) on the same stage-2 handle and returns its snapshot at native ``start_t`` -- the input of loop step
    ``schedule.snapshot_index(start_t)``, the cache's snapshot convention -- or, for the final-start variants,
    its final action. Nothing here is executed or counted as an executed stage-3 call.
    """
    head = runner._model.action_head  # noqa: SLF001 - same seam as groot_warm_variant_stage3
    noise = make_noise(int(seed), tuple(like.shape[-2:]))[None, ...].to(device=like.device)
    snap_index = int(schedule.snapshot_index(float(start_t)))
    got = {}

    def on_step(i, x_in, x_out):
        if int(i) == snap_index:
            got["snap"] = x_in.detach().clone()

    extra = {} if step_fn is None else {"step_fn": step_fn}
    final = _staged.denoise_loop(head, runner._head_inputs(stage2), stage2.action_inputs, noise=noise,  # noqa: SLF001
                                 num_steps=int(schedule.num_steps), start_index=0, on_step=on_step, **extra)
    x = final if variant in GROOT_FINAL_START_VARIANTS else got["snap"]
    return x.to(device=like.device, dtype=like.dtype).reshape(like.shape)


def mid_denoise_loop(action_head: Any, backbone_output: Any, action_input: Any, *, start: torch.Tensor,
                     entry_t: float, num_steps: int, step_fn: Any = None) -> torch.Tensor:
    """Upstream's ascending loop (``openpi.cache.groot.staged.denoise_loop``) on the grid
    ``t_i = entry_t + i * (1 - entry_t) / n``, ``dt = (1 - entry_t) / n``: the same preprocessing, bucket
    discretisation and ``step_fn``. ``entry_t = 0`` reproduces ``denoise_loop(noise=start, num_steps=n)``
    exactly (pinned by the tests)."""
    if not 0.0 <= entry_t < 1.0 or num_steps < 1:
        raise ValueError(f"entry_t={entry_t} / num_steps={num_steps} out of range")
    step_fn = _staged.denoise_step if step_fn is None else step_fn
    processed = action_head.process_backbone_output(backbone_output)
    vl = processed.backbone_features
    embodiment_id = action_input["embodiment_id"]
    state_features = action_head.state_encoder(action_input["state"], embodiment_id)
    batch_size = vl.shape[0]
    actions = start.to(device=vl.device, dtype=vl.dtype)
    dt = (1.0 - float(entry_t)) / num_steps
    for i in range(num_steps):
        t_cont = float(entry_t) + i * dt
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=int(t_cont * action_head.num_timestep_buckets),
                                      device=vl.device)
        actions = step_fn(action_head, vl, state_features, embodiment_id, actions, timesteps_tensor, dt).clone()
    return actions


def shoot_denoise_loop(action_head: Any, backbone_output: Any, action_input: Any, *, start: torch.Tensor,
                       t0: float, dt: float, num_steps: int, step_fn: Any = None) -> torch.Tensor:
    """Upstream's ascending loop on the grid ``t_i = t0 + i * dt`` (GR00T time), the same preprocessing, bucket
    discretisation and ``step_fn`` as ``mid_denoise_loop``, but with a free step size: the shoot variants keep the
    snapshot at its own ``t0`` and take the warm-reset step, so ``t0 + n * dt`` lands past the clean end (1)."""
    if not 0.0 <= t0 < 1.0 or dt <= 0.0 or num_steps < 1:
        raise ValueError(f"t0={t0} / dt={dt} / num_steps={num_steps} out of range")
    step_fn = _staged.denoise_step if step_fn is None else step_fn
    processed = action_head.process_backbone_output(backbone_output)
    vl = processed.backbone_features
    embodiment_id = action_input["embodiment_id"]
    state_features = action_head.state_encoder(action_input["state"], embodiment_id)
    batch_size = vl.shape[0]
    actions = start.to(device=vl.device, dtype=vl.dtype)
    for i in range(num_steps):
        t_cont = float(t0) + i * float(dt)
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=int(t_cont * action_head.num_timestep_buckets),
                                      device=vl.device)
        actions = step_fn(action_head, vl, state_features, embodiment_id, actions, timesteps_tensor, float(dt)).clone()
    return actions


def groot_warm_variant_stage3(runner: Any, stage2: Any, start_x: torch.Tensor, start_t: float, *, schedule: Any,
                              variant: str, step_fn: Any = None, num_steps: int | None = None) -> GrootStage3Output:
    """GR00T mirror of ``exp.step_diag.pi05.warm_variant_stage3`` (the reviewer's dt = 1/remaining proposal).

    The exact resume (``GrootStagedRunner.run_stage3_from``) continues the ascending K-step loop from
    ``start_index = schedule.snapshot_index(start_t)`` with the library grid ``dt = 1/K``. The variants keep
    the same call count ``n = schedule.remaining_steps(start_t)`` but run a NEW ``n``-step loop from
    ``t = 0`` with ``dt = 1/n`` -- i.e. upstream's own ``denoise_loop(noise=<start>, num_steps=n)``.
    ``num_steps`` (GR00T LIBERO, K = 8) overrides ``n`` so an arm keeps RoboCasa's K = 4 (T, N, t) tuple while
    its start stays the snapshot at ``start_t``; left at None, ``n`` is the remaining-step count as before:

    * ``reset_t``    : the start is the cached snapshot ``x_{start_t}`` (fed as if it were noise);
    * ``reset_final``: the start is the payload's final action chunk (the caller substitutes it).
    * ``mid_final``  : the ``reset_final`` start entered at GR00T time 0.25 (pi0.5 flow time 0.75, one K=4 grid
                       step below pure noise) instead of 0, ``n`` steps of ``dt = 0.75/n`` to the clean end
                       (``mid_denoise_loop``; upstream's loop cannot express a non-``i/N`` grid).

    * ``overshoot`` / ``mid_shoot`` / ``mid_shoot50`` (shoot ablation, owner 2026-09-24): no reset -- the snapshot
                       stays at its own GR00T time ``start_t`` and takes ``n`` steps of ``dt = SHOOT_ENTRY_T / n``
                       (the warm-reset / midreset / midreset50 step), running past the clean end
                       (``shoot_denoise_loop``). The step count is stamped in
    ``steps_run`` exactly like the resume so the evidence policy records ``executed_steps = n``.
    """
    if variant not in GROOT_WARM_VARIANTS:
        raise ValueError(f"unknown GR00T warm variant {variant!r}")
    n_steps = int(schedule.remaining_steps(start_t)) if num_steps is None else int(num_steps)
    if n_steps < 1:
        raise ValueError(f"start_t={start_t} leaves no step to run")
    head = runner._model.action_head  # noqa: SLF001 - same seam as the runner's own resume
    backbone_outputs = runner._head_inputs(stage2)  # noqa: SLF001
    if start_x.dim() == 2:
        start_x = start_x[None, ...]
    extra = {} if step_fn is None else {"step_fn": step_fn}
    with runner._timer.measure("stage3_warm"):  # noqa: SLF001
        if variant in GROOT_SHOOT_VARIANTS:
            from exp.step_diag.envs import SHOOT_ENTRY_T

            action_pred = shoot_denoise_loop(head, backbone_outputs, stage2.action_inputs, start=start_x,
                                             t0=float(start_t), dt=SHOOT_ENTRY_T[variant] / n_steps,
                                             num_steps=n_steps, **extra)
        elif variant in ("mid_final", "mid_final50", "mid_snap", "mid_snap50"):
            from exp.step_diag.envs import MID_ENTRY_T_BY_VARIANT

            action_pred = mid_denoise_loop(head, backbone_outputs, stage2.action_inputs, start=start_x,
                                           entry_t=1.0 - MID_ENTRY_T_BY_VARIANT[variant]["groot"], num_steps=n_steps, **extra)
        else:
            action_pred = _staged.denoise_loop(head, backbone_outputs, stage2.action_inputs, noise=start_x,
                                               num_steps=n_steps, start_index=0, **extra)
    runner._model.validate_data(_staged._batch_feature({"action_pred": action_pred}), backbone_outputs,  # noqa: SLF001
                                is_training=False)
    return GrootStage3Output(action_pred=action_pred, start_t=float(start_t), steps_run=n_steps)


def install_warm_variant(runner: Any, orchestrator: Any, variant: str, schedule: Any, *, step_fn: Any = None,
                         self_start: bool = False, num_steps: int | None = None) -> None:
    """Route the executed WARM_START continuation of ``runner`` through a variant (install BEFORE the
    evidence capture wraps the runner, so the capture still sees the variant's output).

    ``reset_final`` needs the hit payload: the orchestrator's ``check`` is wrapped to stash its result
    thread-locally, and the payload's ``action_chunk`` replaces the snapshot the interceptor hands in.
    ``self_start`` (self-start ablation): the start comes from ``groot_self_start`` with the seed the evidence
    wrapper published in ``_SELF.seed``; the retrieved payload is ignored. ``num_steps``: the continuation's
    explicit step count (``groot_warm_variant_stage3``); None keeps ``remaining_steps(start_t)``.
    """
    if variant not in GROOT_WARM_VARIANTS:
        raise ValueError(f"unknown GR00T warm variant {variant!r}")
    local = threading.local()
    if orchestrator is not None:
        inner_check = orchestrator.check

        def stashing_check(*a, **kw):
            result = inner_check(*a, **kw)
            local.cp1 = result
            return result

        orchestrator.check = stashing_check

    def variant_run3f(stage2, start_x, start_t, *, schedule=schedule, **kw):
        if self_start:
            seed = getattr(_SELF, "seed", None)
            if seed is None:
                raise RuntimeError("self start: no decision seed published by the evidence wrapper")
            start_x = groot_self_start(runner, stage2, start_x, start_t, schedule=schedule, variant=variant,
                                       seed=seed, step_fn=step_fn)
            _SELF.info = {"self_start": True, "self_seed": int(seed), "self_direct_nfe": int(schedule.num_steps)}
            return groot_warm_variant_stage3(runner, stage2, start_x, start_t, schedule=schedule, variant=variant,
                                             step_fn=step_fn, num_steps=num_steps)
        if variant in GROOT_FINAL_START_VARIANTS:
            cp1 = getattr(local, "cp1", None)
            payload = None if cp1 is None else getattr(cp1, "payload", None)
            chunk = None if payload is None else getattr(payload, "action_chunk", None)
            if chunk is None:
                raise RuntimeError("reset_final: no WARM_START retrieval payload to take the final chunk from")
            start_x = chunk.to(device=start_x.device, dtype=start_x.dtype).reshape(start_x.shape)
        return groot_warm_variant_stage3(runner, stage2, start_x, start_t, schedule=schedule, variant=variant,
                                         step_fn=step_fn, num_steps=num_steps)

    runner.run_stage3_from = variant_run3f


class _Stage3Capture:
    """Thread-local capture of the last stage-3 output produced through a runner instance."""

    def __init__(self, runner: Any) -> None:
        self._local = threading.local()
        inner3, inner3f = runner.run_stage3, runner.run_stage3_from

        def run_stage3(*a, **kw):
            out = inner3(*a, **kw)
            self._local.last = out
            self._local.calls = getattr(self._local, "calls", 0) + 1
            return out

        def run_stage3_from(*a, **kw):
            out = inner3f(*a, **kw)
            self._local.last = out
            self._local.calls = getattr(self._local, "calls", 0) + 1
            return out

        runner.run_stage3 = run_stage3
        runner.run_stage3_from = run_stage3_from

    def reset(self) -> None:
        self._local.last = None
        self._local.calls = 0

    @property
    def last(self):
        return getattr(self._local, "last", None)

    @property
    def calls(self) -> int:
        return int(getattr(self._local, "calls", 0))


class GrootEvidencePolicy:
    """Evidence-only wrapper for the plain / full / warm arms (see module docstring)."""

    def __init__(self, inner: Any, runner: Any, diag: DiagRecorder, *, schedule_id: str) -> None:
        self._inner = inner
        self._capture = _Stage3Capture(runner)
        self._diag = diag.session()  # one connection = one episode state
        self._schedule_id = schedule_id

    def on_task_begin(self, task_key: str = "") -> None:
        fn = getattr(self._inner, "on_task_begin", None)
        if fn is not None:
            fn(task_key) if _accepts_arg(fn) else fn()

    def on_task_end(self) -> None:
        fn = getattr(self._inner, "on_task_end", None)
        if fn is not None:
            fn()

    def on_episode_start(self, experiment: str = "", task: str = "", episode_id: int = -1,
                         episode_name: str = "", extra_metadata: dict | None = None) -> None:
        self._diag.begin_episode(EpisodeIdentity.from_episode_start(
            experiment=experiment, task=task, episode_id=episode_id, extra_metadata=extra_metadata))
        fn = getattr(self._inner, "on_episode_start", None)
        if fn is not None:
            fn(experiment=experiment, task=task, episode_id=episode_id, episode_name=episode_name,
               extra_metadata=extra_metadata)

    def on_episode_end(self, success: bool) -> None:
        try:
            self._diag.finalize_episode(bool(success), terminal=True)
        finally:
            fn = getattr(self._inner, "on_episode_end", None)
            if fn is not None:
                fn(success)

    def get_action(self, observations: dict) -> dict:
        self._capture.reset()
        _SELF.seed = self._diag.self_start_seed()  # used only by self-start variants
        _SELF.info = None
        out = self._inner.get_action(observations)
        meta = out.get("__hit_meta__") if isinstance(out, dict) else None
        last = self._capture.last
        hit_type = None if meta is None else meta.get("hit_type")
        start_t = None if meta is None else meta.get("start_t")
        a_exec = None if last is None else _storage_chunk(last.action_pred)
        steps = None if last is None else int(getattr(last, "steps_run", 0))
        if hit_type is None:
            hit_type = "MISS"
        if a_exec is not None:
            self._diag.record(a_exec=a_exec, executed_steps=steps, n_stage3_calls=self._capture.calls,
                              hit_type=hit_type, start_t=start_t, schedule_id=self._schedule_id,
                              extra=getattr(_SELF, "info", None) or None)
        else:
            # FULL_HIT (no stage-3 call) or an unexpected path: keep the row, mark the gap.
            self._diag.record(a_exec=np.zeros((1, 1), dtype=np.float32), executed_steps=0, n_stage3_calls=0,
                              hit_type=hit_type, start_t=start_t, schedule_id=self._schedule_id,
                              extra={"a_exec_missing": True})
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _accepts_arg(fn) -> bool:
    import inspect

    try:
        return len(inspect.signature(fn).parameters) >= 1
    except (TypeError, ValueError):
        return False


class GrootDiagPolicy:
    """Teacher-executed shadow policy (mode ``shadow``); see module docstring."""

    def __init__(self, policy: Any, runner: Any, *, orchestrator: Any, diag: DiagRecorder,
                 schedule: Any, shadow: bool = True) -> None:
        self._policy = policy
        self._runner = runner
        self._orchestrator = orchestrator
        self._diag = diag.session()  # one connection = one episode state
        self._schedule = schedule
        self._shadow_on = bool(shadow)
        self._live = (int(runner._model.action_head.num_inference_timesteps) if not shadow
                      else int(runner.live_schedule().num_steps) if hasattr(runner, "live_schedule") else schedule.num_steps)
        if shadow and self._live != schedule.num_steps:
            raise RuntimeError(f"live head runs {self._live} steps but the arm schedule is {schedule.schedule_id}")

    # -- lifecycle -------------------------------------------------------

    def on_task_begin(self, task_key: str = "") -> None:
        if self._orchestrator is not None:
            self._orchestrator.on_task_begin(task_key)

    def on_task_end(self) -> None:
        if self._orchestrator is not None:
            self._orchestrator.on_task_end()

    def on_episode_start(self, experiment: str = "", task: str = "", episode_id: int = -1,
                         episode_name: str = "", extra_metadata: dict | None = None) -> None:
        self._diag.begin_episode(EpisodeIdentity.from_episode_start(
            experiment=experiment, task=task, episode_id=episode_id, extra_metadata=extra_metadata))
        if self._orchestrator is not None:
            self._orchestrator.on_episode_start(task_key=task, episode_id=str(episode_id),
                                                extra_metadata=extra_metadata)

    def on_episode_end(self, success: bool) -> None:
        try:
            self._diag.finalize_episode(bool(success), terminal=True)
        finally:
            if self._orchestrator is not None:
                self._orchestrator.on_episode_end()

    # -- inference -------------------------------------------------------

    def get_action(self, observations: dict) -> dict:
        obs_copy = observations.copy()
        is_batch = _is_batched(obs_copy)
        if not is_batch:
            obs_copy = _unsqueeze_values(obs_copy)
        for key, value in obs_copy.items():
            if not isinstance(value, np.ndarray):
                obs_copy[key] = np.array(value)
        normalized_input = self._policy.apply_transforms(obs_copy)
        with self._runner.session():
            stage1 = self._runner.run_stage1(normalized_input)
        cp1 = None
        retrieval_error = None
        try:
            if self._orchestrator is not None:
                try:
                    cp1 = self._orchestrator.check(CheckpointID.CP1, stage1=stage1)
                except Exception as exc:
                    retrieval_error = f"{type(exc).__name__}: {exc}"
            with self._runner.session():
                stage2 = self._runner.run_stage2_llm(stage1)
                teacher = self._runner.run_stage3(stage2)
            action_cpu = _storage_chunk(teacher.action_pred)
            self._shadow(cp1, stage2, teacher, action_cpu, retrieval_error=retrieval_error)
            if self._orchestrator is not None:
                self._orchestrator.broadcast_action(action_cpu)
        finally:
            if self._orchestrator is not None:
                self._orchestrator.clear()
        unnormalized = self._policy.unapply_transforms({"action": action_cpu[None, ...]})
        return unnormalized if is_batch else _squeeze_values(unnormalized)

    def _shadow(self, cp1, stage2, teacher, action_cpu: torch.Tensor, *, retrieval_error=None) -> None:
        if not self._shadow_on:
            self._diag.record(a_exec=action_cpu, executed_steps=int(getattr(teacher, "steps_run", self._live)),
                              n_stage3_calls=1, hit_type="MISS", start_t=None,
                              schedule_id=f"groot_n15_k{self._live}_v1")
            return
        runner, head = self._runner, self._runner._model.action_head  # noqa: SLF001 - staged runner seam
        def sample(z: torch.Tensor, k: int):
            if retrieval_error is not None:
                raise RuntimeError(retrieval_error)
            with runner.session():
                pred = _staged.denoise_loop(head, runner._head_inputs(stage2), stage2.action_inputs,  # noqa: SLF001
                                            noise=z[None, ...], num_steps=int(k))
            return _storage_chunk(pred)

        def resume(x_t: torch.Tensor, t: float):
            with runner.session():
                out = runner.run_stage3_from(stage2, x_t, float(t), schedule=self._schedule)
            return _storage_chunk(out.action_pred)

        def top1():
            payload = None if cp1 is None else getattr(cp1, "payload", None)
            score = None if cp1 is None else getattr(cp1, "score", None)
            entry = None if cp1 is None else getattr(cp1, "entry_id", None)
            inter = None if payload is None else getattr(payload, "intermediates", None)
            if inter:
                inter = {round(float(t), 4): x for t, x in inter.items()}
            return (None if score is None else float(score), entry, inter or None)

        self._diag.record(a_exec=action_cpu, executed_steps=int(getattr(teacher, "steps_run", self._live)),
                          n_stage3_calls=1, hit_type="MISS", start_t=None, schedule_id=self._schedule.schedule_id,
                          sample=sample, resume=resume, top1=top1,
                          extra={"top1_score": None if cp1 is None or getattr(cp1, "score", None) is None
                                 else float(cp1.score),
                                 "top1_entry_id": None if cp1 is None else getattr(cp1, "entry_id", None)})


__all__ = ["GrootDiagPolicy", "GrootEvidencePolicy"]
