"""Pi0.5 served policy of the step-vs-warm-start line: ``InferenceInterceptor`` + ``DiagRecorder``.

``Pi05DiagInterceptor`` is a subclass composed at server start (``serve_diag_pi05.py`` rebinds
``openpi.cache.interceptor.InferenceInterceptor`` before ``scripts.serve_policy`` builds the
policy). It changes nothing on the executed path: the base class computes and returns the action
exactly as before; the subclass only observes three seams the base class already exposes:

* ``_record_shadow_teacher_arm(cp1_result, stage3)`` -- called by ``infer`` after every
  MISS / WARM_START decision with the executed ``Stage3Output`` in hand;
* ``_record_shadow(cp1_result, cached_action, stage1)`` -- the FULL_HIT mirror;
* ``on_episode_start / on_episode_end`` -- the episode identity and the terminal row.

Stage-2 output is captured per request by wrapping the instance's ``_stage2_fn`` (thread-local,
so two connections never see each other's handle); stage-3 Euler steps are counted by wrapping
the model instance's ``denoise_step`` (thread-local per request), and the executed path's stage-3
entry points (the model instance's ``run_stage3`` / ``run_stage3_from``, outside the shadow
bracket) are counted as ``n_stage3_calls``. Shadow samples call the model
directly (``run_stage3(noise=, num_steps=)`` / ``run_stage3_from``) under ``torch.no_grad`` on
the same stage-2 handle; they never enter the coordinator and never touch the global RNG.

The read-only retrieval winner is taken from the CP1 search the orchestrator already ran for the
decision: the CP1 search strategy is wrapped with a thread-local recorder of its last result
list, so no second search is issued and the orchestrator's verdict / accounting are untouched.
"""

from __future__ import annotations

import logging
import dataclasses
import threading
from typing import Any, Optional

import torch

from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.components.judge import HitType
from openpi.cache.types import PI05_V1, CheckpointID
from openpi.models_pytorch.pi0_pytorch import Stage3Output, _warm_start_num_steps

from exp.step_diag.recorder import DiagRecorder, EpisodeIdentity, make_noise

logger = logging.getLogger("exp.step_diag.pi05")


class _RecordingStrategy:
    """Forwards every attribute to the wrapped search strategy; remembers the last results per thread."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._local = threading.local()

    def search(self, ctx):
        results = self._inner.search(ctx)
        self._local.results = results
        return results

    @property
    def last_results(self):
        return getattr(self._local, "results", None)

    def clear(self) -> None:
        self._local.results = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


WARM_VARIANTS = ("reset_t", "overshoot", "reset_final", "mid_final", "mid_final50", "mid_snap", "mid_snap50")
FINAL_START_VARIANTS = ("reset_final", "mid_final", "mid_final50")  # start = the payload's final action chunk


def warm_variant_stage3(model: Any, stage2: Any, start_x: torch.Tensor, start_t: float, *, num_steps: int,
                        variant: str) -> Stage3Output:
    """Warm-start continuation with ``dt = -1 / remaining_steps`` (the reviewer's proposal, 2026-09-21).

    Both variants run ``n = floor(start_t * num_steps + 0.5)`` Euler steps from the cached ``start_x``
    -- the same call count as ``run_stage3_from`` -- but with the enlarged step ``dt = -1/n``
    instead of the library grid's ``-1/num_steps``:

    * ``reset_t``  : the remaining denoising is treated as a NEW run -- the flow time restarts at
                     ``t = 1`` (the model is told the cache is pure noise) and walks ``1, 1-1/n, ...``;
    * ``overshoot``: the flow time starts at the cache's own ``start_t`` (replayed exactly like
                     ``run_stage3_from``) but advances by ``-1/n`` per step, so it passes ``t = 0``
                     after the first step and the later steps query negative ``t``.
    * ``reset_final``: the ``reset_t`` loop, but the caller hands in the cache's FINAL action chunk
                     (the payload's ``action_chunk``, t = 0) instead of the snapshot at ``start_t``;
                     ``start_t`` then only sets the step budget ``n`` (ablation: budget vs snapshot
                     noise level, 2026-09-22).
    * ``mid_final``: the ``reset_final`` start entered at ``t = MID_ENTRY_T["pi05"]`` = 0.9, one
                     full-schedule grid step below pure noise (not 1), with ``dt = -0.9/n``, ending at ``t = 0`` (owner 2026-09-22: does the cache
                     survive when it is not declared pure noise?).

    Every step goes through ``model.denoise_step`` (the instance attribute, so the interceptor's
    step counter sees it). ``resume`` is not handled here: that is ``model.run_stage3_from``.
    """
    if variant not in WARM_VARIANTS:
        raise ValueError(f"unknown warm variant {variant!r}")
    stage1 = stage2.stage1
    device = stage1.state.device
    bsize = stage1.state.shape[0]
    num_steps = int(num_steps)
    n_steps = _warm_start_num_steps(float(start_t), num_steps)
    if n_steps < 1:
        raise ValueError(f"start_t={start_t} leaves no step to run")
    dt = torch.tensor(-1.0 / n_steps, dtype=torch.float32, device=device)
    if variant in ("mid_final", "mid_final50", "mid_snap", "mid_snap50"):
        from exp.step_diag.envs import MID_ENTRY_T_BY_VARIANT

        entry = MID_ENTRY_T_BY_VARIANT[variant]["pi05"]
        dt = torch.tensor(-entry / n_steps, dtype=torch.float32, device=device)
        timestep = torch.tensor(entry, dtype=torch.float32, device=device)
    elif variant in ("reset_t", "reset_final"):
        timestep = torch.tensor(1.0, dtype=torch.float32, device=device)
    else:
        grid = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        timestep = torch.tensor(1.0, dtype=torch.float32, device=device)
        for _ in range(num_steps - n_steps):  # the full loop's float32 accumulation, as run_stage3_from
            timestep = timestep + grid
    x_t = start_x
    for _ in range(n_steps):
        v_t = model.denoise_step(stage1.state, stage1.prefix_pad_masks, stage2.past_key_values, x_t,
                                 timestep.expand(bsize))
        x_t = x_t + dt * v_t
        timestep = timestep + dt
    return Stage3Output(action_chunk=x_t)


class Pi05DiagInterceptor(InferenceInterceptor):
    """See module docstring. ``diag`` is the recorder; ``mode`` is shadow / plain / full / warm;
    ``exec_steps`` (plain / full only) pins the executed step count of this connection.
    ``self_start`` (self-start ablation arms): the variant's start is taken from a direct full inference
    on the current observation instead of the retrieved cache entry (see ``_self_start``)."""

    def __init__(self, *args: Any, diag: DiagRecorder, mode: str, exec_steps: Optional[int] = None,
                 warm_variant: Optional[str] = None, self_start: bool = False, **kwargs: Any) -> None:
        if warm_variant is not None and warm_variant not in WARM_VARIANTS:
            raise ValueError(f"unknown warm variant {warm_variant!r}")
        if self_start and (warm_variant is None or warm_variant == "overshoot"):
            raise ValueError("self_start needs a reset-family warm variant")
        self._warm_variant = warm_variant
        self._self_start_on = bool(self_start)
        super().__init__(*args, **kwargs)
        shape = (self._model.config.action_horizon, self._model.config.action_dim)
        if shape != diag.spec.action_shape:
            raise ValueError(f"live pi0.5 action shape {shape} != frozen {diag.spec.action_shape}")
        self._diag = diag.session()  # one connection = one episode state
        self._diag_mode = str(mode)
        self._tl = threading.local()
        # plain / full: pin the executed Euler step count on THIS instance's stage-3 binding. The
        # no-cache branch of ``infer`` calls ``self._stage3_fn(stage2, noise=...)`` without
        # ``num_steps`` (model default K), so the module-level ``_NUM_STEPS`` pin alone would not
        # reach it; the shadow samples call the model directly with an explicit ``num_steps``.
        self._exec_steps = None if exec_steps is None else int(exec_steps)
        if self._exec_steps is not None:
            inner_stage3 = self._stage3_fn
            pinned = self._exec_steps

            def _stage3_pinned(stage2, **kw):
                kw["num_steps"] = pinned
                return inner_stage3(stage2, **kw)

            self._stage3_fn = _stage3_pinned
        # stage-2 capture (per request thread)
        inner_stage2 = self._stage2_fn

        def _stage2_capture(stage1):
            out = inner_stage2(stage1)
            self._tl.stage2 = out
            return out

        self._stage2_fn = _stage2_capture
        # Euler-step counter on the model instance (not the class): every stage-3 path of this
        # model -- executed loop, warm resume, shadow samples -- goes through ``denoise_step``.
        model = self._model
        inner_step = model.denoise_step

        def _counted_step(*a, **kw):
            self._tl.n_steps = getattr(self._tl, "n_steps", 0) + 1
            return inner_step(*a, **kw)

        model.denoise_step = _counted_step
        # stage-3 entry-point counter of the executed path (``n_stage3_calls``: 1 per MISS /
        # WARM_START decision, 0 on FULL_HIT), on the model instance so every binding the
        # interceptor may take (``_stage3_fn``, the direct ``model.run_stage3`` branch,
        # ``run_stage3_from``) is seen; the shadow samples bracket themselves with ``in_shadow``.
        inner_run3, inner_run3f = model.run_stage3, model.run_stage3_from

        def _counted_run3(*a, **kw):
            if not getattr(self._tl, "in_shadow", False):
                self._tl.n_calls = getattr(self._tl, "n_calls", 0) + 1
            return inner_run3(*a, **kw)

        variant = self._warm_variant

        def _counted_run3f(stage2, start_x, start_t, *, num_steps=10, **kw):
            if not getattr(self._tl, "in_shadow", False):
                self._tl.n_calls = getattr(self._tl, "n_calls", 0) + 1
                if variant is not None:
                    # warmreset / warmshoot arms: the executed WARM_START continuation uses the
                    # enlarged step (dt = -1/remaining); the shadow bracket keeps the exact resume.
                    if self._self_start_on:
                        start_x = self._self_start(stage2, start_x, start_t, num_steps)
                    elif variant in FINAL_START_VARIANTS:
                        start_x = self._final_chunk_like(start_x)
                    return warm_variant_stage3(model, stage2, start_x, start_t, num_steps=num_steps,
                                               variant=variant)
            return inner_run3f(stage2, start_x, start_t, num_steps=num_steps, **kw)

        model.run_stage3 = _counted_run3
        model.run_stage3_from = _counted_run3f
        # ``_stage3_fn`` was bound to the ORIGINAL method before the instance wrapper above existed
        # (eager binding in the base __init__), so it is counted on its own; the two paths are disjoint.
        inner_stage3_binding = self._stage3_fn

        def _counted_binding(stage2, **kw):
            self._tl.n_calls = getattr(self._tl, "n_calls", 0) + 1
            return inner_stage3_binding(stage2, **kw)

        self._stage3_fn = _counted_binding
        # read-only retrieval winner for the shadow mode
        self._search_rec: Optional[_RecordingStrategy] = None
        orch = self._orchestrator
        if variant in FINAL_START_VARIANTS and orch is not None:
            stash_check = orch.check

            def stashing_check(*a, **kw):
                result = stash_check(*a, **kw)
                self._tl.cp1_result = result
                return result
            orch.check = stashing_check
        if self._diag_mode == "shadow" and orch is not None:
            from openpi.cache.orchestrator import CheckResult

            original_check = orch.check
            def shadow_check(*a, **kw):
                try:
                    result = original_check(*a, **kw)
                    return dataclasses.replace(result, hit_type=HitType.MISS, start_t=None)
                except Exception as exc:
                    self._tl.retrieval_error = f"{type(exc).__name__}: {exc}"
                    return CheckResult(hit_type=HitType.MISS)
            orch.check = shadow_check
            strategies = orch._search_strategies  # noqa: SLF001 - experiment-layer observer
            if CheckpointID.CP1 in strategies:
                self._search_rec = _RecordingStrategy(strategies[CheckpointID.CP1])
                strategies[CheckpointID.CP1] = self._search_rec
        logger.info("Pi05DiagInterceptor mode=%s arm=%s", self._diag_mode, diag.spec.arm_id)

    # -- request boundary ------------------------------------------------

    def infer(self, obs: dict, *, noise=None) -> dict:  # type: ignore[override]
        self._tl.n_steps = 0
        self._tl.n_calls = 0
        self._tl.stage2 = None
        self._tl.retrieval_error = None
        self._tl.self_info = None
        if self._search_rec is not None:
            self._search_rec.clear()
        return super().infer(obs, noise=noise)

    # -- episode boundary ------------------------------------------------

    def on_episode_start(self, experiment: str = "", task: str = "", episode_id: int = -1,
                         episode_name: str = "", extra_metadata: dict | None = None) -> None:
        self._diag.begin_episode(EpisodeIdentity.from_episode_start(
            experiment=experiment, task=task, episode_id=episode_id, extra_metadata=extra_metadata))
        super().on_episode_start(experiment=experiment, task=task, episode_id=episode_id,
                                 episode_name=episode_name, extra_metadata=extra_metadata)

    def on_episode_end(self, success: bool) -> None:
        try:
            self._diag.finalize_episode(bool(success), terminal=True)
        finally:
            super().on_episode_end(success)

    # -- decision seams --------------------------------------------------

    def _record_shadow(self, cp1_result, cached_action, stage1) -> None:
        """FULL_HIT mirror: the executed action is the cached chunk, zero Euler steps."""
        self._diag.record(
            a_exec=cached_action, executed_steps=0, n_stage3_calls=0,
            hit_type="FULL_HIT", start_t=None, schedule_id=PI05_V1.schedule_id,
            extra={"top1_score": _score_of(cp1_result), "top1_entry_id": getattr(cp1_result, "entry_id", None)},
        )

    def _record_shadow_teacher_arm(self, cp1_result, stage3) -> None:
        """MISS / WARM_START: record the executed chunk and, in shadow mode, the sample matrix."""
        executed_steps = int(getattr(self._tl, "n_steps", 0))  # read BEFORE the shadow samples run
        n_calls = int(getattr(self._tl, "n_calls", 0))
        hit = None if cp1_result is None else getattr(cp1_result, "hit_type", None)
        hit_name = None if hit is None else (hit.name if hasattr(hit, "name") else str(hit))
        start_t = None if cp1_result is None else getattr(cp1_result, "start_t", None)
        a_exec = getattr(stage3, "action_chunk", stage3)
        stage2 = getattr(self._tl, "stage2", None)
        sample = resume = top1 = None
        if self._diag_mode == "shadow" and stage2 is not None:
            model, device, k_full = self._model, self._stage3_device, self._diag.spec.k_full

            def sample(z: torch.Tensor, k: int):
                if getattr(self._tl, "retrieval_error", None):
                    raise RuntimeError(self._tl.retrieval_error)
                self._tl.in_shadow = True
                try:
                    with torch.no_grad():
                        out = model.run_stage3(stage2, noise=z.to(device)[None, ...], num_steps=int(k))
                finally:
                    self._tl.in_shadow = False
                return out.action_chunk[0]

            def resume(x_t: torch.Tensor, t: float):
                self._tl.in_shadow = True
                try:
                    with torch.no_grad():
                        out = model.run_stage3_from(
                            stage2, x_t.to(device)[None, ...] if x_t.dim() == 2 else x_t.to(device),
                            float(t), num_steps=k_full)
                finally:
                    self._tl.in_shadow = False
                return out.action_chunk[0]

            def top1():
                return self._top1_payload(cp1_result)

        extra = {"top1_score": _score_of(cp1_result), "top1_entry_id": getattr(cp1_result, "entry_id", None)}
        if getattr(self._tl, "self_info", None):
            extra.update(self._tl.self_info)
        self._diag.record(
            a_exec=a_exec, executed_steps=executed_steps, n_stage3_calls=n_calls,
            hit_type=hit_name if hit_name is not None else "MISS", start_t=start_t,
            schedule_id=PI05_V1.schedule_id, sample=sample, resume=resume, top1=top1, extra=extra,
        )

    def _self_start(self, stage2: Any, like: torch.Tensor, start_t: float, num_steps: int) -> torch.Tensor:
        """Self-start ablation: the start the cache arm would have taken, produced on the spot.

        Runs the policy's full ``num_steps`` loop on this decision's stage-2 handle from private noise
        (``DiagSession.self_start_seed``; the global RNG is not touched) and returns its snapshot at
        ``start_t`` (the input of the step at ``start_t``, the cache's snapshot convention) or, for the
        final-start variants, its final action. The direct run is bracketed like a shadow sample: it is
        not counted as an executed stage-3 call, its Euler steps are removed from ``executed_steps`` and
        reported as ``self_direct_nfe``, so the executed continuation keeps the cache arm's accounting.
        """
        seed = self._diag.self_start_seed()
        if seed is None:
            raise RuntimeError("self start outside an episode")
        noise = make_noise(seed, tuple(like.shape[-2:]))[None, ...].to(device=like.device)
        before = int(getattr(self._tl, "n_steps", 0))
        self._tl.in_shadow = True
        try:
            with torch.no_grad():
                out = self._model.run_stage3(stage2, noise=noise, num_steps=int(num_steps), return_intermediates=True,
                                             save_timesteps=(float(start_t),))
        finally:
            self._tl.in_shadow = False
        self._tl.self_info = {"self_start": True, "self_seed": int(seed),
                              "self_direct_nfe": int(getattr(self._tl, "n_steps", 0)) - before}
        self._tl.n_steps = before
        x = out.action_chunk if self._warm_variant in FINAL_START_VARIANTS else out.intermediates[float(start_t)]
        return x.to(device=like.device, dtype=like.dtype).reshape(like.shape)

    def _final_chunk_like(self, start_x: torch.Tensor) -> torch.Tensor:
        """The hit payload's final action chunk (t = 0), shaped like the snapshot it replaces."""
        cp = getattr(self._tl, "cp1_result", None)
        entry_id = None if cp is None else getattr(cp, "entry_id", None)
        orch = self._orchestrator
        if entry_id is None or orch is None:
            raise RuntimeError("reset_final: no WARM_START retrieval result to take the final chunk from")
        payload = orch._storage.fetch_payload(entry_id)  # noqa: SLF001 - same seam as _top1_payload
        chunk = getattr(payload, "action_chunk", None)
        if chunk is None:
            raise RuntimeError(f"reset_final: payload {entry_id} has no action_chunk")
        return chunk.to(device=start_x.device, dtype=start_x.dtype).reshape(start_x.shape)

    def _top1_payload(self, cp1_result) -> tuple:
        """``(score, entry_id, intermediates)`` of the read-only retrieval winner, or Nones."""
        results = None if self._search_rec is None else self._search_rec.last_results
        if not results:
            return (None, None, None)
        top = results[0]
        orch = self._orchestrator
        if orch is None:
            return (float(top.score), top.id, None)
        payload = orch._storage.fetch_payload(top.id)  # noqa: SLF001 - same seam as shadow_teacher
        inter = getattr(payload, "intermediates", None)
        if inter:
            inter = {round(float(t), 4): x for t, x in inter.items()}
        return (float(top.score), top.id, inter or None)


def _score_of(cp1_result) -> Optional[float]:
    s = None if cp1_result is None else getattr(cp1_result, "score", None)
    return None if s is None else float(s)


__all__ = ["Pi05DiagInterceptor", "HitType"]
