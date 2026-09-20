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

from exp.step_diag.recorder import DiagRecorder, EpisodeIdentity

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


class Pi05DiagInterceptor(InferenceInterceptor):
    """See module docstring. ``diag`` is the recorder; ``mode`` is shadow / plain / full / warm;
    ``exec_steps`` (plain / full only) pins the executed step count of this connection."""

    def __init__(self, *args: Any, diag: DiagRecorder, mode: str, exec_steps: Optional[int] = None,
                 **kwargs: Any) -> None:
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

        def _counted_run3f(*a, **kw):
            if not getattr(self._tl, "in_shadow", False):
                self._tl.n_calls = getattr(self._tl, "n_calls", 0) + 1
            return inner_run3f(*a, **kw)

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
        self._diag.record(
            a_exec=a_exec, executed_steps=executed_steps, n_stage3_calls=n_calls,
            hit_type=hit_name if hit_name is not None else "MISS", start_t=start_t,
            schedule_id=PI05_V1.schedule_id, sample=sample, resume=resume, top1=top1, extra=extra,
        )

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
