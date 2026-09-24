"""Cache-aware GR00T inference: drive CacheOrchestrator around the three-stage split.

What this is
------------
The GR00T counterpart of ``openpi.cache.interceptor.InferenceInterceptor``. It
is a parallel implementation rather than a shared base class, for two reasons
that are not stylistic:

* that module imports jax at import time, and the GR00T virtualenv has no jax,
  so it cannot even be loaded there;
* its coordinator routing, sidecar executors, meta-device sentinels and CP3
  handling are all unused here, and a common base would drag them along.

What *is* shared is everything that matters: the orchestrator, the storage
facade, the judges, gates and search strategies are model-agnostic — they only
ever see ``stage1=<opaque>`` forwarded to a KeyBuilder.

Where it sits
-------------
It satisfies the same minimal protocol as a raw GR00T policy (``get_action``),
so it is injected *inside* ``GrootPolicyAdapter`` rather than wrapped around
it. The adapter validates the wire contract; that check belongs outermost, so
a malformed observation is rejected before it can reach the key builder.

Verdict handling is three-way, exactly as on Pi0.5: FULL_HIT replays the
cached chunk, WARM_START resumes the flow-matching loop from the cached
snapshot at ``start_t`` (``GrootStagedRunner.run_stage3_from``), MISS runs the
language model and the full head. The schedule the snapshot is keyed under is
the library's stamp; the runner refuses to resume under any other loop.

Online write-back records the action chunk only. GR00T libraries are built
offline from collected HDF5 (which carries the snapshots); the load guard pins
``write_policy: never`` so an online entry without intermediates can never be
selected for WARM_START.

Trace mode (plan ``logs/cache_trace_mode_plan.log.md`` §9)
---------------------------------------------------------
With a ``TraceRuntime`` injected, ``get_action`` takes ``_get_action_traced``
instead: every module runs on every decision (key, search, judge, stage 1/2,
the full loop from an explicitly drawn noise, every executable warm tier
resumed from the twin top-1, the top-1 replay), the real verdict picks which
of them leaves the server, and everything is recorded. Under ``--concurrent``
the stage-3 variants go through the process-level ``BatchingCore`` with a
``GrootStageBatcher`` (same-shape buckets across connections); the shared
model lock then covers stage 1/2, the noise draw and the payload preparation
only, and is released before the variants are submitted. The legacy
``_get_action_impl`` body is byte-for-byte what it was.

Coupling map:
  DEPENDS ON:  GrootStagedRunner, CacheOrchestrator, a Gr00tPolicy-shaped object,
               openpi.cache.groot.batcher / openpi.cache.trace.groot (trace only)
  CONSUMED BY: GrootPolicyAdapter (as the injected policy)
  IF CHANGED:  the closed-loop smoke gate must be re-run
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Optional

import numpy as np
import torch

from openpi.cache.components.judge import HitType
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID, DenoiseSchedule, schedule_from_id

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Batching helpers (mirror gr00t.model.policy, which we must not import)
# ------------------------------------------------------------------


def _is_batched(obs: dict[str, Any]) -> bool:
    for key, value in obs.items():
        if "state" in key and len(value.shape) < 3:  # (B, Time, Dim)
            return False
    return True


def _unsqueeze_values(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            out[key] = np.expand_dims(value, axis=0)
        elif isinstance(value, list):
            out[key] = np.expand_dims(np.array(value), axis=0)
        elif isinstance(value, torch.Tensor):
            out[key] = value.unsqueeze(0)
        else:
            out[key] = value
    return out


def _squeeze_values(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            out[key] = np.squeeze(value, axis=0)
        elif isinstance(value, torch.Tensor):
            out[key] = value.squeeze(0)
        else:
            out[key] = value
    return out


def _json_or_none(value: Optional[dict]) -> Optional[str]:
    if value is None:
        return None
    import json

    return json.dumps(value, sort_keys=True, default=str)


# ------------------------------------------------------------------
# Interceptor
# ------------------------------------------------------------------


class GrootCacheInterceptor:
    """Drop-in replacement for a GR00T policy that consults the cache at CP1.

    Args:
        policy: a ``Gr00tPolicy``-shaped object. Only its transform pipeline is
            used; inference goes through ``runner`` so the two halves can be
            timed and separated.
        runner: the staged runner wrapping the same policy's model.
        orchestrator: when ``None`` every call runs both stages, which is the
            teacher-only path used for library collection.
        timer: shared with ``runner``; owns the checkpoint and end-to-end
            probes. ``None`` installs a disabled timer.
        trace: a ``TraceRuntime`` switches ``get_action`` to the trace path
            (plan §9). The remaining keyword arguments only exist for it:
        coordinator: the process-level ``BatchingCore`` over a
            ``GrootStageBatcher`` (concurrent trace serving); stage-3 variants
            are then batched across connections. Refused without ``trace``.
        bundle_id: the coordinator bundle this connection submits under.
        trace_vision_fields: camera list in image-token run order, the slice
            layout of the recorded prefix tokens. Must equal the configured CP1
            key builder's ``vision_fields``; required when no CP1 builder can
            name it (CP2 or no library).
        model_lock: the lock shared by every connection's stage 1/2 (trace
            concurrent mode); ``None`` runs unlocked (single connection).
    """

    def __init__(
        self,
        policy: Any,
        runner: GrootStagedRunner,
        *,
        orchestrator: Optional[CacheOrchestrator] = None,
        timer: Optional[SystemTimer] = None,
        trace: Optional[Any] = None,
        coordinator: Optional[Any] = None,
        bundle_id: str = "default",
        trace_vision_fields: Optional[tuple[str, ...]] = None,
        model_lock: Optional[Any] = None,
    ) -> None:
        self._policy = policy
        self._runner = runner
        self._orchestrator = orchestrator
        self._timer = timer if timer is not None else SystemTimer(enabled=False)
        self._timer.register_probe("total_inference", backend="cpu")
        if trace is None and (coordinator is not None or model_lock is not None):
            raise ValueError(
                "coordinator / model_lock are trace-mode arguments: without a "
                "TraceRuntime the legacy path keeps its whole-infer lock"
            )
        self._trace = trace
        twins = getattr(trace, "twins", None) if trace is not None else None
        if orchestrator is not None and twins is not None:
            # The runtime owns the twin component set; the orchestrator was
            # built before it by the serving entry point (plan §4.2).
            orchestrator.attach_trace_twins(twins)
        self._coordinator = coordinator
        self._bundle_id = str(bundle_id)
        self._model_lock = model_lock if model_lock is not None else contextlib.nullcontext()
        self._trace_vision_fields: Optional[tuple[str, ...]] = None
        self._trace_state_index: Optional[torch.Tensor] = None
        if trace is not None:
            self._trace_vision_fields = self._resolve_trace_vision_fields(
                orchestrator, trace_vision_fields
            )
            for probe in ("stage3_trace_full", "stage3_trace_warm"):
                self._timer.register_probe(probe, backend="cpu")
        # CP2-only (ActionCache-style post-backbone arm): frozen from THIS
        # wrapper's orchestrator, so a hot-swapped bundle never changes what an
        # already-bound connection does. The GR00T guard makes {cp1} and {cp2}
        # mutually exclusive, so this is a two-way switch.
        _has_cp = getattr(orchestrator, "has_checkpoint", None)
        self._cp2_only = bool(
            orchestrator is not None and _has_cp is not None and _has_cp(CheckpointID.CP2)
        )
        _meta = getattr(orchestrator, "artifact_meta", None) if self._cp2_only else None
        self._cp2_library_sha256 = (
            _meta.get("library_sha256") if isinstance(_meta, dict) else None
        )
        if orchestrator is not None:
            self._timer.register_probe("cp2_sum" if self._cp2_only else "cp1_sum", backend="cpu")

    # -- TaskLifecycle ---------------------------------------------------

    def on_task_begin(self) -> None:
        self._timer.on_task_begin()
        if self._orchestrator is not None:
            self._orchestrator.on_task_begin()

    def on_episode_start(
        self,
        experiment: str = "",
        task: str = "",
        episode_id: int = -1,
        episode_name: str = "",
        extra_metadata: dict | None = None,
    ) -> None:
        trace = getattr(self, "_trace", None)
        if trace is not None:
            from openpi.cache.trace.types import EpisodeIdentity

            # The live loop must still be the schedule the plan was built for:
            # every warm tier and every snapshot name is keyed by it.
            live = self._runner.live_schedule()
            if live != trace.plan.schedule:
                raise RuntimeError(
                    f"trace: action head runs {live.schedule_id} but the trace plan "
                    f"was built for {trace.plan.schedule.schedule_id}"
                )
            self._trace_state_index = None
            trace.sink.on_episode_start(
                EpisodeIdentity(
                    experiment=str(experiment),
                    task=str(task),
                    episode_id=int(episode_id),
                    episode_name=str(episode_name),
                    extra_metadata=dict(extra_metadata or {}),
                )
            )
        del experiment, episode_name  # accepted for signature parity
        if self._orchestrator is not None:
            self._orchestrator.on_episode_start(
                task_key=task,
                episode_id=str(episode_id),
                extra_metadata=extra_metadata,
            )

    def on_episode_end(self, success: bool) -> None:
        # `success` is dropped by the orchestrator: its episode hook takes no
        # outcome. It still must be called even under write_policy=never,
        # because closing the search session happens in its finally block.
        if self._orchestrator is not None:
            self._orchestrator.on_episode_end()
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace.sink.on_episode_end(bool(success))
        self._timer.on_task_end()
        self._timer.on_task_begin()

    def on_task_end(self) -> None:
        self._timer.on_task_end()
        if self._orchestrator is not None:
            self._orchestrator.on_task_end()
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace.sink.on_task_end()

    @property
    def traced(self) -> bool:
        """Whether this interceptor serves the trace path."""
        return getattr(self, "_trace", None) is not None

    def close_trace_episode(self) -> None:
        """Close the trace episode as non-terminal and release its twin sessions.

        For a server whose untraced stack never receives the connection
        lifecycle (the GR00T LIBERO adapter): a dropped connection must not
        leave its trace episode open, and the real components must not get a
        lifecycle call the untraced server would not make.
        """
        trace = getattr(self, "_trace", None)
        if trace is not None:
            try:
                trace.sink.on_task_end()
            finally:
                if self._orchestrator is not None:
                    self._orchestrator.close_trace_search_sessions()

    # -- observability ---------------------------------------------------

    @staticmethod
    def _build_hit_meta(
        cp1_result,
        *,
        checkpoint: Optional[str] = None,
        library_sha256: Optional[str] = None,
        online_rit: Optional[dict] = None,
    ) -> dict:
        """Same field set as the Pi0.5 interceptor, so one analysis path reads both.

        ``start_t`` is the real resume point on a WARM_START and ``None``
        otherwise; downstream cost summaries price a warm start by it, so a
        placeholder here would silently mis-price every warm step.

        Two additive fields mirror the Pi0.5 wire: ``checkpoint`` names the
        verdict's checkpoint (``"CP1"`` / ``"CP2"``, ``None`` when no
        orchestrator ran) and ``score`` is the fused score whatever the
        checkpoint. ``cp1_score`` keeps its legacy meaning -- filled on CP1,
        ``None`` on CP2. On CP2 the loaded library's ``library_sha256`` rides
        along (absent, not None, elsewhere) so the client ledger can prove
        which artifact the server searched.
        """
        if cp1_result is None:
            return {
                "hit_type": "MISS",
                "start_t": None,
                "winner_id": None,
                "cp1_score": None,
                "searched": True,
                "checkpoint": None,
                "score": None,
            }
        cp_name = "CP1" if checkpoint is None else checkpoint
        meta = {
            "hit_type": cp1_result.hit_type.name,
            "start_t": (
                cp1_result.start_t
                if cp1_result.hit_type == HitType.WARM_START
                else None
            ),
            "winner_id": cp1_result.entry_id,
            "cp1_score": cp1_result.score if cp_name == "CP1" else None,
            "searched": cp1_result.searched,
            "checkpoint": cp_name,
            "score": cp1_result.score,
        }
        if library_sha256 is not None:
            meta["library_sha256"] = library_sha256
        if online_rit is not None:
            # Additive, present only when the served judge is ``online_rit``:
            # the decision snapshot (q_pre / cuts) and this step's feedback.
            meta["online_rit"] = online_rit
        return meta

    # -- inference -------------------------------------------------------

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Run inference and invalidate an online stream on any failed request."""
        try:
            return self._get_action_impl(observations)
        except Exception:
            orchestrator = self._orchestrator
            if orchestrator is not None and hasattr(orchestrator, "continuation_spec"):
                try:
                    if orchestrator.continuation_spec(CheckpointID.CP1) is not None:
                        orchestrator.record_continuation(
                            CheckpointID.CP1, None, [], invalid_reasons=["inference request failed"]
                        )
                    # The twin stream (trace mode) is invalidated the same way.
                    twin_spec = getattr(orchestrator, "twin_continuation_spec", None)
                    if twin_spec is not None and twin_spec(CheckpointID.CP1) is not None:
                        orchestrator.twin_record_continuation(
                            CheckpointID.CP1, None, [], invalid_reasons=["inference request failed"]
                        )
                except Exception:
                    logger.exception("Could not persist the invalid online stream after inference failure")
            raise

    def _get_action_impl(self, observations: dict[str, Any]) -> dict[str, Any]:
        """One cache-aware inference cycle.

        Mirrors ``Gr00tPolicy.get_action`` step for step so that, with the
        orchestrator disabled, the numbers are the unsplit model's. The cache
        check sits between the two stages, deliberately outside the inference
        context: tensors born inside it stay inference tensors even after a
        `.cpu()`, and the cache keeps them across steps.
        """
        if getattr(self, "_trace", None) is not None:
            return self._get_action_traced(observations)

        with self._timer.measure("total_inference"):
            obs_copy = observations.copy()
            is_batch = _is_batched(obs_copy)
            if not is_batch:
                obs_copy = _unsqueeze_values(obs_copy)
            for key, value in obs_copy.items():
                if not isinstance(value, np.ndarray):
                    obs_copy[key] = np.array(value)

            normalized_input = self._policy.apply_transforms(obs_copy)

            if self._cp2_only:
                return self._get_action_cp2(normalized_input, is_batch)

            with self._runner.session():
                stage1 = self._runner.run_stage1(normalized_input)

            cp1_result = None
            online_diag: Optional[dict] = None
            try:
                if self._orchestrator is not None:
                    with self._timer.measure("cp1_sum"):
                        cp1_result = self._orchestrator.check(
                            CheckpointID.CP1, stage1=stage1
                        )
                # Online RIT feedback contract; ``None`` for every other judge,
                # in which case every branch below is the legacy path verbatim.
                spec = (
                    self._orchestrator.continuation_spec(CheckpointID.CP1)
                    if self._orchestrator is not None
                    and hasattr(self._orchestrator, "continuation_spec")
                    else None
                )

                hit_type = None if cp1_result is None else cp1_result.hit_type
                if hit_type == HitType.FULL_HIT:
                    chunk = cp1_result.payload.action_chunk
                elif hit_type == HitType.WARM_START:
                    payload = cp1_result.payload
                    start_t = cp1_result.start_t
                    # The orchestrator already proved start_t is a key of the
                    # payload; the schedule comes from the library, and the
                    # runner refuses it unless the head is running that loop.
                    schedule = self._library_schedule(payload)
                    with self._runner.session():
                        stage2 = self._runner.run_stage2_llm(stage1)
                        # The capture kwarg rides only for the online judge so
                        # every legacy call (and its test spies) stays verbatim.
                        capture = {"capture_first_step": True} if spec is not None else {}
                        out = self._runner.run_stage3_from(
                            stage2,
                            payload.intermediates[start_t],
                            start_t,
                            schedule=schedule,
                            **capture,
                        )
                        chunk = out.action_pred
                        if spec is not None:
                            online_diag = self._continuation_feedback(
                                spec,
                                stage2,
                                payload,
                                schedule,
                                executed=(start_t, out.first_step_input, out.first_step_x),
                            )
                elif (
                    spec is not None
                    and cp1_result is not None
                    and cp1_result.entry_id is not None
                    and getattr(cp1_result, "searched", True)
                ):
                    # MISS with a candidate under the online judge: the same
                    # two calls ``run_stage2`` makes (staged.py run_stage2),
                    # split so the backbone output is in hand for the side
                    # evaluation of every tier from the candidate's snapshots.
                    with self._runner.session():
                        stage2 = self._runner.run_stage2_llm(stage1)
                        chunk = self._runner.run_stage3(stage2).action_pred
                        payload = self._orchestrator.peek_payload(cp1_result.entry_id)
                        schedule = self._library_schedule(payload)
                        online_diag = self._continuation_feedback(
                            spec, stage2, payload, schedule, executed=None
                        )
                else:
                    with self._runner.session():
                        chunk = self._runner.run_stage2(stage1).action_pred
                    if spec is not None and cp1_result is not None:
                        # No candidate or gate skip: nothing to compare, but the
                        # decision still gets its (empty) feedback row.
                        online_diag = self._orchestrator.record_continuation(
                            CheckpointID.CP1, None, []
                        )

                action_cpu = self._to_storage_tensor(chunk)

                if self._orchestrator is not None:
                    self._orchestrator.broadcast_action(action_cpu)
                    if cp1_result.query_keys is not None:
                        self._orchestrator.buffer_for_write(
                            cp1_result.query_keys, action_cpu
                        )
            finally:
                if self._orchestrator is not None:
                    self._orchestrator.clear()

            unnormalized = self._policy.unapply_transforms(
                {"action": action_cpu[None, ...]}
            )
            if not is_batch:
                unnormalized = _squeeze_values(unnormalized)

        unnormalized["__hit_meta__"] = self._build_hit_meta(cp1_result, online_rit=online_diag)
        return unnormalized

    def _continuation_feedback(self, spec, stage2, payload, schedule, *, executed) -> dict:
        """Turn this decision's first-step updates into feedback and commit it.

        ``executed`` is ``(start_t, x_in, x_out)`` of the warm start that ran
        (its first step is free) or ``None`` on a MISS. Under ``fm1`` every
        other tier is side-evaluated in one batched step from the candidate's
        stored snapshots; under ``fm0`` only the executed tier contributes.
        Runs inside the runner session (the side step needs the same autocast
        as the loop). A non-finite disagreement is counted, never learned.
        """
        from openpi.cache.components.online_rit import feedback_from_updates

        executed_index = None
        if executed is not None:
            executed_index = spec.tier_by_start_t(executed[0]).index
        side: list = []
        batch = 0
        if spec.feedback_mode == "fm1":
            others = [t for t in spec.tiers if t.index != executed_index]
            snaps = []
            for t in others:
                payload.validate_for_warm_start(schedule, t.start_t)
                snaps.append((t.start_t, payload.intermediates[t.start_t]))
            pairs = self._runner.first_step_updates(stage2, snaps, schedule=schedule)
            batch = len(snaps)
            side = [(t.start_t, x_in, x_out) for t, (x_in, x_out) in zip(others, pairs)]
        feedback, reasons = feedback_from_updates(spec, payload, schedule, executed=executed, side=side)
        snapshot = self._orchestrator.pending_decision(CheckpointID.CP1)
        return self._orchestrator.record_continuation(
            CheckpointID.CP1,
            snapshot,
            feedback,
            n_rejected=len(reasons),
            fb_batch_size=batch,
            invalid_reasons=reasons,
        )

    def _get_action_cp2(self, normalized_input: dict, is_batch: bool) -> dict[str, Any]:
        """The CP2-only decision cycle (ActionCache-style arm, plan §3.1).

        stage 1 -> stage 2 (LM only) -> encoded key source, all in one session;
        exactly one ``check(CP2)`` per decision, no CP1 and no CP3 probe.
        FULL_HIT replays the cached chunk without touching the action head's
        denoise loop; WARM_START resumes ``run_stage3_from`` at the library
        snapshot; MISS runs upstream's full ``get_action`` from the already
        computed stage-2 output (``run_stage3``), the same numbers as
        ``run_stage2`` would give.
        """
        with self._runner.session():
            stage1 = self._runner.run_stage1(normalized_input)
            stage2 = self._runner.run_stage2_llm(stage1)
            source = self._runner.run_cp2_key_source(stage2)

        cp2_result = None
        try:
            with self._timer.measure("cp2_sum"):
                cp2_result = self._orchestrator.check(
                    CheckpointID.CP2, stage2=stage2, cp2_source=source
                )
            hit_type = cp2_result.hit_type
            if hit_type == HitType.FULL_HIT:
                chunk = cp2_result.payload.action_chunk
            elif hit_type == HitType.WARM_START:
                payload = cp2_result.payload
                start_t = cp2_result.start_t
                schedule = self._library_schedule(payload)
                with self._runner.session():
                    chunk = self._runner.run_stage3_from(
                        stage2,
                        payload.intermediates[start_t],
                        start_t,
                        schedule=schedule,
                    ).action_pred
            else:
                with self._runner.session():
                    chunk = self._runner.run_stage3(stage2).action_pred

            action_cpu = self._to_storage_tensor(chunk)
            self._orchestrator.broadcast_action(action_cpu)
            if cp2_result.query_keys is not None:
                self._orchestrator.buffer_for_write(cp2_result.query_keys, action_cpu)
        finally:
            source = None
            self._orchestrator.clear()

        unnormalized = self._policy.unapply_transforms({"action": action_cpu[None, ...]})
        if not is_batch:
            unnormalized = _squeeze_values(unnormalized)
        unnormalized["__hit_meta__"] = self._build_hit_meta(
            cp2_result, checkpoint="CP2", library_sha256=self._cp2_library_sha256
        )
        return unnormalized

    # -- trace mode (plan §9) --------------------------------------------

    @staticmethod
    def _resolve_trace_vision_fields(
        orchestrator: Optional[Any], requested: Optional[tuple[str, ...]]
    ) -> tuple[str, ...]:
        """The camera layout the recorded prefix tokens are sliced by.

        The configured CP1 builder is the authority when there is one; an
        explicit ``trace_vision_fields`` must agree with it. Without a CP1
        builder (CP2 recipe, no library) the caller must name the served
        camera mapping -- guessing three cameras on a two-camera LIBERO
        checkpoint would reject every observation at the first step.
        """
        builder = getattr(orchestrator, "key_builder", None) if orchestrator is not None else None
        configured = getattr(builder, "vision_fields", None)
        if configured is not None:
            configured = tuple(configured)
            if requested is not None and tuple(requested) != configured:
                raise ValueError(
                    f"trace_vision_fields {tuple(requested)} disagree with the configured "
                    f"key builder's {configured}"
                )
            return configured
        if requested is None:
            raise ValueError(
                "trace_vision_fields is required: no CP1 key builder names the served "
                "camera layout (CP2 recipe or no library)"
            )
        return tuple(requested)

    def _get_action_traced(self, observations: dict[str, Any]) -> dict[str, Any]:
        """One decision with every module running, the verdict's arm served (plan §9).

        Order of operations under the shared model lock: apply_transforms ->
        stage 1 -> check
        (CP1, or stage-2 LLM + CP2 key source -> check(CP2)) -> stage-2 LLM
        -> noise draw -> variant table (top-1 snapshots moved to the device)
        -> host bucket identity + ready events. The lock is released before
        the variants are run (coordinator: one ``submit_many``; direct: same
        thread). The continuation feedback of the real and the twin judge and
        the legacy bookkeeping follow, then the record.
        """
        from openpi.cache.trace import groot as _tg
        from openpi.cache.trace.records import search_trace_from_check
        from openpi.serving.batching_core import record_ready_events

        tr = self._trace
        plan = tr.plan
        sink = tr.sink
        sink.begin_step()
        timing: dict[str, float] = {}
        t_wall = time.perf_counter()
        try:
            with self._timer.measure("total_inference"):
                obs_copy = observations.copy()
                is_batch = _is_batched(obs_copy)
                if not is_batch:
                    obs_copy = _unsqueeze_values(obs_copy)
                for key, value in obs_copy.items():
                    if not isinstance(value, np.ndarray):
                        obs_copy[key] = np.array(value)
                raw_images, prompt_text, raw_state, raw_layout = _tg.capture_raw_observation(
                    obs_copy
                )
                cp = None
                cond = None
                events: tuple = ()
                with self._model_lock:
                    # The policy's transform chain is shared by every connection
                    # and is not thread-safe (einops registers its backends
                    # lazily); the legacy path runs it under the whole-infer
                    # lock, so it stays under the shared lock here too.
                    normalized_input = self._policy.apply_transforms(obs_copy)
                    tokenized = _tg.tokenized_prompt(normalized_input)
                    t0 = time.perf_counter()
                    with self._runner.session():
                        stage1 = self._runner.run_stage1(normalized_input)
                    timing["stage1_ms"] = (time.perf_counter() - t0) * 1000.0

                    if self._cp2_only:
                        t0 = time.perf_counter()
                        with self._runner.session():
                            stage2 = self._runner.run_stage2_llm(stage1)
                            source = self._runner.run_cp2_key_source(stage2)
                        timing["stage2_ms"] = (time.perf_counter() - t0) * 1000.0
                        t0 = time.perf_counter()
                        with self._timer.measure("cp2_sum"):
                            cp = self._orchestrator.check(
                                CheckpointID.CP2,
                                trace=True,
                                fetch_top1=plan.fetch_top1,
                                stage2=stage2,
                                cp2_source=source,
                            )
                        source = None
                        timing["cp2_ms"] = (time.perf_counter() - t0) * 1000.0
                    else:
                        if self._orchestrator is not None:
                            t0 = time.perf_counter()
                            with self._timer.measure("cp1_sum"):
                                cp = self._orchestrator.check(
                                    CheckpointID.CP1,
                                    trace=True,
                                    fetch_top1=plan.fetch_top1,
                                    stage1=stage1,
                                )
                            timing["cp1_ms"] = (time.perf_counter() - t0) * 1000.0
                        t0 = time.perf_counter()
                        with self._runner.session():
                            stage2 = self._runner.run_stage2_llm(stage1)
                        timing["stage2_ms"] = (time.perf_counter() - t0) * 1000.0

                    effective_hit = cp.hit_type if cp is not None else HitType.MISS

                    # Noise for the full loop: upstream's own draw on a MISS
                    # (global RNG, as the legacy path consumes it), a private
                    # identity-seeded generator on a hit so the extra inference
                    # never shifts the global stream (plan §9-4).
                    gen = None
                    if effective_hit != HitType.MISS:
                        gen = tr.noise_generator(
                            device=stage2.backbone_features.device,
                            identity=sink.identity,
                            step_idx=sink.step_idx,
                        )
                    with self._runner.session():
                        z = self._runner.sample_noise(stage2, generator=gen)

                    capture_first = self._continuation_wanted()
                    variants, tier_status, warm_index_map, warm_exec_name = (
                        self._trace_variants(stage2, cp, z=z, capture_first_step=capture_first)
                    )
                    if self._coordinator is not None:
                        from openpi.cache.groot.batcher import stage3_input

                        cond = stage3_input(self._runner, stage2)
                        events = record_ready_events(z.device)

                t0 = time.perf_counter()
                outs = self._run_trace_variants(stage2, cond, variants, events)
                timing["stage3_variants_ms"] = (time.perf_counter() - t0) * 1000.0
                full = outs["full"]
                executed_arm, executed = self._select_executed(cp, outs, warm_exec_name)

                online_diag: Optional[dict] = None
                twin_diag: Optional[dict] = None
                try:
                    if self._orchestrator is not None:
                        with self._model_lock:
                            online_diag = self._trace_real_feedback(cp, stage2, executed)
                            twin_diag = self._trace_twin_feedback(cp, stage2, outs)
                    if effective_hit == HitType.FULL_HIT:
                        chunk = cp.payload.action_chunk
                    else:
                        chunk = executed.action
                    action_cpu = self._to_storage_tensor(chunk)
                    if self._orchestrator is not None:
                        self._orchestrator.broadcast_action(action_cpu)
                        if cp is not None and cp.query_keys is not None:
                            self._orchestrator.buffer_for_write(cp.query_keys, action_cpu)
                finally:
                    if self._orchestrator is not None:
                        self._orchestrator.clear()

                with self._model_lock:
                    unnormalized = self._policy.unapply_transforms(
                        {"action": action_cpu[None, ...]}
                    )
                if not is_batch:
                    unnormalized = _squeeze_values(unnormalized)

            if self._cp2_only:
                hit_meta = self._build_hit_meta(
                    cp, checkpoint="CP2", library_sha256=self._cp2_library_sha256
                )
            else:
                hit_meta = self._build_hit_meta(cp, online_rit=online_diag)
            unnormalized["__hit_meta__"] = dict(hit_meta)
            timing["wall_ms"] = (time.perf_counter() - t_wall) * 1000.0

            # ---- record ----
            ct = getattr(cp, "trace", None) if cp is not None else None
            top1 = ct.top1_payload if ct is not None else None
            full_action = full.action[0].detach().cpu().float().numpy()
            action_full_hit = None
            if top1 is not None and getattr(top1, "action_chunk", None) is not None:
                action_full_hit = top1.action_chunk.detach().cpu().float().numpy()
            action_warm = {
                idx: (outs[name].action[0].detach().cpu().float().numpy() if name in outs else None)
                for idx, name in warm_index_map.items()
            }
            action_warm_exec = (
                outs[warm_exec_name].action[0].detach().cpu().float().numpy()
                if warm_exec_name is not None
                else None
            )
            step = _tg.build_step_trace(
                plan=plan,
                stage1=stage1,
                vision_fields=self._trace_vision_fields,
                expected_state_index=self._trace_state_index,
                raw_images=raw_images,
                prompt=prompt_text,
                raw_state=raw_state,
                raw_state_layout=raw_layout,
                tokenized=tokenized,
                query_keys=cp.query_keys if cp is not None else None,
                search=search_trace_from_check(ct),
                full_action=full_action,
                full_output=full,
                action_full_hit=action_full_hit,
                action_warm=action_warm,
                action_warm_exec=action_warm_exec,
                action_executed=action_cpu.numpy(),
                executed_arm=executed_arm,
                verdict=hit_meta,
                tier_status=tier_status,
                timing_ms=timing,
                warm_index_map={
                    idx: {
                        "arm": name,
                        "start_t": plan.schedule.snapshot_t(idx),
                        "schedule_id": plan.schedule.schedule_id,
                        "entry_id": ct.top1_entry_id if ct is not None else None,
                    }
                    for idx, name in warm_index_map.items()
                },
                real_continuation_json=_json_or_none(online_diag),
                twin_continuation_json=_json_or_none(twin_diag),
            )
            if self._trace_state_index is None and plan.record_prefix_tokens:
                self._trace_state_index = stage1.state_mask[0, -1].clone()
            writer_error = None
            try:
                sink.record_step(step)
            except Exception as exc:
                if plan.fail_loud:
                    raise
                writer_error = repr(exc)
                logger.warning("trace: record_step failed (diagnostic mode): %r", exc)
            unnormalized["__hit_meta__"]["trace"] = {
                "executed_arm": executed_arm,
                "top1_entry_id": ct.top1_entry_id if ct is not None else None,
                "tier_status": dict(tier_status),
                **({"writer_error": writer_error} if writer_error else {}),
            }
            return unnormalized
        finally:
            sink.finish_step()

    def _continuation_wanted(self) -> bool:
        """Whether any online judge (real or twin) will read first-step captures."""
        orch = self._orchestrator
        if orch is None or self._cp2_only:
            return False
        spec = getattr(orch, "continuation_spec", None)
        twin = getattr(orch, "twin_continuation_spec", None)
        return bool(
            (spec is not None and spec(CheckpointID.CP1) is not None)
            or (twin is not None and twin(CheckpointID.CP1) is not None)
        )

    def _trace_variants(self, stage2, cp, *, z: torch.Tensor, capture_first_step: bool):
        """Variant table for one decision (plan §5 step 9, GR00T flavour).

        ``full`` runs the whole loop from ``z``; every executable warm tier
        resumes from the twin top-1's snapshot; ``warm_exec`` guarantees a real
        WARM_START verdict whose ``(winner, start_t)`` is not among them still
        has an arm to serve. The real payload passes the real check
        (``_library_schedule`` + ``validate_for_warm_start``), as on the
        legacy path.
        """
        from openpi.cache.trace.types import ARM_WARM_EXEC, warm_arm_name

        plan = self._trace.plan
        schedule = plan.schedule
        ct = getattr(cp, "trace", None) if cp is not None else None
        top1 = ct.top1_payload if ct is not None else None
        top1_id = ct.top1_entry_id if ct is not None else None
        device = stage2.backbone_features.device
        expected = tuple(z.shape[1:])
        variants: list[tuple[str, str, dict]] = []
        tier_status: dict[str, str] = {}
        warm_index_map: dict[int, str] = {}

        self._precheck_variant_tensor(z[0], "noise", expected, device, dtype=None)
        variants.append(
            (
                "full",
                "miss",
                {"noise": z, "num_steps": schedule.num_steps, "save_timesteps": plan.save_timesteps},
            )
        )
        computed: set[tuple[Optional[str], float]] = set()
        for idx in plan.warm_tiers:
            name = warm_arm_name(idx)
            t = schedule.snapshot_t(idx)
            if top1 is None:
                tier_status[name] = "no_top1"
                continue
            try:
                top1.validate_for_warm_start(schedule, t)
            except (KeyError, ValueError, AttributeError) as exc:
                tier_status[name] = f"no_snapshot:{type(exc).__name__}"
                continue
            start_x = top1.intermediates[t].to(device)
            if start_x.dim() == 3:
                start_x = start_x[0]
            self._precheck_variant_tensor(start_x, name, expected, device, dtype=torch.float32)
            variants.append(
                (
                    name,
                    "warm",
                    {
                        "start_x": start_x,
                        "start_t": t,
                        "num_steps": schedule.num_steps,
                        "capture_first_step": capture_first_step,
                    },
                )
            )
            warm_index_map[idx] = name
            tier_status[name] = "ok"
            computed.add((top1_id, round(float(t), 4)))

        warm_exec_name = None
        if cp is not None and cp.hit_type == HitType.WARM_START:
            start_t = cp.start_t
            lib_schedule = self._library_schedule(cp.payload)
            cp.payload.validate_for_warm_start(lib_schedule, start_t)
            key = (cp.entry_id, round(float(start_t), 4))
            if key not in computed:
                start_x = cp.payload.intermediates[start_t].to(device)
                if start_x.dim() == 3:
                    start_x = start_x[0]
                self._precheck_variant_tensor(
                    start_x, ARM_WARM_EXEC, expected, device, dtype=torch.float32
                )
                variants.append(
                    (
                        ARM_WARM_EXEC,
                        "warm",
                        {
                            "start_x": start_x,
                            "start_t": start_t,
                            "num_steps": lib_schedule.num_steps,
                            "capture_first_step": capture_first_step,
                        },
                    )
                )
                warm_exec_name = ARM_WARM_EXEC
                reason = "winner_differs_from_top1" if cp.entry_id != top1_id else "tier_not_enumerated"
                tier_status[ARM_WARM_EXEC] = f"warm_exec:{reason}"
        return variants, tier_status, warm_index_map, warm_exec_name

    @staticmethod
    def _precheck_variant_tensor(
        t: torch.Tensor, name: str, expected: tuple, device, *, dtype
    ) -> None:
        """Refuse a malformed variant input on THIS connection, before it is shared."""
        if tuple(t.shape) != tuple(expected):
            raise ValueError(
                f"trace variant {name!r}: expected shape {tuple(expected)}, got {tuple(t.shape)}"
            )
        if dtype is not None and t.dtype != dtype:
            raise ValueError(f"trace variant {name!r}: expected {dtype}, got {t.dtype}")
        if str(t.device) != str(torch.device(device)):
            raise ValueError(f"trace variant {name!r}: expected device {device}, got {t.device}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"trace variant {name!r}: non-finite values")

    def _run_trace_variants(self, stage2, cond, variants, events) -> dict[str, Any]:
        """Run every variant: one ``submit_many`` under the coordinator, direct
        runner calls otherwise. Returns ``{name: Stage3VariantOutput}``."""
        from openpi.cache.groot import batcher as _gb
        from openpi.serving.batching_core import Stage3MissPayload, Stage3WarmStartPayload

        outs: dict[str, Any] = {}
        if self._coordinator is not None:
            payloads = []
            for name, kind, spec in variants:
                if kind == "miss":
                    payloads.append(
                        Stage3MissPayload(
                            stage2_out=cond,
                            noise=spec["noise"][0],
                            num_steps=spec["num_steps"],
                            save_timesteps=spec["save_timesteps"],
                            ready_events=events,
                        )
                    )
                else:
                    payloads.append(
                        Stage3WarmStartPayload(
                            stage2_out=cond,
                            start_x=spec["start_x"],
                            start_t=spec["start_t"],
                            num_steps=spec["num_steps"],
                            capture_first_step=spec["capture_first_step"],
                            ready_events=events,
                        )
                    )
            with self._timer.measure("stage3_trace_full"):
                results = self._coordinator.submit_many_to_stage(3, self._bundle_id, payloads)
            for (name, _kind, _spec), out in zip(variants, results, strict=True):
                outs[name] = out
            return outs
        schedule = self._trace.plan.schedule
        for name, kind, spec in variants:
            if kind == "miss":
                with self._timer.measure("stage3_trace_full"):
                    out, caps = _gb.run_miss(
                        self._runner,
                        stage2,
                        spec["noise"],
                        schedule=schedule,
                        capture=spec["save_timesteps"] is not None,
                    )
                outs[name] = _gb.split_miss(
                    out, caps, schedule=schedule, save_timesteps_per_request=[spec["save_timesteps"]]
                )[0]
            else:
                with self._timer.measure("stage3_trace_warm"):
                    out = _gb.run_warm(
                        self._runner,
                        stage2,
                        spec["start_x"][None, ...],
                        spec["start_t"],
                        schedule=schedule,
                        capture_first_step=spec["capture_first_step"],
                    )
                outs[name] = _gb.split_warm(out, 1)[0]
        return outs

    def _select_executed(self, cp, outs: dict[str, Any], warm_exec_name):
        """``(executed_arm, Stage3VariantOutput | None)`` for the real verdict."""
        from openpi.cache.trace.types import ARM_FULL_HIT, ARM_FULL_INFERENCE, warm_arm_name

        if cp is None or cp.hit_type == HitType.MISS:
            return ARM_FULL_INFERENCE, outs["full"]
        if cp.hit_type == HitType.FULL_HIT:
            return ARM_FULL_HIT, None
        if warm_exec_name is not None:
            return warm_exec_name, outs[warm_exec_name]
        name = warm_arm_name(self._trace.plan.schedule.snapshot_index(cp.start_t))
        if name not in outs:
            raise RuntimeError(
                f"trace: WARM_START verdict start_t={cp.start_t} has no computed variant "
                f"(have {sorted(outs)})"
            )
        return name, outs[name]

    def _trace_real_feedback(self, cp, stage2, executed) -> Optional[dict]:
        """The real online judge's feedback, under the legacy conditions (plan §9-5).

        WARM_START: the executed pair is the served variant's captured first
        step (fm0: only that; fm1: the other tiers side-evaluated by the
        original helper). MISS with a candidate: the candidate's payload is
        side-evaluated as before. No candidate / gate skip: the empty row.
        FULL_HIT: nothing, as on the legacy path.
        """
        orch = self._orchestrator
        if self._cp2_only or cp is None or not hasattr(orch, "continuation_spec"):
            return None
        spec = orch.continuation_spec(CheckpointID.CP1)
        if spec is None:
            return None
        hit = cp.hit_type
        if hit == HitType.WARM_START:
            payload = cp.payload
            schedule = self._library_schedule(payload)
            with self._runner.session():
                return self._continuation_feedback(
                    spec,
                    stage2,
                    payload,
                    schedule,
                    executed=(cp.start_t, executed.first_step_input, executed.first_step_x),
                )
        if hit == HitType.FULL_HIT:
            return None
        if cp.entry_id is not None and getattr(cp, "searched", True):
            payload = orch.peek_payload(cp.entry_id)
            schedule = self._library_schedule(payload)
            with self._runner.session():
                return self._continuation_feedback(spec, stage2, payload, schedule, executed=None)
        return orch.record_continuation(CheckpointID.CP1, None, [])

    def _trace_twin_feedback(self, cp, stage2, outs: dict[str, Any]) -> Optional[dict]:
        """Close the twin online judge's decision with the same contract (plan §9-6).

        The twin candidate is the twin top-1, whose first-step pairs the warm
        variants already captured under the same ``(entry, tier, schedule)``;
        those are reused, any tier without a computed variant is side-evaluated
        once, and a tier whose snapshot is missing is recorded as rejected.
        """
        from openpi.cache.components.online_rit import feedback_from_updates
        from openpi.cache.trace.types import warm_arm_name

        orch = self._orchestrator
        if self._cp2_only or cp is None or not hasattr(orch, "twin_continuation_spec"):
            return None
        spec = orch.twin_continuation_spec(CheckpointID.CP1)
        ct = getattr(cp, "trace", None)
        if spec is None or ct is None:
            return None
        pending = orch.twin_pending_decision(CheckpointID.CP1)
        top1 = ct.top1_payload
        if pending is None or top1 is None:
            return orch.twin_record_continuation(CheckpointID.CP1, None, [])
        verdict = ct.twin_verdict
        hit = getattr(verdict, "hit_type", HitType.MISS)
        if hit == HitType.FULL_HIT:
            return None
        schedule = self._trace.plan.schedule

        def _pair(t: float):
            out = outs.get(warm_arm_name(schedule.snapshot_index(t)))
            if out is None or out.first_step_input is None or out.first_step_x is None:
                return None
            return out.first_step_input, out.first_step_x

        executed = None
        executed_index = None
        reasons: list[str] = []
        to_evaluate: list[tuple[float, torch.Tensor]] = []
        if hit == HitType.WARM_START:
            t = verdict.start_t
            executed_index = spec.tier_by_start_t(t).index
            pair = _pair(t)
            if pair is not None:
                executed = (t, pair[0], pair[1])
            else:
                to_evaluate.append((t, top1.intermediates[t]))
        side: list[tuple[float, torch.Tensor, torch.Tensor]] = []
        if spec.feedback_mode == "fm1":
            for tier in spec.tiers:
                if tier.index == executed_index:
                    continue
                pair = _pair(tier.start_t)
                if pair is not None:
                    side.append((tier.start_t, pair[0], pair[1]))
                    continue
                try:
                    top1.validate_for_warm_start(schedule, tier.start_t)
                except (KeyError, ValueError, AttributeError) as exc:
                    reasons.append(f"twin:{tier.name}:{exc}")
                    continue
                to_evaluate.append((tier.start_t, top1.intermediates[tier.start_t]))
        batch = 0
        if to_evaluate:
            with self._runner.session():
                pairs = self._runner.first_step_updates(stage2, to_evaluate, schedule=schedule)
            batch = len(to_evaluate)
            for (t, _x), (x_in, x_out) in zip(to_evaluate, pairs, strict=True):
                if executed is None and hit == HitType.WARM_START and t == verdict.start_t:
                    executed = (t, x_in, x_out)
                else:
                    side.append((t, x_in, x_out))
        feedback, fb_reasons = feedback_from_updates(
            spec, top1, schedule, executed=executed, side=side
        )
        reasons.extend(fb_reasons)
        return orch.twin_record_continuation(
            CheckpointID.CP1,
            pending,
            feedback,
            n_rejected=len(reasons),
            fb_batch_size=batch,
            invalid_reasons=reasons,
        )

    def _library_schedule(self, payload) -> DenoiseSchedule:
        """Resolve the loop a payload's snapshots were taken from.

        The artifact-level ``schedule_id`` is the authority when the storage
        exposes one; the entry's own ``denoising_num_steps`` must agree with it.
        Without artifact metadata the entry must still carry its own explicit
        identity. The runner checks that identity against the live head.
        """
        meta = getattr(self._orchestrator, "artifact_meta", None) or {}
        library_id = meta.get("schedule_id")
        payload_id = payload.schedule_id
        if library_id is not None and payload_id != library_id:
            raise RuntimeError(
                f"WARM_START payload is stamped {payload_id!r} but its library is "
                f"stamped {library_id!r}."
            )
        schedule_id = library_id or payload_id
        if schedule_id is None:
            raise RuntimeError("WARM_START payload and library carry no schedule_id")
        schedule = schedule_from_id(schedule_id)
        if payload.denoising_num_steps != schedule.num_steps:
            raise RuntimeError(
                "WARM_START payload carries denoising_num_steps="
                f"{payload.denoising_num_steps} but its schedule is {schedule_id} "
                f"({schedule.num_steps} steps)."
            )
        return schedule

    @staticmethod
    def _to_storage_tensor(chunk: torch.Tensor) -> torch.Tensor:
        """Normalise an action chunk to the storage contract: [H, D] CPU fp32.

        A cache hit hands back an already-unbatched payload tensor while a miss
        produces the model's ``[1, H, D]``; both end up the same shape here.

        The final clone is what actually lets the tensor outlive the inference
        context. Converting inside that context is not enough — the result is
        still an inference tensor, and the first in-place write from the
        storage or normaliser layers would raise.
        """
        if chunk.dim() == 3:
            chunk = chunk[0]
        out = chunk.detach().cpu().float().contiguous()
        if out.is_inference():
            out = out.clone()
        return out
