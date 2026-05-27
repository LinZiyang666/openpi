"""Episode-level scheduler: the no-bubble, affinity-aware core (plan §7).

Pure state machine — no I/O, no ctl, no threads of its own. The driver polls it
single-threaded-ish (a lock guards concurrent access from the barrier thread).
The driver performs the actual stage setup/teardown (on_stage_begin/complete)
and feeds results back; the scheduler decides *what is ready* and *when a stage
may activate*.

Key behaviours:
  - never idle: if a server has any ready episode, ``next_task`` returns it.
  - yaml affinity: per server, activation is capped per phase. Both warmup and
    eval default to 2 concurrent yamls to fill barrier/straggler bubbles (the
    2nd yaml activates only when the 1st's ready episodes are exhausted and
    workers are idle; same-keybuilder yamls share one backend via BackendPool,
    so the extra yaml adds ~no GPU memory). Set ``eval_concurrency=1`` for the
    tightest memory footprint at the cost of an end-of-yaml idle bubble.
  - warmup atomicity: a retriable warmup failure invalidates the *whole* stage
    (episodes re-pending, stage back to SETUP_PENDING) — never per-episode
    (would duplicate the appended server dump). Eval retries per-episode.
  - dependency gating: a downstream stage cannot activate until its upstreams
    are DONE (the warmup->eval barrier).

Coupling map:
  DEPENDS ON:  task.py, standard library
  CONSUMED BY: driver.py
  IF CHANGED:  driver poll loop
"""

from __future__ import annotations

import dataclasses
import enum
import threading
import time

from openpi.conductor import task as _task


class StageState(enum.Enum):
    BLOCKED = "blocked"  # upstream not all done OR not yet activated
    SETUP_PENDING = "setup_pending"  # activated; driver must run on_stage_begin
    SETUP_RUNNING = "setup_running"  # driver is running on_stage_begin
    READY = "ready"  # episodes dispatchable
    COMPLETE_PENDING = "complete_pending"  # all episodes terminal; driver must run on_stage_complete
    COMPLETE_RUNNING = "complete_running"
    DONE = "done"
    FAILED = "failed"  # exhausted retries / fatal


class _StageRT:
    """Mutable per-stage runtime state."""

    def __init__(self, stage: _task.Stage) -> None:
        self.stage = stage
        self.state = StageState.BLOCKED
        self.pending: list[str] = [e.task_uid for e in stage.episodes]
        self.dispatched: set[str] = set()
        self.done_ok: set[str] = set()
        self.done_fail: set[str] = set()
        self.stage_attempts = 0  # warmup stage-level retry counter
        self.setup_attempts = 0  # setup/complete hook retry counter
        self.fatal = False

    @property
    def total(self) -> int:
        return len(self.stage.episodes)

    def reset_for_rerun(self) -> None:
        """Whole-stage invalidation (warmup atomicity)."""
        self.pending = [e.task_uid for e in self.stage.episodes]
        self.dispatched.clear()
        self.done_ok.clear()
        self.done_fail.clear()


class EpisodeScheduler:
    """See module docstring. Thread-safe via a single internal lock."""

    def __init__(
        self,
        graph: _task.TaskGraph,
        *,
        warmup_concurrency: int = 2,
        eval_concurrency: int = 2,
        max_episode_retries: int = 3,
        max_warmup_stage_retries: int = 3,
        max_setup_retries: int = 3,
    ) -> None:
        graph.validate()
        self._graph = graph
        self._warmup_cc = warmup_concurrency
        self._eval_cc = eval_concurrency
        self._max_ep_retries = max_episode_retries
        self._max_wu_retries = max_warmup_stage_retries
        self._max_setup_retries = max_setup_retries
        self._lock = threading.RLock()
        self._rt: dict[str, _StageRT] = {sid: _StageRT(s) for sid, s in graph.stages.items()}
        self._ep_index: dict[str, tuple[str, _task.EpisodeTask]] = {}
        for sid, s in graph.stages.items():
            for ep in s.episodes:
                self._ep_index[ep.task_uid] = (sid, ep)
        self._ep_attempts: dict[str, int] = {}
        self._dispatch_ts: dict[str, float] = {}  # task_uid -> dispatch monotonic ts
        self._dispatch_gen: dict[str, int] = {}  # task_uid -> current dispatch generation (G2R3 fence)

    # ------------------------------------------------------------------
    # resume support (journal)
    # ------------------------------------------------------------------

    def mark_preexisting_done(self, done_uids: set[str]) -> None:
        """Pre-seed eval episodes already completed (journal replay; plan §8.1).

        Only applies to eval episodes — warmup is stage-atomic and must rerun
        whole if its stage is not yet complete (plan §8.2), so warmup uids here
        are ignored.
        """
        with self._lock:
            for uid in done_uids:
                entry = self._ep_index.get(uid)
                if entry is None:
                    continue
                sid, ep = entry
                if ep.phase != "eval":
                    continue
                rt = self._rt[sid]
                if uid in rt.pending:
                    rt.pending.remove(uid)
                    rt.done_ok.add(uid)

    # ------------------------------------------------------------------
    # activation (dependency gating + per-server phase concurrency caps)
    # ------------------------------------------------------------------

    def _deps_satisfied(self, stage_id: str) -> bool:
        return all(self._rt[u].state == StageState.DONE for u in self._graph.upstreams(stage_id))

    def _active_count(self, server_key: str, phase: str) -> int:
        # "active" = activated and not finished (not BLOCKED/DONE/FAILED).
        live = {
            StageState.SETUP_PENDING,
            StageState.SETUP_RUNNING,
            StageState.READY,
            StageState.COMPLETE_PENDING,
            StageState.COMPLETE_RUNNING,
        }
        return sum(
            1
            for rt in self._rt.values()
            if rt.stage.server.key == server_key and rt.stage.phase == phase and rt.state in live
        )

    def _refresh_activation(self) -> None:
        # Deterministic order (stage_id) so behaviour is reproducible/testable.
        for sid in sorted(self._rt):
            rt = self._rt[sid]
            if rt.state != StageState.BLOCKED or rt.fatal:
                continue
            if not self._deps_satisfied(sid):
                continue
            cap = self._warmup_cc if rt.stage.phase == "warmup" else self._eval_cc
            if self._active_count(rt.stage.server.key, rt.stage.phase) < cap:
                rt.state = StageState.SETUP_PENDING

    # ------------------------------------------------------------------
    # driver-facing stage lifecycle queries
    # ------------------------------------------------------------------

    def pending_setups(self) -> list[_task.Stage]:
        with self._lock:
            self._refresh_activation()
            return [rt.stage for rt in self._rt.values() if rt.state == StageState.SETUP_PENDING]

    def mark_setup_running(self, stage_id: str) -> None:
        """Driver is running this stage's on_stage_begin hook."""
        with self._lock:
            self._rt[stage_id].state = StageState.SETUP_RUNNING

    def mark_setup_done(self, stage_id: str) -> None:
        with self._lock:
            rt = self._rt[stage_id]
            # A zero-episode stage (pure setup) OR one whose episodes were all
            # pre-completed on resume (pending+dispatched empty) goes straight to
            # complete; otherwise it becomes dispatchable. Without the resume
            # check such a stage would stall in READY forever (no result arrives
            # to trigger _maybe_complete) and all_done() would never hold.
            if rt.total == 0 or (not rt.pending and not rt.dispatched):
                rt.state = StageState.COMPLETE_PENDING
            else:
                rt.state = StageState.READY

    def pending_completes(self) -> list[_task.Stage]:
        with self._lock:
            return [rt.stage for rt in self._rt.values() if rt.state == StageState.COMPLETE_PENDING]

    def mark_complete_running(self, stage_id: str) -> None:
        """Driver is running this stage's on_stage_complete hook."""
        with self._lock:
            self._rt[stage_id].state = StageState.COMPLETE_RUNNING

    def mark_complete_done(self, stage_id: str) -> None:
        """on_stage_complete finished; the stage is fully done."""
        with self._lock:
            self._rt[stage_id].state = StageState.DONE

    def mark_setup_failed(self, stage_id: str, *, fatal: bool = False) -> None:
        """on_stage_begin raised. ``fatal`` (non-retriable, e.g. ConfigValidationError)
        OR exceeding max_setup_retries → mark the stage FAILED and cascade to its
        downstream — never retry a config error forever (plan §9.1). Otherwise roll
        back to SETUP_PENDING so a transient ctl error self-recovers."""
        with self._lock:
            rt = self._rt[stage_id]
            rt.setup_attempts += 1
            if fatal or rt.setup_attempts > self._max_setup_retries:
                rt.fatal = True
                rt.state = StageState.FAILED
                self._cascade_fail(stage_id)
            else:
                rt.state = StageState.SETUP_PENDING

    def mark_complete_failed(self, stage_id: str, *, fatal: bool = False) -> None:
        """on_stage_complete raised. Same fatal / bounded-retry policy as
        mark_setup_failed; rolls back to COMPLETE_PENDING when retriable."""
        with self._lock:
            rt = self._rt[stage_id]
            rt.setup_attempts += 1
            if fatal or rt.setup_attempts > self._max_setup_retries:
                rt.fatal = True
                rt.state = StageState.FAILED
                self._cascade_fail(stage_id)
            else:
                rt.state = StageState.COMPLETE_PENDING

    # ------------------------------------------------------------------
    # episode dispatch + result
    # ------------------------------------------------------------------

    def next_task(self, server_key: str) -> _task.EpisodeTask | None:
        """Return one ready episode for a worker bound to ``server_key``.

        Never idle: returns any ready episode on that server. Affinity is
        enforced upstream by activation caps (only a bounded set of yamls is
        ever READY at once per server).
        """
        with self._lock:
            self._refresh_activation()
            for sid in sorted(self._rt):
                rt = self._rt[sid]
                if rt.state != StageState.READY or rt.stage.server.key != server_key:
                    continue
                if not rt.pending:
                    continue
                uid = rt.pending.pop(0)
                rt.dispatched.add(uid)
                self._dispatch_ts[uid] = time.monotonic()
                gen = self._dispatch_gen.get(uid, 0) + 1
                self._dispatch_gen[uid] = gen
                return dataclasses.replace(self._ep_index[uid][1], attempt=gen)
            return None

    def mark_result(self, task_uid: str, *, success: bool, retriable: bool, attempt: int | None = None) -> None:
        """Record an episode result and advance stage state.

        ``attempt`` (set on worker-reported results) fences a stale result from a
        superseded dispatch: a timed-out + requeued task is re-dispatched at a
        higher generation, so the old worker's late result (lower attempt) is
        rejected (G2R3). The timeout requeue passes ``attempt=None`` (it scans
        the live ``_dispatch_ts`` so it always targets the current dispatch); the
        disconnect requeue passes the generation the dropped connection held, so
        a late drop cannot invalidate a newer dispatch re-assigned elsewhere.

        - success or fatal failure: count terminal.
        - retriable warmup failure: invalidate the whole stage (re-pending,
          back to SETUP_PENDING) — stage-atomic (plan §8.2/§9.1).
        - retriable eval failure: re-pending that single episode.
        """
        with self._lock:
            entry = self._ep_index.get(task_uid)
            if entry is None:
                return
            sid, ep = entry
            rt = self._rt[sid]
            if attempt is not None and attempt != self._dispatch_gen.get(task_uid):
                # Result from a superseded dispatch (this attempt timed out and the
                # task was re-dispatched at a higher generation). Reject so it is
                # not accepted as the current dispatch (G2R3 stale-result fence).
                return
            if task_uid not in rt.dispatched:
                # Not currently dispatched → stale / duplicate / post-reset
                # result (e.g. a warmup episode still in-flight on an old worker
                # when its stage was invalidated, or a result already recorded).
                # Ignore: makes mark_result idempotent and blocks the rerun-time
                # double-append that would bias the warmup buffer (plan §8.2).
                return
            rt.dispatched.discard(task_uid)
            self._dispatch_ts.pop(task_uid, None)

            if success:
                rt.done_ok.add(task_uid)
                self._maybe_complete(rt)
                return

            if not retriable:
                # Fatal: do not retry; count as terminal failure.
                rt.done_fail.add(task_uid)
                self._maybe_complete(rt)
                return

            # Retriable failure.
            if ep.phase == "warmup":
                self._invalidate_warmup_stage(rt)
            else:
                attempts = self._ep_attempts.get(task_uid, 0) + 1
                self._ep_attempts[task_uid] = attempts
                if attempts > self._max_ep_retries:
                    rt.done_fail.add(task_uid)
                    self._maybe_complete(rt)
                else:
                    rt.pending.append(task_uid)  # re-queue this episode

    def requeue_timed_out(self, *, timeout_s: float, now: float | None = None) -> list[str]:
        """Requeue episodes dispatched more than ``timeout_s`` ago (wall-clock
        episode timeout, plan §9.2 — catches a worker stuck inside infer that
        holds a task without reporting). Retriable: eval re-queues the episode,
        warmup invalidates its stage. Returns the requeued uids."""
        now = time.monotonic() if now is None else now
        with self._lock:
            stale = [uid for uid, ts in self._dispatch_ts.items() if now - ts > timeout_s]
        for uid in stale:
            self.mark_result(uid, success=False, retriable=True)
        return stale

    def _invalidate_warmup_stage(self, rt: _StageRT) -> None:
        rt.stage_attempts += 1
        if rt.stage_attempts > self._max_wu_retries:
            rt.fatal = True
            rt.state = StageState.FAILED
            self._cascade_fail(rt.stage.stage_id)
            return
        rt.reset_for_rerun()
        rt.state = StageState.SETUP_PENDING  # driver will re-run on_stage_begin (clears dump first)

    def _cascade_fail(self, stage_id: str) -> None:
        # A FAILED (retry-exhausted) stage can never reach DONE, so its downstream
        # stages can never satisfy their dependency. Mark them FAILED too — else
        # they would sit BLOCKED forever and all_done() would never hold (hang).
        for ds in self._graph.downstreams(stage_id):
            ds_rt = self._rt[ds]
            if ds_rt.state not in (StageState.DONE, StageState.FAILED):
                ds_rt.fatal = True
                ds_rt.state = StageState.FAILED
                self._cascade_fail(ds)

    def _maybe_complete(self, rt: _StageRT) -> None:
        if rt.state in (StageState.READY, StageState.SETUP_RUNNING) and not rt.pending and not rt.dispatched:
            rt.state = StageState.COMPLETE_PENDING

    # ------------------------------------------------------------------
    # global progress
    # ------------------------------------------------------------------

    def all_done(self) -> bool:
        with self._lock:
            return all(rt.state in (StageState.DONE, StageState.FAILED) for rt in self._rt.values())

    def stage_state(self, stage_id: str) -> StageState:
        with self._lock:
            return self._rt[stage_id].state

    def progress(self) -> dict[str, int]:
        """Coarse counts for the monitor/global summary."""
        with self._lock:
            done = sum(len(rt.done_ok) for rt in self._rt.values())
            failed = sum(len(rt.done_fail) for rt in self._rt.values())
            total = sum(rt.total for rt in self._rt.values())
            return {"episodes_done": done, "episodes_failed": failed, "episodes_total": total}
