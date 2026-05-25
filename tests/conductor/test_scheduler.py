"""EpisodeScheduler state-machine tests (M2) — the no-bubble/affinity core."""

from __future__ import annotations

import time

from openpi.conductor import task as T  # noqa: N812
from openpi.conductor.scheduler import EpisodeScheduler
from openpi.conductor.scheduler import StageState

SRV = T.ServerEndpoint("h", 8000)


def _episodes(stage_id, yaml_id, phase, n):
    return [
        T.EpisodeTask(
            task_uid=T.make_task_uid(yaml_id, phase, 0, i),
            yaml_id=yaml_id,
            phase=phase,
            experiment="exp",
            task_id=0,
            episode_idx=i,
            orig_init_state_idx=i,
            server_host=SRV.host,
            server_port=SRV.port,
            bundle_id=yaml_id,
        )
        for i in range(n)
    ]


def _chain_graph(yaml_id="y", n_warmup=1, n_eval=2, server=SRV):
    """warmup stage -> eval stage, linked by a calib artifact."""
    g = T.TaskGraph()
    w_id, e_id = f"{yaml_id}:warmup", f"{yaml_id}:eval"
    g.add_calibration(
        T.CalibrationArtifact(calib_id=yaml_id, source="warmup_stage", warmup_stage_id=w_id, cleanup_id=yaml_id)
    )
    g.add_stage(
        T.Stage(
            w_id,
            yaml_id,
            "warmup",
            server,
            episodes=_episodes(w_id, yaml_id, "warmup", n_warmup),
            produces_calib_id=yaml_id,
        )
    )
    g.add_stage(
        T.Stage(
            e_id, yaml_id, "eval", server, episodes=_episodes(e_id, yaml_id, "eval", n_eval), consumes_calib_id=yaml_id
        )
    )
    g.add_dependency(w_id, e_id)
    return g


def _advance_to_ready(sched, stage_id):
    """Simulate the driver running on_stage_begin for a stage."""
    assert any(s.stage_id == stage_id for s in sched.pending_setups())
    sched.mark_setup_running(stage_id)
    sched.mark_setup_done(stage_id)


def _drain_stage(sched, server_key, stage_id):
    """Dispatch+succeed every episode of a READY stage, then complete it."""
    while True:
        t = sched.next_task(server_key)
        if t is None:
            break
        sched.mark_result(t.task_uid, success=True, retriable=False)
    assert sched.stage_state(stage_id) == StageState.COMPLETE_PENDING
    sched.mark_complete_running(stage_id)
    sched.mark_complete_done(stage_id)


# ------------------------------------------------------------------
# dependency gating (warmup -> eval barrier)
# ------------------------------------------------------------------


def test_eval_blocked_until_warmup_done():
    s = EpisodeScheduler(_chain_graph())
    setups = {st.stage_id for st in s.pending_setups()}
    assert setups == {"y:warmup"}  # eval not yet activatable

    _advance_to_ready(s, "y:warmup")
    _drain_stage(s, SRV.key, "y:warmup")

    setups = {st.stage_id for st in s.pending_setups()}
    assert "y:eval" in setups  # now unblocked


def test_never_idle_returns_ready_episode():
    s = EpisodeScheduler(_chain_graph(n_warmup=2))
    _advance_to_ready(s, "y:warmup")
    t = s.next_task(SRV.key)
    assert t is not None
    assert t.phase == "warmup"


def test_no_task_when_nothing_ready():
    s = EpisodeScheduler(_chain_graph())
    # warmup is only SETUP_PENDING (not yet READY) -> next_task gives nothing
    assert s.next_task(SRV.key) is None


# ------------------------------------------------------------------
# activation caps (affinity: warmup relaxed, eval tight)
# ------------------------------------------------------------------


def test_warmup_concurrency_cap():
    g = T.TaskGraph()
    for k in range(3):
        yid = f"y{k}"
        wid = f"{yid}:warmup"
        g.add_calibration(
            T.CalibrationArtifact(calib_id=yid, source="warmup_stage", warmup_stage_id=wid, cleanup_id=yid)
        )
        g.add_stage(T.Stage(wid, yid, "warmup", SRV, episodes=_episodes(wid, yid, "warmup", 1), produces_calib_id=yid))
    s = EpisodeScheduler(g, warmup_concurrency=2)
    setups = {st.stage_id for st in s.pending_setups()}
    assert len(setups) == 2  # only 2 warmups activate at once


def test_eval_concurrency_cap_is_one():
    g = T.TaskGraph()
    for k in range(2):
        yid = f"y{k}"
        eid = f"{yid}:eval"
        g.add_stage(T.Stage(eid, yid, "eval", SRV, episodes=_episodes(eid, yid, "eval", 1)))
    s = EpisodeScheduler(g, eval_concurrency=1)
    setups = [st.stage_id for st in s.pending_setups()]
    assert len(setups) == 1  # eval is tight: one yaml at a time per server


# ------------------------------------------------------------------
# warmup atomicity vs eval per-episode retry
# ------------------------------------------------------------------


def test_warmup_failure_invalidates_whole_stage():
    s = EpisodeScheduler(_chain_graph(n_warmup=3))
    _advance_to_ready(s, "y:warmup")
    t = s.next_task(SRV.key)  # dispatch one of three warmup episodes
    s.mark_result(t.task_uid, success=True, retriable=False)  # one succeeds
    t2 = s.next_task(SRV.key)
    s.mark_result(t2.task_uid, success=False, retriable=True)  # one retriable-fails
    # whole stage invalidated: back to SETUP_PENDING, all 3 episodes re-pending
    assert s.stage_state("y:warmup") == StageState.SETUP_PENDING
    _advance_to_ready(s, "y:warmup")
    dispatched = [s.next_task(SRV.key) for _ in range(3)]
    assert all(d is not None for d in dispatched)  # all 3 available again


def test_eval_failure_requeues_single_episode():
    s = EpisodeScheduler(_chain_graph(n_warmup=1, n_eval=2))
    _advance_to_ready(s, "y:warmup")
    _drain_stage(s, SRV.key, "y:warmup")
    _advance_to_ready(s, "y:eval")
    a = s.next_task(SRV.key)
    b = s.next_task(SRV.key)
    assert {a.episode_idx, b.episode_idx} == {0, 1}
    s.mark_result(a.task_uid, success=False, retriable=True)  # one eval requeues
    s.mark_result(b.task_uid, success=True, retriable=False)
    again = s.next_task(SRV.key)  # only the failed episode comes back
    assert again is not None
    assert again.task_uid == a.task_uid


def test_eval_retry_exhaustion_marks_fatal():
    s = EpisodeScheduler(_chain_graph(n_warmup=1, n_eval=1), max_episode_retries=2)
    _advance_to_ready(s, "y:warmup")
    _drain_stage(s, SRV.key, "y:warmup")
    _advance_to_ready(s, "y:eval")
    for _ in range(3):  # initial + 2 retries = 3 attempts, then exhausted
        t = s.next_task(SRV.key)
        if t is None:
            break
        s.mark_result(t.task_uid, success=False, retriable=True)
    assert s.next_task(SRV.key) is None  # no more retries
    assert s.stage_state("y:eval") == StageState.COMPLETE_PENDING


def test_warmup_stage_retry_exhaustion_fails():
    s = EpisodeScheduler(_chain_graph(n_warmup=1), max_warmup_stage_retries=1)
    for _ in range(3):
        if s.stage_state("y:warmup") in (StageState.DONE, StageState.FAILED):
            break
        if any(st.stage_id == "y:warmup" for st in s.pending_setups()):
            _advance_to_ready(s, "y:warmup")
        t = s.next_task(SRV.key)
        if t is not None:
            s.mark_result(t.task_uid, success=False, retriable=True)
    assert s.stage_state("y:warmup") == StageState.FAILED


# ------------------------------------------------------------------
# resume
# ------------------------------------------------------------------


def test_resume_skips_done_eval_but_not_warmup():
    s = EpisodeScheduler(_chain_graph(n_warmup=1, n_eval=2))
    warm_uid = T.make_task_uid("y", "warmup", 0, 0)
    eval_uid0 = T.make_task_uid("y", "eval", 0, 0)
    s.mark_preexisting_done({warm_uid, eval_uid0})
    # warmup not skipped: it still activates and must run whole
    _advance_to_ready(s, "y:warmup")
    assert s.stage_state("y:warmup") == StageState.READY
    _drain_stage(s, SRV.key, "y:warmup")
    # eval: only episode 1 remains (episode 0 was pre-done)
    _advance_to_ready(s, "y:eval")
    t = s.next_task(SRV.key)
    assert t is not None
    assert t.episode_idx == 1
    assert s.next_task(SRV.key) is None


# ------------------------------------------------------------------
# full flow
# ------------------------------------------------------------------


def test_full_warmup_eval_flow_reaches_all_done():
    s = EpisodeScheduler(_chain_graph(n_warmup=2, n_eval=3))
    _advance_to_ready(s, "y:warmup")
    _drain_stage(s, SRV.key, "y:warmup")
    _advance_to_ready(s, "y:eval")
    _drain_stage(s, SRV.key, "y:eval")
    assert s.all_done()
    prog = s.progress()
    assert prog["episodes_done"] == 5
    assert prog["episodes_total"] == 5


# ------------------------------------------------------------------
# Round-1 self-review regressions
# ------------------------------------------------------------------


def test_all_predone_eval_stage_completes_not_stalls():
    # All eval episodes already journal-done on resume -> the stage must reach
    # COMPLETE_PENDING at setup, not stall forever in READY (Blocking #1).
    s = EpisodeScheduler(_chain_graph(n_warmup=1, n_eval=2))
    _advance_to_ready(s, "y:warmup")
    _drain_stage(s, SRV.key, "y:warmup")
    s.mark_preexisting_done({T.make_task_uid("y", "eval", 0, 0), T.make_task_uid("y", "eval", 0, 1)})
    assert any(st.stage_id == "y:eval" for st in s.pending_setups())
    s.mark_setup_running("y:eval")
    s.mark_setup_done("y:eval")
    assert s.stage_state("y:eval") == StageState.COMPLETE_PENDING
    s.mark_complete_running("y:eval")
    s.mark_complete_done("y:eval")
    assert s.all_done()


def test_mark_result_idempotent_on_non_dispatched():
    s = EpisodeScheduler(_chain_graph(n_warmup=1, n_eval=1))
    _advance_to_ready(s, "y:warmup")
    t = s.next_task(SRV.key)
    s.mark_result(t.task_uid, success=True, retriable=False)
    s.mark_result(t.task_uid, success=False, retriable=True)  # duplicate -> ignored
    prog = s.progress()
    assert prog["episodes_done"] == 1
    assert prog["episodes_failed"] == 0  # not double-counted


def test_stale_warmup_inflight_result_ignored_after_invalidation():
    s = EpisodeScheduler(_chain_graph(n_warmup=2))
    _advance_to_ready(s, "y:warmup")
    a = s.next_task(SRV.key)
    b = s.next_task(SRV.key)
    s.mark_result(a.task_uid, success=False, retriable=True)  # invalidates whole stage
    assert s.stage_state("y:warmup") == StageState.SETUP_PENDING
    # b was in-flight on an old worker; its late result must be ignored (no
    # double-append / premature completion of the rerun).
    s.mark_result(b.task_uid, success=True, retriable=False)
    assert s.stage_state("y:warmup") == StageState.SETUP_PENDING


def test_warmup_failure_cascades_to_downstream_eval():
    # warmup retry-exhausted -> FAILED; its eval downstream must also FAIL, else
    # eval sits BLOCKED forever and all_done() would hang.
    s = EpisodeScheduler(_chain_graph(n_warmup=1, n_eval=2), max_warmup_stage_retries=0)
    _advance_to_ready(s, "y:warmup")
    t = s.next_task(SRV.key)
    s.mark_result(t.task_uid, success=False, retriable=True)  # exhausts (max=0)
    assert s.stage_state("y:warmup") == StageState.FAILED
    assert s.stage_state("y:eval") == StageState.FAILED  # cascaded
    assert s.all_done()  # no hang


def test_eval_concurrency_releases_next_yaml_after_done():
    g = T.TaskGraph()
    for k in range(2):
        yid = f"y{k}"
        eid = f"{yid}:eval"
        g.add_stage(T.Stage(eid, yid, "eval", SRV, episodes=_episodes(eid, yid, "eval", 1)))
    s = EpisodeScheduler(g, eval_concurrency=1)
    first = [st.stage_id for st in s.pending_setups()]
    assert len(first) == 1  # only one eval yaml active at a time
    active = first[0]
    _advance_to_ready(s, active)
    _drain_stage(s, SRV.key, active)  # run + complete the active yaml
    nxt = [st.stage_id for st in s.pending_setups()]
    assert len(nxt) == 1
    assert nxt[0] != active  # slot released -> the other yaml now activates


def test_requeue_timed_out_requeues_stuck_episode():
    s = EpisodeScheduler(_chain_graph(n_warmup=1, n_eval=2))
    _advance_to_ready(s, "y:warmup")
    t = s.next_task(SRV.key)  # dispatched now
    assert s.requeue_timed_out(timeout_s=100.0) == []  # nothing timed out yet
    requeued = s.requeue_timed_out(timeout_s=0.0, now=time.monotonic() + 1000.0)
    assert t.task_uid in requeued
    # warmup episode timed out -> whole stage invalidated (stage-atomic)
    assert s.stage_state("y:warmup") == StageState.SETUP_PENDING


def test_setup_retry_exhaustion_fails_stage():
    s = EpisodeScheduler(_chain_graph(n_warmup=1), max_setup_retries=1)
    for _ in range(4):
        if "y:warmup" not in {st.stage_id for st in s.pending_setups()}:
            break
        s.mark_setup_running("y:warmup")
        s.mark_setup_failed("y:warmup", fatal=False)  # retriable setup failure
    assert s.stage_state("y:warmup") == StageState.FAILED  # exhausted -> FAILED


def test_stale_attempt_result_fenced_after_requeue():
    # G2R3: after a timeout requeue + re-dispatch of the same uid, the OLD
    # attempt's late result must be fenced (rejected), not accepted as current.
    s = EpisodeScheduler(_chain_graph(n_warmup=1, n_eval=1))
    _advance_to_ready(s, "y:warmup")
    t1 = s.next_task(SRV.key)
    assert t1.attempt == 1
    s.requeue_timed_out(timeout_s=0.0, now=time.monotonic() + 1000.0)  # times out -> invalidate
    assert s.stage_state("y:warmup") == StageState.SETUP_PENDING
    _advance_to_ready(s, "y:warmup")
    t2 = s.next_task(SRV.key)
    assert t2.task_uid == t1.task_uid  # same uid
    assert t2.attempt == 2  # new generation
    # old worker (attempt 1) finally reports -> fenced, stage NOT completed
    s.mark_result(t1.task_uid, success=True, retriable=False, attempt=1)
    assert s.stage_state("y:warmup") == StageState.READY
    # current attempt (2) result is accepted
    s.mark_result(t2.task_uid, success=True, retriable=False, attempt=2)
    assert s.stage_state("y:warmup") == StageState.COMPLETE_PENDING
