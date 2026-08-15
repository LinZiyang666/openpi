"""Scheduler-accepted plumbing for the X14 RL router batches.

An RL batch may only train on episodes that the scheduler actually took as the
live dispatch's outcome. The journal alone cannot express that: a stale result
from a superseded dispatch is journaled with exactly the same
``status="failed"`` / ``status="done"`` shape as the real one, so an offline
packager reading only the ledger would happily admit a rollout that the
scheduler discarded — and credit its reward to the wrong trajectory.

These tests pin the three links of that chain:

  1. ``EpisodeScheduler.mark_result`` reports acceptance;
  2. ``ConductorDriver.handle_result`` carries it (plus attempt / error) into
     the journal;
  3. the journal keeps pre-X14 records byte-identical, so every existing
     consumer and every existing resume path is untouched.
"""

from __future__ import annotations

import json

from openpi.conductor import protocol as P  # noqa: N812
from openpi.conductor import task as T  # noqa: N812
from openpi.conductor.driver import ConductorDriver
from openpi.conductor.journal import Journal
from openpi.conductor.scheduler import EpisodeScheduler
from openpi.conductor.strategy import ExperimentStrategy

SRV = T.ServerEndpoint("h", 8000)


def _episodes(yaml_id: str, phase: str, n: int) -> list[T.EpisodeTask]:
    return [
        T.EpisodeTask(
            task_uid=T.make_task_uid(yaml_id, phase, 0, i),
            yaml_id=yaml_id, phase=phase, experiment="exp", task_id=0,
            episode_idx=i, orig_init_state_idx=i,
            server_host=SRV.host, server_port=SRV.port, bundle_id=yaml_id,
        )
        for i in range(n)
    ]


def _eval_graph(yaml_id: str = "rlr", n: int = 3) -> T.TaskGraph:
    """A single eval stage — the shape an RL training batch actually dispatches."""
    g = T.TaskGraph()
    g.add_stage(T.Stage(f"{yaml_id}:eval", yaml_id, "eval", SRV,
                        episodes=_episodes(yaml_id, "eval", n)))
    return g


def _ready_scheduler(yaml_id: str = "rlr", n: int = 3) -> EpisodeScheduler:
    """Scheduler with its single stage advanced to READY (what the driver's
    stage loop does via on_stage_begin)."""
    sched = EpisodeScheduler(_eval_graph(yaml_id, n))
    sched.mark_setup_running(f"{yaml_id}:eval")
    sched.mark_setup_done(f"{yaml_id}:eval")
    return sched


# ---------------------------------------------------------------------------
# 1. mark_result acceptance semantics
# ---------------------------------------------------------------------------


def test_live_dispatch_result_is_accepted() -> None:
    sched = _ready_scheduler()
    task = sched.next_task(SRV.key)
    assert sched.mark_result(task.task_uid, success=True, retriable=False,
                             attempt=task.attempt) is True


def test_stale_attempt_is_rejected() -> None:
    """A requeued task is re-dispatched at a higher generation; the old worker's
    late report must not be credited to the live attempt."""
    sched = _ready_scheduler()
    first = sched.next_task(SRV.key)
    sched.mark_result(first.task_uid, success=False, retriable=True, attempt=first.attempt)
    second = sched.next_task(SRV.key)
    while second.task_uid != first.task_uid:            # drain to the requeued one
        second = sched.next_task(SRV.key)
    assert second.attempt > first.attempt

    assert sched.mark_result(first.task_uid, success=True, retriable=False,
                             attempt=first.attempt) is False
    assert sched.mark_result(second.task_uid, success=True, retriable=False,
                             attempt=second.attempt) is True


def test_duplicate_result_is_rejected() -> None:
    sched = _ready_scheduler()
    task = sched.next_task(SRV.key)
    assert sched.mark_result(task.task_uid, success=True, retriable=False,
                             attempt=task.attempt) is True
    assert sched.mark_result(task.task_uid, success=True, retriable=False,
                             attempt=task.attempt) is False


def test_unknown_uid_is_rejected() -> None:
    sched = _ready_scheduler()
    assert sched.mark_result("no-such-uid", success=True, retriable=False) is False


def test_retriable_failure_is_accepted_but_not_terminal() -> None:
    """Accepted means "this came from the live dispatch", not "this is final" —
    the driver still journals only terminal records."""
    sched = _ready_scheduler()
    task = sched.next_task(SRV.key)
    assert sched.mark_result(task.task_uid, success=False, retriable=True,
                             attempt=task.attempt) is True


def test_fatal_failure_is_accepted() -> None:
    sched = _ready_scheduler()
    task = sched.next_task(SRV.key)
    assert sched.mark_result(task.task_uid, success=False, retriable=False,
                             attempt=task.attempt) is True


def test_timeout_requeue_still_works_ignoring_the_return_value() -> None:
    """Pre-existing callers discard the value; the additive return must not
    change their behaviour."""
    sched = _ready_scheduler()
    task = sched.next_task(SRV.key)
    requeued = sched.requeue_timed_out(timeout_s=-1.0)
    assert requeued == [task.task_uid]


# ---------------------------------------------------------------------------
# 2. Journal additive fields
# ---------------------------------------------------------------------------


def _lines(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_legacy_journal_record_is_byte_identical(tmp_path) -> None:
    """Every pre-X14 caller omits the new kwargs; those lines must not change,
    or a resume replaying an old journal would see an unfamiliar schema."""
    journal = Journal(tmp_path / "j.jsonl")
    journal.record(task_uid="u", yaml_id="y", phase="eval", status="done", success=True)
    (row,) = _lines(journal.path)
    assert set(row) == {"task_uid", "yaml_id", "phase", "status", "success", "ts"}


def test_journal_records_attempt_accepted_and_error(tmp_path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    journal.record(task_uid="u", yaml_id="y", phase="eval", status="failed",
                   success=False, attempt=2, accepted=False, error="SidecarError: down")
    (row,) = _lines(journal.path)
    assert row["attempt"] == 2 and row["accepted"] is False
    assert row["error"] == "SidecarError: down"


def test_replay_done_uids_ignores_the_new_fields(tmp_path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    journal.record(task_uid="u1", yaml_id="y", phase="eval", status="done",
                   success=True, attempt=1, accepted=True)
    journal.record(task_uid="u2", yaml_id="y", phase="eval", status="failed",
                   success=False, attempt=3, accepted=False, error="boom")
    assert journal.replay_done_uids() == {"u1", "u2"}


# ---------------------------------------------------------------------------
# 3. Driver end-to-end
# ---------------------------------------------------------------------------


class _EvalOnlyStrategy(ExperimentStrategy):
    def plan(self, yamls, server_assignment):
        g = T.TaskGraph()
        for y in yamls:
            g.add_stage(T.Stage(f"{y}:eval", y, "eval", server_assignment[y],
                                episodes=_episodes(y, "eval", 2)))
        return g

    def on_stage_begin(self, stage, ctl, ctx):
        pass

    def on_stage_complete(self, stage, ctl, ctx):
        pass


class _FakeCtl:
    def __init__(self, server):
        self.server = server

    def load_cache_config(self, **kw):
        return {"__ack__": "load_cache_config", "version": 1}


def _driver(tmp_path) -> ConductorDriver:
    driver = ConductorDriver(
        _EvalOnlyStrategy(),
        yaml_weights={"rlr": 1},
        servers=[SRV],
        journal_path=str(tmp_path / "journal.jsonl"),
        ctl_factory=_FakeCtl,
    )
    driver.drive_stages_once()  # runs on_stage_begin -> stage becomes READY
    return driver


def _report(driver: ConductorDriver, task: T.EpisodeTask, *, success: bool,
            error: str | None = None, attempt: int | None = None) -> None:
    result = T.EpisodeResult(
        task.task_uid, success=success, n_steps=10, error=error,
        attempt=task.attempt if attempt is None else attempt,
    )
    driver.handle_result(P.result_to_wire(result))


def test_driver_journals_accepted_true_for_a_live_result(tmp_path) -> None:
    driver = _driver(tmp_path)
    task = driver._scheduler.next_task(SRV.key)
    _report(driver, task, success=True)
    (row,) = _lines(driver._journal.path)
    assert row["accepted"] is True and row["attempt"] == task.attempt
    assert "error" not in row  # omitted when None, so success rows stay lean


def test_driver_journals_accepted_false_for_a_stale_result(tmp_path) -> None:
    """This is the row an RL batch packager must be able to reject. Without
    ``accepted`` it is indistinguishable from the live dispatch's record."""
    driver = _driver(tmp_path)
    task = driver._scheduler.next_task(SRV.key)
    _report(driver, task, success=False, error="ConnectionResetError: drop")  # requeue
    live = driver._scheduler.next_task(SRV.key)
    while live.task_uid != task.task_uid:
        live = driver._scheduler.next_task(SRV.key)

    _report(driver, task, success=True, attempt=task.attempt)   # stale worker reports
    _report(driver, live, success=True)                          # live dispatch reports

    rows = [r for r in _lines(driver._journal.path) if r["task_uid"] == task.task_uid]
    stale = [r for r in rows if r["attempt"] == task.attempt and r["status"] == "done"]
    fresh = [r for r in rows if r["attempt"] == live.attempt]
    assert stale and stale[0]["accepted"] is False
    assert fresh and fresh[0]["accepted"] is True


def test_driver_journals_the_error_on_a_fatal_result(tmp_path) -> None:
    """A non-retriable episode error (the sidecar wrapper raises
    ``FatalEpisodeError`` so the arm is not silently retried into a different
    verdict) reaches the ledger with its cause intact."""
    driver = _driver(tmp_path)
    task = driver._scheduler.next_task(SRV.key)
    _report(driver, task, success=False, error="FatalEpisodeError: student sidecar down")
    (row,) = _lines(driver._journal.path)
    assert row["status"] == "failed" and row["accepted"] is True
    assert row["error"] == "FatalEpisodeError: student sidecar down"


def test_retriable_failure_is_not_journaled(tmp_path) -> None:
    """A requeued episode is not terminal, so the ledger must stay silent — an
    RL batch counts slots, and a phantom terminal row would close one early."""
    driver = _driver(tmp_path)
    task = driver._scheduler.next_task(SRV.key)
    _report(driver, task, success=False, error="TimeoutError: worker stuck")
    assert _lines(driver._journal.path) == []


# ---------------------------------------------------------------------------
# 4. Sidecar failures must land in the ledger with their cause
# ---------------------------------------------------------------------------


def test_real_sidecar_error_is_classified_fatal() -> None:
    """The production exception type, not a hand-written marker string.

    ``SidecarExecutor`` is fail-closed by construction — it raises rather than
    falling back to the teacher — so a retry re-runs a whole episode against a
    sidecar that is almost certainly still down. Classified retriable, the row
    is never journaled at all (the driver records only terminal results) and the
    episode vanishes from the ledger instead of being explained.
    """
    from openpi.cache.sidecar_executor import SidecarError
    from openpi.conductor.driver import is_retriable_error

    exc = SidecarError("sidecar[127.0.0.1:7002]: request failed within 30.0s")
    assert isinstance(exc, RuntimeError)          # why it slipped through before
    assert is_retriable_error(repr(exc)) is False
    # Transport blips stay retriable — the change is scoped to the sidecar.
    assert is_retriable_error("ConnectionResetError: peer reset") is True


def test_sidecar_failure_reaches_the_journal_with_its_error(tmp_path) -> None:
    """§3.2/§3.6 audit chain: sidecar exception -> terminal row carrying the
    error -> excluded from the training manifest."""
    from openpi.cache.sidecar_executor import SidecarError

    driver = _driver(tmp_path)
    task = driver._scheduler.next_task(SRV.key)
    _report(driver, task, success=False, error=repr(SidecarError("student down")))

    (row,) = _lines(driver._journal.path)
    assert row["status"] == "failed" and row["accepted"] is True
    assert "SidecarError" in row["error"]


def test_a_sidecar_failed_episode_is_rejected_by_the_manifest(tmp_path) -> None:
    """End of the chain: the journaled error is what keeps the episode out of a
    training batch."""
    from openpi.cache.sidecar_executor import SidecarError

    from exp.rl_router.batch_package import build_batch_manifest

    driver = _driver(tmp_path)
    task = driver._scheduler.next_task(SRV.key)
    _report(driver, task, success=False, error=repr(SidecarError("student down")))

    manifest = build_batch_manifest(
        batch_id="b0", weights_version="v1", expected_slots=[task.task_uid],
        journal=_lines(driver._journal.path), client_rows=[], shards=[],
    )
    assert not manifest.complete
    assert manifest.rejected[0]["reason"] == "episode_error"
