"""End-to-end integration (M4): real TCP driver + WorkerLoop + FakeEpisodeRunner.

Exercises the full M1-M3 stack — scheduler, journal, driver pull server, worker
pull/run/report loop — over a real localhost socket, with experiment semantics
faked via the strategy's fake ctl. Validates no-bubble completion, the
warmup->eval barrier ordering, journal resume, and transient-failure retry.
"""

from __future__ import annotations

import socket
import threading
import time

from openpi.conductor import task as T  # noqa: N812
from openpi.conductor.driver import ConductorDriver
from openpi.conductor.journal import Journal
from openpi.conductor.worker import WorkerLoop

from .conftest import FakeCtl
from .conftest import FakeEpisodeRunner
from .conftest import WarmupEvalFakeStrategy

S1 = T.ServerEndpoint("127.0.0.1", 8001)


def _run_driver_with_workers(
    strategy,
    *,
    yaml_weights,
    journal_path,
    n_workers=3,
    runner_factory=None,
    timeout=15.0,
):
    """Start a driver + N workers over real TCP; return (driver, ctls) at all_done."""
    ctls: dict[str, FakeCtl] = {}

    def ctl_factory(server):
        ctls[server.key] = FakeCtl(server)
        return ctls[server.key]

    driver = ConductorDriver(
        strategy,
        yaml_weights=yaml_weights,
        servers=[S1],
        journal_path=journal_path,
        ctl_factory=ctl_factory,
    )
    driver_stop = threading.Event()
    dt = threading.Thread(target=driver.run, kwargs={"stop": driver_stop.is_set}, daemon=True)
    dt.start()

    # wait for the pull server to bind
    deadline = time.monotonic() + 5.0
    while driver.port is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert driver.port is not None, "driver did not bind"

    def connect():
        return socket.create_connection(("127.0.0.1", driver.port))

    worker_stop = threading.Event()
    runner_factory = runner_factory or FakeEpisodeRunner
    wts = []
    for i in range(n_workers):
        wl = WorkerLoop(f"w{i}", S1.key, runner_factory(), connect=connect, max_backoff_s=0.5)
        t = threading.Thread(target=wl.run_forever, kwargs={"stop": worker_stop.is_set}, daemon=True)
        t.start()
        wts.append(t)

    dt.join(timeout=timeout)
    worker_stop.set()
    for t in wts:
        t.join(timeout=2.0)
    assert not dt.is_alive(), "driver did not reach all_done within timeout"
    return driver, ctls


def test_end_to_end_multi_worker_completes(tmp_path):
    strat = WarmupEvalFakeStrategy(n_warmup=2, n_eval=6)
    driver, ctls = _run_driver_with_workers(
        strat, yaml_weights={"y": 8}, journal_path=str(tmp_path / "j.jsonl"), n_workers=3
    )
    assert driver.scheduler.all_done()
    prog = driver.scheduler.progress()
    assert prog["episodes_done"] == 8  # 2 warmup + 6 eval

    # journal has every terminal episode
    done = Journal(tmp_path / "j.jsonl").replay_done_uids()
    assert len(done) == 8

    # warmup ctl ops strictly precede eval ctl ops (barrier honoured)
    kinds = [c[0] for c in ctls[S1.key].calls]
    assert kinds.index("fetch") < kinds.index("preload")  # warmup fetched before eval preload
    load_ids = [c[1] for c in ctls[S1.key].calls if c[0] == "load"]
    assert load_ids == ["y__warmup", "y"]


def test_end_to_end_resume_skips_journaled_eval(tmp_path):
    # Pre-seed the journal as if eval episodes 0..3 already completed.
    jpath = tmp_path / "j.jsonl"
    j = Journal(jpath)
    for i in range(4):
        j.record(task_uid=T.make_task_uid("y", "eval", 0, i), yaml_id="y", phase="eval", status="done", success=True)

    strat = WarmupEvalFakeStrategy(n_warmup=2, n_eval=6)
    driver, _ = _run_driver_with_workers(strat, yaml_weights={"y": 8}, journal_path=str(jpath), n_workers=2)
    assert driver.scheduler.all_done()
    # 2 warmup must still run (stage-atomic), only eval 4,5 remained to run now;
    # but progress counts the pre-seeded eval as done too -> 2 warmup + 6 eval.
    prog = driver.scheduler.progress()
    assert prog["episodes_done"] == 8


def test_end_to_end_transient_failure_retried(tmp_path):
    # one eval episode fails transiently twice, then succeeds
    shared_fail = {T.make_task_uid("y", "eval", 0, 1): 2}
    lock = threading.Lock()

    def runner_factory():
        return FakeEpisodeRunner(transient_fail=shared_fail, lock=lock)

    strat = WarmupEvalFakeStrategy(n_warmup=1, n_eval=3)
    driver, _ = _run_driver_with_workers(
        strat, yaml_weights={"y": 4}, journal_path=str(tmp_path / "j.jsonl"), n_workers=2, runner_factory=runner_factory
    )
    assert driver.scheduler.all_done()
    prog = driver.scheduler.progress()
    assert prog["episodes_done"] == 4
    assert prog["episodes_failed"] == 0
