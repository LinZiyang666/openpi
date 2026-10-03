"""Hermetic queue probes and resume transitions: no network, servers, or real worker processes."""

import os
import subprocess
import threading

import pytest

from exp.step_diag.ops import self13_queue as Q


@pytest.mark.parametrize("rc,output", [
    (1, "connection lost"), (99, "timeout"), (124, ""), (255, "ssh unreachable"), (0, ""),
    (0, "SESSION_STATE=gone DONE_LINES=0"),
    (0, "SESSION_STATE=gone DONE_LINES= CELL_PROBE_DONE"),
    (0, "SESSION_STATE=gone DONE_LINES=0 SDCELL_EXIT=bad CELL_PROBE_DONE"),
])
def test_uncertain_probe_never_finishes_a_cell(monkeypatch, rc, output):
    monkeypatch.setattr(Q, "sh", lambda *args, **kwargs: (rc, output))
    assert Q.cell_exit("wls", "/tmp/sdiag/cell.log") is None


@pytest.mark.parametrize("session_rc,content,expected", [
    (0, "still running", None),
    (0, "SDCELL_EXIT=0\n", None),
    (1, "SDCELL_EXIT=0\n", 0),
    (1, "SDCELL_EXIT=7\n", 7),
    (1, "\x00\x00[step_diag] DONE arm=selfwarmreset_t0.2\n", 0),
    (1, "[step_diag] INCOMPLETE arm=selfwarmreset_t0.2\n", 1),
    (1, None, 1),
    (127, "SDCELL_EXIT=0\n", None),
    (2, "SDCELL_EXIT=0\n", None),
])
def test_remote_probe_shell_distinguishes_lifecycle_states(tmp_path, monkeypatch, session_rc, content, expected):
    # A private tmux stub executes only its configured exit code; no real sessions are queried.
    binary = tmp_path / "bin"
    binary.mkdir()
    tmux = binary / "tmux"
    tmux.write_text(f"#!/bin/sh\nexit {session_rc}\n")
    tmux.chmod(0o755)
    logpath = tmp_path / "cell with 'quotes'.log"
    if content is not None:
        logpath.write_text(content)

    def local_probe(host, cmd):
        assert host == Q.HOSTS["wls"]["worker"]
        proc = subprocess.run(["bash", "--noprofile", "--norc", "-c", cmd], capture_output=True, text=True,
                              env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"}, check=False)
        return proc.returncode, proc.stdout + proc.stderr

    monkeypatch.setattr(Q, "sh", local_probe)
    assert Q.cell_exit("wls", str(logpath)) == expected


@pytest.mark.parametrize("already_done", [False, True])
def test_resume_attaches_without_spending_retry_on_unknown_probe(monkeypatch, already_done):
    host, port, teacher = "wls", 23140, "pi05"
    key = f"{host}:{port}"
    job = {"id": "job", "teacher": teacher, "arm": "selfwarmreset_t0.2", "task": "CloseFridge",
           "lane": "main", "status": "running", "slot": key, "tries": 1, "cell": "existing"}
    state = {"jobs": [job], "slots": {}}
    budget = Q.Budget(Q.DEFAULT_CAPS)
    budget.try_take(host, teacher, True, force=True)
    starts, reaps = [], []
    probes = iter([0] if already_done else [None, None, 0])
    monkeypatch.setattr(Q, "_stop", threading.Event())
    monkeypatch.setattr(Q, "save_state", lambda state: None)
    monkeypatch.setattr(Q, "log", lambda message: None)
    monkeypatch.setattr(Q, "start_server", lambda *args: "config-sha")
    monkeypatch.setattr(Q, "stop_server", lambda *args: None)
    monkeypatch.setattr(Q, "start_cell", lambda *args: (starts.append(args) or (0, "already running")))
    monkeypatch.setattr(Q, "cell_exit", lambda *args: next(probes))
    monkeypatch.setattr(Q, "reap_orphans", lambda *args: reaps.append(args))
    monkeypatch.setattr(Q.time, "sleep", lambda delay: None)
    monkeypatch.setattr(Q, "sh", lambda *args: pytest.fail("a successful resume must not rotate logs or retry"))
    Q.slot_main(state, budget, host, teacher, port, "0", job["id"])
    assert job["status"] == "done" and job["tries"] == 1
    assert len(starts) == int(not already_done) and reaps == [(host, port)]
    assert budget.count[host][teacher] == 0 and not state["slots"]


def test_stopping_an_unknown_cell_keeps_its_resume_state(monkeypatch):
    host, port, teacher = "wls", 23140, "pi05"
    key = f"{host}:{port}"
    job = {"id": "job", "teacher": teacher, "arm": "selfwarmreset_t0.2", "task": "CloseFridge",
           "lane": "main", "status": "running", "slot": key, "tries": 1, "cell": "existing"}
    state = {"jobs": [job], "slots": {}}
    budget = Q.Budget(Q.DEFAULT_CAPS)
    budget.try_take(host, teacher, True, force=True)
    stop = threading.Event()
    monkeypatch.setattr(Q, "_stop", stop)
    monkeypatch.setattr(Q, "save_state", lambda state: None)
    monkeypatch.setattr(Q, "log", lambda message: None)
    monkeypatch.setattr(Q, "start_server", lambda *args: "config-sha")
    monkeypatch.setattr(Q, "stop_server", lambda *args: None)
    monkeypatch.setattr(Q, "start_cell", lambda *args: (0, "already running"))
    monkeypatch.setattr(Q, "cell_exit", lambda *args: None)
    monkeypatch.setattr(Q.time, "sleep", lambda delay: stop.set())
    monkeypatch.setattr(Q, "reap_orphans", lambda *args: pytest.fail("unknown cell must not be reaped"))
    monkeypatch.setattr(Q, "sh", lambda *args: pytest.fail("unknown cell must not have its log rotated"))
    Q.slot_main(state, budget, host, teacher, port, "0", job["id"])
    assert job["status"] == "running" and job["tries"] == 1 and job["slot"] == key


def test_slot_sheds_its_server_between_cells_while_the_host_is_over_budget(monkeypatch):
    """Resumed cells are forced into the budget at a restart; a slot whose cell ended on an over-budget host returns
    its server instead of claiming the next job, and only until the host fits again."""
    host, port, teacher = "h100", 23250, "groot"
    key = f"{host}:{port}"
    jobs = [{"id": "g1", "teacher": teacher, "arm": "warmshoot_t0.75", "lane": "main", "task": "CloseFridge",
             "status": "pending", "tries": 0, "slot": None, "cell": None}]
    state = {"jobs": jobs, "slots": {key: {"arm": "warmshoot_t0.75", "job": "g0"}}}
    budget = Q.Budget(Q.DEFAULT_CAPS)
    n_fit = int(Q.HOSTS[host]["ram_budget"] // Q.COST[teacher][1])
    for _ in range(n_fit + 1):  # one more than fits, as after a restart
        budget.try_take(host, teacher, False, force=True)
    assert budget.over(host)
    stop = threading.Event()
    events = []
    monkeypatch.setattr(Q, "_stop", stop)
    monkeypatch.setattr(Q, "save_state", lambda st: None)
    monkeypatch.setattr(Q, "log", lambda msg: events.append(msg))
    monkeypatch.setattr(Q, "libero_pi05_waiting", lambda *a: False)
    monkeypatch.setattr(Q, "stop_server", lambda h, p: events.append(f"stop {p}"))
    monkeypatch.setattr(Q, "start_server", lambda *a: pytest.fail("a shedding slot must not start a server"))
    monkeypatch.setattr(Q.time, "sleep", lambda s: stop.set())
    Q.slot_main(state, budget, host, teacher, port, "0", "g0")
    assert f"stop {port}" in events and any("SLOT_YIELD" in e and "host budget" in e for e in events)
    assert not budget.over(host) and budget.count[host][teacher] == n_fit and jobs[0]["status"] == "pending"


def test_host_overrides_repair_a_server_host_with_another_worker(monkeypatch):
    hosts = {h: dict(v) for h, v in Q.HOSTS.items()}
    monkeypatch.setattr(Q, "HOSTS", hosts)
    got = Q.apply_host_overrides('{"h100": {"worker": "timan107", "worker_gpus": ["4", "5"]}}')
    assert got == {"h100": {"worker": "timan107", "worker_gpus": ["4", "5"]}}
    assert hosts["h100"]["worker"] == "timan107" and hosts["h100"]["addr"] == "149.165.153.233"
    assert Q.apply_host_overrides("{}") == {}
    for bad in ('{"nohost": {"worker": "x"}}', '{"h100": {"addr": "x"}}'):
        with pytest.raises(ValueError):
            Q.apply_host_overrides(bad)


def test_server_launch_moves_the_previous_serve_output_aside(tmp_path, monkeypatch):
    """A failure line an earlier launch left in serve_<port>.out must not fail the next start on the port."""
    host, port = "wls", 23154
    cmds = []
    probes = iter([None, "config-sha"])
    monkeypatch.setattr(Q, "_server_probe", lambda *a: next(probes))
    monkeypatch.setattr(Q, "stop_server", lambda *a: None)
    monkeypatch.setattr(Q, "sh", lambda h, cmd, *a, **k: (cmds.append(cmd) or (0, "")))
    monkeypatch.setattr(Q.time, "sleep", lambda s: None)
    assert Q.start_server(host, "groot", "selfresetfinal_t0.5", port) == "config-sha"
    launch = cmds[0]
    assert launch.index(f"mv -f /tmp/sdiag/serve_{port}.out") < launch.index("nohup bash")
    # the rotation step itself never fails the launch chain when there is nothing to move
    out = tmp_path / "serve.out"
    step = Q.rotate_serve_out(port).replace("/tmp/sdiag/", f"{tmp_path}/")
    assert subprocess.run(["bash", "-c", f"{step} && echo ok"], capture_output=True, text=True).stdout.strip() == "ok"
    out.write_text("server readiness failed: old\n")
    moved = tmp_path / f"serve_{port}.out"
    out.rename(moved)
    subprocess.run(["bash", "-c", step], check=True)
    assert not moved.exists() and (tmp_path / f"serve_{port}.out.prev").read_text().startswith("server readiness failed")
