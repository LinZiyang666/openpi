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
