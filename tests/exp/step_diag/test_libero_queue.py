"""Hermetic tests of the LIBERO self-start queue (``ops/libero_queue.py``) and its cell launcher
(``ops/run_lib_cell.sh``): no network, servers or simulators."""

import json
import os
import pathlib
import re
import subprocess

import pytest

from exp.step_diag import envs as E
from exp.step_diag.ops import libero_queue as L

REPO = pathlib.Path(__file__).resolve().parents[3]


def test_jobs_cover_every_arm_suite_and_task_once():
    jobs = L.build_jobs()
    assert len({j["id"] for j in jobs}) == len(jobs) == (9 + 29) * 2 * 10
    for teacher, env_ids in L.ENV_IDS.items():
        for env_id in env_ids:
            assert E.resolve_env(env_id).benchmark in E.LIBERO_SELF_SUITES
            cells = {(j["arm"], j["task"]) for j in jobs if j["env_id"] == env_id}
            assert cells == {(a, t) for a in E.LIBERO_SELF_ARMS_BY_POLICY[teacher] for t in range(10)}


def _serve_modes(script: str) -> set:
    text = (REPO / "exp/step_diag/ops" / script).read_text()
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("shadow|warm|"))
    return set(line.strip().split(")")[0].split("|")) | {"plain", "full"}


@pytest.mark.parametrize("teacher", ["pi05", "groot"])
def test_every_arm_maps_to_a_served_mode_and_an_existing_yaml(teacher):
    modes = _serve_modes("serve_pi05.sh" if teacher == "pi05" else "serve_groot.sh")
    for env_id in L.ENV_IDS[teacher]:
        for arm in E.LIBERO_SELF_ARMS_BY_POLICY[teacher]:
            mode, arg = L.server_args(env_id, arm)
            assert mode in modes, (arm, mode)
            if mode == "plain":
                assert arg == arm.removeprefix("plain_k")
            elif mode == "full":
                assert arg == "-"
            else:
                rel = pathlib.Path(arg).relative_to(L.SERVER_REPO)
                assert (REPO / rel).is_file(), arg
                assert f"warm_t{E.warm_t_of(arm):g}.yaml" == rel.name
    assert L.server_args("groot_libero_10", "selfmidreset_t0.75_n1")[0] == "selfmidreset"
    assert L.server_args("groot_libero_10", "midreset50_t0.5_n2")[0] == "midreset50"


def test_groot_server_env_names_the_suite_checkpoint():
    assert "SD_CKPT=/home/exouser/ckpt/n15_libero_10" in L.server_env("h100", "groot", "groot_libero_10")
    assert "SD_CKPT=/data/ckpt/n15_libero_spatial" in L.server_env("wls", "groot", "groot_libero_spatial")
    assert "pi05_libero_pytorch" in L.server_env("h100", "pi05", "pi05_libero_10")


def test_ports_do_not_collide_with_the_robocasa_queue():
    for host, h in L.HOSTS.items():
        mine = set(h["ports"]["pi05"]) | set(h["ports"]["groot"])
        rc = set(L.RC.HOSTS[host]["ports"]["pi05"]) | set(L.RC.HOSTS[host]["ports"]["groot"])
        assert not mine & rc and len(mine) == len(h["ports"]["pi05"]) + len(h["ports"]["groot"])


def _rc_state(tmp_path, pending, slots, teacher="groot", stale=()):
    """A RoboCasa queue state: ``pending`` waiting jobs of ``teacher``, one running cell per live slot in ``slots``
    and a finished job on every ``stale`` slot entry (a slot that yielded: no server)."""
    path = tmp_path / "rc_state.json"
    jobs = [{"id": f"j{i}", "teacher": teacher, "status": "pending" if i < pending else "done"} for i in range(4)]
    jobs += [{"id": f"run{k}", "teacher": teacher, "status": "running"} for k in slots]
    entries = {k: {"arm": "a", "job": f"run{k}"} for k in slots}
    entries.update({k: {"arm": "a", "job": "j3"} for k in stale})
    path.write_text(json.dumps({"jobs": jobs, "slots": entries}))
    return path


def test_budget_waits_for_robocasa_and_fits_next_to_its_live_servers(tmp_path, monkeypatch):
    caps = {"h100": {"pi05": 99, "groot": 99}, "wls": {"pi05": 99, "groot": 99}}
    alive = [True]
    monkeypatch.setattr(L, "rc_queue_alive", lambda *a: alive[0])
    # a RoboCasa job waits for a live slot: the host is RoboCasa's
    b = L.Budget(caps, _rc_state(tmp_path, 1, ["wls:23140"]))
    assert not b.try_take("wls", "groot", True)
    # nothing pending: fit next to RoboCasa's four live pi0.5 servers on wls (4 x 7.8 of 44 GB)
    b = L.Budget(caps, _rc_state(tmp_path, 0, [f"wls:{p}" for p in range(23140, 23144)]))
    taken = 0
    while b.try_take("wls", "groot", True):
        taken += 1
    assert taken == int((L.RC.HOSTS["wls"]["gpu_budget"] - 4 * L.RC.COST["pi05"][0]) // L.COST["groot"][0])
    # the RoboCasa queue is gone (a stuck pending job no one will claim): the whole host budget
    alive[0] = False
    b = L.Budget(caps, _rc_state(tmp_path, 1, []))
    taken = 0
    while b.try_take("h100", "pi05", True):
        taken += 1
    g, r = L.COST["pi05"]
    assert taken == int(min(L.RC.HOSTS["h100"]["gpu_budget"] // g, L.RC.HOSTS["h100"]["ram_budget"] // r))
    # no RoboCasa state at all; an unreadable one blocks
    assert L.Budget(caps, tmp_path / "missing.json").try_take("wls", "pi05", True)
    (tmp_path / "bad.json").write_text("{")
    assert not L.Budget(caps, tmp_path / "bad.json").try_take("wls", "pi05", True)


def test_budget_caps_apply_only_while_the_other_teacher_has_work(tmp_path):
    caps = {"h100": {"pi05": 1, "groot": 1}, "wls": {"pi05": 1, "groot": 1}}
    b = L.Budget(caps, _rc_state(tmp_path, 0, []))
    assert b.try_take("h100", "pi05", True) and not b.try_take("h100", "pi05", True)
    assert b.try_take("h100", "pi05", False)
    b.give_back("h100", "pi05")
    assert b.count["h100"]["pi05"] == 1 and b.used["h100"][0] == pytest.approx(L.COST["pi05"][0])


def test_run_prefix_and_tag_name_the_suite_and_task():
    job = {"env_id": "groot_libero_10", "arm": "warm_t0.875", "task": 3}
    assert L.cell_tag(23170, job) == "q23170l10t3"
    assert L.cell_log(23170, job) == "/tmp/sdiag/sdlib_q23170l10t3_warm_t0_875_sdiag_libero_self.log"
    other = {"env_id": "groot_libero_spatial", "arm": "warm_t0.875", "task": 3}
    assert L.cell_tag(23170, job) != L.cell_tag(23170, other)


def test_cell_launcher_session_matches_the_queue_log_and_driver_arguments(tmp_path):
    """Run ``run_lib_cell.sh`` against a tmux stub that records its arguments (no session is started)."""
    repo = tmp_path / "repo"
    (repo / "exp/common/data/db_init/libero/libero_10_apool").mkdir(parents=True)
    stub = tmp_path / "bin"
    stub.mkdir()
    record = tmp_path / "tmux_args"
    (stub / "tmux").write_text(f"#!/bin/sh\n[ \"$3\" = has-session ] && exit 1\nprintf '%s\\n' \"$@\" > {record}\n")
    (stub / "tmux").chmod(0o755)
    job = {"env_id": "groot_libero_10", "arm": "selfmidreset_t0.75_n1", "task": 4}
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "SD_LIB_REPO": str(repo), "SD_TMUX_SOCKET": "sdiag",
           "SD_RUN_PREFIX": "sdlq23170l10t4", "SD_OUT_ROOT": "/out", "SD_GPUS": "5"}
    out = subprocess.run(["bash", str(REPO / "exp/step_diag/ops/run_lib_cell.sh"), job["env_id"], job["arm"],
                          "ziyanglin.com:23170", "4", "c" * 64, L.cell_tag(23170, job)],
                         capture_output=True, text=True, env=env, check=True).stdout
    name = re.search(r"started (\S+)", out).group(1)
    assert f"/tmp/sdiag/{name}.log" == L.cell_log(23170, job)
    args = record.read_text().splitlines()
    assert args[:5] == ["-L", "sdiag", "new", "-s", name]
    launch = args[-1]
    for token in ("--env-id groot_libero_10", "--arm-id selfmidreset_t0.75_n1", "--experiment-id sdiag_libero_self",
                  "--servers ziyanglin.com:23170", "--tasks 4", "--gpu-ids 5", "--run-prefix sdlq23170l10t4",
                  "--out-root /out", f"{repo}/exp/common/data/db_init/libero/libero_10_apool", "SDCELL_EXIT="):
        assert token in launch.replace("\\ ", " "), token


def test_cell_launcher_refuses_an_unknown_env_or_a_missing_pool(tmp_path):
    script = str(REPO / "exp/step_diag/ops/run_lib_cell.sh")
    env = {**os.environ, "SD_LIB_REPO": str(tmp_path)}
    for env_id in ("pi05_rc", "groot_libero_spatial"):
        proc = subprocess.run(["bash", script, env_id, "full", "h:1", "0", "c", "t"], capture_output=True, text=True,
                              env=env)
        assert proc.returncode == 1


def test_pi05_libero_waits_only_for_robocasa_pi05_and_groot_libero_for_all(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "rc_queue_alive", lambda *a: True)
    caps = L.DEFAULT_CAPS
    live = ["h100:23250"]
    groot_pending = L.Budget(caps, _rc_state(tmp_path, 2, live, teacher="groot"))
    assert groot_pending.try_take("h100", "pi05", True)  # pi0.5 LIBERO ranks before RoboCasa GR00T
    assert not groot_pending.try_take("h100", "groot", True)
    pi05_pending = L.Budget(caps, _rc_state(tmp_path, 2, live, teacher="pi05"))
    assert not pi05_pending.try_take("h100", "pi05", True)


# ------------------------------------------------------------------
# The RoboCasa queue's side of the shared hosts
# ------------------------------------------------------------------


def _lib_state(tmp_path, slots, pi05_pending):
    path = tmp_path / "lib_state.json"
    jobs = [{"id": "a", "teacher": "pi05", "status": "pending" if pi05_pending else "done"},
            {"id": "b", "teacher": "groot", "status": "pending"}]
    path.write_text(json.dumps({"jobs": jobs, "slots": {k: {"env_id": e, "arm": "full", "job": "a"} for k, e in slots}}))
    return path


def test_robocasa_budget_fits_next_to_live_libero_servers(tmp_path, monkeypatch):
    RC = L.RC
    path = _lib_state(tmp_path, [("wls:23160", "pi05_libero_10"), ("wls:23170", "groot_libero_10"),
                                 ("h100:23260", "pi05_libero_spatial")], True)
    assert RC.libero_usage("wls", path) == pytest.approx((L.COST["pi05"][0] + L.COST["groot"][0],
                                                          L.COST["pi05"][1] + L.COST["groot"][1]))
    assert RC.libero_usage("wls", tmp_path / "missing.json") == (0.0, 0.0)
    monkeypatch.setattr(RC, "LIBERO_STATE", path)
    b = RC.Budget({"h100": {"pi05": 99, "groot": 99}, "wls": {"pi05": 99, "groot": 99}})
    taken = 0
    while b.try_take("wls", "groot", False):
        taken += 1
    free = RC.HOSTS["wls"]["gpu_budget"] - L.COST["pi05"][0] - L.COST["groot"][0]
    assert taken == int(free // RC.COST["groot"][0])
    assert b.try_take("wls", "groot", False, force=True)  # a resumed cell always holds its server


@pytest.mark.parametrize("pi05_pending,alive,expected", [(True, 0, True), (True, 1, False), (False, 0, False)])
def test_libero_pi05_waiting_needs_pending_jobs_and_a_live_queue(tmp_path, monkeypatch, pi05_pending, alive, expected):
    RC = L.RC
    path = _lib_state(tmp_path, [], pi05_pending)
    monkeypatch.setattr(RC.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, alive))
    assert RC.libero_pi05_waiting(path) is expected
    assert RC.libero_pi05_waiting(tmp_path / "missing.json") is False


def test_robocasa_groot_slot_hands_over_between_cells(monkeypatch):
    """A GR00T slot that finished its cell stops its server, returns its budget and leaves the state's live slots
    while pi0.5 LIBERO jobs are pending; its pending RoboCasa jobs stay pending."""
    import threading

    RC = L.RC
    host, port = "h100", 23250
    key = f"{host}:{port}"
    jobs = [{"id": "g1", "teacher": "groot", "arm": "warmshoot_t0.75", "lane": "main", "task": "CloseFridge",
             "status": "pending", "tries": 0, "slot": None, "cell": None}]
    state = {"jobs": jobs, "slots": {key: {"arm": "warmshoot_t0.75", "job": "g0"}}}
    budget = RC.Budget(RC.DEFAULT_CAPS)
    budget.try_take(host, "groot", True, force=True)
    stop = threading.Event()
    events = []
    monkeypatch.setattr(RC, "_stop", stop)
    monkeypatch.setattr(RC, "save_state", lambda st: events.append(("save", dict(st["slots"]))))
    monkeypatch.setattr(RC, "log", lambda msg: events.append(("log", msg)))
    monkeypatch.setattr(RC, "libero_pi05_waiting", lambda *a: True)
    monkeypatch.setattr(RC, "stop_server", lambda h, p: events.append(("stop", p)))
    monkeypatch.setattr(RC, "start_server", lambda *a: pytest.fail("a yielding slot must not start a server"))
    monkeypatch.setattr(RC.time, "sleep", lambda s: stop.set())
    # a resume id whose job has already finished: the slot starts out holding its server and budget, as after a cell
    RC.slot_main(state, budget, host, "groot", port, "0", "g0")
    kinds = [e[0] for e in events]
    assert ("stop", port) in events and kinds.index("stop") < max(i for i, e in enumerate(events) if e[0] == "save")
    assert any(e[0] == "log" and "SLOT_YIELD" in e[1] and "pi0.5 LIBERO" in e[1] for e in events)
    assert key not in state["slots"] and budget.count[host]["groot"] == 0
    assert jobs[0]["status"] == "pending"


def test_stale_robocasa_slot_entries_hold_no_capacity(tmp_path, monkeypatch):
    """A RoboCasa slot entry whose job is no longer running (a slot that yielded) costs no budget."""
    monkeypatch.setattr(L, "rc_queue_alive", lambda *a: True)
    stale = [f"h100:{p}" for p in range(23250, 23258)]
    blocks, g, r = L.rc_usage("h100", "pi05", _rc_state(tmp_path, 1, [], stale=stale))
    assert (blocks, g, r) == (False, 0.0, 0.0)
    blocks, g, r = L.rc_usage("h100", "pi05", _rc_state(tmp_path, 0, ["h100:23240"], teacher="pi05", stale=stale))
    assert not blocks and (g, r) == L.RC.COST["pi05"]


def test_groot_libero_waits_while_robocasa_groot_slots_yield(tmp_path, monkeypatch):
    """RoboCasa GR00T slots that yielded to pi0.5 LIBERO run no cell, yet their pending jobs rank before GR00T LIBERO:
    while the RoboCasa queue runs, GR00T LIBERO stays blocked (2026-09-25 00:0x: it had started)."""
    stale = [f"h100:{p}" for p in range(23250, 23258)]
    path = _rc_state(tmp_path, 3, [], teacher="groot", stale=stale)
    monkeypatch.setattr(L, "rc_queue_alive", lambda *a: True)
    assert L.rc_usage("h100", "groot", path)[0] is True
    assert L.rc_usage("h100", "pi05", path)[0] is False
    assert not L.Budget(L.DEFAULT_CAPS, path).try_take("wls", "groot", True)
    monkeypatch.setattr(L, "rc_queue_alive", lambda *a: False)
    assert L.rc_usage("h100", "groot", path)[0] is False


def test_rc_queue_alive_asks_tmux_then_the_state_file_age(monkeypatch, tmp_path):
    """A RoboCasa queue whose tmux session is gone still counts as alive while its state file is fresh (the restart
    window of 2026-09-25 09:30 leaked 3 GR00T LIBERO jobs); an old or missing state file does not."""
    import time as _time

    calls = []
    rc = {"code": 1}
    monkeypatch.setattr(L.subprocess, "run",
                        lambda argv, **k: calls.append(argv) or subprocess.CompletedProcess(argv, rc["code"]))
    state = tmp_path / "rc_state.json"
    assert L.rc_queue_alive(state) is False and calls == [["tmux", "has-session", "-t", "sdq"]]
    state.write_text("{}")
    assert L.rc_queue_alive(state) is True
    old = _time.time() - L.RC_RESTART_GRACE_S - 5
    os.utime(state, (old, old))
    assert L.rc_queue_alive(state) is False
    rc["code"] = 0
    assert L.rc_queue_alive(state) is True


def test_resumed_libero_slot_yields_when_robocasa_ranks_first(monkeypatch):
    """A slot resumed at a restart holds its budget without passing ``try_take``: before its next claim it re-checks
    the RoboCasa gate and, when blocked, stops its server, returns the budget and leaves the live slots."""
    import threading

    host, port = "h100", 23274
    key = f"{host}:{port}"
    jobs = [{"id": "g1", "teacher": "groot", "env_id": "groot_libero_spatial", "arm": "warmreset_t0.75_n1", "task": 1,
             "status": "pending", "tries": 0, "slot": None, "cell": None}]
    state = {"jobs": jobs, "slots": {key: {"env_id": "groot_libero_spatial", "arm": "warmreset_t0.75_n1", "job": "g0"}}}
    budget = L.Budget(L.DEFAULT_CAPS)
    budget.try_take(host, "groot", True, force=True)
    stop = threading.Event()
    events = []
    monkeypatch.setattr(L, "_stop", stop)
    monkeypatch.setattr(L, "save_state", lambda st: events.append(("save", dict(st["slots"]))))
    monkeypatch.setattr(L, "log", lambda msg: events.append(("log", msg)))
    monkeypatch.setattr(L, "rc_usage", lambda *a, **k: (True, 0.0, 0.0))
    monkeypatch.setattr(L, "stop_server", lambda h, p: events.append(("stop", p)))
    monkeypatch.setattr(L, "start_server", lambda *a: pytest.fail("a yielding slot must not start a server"))
    monkeypatch.setattr(L, "start_cell", lambda *a: pytest.fail("a yielding slot must not start a cell"))
    monkeypatch.setattr(L.time, "sleep", lambda s: stop.set())
    L.slot_main(state, budget, host, "groot", port, "0", "g0")  # "g0" already finished: the slot enters holding
    kinds = [e[0] for e in events]
    assert ("stop", port) in events and kinds.index("stop") < max(i for i, e in enumerate(events) if e[0] == "save")
    assert any(e[0] == "log" and "SLOT_YIELD" in e[1] for e in events)
    assert key not in state["slots"] and budget.count[host]["groot"] == 0 and jobs[0]["status"] == "pending"
