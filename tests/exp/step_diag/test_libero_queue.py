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
    assert "SD_CKPT=/home/weiland/ckpt_n15_libero_spatial" in L.server_env("wls", "groot", "groot_libero_spatial")
    assert "pi05_libero_pytorch" in L.server_env("h100", "pi05", "pi05_libero_10")


def test_ports_do_not_collide_with_the_robocasa_queue():
    for host, h in L.HOSTS.items():
        mine = set(h["ports"]["pi05"]) | set(h["ports"]["groot"])
        rc = set(L.RC.HOSTS[host]["ports"]["pi05"]) | set(L.RC.HOSTS[host]["ports"]["groot"])
        assert not mine & rc and len(mine) == len(h["ports"]["pi05"]) + len(h["ports"]["groot"])


def _rc_state(tmp_path, pending, slots):
    path = tmp_path / "rc_state.json"
    jobs = [{"id": f"j{i}", "status": "pending" if i < pending else "done"} for i in range(4)]
    path.write_text(json.dumps({"jobs": jobs, "slots": {k: {"arm": "a", "job": "j"} for k in slots}}))
    return path


def test_budget_waits_for_robocasa_and_fits_next_to_its_live_servers(tmp_path):
    caps = {"h100": {"pi05": 99, "groot": 99}, "wls": {"pi05": 99, "groot": 99}}
    # a RoboCasa job waits for a live slot: the host is RoboCasa's
    b = L.Budget(caps, _rc_state(tmp_path, 1, ["wls:23140"]))
    assert not b.try_take("wls", "groot", True)
    # nothing pending: fit next to RoboCasa's four live pi0.5 servers on wls (4 x 7.8 of 44 GB)
    b = L.Budget(caps, _rc_state(tmp_path, 0, [f"wls:{p}" for p in range(23140, 23144)]))
    taken = 0
    while b.try_take("wls", "groot", True):
        taken += 1
    assert taken == int((L.RC.HOSTS["wls"]["gpu_budget"] - 4 * L.RC.COST["pi05"][0]) // L.COST["groot"][0])
    # every RoboCasa slot gone (even with a stuck pending job): the whole host budget
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
