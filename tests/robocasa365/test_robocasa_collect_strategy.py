"""Non-manual tests for run_collect.py: identity, affinity, ctl, spawn, run-plan.

Everything here runs without GPU / sim / network: servers are plain
``ServerEndpoint`` values, the spawn function is intercepted before Popen, and
run-plans live in tmp_path.
"""

from __future__ import annotations

import json
import subprocess
from unittest import mock

import pytest

from openpi.conductor.driver import assign_servers
from openpi.conductor.task import ServerEndpoint

from exp.robocasa365.run_collect import (
    RobocasaCollectStrategy,
    _NoOpCtl,
    build_run_plan,
    build_yaml_id,
    compute_plan_hash,
    load_env_config,
    robocasa_spawn_fn,
    validate_teacher_endpoints,
    write_run_plan,
)

TASKS = [("OpenCabinet", 3), ("CloseDrawer", 2)]


def _strategy(teacher="pi05", layout=1, style=1, **kw):
    return RobocasaCollectStrategy(
        teacher=teacher, layout=layout, style=style, base_seed=0, replan_steps=5, tasks=TASKS, **kw
    )


def _planned(strategy, ports=(8010,)):
    servers = [ServerEndpoint("127.0.0.1", p) for p in ports]
    weights = {yid: n for yid, (_, n) in zip(strategy.yaml_ids, TASKS)}
    assignment = assign_servers(weights, servers, None, {s.key: 1 for s in servers})
    return strategy.plan(sorted(weights), assignment)


def _uids(graph):
    return [ep.task_uid for stage in graph.stages.values() for ep in stage.episodes]


# ------------------------------------------------------------------
# Identity (frozen §4.3.2)
# ------------------------------------------------------------------


def test_identity_formulas():
    strategy = _strategy()
    assert strategy.run_id == "collect_l1s1_pi05"
    assert build_yaml_id(strategy.run_id, "OpenCabinet") == "collect_l1s1_pi05__OpenCabinet"
    graph = _planned(strategy)
    first = graph.stages["collect_l1s1_pi05__OpenCabinet"].episodes[0]
    assert first.task_uid == "collect_l1s1_pi05__OpenCabinet:eval:0:0"


def test_uids_globally_unique_across_teachers_and_scenes():
    all_uids: list[str] = []
    for teacher in ("pi05", "groot_tp"):
        for layout, style in ((1, 1), (5, 7)):
            all_uids += _uids(_planned(_strategy(teacher=teacher, layout=layout, style=style)))
    assert len(all_uids) == len(set(all_uids)), "teacher/scene identity must be encoded in yaml_id"


def test_plan_is_deterministic_and_validates():
    strategy = _strategy()
    g1, g2 = _planned(strategy), _planned(strategy)
    assert _uids(g1) == _uids(g2)
    g1.validate()  # raises on cycles / dangling calibs


def test_stage_and_task_shape():
    graph = _planned(_strategy())
    for stage in graph.stages.values():
        assert stage.phase == "eval"
        assert stage.produces_calib_id is None and stage.consumes_calib_id is None
        for ep in stage.episodes:
            assert ep.bundle_id == "default"  # the only id a bare server acks
            assert ep.experiment == "pi05" and "/" not in ep.experiment
            assert ep.orig_init_state_idx == ep.episode_idx  # seed offset carrier
            for key in ("task_name", "layout", "style", "teacher", "base_seed", "replan_steps"):
                assert key in ep.extra, key


def test_extension_batch_continues_episode_range():
    strategy = _strategy(batch=2, episode_lo={"OpenCabinet": 3, "CloseDrawer": 2})
    graph = _planned(strategy)
    idxs = [ep.episode_idx for ep in graph.stages["collect_l1s1_pi05__OpenCabinet"].episodes]
    assert idxs == [3, 4, 5]  # seeds continue where batch 1 ended, no overlap


# ------------------------------------------------------------------
# Teacher<->server affinity (frozen §4.3.4a)
# ------------------------------------------------------------------

ENV_CONFIG = {
    "PI05_SERVERS": "127.0.0.1:8010,127.0.0.1:8011",
    "GROOT_TP_SERVERS": "127.0.0.1:8020",
}


def test_affinity_rejects_foreign_endpoint_before_graph():
    servers = [ServerEndpoint("127.0.0.1", 8010), ServerEndpoint("127.0.0.1", 8020)]
    with pytest.raises(SystemExit, match="8020"):
        validate_teacher_endpoints("pi05", servers, ENV_CONFIG)


def test_affinity_accepts_homogeneous_group_and_assignment_stays_inside():
    servers = [ServerEndpoint("127.0.0.1", 8010), ServerEndpoint("127.0.0.1", 8011)]
    validate_teacher_endpoints("pi05", servers, ENV_CONFIG)
    strategy = _strategy()
    weights = {yid: n for yid, (_, n) in zip(strategy.yaml_ids, TASKS)}
    assignment = assign_servers(weights, servers, None, {s.key: 1 for s in servers})
    allowed = {s.key for s in servers}
    assert all(ep.key in allowed for ep in assignment.values())
    # Dual-route stub check (D-K lower half): with per-server capacity 1 the two
    # yamls land on distinct endpoints — both routes are exercised.
    assert len({ep.key for ep in assignment.values()}) == 2


def test_env_config_parser(tmp_path):
    path = tmp_path / "collect.env"
    path.write_text("# comment\nPI05_SERVERS=a:1,b:2\n\nWORKER_PYTHON=/x/python\n")
    values = load_env_config(path)
    assert values == {"PI05_SERVERS": "a:1,b:2", "WORKER_PYTHON": "/x/python"}


# ------------------------------------------------------------------
# _NoOpCtl
# ------------------------------------------------------------------


def test_noop_ctl_is_socket_free():
    ctl = _NoOpCtl()
    assert ctl.unload_warmup_buffer("anything") == {}
    ctl.close()
    # No websocket attribute of any kind — constructing it must not connect.
    assert not any("socket" in attr or "_ws" in attr for attr in vars(ctl))


# ------------------------------------------------------------------
# Spawn safety (frozen §4.3.4)
# ------------------------------------------------------------------


def test_spawn_kwargs_are_safe_and_parameterized(monkeypatch):
    captured: dict = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return mock.Mock()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    # The spawn function deliberately inherits the ambient environment (minus
    # venv vars); drop the host's LD_LIBRARY_PATH so the no-hardcoded-paths
    # assertion below only sees what the code itself constructed.
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    from openpi.conductor.agent import WorkerSpec

    spec = WorkerSpec(worker_id="w0", server_key="10.0.0.5:8010", gpu_id="1")
    robocasa_spawn_fn(
        spec,
        "10.0.0.9",
        41000,
        worker_python="/venvA/bin/python",
        robocasa_cwd="/cwd/robocasa365",
        repo_root="/repo",
        egl_lib_dir="/egl/lib",
        egl_vendor_dir="/egl/vendor",
        teacher="pi05",
        connect_deadline_s=60.0,
        episode_deadline_s=900.0,
        terminate_grace_s=30.0,
    )
    # SAFETY: without a fresh session WorkerAgent.stop()'s killpg targets the
    # agent's own process group.
    assert captured["start_new_session"] is True
    assert captured["cwd"] == "/cwd/robocasa365"
    env = captured["env"]
    assert env["MUJOCO_GL"] == "egl"
    assert env["__EGL_VENDOR_LIBRARY_DIRS"] == "/egl/vendor"
    assert env["LD_LIBRARY_PATH"].startswith("/egl/lib")
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PYTHONPATH"].split(":")[0] == "/repo"
    assert captured["cmd"][0] == "/venvA/bin/python"
    assert "--teacher" in captured["cmd"]
    # All paths flow from the parameters above; nothing about weilandserver may
    # be baked into the code path under test.
    joined = " ".join(captured["cmd"]) + " " + env["PYTHONPATH"] + " " + env["LD_LIBRARY_PATH"]
    assert "/home/weiland" not in joined


# ------------------------------------------------------------------
# Run-plan artifact (frozen §4.3.6-(6))
# ------------------------------------------------------------------


def _plan_payload(strategy=None, root="/data/x/build_l1s1"):
    strategy = strategy or _strategy()
    return build_run_plan(strategy, _planned(strategy), root)


def test_run_plan_shape_and_hash_excludes_itself():
    payload = _plan_payload()
    assert payload["uids"] == _uids(_planned(_strategy()))
    assert payload["prefixes"][payload["uids"][0]] == "pi05/OpenCabinet/episode_0000"
    assert payload["params"]["collect_root"] == "/data/x/build_l1s1"  # output-affecting param is hashed
    assert payload["plan_hash"] == compute_plan_hash(payload)
    # Hash is over the body only: recomputing with the hash field present must
    # give the same digest (i.e. plan_hash is excluded from its own input).
    tampered = dict(payload)
    tampered["plan_hash"] = "0" * 64
    assert compute_plan_hash(tampered) == payload["plan_hash"]


def test_run_plan_write_is_exclusive_and_resume_checks_hash(tmp_path):
    path = tmp_path / "run_plan_collect_l1s1_pi05_b01.json"
    payload = _plan_payload()
    write_run_plan(path, payload)
    on_disk = json.loads(path.read_text())
    assert on_disk["plan_hash"] == payload["plan_hash"]

    # Same parameters => same hash => resume proceeds without touching the file.
    write_run_plan(path, _plan_payload())
    # Changed output root => different hash => refuse to resume.
    with pytest.raises(SystemExit, match="mismatch"):
        write_run_plan(path, _plan_payload(root="/data/OTHER/build_l1s1"))
    # A corrupt stored hash is caught on every read/resume.
    on_disk["plan_hash"] = "0" * 64
    path.write_text(json.dumps(on_disk))
    with pytest.raises(SystemExit, match="corrupt"):
        write_run_plan(path, payload)
