"""MetaWorld conductor runner, worker entry and warm reset env spec (fake client / env)."""

from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np
import pytest

from exp.metaworld import tasks as T
from exp.metaworld import warm_reset_env as W
from exp.metaworld.episode_runner import MetaworldEpisodeRunner, decision_row
from openpi.conductor.agent import WorkerSpec
from openpi.conductor.task import EpisodeTask, make_task_uid
from tests.exp.metaworld.test_env import FakeEnv


def make_task(
    task_name="push-back-v3",
    episode_idx=1,
    orig=6,
    *,
    yaml_id="wr_x_full",
    port=8000,
    attempt=1,
    extra=None,
):
    task_id = T.task_id_of(task_name)
    return EpisodeTask(
        task_uid=make_task_uid(yaml_id, "eval", task_id, episode_idx),
        yaml_id=yaml_id,
        phase="eval",
        experiment=T.BENCHMARK,
        task_id=task_id,
        episode_idx=episode_idx,
        orig_init_state_idx=orig,
        server_host="127.0.0.1",
        server_port=port,
        bundle_id=yaml_id,
        attempt=attempt,
        extra={"num_trials_per_task": 2, **(extra or {})},
    )


class FakeClient:
    def __init__(self, hit_meta=None, fail_at=None):
        self.calls = []
        self.hit_meta = hit_meta
        self.fail_at = fail_at
        self.n_infer = 0
        self.closed = False

    def select_bundle(self, bundle):
        self.calls.append(("select_bundle", bundle))

    def episode_start(self, **kw):
        self.calls.append(("episode_start", kw))

    def episode_end(self, **kw):
        self.calls.append(("episode_end", kw))

    def infer(self, request):
        self.n_infer += 1
        if self.fail_at == self.n_infer:
            raise ConnectionError("server went away")
        out = {"actions": np.zeros((5, 4), dtype=np.float32)}
        if self.hit_meta is not None:
            out["__hit_meta__"] = self.hit_meta
        return out

    def close(self):
        self.closed = True


def make_runner(clients, envs, success_at=None):
    made_clients, made_envs = [], []

    def client_factory(server):
        client = clients.pop(0)
        made_clients.append((server.key, client))
        return client

    def env_factory(task, idx, seed):
        env = FakeEnv(success_at=success_at)
        made_envs.append((task, idx, seed, env))
        env.close = lambda: setattr(env, "closed", True)
        return env

    runner = MetaworldEpisodeRunner(
        seed=7, replan_steps=5, env_factory=env_factory, client_factory=client_factory
    )
    return runner, made_clients, made_envs


def test_episode_identity_rows_and_outcome():
    hit = {
        "hit_type": "WARM_START",
        "start_t": np.float64(0.2),
        "cp1_score": float("nan"),
        "warm_reset": {
            "decision_nfe": 12,
            "n_steps": np.int64(2),
            "self_direct_nfe": 10,
            "t": [1.0, 0.5],
        },
    }
    client = FakeClient(hit_meta=hit)
    runner, _clients, envs = make_runner([client], [], success_at=12)
    task = make_task()
    progress = []
    result = runner.run(
        task, lambda step, rate, hit_type: progress.append((step, hit_type))
    )
    assert (result.success, result.n_steps) == (True, 12)
    assert envs[0][:3] == ("push-back-v3", 6, 7)
    assert envs[0][3].closed
    kinds = [c[0] for c in client.calls]
    assert kinds == ["select_bundle", "episode_start", "episode_end"]
    start = client.calls[1][1]
    assert start == {
        "experiment": "metaworld_mt50",
        "task": "push-back-v3",
        "episode_id": 1,
        "extra_metadata": {
            "task_id": task.task_id,
            "orig_init_state_idx": 6,
            "task_uid": task.task_uid,
            "attempt": 1,
            "seed": 7,
        },
    }
    assert client.calls[2][1] == {"success": True}
    decisions = [r for r in result.per_step_rows if "hit_type" in r]
    assert [r["step_idx"] for r in decisions] == [0, 5, 10]
    assert all(r["hit_type"] == "WARM_START" and r["start_t"] == 0.2 for r in decisions)
    assert decisions[0]["cp1_score"] is None
    assert (
        decisions[0]["warm_reset_decision_nfe"] == 12
        and decisions[0]["warm_reset_n_steps"] == 2
    )
    assert type(decisions[0]["warm_reset_n_steps"]) is int
    assert "warm_reset_t" not in decisions[0]
    assert all(
        r["task_uid"] == task.task_uid
        and r["yaml_id"] == task.yaml_id
        and r["orig_init_state_idx"] == 6
        for r in decisions
    )
    summary = [r for r in result.per_step_rows if r.get("_kind") == "episode_summary"]
    assert len(summary) == 1 and "hit_type" not in summary[0]
    assert summary[0]["n_decisions"] == 3 and summary[0]["end_reason"] == "success"
    # the driver writes rows with allow_nan=False
    json.dumps(result.per_step_rows, allow_nan=False)
    assert progress[0] == (0, "WARM_START") and len(progress) == 12


def test_rows_without_hit_meta_still_count_decisions():
    runner, _, _ = make_runner([FakeClient()], [])
    result = runner.run(make_task(), lambda *a: None)
    decisions = [r for r in result.per_step_rows if "hit_type" in r]
    assert len(decisions) == 32 and all(r["hit_type"] is None for r in decisions)
    assert [r["step_idx"] for r in decisions] == list(range(0, 160, 5))
    assert result.success is False and result.n_steps == 160


def test_connection_reuse_bundle_switch_and_reconnect():
    first, second = FakeClient(), FakeClient()
    runner, clients, _ = make_runner([first, second], [], success_at=1)
    runner.run(make_task(yaml_id="a"), lambda *a: None)
    runner.run(make_task(yaml_id="a", episode_idx=0), lambda *a: None)
    runner.run(make_task(yaml_id="b"), lambda *a: None)
    assert [c for c in first.calls if c[0] == "select_bundle"] == [
        ("select_bundle", "a"),
        ("select_bundle", "b"),
    ]
    runner.run(make_task(yaml_id="b", port=8001), lambda *a: None)
    assert first.closed and [k for k, _ in clients] == [
        "127.0.0.1:8000",
        "127.0.0.1:8001",
    ]


def test_failure_drops_connection_and_ends_episode():
    client = FakeClient(fail_at=2)
    runner, _, envs = make_runner([client, FakeClient()], [])
    with pytest.raises(ConnectionError):
        runner.run(make_task(), lambda *a: None)
    assert client.closed and envs[0][3].closed
    assert client.calls[-1] == ("episode_end", {"success": False})
    runner.run(make_task(), lambda *a: None)  # reconnects with a fresh client


def test_task_name_cross_check():
    runner, _, _ = make_runner([FakeClient()], [])
    with pytest.raises(ValueError, match="disagrees"):
        runner.run(make_task(extra={"task_name": "reach-v3"}), lambda *a: None)
    bad = make_task()
    bad = dataclasses.replace(bad, task_id=50)
    with pytest.raises(ValueError, match="MT50"):
        MetaworldEpisodeRunner.task_name(bad)


def test_decision_row_drops_non_scalars():
    row = decision_row(
        make_task(),
        "push-back-v3",
        5,
        {"__hit_meta__": {"hit_type": "MISS", "winner_id": [1, 2], "miss_nfe": 2}},
    )
    assert (
        row["hit_type"] == "MISS" and row["winner_id"] is None and row["step_idx"] == 5
    )
    assert row["miss_nfe"] == 2 and "warm_reset_decision_nfe" not in row


# ------------------------------------------------------------------
# Worker spawn and entry
# ------------------------------------------------------------------


def test_worker_spawn_command_and_environment(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/repo/.venv")
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    spec = WorkerSpec(
        worker_id="mw-0-1",
        server_key="127.0.0.1:23192",
        gpu_id="0",
        replan_steps=5,
        seed=7,
        env={"MUJOCO_EGL_DEVICE_ID": "0"},
    )
    popen = []
    monkeypatch.setattr(
        W.subprocess, "Popen", lambda cmd, **kw: popen.append((cmd, kw)) or "handle"
    )
    assert (
        W.spawn_worker(
            spec, "127.0.0.1", 9100, worker_python="/sim/python", repo_root="/repo"
        )
        == "handle"
    )
    cmd, kw = popen[0]
    assert cmd == [
        "/sim/python", "-m", "exp.metaworld.worker_entry", "--worker-id", "mw-0-1",
        "--server-key", "127.0.0.1:23192", "--driver-host", "127.0.0.1", "--driver-port", "9100",
        "--seed", "7", "--replan-steps", "5",
    ]  # fmt: skip
    assert kw["start_new_session"] is True
    env = kw["env"]
    assert "VIRTUAL_ENV" not in env
    assert env["PYTHONPATH"] == os.pathsep.join(("/repo", "/repo/src"))
    assert env["MUJOCO_GL"] == "egl"
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["MUJOCO_EGL_DEVICE_ID"] == "0"


def test_worker_entry_imports_without_simulator():
    from exp.metaworld import worker_entry

    args = worker_entry.parser().parse_args(
        [
            "--worker-id",
            "w",
            "--server-key",
            "h:1",
            "--driver-host",
            "h",
            "--driver-port",
            "2",
        ]
    )
    assert (args.seed, args.replan_steps) == (7, 5)
    assert "metaworld" not in sys.modules
