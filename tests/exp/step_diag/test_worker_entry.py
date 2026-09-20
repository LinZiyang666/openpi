"""CPU tests for exp/step_diag/worker_entry.py with the same injected fakes the production runner tests
use: the client proxy stamps the arm identity into episode_start and keeps episode_end's signature,
every infer is counted with its hit meta, the counting env records the successful last step the
runner's n_steps omits, one summary row is appended per episode, and the spawn mirror matches
run_collect.robocasa_spawn_fn argv/env exactly."""

from __future__ import annotations

import subprocess
from typing import Any

import numpy as np

from exp.robocasa365.episode_runner import PROMPT_SOURCE_KEY
from exp.robocasa365.run_collect import robocasa_spawn_fn
from exp.step_diag import worker_entry as WE
from openpi.conductor import WorkerSpec
from openpi.conductor.task import EpisodeTask

PROMPT = "open the cabinet door"


def _task(**over: Any) -> EpisodeTask:
    extra = {"task_name": "OpenCabinet", "layout": 1, "style": 1, "teacher": "pi05", "base_seed": 2_000_000,
             "replan_steps": 2, "arm_id": "plain_k2", "experiment_id": "sdiag_rc_v1", "lane": "main",
             "launch_id": "L-abc", "config_sha": "cfg-1"}
    extra.update(over)
    extra = {k: v for k, v in extra.items() if v is not None}
    return EpisodeTask(task_uid="sdiag-plain_k2__l1s1_pi05__OpenCabinet:eval:9:7", yaml_id="sdiag-plain_k2__l1s1_pi05__OpenCabinet",
                       phase="eval", experiment="pi05", task_id=9, episode_idx=7, orig_init_state_idx=7,
                       server_host="127.0.0.1", server_port=8010, bundle_id="default", attempt=2, extra=extra)


def _obs():
    return {PROMPT_SOURCE_KEY: PROMPT, "x": np.zeros(3)}


class _FakeEnv:
    def __init__(self, succeed_at=None):
        self.steps = 0
        self.reset_seeds = []
        self._succeed_at = succeed_at
    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.steps = 0
        return _obs(), {}
    def step(self, action):
        self.steps += 1
        return _obs(), 0.0, False, False, {"success": self._succeed_at is not None and self.steps >= self._succeed_at}
    def close(self):
        pass


class _FakeClient:
    def __init__(self, meta=True):
        self.calls = []
        self._meta = meta
    def select_bundle(self, bundle_id):
        self.calls.append(("select_bundle", {"bundle_id": bundle_id}))
    def episode_start(self, **kwargs):
        self.calls.append(("episode_start", kwargs))
    def infer(self, payload):
        self.calls.append(("infer", payload))
        out = {"actions": np.zeros((50, 12))}
        if self._meta:
            out["__hit_meta__"] = {"hit_type": "MISS", "start_t": None, "cp1_score": 0.5, "big": np.zeros(4)}
        return out
    def episode_end(self, success):
        self.calls.append(("episode_end", {"success": success}))
        return {}
    def close(self):
        pass
    def named(self, name):
        return [p for c, p in self.calls if c == name]


class _FakeAdapter:
    def env_kwargs(self):
        return {}
    def build_observation(self, obs, prompt):
        return {"prompt": prompt}
    def iter_actions(self, response, replan_steps):
        return iter(list(response["actions"])[:replan_steps])


def _runner(client, env, horizon=5):
    return WE.StepDiagEpisodeRunner(
        _FakeAdapter(), client_factory=lambda server: client, gym_make=lambda task, layout, style, **kw: env,
        horizon_fn=lambda name: horizon, handshake_probe=lambda server, timeout_s: None,
        connect_deadline_s=0.05, connect_retries=1)


def test_stamp_counts_and_summary_row_on_success():
    client, env = _FakeClient(), _FakeEnv(succeed_at=3)
    runner = _runner(client, env, horizon=10)
    result = runner.run(_task(), lambda *a: None)
    start = client.named("episode_start")[0]
    meta = start["extra_metadata"]
    assert meta["task_uid"].endswith(":9:7") and meta["attempt"] == 2 and meta["seed"] == 2_000_007
    assert {k: meta[k] for k in WE.STAMP_KEYS} == {"launch_id": "L-abc", "arm_id": "plain_k2", "experiment_id": "sdiag_rc_v1",
                                                   "config_sha": "cfg-1", "lane": "main", "layout": 1, "style": 1}
    assert client.named("episode_end") == [{"success": True}]  # signature unchanged, nothing extra
    # success on env step 3 -> runner n_steps 2 (not incremented on the success step), env count 3
    assert result.success and result.n_steps == 2 and env.steps == 3
    summary = [r for r in result.per_step_rows if r.get("row") == "episode_summary"]
    assert len(summary) == 1
    s = summary[0]
    assert s["step_idx"] == WE.SUMMARY_STEP_IDX and s["reported_n_steps"] == 2 and s["n_env_steps"] == 3
    assert s["n_decisions"] == 2 and s["n_hit_meta"] == 2  # replan 2: infers at step 0 and 2
    assert s["hit_meta"][0] == {"decision_idx": 0, "hit_type": "MISS", "start_t": None, "cp1_score": 0.5}  # arrays dropped
    assert s["task_uid"] == result.task_uid and s["attempt"] == 2 and s["init_idx"] == 7 and s["seed"] == 2_000_007
    assert s["arm_id"] == "plain_k2" and s["launch_id"] == "L-abc" and s["success"] is True and s["task_id"] == 9
    # the runner's own hit-meta rows are still there, untouched
    assert len([r for r in result.per_step_rows if "row" not in r]) == 2


def test_no_hit_meta_still_counts_decisions_and_stamp_reset_between_episodes():
    client, env = _FakeClient(meta=False), _FakeEnv()
    runner = _runner(client, env, horizon=6)
    r1 = runner.run(_task(), lambda *a: None)
    s1 = r1.per_step_rows[-1]
    assert s1["n_decisions"] == 3 and s1["n_hit_meta"] == 0 and s1["hit_meta"] == [] and s1["n_env_steps"] == 6
    assert s1["reported_n_steps"] == 6 and not r1.success
    r2 = runner.run(_task(launch_id="L-2", arm_id=None), lambda *a: None)
    meta2 = client.named("episode_start")[1]["extra_metadata"]
    assert meta2["launch_id"] == "L-2" and "arm_id" not in meta2
    assert r2.per_step_rows[-1]["n_decisions"] == 3 and r2.per_step_rows[-1]["launch_id"] == "L-2"


def test_spawn_mirror_matches_production_argv_and_env(monkeypatch):
    captured = []

    class _P:
        def __init__(self, cmd, **kw):
            captured.append((list(cmd), kw))

    monkeypatch.setattr(subprocess, "Popen", _P)
    kw = dict(worker_python="/py", robocasa_cwd="/rc", repo_root="/repo", egl_lib_dir="/egl", egl_vendor_dir="/vendor",
              teacher="pi05", connect_deadline_s=1.5, episode_deadline_s=2.5, terminate_grace_s=3.5,
              max_cached_envs=1, pinned_objects_path="/pins.json")
    spec = WorkerSpec(worker_id="w0", server_key="h:1", gpu_id="2")
    robocasa_spawn_fn(spec, "d", 7, **kw)
    WE.step_diag_spawn_fn(spec, "d", 7, **kw)
    (cmd_a, kw_a), (cmd_b, kw_b) = captured
    assert cmd_a[2] == "exp.robocasa365.worker_entry" and cmd_b[2] == "exp.step_diag.worker_entry"
    assert cmd_a[:2] + cmd_a[3:] == cmd_b[:2] + cmd_b[3:]
    assert kw_a["env"] == kw_b["env"] and kw_a["cwd"] == kw_b["cwd"] == "/rc" and kw_b["start_new_session"] is True
    # without the optional flags the mirror stays byte-identical too
    captured.clear()
    kw2 = dict(kw, max_cached_envs=None, pinned_objects_path=None)
    robocasa_spawn_fn(spec, "d", 7, **kw2)
    WE.step_diag_spawn_fn(spec, "d", 7, **kw2)
    (cmd_a, _), (cmd_b, _) = captured
    assert "--pinned-objects" not in cmd_b and "--max-cached-envs" not in cmd_b and cmd_a[3:] == cmd_b[3:]


def test_build_runner_uses_step_diag_runner():
    from exp.robocasa365.worker_entry import parse_args
    args = parse_args(["--worker-id", "w", "--server-key", "h:1", "--driver-host", "d", "--driver-port", "1",
                       "--teacher", "pi05", "--connect-deadline-s", "1", "--episode-deadline-s", "2",
                       "--terminate-grace-s", "3", "--max-cached-envs", "1"])
    wd = WE.build_runner(args)
    assert isinstance(wd._inner, WE.StepDiagEpisodeRunner)
