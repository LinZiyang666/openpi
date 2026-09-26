"""MetaWorld as a warm reset entry environment: plugin registration, CLI, agent and admission.

The end-to-end test serves every dispatched episode of a six-arm library-free
plan through the Pi0.5 CPU stub stack the entry's own tests use, but drives it
with the real ``MetaworldEpisodeRunner`` + ``run_episode`` over a fake
simulator, stamps the rows as the driver does, and requires ``admit`` to accept
every episode with the measured NFE of each arm.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from exp.metaworld import tasks as T
from exp.metaworld import warm_reset_env as W
from exp.metaworld.episode_runner import MetaworldEpisodeRunner
from exp.warm_reset import envs as R
from exp.warm_reset.admit import admit
from exp.warm_reset.conductor import WarmResetStrategy, build_driver, worker_agent
from exp.warm_reset.plan import arm_kind, config_of, miss_steps_of, prepare, read_plan
from exp.warm_reset.run import main
from openpi.conductor.task import EpisodeTask, ServerEndpoint
from tests.exp.metaworld.test_env import FakeEnv

REPO = pathlib.Path(__file__).resolve().parents[3]
TWO_TASKS = [
    {"task_id": T.task_id_of("reach-v3"), "name": "reach-v3", "init_indices": [0, 1]},
    {
        "task_id": T.task_id_of("push-back-v3"),
        "name": "push-back-v3",
        "init_indices": [0, 1],
    },
]


def plan_of(tmp_path, *, arms=W.ARMS, tasks=None, seed=7):
    root = tmp_path / "run"
    plan = prepare(
        out=root,
        env_id=W.ENV_ID,
        base_yaml=None,
        self_trigger="always",
        arms=list(arms),
        tasks=tasks or TWO_TASKS,
        servers=["127.0.0.1:23192"],
        evidence_root=str(tmp_path / "ev"),
        namespace="mw_paired",
        rollout={"replan_steps": 5, "seed": seed, "init_states_dir": ""},
    )
    return root, plan


def graph_tasks(plan):
    strategy = WarmResetStrategy(plan)
    ids = [a["yaml_id"] for a in plan["arms"]]
    strategy.plan(ids, dict.fromkeys(ids, ServerEndpoint("127.0.0.1", 23192)))
    return strategy.tasks


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------


def test_registered_through_the_plugin_hook():
    assert W.ENV_ID in R.env_ids()
    assert R.get_env(W.ENV_ID) is W.ENV
    assert not any("metaworld" in name for name in R._PLUGIN_ERRORS)
    env = W.ENV
    assert (env.policy, env.benchmark, env.action_horizon, env.k_full) == (
        "pi05",
        "metaworld_mt50",
        5,
        10,
    )
    assert env.schedule_id == "pi05_v1" and env.default_arms == W.ARMS
    # a re-registration of the same environment (e.g. a module reload) is accepted
    assert R.register_env(dataclasses.replace(env, adapter=W.MetaworldAdapter())) == env


def test_plugin_module_is_import_light():
    code = (
        "import sys, exp.metaworld.warm_reset_env as w; "
        "print(sorted(m for m in ('metaworld', 'mujoco', 'torch', 'jax') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'src'}"},
    )
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_arm_kinds_and_step_counts():
    env = W.ENV
    assert [arm_kind(env, a, "always") for a in W.ARMS] == ["miss"] * 3 + [
        "self_only"
    ] * 3
    assert [miss_steps_of(env, a) for a in W.ARMS[:3]] == [10, 2, 1]
    # every self arm continues 2 Euler steps after its K = 10 self start
    assert env.schedule.remaining_steps(0.2) == 2


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------


def test_adapter_task_export():
    a, env = W.ENV.adapter, W.ENV
    everything = a.export_tasks(
        env, task_ids="all", task_names="", episodes=20, init_offset=0
    )
    assert [t["name"] for t in everything] == list(T.TASK_NAMES)
    assert everything[0]["init_indices"] == list(range(20))
    by_name = a.export_tasks(
        env,
        task_ids="all",
        task_names="push-back-v3, reach-v3",
        episodes=2,
        init_offset=3,
    )
    assert by_name == [
        {"task_id": 42, "name": "push-back-v3", "init_indices": [3, 4]},
        {"task_id": 43, "name": "reach-v3", "init_indices": [3, 4]},
    ]
    by_id = a.export_tasks(
        env, task_ids="43,0", task_names="", episodes=1, init_offset=0
    )
    assert [t["name"] for t in by_id] == ["reach-v3", "assembly-v3"]
    with pytest.raises(ValueError):
        a.export_tasks(env, task_ids="all", task_names="", episodes=51, init_offset=0)


def test_adapter_rollout_extra_identity_pairing():
    a, env = W.ENV.adapter, W.ENV
    a.validate_rollout(env, {"seed": 7, "replan_steps": 5, "init_states_dir": ""})
    for bad in (
        {"seed": 0, "replan_steps": 5},
        {"seed": 7, "replan_steps": 4},
        {"seed": 7, "replan_steps": 5, "init_states_dir": "/pool"},
    ):
        with pytest.raises(ValueError):
            a.validate_rollout(env, bad)
    manifest = {"rollout": {"seed": 7}}
    assert a.experiment(env, manifest) == "metaworld_mt50"
    assert a.episode_extra(env, manifest, TWO_TASKS[1]) == {"task_name": "push-back-v3"}
    with pytest.raises(ValueError, match="task_id"):
        a.episode_extra(env, manifest, {"task_id": 0, "name": "reach-v3"})
    episode = EpisodeTask(
        task_uid="u", yaml_id="y", phase="eval", experiment="metaworld_mt50", task_id=43,
        episode_idx=1, orig_init_state_idx=9, server_host="h", server_port=1, bundle_id="y",
    )  # fmt: skip
    assert a.identity(env, manifest, episode) == {"seed": 7}
    assert a.pairing(env, manifest, episode) == {"init_idx": 9, "env_seed": 7}


# ------------------------------------------------------------------
# CLI and agent
# ------------------------------------------------------------------


def test_cli_tasks_and_prepare_without_base_yaml(tmp_path):
    tasks = tmp_path / "tasks.json"
    assert (
        main(
            ["tasks", "--env", W.ENV_ID, "--task-names", "reach-v3,push-back-v3",
             "--episodes", "2", "--out", str(tasks)]
        )
        == 0
    )  # fmt: skip
    root = tmp_path / "run"
    assert (
        main(
            ["prepare", "--env", W.ENV_ID, "--tasks", str(tasks), "--arms", "all",
             "--self-trigger", "always", "--servers", "127.0.0.1:23192",
             "--server-evidence-root", str(tmp_path / "ev"), "--namespace", "mw",
             "--out", str(root)]
        )
        == 0
    )  # fmt: skip
    plan = read_plan(root)
    assert [a["arm"] for a in plan["arms"]] == list(W.ARMS)
    assert plan["rollout"]["seed"] == 7 and plan["rollout"]["replan_steps"] == 5
    for record in plan["arms"]:
        cfg = config_of(record["yaml"])
        if record["arm"] in W.ARMS[:3]:
            assert cfg.warm_reset is None
            assert cfg.miss.num_steps == miss_steps_of(W.ENV, record["arm"])
        else:
            assert cfg.miss is None
            assert cfg.warm_reset.trigger == "always" and cfg.warm_reset.start_t == 0.2
            assert cfg.warm_reset.start.source == "self"
    episodes = graph_tasks(plan)
    assert len(episodes) == 6 * 2 * 2
    assert {t.experiment for t in episodes} == {"metaworld_mt50"}
    assert all(
        t.extra == {"num_trials_per_task": 2, "task_name": T.TASK_NAMES[t.task_id]}
        and t.orig_init_state_idx == t.episode_idx
        for t in episodes
    )
    with pytest.raises(ValueError, match="seed"):
        plan_of(tmp_path / "bad", seed=0)


def test_worker_agent_launches_the_metaworld_worker(tmp_path):
    _root, plan = plan_of(tmp_path)
    agent = worker_agent(
        plan,
        server="127.0.0.1:23192",
        driver_host="127.0.0.1",
        driver_port=23193,
        gpus=["0"],
        workers_per_gpu=3,
        prefix="mw",
        rc_options={"worker_python": "/sim/python"},
    )
    assert agent._spawn_fn.func is W.spawn_worker
    assert agent._spawn_fn.keywords == {"worker_python": "/sim/python"}
    assert [s.worker_id for s in agent._specs] == ["mw-0-0", "mw-0-1", "mw-0-2"]
    assert all(s.seed == 7 and s.replan_steps == 5 for s in agent._specs)
    default = worker_agent(
        plan, server="127.0.0.1:23192", driver_host="h", driver_port=1, gpus=["0"],
        workers_per_gpu=1, prefix="mw",
    )  # fmt: skip
    assert default._spawn_fn.keywords == {"worker_python": W.DEFAULT_WORKER_PYTHON}
    with pytest.raises(ValueError, match="worker-python"):
        worker_agent(
            plan, server="127.0.0.1:23192", driver_host="h", driver_port=1, gpus=["0"],
            workers_per_gpu=1, prefix="mw", conda_env="libero",
        )  # fmt: skip


# ------------------------------------------------------------------
# End to end: runner rows + server evidence -> admission
# ------------------------------------------------------------------


class ServedClient:
    """The runner's client over an in-process served stack (evidence wrapper + interceptor)."""

    def __init__(self, served, obs):
        self.served, self.obs = served, obs

    def select_bundle(self, bundle):
        pass

    def episode_start(self, **kw):
        self.served.on_episode_start(**kw)

    def infer(self, request):
        assert request["observation/image"].ndim == 3
        meta = self.served.infer(self.obs)["__hit_meta__"]
        return {"actions": np.zeros((5, 4), dtype=np.float32), "__hit_meta__": meta}

    def episode_end(self, success):
        self.served.on_episode_end(success)
        self.served.on_task_end()

    def close(self):
        pass


def _serve(tmp_path, root, plan, success_at):
    from openpi.cache.config import build_cache_components
    from openpi.cache.interceptor import InferenceInterceptor
    from openpi.cache.orchestrator import CacheOrchestrator
    from openpi.cache.timing import SystemTimer
    from openpi.cache.warm_reset.pi05 import build_pi05_warm_reset
    from tests.cache.test_interceptor import FakePolicy
    from tests.cache.warm_reset._support import Pi05Model, obs_for

    build_driver(
        root,
        plan,
        bind_host="127.0.0.1",
        port=0,
        concurrency=2,
        ctl_factory=lambda _: None,
    )
    execution = json.loads((root / "execution.json").read_text())
    arms = {a["yaml_id"]: a for a in plan["arms"]}
    journal, per_step = [], []
    for raw in execution["tasks"]:
        task = EpisodeTask(**raw)
        arm = arms[task.yaml_id]
        yaml_path = tmp_path / f"{task.yaml_id}.yaml"
        yaml_path.write_text(arm["yaml"], encoding="utf-8")
        cfg = config_of(arm["yaml"])
        comp = build_cache_components(cfg)
        orch = CacheOrchestrator(
            storage=comp["storage"], key_builder=comp["key_builder"], gates=comp["gates"],
            judges=comp["judges"], search_strategies=comp["search_strategies"],
            timer=comp["timer"], write_policy=comp.get("write_policy"),
        )  # fmt: skip
        parts = build_pi05_warm_reset(
            cfg,
            bundle_id=task.bundle_id,
            yaml_id=task.yaml_id,
            yaml_path=str(yaml_path),
        )
        served = parts.wrap(
            InferenceInterceptor(
                FakePolicy(Pi05Model()), timer=SystemTimer(enabled=False), orchestrator=orch,
                eager=True, bundle_id=task.bundle_id, **parts.interceptor_kwargs(),
            )
        )  # fmt: skip
        client = ServedClient(served, obs_for(None))
        runner = MetaworldEpisodeRunner(
            env_factory=lambda name, idx, seed, _i=task.orig_init_state_idx: FakeEnv(
                success_at=success_at(name, _i)
            ),
            client_factory=lambda server, _c=client: _c,
        )
        result = runner.run(task, lambda *a: None)
        # the driver's stamping (ConductorDriver.handle_result)
        for row in result.per_step_rows:
            row.setdefault("success", result.success)
            row.setdefault("task_uid", result.task_uid)
            row.setdefault("attempt", task.attempt)
            row.update(accepted=True, yaml_id=task.yaml_id, run_id=execution["run_id"])
        per_step.extend(result.per_step_rows)
        journal.append(
            {"run_id": execution["run_id"], "task_uid": task.task_uid, "yaml_id": task.yaml_id,
             "phase": "eval", "status": "done" if result.success else "failed",
             "success": result.success, "attempt": task.attempt, "accepted": True, "error": None}
        )  # fmt: skip
    for name, rows in (("journal.jsonl", journal), ("per_step.jsonl", per_step)):
        (root / name).write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
        )
    return per_step


def test_six_arm_run_is_admitted_with_measured_nfe(tmp_path):
    root, plan = plan_of(tmp_path)
    # idx 0 succeeds at policy step 12 (3 decisions); idx 1 runs to the 160-step cap (32)
    per_step = _serve(tmp_path, root, plan, lambda name, idx: 12 if idx == 0 else None)
    report = admit(root)
    assert report["ok"] and not report["global_problems"], report
    decisions = 2 * (3 + 32)  # two tasks, per task one idx-0 and one idx-1 episode
    nfe = {"full": 10, "plain_k2": 2, "plain_k1": 1}
    for arm in W.ARMS:
        cell = report["arms"][arm]
        assert (cell["expected"], cell["admitted"], cell["successes"]) == (4, 4, 2), arm
        assert cell["measured_total_nfe"] == decisions * nfe.get(arm, 10 + 2), arm
    rows = [r for r in per_step if "hit_type" in r]
    by_yaml = {a["yaml_id"]: a["arm"] for a in plan["arms"]}
    for row in rows:
        arm = by_yaml[row["yaml_id"]]
        if arm in nfe:
            assert row["hit_type"] == "MISS" and row["miss_nfe"] == nfe[arm]
        else:
            assert row["hit_type"] == "SELF_ONLY" and row["start_t"] == 0.2
            assert row["warm_reset_decision_nfe"] == 12


def test_admission_rejects_a_wrong_bench_seed(tmp_path):
    root, plan = plan_of(tmp_path, arms=["full"])
    _serve(tmp_path, root, plan, lambda name, idx: 3)
    assert admit(root)["ok"]
    # a worker that ran another bench seed: its episode_start identity no longer matches
    root2, plan2 = plan_of(tmp_path / "b", arms=["full"])
    original = MetaworldEpisodeRunner.__init__

    def seeded(self, **kw):
        original(self, **{**kw, "seed": 8})

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MetaworldEpisodeRunner, "__init__", seeded)
        _serve(tmp_path / "b", root2, plan2, lambda name, idx: 3)
    report = admit(root2)
    assert not report["ok"]
    assert all(
        not ep["admitted"] and "identity_mismatch" in ep["problems"]
        for ep in report["episodes"]
    )
