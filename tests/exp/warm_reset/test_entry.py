"""Experiment entry checks: frozen arms, standard workers and trusted admission."""

import copy
import dataclasses
import json
from types import SimpleNamespace

import pytest
import yaml

from exp.step_diag.envs import ENVS
from exp.warm_reset.admit import admit, trusted_expected
from exp.warm_reset.conductor import WarmResetStrategy, build_driver, worker_agent
from exp.warm_reset.plan import arm_block, config_of, default_arms, prepare, read_plan
from exp.warm_reset.run import main
from openpi.cache.warm_reset.evidence import ExpectedEpisode
from openpi.cache.warm_reset.types import WarmResetSpec
from openpi.conductor.task import ServerEndpoint
from tests.cache.warm_reset._arms import spec_for
from tests.cache.warm_reset._support import pi05_config


def make_plan(tmp_path, env_id="pi05_libero_10", arms=None):
    """Prepare a run with a frozen server-side library path."""
    cfg = pi05_config(None, start_t=ENVS[env_id].warm_ts[0])
    cfg.key_builder.type = "placeholder"
    cfg.write_policy.type = "never"
    cfg.backend.in_memory.preload_path = "/server/library.pkl"
    cfg.denoise_schedule = ENVS[env_id].schedule_id
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(dataclasses.asdict(cfg)))
    root = tmp_path / "run"
    plan = prepare(
        out=root,
        env_id=env_id,
        base_yaml=base,
        arms=arms or ["warmreset_t0.2", "selfresetfinal_t0.2"],
        tasks=[{"task_id": 2, "name": "task name", "init_indices": [4, 8]}],
        servers=["localhost:8000"],
        evidence_root=str(tmp_path / "ev"),
        namespace="paired",
        rollout={
            "replan_steps": 5,
            "seed": 7,
            "base_seed": 3000000,
            "layout": 1,
            "style": 2,
        },
    )
    return root, plan


@pytest.mark.parametrize("env_id", list(ENVS))
def test_all_default_arms_match_existing_protocol_and_hotload(tmp_path, env_id):
    root, plan = make_plan(tmp_path, env_id, default_arms(env_id))
    assert read_plan(root) == plan
    assert len(plan["arms"]) >= 7
    env = ENVS[env_id]
    for arm in plan["arms"]:
        cfg = config_of(arm["yaml"])
        actual = WarmResetSpec.from_config(cfg.warm_reset) if cfg.warm_reset else None
        expected = spec_for(
            env.policy,
            arm["arm"],
            namespace="paired",
            evidence_dir=plan["evidence_dir"],
        )
        assert actual == expected
    strategy = WarmResetStrategy(plan)
    graph = strategy.plan(
        [a["yaml_id"] for a in plan["arms"]],
        {a["yaml_id"]: ServerEndpoint("localhost", 8000) for a in plan["arms"]},
    )
    calls = []
    for stage in graph.stages.values():
        strategy.on_stage_begin(
            stage,
            SimpleNamespace(load_cache_config=lambda **kw: calls.append(kw)),
            None,
        )
    assert [c["yaml_content"] for c in calls] == [a["yaml"] for a in plan["arms"]]
    assert all(
        t.bundle_id == t.yaml_id and t.orig_init_state_idx in (4, 8)
        for t in strategy.tasks
    )
    if env.benchmark == "robocasa365":
        expected_teacher = "groot_tp" if env.policy == "groot" else "pi05"
        assert all(t.extra["teacher"] == expected_teacher for t in strategy.tasks)
        assert all(
            t.extra["task_name"] == "task name" and t.extra["base_seed"] == 3000000
            for t in strategy.tasks
        )


def test_prepare_and_execution_refuse_reuse_and_yaml_mutation(tmp_path):
    root, plan = make_plan(tmp_path)
    build_driver(
        root,
        plan,
        bind_host="127.0.0.1",
        port=0,
        concurrency=2,
        ctl_factory=lambda _: None,
    )
    execution = json.loads((root / "execution.json").read_text())
    assert len(execution["tasks"]) == 4 and execution["run_id"]
    with pytest.raises(FileExistsError):
        build_driver(root, plan, bind_host="127.0.0.1", port=0, concurrency=2)
    path = next((root / "yamls").glob("*.yaml"))
    path.write_text(path.read_text() + "\n# drift\n")
    with pytest.raises(ValueError, match="YAML changed"):
        read_plan(root)


@pytest.mark.parametrize("env_id", list(ENVS))
def test_workers_use_existing_islands_with_correct_policy_knobs(tmp_path, env_id):
    _root, plan = make_plan(tmp_path, env_id, default_arms(env_id)[:1])
    rc = {
        "worker_python": "/island/python",
        "robocasa_cwd": "/sim",
        "egl_lib_dir": "/lib",
        "egl_vendor_dir": "/vendor",
        "connect_deadline_s": 60,
        "episode_deadline_s": 1500,
        "terminate_grace_s": 5,
        "max_cached_envs": 1,
    }
    agent = worker_agent(
        plan,
        server="localhost:8000",
        driver_host="driver",
        driver_port=9100,
        gpus=["2", "3"],
        workers_per_gpu=2,
        prefix="host",
        conda_env="libero",
        rc_options=rc,
    )
    assert len(agent._specs) == 4
    assert {s.gpu_id for s in agent._specs} == {"2", "3"}
    assert all(s.env == {"MUJOCO_EGL_DEVICE_ID": s.gpu_id} for s in agent._specs)
    assert {s.resize_size for s in agent._specs} == {
        256 if env_id.startswith("groot") else 224
    }
    assert {s.replan_steps for s in agent._specs} == {5}
    if ENVS[env_id].benchmark == "robocasa365":
        assert agent._spawn_fn.keywords["teacher"] == (
            "groot_tp" if env_id.startswith("groot") else "pi05"
        )
    else:
        assert all(
            s.task_suite_name == ENVS[env_id].benchmark and s.conda_env == "libero"
            for s in agent._specs
        )


def trusted_fixture(tmp_path, env_id="pi05_libero_10"):
    """Make trusted dispatch/terminal/worker records independent of server rows."""
    root, plan = make_plan(tmp_path, env_id)
    strategy = WarmResetStrategy(plan)
    strategy.plan(
        [a["yaml_id"] for a in plan["arms"]],
        {a["yaml_id"]: ServerEndpoint("localhost", 8000) for a in plan["arms"]},
    )
    task = strategy.tasks[0]
    term = {
        "task_uid": task.task_uid,
        "yaml_id": task.yaml_id,
        "run_id": "run",
        "phase": "eval",
        "status": "failed",
        "success": False,
        "accepted": True,
        "attempt": 2,
    }
    rows = [
        dict(term, step_idx=i * 5, hit_type="WARM_START", start_t=0.2) for i in range(3)
    ]
    args = {
        "task": task,
        "task_name": "task name",
        "arm": plan["arms"][0],
        "manifest": plan,
        "run_id": "run",
        "terminals": [term],
        "per_step": rows,
    }
    return root, plan, args


def test_terminal_failure_is_valid_and_retry_seed_comes_from_dispatch(tmp_path):
    _, _, args = trusted_fixture(tmp_path, "pi05_rc")
    expected = trusted_expected(**args)
    assert isinstance(expected, ExpectedEpisode)
    assert (
        expected.n_decisions == 3
        and expected.attempt == 2
        and expected.outcome is False
    )
    assert expected.identity["seed"] == 3000004
    assert expected.identity["orig_init_state_idx"] == 4


def test_accepted_nonterminal_retry_rows_do_not_contaminate_terminal_attempt(tmp_path):
    _, _, args = trusted_fixture(tmp_path)
    old = [dict(row, attempt=1) for row in args["per_step"]]
    args["per_step"].extend(old)
    expected = trusted_expected(**args)
    assert expected.attempt == 2 and expected.n_decisions == 3


@pytest.mark.parametrize(
    "mutation",
    [
        "stale",
        "success",
        "yaml",
        "duplicate",
        "missing",
        "bool_step",
        "miss",
        "start_t",
        "duplicate_terminal",
        "error",
        "no_terminal",
        "status",
    ],
)
def test_trusted_input_corruption_is_not_silently_filtered(tmp_path, mutation):
    _, _, args = trusted_fixture(tmp_path)
    rows, term = args["per_step"], args["terminals"][0]
    if mutation == "stale":
        rows[0]["attempt"] = 1
    elif mutation == "success":
        rows[0]["success"] = True
    elif mutation == "yaml":
        rows[0]["yaml_id"] = "other"
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "missing":
        rows.pop(1)
    elif mutation == "bool_step":
        rows[0]["step_idx"] = False
    elif mutation == "miss":
        rows[0]["hit_type"] = "MISS"
    elif mutation == "start_t":
        rows[0]["start_t"] = 0.1
    elif mutation == "duplicate_terminal":
        args["terminals"].append(copy.deepcopy(term))
    elif mutation == "error":
        term["error"] = "crashed"
    elif mutation == "no_terminal":
        args["terminals"] = []
    elif mutation == "status":
        term["status"] = "done"
    with pytest.raises(ValueError):
        trusted_expected(**args)


def test_admission_requires_every_planned_episode_and_corrupt_files_fail_closed(
    tmp_path,
):
    root, plan = make_plan(tmp_path)
    build_driver(
        root,
        plan,
        bind_host="127.0.0.1",
        port=0,
        concurrency=2,
        ctl_factory=lambda _: None,
    )
    (root / "journal.jsonl").write_text("{broken")
    (root / "per_step.jsonl").write_text("")
    result = admit(root)
    assert not result["ok"] and result["global_problems"]
    assert len(result["episodes"]) == 4
    assert all(not r["admitted"] and r["total_nfe"] is None for r in result["episodes"])
    assert main(["admit", "--run-dir", str(root)]) == 2
    execution = json.loads((root / "execution.json").read_text())
    execution["tasks"].pop()
    (root / "execution.json").write_text(json.dumps(execution))
    with pytest.raises(ValueError, match="roster"):
        admit(root)


def test_complete_run_is_admitted_with_measured_nfe(tmp_path):
    """Positive path: frozen plan + driver records + real served evidence rows join and price K + N."""
    from examples.libero.episode_runner import _episode_extra_metadata
    from openpi.cache.interceptor import InferenceInterceptor
    from openpi.cache.timing import SystemTimer
    from openpi.cache.warm_reset.pi05 import build_pi05_warm_reset
    from openpi.conductor.task import EpisodeTask
    from tests.cache.test_interceptor import FakePolicy
    from tests.cache.warm_reset._support import Pi05Model, obs_for, pi05_payload, warm_orchestrator

    root, plan = make_plan(tmp_path)
    build_driver(root, plan, bind_host="127.0.0.1", port=0, concurrency=2, ctl_factory=lambda _: None)
    execution = json.loads((root / "execution.json").read_text())
    arms = {a["yaml_id"]: a for a in plan["arms"]}
    journal, per_step = [], []
    for i, raw in enumerate(execution["tasks"]):
        task, success = EpisodeTask(**raw), i % 2 == 0
        arm = arms[task.yaml_id]
        yaml_path = tmp_path / f"{task.yaml_id}.yaml"
        yaml_path.write_text(arm["yaml"], encoding="utf-8")
        parts = build_pi05_warm_reset(
            config_of(arm["yaml"]), bundle_id=task.bundle_id, yaml_id=task.yaml_id, yaml_path=str(yaml_path)
        )
        model = Pi05Model()
        orch, _ = warm_orchestrator(model, arm["start_t"], pi05_payload(arm["start_t"]))
        served = parts.wrap(InferenceInterceptor(
            FakePolicy(model), timer=SystemTimer(enabled=False), orchestrator=orch, eager=True,
            bundle_id=task.bundle_id, warm_reset=parts.executor,
        ))
        served.on_episode_start(experiment=task.experiment, task="task name", episode_id=task.episode_idx,
                                extra_metadata=_episode_extra_metadata(task))
        stamp = {"run_id": execution["run_id"], "task_uid": task.task_uid, "yaml_id": task.yaml_id,
                 "attempt": task.attempt, "accepted": True, "success": success}
        for step in range(3):
            meta = served.infer(obs_for(None))["__hit_meta__"]
            per_step.append(dict(stamp, step_idx=5 * step, hit_type=meta["hit_type"], start_t=meta["start_t"]))
        served.on_episode_end(success)
        served.on_task_end()
        journal.append(dict(stamp, phase="eval", status="done" if success else "failed", error=None))
    for name, rows in (("journal.jsonl", journal), ("per_step.jsonl", per_step)):
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = admit(root)
    assert report["ok"] and not report["global_problems"], report
    by_arm = {a["arm"]: a for a in plan["arms"]}
    summary = {arm: report["arms"][arm] for arm in by_arm}
    assert summary["warmreset_t0.2"] == {"expected": 2, "admitted": 2, "successes": 1, "measured_total_nfe": 12}
    assert summary["selfresetfinal_t0.2"] == {"expected": 2, "admitted": 2, "successes": 1,
                                              "measured_total_nfe": 72}


def test_rollout_drift_is_rejected_and_cli_writes_a_failed_report(tmp_path):
    root, plan = make_plan(tmp_path)
    build_driver(
        root,
        plan,
        bind_host="127.0.0.1",
        port=0,
        concurrency=2,
        ctl_factory=lambda _: None,
    )
    plan["rollout"]["seed"] += 1
    (root / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan changed"):
        admit(root)
    assert main(["admit", "--run-dir", str(root)]) == 2
    assert not json.loads((root / "admission.json").read_text())["ok"]


def test_cli_exports_rc_names_and_original_indices_without_simulator(tmp_path):
    out = tmp_path / "tasks.json"
    assert (
        main(
            [
                "tasks",
                "--env",
                "pi05_rc",
                "--task-names",
                "OpenDrawer,CloseFridge",
                "--episodes",
                "2",
                "--init-offset",
                "10",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text()) == [
        {"task_id": 0, "name": "OpenDrawer", "init_indices": [10, 11]},
        {"task_id": 1, "name": "CloseFridge", "init_indices": [10, 11]},
    ]


@pytest.mark.parametrize(
    "arm", ["full", "plain_k2", "../warmreset_t0.2", "warm_t0.2_n1", "midshoot_t0.2"]
)
def test_unsupported_arm_ids_fail_before_any_dispatch(arm):
    with pytest.raises(ValueError):
        arm_block("pi05_rc", arm, evidence_dir="/ev", namespace="ns")
