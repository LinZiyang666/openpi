"""Entry completion: full / plain_k / library-free self arms, GR00T per-K endpoints, pinned RoboCasa,
stable RoboCasa experiment id, init-pool digest, the env registry hook and backward compatibility."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from exp.step_diag import envs as E
from exp.warm_reset import envs as R
from exp.warm_reset.admit import admit, trusted_expected
from exp.warm_reset.conductor import WarmResetStrategy, build_driver, worker_agent
from exp.warm_reset.plan import config_of, prepare, read_plan, required_steps
from exp.warm_reset.run import _k_servers, main
from openpi.cache.config import is_library_free
from openpi.cache.warm_reset.evidence import ExpectedEpisode
from openpi.cache.warm_reset.types import MissSpec, WarmResetSpec
from openpi.conductor.task import EpisodeTask, ServerEndpoint
from tests.cache.warm_reset._support import pi05_config

PIN_MANIFEST = str(Path(__file__).resolve().parents[3] / "exp/robocasa365/config/pnp_pinned_objects.json")
ROLLOUT = {"replan_steps": 5, "seed": 7, "base_seed": 3000000, "layout": 1, "style": 2}
OLD_PLAN_KEYS = {"schema", "token", "env_id", "schedule_id", "k", "tasks", "servers", "evidence_dir", "namespace",
                 "rollout", "arms"}
OLD_ARM_KEYS = {"arm", "yaml_id", "yaml", "yaml_sha256", "start_t", "spec_digest"}


def base_yaml(tmp_path, env_id: str) -> Path:
    """A frozen-library base (the ``test_entry`` shape)."""
    env = E.ENVS[env_id]
    cfg = pi05_config(None, start_t=env.warm_ts[0])
    cfg.key_builder.type = "placeholder"
    cfg.write_policy.type = "never"
    cfg.backend.in_memory.preload_path = "/server/library.pkl"
    cfg.denoise_schedule = env.schedule_id
    path = tmp_path / f"base_{env_id}.yaml"
    path.write_text(yaml.safe_dump(dataclasses.asdict(cfg)))
    return path


def plan_of(tmp_path, env_id, arms, *, base=True, tasks=None, name="run", rollout=None, **kw):
    root = tmp_path / name
    plan = prepare(
        out=root, env_id=env_id, base_yaml=base_yaml(tmp_path, env_id) if base else None, arms=arms,
        tasks=tasks or [{"task_id": 2, "name": "task name", "init_indices": [4, 8]}],
        servers=kw.pop("servers", ["localhost:8000"]), evidence_root=str(tmp_path / "ev"), namespace="paired",
        rollout=dict(rollout or ROLLOUT), **kw,
    )
    return root, plan


def graph_tasks(plan, server=ServerEndpoint("localhost", 8000)):
    strategy = WarmResetStrategy(plan)
    strategy.plan([a["yaml_id"] for a in plan["arms"]], dict.fromkeys((a["yaml_id"] for a in plan["arms"]), server))
    return strategy.tasks


# ------------------------------------------------------------------
# Arm kinds
# ------------------------------------------------------------------


def test_miss_and_library_free_self_arms_need_no_base_yaml(tmp_path):
    root, plan = plan_of(tmp_path, "pi05_libero_10",
                         ["full", "plain_k2", "selfwarmreset_t0.2", "selfresetfinal_t0.2", "selfmidfinal_t0.2"],
                         base=False, self_trigger="always")
    assert read_plan(root) == plan
    arms = {a["arm"]: a for a in plan["arms"]}
    for aid in ("full", "plain_k2"):
        cfg = config_of(arms[aid]["yaml"])
        assert arms[aid]["kind"] == "miss" and arms[aid]["start_t"] is None
        assert arms[aid]["steps"] == cfg.miss.num_steps == (10 if aid == "full" else 2)
        assert arms[aid]["spec_digest"] == MissSpec.from_config(cfg.miss).digest()
        assert is_library_free(cfg) and not cfg.backend.in_memory.preload_path
    for aid in ("selfwarmreset_t0.2", "selfresetfinal_t0.2", "selfmidfinal_t0.2"):
        cfg = config_of(arms[aid]["yaml"])
        spec = WarmResetSpec.from_config(cfg.warm_reset)
        assert arms[aid]["kind"] == "self_only" and spec.always and spec.start_t == 0.2 == arms[aid]["start_t"]
        assert spec.seed_namespace == "paired" and is_library_free(cfg)
    assert graph_tasks(plan)[0].experiment == "libero_10"


def test_library_arms_still_need_the_base_yaml(tmp_path):
    with pytest.raises(ValueError, match="--base-yaml"):
        plan_of(tmp_path, "pi05_libero_10", ["warmreset_t0.2", "full"], base=False)
    with pytest.raises(ValueError, match="--base-yaml"):  # a verdict-triggered self arm reads the library
        plan_of(tmp_path, "pi05_libero_10", ["selfwarmreset_t0.2"], base=False)


@pytest.mark.parametrize("arm", ["plain_k0", "plain_k", "full_k2", "../full"])
def test_invalid_miss_arm_ids(tmp_path, arm):
    with pytest.raises(ValueError):
        plan_of(tmp_path, "pi05_libero_10", [arm], base=False)


def test_mixed_run_keeps_warm_arms_byte_identical(tmp_path):
    """Warm arms of a run with new-kind arms get exactly the yaml a warm-only run gets."""
    _, old = plan_of(tmp_path, "pi05_libero_10", ["warmreset_t0.2", "warm_t0.2"], name="old")
    _, new = plan_of(tmp_path, "pi05_libero_10", ["warmreset_t0.2", "warm_t0.2", "full", "selfresetfinal_t0.2"],
                     name="new", self_trigger="always")

    def norm(plan, arm):
        rec = next(a for a in plan["arms"] if a["arm"] == arm)
        return rec["yaml"].replace(plan["evidence_dir"], "EV")

    for arm in ("warmreset_t0.2", "warm_t0.2"):
        assert norm(old, arm) == norm(new, arm)
    assert "miss:" not in norm(old, "warmreset_t0.2") and "trigger" not in norm(old, "warmreset_t0.2")


def test_default_options_write_the_pre_registry_plan_layout(tmp_path):
    _, plan = plan_of(tmp_path, "pi05_rc", ["warmreset_t0.2", "warm_t0.2"])
    assert set(plan) == OLD_PLAN_KEYS
    assert all(set(a) == OLD_ARM_KEYS for a in plan["arms"])


def test_pre_registry_plans_replan_to_the_same_episode_tasks(tmp_path):
    """The EpisodeTask fields the pre-registry strategy wrote, per benchmark (admission replans with them)."""
    _, rc = plan_of(tmp_path, "pi05_rc", ["warmreset_t0.2"])
    task = graph_tasks(rc)[0]
    assert task.experiment == f"warm_reset_{rc['token']}"
    assert task.extra == {"num_trials_per_task": 2, "task_name": "task name", "teacher": "pi05",
                          "base_seed": 3000000, "layout": 1, "style": 2, "replan_steps": 5}
    _, groot_rc = plan_of(tmp_path, "groot_rc", ["warmreset_t0.75"], name="grc")
    assert graph_tasks(groot_rc)[0].extra["teacher"] == "groot_tp"
    _, lib = plan_of(tmp_path, "groot_libero_10", ["warmreset_t0.75_n1"], name="lib")
    task = graph_tasks(lib)[0]
    assert task.experiment == "libero_10" and task.extra == {"num_trials_per_task": 2}
    assert (task.bundle_id, task.orig_init_state_idx, task.episode_idx) == (task.yaml_id, 4, 0)


# ------------------------------------------------------------------
# GR00T per-step-count endpoints
# ------------------------------------------------------------------

GROOT_ARMS = ["full", "plain_k1", "plain_k2", "warmreset_t0.75_n1", "selfresetfinal_t0.5_n2"]


def test_groot_arms_are_pinned_to_their_step_count_endpoints(tmp_path):
    root, plan = plan_of(tmp_path, "groot_libero_10", GROOT_ARMS, self_trigger="always",
                         servers=["h:9000"], k_servers={1: ["h:9001"], 2: ["h:9002", "h:9003"]})
    assert read_plan(root) == plan
    assert [f"{s['host']}:{s['port']}" for s in plan["servers"]] == ["h:9000", "h:9001", "h:9002", "h:9003"]
    assert plan["server_steps"] == {"h:9000": 8, "h:9001": 1, "h:9002": 2, "h:9003": 2}
    ends = {a["arm"]: a["endpoints"] for a in plan["arms"]}
    assert ends == {"full": ["h:9000"], "plain_k1": ["h:9001"], "plain_k2": ["h:9002", "h:9003"],
                    "warmreset_t0.75_n1": ["h:9000"], "selfresetfinal_t0.5_n2": ["h:9000"]}
    placed = {t.yaml_id: f"{t.server_host}:{t.server_port}" for t in graph_tasks(plan, ServerEndpoint("h", 9002))}
    by_arm = {a["arm"]: placed[a["yaml_id"]] for a in plan["arms"]}
    assert by_arm == {"full": "h:9000", "plain_k1": "h:9001", "plain_k2": "h:9002",
                      "warmreset_t0.75_n1": "h:9000", "selfresetfinal_t0.5_n2": "h:9000"}
    agent = worker_agent(plan, server="h:9001", driver_host="d", driver_port=1, gpus=["0"], workers_per_gpu=1,
                         prefix="p")
    assert agent._specs[0].server_key == "h:9001"


def test_groot_miss_arm_without_its_endpoint_is_refused(tmp_path):
    with pytest.raises(ValueError, match="--denoising-steps 2"):
        plan_of(tmp_path, "groot_libero_10", ["plain_k2"], base=False, servers=["h:9000"],
                k_servers={1: ["h:9001"]})
    with pytest.raises(ValueError, match="--denoising-steps 1"):
        plan_of(tmp_path, "groot_rc", ["plain_k1"], base=False, name="rc")
    _, plan = plan_of(tmp_path, "groot_rc", ["full"], base=False, name="full_only")  # K = 4: the default servers
    assert "server_steps" not in plan and "endpoints" not in plan["arms"][0]


@pytest.mark.parametrize("k_servers,match", [
    ({8: ["h:9001"]}, "other than K=8"),
    ({1: ["h:9000"]}, "two step counts"),
    ({1: []}, "endpoints"),
])
def test_invalid_k_servers(tmp_path, k_servers, match):
    with pytest.raises(ValueError, match=match):
        plan_of(tmp_path, "groot_libero_10", ["full"], base=False, servers=["h:9000"], k_servers=k_servers)


def test_pi05_sets_miss_steps_per_bundle_not_per_endpoint(tmp_path):
    with pytest.raises(ValueError, match="GR00T option"):
        plan_of(tmp_path, "pi05_libero_10", ["plain_k2"], base=False, k_servers={2: ["h:9001"]})


def test_admission_refuses_a_dispatch_to_the_wrong_step_count(tmp_path):
    root, plan = plan_of(tmp_path, "groot_libero_10", ["full", "plain_k1"], base=False, servers=["h:9000"],
                         k_servers={1: ["h:9001"]})
    build_driver(root, plan, bind_host="127.0.0.1", port=0, concurrency=2, ctl_factory=lambda _: None)
    execution = json.loads((root / "execution.json").read_text())
    plain = next(a["yaml_id"] for a in plan["arms"] if a["arm"] == "plain_k1")
    assert {t["server_port"] for t in execution["tasks"] if t["yaml_id"] == plain} == {9001}
    for t in execution["tasks"]:
        if t["yaml_id"] == plain:
            t["server_port"] = 9000
    (root / "execution.json").write_text(json.dumps(execution))
    with pytest.raises(ValueError, match="another step count"):
        admit(root)


def test_cli_parses_k_servers():
    assert _k_servers("1=h:1, 2=h:2,2=h:3") == {1: ["h:1"], 2: ["h:2", "h:3"]}
    with pytest.raises(ValueError):
        _k_servers("h:1")


# ------------------------------------------------------------------
# RoboCasa: pinned objects and a stable experiment id
# ------------------------------------------------------------------

PNP = [{"task_id": 0, "name": "PickPlaceCounterToStove", "init_indices": [0, 1]}]
RC_OPTIONS = {"worker_python": "/island/python", "robocasa_cwd": "/sim", "egl_lib_dir": "/lib",
              "egl_vendor_dir": "/vendor", "connect_deadline_s": 60, "episode_deadline_s": 1500,
              "terminate_grace_s": 5, "max_cached_envs": 1}


def test_pinned_robocasa_freezes_the_pin_table_and_stamps_every_episode(tmp_path):
    from exp.robocasa365.episode_runner import PIN_EXTRA_KEYS
    from exp.robocasa365.pinned_objects import compute_pin_task_id, load_pin_manifest

    pin_id, table = load_pin_manifest(PIN_MANIFEST)
    root, plan = plan_of(tmp_path, "pi05_rc", ["warmreset_t0.2", "full"], tasks=PNP, pinned_objects=PIN_MANIFEST)
    assert read_plan(root) == plan and plan["pin_id"] == pin_id == E.canonical_pin_id()
    assert plan["pins"] == {"PickPlaceCounterToStove": table["PickPlaceCounterToStove"]}
    tasks = graph_tasks(plan)
    for task in tasks:
        assert all(key in task.extra for key in PIN_EXTRA_KEYS)
        assert task.extra["pin_task_id"] == compute_pin_task_id("PickPlaceCounterToStove", task.extra["pinned_objects"])
    env = R.get_env("pi05_rc")
    ident = env.adapter.identity(env, plan, tasks[0])
    assert ident == {"seed": 3000000 + tasks[0].orig_init_state_idx, "pin_id": pin_id,
                     "pin_task_id": tasks[0].extra["pin_task_id"]}
    assert env.adapter.pairing(env, plan, tasks[0])["pin_id"] == pin_id
    agent = worker_agent(plan, server="localhost:8000", driver_host="d", driver_port=1, gpus=["0"],
                         workers_per_gpu=1, prefix="p", rc_options=RC_OPTIONS, pinned_objects=PIN_MANIFEST)
    assert agent._spawn_fn.keywords["pinned_objects_path"] == str(Path(PIN_MANIFEST).resolve())
    with pytest.raises(ValueError, match="pinned plan needs --pinned-objects"):
        worker_agent(plan, server="localhost:8000", driver_host="d", driver_port=1, gpus=["0"],
                     workers_per_gpu=1, prefix="p", rc_options=RC_OPTIONS)


def test_unpinned_robocasa_refuses_a_pinned_agent_and_stays_unstamped(tmp_path):
    from exp.robocasa365.episode_runner import PIN_EXTRA_KEYS

    _, plan = plan_of(tmp_path, "pi05_rc", ["warmreset_t0.2"], tasks=PNP)
    assert "pin_id" not in plan and not any(k in graph_tasks(plan)[0].extra for k in PIN_EXTRA_KEYS)
    with pytest.raises(ValueError, match="unpinned plan refuses it"):
        worker_agent(plan, server="localhost:8000", driver_host="d", driver_port=1, gpus=["0"],
                     workers_per_gpu=1, prefix="p", rc_options=RC_OPTIONS, pinned_objects=PIN_MANIFEST)


def test_pin_options_are_refused_where_they_do_not_apply(tmp_path):
    with pytest.raises(ValueError, match="not supported by the libero adapter"):
        plan_of(tmp_path, "pi05_libero_10", ["warmreset_t0.2"], pinned_objects=PIN_MANIFEST)
    with pytest.raises(ValueError, match="no slot map"):
        plan_of(tmp_path, "pi05_rc", ["warmreset_t0.2"], name="x", pinned_objects=PIN_MANIFEST,
                tasks=[{"task_id": 0, "name": "CloseFridge", "init_indices": [0]}])


def test_stable_robocasa_experiment_id(tmp_path):
    _, plan = plan_of(tmp_path, "pi05_rc", ["selfwarmreset_t0.2"], experiment_id="wr_rc_main_v1")
    assert plan["experiment_id"] == "wr_rc_main_v1"
    assert {t.experiment for t in graph_tasks(plan)} == {"wr_rc_main_v1"}
    with pytest.raises(ValueError, match="not supported by the libero adapter"):
        plan_of(tmp_path, "pi05_libero_10", ["full"], base=False, name="l", experiment_id="x")
    with pytest.raises(ValueError, match="experiment id"):
        plan_of(tmp_path, "pi05_rc", ["full"], base=False, name="bad", experiment_id="a/b")


def test_stable_experiment_id_makes_self_seeds_common_across_runs(tmp_path):
    from openpi.cache.warm_reset.types import EpisodeDigestSeed

    seeds = []
    for name in ("a", "b"):
        _, plan = plan_of(tmp_path, "pi05_rc", ["selfwarmreset_t0.2"], name=name, experiment_id="wr_rc_main_v1")
        task = graph_tasks(plan)[0]
        policy = EpisodeDigestSeed("paired", ("experiment", "task", "orig_init_state_idx", "attempt"))
        seeds.append(policy.seed({"experiment": task.experiment, "task": "task name",
                                  "orig_init_state_idx": task.orig_init_state_idx, "attempt": 1}, 0))
    assert seeds[0] == seeds[1]


# ------------------------------------------------------------------
# LIBERO init-pool digest
# ------------------------------------------------------------------


def test_init_pool_digest_is_recorded(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "task.init").write_bytes(b"states")
    rollout = dict(ROLLOUT, init_states_dir=str(pool))
    _, plan = plan_of(tmp_path, "pi05_libero_10", ["full"], base=False, rollout=rollout)
    assert plan["init_pool_sha256"] == E.sha256_tree(pool)
    env = R.get_env("pi05_libero_10")
    assert env.adapter.pairing(env, plan, graph_tasks(plan)[0])["init_pool_sha256"] == E.sha256_tree(pool)
    with pytest.raises(ValueError, match="disagrees"):
        plan_of(tmp_path, "pi05_libero_10", ["full"], base=False, name="bad", rollout=rollout,
                init_pool_sha256="0" * 64)
    remote = dict(ROLLOUT, init_states_dir="/worker/only/pool")
    _, given = plan_of(tmp_path, "pi05_libero_10", ["full"], base=False, name="given", rollout=remote,
                       init_pool_sha256="ab" * 32)
    assert given["init_pool_sha256"] == "ab" * 32
    _, unknown = plan_of(tmp_path, "pi05_libero_10", ["full"], base=False, name="unknown", rollout=remote)
    assert unknown["init_pool_sha256"] is None
    assert env.adapter.pairing(env, unknown, graph_tasks(unknown)[0])["init_pool_sha256"] == "path:/worker/only/pool"


# ------------------------------------------------------------------
# Registry hook
# ------------------------------------------------------------------


class _ToyAdapter(R.BenchmarkAdapter):
    name = "toy"

    def validate_rollout(self, env, rollout):
        R._nonneg_int(rollout, "seed", "toy")

    def _task_pairs(self, env, *, task_ids, task_names):
        return [(0, "reach"), (1, "push")]

    def experiment(self, env, manifest):
        return "toy_mt"

    def episode_extra(self, env, manifest, task):
        return {"toy_task": task["name"]}

    def identity(self, env, manifest, episode):
        return {"seed": manifest["rollout"]["seed"]}

    def pairing(self, env, manifest, episode):
        return {"init_idx": episode.orig_init_state_idx, "env_seed": manifest["rollout"]["seed"]}

    def worker_agent(self, env, manifest, *, specs, driver_host, driver_port, options):
        return SimpleNamespace(specs=specs, options=options)


@pytest.fixture
def toy_env(monkeypatch):
    monkeypatch.setattr(R, "_REGISTRY", dict(R._REGISTRY))
    return R.register_env(R.EntryEnv(env_id="pi05_toy", policy="pi05", benchmark="toy_mt", action_horizon=5,
                                     k_full=10, schedule_id="pi05_v1", adapter=_ToyAdapter(),
                                     default_arms=("full", "plain_k2")))


def test_registered_env_drives_every_subcommand(tmp_path, toy_env):
    out = tmp_path / "tasks.json"
    assert main(["tasks", "--env", "pi05_toy", "--episodes", "2", "--out", str(out)]) == 0
    tasks = json.loads(out.read_text())
    assert [t["name"] for t in tasks] == ["reach", "push"]
    root = tmp_path / "run"
    assert main(["prepare", "--env", "pi05_toy", "--tasks", str(out), "--servers", "h:1",
                 "--server-evidence-root", str(tmp_path / "ev"), "--namespace", "ns", "--out", str(root),
                 "--self-trigger", "always", "--arms", "full,plain_k2,selfwarmreset_t0.2"]) == 0
    plan = read_plan(root)
    assert [a["arm"] for a in plan["arms"]] == ["full", "plain_k2", "selfwarmreset_t0.2"]
    task = graph_tasks(plan)[0]
    assert task.experiment == "toy_mt" and task.extra == {"num_trials_per_task": 2, "toy_task": "reach"}
    agent = worker_agent(plan, server="h:1", driver_host="d", driver_port=1, gpus=["0"], workers_per_gpu=2,
                         prefix="p")
    assert len(agent.specs) == 2 and agent.specs[0].replan_steps == 5
    with pytest.raises(ValueError, match="already registered"):
        R.register_env(dataclasses.replace(toy_env, adapter=_ToyAdapter()))
    with pytest.raises(ValueError, match="policy"):
        R.register_env(dataclasses.replace(toy_env, env_id="x", policy="act"))


def test_plugins_are_discovered_by_convention_and_failures_stay_local(tmp_path, monkeypatch):
    (tmp_path / "toy").mkdir()
    (tmp_path / "toy" / "warm_reset_env.py").write_text("raise ImportError('no simulator here')\n")
    (tmp_path / "other").mkdir()
    monkeypatch.setattr(R, "_EXP_ROOT", tmp_path)
    monkeypatch.setattr(R, "_discovered", False)
    monkeypatch.setattr(R, "_PLUGIN_ERRORS", {})
    imported = []

    def fake_import(name):
        imported.append(name)
        raise ImportError("no simulator here")

    monkeypatch.setattr(R.importlib, "import_module", fake_import)
    assert "pi05_libero_10" in R.env_ids()
    assert imported == ["exp.toy.warm_reset_env"]
    with pytest.raises(KeyError, match="no simulator here"):
        R.get_env("pi05_metaworld_like")


def test_repo_plugins_import(monkeypatch):
    """Every plugin present in this tree imports and registers without error."""
    monkeypatch.setattr(R, "_discovered", False)
    monkeypatch.setattr(R, "_PLUGIN_ERRORS", {})
    R.env_ids()
    assert R._PLUGIN_ERRORS == {}


# ------------------------------------------------------------------
# End to end: serve, admit, analyse (Pi0.5 CPU stub)
# ------------------------------------------------------------------


def _serve_run(tmp_path, root, plan, *, success=lambda i: i % 2 == 0, decisions=3):
    """Serve every dispatched episode through the stack ``_wrap_policy`` builds for the arm's yaml."""
    from examples.libero.episode_runner import _episode_extra_metadata
    from openpi.cache.config import build_cache_components
    from openpi.cache.interceptor import InferenceInterceptor
    from openpi.cache.orchestrator import CacheOrchestrator
    from openpi.cache.timing import SystemTimer
    from openpi.cache.warm_reset.pi05 import build_pi05_warm_reset
    from tests.cache.test_interceptor import FakePolicy
    from tests.cache.warm_reset._support import Pi05Model, obs_for

    build_driver(root, plan, bind_host="127.0.0.1", port=0, concurrency=2, ctl_factory=lambda _: None)
    execution = json.loads((root / "execution.json").read_text())
    arms = {a["yaml_id"]: a for a in plan["arms"]}
    names = {t["task_id"]: t["name"] for t in plan["tasks"]}
    journal, per_step = [], []
    for i, raw in enumerate(execution["tasks"]):
        task, ok = EpisodeTask(**raw), success(i)
        arm = arms[task.yaml_id]
        yaml_path = tmp_path / f"{task.yaml_id}.yaml"
        yaml_path.write_text(arm["yaml"], encoding="utf-8")
        cfg = config_of(arm["yaml"])
        comp = build_cache_components(cfg)
        orch = CacheOrchestrator(storage=comp["storage"], key_builder=comp["key_builder"], gates=comp["gates"],
                                 judges=comp["judges"], search_strategies=comp["search_strategies"],
                                 timer=comp["timer"], write_policy=comp.get("write_policy"))
        parts = build_pi05_warm_reset(cfg, bundle_id=task.bundle_id, yaml_id=task.yaml_id, yaml_path=str(yaml_path))
        served = parts.wrap(InferenceInterceptor(FakePolicy(Pi05Model()), timer=SystemTimer(enabled=False),
                                                 orchestrator=orch, eager=True, bundle_id=task.bundle_id,
                                                 **parts.interceptor_kwargs()))
        served.on_episode_start(experiment=task.experiment, task=names[task.task_id], episode_id=task.episode_idx,
                                extra_metadata=_episode_extra_metadata(task))
        stamp = {"run_id": execution["run_id"], "task_uid": task.task_uid, "yaml_id": task.yaml_id,
                 "attempt": task.attempt, "accepted": True, "success": ok}
        for step in range(decisions):
            meta = served.infer(obs_for(None))["__hit_meta__"]
            per_step.append(dict(stamp, step_idx=5 * step, hit_type=meta["hit_type"], start_t=meta["start_t"]))
        served.on_episode_end(ok)
        served.on_task_end()
        journal.append(dict(stamp, phase="eval", status="done" if ok else "failed", error=None))
    for name, rows in (("journal.jsonl", journal), ("per_step.jsonl", per_step)):
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


E2E_ARMS = ["full", "plain_k2", "selfwarmreset_t0.2", "selfresetfinal_t0.2"]


def test_library_free_run_is_admitted_with_measured_nfe(tmp_path):
    root, plan = plan_of(tmp_path, "pi05_libero_10", E2E_ARMS, base=False, self_trigger="always")
    _serve_run(tmp_path, root, plan)
    report = admit(root)
    assert report["ok"] and not report["global_problems"], report
    assert report["arms"]["full"] == {"expected": 2, "admitted": 2, "successes": 1, "measured_total_nfe": 60}
    assert report["arms"]["plain_k2"]["measured_total_nfe"] == 12
    assert report["arms"]["selfwarmreset_t0.2"]["measured_total_nfe"] == 72  # (K + N) x 3 x 2
    by_arm = {}
    for rep in report["episodes"]:
        by_arm.setdefault(rep["arm"], []).append(rep)
    assert all(r["miss_nfe"] == 30 and r["evidence_scope"] == "server+worker" for r in by_arm["full"])
    assert all("miss_nfe" not in r for r in by_arm["selfwarmreset_t0.2"])


def test_worker_rows_of_another_kind_are_rejected(tmp_path):
    _, plan = plan_of(tmp_path, "pi05_libero_10", ["full", "selfwarmreset_t0.2"], base=False,
                         self_trigger="always")
    strategy = WarmResetStrategy(plan)
    strategy.plan([a["yaml_id"] for a in plan["arms"]],
                  {a["yaml_id"]: ServerEndpoint("localhost", 8000) for a in plan["arms"]})
    for arm, wrong in ((plan["arms"][0], {"hit_type": "WARM_START", "start_t": 0.2}),
                       (plan["arms"][1], {"hit_type": "MISS", "start_t": None})):
        task = next(t for t in strategy.tasks if t.yaml_id == arm["yaml_id"])
        term = {"task_uid": task.task_uid, "yaml_id": task.yaml_id, "run_id": "run", "phase": "eval",
                "status": "done", "success": True, "accepted": True, "attempt": 1}
        right = {"hit_type": "MISS", "start_t": None} if arm["arm"] == "full" else {"hit_type": "SELF_ONLY",
                                                                                    "start_t": 0.2}
        args = {"task": task, "task_name": "task name", "arm": arm, "manifest": plan, "run_id": "run",
                "terminals": [term]}
        good = trusted_expected(**args, per_step=[dict(term, step_idx=0, **right)])
        assert isinstance(good, ExpectedEpisode)
        with pytest.raises(ValueError, match="requested arm"):
            trusted_expected(**args, per_step=[dict(term, step_idx=0, **wrong)])


def _write_admission(root) -> dict:
    report = admit(root)
    (root / "admission.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_analysis_pairs_framework_cells_and_merges_segments(tmp_path):
    from exp.warm_reset import analysis as AN

    tasks_a = [{"task_id": 2, "name": "task name", "init_indices": [0, 1]}]
    tasks_b = [{"task_id": 2, "name": "task name", "init_indices": [2, 3]}]
    runs = []
    for name, tasks in (("seg_a", tasks_a), ("seg_b", tasks_b)):
        root, plan = plan_of(tmp_path, "pi05_libero_10", E2E_ARMS, base=False, self_trigger="always",
                             tasks=tasks, name=name)
        _serve_run(tmp_path / name, root, plan, success=lambda i: i % 3 != 0)
        assert _write_admission(root)["ok"]
        runs.append(AN.load_run(root))
    res = AN.analyze_runs(runs)
    assert list(res) == ["m2"]
    panel = res["m2"]
    assert panel["status"] == "complete" and panel["missing_arms"] == []
    task = panel["tasks"]["task name"]
    assert not task["comparison_problems"]
    cells = task["cells"]
    assert set(cells) == set(E2E_ARMS)
    assert cells["full"]["n"] == 4 and cells["full"]["mean_executed_steps"] == 10.0
    assert cells["plain_k2"]["equal_nfe"] and cells["plain_k2"]["mean_executed_steps"] == 2.0
    sw = cells["selfwarmreset_t0.2"]
    assert sw["complete"] and sw["equal_nfe"] and sw["nfe_per_decision"] == {"continuation": 2.0,
                                                                             "self_start": 10.0, "total": 12.0}
    assert sw["episode_nfe_mean"]["total"] == 36.0 and sw["framework"] == "warm_reset"
    assert set(task["paired_deltas"]) == {"selfwarmreset_t0.2 - full", "selfwarmreset_t0.2 - plain_k2",
                                          "selfresetfinal_t0.2 - full", "selfresetfinal_t0.2 - plain_k2"}
    assert task["paired_deltas"]["selfwarmreset_t0.2 - full"]["n"] == 4
    md = AN.markdown(res)
    assert "status: **COMPLETE**" in md and "| selfwarmreset_t0.2 | 4 |" in md
    lengths = AN.success_lengths(runs)
    row = lengths["decisions"]["pi05_libero_10_m2"]["arms"]["selfwarmreset_t0.2"]
    assert row["paired"]["n_pairs"] >= 1 and set(lengths) == {"decisions"}
    out = tmp_path / "a.json"
    assert AN.main(["--run-dir", str(runs[0]["root"]), "--run-dir", str(runs[1]["root"]),
                    "--out-json", str(out), "--out-md", str(tmp_path / "a.md"),
                    "--success-length-out", str(tmp_path / "len.json")]) == 0
    assert json.loads(out.read_text())["m2"]["tasks"]["task name"]["cells"]["full"]["n"] == 4


def test_analysis_refuses_runs_of_different_environments(tmp_path):
    from exp.warm_reset import analysis as AN

    runs = []
    for env_id in ("pi05_libero_10", "pi05_libero_spatial"):
        root, plan = plan_of(tmp_path, env_id, ["full"], base=False, name=env_id)
        _serve_run(tmp_path / env_id, root, plan)
        _write_admission(root)
        runs.append(AN.load_run(root))
    with pytest.raises(ValueError, match="different environments"):
        AN.analyze_runs(runs)


def test_analysis_needs_this_executions_admission(tmp_path):
    from exp.warm_reset import analysis as AN

    root, plan = plan_of(tmp_path, "pi05_libero_10", ["full"], base=False)
    _serve_run(tmp_path, root, plan)
    with pytest.raises(ValueError, match="admission.json"):
        AN.load_run(root)
    report = _write_admission(root)
    report["run_id"] = "other"
    (root / "admission.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not this execution"):
        AN.load_run(root)


def _synthetic_groot_run(tmp_path, successes: dict) -> dict:
    """A GR00T LIBERO framework run with a synthetic admission (the analysis reads admission.json only)."""
    from exp.warm_reset import analysis as AN
    from tests.exp.step_diag.test_libero_selfstart import TASK_NAMES

    tasks = [{"task_id": 0, "name": TASK_NAMES[0], "init_indices": list(range(len(next(iter(successes.values())))))}]
    root, plan = plan_of(tmp_path, "groot_libero_10", list(successes), base=False, self_trigger="always",
                         tasks=tasks, servers=["h:9000"], k_servers={1: ["h:9001"]},
                         rollout=dict(ROLLOUT, init_states_dir="/worker/pool"), init_pool_sha256="pool")
    build_driver(root, plan, bind_host="127.0.0.1", port=0, concurrency=2, ctl_factory=lambda _: None)
    execution = json.loads((root / "execution.json").read_text())
    arms = {a["yaml_id"]: a for a in plan["arms"]}
    episodes = []
    for raw in execution["tasks"]:
        task = EpisodeTask(**raw)
        aid = arms[task.yaml_id]["arm"]
        ok = bool(successes[aid][task.orig_init_state_idx])
        steps = 8 if aid == "full" else 1
        rep = {"task_uid": task.task_uid, "arm": aid, "admitted": True, "success": ok, "attempt": 1,
               "n_decisions": 3, "problems": {}, "evidence_scope": "server+worker",
               "continuation_nfe": None if aid in ("full", "plain_k1") else 3,
               "self_start_nfe": 24 if aid.startswith("self") else 0, "total_nfe": 3 * steps}
        if aid in ("full", "plain_k1"):
            rep["miss_nfe"] = 3 * steps
        if aid.startswith("self"):
            rep["total_nfe"] = 27
        episodes.append(rep)
    (root / "admission.json").write_text(json.dumps({
        "schema": "warm_reset_admission_v1", "run_id": execution["run_id"], "ok": True, "global_problems": {},
        "arms": {}, "episodes": episodes}))
    return AN.load_run(root)


def test_framework_cells_never_pair_with_step_diag_cells_unless_flagged(tmp_path):
    from exp.warm_reset import analysis as AN
    from tests.exp.step_diag.test_libero_selfstart import _write_libero_arm

    run = _synthetic_groot_run(tmp_path / "fw", {"full": [1, 1, 0, 1], "plain_k1": [0, 1, 0, 0],
                                                 "selfmidreset_t0.75_n1": [1, 1, 1, 0]})
    sd_arms, sd_srv = tmp_path / "sd_arms", tmp_path / "sd_srv"
    _write_libero_arm(sd_arms, sd_srv, "selfmidreset_t0.75_n1", {0: [1, 0, 1, 0]})
    env = R.get_env("groot_libero_10")
    args = SimpleNamespace(step_diag_arms="selfmidreset_t0.75_n1", step_diag_root=str(sd_arms),
                           step_diag_server_rows=str(sd_srv))
    source = AN._step_diag_source(args, env)
    plain = AN.analyze_runs([run], step_diag=source)["m1"]
    task = next(iter(plain["tasks"].values()))
    assert {"selfmidreset_t0.75_n1", "step_diag:selfmidreset_t0.75_n1"} <= set(task["cells"])
    assert task["cells"]["step_diag:selfmidreset_t0.75_n1"]["framework"] == "step_diag"
    assert not any("step_diag:" in k for k in task["paired_deltas"])
    assert "selfmidreset_t0.75_n1 - full" in task["paired_deltas"]
    flagged = AN.analyze_runs([run], step_diag=source, allow_cross_framework=True)["m1"]
    task = next(iter(flagged["tasks"].values()))
    delta = task["paired_deltas"]["selfmidreset_t0.75_n1 - step_diag:selfmidreset_t0.75_n1 [cross-framework]"]
    assert delta["n"] == 4 and delta["point"] == pytest.approx(0.25)
    macro = flagged.get("macro") or {"paired_deltas": {}}
    assert not any("cross-framework" in k for k in macro["paired_deltas"])


@pytest.mark.parametrize("env_id", ["groot_libero_10", "groot_rc"])
def test_groot_library_free_arm_yamls_pass_the_serving_guards_of_their_endpoint(tmp_path, env_id):
    """Every library-free GR00T arm yaml loads, builds its storage and passes the GR00T guard
    against the live step count of the endpoint it is pinned to (and fails on another)."""
    from openpi.cache.config import ConfigValidationError, build_shared_storage
    from openpi.cache.groot.load_guard import (
        validate_artifact_identity,
        validate_groot_cache_config,
    )

    k = E.ENVS[env_id].k_full
    self_arm = "selfresetfinal_t0.5_n2" if env_id == "groot_libero_10" else "selfresetfinal_t0.5"
    _, plan = plan_of(tmp_path, env_id, ["full", "plain_k1", "plain_k2", self_arm], base=False,
                      self_trigger="always", servers=["h:9000"], k_servers={1: ["h:9001"], 2: ["h:9002"]})
    env = R.get_env(env_id)
    for arm in plan["arms"]:
        cfg = config_of(arm["yaml"])
        live = required_steps(env, arm["arm"])
        assert is_library_free(cfg)
        validate_groot_cache_config(cfg, allow_hysteresis_gate=True, num_inference_timesteps=live)
        storage = build_shared_storage(cfg)
        validate_artifact_identity(storage, cfg)
        wrong = 2 if live != 2 else k
        with pytest.raises(ConfigValidationError):
            validate_groot_cache_config(cfg, allow_hysteresis_gate=True, num_inference_timesteps=wrong)
