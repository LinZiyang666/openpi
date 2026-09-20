"""G2 regression coverage for execution seams and end-to-end evidence admission."""

from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from exp.step_diag import recorder as R
from exp.step_diag import groot as G
from exp.step_diag import run_diag as D
from exp.step_diag.run_libero_diag import LiberoDiagRunner, LiberoDiagStrategy
from exp.step_diag.analysis import aggregate_arms as B
from exp.step_diag.analysis import analyze_shadow as A
from openpi.conductor import ServerEndpoint
from openpi.conductor.task import EpisodeTask
from tests.exp.step_diag.test_aggregate import _write_arm
from tests.exp.step_diag.test_groot_policy import (
    _FakeRunner,
    _FakePolicy,
    _spec,
    _start,
    _rows,
)
from tests.exp.step_diag.test_worker_entry import _FakeClient, _FakeEnv


def test_groot_plain_one_step_does_not_construct_snapshot_schedule(tmp_path):
    runner = _FakeRunner(k=1)
    runner.live_schedule = lambda: (_ for _ in ()).throw(
        AssertionError("one-step has no snapshot schedule")
    )
    rec = R.DiagRecorder(
        _spec(mode="plain", arm_id="plain_k1", k_set=(), warm_ts=(), exec_steps=1),
        tmp_path,
    )
    policy = G.GrootDiagPolicy(
        _FakePolicy(), runner, orchestrator=None, diag=rec, schedule=None, shadow=False
    )
    _start(policy)
    policy.get_action({"state": np.zeros(4)})
    policy.on_episode_end(True)
    assert _rows(rec)[0]["executed_steps"] == 1
    assert _rows(rec)[0]["schedule_id"] == "groot_n15_k1_v1"


def test_groot_batch_of_one_uses_production_batch_contract():
    assert G._is_batched(
        {"video.front": np.zeros((1, 1, 8, 8, 3)), "state.x": np.zeros((1, 1, 4))}
    )
    assert not G._is_batched(
        {"video.front": np.zeros((1, 8, 8, 3)), "state.x": np.zeros((1, 4))}
    )


def test_libero_strategy_freezes_pool_and_preserves_task_ids():
    args = SimpleNamespace(
        env_id="pi05_libero_10", experiment_id="formal", config_sha="cfg"
    )
    strategy = LiberoDiagStrategy(
        args, [f"task{i}" for i in range(10)], "pool", "launch"
    )
    server = ServerEndpoint("host", 1)
    graph = strategy.plan([], {y: server for y in strategy.yaml_ids})
    assert len(strategy.expected()) == 100
    stage = graph.stages[strategy.yaml_ids[9]]
    assert (
        stage.episodes[-1].task_id == 9 and stage.episodes[-1].orig_init_state_idx == 9
    )
    assert stage.episodes[-1].extra["init_pool_sha256"] == "pool"
    different = LiberoDiagStrategy(args, strategy.names, "different-pool", "launch")
    assert strategy.run_id != different.run_id


def test_libero_runner_counts_stamps_and_checks_actual_pool():
    client, env = _FakeClient(), _FakeEnv()
    args = SimpleNamespace(seed=7)
    task = EpisodeTask(
        task_uid="u",
        yaml_id="y",
        phase="eval",
        experiment="libero_10",
        task_id=2,
        episode_idx=3,
        orig_init_state_idx=3,
        server_host="host",
        server_port=1,
        bundle_id="default",
        attempt=2,
        extra=dict(
            num_trials_per_task=10,
            launch_id="launch",
            arm_id="shadow",
            experiment_id="formal",
            config_sha="cfg",
            init_pool_sha256="pool",
            seed=7,
        ),
    )

    def episode(env, client, *args, **kwargs):
        env.reset()
        client.infer({})
        env.step(None)
        client.infer({})
        env.step(None)
        return True, [], [], [], 2

    runner = LiberoDiagRunner(
        args,
        lambda _: (env, None, "task", 10),
        client_factory=lambda _: client,
        pool_sha="pool",
        run_episode_fn=episode,
    )
    result = runner.run(task, lambda *args: None)
    summary = next(r for r in result.per_step_rows if r.get("row") == "episode_summary")
    assert (
        summary["n_decisions"] == 2
        and summary["n_env_steps"] == 2
        and summary["success"]
    )
    stamp = client.named("episode_start")[0]["extra_metadata"]
    assert (
        stamp["task_uid"] == "u"
        and stamp["attempt"] == 2
        and stamp["init_pool_sha256"] == "pool"
    )
    with pytest.raises(ValueError, match="pool bytes"):
        runner.run(
            replace(task, extra={**task.extra, "init_pool_sha256": "other"}),
            lambda *args: None,
        )


def test_task_server_mapping_does_not_depend_on_arm_or_episode_budget():
    slots = [ServerEndpoint("host", p) for p in (1, 2, 3)]
    mappings = []
    for arm, n in [("full", 50), ("plain_k2", 100)]:
        strategy = D.StepDiagStrategy(
            arm_id=arm,
            experiment_id="formal",
            config_sha=arm,
            lane="main",
            teacher="pi05",
            layout=1,
            style=1,
            base_seed=2_000_000,
            replan_steps=5,
            slots=slots,
            tasks=[("CloseFridge", 50), ("OpenDrawer", n)],
        )
        graph = strategy.plan([], {y: slots[0] for y in strategy.yaml_ids})
        mappings.append(
            {s.episodes[0].task_id: s.server.key for s in graph.stages.values()}
        )
    assert mappings[0] == mappings[1]


@pytest.mark.parametrize(
    "defect", ["missing_manifest", "missing_worker_summary", "duplicate_finalize"]
)
def test_new_evidence_requirements_have_a_positive_control(tmp_path, defect):
    arms, rows = tmp_path / "arms", tmp_path / "rows"
    _write_arm(arms, rows, "plain_k2", "plain", 2, {"CloseFridge": [1, 0]})
    arm = B.load_arm(arms / "pi05/plain_k2")
    server = B.load_server_rows(rows / "pi05/plain_k2")
    assert B.cell_admission(arm, server, "CloseFridge", kind="plain", m=2)["equal_nfe"]
    uid = next(iter(arm["expected"]))
    if defect == "missing_manifest":
        server[(uid, 1)]["manifest"] = None
    elif defect == "missing_worker_summary":
        arm["summaries"].pop(uid)
    else:
        server[(uid, 1)]["duplicate_finalize"] = True
    assert not B.cell_admission(arm, server, "CloseFridge", kind="plain", m=2)[
        "equal_nfe"
    ]


def test_runtime_and_worker_facts_are_notes_not_gates(tmp_path):
    """Workers on different GPU slots / hosts and a server restarted on another runtime stay
    admissible (owner ruling 2026-09-12); a different checkpoint or experiment does not."""
    arms, rows = tmp_path / "arms", tmp_path / "rows"
    _write_arm(arms, rows, "plain_k2", "plain", 2, {"CloseFridge": [1, 0, 1, 0]})
    arm = B.load_arm(arms / "pi05/plain_k2")
    server = B.load_server_rows(rows / "pi05/plain_k2")
    for i, (uid, attempts) in enumerate(arm["summaries"].items()):
        attempts[1]["worker_runtime"] = {**attempts[1]["worker_runtime"], "gpu_slot": str(i % 2), "host": f"timan10{7 + i % 2}"}
    cell = B.cell_admission(arm, server, "CloseFridge", kind="plain", m=2)
    assert cell["equal_nfe"] and len(cell["worker_identities"]) == 2
    cells = {"plain_k2": cell, "full": dict(cell), "warm_t0.2": dict(cell)}
    assert B.comparison_problems(cells, {a: arm for a in cells}) == []
    assert B.comparison_notes(cells) == {"worker_islands": 2, "serving_runtimes": 1, "uniform": False}
    # a different checkpoint identity on one arm breaks the comparison
    other = dict(cell, comparison_identities=["other-model"])
    assert B.comparison_problems({**cells, "full": other}, {a: arm for a in cells}) == ["comparison_identities_mismatch"]


def test_shadow_rng_distinguishes_libero_init_indices(tmp_path):
    spec = R.DiagSpec(
        experiment_id="formal",
        env_id="pi05_libero_10",
        arm_id="shadow",
        mode="shadow",
        k_full=10,
        k_set=(1,),
        action_shape=(10, 32),
        n_dense_extra=0,
    )
    rec = R.DiagRecorder(spec, tmp_path)
    for idx in (0, 1):
        rec.begin_episode(
            R.EpisodeIdentity("libero_10", "task", idx, f"u{idx}", 1, 0, idx, None)
        )
        rec.record(
            a_exec=torch.zeros(10, 32),
            executed_steps=10,
            n_stage3_calls=1,
            hit_type="MISS",
            start_t=None,
            schedule_id="pi05_v1",
            sample=lambda z, k: z,
        )
        rec.finalize_episode(True)
    rows = [json.loads(line) for line in rec.rows_path.read_text().splitlines()]
    assert rows[0]["noise_ids"] != rows[2]["noise_ids"]


def test_all_four_bootstrap_quantities_gate_degenerate_verdict():
    from tests.exp.step_diag.test_aggregate import _macro

    macro = _macro(0.4, 0.35)
    macro["H25"]["degenerate"] = True
    assert (
        B.verdict(macro, {"flat": {"lower": -0.01, "upper": 0.01}}, True)["verdict"]
        == "inconclusive"
    )


def test_shadow_joins_real_recorder_to_conductor_authority(tmp_path):
    arms, rows_root = tmp_path / "arms", tmp_path / "rows"
    _write_arm(
        arms, rows_root, "shadow", "plain", 10, {"CloseFridge": [1] * 10}, n_decisions=1
    )
    driver_dir, rows_dir = arms / "pi05/shadow", rows_root / "pi05/shadow"
    arm = B.load_arm(driver_dir)
    launch = next(iter(arm["launches"].values()))
    (rows_dir / "rows_x.jsonl").unlink()
    spec = R.DiagSpec(
        experiment_id="e",
        env_id="pi05_rc",
        arm_id="shadow",
        mode="shadow",
        k_full=10,
        k_set=(1, 2, 3, 5),
        warm_ts=(0.1, 0.2, 0.3),
        action_shape=(50, 32),
        config_sha=launch["config_sha"],
    )
    rec = R.DiagRecorder(spec, rows_dir, launch_id="server")
    for ident in arm["expected"].values():
        idx = ident["init_idx"]
        ep = R.EpisodeIdentity(
            "robocasa365",
            ident["task"],
            idx,
            ident["task_uid"],
            1,
            0,
            idx,
            ident["env_seed"],
            lane="main",
            extra=dict(
                launch_id="L0",
                arm_id="shadow",
                experiment_id="e",
                config_sha=spec.config_sha,
                layout=1,
                style=1,
            ),
        )
        rec.begin_episode(ep)
        rec.record(
            a_exec=torch.zeros(50, 32),
            executed_steps=10,
            n_stage3_calls=1,
            hit_type="MISS",
            start_t=None,
            schedule_id="pi05_v1",
            sample=lambda z, k: z + k,
        )
        rec.finalize_episode(True)
    weights_info = {"env_id": "pi05_rc", "library_sha256": "library"}
    accepted, rejected = A.accepted_shadow_episodes(
        driver_dir, rows_dir, "pi05_rc", weights_info
    )
    assert len(accepted) == 10 and rejected == {}
    res = A.admit_and_measure(
        A.read_rows([rec.rows_path]),
        rows_dir,
        torch.ones(32),
        torch.arange(32) < 12,
        h_exec=5,
        n_primary=4,
        min_coverage=0.9,
        min_episodes=8,
        journal_accepted=accepted,
    )
    assert len(res["episodes"]) == 10 and res["per_task"]["CloseFridge"]["published"]
    journal = driver_dir / "journal_0.jsonl"
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    records[0]["run_id"] = "wrong-driver"
    journal.write_text("".join(json.dumps(r) + "\n" for r in records))
    accepted, rejected = A.accepted_shadow_episodes(
        driver_dir, rows_dir, "pi05_rc", weights_info
    )
    assert len(accepted) == 9 and rejected["driver_run_mismatch"] == 1
    with pytest.raises(ValueError, match="weights"):
        A.accepted_shadow_episodes(
            driver_dir, rows_dir, "pi05_rc", {"env_id": "pi05_rc"}
        )


def test_pi05_shadow_retrieval_error_keeps_teacher_action(tmp_path):
    from exp.step_diag.pi05 import Pi05DiagInterceptor
    from openpi.cache.timing import SystemTimer
    from tests.cache.conftest import make_orchestrator
    from tests.cache.test_interceptor import FakePolicy, _make_obs
    from tests.exp.step_diag.test_pi05_interceptor import (
        _StagedFakeModel,
        _episode_start,
        _spec,
    )

    model = _StagedFakeModel()
    orch, _, _ = make_orchestrator()

    def broken(*args, **kwargs):
        raise RuntimeError("retrieval failed")

    orch.check = broken
    rec = R.DiagRecorder(_spec("shadow"), tmp_path)
    served = Pi05DiagInterceptor(
        FakePolicy(model),
        timer=SystemTimer(enabled=False),
        orchestrator=orch,
        diag=rec,
        mode="shadow",
    )
    _episode_start(served)
    noise = torch.randn(1, 50, 32)
    action = served.infer(_make_obs(), noise=noise.numpy())["actions"]
    expected = (
        _StagedFakeModel().run_stage3(None, noise=noise, num_steps=10).action_chunk[0]
    )
    np.testing.assert_array_equal(np.asarray(action), expected.numpy())
    served.on_episode_end(True)
    row = json.loads(rec.rows_path.read_text().splitlines()[0])
    assert row["status"] == "error" and "retrieval failed" in row["error_reason"]


def test_missing_label_strata_keep_absent_server_episodes_and_exclude_retries(tmp_path):
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    _write_arm(
        arms, srv, "shadow", "plain", 10, {"CloseFridge": [1, 0, 1]},
        drop_server=("CloseFridge:1",), drop_journal=("CloseFridge:2",),
    )
    arm = B.load_arm(arms / "pi05" / "shadow")
    rows = A.read_rows([srv / "pi05" / "shadow" / "rows_x.jsonl"])
    admitted = [{"task_uid": "CloseFridge:0", "attempt": 1, "n_valid": 3}]
    baseline = A.authoritative_stratified(arm, rows, admitted)
    assert baseline["success"]["episodes"] == baseline["success"]["admitted"] == 1
    assert baseline["success"]["valid_labels"] == 3
    assert baseline["failure"]["episodes"] == baseline["failure"]["rejected"] == 1
    assert baseline["failure"]["missing_decision_rows"] == 3
    assert baseline["failure"]["missing_finalize_episodes"] == 1
    assert baseline["unknown"]["episodes"] == 1
    assert baseline["unknown"]["unknown_decision_count_episodes"] == 1
    retry = [dict(r, attempt=2, outcome=False) for r in rows if r["task_uid"] == "CloseFridge:0"]
    assert A.authoritative_stratified(arm, rows + retry, admitted) == baseline


def test_libero_worker_imports_without_serving_dependencies():
    import subprocess
    import sys

    source = """
import sys, importlib.abc
class RejectServingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('openpi.cache', 'jax', 'gr00t')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectServingImports())
import exp.step_diag.run_libero_diag
"""
    subprocess.run(
        [sys.executable, "-c", source], check=True, capture_output=True, text=True
    )


def test_groot_rc_served_builder_accepts_plain_one_step(tmp_path, monkeypatch):
    import sys
    import types
    from exp.robocasa365 import serve_groot_n15 as server
    from exp.step_diag import serve_diag_groot as serving

    packages = {
        name: types.ModuleType(name)
        for name in ("gr00t", "gr00t.model", "gr00t.model.policy")
    }
    packages["gr00t"].model = packages["gr00t.model"]
    packages["gr00t.model"].policy = packages["gr00t.model.policy"]

    class Policy:
        def __init__(self, denoising_steps):
            self.model = SimpleNamespace(
                action_head=SimpleNamespace(
                    num_inference_timesteps=denoising_steps,
                    config=SimpleNamespace(action_horizon=16, action_dim=32),
                )
            )

    packages["gr00t.model.policy"].Gr00tPolicy = Policy
    for name, package in packages.items():
        monkeypatch.setitem(sys.modules, name, package)

    def runner(policy, *args):
        def no_snapshot_schedule():
            raise AssertionError("plain k=1 cannot construct a snapshot schedule")

        return None, SimpleNamespace(
            _model=policy.model, live_schedule=no_snapshot_schedule
        )

    monkeypatch.setattr(serving, "_orchestrator_and_runner", runner)
    monkeypatch.setattr(server, "_build_served_policy", server._build_served_policy)
    args, _ = serving.parse(
        [
            "--benchmark",
            "rc",
            "--env-id",
            "groot_rc",
            "--mode",
            "plain",
            "--exec-steps",
            "1",
            "--arm-id",
            "plain_k1",
            "--experiment-id",
            "formal",
            "--diag-out",
            str(tmp_path),
        ]
    )
    rec = R.DiagRecorder(
        R.DiagSpec(
            experiment_id="formal",
            env_id="groot_rc",
            arm_id="plain_k1",
            mode="plain",
            k_full=4,
            k_set=(),
            action_shape=(16, 32),
        ),
        tmp_path,
    )
    serving.install_rc(args, rec)
    policy = packages["gr00t.model.policy"].Gr00tPolicy(denoising_steps=4)
    served, description = server._build_served_policy(policy, SimpleNamespace())
    assert served._live == 1 and "k=1" in description
