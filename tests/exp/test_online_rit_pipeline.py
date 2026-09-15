"""Production producer/consumer regressions for the online RIT review repairs."""

from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from exp.online_rit.common import SCHEDULE, canonical_sha256, load_ledger, sha256_file, write_json, write_jsonl
from tests.exp.test_online_rit_exp import complete_cost_record, parity_fixture, synthetic_table


def test_actual_calibration_producers_feed_init_and_reject_foreign_scales(tmp_path, monkeypatch):
    from exp.online_rit import fit_init_curves, replay_sim
    from exp.online_rit.analysis import signal_check

    table, scales, signal, knots, replay, parity = [tmp_path / n for n in ("table.jsonl", "scales.npz", "signal.json", "knots.json", "replay.json", "parity.json")]
    write_jsonl(table, synthetic_table(n_traj=64, steps=40, n_library=4))
    gate = parity_fixture()
    np.savez(scales, meta_json=json.dumps({"library_sha256": gate["identity"]["library_sha256"], "schedule_id": SCHEDULE.schedule_id}))
    gate["identity"]["scales_sha256"] = sha256_file(scales)
    write_json(parity, gate)
    write_json(str(table) + ".record.json", {"smoke": False, "out_sha256": sha256_file(table), "identity": gate["identity"],
                                            "parity_gate_sha256": sha256_file(parity), "parity_gate": str(parity)})
    monkeypatch.setattr(sys, "argv", ["signal_check", "--table", str(table), "--out", str(signal)])
    signal_check.main()
    assert json.loads(signal.read_text())["release"]["status"] == "PASS"
    fit_init_curves.cmd_knots(SimpleNamespace(table=str(table), out=str(knots), n_seg=8))
    monkeypatch.setattr(sys, "argv", ["replay_sim", "--table", str(table), "--knots", str(knots), "--out", str(replay), "--gate-theta", "0.5", "--delta", "0.3"])
    replay_sim.main()
    assert json.loads(replay.read_text())["release_gate"]["status"] == "PASS"
    args = SimpleNamespace(table=str(table), scales=str(scales), signal=str(signal), knots=str(knots), replay=str(replay), parity=str(parity), h_exec=5, out_dir=str(tmp_path / "init"))
    fit_init_curves.cmd_init(args)
    state = json.loads((tmp_path / "init/init_state.json").read_text())
    assert state["fixed_params"]["scales_sha256"] == sha256_file(scales)
    fit_init_curves.cmd_rprime(args)
    foreign = tmp_path / "foreign.npz"
    np.savez(foreign, meta_json=json.dumps({"library_sha256": "foreign", "schedule_id": SCHEDULE.schedule_id}))
    with pytest.raises(SystemExit, match="scales/library"):
        fit_init_curves.cmd_init(SimpleNamespace(**{**vars(args), "scales": str(foreign)}))
    changed = copy.deepcopy(gate)
    changed["identity"]["corpus_manifest_sha256"] = "foreign"
    write_json(parity, changed)
    with pytest.raises(SystemExit, match="not bound"):
        fit_init_curves.cmd_init(args)
    with pytest.raises(SystemExit, match="parity file"):
        fit_init_curves.cmd_rprime(args)


def test_reduced_parity_cannot_be_used_as_a_formal_pass():
    from exp.online_rit.provenance import validate_parity

    gate = parity_fixture()
    gate.update(n_rows=10, per_task_rows={str(i): 1 for i in range(10)}, sample=gate["sample"][:10])
    with pytest.raises(SystemExit, match="200-row"):
        validate_parity(gate, gate["identity"])


def test_materialized_pools_bind_loader_contents_and_index_map(tmp_path):
    from exp.ablation_study.cache_size.run_size_eval import load_apool_digest
    from exp.online_rit.cohorts import validate_pool
    from exp.online_rit.library_prep import cmd_pools

    source = tmp_path / "a"
    source.mkdir()
    order = {"suite": "libero_10", "assignment": {str(t): {"task_name": f"task{t}"} for t in range(10)}}
    for t in range(10):
        torch.save(torch.arange(50 * 3).reshape(50, 3).float() + t * 1000, source / f"task{t}.init")
    write_json(tmp_path / "order.json", order)
    out = tmp_path / "pools"
    cmd_pools(SimpleNamespace(task_order=str(tmp_path / "order.json"), apool_dir=str(source), out_dir=str(out), per_side=25, seed=20260914))
    manifest = json.loads((out / "init_pools_manifest.json").read_text())
    for name, trials in (("adapt", 25), ("terminal", 25), ("smoke", 1)):
        record = load_apool_digest(str(out / f"apool_{name}.yaml"), expect_per_task=trials)
        mapping = validate_pool(manifest, name, record, trials)
        states = torch.load(out / f"{name}_pool/task0.init", weights_only=False)
        parent = torch.load(source / "task0.init", weights_only=False)
        assert torch.equal(states[-1], parent[mapping[0][-1]])
    wrong = load_apool_digest(str(out / "apool_terminal.yaml"), expect_per_task=25)
    with pytest.raises(SystemExit, match="differs"):
        validate_pool(manifest, "adapt", wrong, 25)
    tampered = copy.deepcopy(manifest)
    tampered["adapt"]["0"][0] = 49
    with pytest.raises(SystemExit, match="mapping changed"):
        validate_pool(tampered, "adapt", load_apool_digest(str(out / "apool_adapt.yaml"), expect_per_task=25), 25)


def test_real_cost_reader_and_consumers_charge_feedback_and_snapshot_bounds(tmp_path):
    from exp.online_rit.aggregate_online import aggregate
    from exp.online_rit.ir_replay import Episode, GateParams, replay_ir

    doc = complete_cost_record()
    doc.update(stage1_ms=1., stage2_ms=2., stage3_ladder_ms={"1": 8., "2": 9., "4": 11., "8": 25.},
               captured_stage3_ms={"1": 8.5, "2": 9.5, "4": 11.5},
               executed_feedback_ms={"1": 1., "2": 1., "4": 1.},
               dispatch_ms={"online": 2., "threshold": 1.},
               commit_ms={f"{m}:{n}": (2. if m == "learning" else 1.) for m in ("frozen", "learning") for n in (0, 1, 3)},
               snapshot_ms={"learning": 4., "frozen": 3.})
    path = tmp_path / "cost.json"
    write_json(path, doc)
    ledger = load_ledger(path)
    assert ledger.warm_ms(.875) == 11. and ledger.miss_ms == 28.
    assert ledger.decision_ms("WARM_START", .875, 0, online=True) == pytest.approx(11 + .5 + 1 + 2 + 2 + 4/200)
    frozen = ledger.decision_ms("WARM_START", .875, 0, online=True, update_enabled=False)
    assert frozen == 11 + .5 + 1 + 2 + 1
    replay = replay_ir([Episode("e", (.9,))], [0., float("inf"), float("inf")], ledger, GateParams(.5), feedback_mode="fm0")
    uid = "a:eval:0:0"
    write_jsonl(tmp_path / "journal.jsonl", [{"yaml_id": "a", "task_uid": uid, "attempt": 0, "accepted": True, "run_id": "r", "status": "done"}])
    write_jsonl(tmp_path / "per_step.jsonl", [{"yaml_id": "a", "task_uid": uid, "attempt": 0, "accepted": True, "run_id": "r", "step_idx": 0, "hit_type": "WARM_START", "start_t": .875, "online_rit": {"feedback_mode": "fm0", "update_enabled": True, "fb_batch_size": 0}}])
    result = aggregate(tmp_path, ledger)["a"]
    assert result["ir_percent"] == pytest.approx(replay.ir_percent)
    assert result["ir_percent_no_fb"] == pytest.approx(replay.ir_percent_no_fb)
    del doc["commit_ms"]["learning:1"]
    write_json(path, doc)
    with pytest.raises(SystemExit, match="incomplete"):
        load_ledger(path)


def test_original_init_pairing_resamples_risk_counts_and_rejects_missing_identity():
    from exp.online_rit.aggregate_online import paired_bootstrap

    def record(original, success, k, n):
        return {"suite": "s", "parent_pool_sha256": "parent", "task_id": 0, "orig_init_state_idx": original,
                "success": success, "violation": {"7": {"k": k, "n": n}}}
    a = {"full:0": record(10, True, 1, 2), "full:1": record(20, False, 1, 10)}
    b = {"subset:0": record(20, True, 0, 10), "subset:1": record(10, True, 0, 2)}
    result = paired_bootstrap(a, b, n_boot=100)
    assert result["n_shared"] == 2 and result["diff"] == .5
    assert result["risk"]["all"]["diff"] == pytest.approx(-2/12)
    assert result["risk"]["7"]["n_boot_valid"] == 100
    assert paired_bootstrap(a, {"other": record(30, True, 0, 0)})["n_shared"] == 0
    with pytest.raises(SystemExit, match="lacks original"):
        paired_bootstrap({"old": True}, b)


def test_search_evaluates_extra_q_values_without_losing_uniform_budget():
    from exp.online_rit.ir_replay import GateParams, ReplayResult, search_delta

    q = .123456789
    seen = set()
    def evaluate(d):
        seen.add(d)
        value = 70. if d == q else 90.
        return ReplayResult(value, value, 1, {}, {}, str(value))
    result = search_delta([], lambda d: [], SimpleNamespace(source="synthetic"), GateParams(.5),
                          delta_lo=0., delta_hi=1., targets=(70., 60.), extra_deltas=(q,), evaluate=evaluate)
    assert result["targets"]["70"]["found"] and not result["targets"]["60"]["found"]
    assert result["uniform_grid_points"] == 4097 and result["n_evaluated"] == 4098
    assert set(np.linspace(0., 1., 4097)) <= seen


def test_smoke_emit_and_completed_terminal_export_produce_launchable_cohorts(tmp_path, monkeypatch):
    from exp.online_rit.emit_online_arms import emit
    from exp.online_rit.fit_init_curves import fit_rprime
    from exp.online_rit import pick_terminal_state
    from openpi.cache.components.online_rit import OnlineRiskCurves
    from openpi.cache.online_state import CurveRegistry
    from tests.exp.test_online_rit_exp import KNOTS, TEMPLATE

    for name in ("scales.npz", "library.pkl", "init.json"):
        (tmp_path / name).write_bytes(b"synthetic identity fixture")
    write_json(tmp_path / "knots.json", {"knots": KNOTS})
    write_json(tmp_path / "r_arm_record.json", {"gate_theta": .5})
    write_json(tmp_path / "rfit.json", fit_rprime(synthetic_table(n_traj=12), KNOTS))
    write_json(tmp_path / "ladder.json", {"provisional": False, "targets": {"70": {"found": True, "duplicate_of": None, "delta": .3}}})
    args = SimpleNamespace(suite="libero_10", template=str(TEMPLATE), r_arm_record=str(tmp_path / "r_arm_record.json"),
                           knots=str(tmp_path / "knots.json"), scales=str(tmp_path / "scales.npz"), init_state=str(tmp_path / "init.json"),
                           rprime_fit=str(tmp_path / "rfit.json"), online_ladder=str(tmp_path / "ladder.json"), rprime_ladder=str(tmp_path / "ladder.json"),
                           library_pkl=str(tmp_path / "library.pkl"), state_log_root=str(tmp_path / "states"), out_dir=str(tmp_path / "config"),
                           run_tag="smoke_test", smoke=True, s3b_pkl="", s3b_targets=set(), fm0_targets=set())
    record = emit(args)
    assert len(record["arms"]) == 2 and set(record["matrices"]) == {"smoke"}
    matrix = yaml.safe_load((tmp_path / "config/libero_10/arm_matrix_smoke_test_smoke.yaml").read_text())
    assert matrix["cohort"]["trials"] == 1 and matrix["cohort"]["pool"] == "smoke"
    source_arm, source_meta = next((name, rec) for name, rec in record["arms"].items() if rec["rule"] == "ocold")
    source = source_meta["yaml"]
    params = {"scales_sha256": sha256_file(args.scales), "h_exec": 5, "schedule_id": SCHEDULE.schedule_id}
    registry = CurveRegistry(state_log_root=str(tmp_path / "terminal_states"), server_instance_id="server")
    key = registry.attach(yaml_id=source_arm, library_sha256=sha256_file(args.library_pkl), fingerprint="fixture", factory=lambda: OnlineRiskCurves(knots=KNOTS, tier_indices=[7, 6, 4], alpha=.05, window=128, n_min=20, fixed_params=params))
    registry.flush(key, "task_end")
    mapping = {str(t): list(range(25)) for t in range(10)}
    manifest = {"suite": "libero_10", "parent_pool_sha256": "parent", "adapt": mapping,
                "terminal": {str(t): list(range(25, 50)) for t in range(10)},
                "pool_records": {"adapt": {"total_inits": 250, "rollup_sha256": "adapt", "per_task_digests": {}, "index_map_sha256": canonical_sha256(mapping)}}}
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    data = tmp_path / "run"
    journal, steps = [], []
    for t in range(10):
        for i in range(25):
            uid = f"{source_arm}:eval:{t}:{i}"
            common = {"yaml_id": source_arm, "task_uid": uid, "run_id": "run", "attempt": 0, "accepted": True}
            journal.append({**common, "status": "failed", "error": None})
            steps.append({**common, "suite": "libero_10", "parent_pool_sha256": "parent", "task_id": t, "orig_init_state_idx": i,
                          "step_idx": 0, "hit_type": "MISS", "online_rit": {"server_instance_id": "server", "learned": False, "fb": []}})
    write_jsonl(data / "journal.jsonl", journal)
    write_jsonl(data / "per_step.jsonl", steps)
    write_json(data / "per_step.jsonl.launch.json", {"yaml_sha256": {source_arm: sha256_file(source)}, "init_map_sha256": sha256_file(manifest_path), "init_map_key": "adapt"})
    out = tmp_path / "frozen.yaml"
    argv = ["pick_terminal", "--state-dir", str(registry.log_dir(key)), "--source-yaml", source, "--out-yaml", str(out),
            "--terminal-copy", str(tmp_path / "terminal.json"), "--data-dir", str(data), "--pool-manifest", str(manifest_path)]
    monkeypatch.setattr(sys, "argv", argv)
    pick_terminal_state.main()
    frozen = yaml.safe_load(out.read_text())["checkpoints"]["cp1"]["judge"]
    assert frozen["update_enabled"] is False
    assert yaml.safe_load(out.with_suffix(".matrix.yaml").read_text())["cohort"]["pool"] == "terminal"
    write_jsonl(data / "journal.jsonl", journal[:-1])
    with pytest.raises(SystemExit, match="episode set"):
        pick_terminal_state.main()
