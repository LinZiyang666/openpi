"""RIT-Pareto glue: cohort sampling, export, arm emission, runner gates, aggregation."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch
import yaml

from exp.dispatch_surface.collect_query_cohort import cmd_plan
from exp.dispatch_surface.emit_precheck_yamls import LAYER_PRIMARY, LAYER_SECONDARY
from exp.gate_threshold_pareto.emit_gtp_yamls import GATE_J, GATE_L, GATE_PROBE_INTERVAL
from exp.gate_threshold_pareto.run_gtp import validate_arms
from exp.rit_pareto import aggregate_rit, emit_arms, export_rit, shadow_cohort
from openpi.cache.components.surface_judge import load_surface_artifact
from openpi.cache.config import load_cache_config

SPATIAL_NAMES = {
    0: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    1: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    2: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    3: "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    4: "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    5: "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    6: "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    7: "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    8: "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    9: "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
}


# ------------------------------------------------------------------
# Shadow cohort sampling
# ------------------------------------------------------------------


def test_sample_assignment_is_seeded_and_disjoint():
    a = shadow_cohort.sample_assignment(SPATIAL_NAMES, seed=1, fit_per_task=5, cal_per_task=10)
    b = shadow_cohort.sample_assignment(SPATIAL_NAMES, seed=1, fit_per_task=5, cal_per_task=10)
    c = shadow_cohort.sample_assignment(SPATIAL_NAMES, seed=2, fit_per_task=5, cal_per_task=10)
    assert a == b
    assert a != c
    for tid, info in a.items():
        assert info["task_name"] == SPATIAL_NAMES[tid]
        picked = info["fit"] + info["cal"]
        assert len(picked) == 15 and len(set(picked)) == 15
        assert all(0 <= i < 50 for i in picked)
        assert info["fit"] == sorted(info["fit"]) and info["cal"] == sorted(info["cal"])
        assert not set(info["fit"]) & set(info["cal"])
    # A task's draw does not move when another task's quota is unchanged.
    assert a[3] == shadow_cohort.sample_assignment({3: SPATIAL_NAMES[3]}, seed=1, fit_per_task=5, cal_per_task=10)[3]


def test_sample_assignment_rejects_zero_quota_and_oversize():
    with pytest.raises(ValueError, match="positive"):
        shadow_cohort.sample_assignment(SPATIAL_NAMES, seed=1, fit_per_task=0, cal_per_task=10)
    with pytest.raises(ValueError, match="exceeds"):
        shadow_cohort.sample_assignment(SPATIAL_NAMES, seed=1, fit_per_task=30, cal_per_task=30)


def _fake_apool(root: pathlib.Path) -> pathlib.Path:
    apool = root / "apool"
    apool.mkdir()
    for tid, name in SPATIAL_NAMES.items():
        states = torch.arange(50 * 4, dtype=torch.float32).reshape(50, 4) + 1000 * tid
        torch.save(states, apool / f"{name}.init")
    return apool


def _task_order_manifest(root: pathlib.Path) -> pathlib.Path:
    path = root / "order.json"
    path.write_text(json.dumps({"assignment": {str(t): {"task_name": n} for t, n in SPATIAL_NAMES.items()}}))
    return path


def test_sample_materialises_pool_and_feeds_the_phase0_plan(tmp_path):
    apool = _fake_apool(tmp_path)
    order = _task_order_manifest(tmp_path)
    manifest = tmp_path / "shadow_manifest.json"
    pool = tmp_path / "shadow_pool"

    class Args:
        suite = "libero_spatial"
        apool_dir = str(apool)
        task_order_manifest = str(order)
        seed = 20260901
        fit_per_task = 5
        cal_per_task = 10
        pool_out = str(pool)
        out_manifest = str(manifest)

    shadow_cohort.cmd_sample(Args)
    m = json.loads(manifest.read_text())
    assert m["quota"] == {"fit": 5, "cal": 10} and m["seed"] == 20260901
    assert sorted(int(t) for t in m["assignment"]) == list(range(10))
    for tid, info in m["assignment"].items():
        name = info["task_name"]
        states = torch.load(pool / f"{name}.init", weights_only=False)
        picked = sorted(info["fit"] + info["cal"])
        assert states.shape[0] == 15
        full = torch.load(apool / f"{name}.init", weights_only=False)
        assert torch.equal(states, full[picked])
        assert m["pool_digests"]["shadow"][name]["indices"] == picked

    class PlanArgs:
        split_manifest = str(manifest)
        pool_dir = str(pool)
        out = str(tmp_path / "plan.json")
        fit_per_task = None
        cal_per_task = None

    cmd_plan(PlanArgs)
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert len(plan["episodes"]) == 150
    for e in plan["episodes"]:
        info = m["assignment"][str(e["task_id"])]
        picked = sorted(info["fit"] + info["cal"])
        assert picked[e["subset_init_state_idx"]] == e["orig_init_state_idx"]
        assert e["split"] == ("fit" if e["orig_init_state_idx"] in info["fit"] else "cal")

    # A second sample into the same pool dir is refused.
    with pytest.raises(SystemExit, match="must be empty"):
        shadow_cohort.cmd_sample(Args)


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------


def _synthetic_table(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.6, 1.0, size=n)
    base = 12.0 * (1.0 - s) + 1.0
    y10 = base + rng.gamma(2.0, 0.6, size=n)
    y7 = 0.7 * base + rng.gamma(2.0, 0.5, size=n)
    return s, y7, y10


FAKE_CONTRACT = {
    "key_builder_digest": "kb", "search_digest": "sd", "library_sha256": "0" * 64,
    "library_entry_count": 10, "action_dim": 32, "num_steps": 10, "h_exec": 5,
    "policy_fingerprint": "pf", "top_k": 1,
}


def _contract(tmp_path) -> dict:
    """Yaml-side digests from the real spatial template, identity fields faked."""
    from openpi.cache.config import compute_surface_retrieval_contract

    doc = export_rit.build_calibration_yaml("libero_spatial", "/srv/lib.pkl")
    path = tmp_path / "calib_for_contract.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    contract = compute_surface_retrieval_contract(load_cache_config(str(path)))
    contract.update({k: v for k, v in FAKE_CONTRACT.items() if k not in ("key_builder_digest", "search_digest")})
    return contract


def _export(tmp_path, targets=(20.0, 40.0, 60.0, 80.0, 95.0)):
    s, y7, y10 = _synthetic_table()
    contract = _contract(tmp_path)
    w = np.linspace(0.5, 1.5, 32).astype(np.float32)
    mask = np.ones(32, dtype=bool)
    identity = {"suite": "libero_spatial", "ref_mode": "tau1", "table_sha256": "t" * 64,
                "weights_sha256": "w" * 64, "cache_yaml_sha256": "c" * 64,
                "library_pkl": "/srv/lib.pkl", "library_sha256": "0" * 64,
                "n_episodes": 150, "n_tasks": 10}
    out = tmp_path / "export"
    rec = export_rit.fit_export(
        s=s, y7=y7, y10=y10, targets=list(targets), alpha=0.05, h_exec=5, w=w, active_mask=mask,
        contract=contract, identity=identity, out_dir=out,
    )
    return rec, out, s


def test_fit_export_writes_loadable_ir_addressed_artifacts(tmp_path):
    rec, out, s = _export(tmp_path)
    assert set(rec["artifacts"]) == {"ir20", "ir40", "ir60", "ir80", "ir95"}
    prev = -1.0
    for name, art in sorted(rec["artifacts"].items(), key=lambda kv: kv[1]["target_ir"]):
        a = load_surface_artifact(art["path"])
        assert not a.uses_disagreement and a.k == 1 and a.h_exec == 5
        assert float(a.s_min_full[0]) == art["theta_full"] and float(a.s_min_warm[0]) == art["theta_warm"]
        assert a.s_min_warm[0] <= a.s_min_full[0]
        assert a.retrieval_contract == rec["retrieval_contract"]
        assert a.retrieval_contract["library_sha256"] == FAKE_CONTRACT["library_sha256"]
        assert a.meta["target_ir"] == art["target_ir"] and a.meta["ref_mode"] == "tau1"
        assert abs(art["ir_gap"]) <= 0.5
        assert art["predicted_ir"] > prev
        prev = art["predicted_ir"]
        assert export_rit._sha(pathlib.Path(art["path"])) == art["output_sha256"]
    fit_rec = json.loads((out / export_rit.FIT_RECORD_NAME).read_text())
    assert fit_rec["n_rows"] == len(s) and fit_rec["gate_theta"] == rec["gate_theta"]
    assert rec["fit_record_sha256"] == export_rit._sha(out / export_rit.FIT_RECORD_NAME)
    assert rec["gate_theta"] == export_rit.solve_gate_theta(s)
    # top-15% of the shadow scores are admitted by the gate
    assert 0.80 <= float(np.mean(s >= rec["gate_theta"])) <= 0.90


def test_fit_export_refuses_non_empty_out_dir_and_bad_targets(tmp_path):
    rec, out, _ = _export(tmp_path)
    s, y7, y10 = _synthetic_table()
    with pytest.raises(SystemExit, match="must be empty"):
        export_rit.fit_export(s=s, y7=y7, y10=y10, targets=[50.0], alpha=0.05, h_exec=5,
                              w=np.ones(32), active_mask=np.ones(32, bool), contract=FAKE_CONTRACT,
                              identity={}, out_dir=out)
    with pytest.raises(SystemExit, match="strictly increasing"):
        export_rit.parse_targets("50,40")
    with pytest.raises(SystemExit, match="lie in"):
        export_rit.parse_targets("0,50")
    assert export_rit.parse_targets("20,82.5") == [20.0, 82.5]
    assert export_rit.target_name(82.5) == "ir82p5" and export_rit.target_name(20) == "ir20"


def test_calibration_yaml_uses_the_gtp_template_with_calibration_judge(tmp_path):
    doc = export_rit.build_calibration_yaml("libero_spatial", "/srv/lib.pkl")
    cp1 = doc["checkpoints"]["cp1"]
    assert cp1["judge"] == {"type": "always_hit"} and cp1["gate"] == {"type": "always_search"}
    assert doc["backend"]["in_memory"]["preload_path"] == "/srv/lib.pkl"
    assert doc["write_policy"] == {"type": "never"}
    assert doc["keys"]["vision_1"]["weight"] == 0.5  # spatial fusion weights inherited
    path = tmp_path / "calib.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    cfg = load_cache_config(str(path))
    assert cfg.checkpoints["cp1"].judge.type == "always_hit"


# ------------------------------------------------------------------
# Arm emission + runner gates
# ------------------------------------------------------------------


def test_emit_arms_two_layers_round_trip_and_runner_accepts_them(tmp_path):
    rec, out, _ = _export(tmp_path, targets=(30.0, 60.0))
    emitted = emit_arms.emit(out / export_rit.EXPORT_RECORD_NAME, suite="libero_spatial",
                             preload_path="/srv/lib.pkl", out_dir=tmp_path / "arms")
    assert set(emitted["layers"]) == {LAYER_PRIMARY, LAYER_SECONDARY}
    ng = yaml.safe_load(pathlib.Path(emitted["layers"][LAYER_PRIMARY]["matrix"]).read_text())
    hg = yaml.safe_load(pathlib.Path(emitted["layers"][LAYER_SECONDARY]["matrix"]).read_text())
    assert [r["arm"] for r in ng["arms"]] == ["rit_sp_ng_ir30", "rit_sp_ng_ir60"]
    assert [r["arm"] for r in hg["arms"]] == ["rit_sp_hg_ir30", "rit_sp_hg_ir60"]
    for row in ng["arms"]:
        cfg = load_cache_config(row["yaml"])
        cp1 = cfg.checkpoints["cp1"]
        assert cp1.judge.type == "dispatch_surface" and cp1.gate.type == "always_search"
        assert cfg.backend.in_memory.preload_path == "/srv/lib.pkl"
    for row in hg["arms"]:
        cp1 = load_cache_config(row["yaml"]).checkpoints["cp1"]
        assert cp1.gate.type == "score_hysteresis"
        assert cp1.gate.theta_low == cp1.gate.theta_high == rec["gate_theta"]
        assert (cp1.gate.j, cp1.gate.probe_interval, cp1.gate.L) == (GATE_J, GATE_PROBE_INTERVAL, GATE_L)
    # runner gates: the declared judge/gate pass, the GTP defaults reject them
    assert validate_arms(ng["arms"], phase="eval", judge_type="dispatch_surface", eval_gate="always_search")
    assert validate_arms(hg["arms"], phase="eval", judge_type="dispatch_surface")
    with pytest.raises(SystemExit, match="expected 'threshold'"):
        validate_arms(ng["arms"], phase="eval")
    with pytest.raises(SystemExit, match="expected 'score_hysteresis'"):
        validate_arms(ng["arms"], phase="eval", judge_type="dispatch_surface")
    with pytest.raises(SystemExit, match="expected 'always_search'"):
        validate_arms(hg["arms"], phase="eval", judge_type="dispatch_surface", eval_gate="always_search")
    with pytest.raises(SystemExit, match="judge_type must be"):
        validate_arms(hg["arms"], phase="eval", judge_type="bogus")
    # a tampered artifact is refused at emit time
    art = pathlib.Path(rec["artifacts"]["ir30"]["path"])
    art.write_bytes(art.read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="missing or changed"):
        emit_arms.emit(out / export_rit.EXPORT_RECORD_NAME, suite="libero_spatial",
                       preload_path="/srv/lib.pkl", out_dir=tmp_path / "arms2")


def test_emit_refuses_a_different_library_than_the_export(tmp_path):
    _, out, _ = _export(tmp_path, targets=(50.0,))
    with pytest.raises(SystemExit, match="would load"):
        emit_arms.emit(out / export_rit.EXPORT_RECORD_NAME, suite="libero_spatial",
                       preload_path="/srv/other.pkl", out_dir=tmp_path / "arms")


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------


def test_aggregate_prices_three_tiers_and_matches_accepted_attempts(tmp_path):
    from exp.dispatch_surface.analysis.analytic_cost import unit_cost

    journal = [
        {"yaml_id": "rit_sp_ng_ir40", "task_uid": "a", "attempt": 2, "status": "done", "accepted": True},
        {"yaml_id": "rit_sp_ng_ir40", "task_uid": "a", "attempt": 1, "status": "failed", "accepted": False},
        {"yaml_id": "rit_sp_ng_ir40", "task_uid": "b", "attempt": 1, "status": "failed", "accepted": True},
        {"yaml_id": "rit_sp_ng_ir40", "task_uid": "c", "attempt": 1, "status": "running", "accepted": True},
    ]
    steps = [
        {"yaml_id": "rit_sp_ng_ir40", "task_uid": "a", "attempt": 2, "hit_type": "FULL_HIT", "start_t": None},
        {"yaml_id": "rit_sp_ng_ir40", "task_uid": "a", "attempt": 2, "hit_type": "WARM_START", "start_t": 0.3},
        {"yaml_id": "rit_sp_ng_ir40", "task_uid": "a", "attempt": 1, "hit_type": "MISS", "start_t": None},
        {"yaml_id": "rit_sp_ng_ir40", "task_uid": "b", "attempt": 1, "hit_type": "MISS", "start_t": None},
        {"yaml_id": "rit_sp_ng_ir40", "task_uid": "b", "attempt": 1, "hit_type": None, "start_t": None},
    ]
    (tmp_path / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in journal))
    (tmp_path / "per_step.jsonl").write_text("".join(json.dumps(r) + "\n" for r in steps))
    res = aggregate_rit.aggregate(tmp_path)
    arm = res["rit_sp_ng_ir40"]
    assert arm["n_ep"] == 2 and arm["success_rate"] == 0.5 and arm["target_ir"] == 40.0
    assert {k: arm["counts"][k] for k in ("FULL_HIT", "WARM_START", "MISS")} == {"FULL_HIT": 1, "WARM_START": 1, "MISS": 1}
    assert arm["counts"]["WARM_START@0.3"] == 1 and arm["label"] == "IR=40"
    expected = (unit_cost("FULL_HIT", None) + unit_cost("WARM_START", 0.3) + unit_cost("MISS", None))
    assert arm["ir_percent"] == pytest.approx(100.0 * expected / (3 * unit_cost("MISS", None)))
    assert aggregate_rit.target_of("rit_l10_hg_ir82p5") == 82.5
    assert aggregate_rit.target_of("gtp_ws_sp_fh05") is None


def test_plot_writes_png_and_pdf(tmp_path):
    arms = {f"rit_sp_ng_ir{t}": {"ir_percent": float(t) + 1, "success_rate": 0.5 + t / 400, "target_ir": float(t)}
            for t in (20, 40, 60)}
    paths = aggregate_rit.plot_suite("libero_spatial", {"no gate": arms, "H gate": arms}, tmp_path / "fig",
                                     gst=[(30.0, 0.5, "gtp_ws_sp_fh10")])
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)


def test_gpu_slots_default_and_override():
    from exp.gate_threshold_pareto.run_gtp import gpu_slots

    assert gpu_slots(8) == [str(i) for i in range(8)]
    assert gpu_slots(8, "0,1,2,3,4,6") == ["0", "1", "2", "3", "4", "6"]
    with pytest.raises(SystemExit, match="comma list"):
        gpu_slots(8, "0,x")
    with pytest.raises(SystemExit, match="positive"):
        gpu_slots(0)


def test_pareto_front_keeps_only_non_dominated_arms():
    pts = [(40.0, 0.80, 20.0, "a"), (45.0, 0.78, 25.0, "b"), (50.0, 0.90, 30.0, "c"),
           (50.0, 0.85, 35.0, "c2"), (60.0, 0.88, 40.0, "d"), (70.0, 0.95, 45.0, "e")]
    front = aggregate_rit.pareto_front(pts)
    assert [p[3] for p in front] == ["a", "c", "e"]
    # ties on IR: the higher success wins, the other is dominated
    assert all(p[3] != "c2" for p in front)
