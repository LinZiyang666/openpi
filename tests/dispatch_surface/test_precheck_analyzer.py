"""Tests for analyze_precheck: frontier branches, gates, discipline refusals."""

from __future__ import annotations

import numpy as np
import pytest

import exp.dispatch_surface.power_sim_cost_blocks as power_mod
from exp.dispatch_surface.analysis.analyze_precheck import (
    COMPUTE_GATE,
    compute_unit_cost,
    frontier_record,
    gate1,
    gate2,
    load_sr_outcomes,
)

T_ARMS = [("t1", 0.90, 100.0, 200.0), ("t2", 0.94, 120.0, 220.0), ("t3", 0.97, 140.0, 240.0)]


@pytest.fixture(autouse=True)
def _small_but_frozen_power_replay(monkeypatch):
    monkeypatch.setattr(power_mod, "POWER_SEED", 1)
    monkeypatch.setattr(power_mod, "POWER_N_SIM", 20)
    monkeypatch.setattr(power_mod, "POWER_N_BOOT", 50)


# ------------------------------------------------------------------
# Frontier branches
# ------------------------------------------------------------------


def test_bracket_branch_interpolates():
    rec = frontier_record(T_ARMS, (0.92, 80.0, 150.0))
    assert rec["branch"] == "bracket" and rec["D_sr"] == 0.0
    # SR 0.92 sits halfway between t1 (0.90) and t2 (0.94): comparator
    # compute = 110, latency = 210.
    assert rec["D_c"] == pytest.approx((80 - 110) / 110)
    assert rec["D_l"] == pytest.approx((150 - 210) / 210)


def test_high_branch_clamps_to_argmax_sr_cell():
    rec = frontier_record(T_ARMS, (0.99, 100.0, 200.0))
    assert rec["branch"] == "high"
    assert rec["D_sr"] == pytest.approx(0.99 - 0.97)
    assert rec["D_c"] == pytest.approx((100 - 140) / 140)


def test_low_branch_clamps_and_stays_in_cost_distribution():
    rec = frontier_record(T_ARMS, (0.80, 50.0, 100.0))
    assert rec["branch"] == "low"
    assert rec["D_sr"] == pytest.approx(0.80 - 0.90)
    # Cost differences are still produced — no deletion path exists.
    assert np.isfinite(rec["D_c"]) and np.isfinite(rec["D_l"])


def test_interpolation_does_not_zero_out_a_known_compute_gap():
    # SV compute is a KNOWN 40% below the frontier at matched SR; the record
    # must reflect that gap, never construct it away (Round 3 B2 regression).
    recs = [frontier_record(T_ARMS, (0.94, 120.0 * 0.6, 220.0 * 0.6)) for _ in range(100)]
    d_c = np.array([r["D_c"] for r in recs])
    assert np.quantile(d_c, 0.95) == pytest.approx(-0.4)


def test_nonpositive_comparator_cost_refused():
    with pytest.raises(SystemExit):
        frontier_record([("t1", 0.9, 0.0, 1.0), ("t2", 0.95, 1.0, 1.0),
                         ("t3", 0.97, 1.0, 1.0)], (0.90, 1.0, 1.0))


# ------------------------------------------------------------------
# Gates
# ------------------------------------------------------------------


def _records(d_sr=0.0, d_c=-0.10, d_l=-0.02, n=1000, branch="bracket"):
    return [{"branch": branch, "D_sr": d_sr, "D_c": d_c, "D_l": d_l} for _ in range(n)]


def test_gate1_passes_on_clear_win():
    assert gate1(_records())["pass"] is True


def test_gate1_fails_on_compute_gate():
    assert gate1(_records(d_c=COMPUTE_GATE + 0.01))["pass"] is False


def test_gate1_latency_regression_fails():
    assert gate1(_records(d_l=0.01))["pass"] is False


def test_gate1_low_replicates_fail_sr_floor():
    recs = _records(n=900) + _records(d_sr=-0.05, n=100, branch="low")
    out = gate1(recs)
    assert out["branch_shares"]["low"] == pytest.approx(0.1)
    assert out["pass"] is False  # 5% lower quantile of D_sr < 0


def test_gate2_requires_sr_gain_and_latency_nonregression():
    n = 1000
    win = gate2(np.full(n, 0.02), np.full(n, 0.01), np.full(n, -0.01))
    assert win["pass"] is True
    no_sr = gate2(np.full(n, 0.0), np.full(n, 0.01), np.full(n, -0.01))
    assert no_sr["pass"] is False
    lat_reg = gate2(np.full(n, 0.02), np.full(n, 0.01), np.full(n, 0.02))
    assert lat_reg["pass"] is False  # v may not buy SR with latency


def test_no_gate3_confirmatory_path_exists():
    import exp.dispatch_surface.analysis.analyze_precheck as mod
    import inspect

    src = inspect.getsource(mod)
    assert "gate3" not in src.lower().replace("no gate 3", "")


# ------------------------------------------------------------------
# Discipline refusals
# ------------------------------------------------------------------


def test_unpaired_journal_refused(tmp_path):
    rows = [
        {"yaml_id": "a", "task_uid": "a:eval:0:1", "status": "done",
         "success": True, "accepted": True},
        {"yaml_id": "b", "task_uid": "b:eval:0:2", "status": "done",
         "success": True, "accepted": True},
    ]
    p = tmp_path / "journal.jsonl"
    import json

    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit):
        load_sr_outcomes(str(p), ["a", "b"])


def test_paired_journal_loads(tmp_path):
    import json

    rows = []
    for arm in ("a", "b"):
        for init in (1, 2):
            rows.append({"yaml_id": arm, "task_uid": f"{arm}:eval:0:{init}",
                         "status": "done", "success": init == 1, "accepted": True})
    p = tmp_path / "journal.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = load_sr_outcomes(str(p), ["a", "b"])
    assert out["a"][(0, 1)] == 1 and out["a"][(0, 2)] == 0


def test_duplicate_accepted_rows_refused(tmp_path):
    import json

    rows = [
        {"yaml_id": "a", "task_uid": "a:eval:0:1", "status": "done",
         "success": True, "accepted": True},
        {"yaml_id": "a", "task_uid": "a:eval:0:1", "status": "failed",
         "success": False, "accepted": True},
    ]
    p = tmp_path / "journal.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit):
        load_sr_outcomes(str(p), ["a"])


def test_equally_incomplete_grid_refused_against_expected(tmp_path):
    """Two arms missing the SAME cell used to pass the old mutual check."""
    import json

    rows = []
    for arm in ("a", "b"):
        rows.append({"yaml_id": arm, "task_uid": f"{arm}:eval:0:0",
                     "status": "done", "success": True, "accepted": True})
    p = tmp_path / "journal.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit):
        load_sr_outcomes(str(p), ["a", "b"], expected_grid={(0, 0), (0, 1)})


# ------------------------------------------------------------------
# Cost row parsing (G2-B3 golden rows)
# ------------------------------------------------------------------


def _row(name, elapsed=None, **extra):
    r = {"timestamp": 0.0, "task_id": 1, "name": name, **extra}
    if elapsed is not None:
        r["elapsed_ms"] = elapsed
    return r


def test_compute_unit_cost_mixed_verdicts_golden():
    # 3 decisions: FULL_HIT (stage1 only), WARM_START (1+2+warm), MISS (1+2+flow).
    rows = [
        _row("stage1_vision", 10.0), _row("total_inference", 11.0),
        _row("stage1_vision", 10.0), _row("stage2_llm", 27.0),
        _row("stage3_warm", 9.0), _row("total_inference", 47.0),
        _row("stage1_vision", 10.0), _row("stage2_llm", 27.0),
        _row("stage3_flow", 29.0), _row("total_inference", 67.0),
    ]
    # Stage sum = 10 + (10+27+9) + (10+27+29) = 122 over 3 decisions.
    assert compute_unit_cost(rows, "t") == pytest.approx(122.0 / 3)


def test_compute_unit_cost_rejects_legacy_field():
    """Regression for G2-B3: a dur_ms row must be refused, never read as 0."""
    rows = [_row("total_inference", 11.0),
            {"timestamp": 0.0, "task_id": 1, "name": "stage1_vision", "dur_ms": 10.0}]
    with pytest.raises(SystemExit):
        compute_unit_cost(rows, "t")


def test_compute_unit_cost_rejects_missing_decisions_or_stages():
    with pytest.raises(SystemExit):
        compute_unit_cost([_row("stage1_vision", 10.0)], "t")  # no decisions
    with pytest.raises(SystemExit):
        compute_unit_cost([_row("total_inference", 11.0)], "t")  # no stages
    with pytest.raises(SystemExit):
        compute_unit_cost([_row("total_inference", 11.0),
                           _row("stage1_vision", float("nan"))], "t")


# ------------------------------------------------------------------
# check_discipline provenance fixture (G2-B2 / B6)
# ------------------------------------------------------------------


def _discipline_fixture(tmp_path, *, s0_delta=0.5, pool_digest="PD",
                        cost_fp="fp", power_blocks=None, s0_inputs=None,
                        launch_trials=30, cuda=True, aprime_cost=None,
                        seed_b=11, tamper_arm_yaml=False):
    import hashlib
    import json

    import yaml as _yaml

    from exp.dispatch_surface.power_sim_cost_blocks import record_digest, simulate
    from tests.cache.components.test_surface_judge import make_artifact

    sv_inputs = {"table": "T", "cohort_manifest": "C", "weights_npz": "W",
                 "rebuild_record": "R", "split_manifest": "S", "cache_yaml": "Y"}
    (tmp_path / "sv_dir").mkdir(exist_ok=True)
    (tmp_path / "s0_dir").mkdir(exist_ok=True)
    sv_path = make_artifact(tmp_path / "sv_dir", delta=0.5,
                            meta={"input_digests": sv_inputs})
    s0_path = make_artifact(
        tmp_path / "s0_dir", uses_disagreement=False, delta=s0_delta,
        meta={"input_digests": s0_inputs if s0_inputs is not None else sv_inputs},
    )
    arms, yaml_shas = {}, {}
    for arm, art in (("dsp_sv", sv_path), ("dsp_s0", s0_path)):
        y = tmp_path / f"{arm}.yaml"
        y.write_text(_yaml.safe_dump(
            {"checkpoints": {"cp1": {"judge": {"surface_artifact_path": str(art)}}}}))
        arms[arm] = str(y)
        yaml_shas[arm] = hashlib.sha256(y.read_bytes()).hexdigest()
    matrix = {"arms": arms, "arm_yaml_sha256": yaml_shas,
              "core_arms": ["dsp_sv", "dsp_s0"]}
    matrix_path = tmp_path / "arm_matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True))
    if tamper_arm_yaml:
        # Drift an arm yaml AFTER the matrix froze its digest.
        pathlib_y = tmp_path / "dsp_sv.yaml"
        pathlib_y.write_text(pathlib_y.read_text() + "# drifted\n")
    matrix_sha = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    (tmp_path / "fit_record.json").write_text(json.dumps({"delta_star": 0.5}))
    from openpi.cache.components.surface_judge import load_surface_artifact

    contract = load_surface_artifact(sv_path).retrieval_contract
    aprime_launch = "APRIME"
    (tmp_path / "launch.json").write_text(json.dumps({
        "library_sha256": contract["library_sha256"],
        "contract_binding": {"policy_fingerprint": contract["policy_fingerprint"]},
        "core_arms": ["dsp_sv", "dsp_s0"],
        "trials_per_task": launch_trials,
        "aprime_content_sha256": aprime_launch,
    }))
    # A REAL, replay-validated power record (tiny sim + variance source file).
    variance_src = tmp_path / "variance.json"
    variance_src.write_text(json.dumps({
        "schema_version": 1, "sigma_compute": 0.01, "sigma_latency": 0.01,
    }, sort_keys=True))
    power = simulate(sigma_compute=0.01, sigma_latency=0.01, seed=1, n_sim=20, n_boot=50)
    power["variance_source"] = str(variance_src)
    power["variance_source_sha256"] = hashlib.sha256(variance_src.read_bytes()).hexdigest()
    if power_blocks is not None:
        power["chosen_r"] = power_blocks
    power["record_digest"] = record_digest(power)
    (tmp_path / "power.json").write_text(json.dumps(power))
    cost_dir = tmp_path / "cost"
    cost_dir.mkdir(exist_ok=True)
    for name, level in (("compute", "SNAPSHOT"), ("latency", "OFF")):
        (cost_dir / f"manifest_{name}.json").write_text(json.dumps({
            "arm_orders": {"0": ["dsp_sv"]}, "blocks": 5,
            "block_pool_digest": pool_digest,
            "aprime_content_sha256": aprime_cost if aprime_cost is not None else aprime_launch,
            "arm_matrix_sha256": matrix_sha,
            "monitor_level": level,
            "cuda_available": cuda,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "stage_devices": {"stage1": "cuda:0", "stage2": "cuda:0", "stage3": "cuda:0"},
            "stage_probe_backends": {
                "stage1": "cuda" if name == "compute" else "cpu",
                "stage2": "cuda" if name == "compute" else "cpu",
                "stage3": "cuda" if name == "compute" else "cpu",
            },
            "seed": seed_b,
            "policy_fingerprint": contract["policy_fingerprint"] if cost_fp == "fp" else "OTHER",
            "library_sha256": contract["library_sha256"],
            "power_record_digest": record_digest(power),
        }))

    args = type("Args", (), {})()
    args.fit_record = str(tmp_path / "fit_record.json")
    args.launch_manifest = str(tmp_path / "launch.json")
    args.cost_dir = str(cost_dir)
    args.power_record = str(tmp_path / "power.json")
    args.arm_matrix = str(matrix_path)
    args.blocks = 5
    args.trials = 30
    return args, matrix


def test_check_discipline_accepts_consistent_fixture(tmp_path):
    from exp.dispatch_surface.analysis.analyze_precheck import check_discipline

    args, matrix = _discipline_fixture(tmp_path)
    out = check_discipline(args, matrix)
    assert out["delta_star"] == 0.5 and out["blocks"] == 5


def test_check_discipline_rejects_incomplete_compute_probe_map(tmp_path):
    import json
    import pathlib

    from exp.dispatch_surface.analysis.analyze_precheck import check_discipline

    args, matrix = _discipline_fixture(tmp_path)
    path = pathlib.Path(args.cost_dir) / "manifest_compute.json"
    manifest = json.loads(path.read_text())
    manifest["stage_probe_backends"] = {"stage1": "cuda"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        check_discipline(args, matrix)


@pytest.mark.parametrize("break_it", [
    "s0_delta", "pool_digest_none", "cost_fp", "power_blocks",
    "s0_inputs", "launch_trials", "cuda", "aprime_mismatch",
    "arm_yaml_drift", "gpu_missing",
])
def test_check_discipline_refusals(tmp_path, break_it):
    from exp.dispatch_surface.analysis.analyze_precheck import check_discipline

    kwargs = {}
    if break_it == "s0_delta":
        kwargs["s0_delta"] = 0.4      # S0 not sharing the frozen delta (G2-B2)
    elif break_it == "pool_digest_none":
        kwargs["pool_digest"] = None  # passes lack a shared block-pool digest
    elif break_it == "cost_fp":
        kwargs["cost_fp"] = "other"   # cost pass ran a different policy
    elif break_it == "power_blocks":
        kwargs["power_blocks"] = 7    # chosen_r not derivable -> validate refuses
    elif break_it == "s0_inputs":
        kwargs["s0_inputs"] = {"table": "OTHER"}  # same delta, DIFFERENT data (G2R2-B6)
    elif break_it == "launch_trials":
        kwargs["launch_trials"] = 25  # frozen 30/task quota broken
    elif break_it == "cuda":
        kwargs["cuda"] = False        # CPU-fallback probes must never adjudicate
    elif break_it == "aprime_mismatch":
        kwargs["aprime_cost"] = "DIFFERENT"  # cost A' != primary launch pool (G2R3-B3)
    elif break_it == "arm_yaml_drift":
        kwargs["tamper_arm_yaml"] = True     # self-reported sha != actual file
    else:
        kwargs["seed_b"] = None       # exercised via gpu/seed field below
        args, matrix = _discipline_fixture(tmp_path)
        # gpu_name missing: rewrite both manifests without it.
        import json as _json
        import pathlib as _pl

        for name in ("compute", "latency"):
            p = _pl.Path(args.cost_dir) / f"manifest_{name}.json"
            man = _json.loads(p.read_text())
            man["gpu_name"] = None
            p.write_text(_json.dumps(man))
        with pytest.raises(SystemExit):
            check_discipline(args, matrix)
        return
    args, matrix = _discipline_fixture(tmp_path, **kwargs)
    with pytest.raises(SystemExit):
        check_discipline(args, matrix)
