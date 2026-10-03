"""Reader-contract fakes exercise scientific denominators, masks and support."""
import copy
import json
from pathlib import Path
import subprocess
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from exp.offline_search.debug.tools.decision import common as C
from exp.offline_search.debug.tools.decision import (provenance, divergence, follow_vs_look,
    camera_shadow, stage_ledger, call_value, exposure_hazard, churn, trigger_vs_onset)
from exp.offline_search.debug.tools.decision.kernels import kernel, overlap


def options(**kwargs):
    values = dict(bootstraps=100, seed=3, onsets=None, triggers=None)
    values.update(kwargs)
    return SimpleNamespace(**values)


class FakeArm:
    """No serving imports; supplied actions/propensities are fully inspectable."""
    def __init__(self, n=6):
        self.arm_name = "fake"
        self.manifest = {"model": "pi05"}
        self.server_meta = {"valid_action_dims": list(range(7)), "gripper_dim": 6,
                            "gripper_threshold": .5, "model": "pi05", "action_scale": [1.] * 32}
        self.ds, self.eps = [], []
        for ep in range(n):
            self.eps.append(dict(episode_key="e%d" % ep, task_id=ep % 2, init=ep // 2,
                accepted=True, success=ep % 2 == 0, n_controls=22, env_seed=7,
                reset_state_sha256="reset%d" % ep, orig_init_state_idx=ep // 2,
                termination_reason="success" if ep % 2 == 0 else "max_steps", error=None))
            for seq in range(4):
                self.ds.append(dict(decision_id="e%d:1:%d" % (ep, seq), episode_key="e%d" % ep,
                    decision_seq=seq, task_id=ep % 2, init=ep // 2, stage_pre="grasp" if seq < 2 else "carry",
                    vision=seq % 2 == 0, src="cache" if seq % 2 == 0 else "cache_tail",
                    camera_mode="full" if seq % 2 == 0 else "blind", chunk_offset=0 if seq % 2 == 0 else 5,
                    blind_age_controls=5 * (seq % 2), n_applied=5, control_idx_start=2 + 5 * seq,
                    stage1_calls=int(seq % 2 == 0), policy_calls=0, camera_completions=0,
                    owner_cost=.152 * (seq % 2 == 0), lib="current", rows=[0, 1, 2], weights=[.25, .25, .5],
                    p_nominal=.5, p_effective=.5, treatment=ep % 2 == 0, eligible=True,
                    coin=.2 if ep % 2 == 0 else .8, diag={}))
        self.ds = pd.DataFrame(self.ds)
        self.eps = pd.DataFrame(self.eps)
        ids = self.ds.decision_id.to_numpy()
        base = np.zeros((len(ids), 10, 32))
        self.blocks = dict(served_chunk=base.copy(), cache_chunk=base.copy())
        shadow = np.ones_like(base)
        shadow[:, :, 7:] = 10000.  # Noise-like padding must never affect errors.
        self.augments = {
            "policy_shadow": dict(decision_id=ids, chunk=shadow),
            "shadow_look": dict(decision_id=ids, cache_chunk=base + .1,
                                rows=np.tile([0, 1, 2], (len(ids), 1)),
                                weights=np.tile([.25, .25, .5], (len(ids), 1))),
            "camera_shadow": dict(decision_id=ids, wrist_cache_chunk=base + .2, third_cache_chunk=base + .3,
                wrist_rows=np.tile([0, 1, 2], (len(ids), 1)), wrist_weights=np.tile([.25, .25, .5], (len(ids), 1)))
        }
        self.ctrl = dict(control_idx=np.arange(22), is_settle=np.arange(22) < 2,
            eef_pos=np.column_stack([np.arange(22) / 10., np.zeros((22, 2))]),
            predicates=(np.arange(22) >= 15).astype(float)[:, None],
            decision_seq=np.r_[-1, -1, np.repeat(np.arange(4), 5)])

    def decisions(self, columns=None, accepted_only=True):
        return self.ds.copy() if columns is None else self.ds[columns].copy()

    def episodes(self, accepted_only=True):
        return self.eps.copy()

    def decision_arrays(self, keys, decision_ids):
        index = {did: i for i, did in enumerate(self.ds.decision_id)}
        return {key: self.blocks[key][[index[did] for did in decision_ids]] for key in keys}

    def aug(self, kind, decision_ids=None):
        if kind not in self.augments:
            raise KeyError(kind)
        return self.augments[kind]

    def catalog(self):
        return pd.DataFrame([
            dict(row=0, task_id=0, episode=0, mode=0, stage_run=0, event_near=False, next=1, progress=.2),
            dict(row=1, task_id=0, episode=0, mode=0, stage_run=0, event_near=False, next=2, progress=.3),
            dict(row=2, task_id=0, episode=1, mode=1, stage_run=1, event_near=True, next=-1, progress=.7)])

    def controls(self, episode_key, keys=None):
        return copy.deepcopy(self.ctrl)


def test_provenance_merges_demo_mass_and_unknowns():
    arm = FakeArm()
    report = provenance.analyze(arm)
    row = report["tables"]["decisions"][0]
    assert row["effective_demos"] == 2
    assert row["row_effective_count"] == pytest.approx(8 / 3)
    assert row["unsupported_next_mass"] == .5
    assert row["cross_stage_kernel"] and row["cross_mode_kernel"]
    arm.ds.at[0, "rows"] = [0, 1, 99]
    row = provenance.analyze(arm)["tables"]["decisions"][0]
    assert row["unknown_mass"] == .5
    assert row["known_demo_mass"] == .5
    assert row["progress_known_mass"] == .5
    assert row["progress"] is None


def test_zero_weight_members_remain_structural():
    arm = FakeArm()
    arm.ds.at[0, "weights"] = [.5, .5, 0.]
    row = provenance.analyze(arm)["tables"]["decisions"][0]
    assert row["unsupported_next_members"] == 1
    assert row["unsupported_next_mass"] == 0
    assert row["cross_stage_kernel"] is False
    assert row["structural_cross_stage_kernel"] is True
    assert overlap([0, 0, 1], [.2, .3, .5], [0, 1], [.5, .5]) == 1


def test_divergence_masks_padding_and_marks_execution():
    arm = FakeArm()
    report = divergence.analyze(arm)
    rows = report["tables"]["offsets"]
    assert rows[0]["rms"] == 1
    assert rows[0]["motion_rms"] == 1
    assert rows[0]["gripper_flip"] is True
    assert rows[4]["executed_offset"] is True
    assert rows[5]["executed_offset"] is False
    assert report["noise_floor_coverage"]["available"] == 0
    assert rows[0]["noise_adjusted_mse"] is None


def test_policy_noise_half_and_negative_adjustment_preserved():
    arm = FakeArm()
    did = arm.ds.decision_id.iloc[0]
    base = arm.augments["policy_shadow"]["chunk"][0]
    arm.blocks["served_chunk"][0] = base
    arm.augments["policy_draws"] = dict(decision_id=np.array([did]), chunks=np.stack([base - 2, base, base + 2])[None])
    report = divergence.analyze(arm)
    noise = report["tables"]["noise_floor"][0]
    row = report["tables"]["offsets"][0]
    assert noise["policy_pair_mse"] == pytest.approx(16 / 3)
    assert row["noise_adjusted_mse"] == pytest.approx(-8 / 3)
    assert report["noise_floor_coverage"]["available"] == 1


def test_manifest_required_and_gripper_threshold_not_inferred():
    arm = FakeArm()
    del arm.server_meta["valid_action_dims"]
    with pytest.raises(C.Unavailable):
        divergence.analyze(arm)
    arm.server_meta["valid_action_dims"] = list(range(7))
    del arm.server_meta["gripper_threshold"]
    assert divergence.analyze(arm)["tables"]["offsets"][0]["gripper_flip"] is None


def test_partial_follow_head_and_kernel_row_map():
    arm = FakeArm()
    arm.ds.loc[1, "n_applied"] = 2
    report = follow_vs_look.analyze(arm)
    row = report["tables"]["decisions"][0]
    assert row["offset_denominator"] == 2
    assert row["motion_rms"] == pytest.approx(.1)
    assert row["kernel_overlap"] == 1
    assert report["coverage"]["denominator"] == 12
    assert len(report["tables"]["follow_gap_map"]) == 3


def test_camera_shadow_missingness_and_retrieval():
    arm = FakeArm()
    arm.augments["camera_shadow"]["third_cache_chunk"][0] = np.nan
    report = camera_shadow.analyze(arm)
    assert report["coverage"]["available"] == 47
    assert report["tables"]["decisions"][0]["kernel_overlap"] == 1


def test_augmentation_alignment_uses_identity():
    arm = FakeArm()
    data = arm.augments["policy_shadow"]
    data["chunk"][0] = 3
    data["decision_id"] = data["decision_id"][::-1]
    data["chunk"] = data["chunk"][::-1]
    aligned = C.augmentation(arm, "policy_shadow", arm.ds.decision_id)
    assert (aligned["chunk"][0] == 3).all()


def test_ledger_additive_and_partial_camera_prices():
    arm = FakeArm()
    arm.ds.loc[0, ["camera_mode", "camera_completions", "policy_calls"]] = ["wrist_only", 1, 1]
    report = stage_ledger.analyze(arm)
    records = report["tables"]["stages"]
    total = records[0]
    assert total["measured_work"] == pytest.approx(11 * .152 + .0646 + .0502 + .848)
    assert sum(row["measured_work"] for row in records[1:]) == pytest.approx(total["measured_work"])
    assert total["measured_IR_controls"] == pytest.approx(5 * total["measured_work"] / 120)
    assert len(report["tables"]["visits"]) == 12
    arm.ds.loc[0, "stage1_calls"] = np.nan
    assert stage_ledger.analyze(arm)["tables"]["stages"][0]["measured_work"] is None


def test_call_population_score_not_independent_anchor_sr():
    arm = FakeArm()
    report = call_value.analyze(arm, options())
    first = next(row for row in report["tables"]["contrasts"] if row.get("stage") == "grasp" and row.get("endpoint") == "final_success" and row.get("estimand") == "first_entry")
    assert first["population"]["estimate"] == 1
    assert first["original_episode_denominator"] == 6
    assert first["treated_ESS"] == 3
    assert first["control_ESS"] == 3
    assert first["population"]["task_init_clusters"] == 6
    assert report["support_reasons"]["not_fresh_anchor"] == 12


def test_call_forced_entries_and_unknown_stages_withheld():
    arm = FakeArm()
    arm.ds.loc[0, "p_effective"] = 1.
    arm.ds.loc[2, "p_effective"] = 0.
    report = call_value.analyze(arm, options())
    assert report["support_reasons"]["forced_no_interior_support"] == 2
    arm.ds["p_effective"] = np.nan
    assert call_value.analyze(arm, options())["status"] == "unavailable"
    arm = FakeArm()
    arm.ds["stage"] = arm.ds.stage_pre
    arm.ds.drop(columns="stage_pre", inplace=True)
    report = call_value.analyze(arm, options())
    assert {row["stage"] for row in report["tables"]["contrasts"]} == {"unknown"}


def test_first_entry_unreached_zero_and_derivative_sum():
    arm = FakeArm()
    arm.ds.loc[arm.ds.episode_key == "e4", "eligible"] = False
    report = call_value.analyze(arm, options())
    row = next(row for row in report["tables"]["contrasts"] if row.get("stage") == "grasp" and row.get("endpoint") == "final_success" and row.get("estimand") == "first_entry")
    assert row["population"]["estimate"] == pytest.approx(2 / 3)
    assert row["conditional_reached"]["estimate"] == pytest.approx(.8)
    assert row["unreached_share"] == pytest.approx(1 / 6)
    arm = FakeArm()
    arm.ds["stage_pre"] = "all"
    report = call_value.analyze(arm, options())
    row = next(row for row in report["tables"]["contrasts"] if row.get("endpoint") == "final_success" and row.get("estimand") == "probability_shift_derivative")
    assert row["population"]["estimate"] == 2


def test_coin_mismatch_and_missing_endpoint_are_explicit():
    arm = FakeArm()
    arm.ds.loc[0, "coin"] = .9
    report = call_value.analyze(arm, options())
    assert report["support_reasons"]["coin_treatment_mismatch"] == 1
    arm.eps["success"] = arm.eps.success.astype(object)
    arm.eps.loc[0, "success"] = None
    report = call_value.analyze(arm, options())
    assert report["status"] == "unavailable"
    assert "accepted outcome unavailable" in report["reason"]


def test_cluster_bootstrap_preserves_repeats_and_fixed_tasks():
    frame = pd.DataFrame([dict(task_id=0, init=0, numerator=1., denominator=1.),
                          dict(task_id=0, init=1, numerator=0., denominator=1.)])
    result = C.cluster_interval(frame, bootstraps=100, seed=10)
    repeated = C.cluster_interval(pd.concat([frame] * 10), bootstraps=100, seed=10)
    assert result["estimate"] == repeated["estimate"]
    assert result["interval"] == repeated["interval"]
    assert repeated["task_init_clusters"] == 2


def test_lottery_uses_anchor_assignment_not_survivors():
    arm = FakeArm()
    for i, row in arm.ds.iterrows():
        ep = int(row.episode_key[1:])
        e = ep % 3
        arm.ds.at[i, "diag"] = dict(support=[0, 1, 2], propensities=[1 / 3] * 3,
                                   drawn_e=e, coin=(e + .5) / 3)
        arm.ds.at[i, "coin"] = (e + .5) / 3
    report = exposure_hazard.analyze(arm, options())
    assert report["coverage"]["available"] == 12
    final = [row for row in report["tables"]["contrasts"] if row.get("endpoint") == "final_success"]
    assert {row["extension_block"] for row in final} == {1, 2}
    assert all(row["original_episode_denominator"] == 6 for row in final)
    arm.ds.at[0, "diag"] = dict(support=[0], propensities=[1., 0., 0.], drawn_e=0, coin=.2)
    assert exposure_hazard.analyze(arm, options())["coverage"]["unavailable"] == 1


def test_nested_adapter_diagnostics_are_read_only():
    row = {"diag": {"layer_0": {"_last_log": {"os_c_p": .25, "os_c_coin": .2}}}}
    original = copy.deepcopy(row)
    assert C.field(row, "os_c_p") == .25
    assert row == original


def test_churn_identity_and_net_symmetric_decomposition():
    reference, candidate = FakeArm(), FakeArm()
    candidate.eps.loc[0, "success"] = False
    candidate.eps.loc[1, "success"] = True
    candidate.eps.loc[2, "success"] = False
    rows, result = churn.compare(reference, candidate, options())
    assert result["loss"] == 2 and result["gain"] == 1
    assert result["churn"] == pytest.approx(result["symmetric_churn"] + abs(result["net_loss_bias"]))
    assert len(rows) == 6
    candidate.eps.loc[0, "env_seed"] = 8
    rows, summary = churn.compare(reference, candidate, options())
    assert summary["status"] == "unavailable"
    assert rows[0]["reason"] == "paired episodes have different env_seed"


def test_trigger_stage_denominator_includes_missed_onsets(tmp_path):
    arm = FakeArm()
    arm.ds["alert"] = 0.
    arm.ds.loc[0, "alert"] = 1.
    labels = [dict(episode_key="e%d" % ep, onset_control=10, onset_interval=[9, 10],
                   status="available", rule_version="test", onset_confidence="heuristic", truth_validated=False)
              for ep in range(6)]
    path = tmp_path / "onsets.json"
    path.write_text(json.dumps(labels))
    report = trigger_vs_onset.analyze(arm, options(onsets=path, triggers=["alert"]))
    row = report["tables"]["triggers"][0]
    assert row["onset_episode_denominator"] == 6
    assert row["hit_rate"] == pytest.approx(1 / 6)
    stage = report["tables"]["stages"][0]
    assert stage["onset_episode_exposure_denominator"] == 6
    assert stage["hit_rate"]["estimate"] == pytest.approx(1 / 6)
    lead = report["tables"]["episodes"][0]
    assert lead["lead_interval_controls"] == [7., 8.]
    assert lead["truth_validated"] is False


def test_missing_onsets_report_unavailable():
    with pytest.raises(C.Unavailable, match="onsets"):
        trigger_vs_onset.analyze(FakeArm(), options())


def test_duplicate_and_conflicting_attempt_identity_rejected():
    arm = FakeArm()
    arm.ds = pd.concat([arm.ds, arm.ds.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate decision"):
        provenance.analyze(arm)
    arm = FakeArm()
    arm.ds.loc[0, "task_id"] = 99
    with pytest.raises(ValueError, match="conflicting"):
        C.inputs(arm)


def test_synthetic_reader_and_all_cli_artifacts(tmp_path):
    from exp.offline_search.debug.fixtures import make_synthetic_arm
    from exp.offline_search.debug import reader
    root = tmp_path / "run"
    make_synthetic_arm(root)
    shutil.copytree(root / "runs/synthetic", root / "runs/replicate")
    arm = reader.open_arm(root, "synthetic")
    for module in (provenance, divergence, follow_vs_look, camera_shadow, stage_ledger, call_value, exposure_hazard):
        report = module.analyze(arm, options())
        assert "coverage" in report
        assert json.dumps(C.clean(report), allow_nan=False)
    for name in ("provenance", "divergence", "follow_vs_look", "camera_shadow", "stage_ledger", "call_value", "exposure_hazard", "trigger_vs_onset"):
        out = tmp_path / name
        proc = subprocess.run([sys.executable, "-m", "exp.offline_search.debug.tools.decision." + name,
            "--run-root", str(root), "--arms", "synthetic", "--out", str(out), "--procs", "2", "--bootstraps", "10"],
            text=True, capture_output=True)
        assert proc.returncode == 0, proc.stderr
        assert (out / (name + ".json")).exists()
        assert (out / (name + ".md")).exists()
        assert list(out.glob("*.csv"))
    proc = subprocess.run([sys.executable, "-m", "exp.offline_search.debug.tools.decision.churn",
        "--run-root", str(root), "--arms", "synthetic", "replicate", "--out", str(tmp_path / "churn"),
        "--bootstraps", "10"], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "churn/churn.md").exists()
    result = json.loads((tmp_path / "churn/churn.json").read_text())
    assert result["arms"]["replicate"]["tables"]["decomposition"][0]["churn"] == 0


def test_missing_client_join_and_gapped_decisions_withhold_causal_estimates():
    arm = FakeArm()
    arm.ds["server_join"] = "verified"
    arm.ds.loc[0, "server_join"] = "missing_client"
    assert call_value.analyze(arm, options())["status"] == "unavailable"
    assert exposure_hazard.analyze(arm, options())["status"] == "unavailable"
    arm = FakeArm()
    arm.ds = arm.ds.drop(index=0)
    assert "sequence has gaps" in call_value.analyze(arm, options())["reason"]


def test_p3_model_pricing_requires_attestation():
    arm = FakeArm()
    arm.manifest["adapter"] = "r6p3.v2"
    assert C.metadata(arm)["model"] == "pi05"  # Explicit arm manifest is attestation.
    del arm.manifest["model"]
    with pytest.raises(C.Unavailable, match="price table"):
        stage_ledger.analyze(arm)
    arm.profile_metadata = {"model": "groot"}
    assert stage_ledger.analyze(arm)["tables"]["stages"][0]["measured_work"] == pytest.approx(12 * .148)


def test_p3_uniform_and_legacy_adapter_fields_resolve():
    arm = FakeArm()
    for i, row in arm.ds.iterrows():
        arm.ds.at[i, "diag"] = {"assignment": {"uniform": row.coin}}
    arm.ds.drop(columns="coin", inplace=True)
    assert call_value.analyze(arm, options())["coverage"]["available"] == 12


def test_bootstrap_single_init_per_task_cannot_estimate_within_task_uncertainty():
    rows = pd.DataFrame([dict(task_id=i, init=0, numerator=float(i % 2), denominator=1.) for i in range(4)])
    report = C.cluster_interval(rows, bootstraps=100)
    assert report["estimate"] == .5
    assert report["interval_status"] == "unavailable"


def test_camera_shadows_do_not_drop_second_pure_policy_library():
    arm = FakeArm()
    n = len(arm.ds)
    full = arm.augments["shadow_look"]
    full["lib"] = np.array(["current"] * n)
    full["lib_dense"] = np.array(["dense"] * n)
    full["cache_chunk_dense"] = full["cache_chunk"] + .1
    camera = arm.augments["camera_shadow"]
    camera["wrist_cache_chunk_dense"] = camera["wrist_cache_chunk"] + .1
    camera["third_cache_chunk_dense"] = camera["third_cache_chunk"] + .1
    report = camera_shadow.analyze(arm)
    assert report["coverage"]["denominator"] == 4 * n
    assert report["coverage"]["available"] == 4 * n
    assert {row["lib"] for row in report["tables"]["decisions"]} == {"current", "dense"}


def test_lottery_future_valve_and_stall_endpoints_require_exact_clocks():
    from exp.offline_search.debug.tools.decision.outcomes import endpoints
    arm = FakeArm()
    arm.ds["shadow_delta"] = .25
    arm.ds["stall_entry"] = False
    arm.ds.loc[1, "stall_entry"] = True
    ds, _ = C.inputs(arm)
    values = endpoints(arm, ds)
    first = values[ds.decision_id.iloc[0]]
    assert first["valve_statistic_5"] == .25
    assert first["stall_entry_5"] == 1.
    assert first["object_relative_drift_5"] is None
    assert first["valve_statistic_20"] is None


def test_blind_trigger_domain_does_not_require_anchor_only_missing_flags(tmp_path):
    arm = FakeArm()
    for i, row in arm.ds.iterrows():
        if not row.vision:
            arm.ds.at[i, "diag"] = {"os_sf_valve_fire": float(i == 1)}
    labels = [dict(episode_key="e%d" % ep, onset_control=10, status="available") for ep in range(6)]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    report = trigger_vs_onset.analyze(arm, options(onsets=path, triggers=["os_sf_valve_fire"]))
    assert report["coverage"]["available"] == 6
    assert report["tables"]["triggers"][0]["hit"] == 1


def test_unknown_failure_onset_is_not_a_false_alert_negative(tmp_path):
    arm = FakeArm()
    arm.ds["alert"] = 1.
    labels = [dict(episode_key="e%d" % ep, onset_control=None, status="available", label="never_reached") for ep in range(6)]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    report = trigger_vs_onset.analyze(arm, options(onsets=path, triggers=["alert"]))
    summary = report["tables"]["triggers"][0]
    assert summary["no_onset_episode_denominator"] == 3
    assert summary["onset_time_unavailable"] == 3
    assert summary["false_alert"] == 3
