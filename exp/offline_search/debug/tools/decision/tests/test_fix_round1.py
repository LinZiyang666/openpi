"""Regressions against stock terminal names, valve landing and real producer fields."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from exp.offline_search.debug import reader
from exp.offline_search.debug.fixtures import make_synthetic_arm
from exp.offline_search.debug.tools.decision import common as C
from exp.offline_search.debug.tools.decision import (call_value, churn, divergence,
    exposure_hazard, follow_vs_look, stage_ledger, trigger_vs_onset)
from exp.offline_search.debug.tools.decision.library import MemberActions
from exp.offline_search.debug.tools.decision.outcomes import endpoints
from .test_profiles import FakeArm, options


def test_step_cap_absorbs_short_failed_terminal_windows():
    arm = FakeArm()
    arm.eps.loc[1, "termination_reason"] = "step_cap"
    decisions, _ = C.inputs(arm)
    last = decisions[decisions.episode_key == "e1"].iloc[-1]
    value = endpoints(arm, decisions)[last.decision_id]
    assert value["valid"] and value["final_success"] == 0.
    for horizon in (5, 10, 20):
        assert value["eef_displacement_" + str(horizon)] == pytest.approx(.5)
        assert value["predicate_delta_" + str(horizon)] == 0.
    assert value["next_look_controls"] is None and value["next_look_censored"]
    arm.eps.loc[1, "termination_reason"] = "incomplete"
    decisions, _ = C.inputs(arm)
    assert endpoints(arm, decisions)[last.decision_id]["eef_displacement_20"] is None


def test_exception_excludes_even_full_windows_without_an_error_string():
    arm = FakeArm()
    arm.eps.loc[0, "termination_reason"] = "exception"
    decisions, _ = C.inputs(arm)
    values = endpoints(arm, decisions)
    for row in decisions[decisions.episode_key == "e0"].to_dict("records"):
        value = values[row["decision_id"]]
        assert value["valid"] is False
        assert all(v is None for k, v in value.items() if k not in ("valid", "next_look_censored", "endpoint_reasons"))
        assert set(value["endpoint_reasons"].values()) == {"exception/infrastructure error is invalid and excluded"}
    for tool in (call_value, exposure_hazard):
        report = tool.analyze(arm, options())
        assert report["status"] == "unavailable" and "exception" in report["reason"]
    pairs, summary = churn.compare(FakeArm(), arm, options())
    assert pairs[0]["status"] == "unavailable" and summary["outcome_available"] == 5


def test_sf_valve_lands_in_vision_blind_extras_and_overrides_stale_diag(tmp_path):
    arm = FakeArm()
    arm.ds["blind_extras"] = [{} for _ in range(len(arm.ds))]
    for i, row in arm.ds.iterrows():
        arm.ds.at[i, "diag"] = {"os_sf_valve_fire": 0.}
        if row.decision_seq == 2:
            arm.ds.at[i, "blind_extras"] = {"os_sf_valve_fire": 0.}
    arm.ds.at[2, "blind_extras"] = {"os_sf_valve_fire": 1.}
    labels = [dict(episode_key="e%d" % i, onset_control=15, status="available") for i in range(6)]
    path = tmp_path / "onsets.json"
    path.write_text(json.dumps(labels))
    report = trigger_vs_onset.analyze(arm, options(onsets=path, triggers=["os_sf_valve_fire"]))
    summary = report["tables"]["triggers"][0]
    assert summary["available"] == 6 and summary["hit"] == 1
    assert report["tables"]["episodes"][0]["lead_controls"] == 3.
    assert report["tables"]["leads"][0]["decision_seq"] == 2
    carry = next(r for r in report["tables"]["stages"] if r["stage"] == "carry")
    assert carry["hit_rate"]["estimate"] == pytest.approx(1 / 6)


def test_stage_proxy_uses_weighted_catalogue_and_keeps_causal_certification_explicit():
    arm = FakeArm()
    arm.ds.drop(columns="stage_pre", inplace=True)
    arm.ds["weights"] = [[.4, .4, .2]] * len(arm.ds)
    decisions, _ = C.inputs(arm)
    assert set(decisions.stage) == {"catalog:0"}
    assert set(decisions.stage_source) == {"catalog_descriptive"}
    assert decisions.stage_mass.iloc[0] == {"0": .8, "1": .2}
    report = call_value.analyze(arm, options())
    assert {r["stage"] for r in report["tables"]["contrasts"]} == {"unknown"}
    assert all(r["stage_pre_reason"] for r in report["tables"]["support"])
    arm.ds["weights"] = [[.25, .25, .5]] * len(arm.ds)
    decisions, _ = C.inputs(arm)
    assert decisions.stage.iloc[0] == "unknown" and "tied" in decisions.stage_reason.iloc[0]
    arm.catalog = lambda: None
    decisions, _ = C.inputs(arm)
    assert decisions.stage_source.iloc[0] == "unavailable"
    assert decisions.stage_reason.iloc[0] == "library row catalog unavailable"


def test_nested_r8_library_size_used_in_both_action_reports():
    arm = FakeArm()
    arm.manifest["arm_spec"] = {"r8": {"library_size": 500, "library": "bpool_cs"}}
    arm.manifest["library_size"] = 1000
    arm.server_meta["library_size"] = 123
    assert C.metadata(arm)["library_size"] == 500
    assert divergence.analyze(arm)["tables"]["offsets"][0]["library_size"] == 500
    assert follow_vs_look.analyze(arm)["tables"]["decisions"][0]["library_size"] == 500
    arm.manifest = {"model": "pi05", "r8": {"library_size": 50}}
    assert C.metadata(arm)["library_size"] == 50
    arm.profile_metadata = {"library_size": 250}
    assert C.metadata(arm)["library_size"] == 250


def member_fixture(arm, tmp_path):
    chunks = np.zeros((4, 10, 32), np.float32)
    chunks[1, :5, :6] = 2.
    chunks[1, 5:, :6] = 4.
    chunks[2, :, :6] = 8.
    chunks[:, :, 6:] = np.arange(4)[:, None, None] * 1e6
    path = tmp_path / "action.npy"
    np.save(path, chunks, allow_pickle=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    arm.server_meta["artifacts"] = {"library_current_action.npy": dict(path=str(path), sha256=sha, status="available")}
    arm.ds["rows"] = [[0, 1, 2]] * len(arm.ds)
    arm.ds["weights"] = [[.5, .5, 0.]] * len(arm.ds)
    return path


def test_member_spread_uses_carried_tail_and_excludes_padding_and_gripper(tmp_path):
    arm = FakeArm()
    member_fixture(arm, tmp_path)
    result = follow_vs_look.analyze(arm)
    row = result["tables"]["decisions"][0]
    assert row["member_spread"] == 2. and row["member_spread_offset"] == 5
    assert row["disagreement_over_member_spread"] == pytest.approx(.05)
    assert result["member_spread_coverage"]["available"] == 12
    arm.ds.loc[1, "src"] = "follow"
    arm.ds.at[1, "diag"] = dict(follow_source="successor_heads", successor_rows=[0, 2, -1])
    row = follow_vs_look.analyze(arm)["tables"]["decisions"][0]
    assert row["member_spread"] == 4. and row["member_spread_offset"] == 0
    assert row["src"] == "follow"
    arm.ds.at[1, "diag"] = {}
    row = follow_vs_look.analyze(arm)["tables"]["decisions"][0]
    assert row["member_spread"] is None and "alignment unavailable" in row["member_spread_reason"]


def test_member_spread_rejects_changed_library_and_marks_missing_store(tmp_path):
    arm = FakeArm()
    row = follow_vs_look.analyze(arm)["tables"]["decisions"][0]
    assert row["member_spread_status"] == "unavailable" and row["member_spread_reason"]
    path = member_fixture(arm, tmp_path)
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(C.Unavailable, match="sha256"):
        MemberActions(arm).spread(arm.ds.iloc[1].to_dict(), 5)


def test_member_spread_remains_available_without_deferred_shadow(tmp_path):
    arm = FakeArm()
    member_fixture(arm, tmp_path)
    del arm.augments["shadow_look"]
    report = follow_vs_look.analyze(arm)
    assert report["status"] == "unavailable" and report["coverage"]["available"] == 0
    assert report["member_spread_coverage"]["available"] == 12
    assert report["tables"]["decisions"][0]["member_spread"] == 2.
    assert report["coverage"]["reason"] == "served_chunk or shadow_look cache_chunk unavailable"


def test_scratch_catalogue_is_keyed_and_attested_by_capture(tmp_path):
    arm = FakeArm()
    arm.manifest["suite"] = "synthetic"
    member_fixture(arm, tmp_path)
    directory = tmp_path / "catalog/pi05_synthetic_current"
    directory.mkdir(parents=True)
    table = arm.catalog()
    path = directory / "rows.parquet"
    table.to_parquet(path, index=False)
    metadata = dict(model="pi05", suite="synthetic", lib="current", lib_sha="fixture",
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    store_arrays={"action": arm.server_meta["artifacts"]["library_current_action.npy"]})
    manifest = directory / "rows.json"
    manifest.write_text(json.dumps(metadata))
    arm.profile_catalog_root = tmp_path
    assert C.catalog(arm).equals(table)
    assert arm.profile_catalog_provenance["path"] == str(path)
    del arm.profile_catalog_table
    arm.server_meta["artifacts"]["library_current_rs.npy"] = {"sha256": "captured-state"}
    metadata["store_arrays"]["rs"] = {"sha256": "different-state"}
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(C.Unavailable, match="rs fingerprint"):
        C.catalog(arm)
    del metadata["store_arrays"]["rs"]
    metadata["store_arrays"]["action"] = {"sha256": "wrong-library"}
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(C.Unavailable, match="fingerprint"):
        C.catalog(arm)
    metadata["lib"] = "dense"
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="identity"):
        C.catalog(arm)


def test_unproduced_drifts_have_source_reasons_and_client_motion_is_available():
    arm = FakeArm()
    # Unsourced diagnostic scalars cannot replace an aligned physical reference.
    arm.ds["robot_demo_drift"] = 100.
    arm.ds["object_relative_drift"] = 100.
    decisions, _ = C.inputs(arm)
    value = endpoints(arm, decisions)[decisions.decision_id.iloc[0]]
    assert value["eef_displacement_5"] == pytest.approx(.5)
    assert value["robot_demo_drift_5"] is None and value["object_relative_drift_5"] is None
    assert "reference unavailable" in value["endpoint_reasons"]["robot_demo_drift_5"]
    assert "object frames unavailable" in value["endpoint_reasons"]["object_relative_drift_5"]
    for i in range(len(arm.ds)):
        e = int(arm.ds.episode_key.iloc[i][1:]) % 3
        arm.ds.at[i, "diag"] = dict(support=[0, 1, 2], propensities=[1 / 3] * 3, drawn_e=e, coin=(e + .5) / 3)
        arm.ds.loc[i, "coin"] = (e + .5) / 3
    report = exposure_hazard.analyze(arm, options())
    absent = report["endpoint_coverage"]["object_relative_drift_5"]
    assert absent["available"] == 0 and absent["unavailable_reasons"]
    row = next(r for r in report["tables"]["contrasts"] if r["endpoint"] == "object_relative_drift_5")
    assert row["status"] == "unavailable" and row["endpoint_unavailable_reasons"]


def test_sources_remain_cache_tail_and_true_follow_despite_sf_diagnostics():
    arm = FakeArm()
    arm.ds.at[1, "diag"] = dict(os_sf_valve_fire=0., os_sf_extension=0.)
    arm.ds.loc[3, "src"] = "follow"
    sources = {r["src"] for r in follow_vs_look.analyze(arm)["tables"]["curves"]}
    assert sources == {"cache_tail", "follow"}
    row = next(r for r in divergence.analyze(arm)["tables"]["offsets"] if r["decision_seq"] == 1)
    assert row["src"] == "cache_tail"


@pytest.mark.parametrize("legacy_init", [True, False])
def test_per_process_reader_layout_tail_reporting_and_subset_init_cli(tmp_path, legacy_init):
    root = tmp_path / "run"
    make_synthetic_arm(root)
    server = root / "runs/synthetic/debug/server_fixture"
    path = server / "decisions.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for row in records:
        row["orig_init_state_idx"] = row["init"] + 100
        if legacy_init:
            # An old server's original-state index is normalized by the reader.
            row["init"] = row.pop("orig_init_state_idx")
    for episode_path in (root / "runs/synthetic/debug/client").glob("*/episode.json"):
        episode = json.loads(episode_path.read_text())
        episode["orig_init_state_idx"] = episode["init"] + 100
        episode_path.write_text(json.dumps(episode))
    halfway = len(records) // 2
    path.rename(server / "old_decisions.txt")
    for pid, rows in ((1, records[:halfway]), (2, records[halfway:])):
        text = "".join(json.dumps(row) + "\n" for row in rows)
        if pid == 2:
            text += '{"killed":'
        (server / ("decisions_%d.jsonl" % pid)).write_text(text)
    (server / "meta.json").rename(server / "meta_1.json")
    (server / "meta_2.json").write_text((server / "meta_1.json").read_text())
    arm = reader.open_arm(root, "synthetic")
    decisions, episodes = C.inputs(arm)
    assert len(decisions) == len(records) and len(arm.server_metas) == 2
    assert (decisions.orig_init_state_idx == decisions.init + 100).all()
    assert set(decisions.init) == set(episodes.init)
    assert arm.read_issues[0]["code"] == "truncated_jsonl_tail"
    assert stage_ledger.analyze(arm)["status"] == "available"
    out = tmp_path / "report"
    proc = subprocess.run([sys.executable, "-m", "exp.offline_search.debug.tools.decision.stage_ledger",
        "--run-root", str(root), "--arms", "synthetic", "--out", str(out), "--bootstraps", "0"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((out / "stage_ledger.json").read_text())["arms"]["synthetic"]
    assert result["input_diagnostics"][0]["code"] == "truncated_jsonl_tail"


@pytest.mark.parametrize("tail", ['{"partial":', json.dumps(dict(episode_key="e1", onset_control=10))])
def test_onset_jsonl_tail_is_skipped_and_reported(tmp_path, tail):
    path = tmp_path / "onsets.jsonl"
    path.write_text(json.dumps(dict(episode_key="e0", onset_control=10)) + '\n' + tail)
    arm = FakeArm()
    arm.ds["alert"] = 0.
    result = trigger_vs_onset.analyze(arm, options(onsets=path, triggers=["alert"]))
    assert result["onset_input_diagnostics"][0]["code"] == "truncated_jsonl_tail"
    assert result["coverage"]["available"] == 1


def test_single_smoke_reference_churn_cli_has_explicit_unavailability(tmp_path):
    root = tmp_path / "run"
    make_synthetic_arm(root)
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-m", "exp.offline_search.debug.tools.decision.churn",
        "--run-root", str(root), "--arms", "synthetic", "--out", str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((out / "churn.json").read_text())["arms"]["synthetic"]
    assert result["status"] == "unavailable" and "distinct candidate/reference" in result["reason"]
    assert result["episode_denominator"] == 6
