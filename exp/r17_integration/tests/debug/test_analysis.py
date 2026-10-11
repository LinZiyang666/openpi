import copy
import json

import numpy as np
import pytest

from exp.r17_integration.debug.analysis_profiles import decision_profile, physical_profile, write_report
from exp.r17_integration.debug.validate import digest
from exp.r17_integration.search.analyze import paired_bootstrap, summarize
from exp.r17_integration.tests.debug.fixtures import arrays, binding, record


def test_report_quality_missing_errors_cost_and_artifacts(tmp_path):
    b = binding()
    journal = [dict(record(n_decisions=1), accepted=True, status="ok", success=False),
               dict(record(init=11, n_decisions=1), accepted=True, status="ok", success=True),
               dict(record(init=12), accepted=False, status="error")]
    decisions = [record(), record(init=11)]
    expected = [record(init=i)["task_uid"] for i in (10, 11, 12)]
    report = summarize(expected, journal, decisions, task_groups={"reach-v3": "easy", "push-v3": "hard"}, binding=b,
                       full_baseline_ms={"binding_digest": digest(b), "absolute_ms": 8.,
                                         "pricing_absolute_ms": 8., "pricing_time_field": "service_ms",
                                         "capture_enabled":False,"pricing_eligible":True})
    assert report["completed"] == 2 and report["sr"] == .5 and report["errors"] == 1
    assert report["missing"] == [record(init=12)["task_uid"]]
    assert report["cost"]["absolute_ms"] == 4 and report["cost"]["normalized_within_model"] == .5
    assert report["per_task"]["push-v3"]["sr"] is None
    values, _ = arrays(b)
    physical = physical_profile({"capabilities": {"task_predicates": {"status": "unsupported", "reason": "no predicate map"}}},
                               {"qpos": np.arange(9, dtype=np.float64)})
    paths = write_report(report, tmp_path / "report", decision_cards=[decision_profile(b, values)], physical_cards=[physical])
    assert all(__import__("pathlib").Path(path).stat().st_size > 0 for path in paths.values())
    saved = json.loads((tmp_path / "report" / "report.json").read_text())
    assert saved["physical_cards"][0]["tools"]["grasp"]["status"] == "unsupported"


def test_missing_cost_is_unknown_and_cross_model_normalization_refused():
    journal = [dict(record(), accepted=True, status="ok")]
    row = record()
    del row["cost_ms"]
    report = summarize([row["task_uid"]], journal, [row], task_groups={"reach-v3": "easy"}, binding=binding())
    assert report["cost"]["absolute_ms"] is None and report["cost"]["status"] == "unsupported"
    with pytest.raises(ValueError, match="baseline"):
        summarize([row["task_uid"]], journal, [record()], task_groups={"reach-v3": "easy"}, binding=binding(),
                  full_baseline_ms={"binding_digest": "another-model", "absolute_ms": 8.})


@pytest.mark.parametrize("model", ["pi05", "groot", "fastwam", "dp"])
def test_profile_uses_active_channels_and_native_grid(model):
    b = binding(model)
    a, _ = arrays(b)
    dim = b["model"]["layout"]["valid_action_dim"]
    first = decision_profile(b, a)
    a["normalized_actions"][:, dim:] = 1e20
    second = decision_profile(b, a)
    assert first == second
    assert second["gripper_channel"] == 3 and len(second["real_cameras"]) == 1
    assert second["native_grid"] == b["model"]["default_sampler"]["native_timesteps"]


def test_paired_bootstrap_semantic_identity_and_ambiguity():
    a = [dict(record(init=i), accepted=True, status="ok", success=False) for i in range(3)]
    b = [dict(row, arm="candidate", success=True) for row in a]
    result = paired_bootstrap(a, b, samples=100)
    assert result["pairs"] == 3 and result["delta_sr"] == 1 and result["ci95"] == [1., 1.]
    b[0]["task_data_hash"] = "unpaired"
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(a, b)


def test_offline_scores_ignore_padding_and_use_mw_gripper3():
    from exp.r17_integration.search.analyze import offline_action_scores
    b = binding("groot")
    teacher = np.zeros((2, 16, 32), np.float32)
    predicted = teacher.copy()
    predicted[:, :, 3] = 2.
    predicted[:, :, 4:] = 1e20
    scores = offline_action_scores(predicted, teacher, layout=b["model"]["layout"])
    assert scores["mean_mse"] == 1. and scores["motion_mse"] == 0. and scores["gripper_mse"] == 4.
    assert scores["status"] == "offline_action_error_only"


def test_missing_native_contact_force_retains_unknown():
    profile = physical_profile({"capabilities": {"contact_force": {"status": "unsupported", "reason": "no force provider"}}},
                               {"contact_force": np.full((1, 6), np.nan, np.float64)})
    assert profile["observed_max_contact_force"] is None
    assert profile["tools"]["forensics"]["status"] == "unsupported"
