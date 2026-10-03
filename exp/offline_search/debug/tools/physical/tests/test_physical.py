from copy import deepcopy
import json
import numpy as np
import pytest

from exp.offline_search.debug.tools.physical import forensics, selfcheck
from exp.offline_search.debug.tools.physical.adapters import EnvironmentAdapter
from exp.offline_search.debug.tools.physical.audits import grasps, drops
from exp.offline_search.debug.tools.physical.common import (
    Unavailable,
    report,
    cluster_interval,
)


def test_reference_is_last_settle_after(physical_episode):
    adapter = EnvironmentAdapter(physical_episode)
    reference, index, status = adapter.reference()
    assert index == 1 and status == "last_settle_after"
    assert reference[0, 2] == 0.1
    physical_episode.controls["obj_pos"][2, 0, 2] = 0.2
    _, data = forensics.physical(physical_episode)
    assert data["relations"][0]["lifted"][2]


def test_drop_label_onset_and_control_decision_joins(physical_episode):
    row, controls, decisions = forensics.timeline_rows(physical_episode)
    assert row["label"] == "drop"
    assert row["onset_control"] == 20
    assert row["onset_decision"] == 3
    assert controls[8]["truth_stage"] == "carry"
    assert controls[20]["truth_stage"] == "release"
    assert decisions[0]["pre_control_idx"] == 1
    assert decisions[0]["truth_stage_pre"] == "settle"
    assert len(controls) == physical_episode.n
    assert not row["truth_validated"]


@pytest.mark.parametrize(
    "variant,label",
    [
        ("miss", "grasp_miss"),
        ("held", "held_not_placed"),
        ("misplace", "misplace"),
        ("success", "success"),
        ("undone", "undone"),
        ("never", "never_reached"),
    ],
)
def test_failure_taxonomy(physical_episode, variant, label):
    ep = physical_episode
    if variant == "miss":
        ep.controls["obj_pos"][2:, 0, 2] = 0.1
    elif variant == "held":
        ep.controls["obj_pos"][8:, 0, 2] = 0.16
        ep.controls["eef_pos"][8:, 2] = 0.18
    elif variant == "misplace":
        ep.controls["obj_pos"][20:, 0] = [0.5, 0, 0.1]
        ep.controls["eef_pos"][20:] = [0.5, 0, 0.12]
    elif variant == "success":
        ep.meta["success"] = True
        ep.controls["predicates"][40:] = 1.0
    elif variant == "undone":
        ep.controls["predicates"][40:50] = 1.0
    else:
        ep.controls["eef_pos"][2:] = [2.0, 0, 0.1]
    assert forensics.analyse(ep)[0]["label"] == label


def test_missing_predicate_is_unavailable_not_false(physical_episode):
    physical_episode.controls["predicates"][15, 0] = np.nan
    tables = forensics.build([physical_episode], procs=1)
    assert tables["episodes_forensics"][0]["status"] == "unavailable"
    assert tables["controls_truth"] == []


def test_unknown_destination_does_not_guess_drop(physical_episode):
    physical_episode.meta["entities"]["predicates"] = [
        ["in", "object", "unknown_region"]
    ]
    assert forensics.analyse(physical_episode)[0]["label"] == "unknown_release"
    assert drops(physical_episode)[0]["status"] == "unavailable"


def test_selfcheck_full_id_mapping_and_resting_contact(physical_episode):
    results = selfcheck.check(physical_episode)
    assert all(r.get("passed", True) for r in results)
    physical_episode.controls["contact_geom"][3, 1] = 158
    errors = selfcheck.check(physical_episode)
    assert any(r["status"] == "unavailable" and "158" in r["reason"] for r in errors)


def test_p3_mapping_known_invalid(physical_episode):
    physical_episode.meta["p3_contact_names_invalid"] = True
    assert any(
        r["status"] == "unavailable" and "P3" in r["reason"]
        for r in selfcheck.check(physical_episode)
    )


def test_mujoco_full_names_aliases_and_quaternion_conventions(physical_episode):
    catalog = physical_episode.meta["entities"]
    catalog.pop("body")
    catalog.pop("geom")
    catalog.update(
        body_names=["table", "object_root", "target_root"],
        geom_names=[None, "table_collision", "object_collision"],
        geom_bodyid=[0, 0, 1],
        movable=[
            dict(name="object_root", body_id=1, aliases=["object"]),
            dict(name="target_root", body_id=2, aliases=["target"], role="fixture"),
        ],
    )
    adapter = EnvironmentAdapter(physical_episode)
    assert adapter.name_to_index["object"] == 0
    assert adapter.settings["obj_quat_order"] == "wxyz"
    assert adapter.contact_map()[2]["entity"] == "object_root"
    assert "unnamed" in adapter.contact_map()[0]["name"]
    _, truth = forensics.physical(physical_episode)
    assert len(truth["relations"]) == 1


def test_generic_adapter_requires_scales_and_explicit_goal_contract(physical_episode):
    physical_episode.meta["suite"] = "metaworld"
    config = dict(
        environment="metaworld",
        near_radius=0.1,
        lift_height=0.03,
        gripper_dim=6,
        close_sign=-1,
        gripper_threshold=0.0,
        goal_objects=[dict(predicate=0, object="object", destination="target")],
    )
    adapter = EnvironmentAdapter(physical_episode, config)
    assert not adapter.closed()[4]
    assert adapter.goals()[0].object == "object"
    with pytest.raises(Unavailable, match="must declare"):
        adapter.setting("place_radius")


def test_object_frame_pregrasp_and_window(physical_episode):
    physical_episode.controls["eef_pos"][3] = [0.0, 0.0, 0.12]
    rows = grasps(physical_episode)
    assert rows[0]["close_control"] == 4
    assert rows[0]["pre_control"] == 3
    assert rows[0]["lift_within_window"]


def test_rotated_object_frame(physical_episode):
    physical_episode.controls["eef_pos"][3] = [0.03, 0, 0.12]
    physical_episode.controls["obj_quat"][:, 0] = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
    row = grasps(physical_episode)[0]
    assert np.allclose(row["object_frame_xyz"], [0.0, -0.03, 0.02])
    assert np.isclose(row["object_frame_yaw"], -np.pi / 2)
    assert row["lift_within_window"]
    assert row["src"] == "cache"


def test_truncated_grasp_is_censored(physical_episode):
    ep = physical_episode
    ep.controls["eef_pos"][3] = [0, 0, 0.12]
    ep.controls["obj_pos"][2:, 0, 2] = 0.1
    ep.controls["action"][:75, 6] = -1
    row = grasps(ep)[0]
    assert row["close_control"] == 75
    assert row["lift_within_window"] is None
    assert row["outcome_status"] == "unavailable"


def test_drop_open_vs_closed_and_issuing_decision(physical_episode):
    slip = drops(physical_episode)[0]
    assert slip["loss_control"] == 20
    assert slip["mechanism"] == "closed_command_slip_candidate"
    assert slip["decision_seq"] == 3
    physical_episode.controls["action"][20:, 6] = -1
    assert drops(physical_episode)[0]["mechanism"] == "premature_open_command"


def test_report_strict_json_and_cluster_bootstrap(tmp_path):
    rows = [
        {"task_id": i, "init": 0, "value": i, "status": "available"} for i in range(3)
    ]
    interval = cluster_interval(rows, "value", draws=100)
    assert interval["clusters"] == 3
    excluded = dict(task_id=999, init=0, value=1000, status="unavailable")
    assert cluster_interval(rows + [excluded], "value", draws=100) == interval
    payload = report(
        tmp_path,
        "example",
        {"rows": rows + [{"status": "unavailable", "value": np.nan}]},
        interval,
    )
    parsed = json.loads((tmp_path / "example.json").read_text())
    assert parsed["tables"]["rows"][-1]["value"] is None
    assert payload["denominators"]["rows"]["unavailable"] == 1
    assert (tmp_path / "example.md").exists()


def test_analysis_does_not_mutate_inputs(physical_episode):
    physical_episode.manifest["physical_adapter"] = {"environment": "libero"}
    before = deepcopy(physical_episode)
    forensics.analyse(physical_episode)
    selfcheck.check(physical_episode)
    drops(physical_episode)
    assert physical_episode.meta == before.meta
    assert physical_episode.manifest == before.manifest
    for k in physical_episode.controls:
        np.testing.assert_array_equal(physical_episode.controls[k], before.controls[k])


def test_control_gaps_cannot_be_analyzed(physical_episode):
    physical_episode.controls["control_idx"][10] += 1
    with pytest.raises(Unavailable, match="contiguous"):
        forensics.analyse(physical_episode)


def test_carry_radius_calibration_is_discovery_only(physical_episode):
    from exp.offline_search.debug.tools.physical.adapters import calibrate

    train = deepcopy(physical_episode)
    train.meta["success"] = True
    holdout = deepcopy(train)
    holdout.meta.update(init=30, episode_key="holdout")
    holdout.controls["eef_pos"] *= 100.0
    assert calibrate([train]) == calibrate([train, holdout])
    config = calibrate([train])
    adapter = EnvironmentAdapter(train, config)
    assert adapter.settings["near_radius"] == 0.08
    assert (
        "holdout"
        not in next(iter(config["near_radius_by_task"].values()))["training_keys"]
    )


def test_named_drop_contacts_and_contact_force(physical_episode):
    physical_episode.controls["contact_force"] = np.tile(
        [1.0, 0, 0, 0, 0, 0], (physical_episode.n, 1)
    )
    row = drops(physical_episode)[0]
    assert row["contacts_status"] == "available"
    contact = row["named_contacts_at_loss"][0]
    assert contact["left"]["body"] == "table"
    assert contact["right"]["entity"] == "object"
    assert contact["normal_force"] == 1.0


def test_resting_check_uses_free_roots_not_articulation_children(physical_episode):
    c = physical_episode.meta["entities"]
    c["movable"][1]["role"] = "movable"
    c["joint_bodyid"] = [1, 2]
    c["joint_type"] = [0, 3]
    result = next(
        r
        for r in selfcheck.check(physical_episode)
        if r["check"] == "resting_objects_contact_table"
    )
    assert result["denominator"] == 1
    assert result["passed"]
