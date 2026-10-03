from copy import deepcopy
import numpy as np
import pytest

from exp.offline_search.debug.tools.physical.common import Unavailable
from exp.offline_search.debug.tools.physical import (
    pairs,
    segmentation,
    segmentation_bench,
)


def make_twin(ep):
    lever = deepcopy(ep)
    lever.meta["arm"] = "lever"
    lever.meta["episode_key"] = "lever_a1"
    lever.controls["action"][12, 0] += 0.01
    lever.controls["qpos"][12:, 0] += 0.02
    lever.controls["eef_pos"][12:, 0] += 0.02
    lever.controls["obj_pos"][20:, 0, 0] += 0.04
    return lever


def test_twins_verify_prefix_and_event_aligned_onset(physical_episode):
    lever = make_twin(physical_episode)
    rec, curves, events = pairs.compare(
        lever,
        physical_episode,
        envelope={"object_gap_threshold": 0.03, "eef_gap_threshold": 0.03},
        twin=True,
    )
    assert rec["first_action_difference"] == 12
    assert rec["first_state_difference"] == 12
    assert rec["object_divergence_onset"] == 20
    assert rec["perturbation_control"] == 12
    assert len(curves) == physical_episode.n
    assert events and all("event_offset" in r for r in events)


def test_pair_seed_and_reset_rejected(physical_episode):
    lever = make_twin(physical_episode)
    lever.meta["env_seed"] = 603
    with pytest.raises(Unavailable, match="env_seed differs"):
        pairs.compare(lever, physical_episode)
    lever.meta["env_seed"] = 7
    lever.meta["reset_state_sha256"] = "different"
    with pytest.raises(Unavailable, match="reset_state_sha256 differs"):
        pairs.compare(lever, physical_episode)


def test_bad_twin_prefix_and_pi05_refused(physical_episode):
    lever = make_twin(physical_episode)
    lever.controls["qpos"][8, 0] += 0.001
    with pytest.raises(Unavailable, match="not bit-identical"):
        pairs.compare(lever, physical_episode, twin=True)
    lever.manifest["model"] = "pi05"
    with pytest.raises(Unavailable, match="GR00T"):
        pairs.compare(lever, physical_episode, twin=True)


def test_uncalibrated_physical_onset_is_unavailable(physical_episode):
    lever = make_twin(physical_episode)
    rec, _, _ = pairs.compare(lever, physical_episode)
    assert rec["first_object_bit_difference"] == 20
    assert rec["object_divergence_onset"] is None
    assert rec["object_onset_status"] == "unavailable"


def test_bitwise_comparison_checks_signed_zero():
    a = np.array([[0.0], [0.0]])
    b = np.array([[0.0], [-0.0]])
    assert pairs.first_bit_difference(a, b) == 1


def test_holdout_never_enters_fits(physical_episode):
    train = deepcopy(physical_episode)
    train.meta["success"] = True
    held = deepcopy(train)
    held.meta.update(init=30, episode_key="holdout")
    held.controls["eef_pos"] *= 1000
    held.controls["action"] *= -1000
    a, b = segmentation.fit([train]), segmentation.fit([train, held])
    assert a == b
    assert "holdout" not in a["training_keys"]


def test_exact_r7_gripper_baseline(physical_episode):
    ep = physical_episode
    ep.meta["success"] = True
    # Modes 0,0,1,0,1,1 -> three-row majority suppresses singleton 1/0.
    modes = [0, 0, 1, 0, 1, 1] + [1] * 10
    for seq, mode in enumerate(modes):
        ep.controls["action"][ep.controls["decision_seq"] == seq, 6] = (
            1.0 if mode else -1.0
        )
    f = segmentation.features(ep)
    fitted = segmentation.fit([ep])
    expected_modes = np.array(modes, bool)
    expected_modes[1:-1] = (
        np.array(modes[:-2]) + np.array(modes[1:-1]) + np.array(modes[2:])
    ) >= 2
    expected = [
        2 + 5 * i for i in np.flatnonzero(expected_modes[1:] != expected_modes[:-1]) + 1
    ]
    assert segmentation.r7_boundaries(ep, f, fitted) == expected


def test_all_causal_candidates_suffix_invariant(physical_episode):
    ep = physical_episode
    ep.meta["success"] = True
    ep.controls["eef_pos"][:, 0] = np.sin(np.arange(ep.n) / 10) * 0.1
    f = segmentation.features(ep)
    fitted = segmentation.fit([ep])
    changed = {k: v.copy() for k, v in f.items()}
    for key in ("position", "translation", "angular", "command"):
        changed[key][45:] += 100.0
    for candidate in segmentation.candidates(fitted):
        a = segmentation.causal_scores(f, fitted, candidate)
        b = segmentation.causal_scores(changed, fitted, candidate)
        np.testing.assert_array_equal(a[:46], b[:46])  # at 45 uses samples <45
        assert segmentation.prefix_audit(f, fitted, candidate)


def test_boundary_matching_is_one_to_one_and_maximal():
    result = segmentation.boundary_quality([0, 4], [3, 7], 4)
    assert result["matched_boundaries"] == 2
    result = segmentation.boundary_quality([1, 2, 3], [2], 1)
    assert result["matched_boundaries"] == 1
    assert result["precision"] == 1 / 3


def test_change_point_and_waypoint_reconstruction():
    x = np.r_[np.zeros((20, 2)), np.ones((20, 2)) * 5]
    assert segmentation.mean_change_points(x, 5.0) == [20]
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    assert 1 in segmentation.rdp(points, 0.1)


def test_risk_uses_predecision_stops_at_first_onset_and_censors(physical_episode):
    train = deepcopy(physical_episode)
    train.meta["success"] = True
    train.meta["episode_key"] = "train"
    held = deepcopy(physical_episode)
    held.meta.update(init=30, episode_key="holdout")
    tables = segmentation_bench.build([train, held])
    held_risk = [r for r in tables["risk_windows"] if r["split"] == "holdout"]
    assert held_risk
    assert all(r["control_start"] <= 20 for r in held_risk)
    assert all(
        r["latest_observed_control"] == r["control_start"] - 1 for r in held_risk
    )
    assert any(r["bad_event_next20"] for r in held_risk)
    assert all(r["passed"] for r in tables["prefix_replay"])
    onsets = {(ep.meta["arm"], ep.key): {"onset_control": None} for ep in (train, held)}
    tables = segmentation_bench.build([train, held], independent_onsets=onsets)
    held_risk = [r for r in tables["risk_windows"] if r["split"] == "holdout"]
    assert any(
        r["status"] == "unavailable" and "censored" in r["reason"] for r in held_risk
    )
    success_risk = [r for r in tables["risk_windows"] if r["episode_key"] == "train"]
    assert any(
        r["competing_success"] and r["status"] == "available" for r in success_risk
    )


def test_risk_metrics_ties_and_cluster_support():
    rows = [
        {
            "status": "available",
            "bad_event_next20": y,
            "score": 0.5,
            "flagged": True,
            "lead_controls": 10,
            "suite": "s",
            "task_id": 0,
            "init": i,
        }
        for i, y in enumerate([False, True])
    ]
    result = segmentation_bench.pr_metrics(rows)
    assert result["auprc"] == 0.5
    assert result["clusters"] == 2


def test_orientation_waypoints_use_train_rotation_scale(physical_episode):
    ep = physical_episode
    ep.meta["success"] = True
    theta = np.sin(np.arange(ep.n) / 8.0) * 0.4
    ep.controls["eef_quat"] = np.column_stack(
        [np.zeros(ep.n), np.zeros(ep.n), np.sin(theta / 2.0), np.cos(theta / 2.0)]
    )
    f = segmentation.features(ep)
    fitted = segmentation.fit([ep])
    points = segmentation.geometry_points(f, fitted)
    assert points.shape == (ep.n, 12)
    knots = sorted(
        set(
            [2]
            + segmentation.retrospective(ep, f, fitted, "waypoint_x1.0")
            + [ep.n - 1]
        )
    )
    error = segmentation.reconstruction(f, knots, 2, ep.n)
    assert error["SO3_rotation_max"] >= 0
    assert error["SO3_rotation_rms"] < 0.2


def test_twin_sigma_diagnostics_join_recorded_arrays(physical_episode):
    lever = make_twin(physical_episode)

    class Reader:
        def __init__(self, value):
            self.value = value

        def decision_arrays(self, keys, ids):
            if keys == ["served_chunk"]:
                return {"served_chunk": np.ones((1, 5, 7)) * self.value}
            return {"state_norm": np.ones((1, 8)) * self.value}

    physical_episode.reader_arm, lever.reader_arm = Reader(0.0), Reader(1.0)
    scales = dict(
        action_dims=list(range(6)),
        action_sigma=np.ones(7),
        state_sigma=np.ones(8),
        state_active=np.ones(8, bool),
        gripper_dim=6,
    )
    row, _, _ = pairs.compare(lever, physical_episode, twin=True, scales=scales)
    assert row["sigma_diagnostics_status"] == "available"
    assert row["action_gap_sigma_rms"] == 1.0
    assert row["state_gap_sigma_lag1"] == 1.0


def test_library_gripper_centers_use_normalized_heads(physical_episode, tmp_path):
    import json

    ep = physical_episode
    ep.meta["success"] = True
    action = np.zeros((2, 5, 7))
    action[0, :, 6], action[1, :, 6] = -0.25, 0.75
    np.save(tmp_path / "action.npy", action)
    np.save(tmp_path / "success.npy", np.ones(2, bool))
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(exec_steps=5, gripper_dim=6))
    )
    fitted = segmentation.fit([ep], library_root=tmp_path)
    assert fitted["gripper_threshold"] == 0.25

    class Reader:
        def decision_arrays(self, keys, ids):
            chunks = np.zeros((len(ids), 5, 7))
            for i, did in enumerate(ids):
                chunks[i, :, 6] = -0.25 if int(did.rsplit(":", 1)[1]) < 6 else 0.75
            return {"served_chunk": chunks}

    ep.reader_arm = Reader()
    f = segmentation.baseline_commands(ep, segmentation.features(ep), fitted)
    assert segmentation.r7_boundaries(ep, f, fitted) == [32]
    assert f["command"][4] == -0.25
