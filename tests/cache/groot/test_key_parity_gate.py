"""The G0-D2 gate implementation itself, on synthetic data.

A gate is only worth running if it fails when it should. These cases pin that:
a benign quantisation passes, a perturbation big enough to flip the retrieved
winner fails, a constant field is excluded from the rank check instead of
dividing by a zero distance, and an episode whose required fields are constant
is reported as unable to answer rather than as a pass.
"""

from __future__ import annotations

import pytest
import torch

from exp.robocasa365.groot_key_parity import REL_ERROR_TOL, check_key_parity

METRICS = {
    "vision_0": "cosine",
    "vision_1": "cosine",
    "prompt_emb": "cosine",
    "robot_state": "l2",
}


def _episode(n_steps=6, dim=32, seed=0):
    """Distinct per-step keys, plus a prompt that never changes within an episode."""
    generator = torch.Generator().manual_seed(seed)
    steps = []
    prompt = torch.randn(dim, generator=generator)
    for t in range(n_steps):
        steps.append(
            {
                "vision_0": torch.randn(dim, generator=generator),
                "vision_1": torch.randn(dim, generator=generator),
                "prompt_emb": prompt.clone(),
                "robot_state": torch.tensor([float(t), 0.0, 1.0]),
            }
        )
    return steps


def _quantised(steps, dtype=torch.float16):
    """What the offline path does to the same values: a storage round trip."""
    return [{k: v.to(dtype).float() for k, v in step.items()} for step in steps]


def test_a_benign_storage_round_trip_passes():
    offline = _episode()
    online = _quantised(offline)  # online is the perturbed one; offline is the library
    report = check_key_parity(online, offline, METRICS)
    assert report.passed, report.summary()
    assert report.error_gate_passed
    assert report.rank_gate_passed


def test_constant_prompt_is_reported_degenerate_and_not_divided_by():
    offline = _episode()
    report = check_key_parity(_quantised(offline), offline, METRICS)
    assert report.degenerate_fields == ["prompt_emb"]
    prompt = report.fields["prompt_emb"]
    assert prompt.rank_ok is None and prompt.min_margin is None
    # The error gate still applies to it.
    assert prompt.max_rel_error <= REL_ERROR_TOL


def test_perturbation_that_flips_the_winner_fails_the_rank_gate():
    offline = _episode()
    online = _quantised(offline)
    # Make step 0's online key look like step 3's library entry.
    online[0]["vision_0"] = offline[3]["vision_0"].clone()
    report = check_key_parity(online, offline, METRICS)
    assert not report.rank_gate_passed
    assert report.fields["vision_0"].rank_ok is False
    assert report.fields["vision_0"].first_rank_failure == 0
    assert not report.passed


def test_error_gate_catches_a_large_drift_even_when_ranking_survives():
    offline = _episode()
    online = _quantised(offline)
    online[2]["robot_state"] = offline[2]["robot_state"] * 1.05  # 5% off
    report = check_key_parity(online, offline, METRICS)
    assert not report.error_gate_passed
    assert not report.passed


def test_episode_without_movement_cannot_answer_the_gate():
    """A frozen robot_state makes the required field degenerate; that is a FAIL."""
    offline = _episode()
    for step in offline:
        step["robot_state"] = torch.tensor([1.0, 0.0, 1.0])
    report = check_key_parity(_quantised(offline), offline, METRICS)
    assert "robot_state" in report.degenerate_fields
    assert not report.required_fields_judged
    assert not report.passed
    assert "cannot answer the gate" in report.summary()


def test_zero_norm_reference_falls_back_to_absolute_error():
    offline = _episode(n_steps=3)
    for index, step in enumerate(offline):
        step["vision_1"] = torch.zeros(32) if index == 0 else torch.randn(32)
    online = [{k: v.clone() for k, v in step.items()} for step in offline]
    online[0]["vision_1"] = torch.zeros(32)  # identical, so absolute error is 0
    report = check_key_parity(online, offline, METRICS)
    assert report.fields["vision_1"].max_rel_error == pytest.approx(0.0, abs=1e-6)


def test_l2_field_is_scored_by_distance_not_cosine():
    """Cosine cannot separate two states on the same ray; L2 can."""
    offline = [
        {"robot_state": torch.tensor([1.0, 0.0])},
        {"robot_state": torch.tensor([2.0, 0.0])},
        {"robot_state": torch.tensor([3.0, 0.0])},
    ]
    online = [{k: v.clone() for k, v in step.items()} for step in offline]
    assert check_key_parity(online, offline, {"robot_state": "l2"}).fields[
        "robot_state"
    ].rank_ok
    # Under cosine every one of these is identical, so the field reads as degenerate.
    cosine_report = check_key_parity(online, offline, {"robot_state": "cosine"})
    assert cosine_report.fields["robot_state"].rank_ok is False


def test_step_count_mismatch_is_refused():
    offline = _episode(n_steps=4)
    with pytest.raises(ValueError, match="online steps vs"):
        check_key_parity(_quantised(offline)[:3], offline, METRICS)
