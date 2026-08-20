"""`sweep_mixture.constant_policy` must produce an exact two-way mixture.

The sweep is the only instrument that says what a *fixed ratio* achieves, and
the whole ts-line verdict ("the router does not beat a constant") rests on it.
So the property under test is not "it runs" but "the constant policy it writes
samples the requested teacher share exactly, whatever the arm set" -- including
R_tc, where the cheap arm is the cache rather than a distilled student. That
case used to exit rather than run, which left the variant the paper is actually
about without a baseline.
"""

import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from exp.rl_router.sweep_mixture import constant_policy  # noqa: E402
from openpi.cache.components.mlp_router_judge import ARM_SETS  # noqa: E402

DIM, HIDDEN = 6, 4
FIELDS = ("vision_0",)
DIMS = {"vision_0": DIM}


def _write(tmp_path, *, arms, p, cheap_arm):
    return constant_policy(
        tmp_path / "constant.pt", p_teacher=p, arms=arms, dim=DIM, hidden=HIDDEN,
        fields=FIELDS, dims=DIMS, mu=torch.zeros(DIM), sigma=torch.ones(DIM),
        cheap_arm=cheap_arm)


def _probs(path, arms, x):
    """Forward the saved router exactly as the judge would."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    h = torch.relu(x @ blob["W1"].T + blob["b1"])
    return torch.softmax(h @ blob["W2"].T + blob["b2"], dim=-1)


@pytest.mark.parametrize(
    ("arms", "cheap"),
    [("ts", "student"), ("tc", "cache"), ("tsc", "student"), ("tsc", "cache")],
)
@pytest.mark.parametrize("p", [0.0, 0.05, 0.3, 0.5, 0.9, 1.0])
def test_realises_the_requested_teacher_share(tmp_path, arms, cheap, p):
    path = _write(tmp_path, arms=arms, p=p, cheap_arm=cheap)
    names = ARM_SETS[arms]
    # Two very different observations: a constant policy must give the same
    # answer for both, otherwise it is not measuring a FIXED ratio.
    x = torch.stack([torch.full((DIM,), -3.0), torch.full((DIM,), 7.5)])
    probs = _probs(path, arms, x)

    assert torch.allclose(probs[0], probs[1], atol=1e-7), "policy depends on the observation"
    got = float(probs[0][names.index("teacher")])
    assert got == pytest.approx(min(max(p, 1e-6), 1 - 1e-6), abs=2e-6)
    assert float(probs[0][names.index(cheap)]) == pytest.approx(1.0 - got, abs=2e-6)
    # every other arm is excluded, not merely small
    for i, n in enumerate(names):
        if n not in ("teacher", cheap):
            assert float(probs[0][i]) < 1e-10


def test_tc_defaults_are_rejected_with_an_actionable_message(tmp_path):
    """R_tc has no student, so the default must fail loudly rather than guess."""
    with pytest.raises(SystemExit) as e:
        _write(tmp_path, arms="tc", p=0.3, cheap_arm="student")
    assert "student" in str(e.value) and "tc" in str(e.value)


def test_teacher_cannot_be_the_cheap_arm(tmp_path):
    with pytest.raises(SystemExit) as e:
        _write(tmp_path, arms="ts", p=0.3, cheap_arm="teacher")
    assert "two distinct arms" in str(e.value)


def test_meta_records_the_arm_set_the_judge_will_check(tmp_path):
    """MlpRouterJudge rejects weights whose meta.arms differs from its config."""
    path = _write(tmp_path, arms="tc", p=0.3, cheap_arm="cache")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["meta"]["arms"] == "tc"
    assert blob["W2"].shape[0] == len(ARM_SETS["tc"])


def test_trunk_is_zero_so_the_mixture_is_state_independent_by_construction(tmp_path):
    path = _write(tmp_path, arms="tc", p=0.3, cheap_arm="cache")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert float(blob["W1"].abs().max()) == 0.0
    assert float(blob["b1"].abs().max()) == 0.0
    assert float(blob["W2"].abs().max()) == 0.0
    # and b2's live gap is the logit of the requested odds
    names = ARM_SETS["tc"]
    gap = float(blob["b2"][names.index("cache")] - blob["b2"][names.index("teacher")])
    assert gap == pytest.approx(math.log(0.7 / 0.3), abs=1e-6)


def test_worker_gpu_id_cycles_an_explicit_device_list() -> None:
    """On a shared box the healthy devices are rarely 0..N-1: an explicit list
    must be cycled verbatim (repetition = weighting), and the modulo default
    must survive unchanged when no list is given."""
    from exp.rl_router.run_rl_router import worker_gpu_id

    ids = [worker_gpu_id(i, gpus=8, gpu_ids="7,7,7,4") for i in range(6)]
    assert ids == ["7", "7", "7", "4", "7", "7"]
    # whitespace and empty segments are operator noise, not devices
    assert worker_gpu_id(1, gpus=8, gpu_ids=" 7 , 4 ,") == "4"
    # no list -> the historical modulo behaviour, exactly
    assert [worker_gpu_id(i, gpus=2) for i in range(4)] == ["0", "1", "0", "1"]
    assert worker_gpu_id(5, gpus=8, gpu_ids="") == "5"
