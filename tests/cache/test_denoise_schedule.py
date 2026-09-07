"""DenoiseSchedule: the index <-> timestep contract both model loops share."""

from __future__ import annotations

import pytest

from openpi.cache.types import (
    CANONICAL_DENOISE_TIMESTEPS,
    DIRECTION_ASC,
    DIRECTION_DESC,
    PI05_V1,
    DenoiseSchedule,
    groot_n15_schedule,
    schedule_from_id,
)


def test_pi05_schedule_reproduces_the_historical_canonical_set():
    """Every existing warm-start yaml validated against exactly this set."""
    legacy = frozenset(round(1.0 - i / 10, 4) for i in range(1, 10))
    assert PI05_V1.timestep_set == legacy
    assert CANONICAL_DENOISE_TIMESTEPS == legacy
    assert PI05_V1.timesteps == (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def test_groot_id_carries_the_step_count_and_is_derived_not_enumerated():
    assert groot_n15_schedule(4).schedule_id == "groot_n15_k4_v1"
    assert groot_n15_schedule(8).schedule_id == "groot_n15_k8_v1"
    assert groot_n15_schedule(6).timesteps == tuple(
        round(i / 6, 4) for i in range(1, 6)
    )


def test_groot_timesteps_ascend_and_remaining_steps_count_down():
    g4 = groot_n15_schedule(4)
    assert g4.timesteps == (0.25, 0.5, 0.75)
    assert [g4.remaining_steps(t) for t in g4.timesteps] == [3, 2, 1]
    g8 = groot_n15_schedule(8)
    assert [g8.remaining_steps(t) for t in g8.timesteps] == [7, 6, 5, 4, 3, 2, 1]


def test_pi05_remaining_steps_match_its_descending_loop():
    assert [PI05_V1.remaining_steps(t) for t in PI05_V1.timesteps] == list(
        range(9, 0, -1)
    )


def test_the_two_families_share_only_the_midpoint():
    """0.5 is a legal timestep of every schedule here; the identity, not the
    value set, is what tells them apart."""
    assert groot_n15_schedule(4).timestep_set & PI05_V1.timestep_set == {0.5}
    assert groot_n15_schedule(8).timestep_set & PI05_V1.timestep_set == {0.5}


def test_snapshot_index_round_trips_and_rejects_foreign_timesteps():
    g8 = groot_n15_schedule(8)
    for index in range(1, 8):
        assert g8.snapshot_index(g8.snapshot_t(index)) == index
    with pytest.raises(ValueError, match="not a recoverable timestep"):
        g8.snapshot_index(0.3)
    with pytest.raises(ValueError, match="not a recoverable timestep"):
        PI05_V1.snapshot_index(0.25)
    with pytest.raises(ValueError):
        g8.snapshot_index(0.0)  # the pure-noise start is not a resume point


def test_snapshot_index_uses_half_up_rounding_not_bankers():
    # 0.5 * 4 = 2.0 exactly; a value a hair below must still land on step 2.
    assert groot_n15_schedule(4).snapshot_index(0.49999) == 2


def test_replay_timestep_accumulates_for_pi05_and_divides_for_groot():
    acc = 1.0
    for _ in range(3):
        acc = acc + (-1.0 / 10)
    assert PI05_V1.replay_timestep(3) == acc
    assert groot_n15_schedule(4).replay_timestep(1) == 0.25


@pytest.mark.parametrize(
    "bad",
    [
        "groot_n15_v1",
        "groot_n15_k0_v1",
        "groot_n15_k1_v1",
        "pi05_v2",
        "",
        "GROOT_N15_K4_V1",
    ],
)
def test_ids_without_a_valid_step_count_are_rejected(bad):
    with pytest.raises(
        ValueError, match="unknown denoise schedule id|num_inference_timesteps"
    ):
        schedule_from_id(bad)


def test_round_trip_through_the_id():
    assert schedule_from_id("groot_n15_k8_v1") == groot_n15_schedule(8)
    assert schedule_from_id("pi05_v1") is PI05_V1


def test_construction_refuses_a_degenerate_loop_or_unknown_direction():
    with pytest.raises(ValueError, match="num_steps"):
        DenoiseSchedule("x", 1, DIRECTION_ASC)
    with pytest.raises(ValueError, match="direction"):
        DenoiseSchedule("x", 4, "sideways")
    with pytest.raises(ValueError):
        groot_n15_schedule(True)
    assert DenoiseSchedule("x", 4, DIRECTION_DESC).timesteps == (0.75, 0.5, 0.25)
