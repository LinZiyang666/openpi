"""Reading warm-start snapshots out of collected HDF5 under the file's own schedule."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from openpi.cache.types import PI05_V1, groot_n15_schedule
from openpi.collect.h5_intermediates import (
    NUM_STEPS_ATTR,
    SCHEDULE_ID_ATTR,
    episode_schedule,
    read_step_intermediates,
    snapshot_indices,
)


def _file(tmp_path, *, schedule=None, indices=(), stamp_steps=None):
    path = tmp_path / "ep.h5"
    with h5py.File(path, "w") as f:
        if schedule is not None:
            f.attrs[SCHEDULE_ID_ATTR] = schedule.schedule_id
            f.attrs[NUM_STEPS_ATTR] = (
                schedule.num_steps if stamp_steps is None else stamp_steps
            )
        grp = f.create_group("step_0000")
        grp.create_dataset("noise_action_0", data=np.zeros((2, 3), np.float32))
        for i in indices:
            grp.create_dataset(
                f"noise_action_{i}", data=np.full((2, 3), float(i), np.float32)
            )
    return path


def test_unstamped_file_reads_as_pi05_and_keeps_the_legacy_tolerance(tmp_path):
    path = _file(tmp_path, indices=(1, 2, 9))
    with h5py.File(path) as f:
        assert episode_schedule(f) is None
        inter, steps = read_step_intermediates(f["step_0000"], None)
    assert steps == PI05_V1.num_steps
    assert set(inter) == {0.9, 0.8, 0.1}
    assert float(inter[0.8][0, 0]) == 2.0


def test_unstamped_file_with_an_index_beyond_the_pi05_loop_is_refused(tmp_path):
    path = _file(tmp_path, indices=(1, 10))
    with h5py.File(path) as f, pytest.raises(ValueError, match="unstamped"):
        read_step_intermediates(f["step_0000"], None)


@pytest.mark.parametrize("num_steps", [4, 8])
def test_stamped_groot_file_maps_indices_to_ascending_timesteps(tmp_path, num_steps):
    schedule = groot_n15_schedule(num_steps)
    path = _file(tmp_path, schedule=schedule, indices=range(1, num_steps))
    with h5py.File(path) as f:
        assert episode_schedule(f) == schedule
        inter, steps = read_step_intermediates(f["step_0000"], schedule)
    assert steps == num_steps
    assert tuple(sorted(inter)) == schedule.timesteps
    # index i is the x consumed by step i, at flow time i/N -- not 1 - i/10.
    assert float(inter[schedule.snapshot_t(1)][0, 0]) == 1.0
    assert 0.9 not in inter


def test_a_four_step_file_is_not_mislabelled_as_pi05(tmp_path):
    """The silent failure this module exists to close: three snapshots read as
    t=0.9/0.8/0.7 (all legal Pi0.5 timesteps) would pass every validator."""
    schedule = groot_n15_schedule(4)
    path = _file(tmp_path, schedule=schedule, indices=(1, 2, 3))
    with h5py.File(path) as f:
        inter, _ = read_step_intermediates(f["step_0000"], episode_schedule(f))
    assert set(inter) == {0.25, 0.5, 0.75}


def test_stamped_file_with_a_gap_is_refused(tmp_path):
    schedule = groot_n15_schedule(4)
    path = _file(tmp_path, schedule=schedule, indices=(1, 3))
    with h5py.File(path) as f, pytest.raises(ValueError, match="do not match"):
        read_step_intermediates(f["step_0000"], schedule)


def test_stamped_file_with_an_extra_index_is_refused(tmp_path):
    schedule = groot_n15_schedule(4)
    path = _file(tmp_path, schedule=schedule, indices=(1, 2, 3, 4))
    with h5py.File(path) as f, pytest.raises(ValueError, match="do not match"):
        read_step_intermediates(f["step_0000"], schedule)


def test_stamp_whose_step_count_disagrees_with_its_id_is_corrupt(tmp_path):
    path = _file(
        tmp_path, schedule=groot_n15_schedule(4), indices=(1, 2, 3), stamp_steps=8
    )
    with h5py.File(path) as f, pytest.raises(ValueError, match="disagrees"):
        episode_schedule(f)


def test_init_noise_is_never_a_snapshot(tmp_path):
    path = _file(tmp_path, schedule=groot_n15_schedule(4), indices=(1, 2, 3))
    with h5py.File(path) as f:
        assert snapshot_indices(f["step_0000"]) == [1, 2, 3]
        inter, _ = read_step_intermediates(f["step_0000"], groot_n15_schedule(4))
    assert 0.0 not in inter


def test_group_without_snapshots_yields_nothing_under_either_reading(tmp_path):
    path = _file(tmp_path, schedule=groot_n15_schedule(4), indices=())
    with h5py.File(path) as f:
        assert read_step_intermediates(f["step_0000"], groot_n15_schedule(4)) == (
            None,
            None,
        )
        assert read_step_intermediates(f["step_0000"], None) == (None, None)
