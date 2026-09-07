"""The offline builder reads snapshots under the file's stamp and seals the library's schedule."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from exp.common.build_in_memory_cache_artifact import build_artifact
from openpi.cache.types import PI05_V1, groot_n15_schedule

BUILDER = "cp1_groot_libero_mean_pool"  # two cameras, 8-wide state


def _write_groot_episode(path, *, schedule=None, snapshots=True, task="t"):
    with h5py.File(path, "w") as f:
        f.attrs["success"] = True
        f.attrs["task"] = task
        if schedule is not None:
            f.attrs["denoise_schedule_id"] = schedule.schedule_id
            f.attrs["denoising_num_steps"] = schedule.num_steps
        for step in range(2):
            g = f.create_group(f"step_{step:04d}")
            g.create_dataset("vision_0", data=np.ones((256, 2048), np.float16))
            g.create_dataset("vision_1", data=np.full((256, 2048), 2.0, np.float16))
            g.create_dataset("prompt_emb", data=np.full((5, 2048), 3.0, np.float16))
            g.create_dataset("robot_state", data=np.arange(8, dtype=np.float32))
            g.create_dataset("clean_action", data=np.ones((16, 32), np.float32))
            if snapshots and schedule is not None:
                g.create_dataset("noise_action_0", data=np.zeros((16, 32), np.float32))
                for i in range(1, schedule.num_steps):
                    g.create_dataset(
                        f"noise_action_{i}",
                        data=np.full((16, 32), float(i), np.float32),
                    )


@pytest.mark.parametrize("num_steps", [4, 8])
def test_stamped_corpus_is_sealed_with_its_schedule_and_keyed_at_i_over_n(
    tmp_path, num_steps
):
    schedule = groot_n15_schedule(num_steps)
    _write_groot_episode(tmp_path / "ep0.h5", schedule=schedule)
    _write_groot_episode(tmp_path / "ep1.h5", schedule=schedule)
    artifact = build_artifact(str(tmp_path), BUILDER, workers=-1)
    assert artifact["schedule_id"] == schedule.schedule_id
    payload = artifact["entries"][0].payload
    assert payload.schedule_id == schedule.schedule_id
    assert payload.denoising_num_steps == num_steps
    assert tuple(sorted(payload.intermediates)) == schedule.timesteps
    # index i landed at t = i/N, and the snapshot value proves which index it was.
    assert float(payload.intermediates[schedule.snapshot_t(1)][0, 0]) == 1.0
    assert 0.9 not in payload.intermediates


def test_unstamped_corpus_without_snapshots_has_no_schedule(tmp_path):
    _write_groot_episode(tmp_path / "ep0.h5")
    artifact = build_artifact(str(tmp_path), BUILDER, workers=-1)
    assert artifact["schedule_id"] is None
    assert artifact["entries"][0].payload.intermediates is None


def test_mixed_schedules_are_refused(tmp_path):
    _write_groot_episode(tmp_path / "ep0.h5", schedule=groot_n15_schedule(4))
    _write_groot_episode(tmp_path / "ep1.h5", schedule=groot_n15_schedule(8))
    with pytest.raises(ValueError, match="different denoise schedules"):
        build_artifact(str(tmp_path), BUILDER, workers=-1)


def test_partially_stamped_corpus_is_refused(tmp_path):
    _write_groot_episode(tmp_path / "ep0.h5", schedule=groot_n15_schedule(4))
    _write_groot_episode(tmp_path / "ep1.h5")
    with pytest.raises(ValueError, match="carry no denoise_schedule_id"):
        build_artifact(str(tmp_path), BUILDER, workers=-1)


def test_pi05_corpus_under_a_groot_builder_is_refused(tmp_path):
    _write_groot_episode(tmp_path / "ep0.h5", schedule=PI05_V1)
    with pytest.raises(ValueError, match="model family"):
        build_artifact(str(tmp_path), BUILDER, workers=-1)


def test_a_stamped_file_missing_a_snapshot_is_refused(tmp_path):
    schedule = groot_n15_schedule(4)
    path = tmp_path / "ep0.h5"
    _write_groot_episode(path, schedule=schedule)
    with h5py.File(path, "a") as f:
        del f["step_0001"]["noise_action_2"]
    with pytest.raises(ValueError, match="do not match"):
        build_artifact(str(tmp_path), BUILDER, workers=-1)


def test_expected_schedule_is_checked_before_building_entries(tmp_path):
    schedule = groot_n15_schedule(8)
    _write_groot_episode(tmp_path / "ep0.h5", schedule=schedule)
    with pytest.raises(ValueError, match="does not match the requested"):
        build_artifact(
            str(tmp_path),
            BUILDER,
            workers=-1,
            expected_schedule_id="groot_n15_k4_v1",
        )
    artifact = build_artifact(
        str(tmp_path),
        BUILDER,
        workers=-1,
        expected_schedule_id=schedule.schedule_id,
    )
    assert artifact["schedule_id"] == schedule.schedule_id
