"""InMemoryBackend refuses libraries whose intermediates come from more than one loop."""

from __future__ import annotations

import pickle

import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID, groot_n15_schedule

DIMS = {"robot_state": 3}


def _entry(
    i: int, *, intermediates=None, num_steps=None, schedule_id=None
) -> CacheEntry:
    return CacheEntry(
        id=f"traj:{i}",
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.full((3,), float(i))},
        payload=CachePayload(
            action_chunk=torch.zeros(16, 32),
            intermediates=intermediates,
            denoising_num_steps=num_steps,
            task_key="t",
            schedule_id=schedule_id,
        ),
        step_idx=i,
        trajectory_id="traj",
    )


def _write(path, entries, *, schedule_id=None):
    doc = {
        "key_builder_type": "cp1_groot_libero_mean_pool",
        "checkpoint_id": "CP1",
        "vector_dims": DIMS,
        "entries": entries,
    }
    if schedule_id is not None:
        doc["schedule_id"] = schedule_id
    with open(path, "wb") as fh:
        pickle.dump(doc, fh)
    return path


def _snapshots(schedule):
    return {t: torch.zeros(16, 32) for t in schedule.timesteps}


def test_stamped_library_records_its_schedule(tmp_path):
    g4 = groot_n15_schedule(4)
    path = _write(
        tmp_path / "a.pkl",
        [_entry(0, intermediates=_snapshots(g4), num_steps=4)],
        schedule_id=g4.schedule_id,
    )
    backend = InMemoryBackend(DIMS)
    backend.load_artifact(path)
    assert backend.artifact_meta["schedule_id"] == "groot_n15_k4_v1"
    assert backend.artifact_meta["denoising_num_steps"] == 4


def test_legacy_library_is_backfilled_as_pi05(tmp_path):
    path = _write(tmp_path / "a.pkl", [_entry(0)])
    backend = InMemoryBackend(DIMS)
    backend.load_artifact(path)
    assert backend.artifact_meta["schedule_id"] == "pi05_v1"
    assert backend.fetch_payload("traj:0").schedule_id == "pi05_v1"


def test_mixed_step_counts_are_refused(tmp_path):
    g4, g8 = groot_n15_schedule(4), groot_n15_schedule(8)
    path = _write(
        tmp_path / "a.pkl",
        [
            _entry(0, intermediates=_snapshots(g4), num_steps=4),
            _entry(1, intermediates=_snapshots(g8), num_steps=8),
        ],
    )
    with pytest.raises(ValueError, match="mixes denoising_num_steps"):
        InMemoryBackend(DIMS).load_artifact(path)


def test_entries_contradicting_the_stamp_are_refused(tmp_path):
    g8 = groot_n15_schedule(8)
    path = _write(
        tmp_path / "a.pkl",
        [_entry(0, intermediates=_snapshots(g8), num_steps=8)],
        schedule_id="groot_n15_k4_v1",
    )
    with pytest.raises(ValueError, match="entries carry"):
        InMemoryBackend(DIMS).load_artifact(path)


def test_snapshots_at_foreign_timesteps_are_refused(tmp_path):
    """A pi0.5-keyed snapshot set under a GR00T stamp: t=0.9 is not a k=4 point."""
    path = _write(
        tmp_path / "a.pkl",
        [
            _entry(
                0,
                intermediates={0.9: torch.zeros(16, 32), 0.5: torch.zeros(16, 32)},
                num_steps=4,
            )
        ],
        schedule_id="groot_n15_k4_v1",
    )
    with pytest.raises(ValueError, match="not recoverable points"):
        InMemoryBackend(DIMS).load_artifact(path)


def test_unknown_stamp_is_refused(tmp_path):
    path = _write(tmp_path / "a.pkl", [_entry(0)], schedule_id="groot_n15_v1")
    with pytest.raises(ValueError, match="unknown denoise schedule id"):
        InMemoryBackend(DIMS).load_artifact(path)


def test_entry_schedule_must_match_the_artifact_stamp(tmp_path):
    g4 = groot_n15_schedule(4)
    path = _write(
        tmp_path / "a.pkl",
        [
            _entry(
                0,
                intermediates=_snapshots(g4),
                num_steps=4,
                schedule_id="groot_n15_k8_v1",
            )
        ],
        schedule_id=g4.schedule_id,
    )
    with pytest.raises(ValueError, match="payload schedule"):
        InMemoryBackend(DIMS).load_artifact(path)
