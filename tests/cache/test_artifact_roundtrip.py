"""Artifact pickle round-trip with new 17-factor key templates."""

from __future__ import annotations

import pickle
from pathlib import Path

import torch

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


def _entry_with_new_factor_keys(eid: str) -> CacheEntry:
    """Entry whose payload.factors carries the refactored 17-factor key
    template ``<descriptor>_<source>_<channel>__p<P>_f<F>``."""
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.zeros(2, dtype=torch.float32)},
        payload=CachePayload(
            action_chunk=torch.zeros(1, 2, dtype=torch.float32),
            factors={
                "jerk_offline_action__p1_f1": 0.42,
                "dispersion_offline_state__p1_f1": 0.5,
                "path_length_offline_state__p3_f3": 1.7,
            },
        ),
        trajectory_id="traj",
    )


def test_pickle_round_trip_preserves_new_factor_key_template(tmp_path: Path) -> None:
    a = torch.ones(2, dtype=torch.float32)
    s = torch.ones(2, dtype=torch.float32)
    library_stats = LibraryStats(
        action_sigma=a, action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones(2, dtype=torch.bool),
    )
    art = {
        "entries": [_entry_with_new_factor_keys(f"e{i}") for i in range(3)],
        "library_stats": library_stats,
    }
    p = tmp_path / "x.pkl"
    with open(p, "wb") as fh:
        pickle.dump(art, fh)
    with open(p, "rb") as fh:
        loaded = pickle.load(fh)
    assert len(loaded["entries"]) == 3
    e = loaded["entries"][0]
    assert "jerk_offline_action__p1_f1" in e.payload.factors
    assert "dispersion_offline_state__p1_f1" in e.payload.factors
    assert isinstance(loaded["library_stats"], LibraryStats)


def test_legacy_factor_keys_round_trip_without_breaking_loader(tmp_path: Path) -> None:
    """Existing pkl files (pre-refactor) carry legacy key names like
    ``f1b_a_jerk__p0_f5``. The pickle format stores them as plain dict
    strings, so the loader still round-trips them — refactor code simply
    does not look them up at verdict time."""
    legacy_factors = {
        "f1b_a_jerk__p0_f5": 0.1,
        "f1b_t_jerk__p0_f5": 0.2,
    }
    entry = CacheEntry(
        id="legacy",
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.zeros(2, dtype=torch.float32)},
        payload=CachePayload(
            action_chunk=torch.zeros(1, 2, dtype=torch.float32),
            factors=legacy_factors,
        ),
        trajectory_id="traj",
    )
    p = tmp_path / "legacy.pkl"
    with open(p, "wb") as fh:
        pickle.dump([entry], fh)
    with open(p, "rb") as fh:
        loaded = pickle.load(fh)
    assert loaded[0].payload.factors == legacy_factors
