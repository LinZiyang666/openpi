"""Orchestrator path: OfflineWriter list runs at episode-end and merges
factor dicts into payload.factors (refactor — new 17-factor key template).
"""

from __future__ import annotations

import torch

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.components.factors.offline import (
    DispersionOfflineState,
    JerkOfflineAction,
)
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


def _entry(eid: str, *, step_idx: int, action: list[float], state: list[float]) -> CacheEntry:
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.tensor(state, dtype=torch.float32)},
        payload=CachePayload(action_chunk=torch.tensor([action], dtype=torch.float32)),
        trajectory_id="traj",
        step_idx=step_idx,
    )


def _lib_stats() -> LibraryStats:
    a = torch.ones(2, dtype=torch.float32)
    s = torch.ones(2, dtype=torch.float32)
    return LibraryStats(
        action_sigma=a, action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones(2, dtype=torch.bool),
    )


def test_offline_writer_signature_takes_library_stats() -> None:
    """OfflineWriter Protocol signature: ``compute_for_episode(entries, library_stats)``."""
    writer = JerkOfflineAction(windows=[(1, 1)])
    entries = [_entry(f"e{i}", step_idx=i, action=[float(i), 0.0], state=[0.0, 0.0]) for i in range(5)]
    rows = writer.compute_for_episode(entries, _lib_stats())
    assert len(rows) == 5
    assert all("jerk_offline_action__p1_f1" in row for row in rows)


def test_offline_writer_required_payload_fields_default_empty() -> None:
    assert JerkOfflineAction(windows=[(1, 1)]).required_payload_fields() == set()
    assert DispersionOfflineState(windows=[(1, 1)]).required_payload_fields() == set()


def test_offline_writer_pipeline_merges_into_payload_factors() -> None:
    """Mimic the orchestrator/_build_entry_chain merge path: each writer's
    output dict is merged into ``entries[i].payload.factors`` (additive)."""
    writers = [
        JerkOfflineAction(windows=[(1, 1)]),
        DispersionOfflineState(windows=[(1, 1)]),
    ]
    entries = [_entry(f"e{i}", step_idx=i, action=[float(i), 0.0], state=[float(i * 0.5), 0.0])
               for i in range(6)]
    library_stats = _lib_stats()

    for writer in writers:
        per_entry = writer.compute_for_episode(entries, library_stats)
        for entry, factors in zip(entries, per_entry, strict=True):
            if entry.payload.factors is None:
                entry.payload.factors = {}
            entry.payload.factors.update(factors)

    interior = entries[2].payload.factors
    assert "jerk_offline_action__p1_f1" in interior
    assert "dispersion_offline_state__p1_f1" in interior
