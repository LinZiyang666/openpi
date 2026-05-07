"""Tests for ``exp/common/factor_postprocess.py`` (refactor)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

# sys.path injection — ``exp/`` is not on the package path when invoked
# directly; mirrors the runner-side bootstrap in build_in_memory_cache_artifact.
_EXP_COMMON = (Path(__file__).resolve().parents[2] / "exp" / "common").as_posix()
if _EXP_COMMON not in sys.path:
    sys.path.insert(0, _EXP_COMMON)

from factor_postprocess import (  # type: ignore[import-not-found]  # noqa: E402
    _load_offline_writers_from_yaml,
    enrich_artifact_with_factors,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _entry(eid: str, *, step_idx: int, action: list[float], state: list[float]) -> CacheEntry:
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.tensor(state, dtype=torch.float32)},
        payload=CachePayload(
            action_chunk=torch.tensor([action], dtype=torch.float32),
        ),
        trajectory_id="traj-fix",
        step_idx=step_idx,
    )


def _ten_entries() -> list[CacheEntry]:
    return [
        _entry(f"e{i}", step_idx=i, action=[float(i), 0.0], state=[float(i * 0.5), 0.0])
        for i in range(10)
    ]


def _lib_stats() -> LibraryStats:
    a = torch.ones(2, dtype=torch.float32)
    s = torch.ones(2, dtype=torch.float32)
    return LibraryStats(
        action_sigma=a, action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones(2, dtype=torch.bool),
    )


def _factors_yaml(tmp_path: Path, factor_types: list[str]) -> Path:
    import yaml
    p = tmp_path / "factors.yaml"
    p.write_text(yaml.safe_dump({
        "factors": [
            {"type": t, "params": {"windows": [{"past": 1, "future": 1}]}}
            for t in factor_types
        ]
    }))
    return p


# ----------------------------------------------------------------------
# enrich_artifact_with_factors — recompute vs reuse library_stats
# ----------------------------------------------------------------------


def test_enrich_returns_recomputed_library_stats_by_default() -> None:
    entries = _ten_entries()
    ls = enrich_artifact_with_factors(entries, [])
    assert isinstance(ls, LibraryStats)
    # Recomputed from entries (action varies 0..9 → non-trivial sigma).
    assert ls.action_sigma.numel() == 2


def test_enrich_reuses_passed_library_stats_without_recomputing() -> None:
    """G1 R3 Item 4 — sentinel survives = no recompute happened."""
    entries = _ten_entries()
    sentinel = _lib_stats()
    sentinel.action_sigma[0] = 7777.0
    out = enrich_artifact_with_factors(entries, [], library_stats=sentinel)
    assert out is sentinel
    assert float(out.action_sigma[0]) == 7777.0


def test_enrich_writes_per_entry_factors_for_offline_writers() -> None:
    from openpi.cache.components.factors.offline import JerkOfflineAction

    entries = _ten_entries()
    writer = JerkOfflineAction(windows=[(1, 1)])
    ls = enrich_artifact_with_factors(entries, [writer], library_stats=_lib_stats())
    # Boundary entries (i=0, i=9) get NaN; interior entries get finite values.
    import math
    assert entries[0].payload.factors is not None
    assert math.isnan(entries[0].payload.factors["jerk_offline_action__p1_f1"])
    assert math.isfinite(entries[5].payload.factors["jerk_offline_action__p1_f1"])


def test_enrich_preserves_existing_factor_keys() -> None:
    """Pre-existing keys in payload.factors are not overwritten by enrich."""
    from openpi.cache.components.factors.offline import JerkOfflineAction

    entries = _ten_entries()
    entries[5].payload.factors = {"legacy_keep_me": 0.42}
    enrich_artifact_with_factors(
        entries, [JerkOfflineAction(windows=[(1, 1)])], library_stats=_lib_stats(),
    )
    assert entries[5].payload.factors["legacy_keep_me"] == 0.42
    assert "jerk_offline_action__p1_f1" in entries[5].payload.factors


# ----------------------------------------------------------------------
# _load_offline_writers_from_yaml — offline-only registry name acceptance
# ----------------------------------------------------------------------


def test_load_offline_writers_from_yaml_happy_path(tmp_path: Path) -> None:
    p = _factors_yaml(tmp_path, ["jerk_offline_action", "dispersion_offline_state"])
    writers = _load_offline_writers_from_yaml(str(p))
    assert len(writers) == 2
    assert all(hasattr(w, "compute_for_episode") for w in writers)


def test_load_offline_writers_rejects_online_factor(tmp_path: Path) -> None:
    """Online factors lack ``compute_for_episode`` and must be rejected."""
    p = _factors_yaml(tmp_path, ["jerk_online_action"])
    with pytest.raises(Exception, match="compute_for_episode"):
        _load_offline_writers_from_yaml(str(p))


def test_load_offline_writers_rejects_unregistered_name(tmp_path: Path) -> None:
    p = _factors_yaml(tmp_path, ["nonexistent_factor"])
    with pytest.raises(ValueError, match="Unknown factor name"):
        _load_offline_writers_from_yaml(str(p))
