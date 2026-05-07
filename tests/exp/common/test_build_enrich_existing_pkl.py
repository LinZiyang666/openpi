"""Smoke tests for ``build_in_memory_cache_artifact.py enrich-existing-pkl``.

Plan §16 B6 / B6.5: this is the acceptance gate for the artifact-rebuild
path. The CLI must (a) read an input pkl with library_stats + entries,
(b) run the new 17-factor OfflineWriter list, (c) merge new factor keys
into ``payload.factors`` without disturbing pre-existing keys, and (d)
NOT recompute library_stats when the input already has it.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_entry(
    eid: str,
    *,
    step_idx: int,
    state: list[float],
    action: list[float],
    legacy_factors: dict[str, float] | None = None,
) -> CacheEntry:
    payload = CachePayload(
        action_chunk=torch.tensor([action], dtype=torch.float32),
        factors=dict(legacy_factors) if legacy_factors else None,
    )
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.tensor(state, dtype=torch.float32)},
        payload=payload,
        step_idx=step_idx,
        trajectory_id="traj-smoke",
    )


def _make_artifact(n: int = 12) -> dict:
    """6+-entry chain so windows up to (1, 1) fit interior entries."""
    entries = []
    for i in range(n):
        entries.append(
            _make_entry(
                f"e{i}",
                step_idx=i,
                state=[float(i), float(i * 0.5)],
                action=[float(i * 0.1), float(i * 0.2)],
                legacy_factors={"legacy_keep_me": float(i)},
            )
        )
    a_sigma = torch.ones(2, dtype=torch.float32)
    s_sigma = torch.ones(2, dtype=torch.float32)
    library_stats = LibraryStats(
        action_sigma=a_sigma, action_active_mask=torch.ones_like(a_sigma, dtype=torch.bool),
        state_sigma=s_sigma, state_active_mask=torch.ones_like(s_sigma, dtype=torch.bool),
    )
    return {"entries": entries, "library_stats": library_stats}


def _write_factors_yaml(path: Path, factor_types: list[str]) -> None:
    spec = {
        "factors": [
            {"type": t, "params": {"windows": [{"past": 1, "future": 1}]}}
            for t in factor_types
        ]
    }
    path.write_text(yaml.safe_dump(spec))


# ----------------------------------------------------------------------
# Happy path: 8 offline factors over a 12-entry chain
# ----------------------------------------------------------------------


def test_enrich_existing_pkl_happy_path(tmp_path: Path) -> None:
    artifact = _make_artifact(n=12)
    in_pkl = tmp_path / "in.pkl"
    out_pkl = tmp_path / "out.pkl"
    factors_yaml = tmp_path / "factors.yaml"
    _write_factors_yaml(factors_yaml, ["jerk_offline_action", "dispersion_offline_state"])
    with open(in_pkl, "wb") as fh:
        pickle.dump(artifact, fh)

    proc = subprocess.run(
        [
            sys.executable, "-m", "exp.common.build_in_memory_cache_artifact",
            "enrich-existing-pkl",
            "--input", str(in_pkl),
            "--factors-yaml", str(factors_yaml),
            "--output", str(out_pkl),
        ],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3],
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"

    with open(out_pkl, "rb") as fh:
        out_art = pickle.load(fh)
    assert len(out_art["entries"]) == 12
    # Refactor: legacy factor keys are preserved (additive merge).
    sample = out_art["entries"][6]   # interior entry has both windows fitting
    assert sample.payload.factors["legacy_keep_me"] == 6.0
    # New offline-factor keys are present (window p1_f1 fits; (P+F+1) = 3 entries).
    assert "jerk_offline_action__p1_f1" in sample.payload.factors
    assert "dispersion_offline_state__p1_f1" in sample.payload.factors


# ----------------------------------------------------------------------
# Library stats are reused (not recomputed) when input pkl carries them.
# ----------------------------------------------------------------------


def test_enrich_existing_pkl_reuses_library_stats(tmp_path: Path) -> None:
    artifact = _make_artifact(n=8)
    # Tag the library_stats with a sentinel value that compute_from_entries
    # would never produce, so we can detect re-computation.
    sentinel = 7777.0
    artifact["library_stats"].action_sigma[0] = sentinel

    in_pkl = tmp_path / "in.pkl"
    out_pkl = tmp_path / "out.pkl"
    factors_yaml = tmp_path / "factors.yaml"
    _write_factors_yaml(factors_yaml, ["jerk_offline_action"])
    with open(in_pkl, "wb") as fh:
        pickle.dump(artifact, fh)

    proc = subprocess.run(
        [
            sys.executable, "-m", "exp.common.build_in_memory_cache_artifact",
            "enrich-existing-pkl",
            "--input", str(in_pkl),
            "--factors-yaml", str(factors_yaml),
            "--output", str(out_pkl),
        ],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3],
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"

    with open(out_pkl, "rb") as fh:
        out_art = pickle.load(fh)
    # Sentinel survived: library_stats was passed through, not recomputed.
    assert float(out_art["library_stats"].action_sigma[0]) == sentinel


# ----------------------------------------------------------------------
# Online factor types are rejected (no compute_for_episode surface).
# ----------------------------------------------------------------------


def test_enrich_existing_pkl_rejects_online_factor(tmp_path: Path) -> None:
    artifact = _make_artifact(n=4)
    in_pkl = tmp_path / "in.pkl"
    out_pkl = tmp_path / "out.pkl"
    factors_yaml = tmp_path / "factors.yaml"
    _write_factors_yaml(factors_yaml, ["jerk_online_action"])  # online — wrong family
    with open(in_pkl, "wb") as fh:
        pickle.dump(artifact, fh)

    proc = subprocess.run(
        [
            sys.executable, "-m", "exp.common.build_in_memory_cache_artifact",
            "enrich-existing-pkl",
            "--input", str(in_pkl),
            "--factors-yaml", str(factors_yaml),
            "--output", str(out_pkl),
        ],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3],
    )
    assert proc.returncode != 0
    assert "compute_for_episode" in proc.stderr or "OfflineWriter" in proc.stderr
    assert not out_pkl.exists()
