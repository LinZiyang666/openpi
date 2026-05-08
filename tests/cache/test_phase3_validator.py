"""Validator tests for Phase 3 ``weighted_sum_zero_nan`` composer (plan §8.2).

The new composer requires:
  - tier_thresholds.full_hit  (mandatory)
  - tier_thresholds.warm_start (mandatory)
  - warm_start <= full_hit (equality allowed)
  - warm_start_t (mandatory)
  - warm_start_t CP1-only (inherited; CP3 has no warm-start payload)
  - non-empty declared dependencies (>= 1 non-zero weight)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
import torch

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.config import ConfigValidationError, load_cache_config
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


# ----------------------------------------------------------------------
# Fixtures (mirror tests/cache/test_warm_fallback_cp_guard.py)
# ----------------------------------------------------------------------


@pytest.fixture
def library_pkl(tmp_path: Path) -> Path:
    a = torch.ones(2, dtype=torch.float32)
    s = torch.ones(2, dtype=torch.float32)
    library_stats = LibraryStats(
        action_sigma=a, action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones(2, dtype=torch.bool),
    )
    entries = [
        CacheEntry(
            id=f"e{i}",
            checkpoint_id=CheckpointID.CP1,
            query_keys={"robot_state": torch.tensor([float(i), 0.0], dtype=torch.float32)},
            payload=CachePayload(
                action_chunk=torch.tensor([[float(i), 0.0]], dtype=torch.float32),
            ),
            trajectory_id="traj",
            step_idx=i,
        )
        for i in range(4)
    ]
    p = tmp_path / "fix.pkl"
    with open(p, "wb") as fh:
        pickle.dump({"entries": entries, "library_stats": library_stats}, fh)
    return p


@pytest.fixture
def calib_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "calib.jsonl"
    with p.open("w") as fh:
        for i in range(60):
            fh.write(json.dumps({"factor_raw": {
                "jerk_online_action__p1_f1": float(i) / 60.0,
            }}) + "\n")
    return p


def _yaml(
    library_pkl: Path,
    calib_path: Path,
    *,
    cp_name: str,
    composer_yaml: str,
) -> str:
    return f"""\
enabled: true
keys:
  robot_state: {{enabled: true, weight: 1.0}}
key_builder: {{type: placeholder}}
backend:
  type: in_memory
  vector_dims: {{robot_state: 2}}
  in_memory:
    preload_path: {library_pkl}
    index_type: brute_force
checkpoints:
  {cp_name}:
    gate: {{type: always_search}}
    search_strategy: {{type: weighted_rrf_knn, rrf_k: 60, top_k: 1}}
    judge:
      type: composite
      normalization:
        type: zscore
        params: {{}}
        stats_source: {{type: offline}}
      factors:
        - type: jerk_online_action
          params: {{windows: [{{past: 1, future: 1}}]}}
      calibration:
        type: percentile_rolling
        params: {{window_size: 50}}
        samples_source:
          type: offline
          offline:
            path: {calib_path}
            format: jsonl
      composer:
{composer_yaml}
"""


_VALID = (
    "        type: weighted_sum_zero_nan\n"
    "        weights: {jerk_online_action__p1_f1: 1.0}\n"
    "        tier_thresholds: {full_hit: 0.5, warm_start: 0.2}\n"
    "        warm_start_t: 0.5"
)


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_valid_phase3_yaml_loads(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    p = tmp_path / "ok.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp1", composer_yaml=_VALID))
    cfg = load_cache_config(p)
    composer = cfg.checkpoints["cp1"].judge.composer
    assert composer.type == "weighted_sum_zero_nan"
    assert composer.tier_thresholds == {"full_hit": 0.5, "warm_start": 0.2}
    assert composer.warm_start_t == 0.5


# ----------------------------------------------------------------------
# Mandatory threshold fields
# ----------------------------------------------------------------------


def test_missing_warm_start_threshold_rejected(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    composer_yaml = (
        "        type: weighted_sum_zero_nan\n"
        "        weights: {jerk_online_action__p1_f1: 1.0}\n"
        "        tier_thresholds: {full_hit: 0.5}\n"
        "        warm_start_t: 0.5"
    )
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp1", composer_yaml=composer_yaml))
    with pytest.raises(ConfigValidationError, match="requires 'warm_start'"):
        load_cache_config(p)


def test_missing_full_hit_threshold_rejected(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    composer_yaml = (
        "        type: weighted_sum_zero_nan\n"
        "        weights: {jerk_online_action__p1_f1: 1.0}\n"
        "        tier_thresholds: {warm_start: 0.2}\n"
        "        warm_start_t: 0.5"
    )
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp1", composer_yaml=composer_yaml))
    with pytest.raises(ConfigValidationError, match="requires 'full_hit'"):
        load_cache_config(p)


# ----------------------------------------------------------------------
# Threshold ordering
# ----------------------------------------------------------------------


def test_warm_start_above_full_hit_rejected(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    """plan §3.2: warm_start <= full_hit; strictly above is rejected."""
    composer_yaml = (
        "        type: weighted_sum_zero_nan\n"
        "        weights: {jerk_online_action__p1_f1: 1.0}\n"
        "        tier_thresholds: {full_hit: 0.3, warm_start: 0.5}\n"
        "        warm_start_t: 0.5"
    )
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp1", composer_yaml=composer_yaml))
    with pytest.raises(ConfigValidationError, match=r"must be <= 'full_hit'"):
        load_cache_config(p)


def test_warm_start_equal_full_hit_accepted(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    """plan §3.2: warm_start == full_hit is legal (degenerate WS path)."""
    composer_yaml = (
        "        type: weighted_sum_zero_nan\n"
        "        weights: {jerk_online_action__p1_f1: 1.0}\n"
        "        tier_thresholds: {full_hit: 0.5, warm_start: 0.5}\n"
        "        warm_start_t: 0.5"
    )
    p = tmp_path / "ok.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp1", composer_yaml=composer_yaml))
    cfg = load_cache_config(p)
    assert cfg.checkpoints["cp1"].judge.composer.tier_thresholds == {
        "full_hit": 0.5, "warm_start": 0.5,
    }


# ----------------------------------------------------------------------
# warm_start_t requirements
# ----------------------------------------------------------------------


def test_missing_warm_start_t_rejected(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    composer_yaml = (
        "        type: weighted_sum_zero_nan\n"
        "        weights: {jerk_online_action__p1_f1: 1.0}\n"
        "        tier_thresholds: {full_hit: 0.5, warm_start: 0.2}"
    )
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp1", composer_yaml=composer_yaml))
    with pytest.raises(ConfigValidationError, match="warm_start_t is required"):
        load_cache_config(p)


def test_warm_start_t_rejected_on_cp3(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    """Inherited CP1-only rule via the (5a-5c) block — CP3 has no warm payload."""
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp3", composer_yaml=_VALID))
    with pytest.raises(ConfigValidationError, match="warm_start_t is only supported on CP1"):
        load_cache_config(p)


# ----------------------------------------------------------------------
# Non-empty declared dependencies
# ----------------------------------------------------------------------


def test_all_zero_weights_rejected(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    composer_yaml = (
        "        type: weighted_sum_zero_nan\n"
        "        weights: {jerk_online_action__p1_f1: 0.0}\n"
        "        tier_thresholds: {full_hit: 0.5, warm_start: 0.2}\n"
        "        warm_start_t: 0.5"
    )
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, cp_name="cp1", composer_yaml=composer_yaml))
    with pytest.raises(ConfigValidationError, match="at least one non-zero weight"):
        load_cache_config(p)
