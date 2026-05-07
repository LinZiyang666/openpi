"""4-layer composite judge yaml end-to-end validator tests (refactor)."""

from __future__ import annotations

import json
import pickle
import textwrap
from pathlib import Path

import pytest
import torch

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.config import ConfigValidationError, load_cache_config
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def library_pkl(tmp_path: Path) -> Path:
    """Tiny in-memory backend artifact carrying library_stats + 4 entries."""
    a_sigma = torch.ones(2, dtype=torch.float32)
    s_sigma = torch.ones(2, dtype=torch.float32)
    library_stats = LibraryStats(
        action_sigma=a_sigma,
        action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s_sigma,
        state_active_mask=torch.ones(2, dtype=torch.bool),
    )
    entries = [
        CacheEntry(
            id=f"e{i}",
            checkpoint_id=CheckpointID.CP1,
            query_keys={
                "robot_state": torch.tensor([float(i), 0.0], dtype=torch.float32),
            },
            payload=CachePayload(
                action_chunk=torch.tensor([[float(i), 0.0]], dtype=torch.float32),
            ),
            trajectory_id="traj-fix",
            step_idx=i,
        )
        for i in range(4)
    ]
    art = {"entries": entries, "library_stats": library_stats}
    p = tmp_path / "fix.pkl"
    with open(p, "wb") as fh:
        pickle.dump(art, fh)
    return p


@pytest.fixture
def calib_jsonl(tmp_path: Path) -> Path:
    """JSONL with 60 samples per key (≥ window_size=50)."""
    p = tmp_path / "calib.jsonl"
    with p.open("w") as fh:
        for i in range(60):
            fh.write(json.dumps({"factor_raw": {
                "jerk_online_action__p1_f1": float(i) / 60.0,
            }}) + "\n")
    return p


def _yaml(library_pkl: Path, calib_path: Path, *, replace: dict[str, str] | None = None) -> str:
    """Build a minimum valid 4-layer composite yaml.

    ``replace`` substitutes substrings to construct invalid variants.
    """
    body = textwrap.dedent(f"""\
    enabled: true
    keys:
      robot_state: {{enabled: true, weight: 1.0}}
    key_builder:
      type: placeholder
    backend:
      type: in_memory
      vector_dims: {{robot_state: 2}}
      in_memory:
        preload_path: {library_pkl}
        index_type: brute_force
    checkpoints:
      cp1:
        gate: {{type: always_search}}
        search_strategy:
          type: weighted_rrf_knn
          rrf_k: 60
          top_k: 1
        judge:
          type: composite
          normalization:
            type: zscore
            params: {{}}
            stats_source: {{type: offline}}
          factors:
            - type: jerk_online_action
              params:
                windows:
                  - {{past: 1, future: 1}}
          calibration:
            type: percentile_rolling
            params: {{window_size: 50}}
            samples_source:
              type: offline
              offline:
                path: {calib_path}
                format: jsonl
          composer:
            type: weighted_sum
            weights: {{jerk_online_action__p1_f1: 1.0}}
            tier_thresholds: {{full_hit: 0.5}}
    """)
    if replace:
        for old, new in replace.items():
            body = body.replace(old, new)
    return body


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_load_valid_4_layer_composite_yaml(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    p = tmp_path / "ok.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl))
    cfg = load_cache_config(p)
    j = cfg.checkpoints["cp1"].judge
    assert j.type == "composite"
    assert j.normalization is not None
    assert j.calibration is not None
    assert j.composer is not None
    assert len(j.factors) == 1


# ----------------------------------------------------------------------
# Validator regressions (plan §13.3)
# ----------------------------------------------------------------------


def test_legacy_factor_name_rejected(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    """Rule 3 — registry no longer registers `f1a_a`."""
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, replace={
        "jerk_online_action": "f1a_a",
    }))
    with pytest.raises(ConfigValidationError, match="not a registered factor name"):
        load_cache_config(p)


def test_zero_zero_window_rejected(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    """Rule 10 — (P, F) = (0, 0) → splice length 1 → reject."""
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, replace={
        "{past: 1, future: 1}": "{past: 0, future: 0}",
    }))
    with pytest.raises(ConfigValidationError, match=r"\(0,0\)"):
        load_cache_config(p)


def test_unknown_calibration_format_rejected(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    """Rule 8 — samples_source.offline.format ∈ {jsonl, pkl}."""
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, replace={
        "format: jsonl": "format: csv",
    }))
    with pytest.raises(ConfigValidationError, match=r"format.*csv"):
        load_cache_config(p)


def test_unsupported_warmup_stats_source_rejected(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    """Rule 7 — Layer 1 stats_source.type must be 'offline'."""
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, replace={
        "stats_source: {type: offline}": "stats_source: {type: warmup}",
    }))
    with pytest.raises(ConfigValidationError, match="stats_source.type='warmup'"):
        load_cache_config(p)


def test_composer_unknown_factor_key_rejected(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    """Rule 4 — composer.weights references a key not produced by Layer 2."""
    p = tmp_path / "bad.yaml"
    p.write_text(_yaml(library_pkl, calib_jsonl, replace={
        "weights: {jerk_online_action__p1_f1: 1.0}":
            "weights: {nonexistent_factor__p9_f9: 1.0}",
    }))
    with pytest.raises(ConfigValidationError):
        load_cache_config(p)
