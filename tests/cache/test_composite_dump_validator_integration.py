"""G2 R1 Item 4 — composite + dump validator integration.

Plan §13.3 rule 11 dereferences ``judge.dump.normalization.stats_source``;
the DumpConfig dataclass must therefore (a) carry the
``normalization`` field and (b) tolerate it being unset (legacy yamls
omit the field). These tests exercise both shapes through
``validate_cache_config``.
"""

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
    art = {"entries": entries, "library_stats": library_stats}
    p = tmp_path / "fix.pkl"
    with open(p, "wb") as fh:
        pickle.dump(art, fh)
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


def _yaml_with_dump(library_pkl: Path, calib_path: Path, *, dump_extra: str) -> str:
    """Composite yaml + dump block; ``dump_extra`` is yaml lines that get
    placed verbatim inside the ``judge.dump`` mapping (already indented
    to the 6-space level used by ``dump:`` body in this template).
    """
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
  cp1:
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
        type: weighted_sum
        weights: {{jerk_online_action__p1_f1: 1.0}}
        tier_thresholds: {{full_hit: 0.5}}
      dump:
        path: /tmp/_x.jsonl
        config_id: dump_smoke
{dump_extra}
        factors:
          - type: jerk_online_action
            params: {{windows: [{{past: 1, future: 1}}]}}
"""


# ----------------------------------------------------------------------
# Dump without normalization — passes
# ----------------------------------------------------------------------


def test_composite_with_dump_no_normalization_passes(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    p = tmp_path / "ok.yaml"
    p.write_text(_yaml_with_dump(library_pkl, calib_jsonl, dump_extra=""))
    cfg = load_cache_config(p)
    assert cfg.checkpoints["cp1"].judge.dump is not None
    assert cfg.checkpoints["cp1"].judge.dump.normalization is None


# ----------------------------------------------------------------------
# Dump with matching normalization — passes (validator rule 11 OK path)
# ----------------------------------------------------------------------


def test_composite_with_dump_matching_normalization_passes(
    tmp_path: Path, library_pkl: Path, calib_jsonl: Path,
) -> None:
    """Same yaml as the no-normalization variant but with matching dump.normalization.

    The dump block in ``_yaml_with_dump`` is indented at 6 spaces (under
    ``      dump:`` from ``checkpoints.cp1.judge``); ``dump_extra`` is
    inserted right after ``config_id`` and must match that indent.
    """
    # dump_extra is inserted at the dump body indent level (8 spaces from
    # column 0; ``dump:`` itself sits at 6 spaces).
    dump_extra = (
        "        normalization:\n"
        "          type: zscore\n"
        "          params: {}\n"
        "          stats_source: {type: offline}"
    )
    p = tmp_path / "match.yaml"
    p.write_text(_yaml_with_dump(library_pkl, calib_jsonl, dump_extra=dump_extra))
    cfg = load_cache_config(p)
    assert cfg.checkpoints["cp1"].judge.dump.normalization is not None
    assert cfg.checkpoints["cp1"].judge.dump.normalization.stats_source.type == "offline"
