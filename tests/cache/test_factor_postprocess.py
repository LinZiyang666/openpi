"""Tests for `exp/common/factor_postprocess.py` helpers.

Covers:
- enrich_artifact_with_factors:
    * groups by trajectory_id, sorts by step_idx
    * step_idx=None entries go to an unnamed bucket + warning
    * multiple writers' factors merge without overwriting
    * empty writers list → only LibraryStats is computed
    * tolerates both torch and numpy entry payloads
- _load_offline_writers_from_yaml:
    * registry.build() round-trip for f1b_a / f1b_t
    * F1a / F2 misconfigured as offline → ConfigValidationError
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import torch
import yaml

from exp.common.factor_postprocess import (
    _load_offline_writers_from_yaml,
    enrich_artifact_with_factors,
)
from openpi.cache.components.factors.source_window import (
    SourceWindowSmoothnessAction,
)
from openpi.cache.config import ConfigValidationError
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID


# ------------------------------------------------------------------
# Mock writer
# ------------------------------------------------------------------


class _FixedWriter:
    """Returns one factor dict per entry under a configurable prefix —
    the factor values just count entries so we can check ordering."""

    def __init__(self, prefix):
        self._prefix = prefix
        self.last_call_entries = None
        self.call_count = 0

    def required_payload_fields(self):
        return set()

    def compute_for_episode(self, entries, library_stats):
        self.last_call_entries = list(entries)
        self.call_count += 1
        return [{f"{self._prefix}_x": float(i)} for i in range(len(entries))]


def _entry(eid, traj_id, step_idx, action_chunk=None) -> CacheEntry:
    if action_chunk is None:
        action_chunk = torch.zeros(1, 2)
    return CacheEntry(
        id=eid, checkpoint_id=CheckpointID.CP1,
        query_keys={}, payload=CachePayload(action_chunk=action_chunk),
        step_idx=step_idx, trajectory_id=traj_id,
    )


# ------------------------------------------------------------------
# enrich: trajectory grouping + step_idx ordering
# ------------------------------------------------------------------


def test_enrich_groups_by_trajectory_id_and_sorts_by_step_idx():
    w = _FixedWriter("f1b_a")
    # Interleaved across two trajectories with shuffled step_idx
    entries = [
        _entry("a-2", "tA", step_idx=2),
        _entry("b-0", "tB", step_idx=0),
        _entry("a-0", "tA", step_idx=0),
        _entry("b-1", "tB", step_idx=1),
        _entry("a-1", "tA", step_idx=1),
    ]
    enrich_artifact_with_factors(entries, [w])

    # Writer was called twice (once per trajectory); each call sees
    # entries in step_idx order
    assert w.call_count == 2
    seen_ids_per_call = []
    # Reconstruct what the writer received by walking factor values
    # (factors are 0,1,2 in call order); pick traj A and traj B
    # programmatically via entry.factors values.
    for entry in entries:
        # the prefix "_x" carries the within-call index
        idx = int(entry.payload.factors["f1b_a_x"])
        seen_ids_per_call.append((entry.trajectory_id, entry.step_idx, idx))

    # Within tA: step_idx 0/1/2 → indices 0/1/2 in the writer's view
    tA = sorted([t for t in seen_ids_per_call if t[0] == "tA"])
    assert tA == [("tA", 0, 0), ("tA", 1, 1), ("tA", 2, 2)]
    tB = sorted([t for t in seen_ids_per_call if t[0] == "tB"])
    assert tB == [("tB", 0, 0), ("tB", 1, 1)]


def test_enrich_step_idx_none_goes_to_unnamed_bucket_with_warning(caplog):
    w = _FixedWriter("f")
    entries = [
        _entry("a", "tA", step_idx=1),
        _entry("b", "tA", step_idx=None),
        _entry("c", "tA", step_idx=0),
    ]
    with caplog.at_level(logging.WARNING):
        enrich_artifact_with_factors(entries, [w])

    assert any("step_idx=None" in rec.message for rec in caplog.records)
    # named entries (c, a) come first sorted; unnamed (b) appended at the
    # end → indices 0 (step 0 = c), 1 (step 1 = a), 2 (None = b)
    by_id = {e.id: e.payload.factors["f_x"] for e in entries}
    assert by_id["c"] == 0.0
    assert by_id["a"] == 1.0
    assert by_id["b"] == 2.0


# ------------------------------------------------------------------
# enrich: multiple writers merge without overwriting
# ------------------------------------------------------------------


def test_enrich_multiple_writers_merge_keys():
    w1 = _FixedWriter("f1b_a")
    w2 = _FixedWriter("f1b_t")
    entries = [_entry("a", "t", 0), _entry("b", "t", 1)]

    enrich_artifact_with_factors(entries, [w1, w2])

    for entry in entries:
        keys = set(entry.payload.factors.keys())
        assert "f1b_a_x" in keys
        assert "f1b_t_x" in keys


# ------------------------------------------------------------------
# enrich: empty writers list → only LibraryStats computed
# ------------------------------------------------------------------


def test_enrich_empty_writers_only_returns_library_stats():
    entries = [_entry("a", "t", 0), _entry("b", "t", 1)]
    ls = enrich_artifact_with_factors(entries, [])

    # No writer ran → no payload.factors written
    for entry in entries:
        assert entry.payload.factors is None
    # But LibraryStats was still computed
    assert ls.action_sigma.shape == (2,)


# ------------------------------------------------------------------
# enrich: numpy / torch dual input bridge (post-detach lifecycle)
# ------------------------------------------------------------------


def test_enrich_accepts_numpy_payload():
    w = _FixedWriter("f")
    entries = [
        _entry("a", "t", 0, action_chunk=np.zeros((1, 2), dtype=np.float32)),
        _entry("b", "t", 1, action_chunk=np.zeros((1, 2), dtype=np.float32)),
    ]
    ls = enrich_artifact_with_factors(entries, [w])

    assert ls.action_sigma.shape == (2,)
    for entry in entries:
        assert "f_x" in entry.payload.factors


# ------------------------------------------------------------------
# enrich: end-to-end with a real F1b writer
# ------------------------------------------------------------------


def test_enrich_with_real_f1b_writer_writes_window_keys():
    writer = SourceWindowSmoothnessAction(
        windows=[(1, 1)],
        descriptors=["jerk"],
        active_eps=0.01,
    )
    entries = [
        _entry(f"e{i}", "t", i, action_chunk=torch.tensor([[float(i), 0.0]]))
        for i in range(5)
    ]
    enrich_artifact_with_factors(entries, [writer])

    # Interior entry 2 has full window and finite jerk; boundary entries
    # 0 and 4 emit NaN per docs §3.4 boundary rule.
    assert not math.isnan(entries[2].payload.factors["f1b_a_jerk__p1_f1"])
    assert math.isnan(entries[0].payload.factors["f1b_a_jerk__p1_f1"])
    assert math.isnan(entries[4].payload.factors["f1b_a_jerk__p1_f1"])


# ------------------------------------------------------------------
# _load_offline_writers_from_yaml
# ------------------------------------------------------------------


def test_load_offline_writers_yaml_returns_writer_instances(tmp_path):
    yml = tmp_path / "factors.yaml"
    yml.write_text(yaml.safe_dump({
        "factors": [
            {"type": "f1b_a", "params": {
                "windows": [{"past": 0, "future": 5}],
                "descriptors": ["jerk"], "active_eps": 0.01,
            }},
            {"type": "f1b_t", "params": {
                "windows": [{"past": 0, "future": 5}],
                "descriptors": ["jerk"], "active_eps": 0.01,
            }},
        ],
    }))

    writers = _load_offline_writers_from_yaml(str(yml))
    assert len(writers) == 2
    assert all(hasattr(w, "compute_for_episode") for w in writers)


def test_load_offline_writers_yaml_rejects_online_only_factor(tmp_path):
    """F1a / F2 have no `compute_for_episode` — misconfiguration as
    OfflineWriter must fail-fast at YAML load."""
    yml = tmp_path / "bad.yaml"
    yml.write_text(yaml.safe_dump({
        "factors": [{"type": "f2", "params": {"K": 3}}],
    }))

    with pytest.raises(ConfigValidationError, match="no compute_for_episode"):
        _load_offline_writers_from_yaml(str(yml))


def test_load_offline_writers_yaml_empty_returns_empty_list(tmp_path):
    yml = tmp_path / "empty.yaml"
    yml.write_text("factors: []\n")
    assert _load_offline_writers_from_yaml(str(yml)) == []
