"""Tests for the l10 D+ init resolver (plan §3-B2)."""

import numpy as np
import pytest

from exp.zixuan_proposal.build_l10_dplus_resolver import (
    match_by_vision_fingerprint,
    resolve_from_recorded_attrs,
)


# ------------------------------------------------------------------
# Canonical default: recorded attrs
# ------------------------------------------------------------------
def test_resolve_from_recorded_attrs():
    rows = [
        {"trajectory_id": "episode_0000_x", "task_id": 0, "orig_init_state_idx": 13},
        {"trajectory_id": "episode_0001_y", "task_id": 0, "orig_init_state_idx": 0},
    ]
    m = resolve_from_recorded_attrs(rows)
    assert m["episode_0000_x"] == (0, 13)
    assert m["episode_0001_y"] == (0, 0)


def test_resolve_rejects_missing_and_duplicate():
    with pytest.raises(ValueError, match="missing"):
        resolve_from_recorded_attrs([{"trajectory_id": "e", "task_id": 0}])
    dup_ident = [
        {"trajectory_id": "a", "task_id": 1, "orig_init_state_idx": 4},
        {"trajectory_id": "b", "task_id": 1, "orig_init_state_idx": 4},
    ]
    with pytest.raises(ValueError, match="duplicate identity"):
        resolve_from_recorded_attrs(dup_ident)


def test_resolve_rejects_duplicate_trajectory_id():
    # A repeated trajectory_id must not silently overwrite a resolved identity.
    dup_tid = [
        {"trajectory_id": "episode_0001", "task_id": 0, "orig_init_state_idx": 2},
        {"trajectory_id": "episode_0001", "task_id": 0, "orig_init_state_idx": 4},
    ]
    with pytest.raises(ValueError, match="duplicate trajectory_id"):
        resolve_from_recorded_attrs(dup_tid)


# ------------------------------------------------------------------
# Vision fingerprint matching
# ------------------------------------------------------------------
def test_fingerprint_unique_match():
    refs = {0: np.array([1.0, 0, 0]), 2: np.array([0, 1.0, 0]), 4: np.array([0, 0, 1.0])}
    q = np.array([0.02, 0.999, 0.0])  # near init 2
    assert match_by_vision_fingerprint(q, refs) == 2


def test_fingerprint_scene_distinguishes_identical_proprioception():
    # Mandatory rejection-fixture intent: two inits that would be identical under
    # robot proprioception are DISTINCT under the scene-sensitive vision fingerprint,
    # so the resolver assigns each to the correct init (never certifies on pose alone).
    refs = {0: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    assert match_by_vision_fingerprint(np.array([0.999, 0.02]), refs) == 0
    assert match_by_vision_fingerprint(np.array([0.02, 0.999]), refs) == 2


def test_fingerprint_rejects_ambiguous():
    refs = {0: np.array([1.0, 0.0]), 2: np.array([1.0, 0.02])}  # nearly identical scenes
    with pytest.raises(ValueError, match="ambiguous"):
        match_by_vision_fingerprint(np.array([1.0, 0.01]), refs)


def test_fingerprint_rejects_miss():
    refs = {0: np.array([1.0, 0.0, 0.0]), 2: np.array([0.0, 1.0, 0.0])}
    with pytest.raises(ValueError, match="abs threshold"):
        match_by_vision_fingerprint(np.array([0.0, 0.0, 1.0]), refs)  # orthogonal to all
