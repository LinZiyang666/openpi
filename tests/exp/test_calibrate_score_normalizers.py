"""Tests for the offline Layer-1 normalizer calibration (Phase 1)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

# Load the calibration script as a module (it lives under exp/, not a package).
_SPEC = importlib.util.spec_from_file_location(
    "calibrate_score_normalizers",
    Path(__file__).resolve().parents[2] / "exp" / "common" / "calibrate_score_normalizers.py",
)
calib = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calib)


def _entry(eid: str, traj: str, task: str, vec: list[float]) -> CacheEntry:
    e = CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={"vision_0": torch.tensor(vec, dtype=torch.float32)},
        payload=CachePayload(action_chunk=torch.zeros(1, 32), task_key=task),
    )
    e.trajectory_id = traj
    return e


def _toy_library():
    # task A: 3 entries (each its own trajectory); task B: 3 entries.
    rng = np.random.default_rng(0)
    entries = []
    for t in range(3):
        v = (rng.standard_normal(4)).tolist()
        entries.append(_entry(f"A{t}", f"A{t}", "taskA", v))
    for t in range(3):
        v = (rng.standard_normal(4)).tolist()
        entries.append(_entry(f"B{t}", f"B{t}", "taskB", v))
    return entries


def test_loeo_excludes_own_trajectory():
    entries = _toy_library()
    same, cross = calib.collect_loeo_scores(
        entries, "vision_0", "cosine", max_queries=999, seed=0,
    )
    # Each entry is its own trajectory, so LOEO removes exactly the self pair.
    # Per query: 2 same-task others + 3 cross-task. 6 queries.
    assert same.size == 6 * 2
    assert cross.size == 6 * 3


def test_select_shortlist_bounded_topk():
    rng = np.random.default_rng(1)
    same = rng.uniform(0.95, 0.999, size=400)   # same-task: more similar
    cross = rng.uniform(0.80, 0.95, size=600)    # cross-task: less similar
    shortlist = calib.select_shortlist(same, cross, "cosine", beta=0.5, lam=0.5)
    assert 1 <= len(shortlist) <= 2
    for cand in shortlist:
        assert cand["method"] in {"affine_clip", "zscore", "logit", "neg_log_one_minus", "power"}
        assert np.isfinite(cand["J"])
        assert "params" in cand
    # Ranking is by J descending.
    assert shortlist[0]["J"] >= shortlist[-1]["J"]


def test_select_shortlist_l2_uses_exp_l2():
    rng = np.random.default_rng(2)
    same = np.abs(rng.standard_normal(400)) * 0.1   # small distances (similar)
    cross = np.abs(rng.standard_normal(600)) * 0.5  # larger distances
    shortlist = calib.select_shortlist(same, cross, "l2", beta=0.5, lam=0.5)
    methods = {c["method"] for c in shortlist}
    # exp_l2 is the only l2-compatible candidate.
    assert methods <= {"affine_clip", "zscore", "exp_l2"}
    assert shortlist  # non-empty


def test_select_shortlist_empty_on_degenerate():
    assert calib.select_shortlist(np.array([]), np.array([]), "cosine", beta=0.5, lam=0.5) == []


def test_collect_loeo_handles_numpy_query_keys():
    # cp1_* artifacts store query_keys as numpy.ndarray (not torch.Tensor); the
    # offline calibration reads the pickle directly and must coerce, not crash.
    entries = []
    for t in range(2):
        e = _entry(f"A{t}", f"A{t}", "taskA", [0.0, 0.0, 0.0, 0.0])
        e.query_keys["vision_0"] = np.random.randn(4).astype(np.float32)  # ndarray
        entries.append(e)
    for t in range(2):
        e = _entry(f"B{t}", f"B{t}", "taskB", [0.0, 0.0, 0.0, 0.0])
        e.query_keys["vision_0"] = np.random.randn(4).astype(np.float32)
        entries.append(e)
    same, cross = calib.collect_loeo_scores(
        entries, "vision_0", "cosine", max_queries=99, seed=0,
    )
    assert same.size + cross.size > 0  # produced scores without crashing
