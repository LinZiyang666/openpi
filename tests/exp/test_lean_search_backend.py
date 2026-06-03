"""Synthetic-data unit tests for LeanSearchBackend (ROUND 3 framework-overhead elimination).

Asserts the lean steady-state search path (whole bucket, top_k==1) returns the SAME winner
and score as the full Round-2 path (within fp32), fires its lean counter, and falls back to
the full path for non-whole-bucket candidates and top_k>1. CPU-only, tiny synthetic
buckets, no 1.1 GB library.
"""

import dataclasses
from types import SimpleNamespace

import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.storage_types import QueryFilter, QuerySpec

from exp.cache_latency_bench.opt.lean_search_backend import LeanSearchBackend

W = {"vision_0": 0.25, "vision_1": 0.4375, "robot_state": 0.3125}
FIELD_SIM = {
    "vision_0": {"type": "cosine"},
    "vision_1": {"type": "cosine"},
    "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 1.0}},
}
SCORE_NORM = {
    "type": "per_field",
    "fields": {
        "vision_0": {"method": "zscore", "params": {"mu": 0.0, "sigma": 1.0, "squash": "tanh"}},
        "vision_1": {"method": "zscore", "params": {"mu": 0.0, "sigma": 1.0, "squash": "tanh"}},
        "robot_state": {"method": "zscore", "params": {"mu": 0.0, "sigma": 1.0, "squash": "tanh"}},
    },
}


def _entry(eid, tk, d, dr, ck="cp1"):
    return SimpleNamespace(
        id=eid, checkpoint_id=ck, payload=SimpleNamespace(task_key=tk),
        query_keys={"vision_0": torch.randn(d), "vision_1": torch.randn(d), "robot_state": torch.randn(dr)},
    )


def _build(n=10, d=8, dr=4):
    es = [_entry(f"e{i}", "A", d, dr) for i in range(n)]
    b = LeanSearchBackend({"vision_0": d, "vision_1": d, "robot_state": dr})
    b._entries = {e.id: e for e in es}
    b._prebuild_task_matrices()
    return b


def _spec(b, top_k=1):
    cands = list(b._entries.values())
    qk = {f: cands[0].query_keys[f] for f in W}
    return QuerySpec(
        query_keys=qk, top_k=top_k, checkpoint_id="cp1", filters=QueryFilter(task_key="A"),
        fusion_weights=W, fusion_method="weighted_score_sum", field_similarity=FIELD_SIM,
        score_normalization=SCORE_NORM,
    )


def test_lean_matches_full_path():
    torch.manual_seed(0)
    b = _build()
    cands = list(b._entries.values())
    spec = _spec(b)
    active = b._iter_active_fields(spec)
    lean = b._search_weighted_score_sum(cands, spec, active)                  # lean override
    full = InMemoryBackend._search_weighted_score_sum(b, cands, spec, active)  # full path (prenorm _compute)
    assert lean[0].id == full[0].id
    assert abs(lean[0].score - full[0].score) < 1e-5  # same operators, only plumbing dropped
    assert b._lean_hits > 0


def test_fallback_partial_bucket():
    torch.manual_seed(1)
    b = _build()
    cands = list(b._entries.values())[1:]  # drops row 0 -> candidate i != matrix row i
    spec = _spec(b)
    active = b._iter_active_fields(spec)
    before = b._lean_hits
    b._search_weighted_score_sum(cands, spec, active)
    assert b._lean_fallbacks > 0
    assert b._lean_hits == before  # lean did NOT fire


def test_fallback_topk_gt_1():
    torch.manual_seed(2)
    b = _build()
    cands = list(b._entries.values())
    spec = _spec(b, top_k=3)
    active = b._iter_active_fields(spec)
    before = b._lean_hits
    res = b._search_weighted_score_sum(cands, spec, active)
    assert b._lean_hits == before  # top_k>1 -> fallback
    assert len(res) == 3


def test_lean_score_full_search_consistent():
    # End-to-end via backend.search(spec): the realized lean path must match a full-path
    # PrenormDot search on the same library.
    torch.manual_seed(3)
    b = _build(n=12)
    spec = _spec(b)
    lean_res = b.search(spec)
    # full path: temporarily force fallback by bypassing the lean override
    full_res = InMemoryBackend.search(b, spec)
    assert lean_res[0].id == full_res[0].id
    assert abs(lean_res[0].score - full_res[0].score) < 1e-5


def test_reordered_bucket_after_verify_falls_back():
    # G2 MAJOR regression: prime _verified_buckets with the in-order bucket, then feed a
    # reordered same-(ckpt,task,n) bucket — the O(1) endpoint check must catch it and fall
    # back rather than returning a misaligned winner.
    torch.manual_seed(6)
    b = _build()
    cands = list(b._entries.values())
    spec = _spec(b)
    active = b._iter_active_fields(spec)
    b._search_weighted_score_sum(cands, spec, active)  # prime cache (lean fires)
    assert b._lean_hits > 0
    hits_before = b._lean_hits
    reordered = list(reversed(cands))  # same (ckpt,task,n), reordered -> endpoints wrong
    b._search_weighted_score_sum(reordered, spec, active)
    assert b._lean_hits == hits_before  # lean did NOT fire on the reordered bucket
    assert b._lean_fallbacks > 0


def test_subset_after_whole_bucket_verify_falls_back():
    # A subset (different n) after a whole-bucket verify must re-scan and fall back.
    torch.manual_seed(7)
    b = _build()
    cands = list(b._entries.values())
    spec = _spec(b)
    active = b._iter_active_fields(spec)
    b._search_weighted_score_sum(cands, spec, active)  # prime whole-bucket verify
    hits_before = b._lean_hits
    b._search_weighted_score_sum(cands[1:], spec, active)  # subset n-1
    assert b._lean_hits == hits_before  # subset -> fallback
