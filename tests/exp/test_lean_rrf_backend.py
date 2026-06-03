"""Unit tests for LeanRRFBackend (ROUND-RRF R3 framework-overhead elimination).

Asserts the lean weighted_rrf path is bit-identical to the parent PrenormDot
``_search_weighted_rrf`` (same mv/argsort/RRF, only plumbing dropped), fires its counter,
and falls back for non-whole-bucket / top_k>1. CPU-only, tiny synthetic buckets.
"""

from types import SimpleNamespace

import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.storage_types import QueryFilter, QuerySpec

from exp.cache_latency_bench.opt.lean_rrf_backend import LeanRRFBackend
from exp.cache_latency_bench.opt.prenorm_dot_backend import PrenormDotBackend

W = {"vision_0": 0.0625, "vision_1": 0.5, "robot_state": 0.4375}
FIELD_SIM = {
    "vision_0": {"type": "cosine"}, "vision_1": {"type": "cosine"},
    "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 1.0}},
}


def _entry(eid, tk, d, dr, ck="cp1"):
    return SimpleNamespace(
        id=eid, checkpoint_id=ck, payload=SimpleNamespace(task_key=tk),
        query_keys={"vision_0": torch.randn(d), "vision_1": torch.randn(d), "robot_state": torch.randn(dr)},
    )


def _mk(cls, entries, d, dr):
    b = cls({"vision_0": d, "vision_1": d, "robot_state": dr})
    b._entries = {e.id: e for e in entries}
    b._prebuild_task_matrices()
    return b


def _spec(cands, top_k=1):
    qk = {f: cands[0].query_keys[f] for f in W}
    return QuerySpec(
        query_keys=qk, top_k=top_k, checkpoint_id="cp1", filters=QueryFilter(task_key="A"),
        fusion_weights=W, fusion_method="weighted_rrf", field_similarity=FIELD_SIM,
        backend_hints={"rrf_k": 60},
    )


def test_lean_rrf_bit_equal_to_prenorm():
    torch.manual_seed(0)
    d, dr = 8, 4
    es = [_entry(f"e{i}", "A", d, dr) for i in range(10)]
    pn = _mk(PrenormDotBackend, es, d, dr)
    lr = _mk(LeanRRFBackend, es, d, dr)
    cands = list(lr._entries.values())
    spec = _spec(cands)
    rp = pn.search(spec)               # parent PrenormDot _search_weighted_rrf (full path)
    rl = lr.search(spec)               # lean RRF
    assert rp[0].id == rl[0].id
    assert rp[0].score == rl[0].score  # bit-equal: same mv → same argsort → same RRF
    assert lr._lean_rrf_hits > 0


def test_lean_rrf_winner_matches_stock():
    torch.manual_seed(1)
    d, dr = 8, 4
    es = [_entry(f"e{i}", "A", d, dr) for i in range(12)]
    stock = InMemoryBackend({"vision_0": d, "vision_1": d, "robot_state": dr})
    stock._entries = {e.id: e for e in es}
    lr = _mk(LeanRRFBackend, es, d, dr)
    spec = _spec(list(lr._entries.values()))
    assert stock.search(spec)[0].id == lr.search(spec)[0].id  # winner parity (prenorm vs cosine)


def test_lean_rrf_fallback_partial_bucket():
    torch.manual_seed(2)
    d, dr = 8, 4
    es = [_entry(f"e{i}", "A", d, dr) for i in range(8)]
    lr = _mk(LeanRRFBackend, es, d, dr)
    cands = list(lr._entries.values())[1:]  # partial bucket -> fallback
    spec = _spec(cands)
    before = lr._lean_rrf_hits
    lr._search_weighted_rrf(cands, spec, lr._iter_active_fields(spec))
    assert lr._lean_rrf_hits == before
    assert lr._lean_rrf_fallbacks > 0


def test_lean_rrf_fallback_topk_gt_1():
    torch.manual_seed(3)
    d, dr = 8, 4
    es = [_entry(f"e{i}", "A", d, dr) for i in range(8)]
    lr = _mk(LeanRRFBackend, es, d, dr)
    cands = list(lr._entries.values())
    spec = _spec(cands, top_k=3)
    before = lr._lean_rrf_hits
    res = lr._search_weighted_rrf(cands, spec, lr._iter_active_fields(spec))
    assert lr._lean_rrf_hits == before  # top_k>1 -> fallback
    assert len(res) == 3
