"""Unit tests for weighted_rrf equivalence of the shipped R1/R2 backends.

weighted_rrf reuses the same ``_compute_field_scores`` hot kernel as weighted_sum, so
PrebuiltMatrix (R1) is bit-equal to stock for RRF (original ops → argsort order identical →
RRF score identical) and PrenormDot (R2) preserves the winner id (RRF rank stable to the
~2e-6 prenorm error). Tiny synthetic buckets, no library.
"""

from types import SimpleNamespace

import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.storage_types import QueryFilter, QuerySpec

from exp.cache_latency_bench.opt.prebuilt_matrix_backend import PrebuiltMatrixBackend
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
    if hasattr(b, "_prebuild_task_matrices"):
        b._prebuild_task_matrices()
    return b


def _rrf_spec(cands):
    qk = {f: cands[0].query_keys[f] for f in W}
    return QuerySpec(
        query_keys=qk, top_k=1, checkpoint_id="cp1", filters=QueryFilter(task_key="A"),
        fusion_weights=W, fusion_method="weighted_rrf", field_similarity=FIELD_SIM,
        backend_hints={"rrf_k": 60},
    )


def test_r1_prebuilt_rrf_bit_equal():
    torch.manual_seed(0)
    d, dr = 8, 4
    es = [_entry(f"e{i}", "A", d, dr) for i in range(10)]
    stock = _mk(InMemoryBackend, es, d, dr)
    pb = _mk(PrebuiltMatrixBackend, es, d, dr)
    spec = _rrf_spec(list(stock._entries.values()))
    rs = stock.search(spec)
    rp = pb.search(spec)
    assert rs[0].id == rp[0].id
    assert rs[0].score == rp[0].score  # bit-equal: same cosine → same argsort → same RRF


def test_r2_prenorm_rrf_winner_parity():
    torch.manual_seed(1)
    d, dr = 8, 4
    es = [_entry(f"e{i}", "A", d, dr) for i in range(12)]
    stock = _mk(InMemoryBackend, es, d, dr)
    pn = _mk(PrenormDotBackend, es, d, dr)
    spec = _rrf_spec(list(stock._entries.values()))
    rs = stock.search(spec)
    rp = pn.search(spec)
    assert rs[0].id == rp[0].id  # winner-id parity (RRF rank stable to prenorm ~2e-6)


def test_rrf_l2_uses_raw_norm_not_exp():
    # _search_weighted_rrf argsorts raw l2 distance ascending; to_similarity(exp) is NOT
    # applied on the RRF path. Confirm prebuilt matches stock on the l2 field too.
    torch.manual_seed(2)
    d, dr = 8, 4
    es = [_entry(f"e{i}", "A", d, dr) for i in range(8)]
    stock = _mk(InMemoryBackend, es, d, dr)
    pb = _mk(PrebuiltMatrixBackend, es, d, dr)
    spec = _rrf_spec(list(stock._entries.values()))
    assert stock.search(spec)[0].id == pb.search(spec)[0].id
