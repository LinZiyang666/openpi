"""Synthetic-data unit tests for PrenormDotBackend (ROUND 2 prenorm-dot).

Asserts cosine buckets are L2-normalized fp32, the prenorm GEMV matches
``F.cosine_similarity`` within fp32 tolerance (NOT bit-identical — Round 2 changed the
numeric path), l2 fields stay bit-exact, non-safe-window inputs fall back to the parent
(no prenorm hit), zero-norm queries are NaN-safe, the prenorm path commutes with the
index_select gather, and the deferred rescore lever raises. CPU-only, tiny synthetic
buckets, no 1.1 GB library.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from openpi.cache.backends.in_memory_backend import InMemoryBackend

from exp.cache_latency_bench.opt.prenorm_dot_backend import PrenormDotBackend


def _entry(eid, task_key, query_keys, checkpoint_id="cp1"):
    return SimpleNamespace(
        id=eid,
        checkpoint_id=checkpoint_id,
        payload=SimpleNamespace(task_key=task_key),
        query_keys=query_keys,
    )


def _build(entries, dims, **kw):
    b = PrenormDotBackend(dims, **kw)
    b._entries = {e.id: e for e in entries}
    b._prebuild_task_matrices()
    return b


def test_cosine_buckets_normalized_fp32():
    torch.manual_seed(0)
    d = 8
    es = [_entry(f"e{i}", "A", {"vision_0": torch.randn(d)}) for i in range(5)]
    b = _build(es, {"vision_0": d})
    key = ("cp1", "A", "vision_0")
    m = b._mat[key]
    assert m.dtype == torch.float32
    norms = m.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert key in b._unit_keys


def test_mv_matches_cosine_within_tol():
    torch.manual_seed(1)
    d = 16
    es = [_entry(f"e{i}", "A", {"vision_0": torch.randn(d)}) for i in range(6)]
    raw = torch.stack([e.query_keys["vision_0"] for e in es])
    b = _build(es, {"vision_0": d})
    cands = list(b._entries.values())
    q = torch.randn(d)
    fs, _ = b._compute_field_scores(q, cands, "vision_0", {"type": "cosine"})
    ref = F.cosine_similarity(q.unsqueeze(0), raw)
    assert torch.allclose(fs, ref, atol=1e-5)  # prenorm-dot ~ cosine (not bit-identical)
    assert b._prenorm_hits > 0


def test_l2_bit_identical_to_parent():
    torch.manual_seed(2)
    d = 6
    es = [_entry(f"e{i}", "A", {"robot_state": torch.randn(d)}) for i in range(4)]
    b = _build(es, {"robot_state": d})
    cands = list(b._entries.values())
    q = torch.randn(d)
    fs, fm = b._compute_field_scores(q, cands, "robot_state", {"type": "l2"})
    bs, bm = InMemoryBackend._compute_field_scores(b, q, cands, "robot_state", {"type": "l2"})
    assert torch.equal(fs, bs)  # l2 path unchanged -> bit-identical
    assert torch.equal(fm, bm)


def test_fallback_cross_bucket_no_prenorm_hit():
    torch.manual_seed(3)
    d = 4
    ea = [_entry(f"a{i}", "A", {"vision_0": torch.randn(d)}) for i in range(3)]
    eb = [_entry(f"b{i}", "B", {"vision_0": torch.randn(d)}) for i in range(3)]
    b = _build(ea + eb, {"vision_0": d})
    before = b._prenorm_hits
    mixed = ea[:2] + eb[:2]
    q = torch.randn(d)
    b._compute_field_scores(q, mixed, "vision_0", {"type": "cosine"})  # cross-bucket -> super()
    assert b._prenorm_hits == before  # prenorm did NOT fire


def test_zero_norm_query_no_nan():
    torch.manual_seed(4)
    d = 4
    es = [_entry(f"e{i}", "A", {"vision_0": torch.randn(d)}) for i in range(3)]
    b = _build(es, {"vision_0": d})
    cands = list(b._entries.values())
    q = torch.zeros(d)  # zero query -> clamp_min(1e-12) guards against NaN
    fs, _ = b._compute_field_scores(q, cands, "vision_0", {"type": "cosine"})
    assert not torch.isnan(fs).any()


def test_prenorm_commutes_with_gather():
    # Whole-bucket (zero-copy) vs subset+reorder (index_select) must agree up to fp32.
    torch.manual_seed(5)
    d = 8
    es = [_entry(f"e{i}", "A", {"vision_0": torch.randn(d)}) for i in range(6)]
    b = _build(es, {"vision_0": d})
    cands = list(b._entries.values())
    q = torch.randn(d)
    full, _ = b._compute_field_scores(q, cands, "vision_0", {"type": "cosine"})       # zero-copy
    sub_cands = cands[1:][::-1]                                                        # subset + reorder
    subset, _ = b._compute_field_scores(q, sub_cands, "vision_0", {"type": "cosine"})  # index_select
    full_by_id = {c.id: full[i] for i, c in enumerate(cands)}
    for i, c in enumerate(sub_cands):
        assert torch.allclose(subset[i], full_by_id[c.id], atol=1e-6)


def test_rescore_lever_deferred_raises():
    d = 4
    es = [_entry(f"e{i}", "A", {"vision_0": torch.randn(d)}) for i in range(3)]
    b = _build(es, {"vision_0": d}, rescore_top_k=8, keep_raw=True)
    cands = list(b._entries.values())
    with pytest.raises(NotImplementedError):
        b._compute_field_scores(torch.randn(d), cands, "vision_0", {"type": "cosine"})
