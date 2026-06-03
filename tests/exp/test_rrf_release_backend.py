"""Unit tests for RrfReleaseVisionBackend (ROUND-RRF R5: lean RRF + vision release via MRO).

Asserts the multiple-inheritance composition resolves correctly (release from
ReleaseVision, lean RRF search from LeanRRF, guard from ReleaseVision), and that
post-release weighted_rrf search is unchanged for an INDEPENDENT query (search reads ``_mat``
+ ``spec.query_keys``, never the freed entry tensors). CPU-only, tiny synthetic buckets.

Note: the query must be independent of the entries (it simulates the keybuilder's live
output). A released entry's own ``query_keys`` are an empty sentinel, so using an entry as
its own query post-release is invalid by construction — exactly as in production replay,
where the query comes from the keybuilder, not from a stored entry.
"""

from types import SimpleNamespace

import pytest
import torch

from openpi.cache.storage_types import QueryFilter, QuerySpec

from exp.cache_latency_bench.opt.lean_rrf_backend import LeanRRFBackend
from exp.cache_latency_bench.opt.safe_release_backend import ReleaseUnsafeError
from exp.cache_latency_bench.opt.safe_release_rrf_backend import RrfReleaseVisionBackend

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
    b._is_frozen = True
    return b


def _spec(q):
    return QuerySpec(
        query_keys=q, top_k=1, checkpoint_id="cp1", filters=QueryFilter(task_key="A"),
        fusion_weights=W, fusion_method="weighted_rrf", field_similarity=FIELD_SIM,
        backend_hints={"rrf_k": 60},
    )


def test_mro_resolution():
    mro = [c.__name__ for c in RrfReleaseVisionBackend.__mro__]
    assert mro[:4] == [
        "RrfReleaseVisionBackend", "ReleaseVisionBackend", "LeanRRFBackend", "LeanSearchBackend",
    ]
    assert RrfReleaseVisionBackend.release_vision.__qualname__.split(".")[0] == "ReleaseVisionBackend"
    assert RrfReleaseVisionBackend._search_weighted_rrf.__qualname__.split(".")[0] == "LeanRRFBackend"
    assert RrfReleaseVisionBackend._compute_field_scores.__qualname__.split(".")[0] == "ReleaseVisionBackend"


def test_post_release_search_unchanged_independent_query():
    torch.manual_seed(0)
    d, dr = 8, 4
    es = [_entry(f"e{i}", "A", d, dr) for i in range(10)]
    rr = _mk(RrfReleaseVisionBackend, es, d, dr)
    q = {"vision_0": torch.randn(d), "vision_1": torch.randn(d), "robot_state": torch.randn(dr)}  # independent
    spec = _spec(q)
    before = rr.search(spec)
    freed = rr.release_vision()
    after = rr.search(spec)
    assert freed > 0 and rr._vision_released
    assert before[0].id == after[0].id
    assert before[0].score == after[0].score  # bit-same: search reads _mat + spec query, not entries
    assert rr._lean_rrf_hits > 0


def test_post_release_matches_lean_rrf():
    # RrfReleaseVision (post-release) winner == LeanRRF (no release) on the same query.
    torch.manual_seed(1)
    d, dr = 8, 4
    es = [_entry(f"e{i}", "A", d, dr) for i in range(12)]
    lr = _mk(LeanRRFBackend, es, d, dr)
    rr = _mk(RrfReleaseVisionBackend, list(lr._entries.values()), d, dr)  # share same entries
    rr.release_vision()
    q = {"vision_0": torch.randn(d), "vision_1": torch.randn(d), "robot_state": torch.randn(dr)}
    spec = _spec(q)
    assert lr.search(spec)[0].id == rr.search(spec)[0].id


def test_release_guard_inherited():
    torch.manual_seed(2)
    d, dr = 8, 4
    ea = [_entry(f"a{i}", "A", d, dr) for i in range(3)]
    eb = [_entry(f"b{i}", "B", d, dr) for i in range(3)]
    rr = _mk(RrfReleaseVisionBackend, ea + eb, d, dr)
    rr.release_vision()
    mixed = ea[:2] + eb[:2]  # cross-bucket -> guard raises (inherited from ReleaseVision)
    with pytest.raises(ReleaseUnsafeError):
        rr._compute_field_scores(torch.randn(d), mixed, "vision_0", {"type": "cosine"})
