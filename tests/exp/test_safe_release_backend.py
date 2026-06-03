"""Unit tests for ReleaseVisionBackend (ROUND 5 memory release).

Asserts: (1) post-release the fast path is bit-identical to pre-release (reads _mat, not
the freed entry tensors); (2) released vision tensors have 0 storage bytes; (3) the
fail-closed guard raises ReleaseUnsafeError on a forced fallback (cross-bucket) rather than
reading the empty sentinel; (4) release refuses without freeze / with an active session.
CPU-only, tiny synthetic buckets, no 1.1 GB library.
"""

from types import SimpleNamespace

import pytest
import torch

from openpi.cache.storage_types import QueryFilter, QuerySpec

from exp.cache_latency_bench.opt.safe_release_backend import ReleaseUnsafeError, ReleaseVisionBackend

W = {"vision_0": 0.25, "vision_1": 0.4375, "robot_state": 0.3125}
FIELD_SIM = {
    "vision_0": {"type": "cosine"}, "vision_1": {"type": "cosine"},
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


def _build(entries, d=8, dr=4):
    b = ReleaseVisionBackend({"vision_0": d, "vision_1": d, "robot_state": dr})
    b._entries = {e.id: e for e in entries}
    b._prebuild_task_matrices()
    b._is_frozen = True  # simulate the frozen pooled backend
    return b


def _spec(cands):
    qk = {f: cands[0].query_keys[f] for f in W}
    return QuerySpec(
        query_keys=qk, top_k=1, checkpoint_id="cp1", filters=QueryFilter(task_key="A"),
        fusion_weights=W, fusion_method="weighted_score_sum", field_similarity=FIELD_SIM,
        score_normalization=SCORE_NORM,
    )


def test_release_then_fast_path_bit_same():
    torch.manual_seed(0)
    b = _build([_entry(f"e{i}", "A", 8, 4) for i in range(10)])
    cands = list(b._entries.values())
    spec = _spec(cands)
    active = b._iter_active_fields(spec)
    before = b._search_weighted_score_sum(cands, spec, active)
    freed = b.release_vision()
    assert freed > 0
    after = b._search_weighted_score_sum(cands, spec, active)
    assert before[0].id == after[0].id
    assert before[0].score == after[0].score  # bit-same: _mat unchanged by the release
    assert b._lean_hits > 0


def test_released_entry_storage_is_zero():
    torch.manual_seed(1)
    b = _build([_entry(f"e{i}", "A", 8, 4) for i in range(6)])
    b.release_vision()
    for e in b._entries.values():
        for f in ("vision_0", "vision_1"):
            assert e.query_keys[f].untyped_storage().nbytes() == 0
        # robot_state is kept (not released)
        assert e.query_keys["robot_state"].untyped_storage().nbytes() > 0


def test_guard_raises_on_forced_fallback():
    torch.manual_seed(2)
    ea = [_entry(f"a{i}", "A", 8, 4) for i in range(3)]
    eb = [_entry(f"b{i}", "B", 8, 4) for i in range(3)]
    b = _build(ea + eb)
    b.release_vision()
    mixed = ea[:2] + eb[:2]  # cross-bucket -> _fast_path_key None -> would read freed tensors
    with pytest.raises(ReleaseUnsafeError):
        b._compute_field_scores(torch.randn(8), mixed, "vision_0", {"type": "cosine"})


def test_release_requires_freeze():
    b = _build([_entry(f"e{i}", "A", 8, 4) for i in range(4)])
    b._is_frozen = False
    with pytest.raises(AssertionError):
        b.release_vision()


def test_release_requires_no_active_session():
    b = _build([_entry(f"e{i}", "A", 8, 4) for i in range(4)])
    b._active_search_sessions.add("sid")
    with pytest.raises(AssertionError):
        b.release_vision()
