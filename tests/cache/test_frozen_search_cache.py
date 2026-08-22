"""Frozen-search cache: exactness, reuse, and invalidation.

The serving path (write_policy=never) repeats searches against an immutable
entry set; `_filtered_candidates` caches the filter result per fingerprint and
`_candidate_matrix` caches the per-field stacked matrix per list object
(ws_search timing lab 2026-08-22: the per-query gather+stack dominated live
search cost). These tests pin the contract: cached results are bit-identical
to uncached ones, hits actually skip the rebuild, and ANY mutation invalidates.
"""

from __future__ import annotations

import gc

import torch

import openpi.cache.backends.in_memory_backend as imb
from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    QueryFilter,
    QuerySpec,
)
from openpi.cache.types import CheckpointID

FIELD_SIM = {
    "vision_0": {"type": "cosine"},
    "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 1.0}},
}


def _entry(eid: str, seed: int, task_key: str) -> CacheEntry:
    g = torch.Generator().manual_seed(seed)
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={
            "vision_0": torch.randn(8, generator=g),
            "robot_state": torch.randn(4, generator=g),
        },
        payload=CachePayload(action_chunk=torch.zeros(10, 32), task_key=task_key),
    )


def _backend(n: int = 12) -> InMemoryBackend:
    backend = InMemoryBackend({"vision_0": 8, "robot_state": 4})
    for i in range(n):
        backend.insert(_entry(f"e{i}", seed=i, task_key="taskA" if i % 2 == 0 else "taskB"))
    return backend


def _spec(seed: int = 100, task_key: str | None = "taskA") -> QuerySpec:
    g = torch.Generator().manual_seed(seed)
    return QuerySpec(
        query_keys={
            "vision_0": torch.randn(8, generator=g),
            "robot_state": torch.randn(4, generator=g),
        },
        top_k=5,
        checkpoint_id=CheckpointID.CP1,
        fusion_method="weighted_score_sum",
        fusion_weights={"vision_0": 0.5, "robot_state": 0.5},
        field_similarity=FIELD_SIM,
        score_normalization={"type": "none"},
        filters=None if task_key is None else QueryFilter(task_key=task_key),
    )


def _as_tuples(results):
    return [(r.id, round(r.score, 6)) for r in results]


class TestExactness:
    def test_repeat_search_identical_and_skips_rebuild(self, monkeypatch):
        backend = _backend()
        first = _as_tuples(backend.search(_spec()))
        assert first, "fixture must produce hits"
        assert len(backend._filtered_cache) == 1
        assert backend._field_matrix_cache

        stacks = []
        real_stack = imb.torch.stack
        monkeypatch.setattr(imb.torch, "stack", lambda *a, **k: stacks.append(1) or real_stack(*a, **k))
        again = _as_tuples(backend.search(_spec()))
        assert again == first
        assert stacks == [], "warm search must not rebuild any matrix"

    def test_cached_equals_uncached_across_paths(self):
        backend = _backend()
        warm = _as_tuples(backend.search(_spec()))
        cold_backend = _backend()  # identical content, cold caches
        assert _as_tuples(cold_backend.search(_spec())) == warm

    def test_rrf_path_cached_and_equal(self):
        backend = _backend()
        spec = _spec()
        spec.fusion_method = "weighted_rrf"
        first = _as_tuples(backend.search(spec))
        assert _as_tuples(backend.search(spec)) == first

    def test_distinct_filters_do_not_collide(self):
        backend = _backend()
        a = {r.id for r in backend.search(_spec(task_key="taskA"))}
        b = {r.id for r in backend.search(_spec(task_key="taskB"))}
        assert a and b and a.isdisjoint(b)
        assert len(backend._filtered_cache) == 2


class TestInvalidation:
    def test_insert_invalidates(self):
        backend = _backend()
        spec = _spec()
        backend.search(spec)
        # A new entry that exactly matches the query must surface post-insert.
        winner = CacheEntry(
            id="winner",
            checkpoint_id=CheckpointID.CP1,
            query_keys=dict(spec.query_keys),
            payload=CachePayload(action_chunk=torch.zeros(10, 32), task_key="taskA"),
        )
        backend.insert(winner)
        assert not backend._filtered_cache and not backend._field_matrix_cache
        assert backend.search(spec)[0].id == "winner"

    def test_delete_invalidates(self):
        backend = _backend()
        spec = _spec()
        top = backend.search(spec)[0].id
        backend.delete([top])
        assert all(r.id != top for r in backend.search(spec))


class TestAdHocLists:
    def test_fresh_subset_lists_stay_correct(self):
        backend = _backend()
        entries = list(backend._entries.values())[:4]
        q = _spec().query_keys["vision_0"]
        s1, m1 = backend._compute_field_scores(q, list(entries), "vision_0", {"type": "cosine"})
        s2, m2 = backend._compute_field_scores(q, list(entries), "vision_0", {"type": "cosine"})
        assert torch.equal(s1, s2) and torch.equal(m1, m2)

    def test_weakref_identity_guard_rejects_foreign_list(self):
        backend = _backend()
        entries = list(backend._entries.values())
        current = InMemoryBackend._CandidateList(entries[:6])
        decoy = InMemoryBackend._CandidateList(entries[6:12])
        # Poison the cache slot for `current`'s id with a ref to the decoy:
        # the guard must treat it as a miss and recompute correctly.
        import weakref

        backend._field_matrix_cache[("vision_0", id(current))] = (
            weakref.ref(decoy), torch.tensor([0]), torch.zeros(1, 8), torch.zeros(6),
            torch.zeros(1),
        )
        q = _spec().query_keys["vision_0"]
        scores, mask = backend._compute_field_scores(q, current, "vision_0", {"type": "cosine"})
        assert int(mask.sum()) == 6  # every entry has the field: real recompute
        ref_backend = _backend()
        ref_scores, _ = ref_backend._compute_field_scores(
            q, list(ref_backend._entries.values())[:6], "vision_0", {"type": "cosine"},
        )
        assert torch.allclose(scores, ref_scores)
        gc.collect()


class TestManualCosineEquivalence:
    def test_matches_torch_cosine_similarity(self):
        # The manual dot/norm path must reproduce F.cosine_similarity within
        # float tolerance (same formula, different summation order).
        import torch.nn.functional as F

        backend = _backend()
        q = _spec().query_keys["vision_0"]
        candidates = backend._filtered_candidates(_spec())
        scores, mask = backend._compute_field_scores(q, candidates, "vision_0", {"type": "cosine"})
        mat = torch.stack([e.query_keys["vision_0"] for e in candidates]).float()
        expected = F.cosine_similarity(q.float().unsqueeze(0), mat)
        assert torch.allclose(scores[mask.bool()], expected[mask.bool()], atol=1e-6)
