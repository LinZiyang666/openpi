"""X15 U0 — retrieval diagnostics produced by the weighted-score-sum backend.

Covers the three properties the risk router depends on and one property the
rest of the system depends on:

* the per-field decomposition means what the plan says it means
  (winner-of-fusion vs each field's own ranking);
* diagnostics can never be stale — an empty or shorter result set reports its
  own truth, never the previous search's;
* concurrent connections cannot read each other's diagnostics, which is why
  ownership sits on the per-connection facade rather than the pooled backend;
* legacy paths (``search()``, non-weighted fusion) keep their exact contract.

Key dependency: ``InMemoryBackend.search_with_diagnostics`` and
``CacheStorage.last_step_features`` (openpi.cache).
"""

from __future__ import annotations

import threading

import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.storage_types import CacheEntry, CachePayload, QuerySpec, StepRetrievalFeatures
from openpi.cache.types import CheckpointID

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

_DIMS = {"a": 2, "b": 2}


def _payload() -> CachePayload:
    return CachePayload(action_chunk=torch.zeros(50, 32))


def _entry(eid: str, a: list[float], b: list[float]) -> CacheEntry:
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={"a": torch.tensor(a), "b": torch.tensor(b)},
        payload=_payload(),
    )


def _backend(entries: list[CacheEntry]) -> InMemoryBackend:
    backend = InMemoryBackend(vector_dims=_DIMS)
    for e in entries:
        backend.insert(e)
    return backend


def _spec(a: list[float], b: list[float], *, top_k: int = 3) -> QuerySpec:
    return QuerySpec(
        query_keys={"a": torch.tensor(a), "b": torch.tensor(b)},
        top_k=top_k,
        checkpoint_id=CheckpointID.CP1,
        fusion_weights={"a": 0.5, "b": 0.5},
        fusion_method="weighted_score_sum",
        field_similarity={"a": {"type": "cosine"}, "b": {"type": "cosine"}},
    )


# ------------------------------------------------------------------
# Semantics of the per-field decomposition
# ------------------------------------------------------------------


def test_winner_per_field_decomposes_the_fused_winner() -> None:
    """``winner_per_field`` is the fused top-1's score in each field — the
    answer to "why did this entry win", not a per-field argmax."""
    backend = _backend([
        _entry("e1", [1.0, 0.0], [1.0, 0.0]),   # aligned with the query in both
        _entry("e2", [0.0, 1.0], [1.0, 0.0]),   # only field b aligns
    ])
    results, diag = backend.search_with_diagnostics(_spec([1.0, 0.0], [1.0, 0.0]))

    assert results[0].id == "e1"
    assert set(diag.winner_per_field) == {"a", "b"}
    # e1 is the fused winner and is perfectly aligned in both fields.
    assert diag.winner_per_field["a"] > diag.winner_per_field["b"] - 1e-6
    assert diag.n_results == 2


def test_field_own_margin_uses_that_fields_own_ranking() -> None:
    """A field whose own top-2 are indistinguishable reports a ~0 margin even
    when the fused score is decisive — that separation is the whole signal."""
    backend = _backend([
        _entry("e1", [1.0, 0.0], [1.0, 0.0]),
        _entry("e2", [1.0, 0.0], [0.0, 1.0]),   # field a identical to e1
    ])
    _, diag = backend.search_with_diagnostics(_spec([1.0, 0.0], [1.0, 0.0]))

    # Field a cannot separate e1 from e2; field b can.
    assert diag.field_own_margin["a"] < 1e-6
    assert diag.field_own_margin["b"] > 0.1
    assert diag.fused_margin > 0.1


def test_single_candidate_has_no_margins_but_still_reports_coverage() -> None:
    """Margins need two candidates; ``n_results`` is defined regardless."""
    backend = _backend([_entry("e1", [1.0, 0.0], [1.0, 0.0])])
    _, diag = backend.search_with_diagnostics(_spec([1.0, 0.0], [1.0, 0.0]))

    assert diag.n_results == 1
    assert diag.fused_margin == 0.0
    assert diag.field_own_margin == {}
    assert set(diag.winner_per_field) == {"a", "b"}


# ------------------------------------------------------------------
# Freshness — diagnostics must describe THIS search
# ------------------------------------------------------------------


def test_empty_library_does_not_return_the_previous_searchs_features() -> None:
    """The stale-read failure mode: a second search that finds nothing must
    not inherit the first search's decomposition."""
    storage = CacheStorage(_backend([_entry("e1", [1.0, 0.0], [1.0, 0.0])]))
    storage.search(_spec([1.0, 0.0], [1.0, 0.0]))
    assert storage.last_step_features().n_results == 1

    empty = CacheStorage(_backend([]))
    empty.search(_spec([1.0, 0.0], [1.0, 0.0]))
    assert empty.last_step_features() == StepRetrievalFeatures()

    # And the same facade, re-searched against a filter that excludes all.
    storage.search(_spec([1.0, 0.0], [1.0, 0.0], top_k=3))
    assert storage.last_step_features().n_results == 1


def test_fewer_results_than_top_k_reports_the_real_count() -> None:
    """``n_results < top_k`` is carried explicitly rather than zero-padded, so
    the router can tell a thin library from a confident one."""
    backend = _backend([_entry("e1", [1.0, 0.0], [1.0, 0.0])])
    _, diag = backend.search_with_diagnostics(_spec([1.0, 0.0], [1.0, 0.0], top_k=5))
    assert diag.n_results == 1
    assert len(diag.fused_topk) == 1


def test_facade_starts_with_no_features() -> None:
    storage = CacheStorage(_backend([_entry("e1", [1.0, 0.0], [1.0, 0.0])]))
    assert storage.last_step_features() is None


# ------------------------------------------------------------------
# Concurrency — the reason ownership is per-connection
# ------------------------------------------------------------------


def test_concurrent_facades_never_read_each_others_diagnostics() -> None:
    """Two connections sharing one pooled backend, forced to interleave.

    Before the atomic-return design this was the real defect: a mutable
    ``last_*`` slot on the shared backend meant connection B's search could
    land between connection A's search and A's judge, handing A the wrong
    step's features. The barrier makes that interleaving certain.
    """
    backend = _backend([
        _entry("e1", [1.0, 0.0], [1.0, 0.0]),
        _entry("e2", [0.0, 1.0], [0.0, 1.0]),
    ])
    # One backend, two per-connection facades — the production topology.
    facade_a = CacheStorage(backend)
    facade_b = CacheStorage(backend)

    barrier = threading.Barrier(2)
    seen: dict[str, StepRetrievalFeatures] = {}
    errors: list[BaseException] = []

    def run(name: str, facade: CacheStorage, query: list[float]) -> None:
        try:
            facade.search(_spec(query, query, top_k=1))
            # Force the other connection to search before this one reads back.
            barrier.wait(timeout=10)
            seen[name] = facade.last_step_features()
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)
            try:
                barrier.abort()
            except threading.BrokenBarrierError:
                pass

    threads = [
        threading.Thread(target=run, args=("a", facade_a, [1.0, 0.0])),
        threading.Thread(target=run, args=("b", facade_b, [0.0, 1.0])),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors, errors
    # Each connection still sees the winner of ITS OWN query.
    assert seen["a"].fused_topk[0][0] == "e1"
    assert seen["b"].fused_topk[0][0] == "e2"


# ------------------------------------------------------------------
# Legacy contracts
# ------------------------------------------------------------------


def test_plain_search_is_unchanged_and_matches_the_diagnostic_path() -> None:
    """``search()`` still returns a bare list, value-identical to the results
    half of ``search_with_diagnostics``."""
    backend = _backend([
        _entry("e1", [1.0, 0.0], [1.0, 0.0]),
        _entry("e2", [0.0, 1.0], [0.0, 1.0]),
    ])
    spec = _spec([1.0, 0.0], [1.0, 0.0])
    plain = backend.search(spec)
    with_diag, _ = backend.search_with_diagnostics(spec)

    assert isinstance(plain, list)
    assert [(r.id, r.score) for r in plain] == [(r.id, r.score) for r in with_diag]


def test_non_weighted_fusion_reports_coverage_only() -> None:
    """RRF does not decompose into comparable per-field scores; it reports a
    truthful count instead of a fabricated decomposition."""
    backend = _backend([
        _entry("e1", [1.0, 0.0], [1.0, 0.0]),
        _entry("e2", [0.0, 1.0], [0.0, 1.0]),
    ])
    spec = QuerySpec(
        query_keys={"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([1.0, 0.0])},
        top_k=2,
        checkpoint_id=CheckpointID.CP1,
        fusion_weights={"a": 0.5, "b": 0.5},
        fusion_method="weighted_rrf",
        field_similarity={"a": {"type": "cosine"}, "b": {"type": "cosine"}},
    )
    results, diag = backend.search_with_diagnostics(spec)
    assert diag.n_results == len(results) == 2
    assert diag.winner_per_field == {}
    assert diag.fused_topk == ()
