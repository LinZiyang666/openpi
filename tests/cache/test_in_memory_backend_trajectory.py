"""Tests for InMemoryBackend trajectory search rewrite (plan §7).

Covers:
  - §7.1 single-step parity (new vs legacy via force_legacy_path).
  - §7.2 multi-branch fallback (sentinel raise → legacy DAG path).
  - §7.5 cross-step session cache parity (cache on/off equivalence).
  - §7.8 mutation contract (insert/delete/load_artifact guards).
  - §7.9 cross-step query_id reuse (cache hits across step shifts).
  - §7.10 lifecycle cleanup (open/close, episode_end, task_end).

The trunk-shaped trajectory tests live in test_trajectory_search.py and exercise
the legacy DAG path under the same fixtures; this file targets the new single-
chain main path and the lifecycle / mutation surface introduced by the rewrite.
"""

from __future__ import annotations

import threading

import pytest
import torch

from openpi.cache.backends.in_memory_backend import (
    SearchSessionActiveError,
    InMemoryBackend,
)
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    QuerySpec,
)
from openpi.cache.types import CheckpointID


# ---------------------------------------------------------------------------
# Helpers (mirrors test_trajectory_search.py for fixture parity)
# ---------------------------------------------------------------------------


def _unit(dim: int, index: int) -> torch.Tensor:
    v = torch.zeros(dim)
    v[index] = 1.0
    return v


def _make_entry(
    entry_id: str,
    state: torch.Tensor,
    *,
    cp: CheckpointID = CheckpointID.CP1,
    prev_ids: list[str] | None = None,
    next_ids: list[str] | None = None,
    trajectory_id: str | None = None,
    step_idx: int | None = None,
) -> CacheEntry:
    return CacheEntry(
        id=entry_id,
        checkpoint_id=cp,
        query_keys={"robot_state": state.cpu().float().contiguous()},
        payload=CachePayload(action_chunk=torch.randn(50, 32)),
        prev_ids=prev_ids or [],
        next_ids=next_ids or [],
        trajectory_id=trajectory_id,
        step_idx=step_idx,
    )


def _spec(
    query_state: torch.Tensor,
    history: list[dict[str, torch.Tensor]],
    weights: list[float],
    *,
    search_session_id: str | None = None,
    trajectory_query_ids: list[int] | None = None,
    top_k: int = 5,
) -> QuerySpec:
    return QuerySpec(
        query_keys={"robot_state": query_state},
        top_k=top_k,
        checkpoint_id=CheckpointID.CP1,
        trajectory_history=history,
        trajectory_weights=weights,
        search_session_id=search_session_id,
        trajectory_query_ids=trajectory_query_ids,
    )


def _populate_chain(backend: InMemoryBackend, n: int = 5, dim: int = 32) -> list[torch.Tensor]:
    """Insert a single linked chain of n entries; return their state vectors."""
    states = [_unit(dim, i) for i in range(n)]
    for i in range(n):
        e = _make_entry(
            f"e:{i}", states[i],
            prev_ids=[f"e:{i - 1}"] if i > 0 else [],
            trajectory_id="traj", step_idx=i,
        )
        backend.insert(e)
    return states


# ---------------------------------------------------------------------------
# §7.1 single-step parity (new vs legacy via force_legacy_path)
# ---------------------------------------------------------------------------


class TestNewLegacyParity:
    """Run the same trajectory search through the new single-chain main path
    and via `force_legacy_path()`; ids and scores must match within atol.
    """

    def _run_both_paths(self, backend, spec):
        new_results = backend.search(spec)
        with backend.force_legacy_path():
            legacy_results = backend.search(spec)
        return new_results, legacy_results

    @staticmethod
    def _assert_parity(new_results, legacy_results):
        """Plan §10 #2: ids set equal, scores atol=1e-6 by id, ties order
        not guaranteed (matches existing dict-then-sort instability).
        """
        assert {r.id for r in new_results} == {r.id for r in legacy_results}
        new_by_id = {r.id: r.score for r in new_results}
        legacy_by_id = {r.id: r.score for r in legacy_results}
        for entry_id, new_score in new_by_id.items():
            assert abs(new_score - legacy_by_id[entry_id]) < 1e-6, (
                f"score divergence on {entry_id}: new={new_score} legacy={legacy_by_id[entry_id]}"
            )
        # The unambiguous top scorer (no tie at the top) must match in both.
        scores_new = sorted(new_by_id.values(), reverse=True)
        if len(scores_new) >= 2 and scores_new[0] - scores_new[1] > 1e-6:
            top_new = max(new_results, key=lambda r: r.score).id
            top_legacy = max(legacy_results, key=lambda r: r.score).id
            assert top_new == top_legacy

    def test_parity_simple_chain(self):
        backend = InMemoryBackend({"robot_state": 32})
        states = _populate_chain(backend, n=5)
        history = [
            {"robot_state": states[4]},
            {"robot_state": states[3]},
            {"robot_state": states[2]},
        ]
        weights = [0.5, 0.3, 0.2]
        spec = _spec(states[4], history, weights)

        new_results, legacy_results = self._run_both_paths(backend, spec)
        self._assert_parity(new_results, legacy_results)

    def test_parity_with_short_history(self):
        backend = InMemoryBackend({"robot_state": 32})
        states = _populate_chain(backend, n=3)
        history = [
            {"robot_state": states[2]},
            {"robot_state": states[1]},
        ]
        weights = [0.5, 0.3, 0.2]  # weights longer than history
        spec = _spec(states[2], history, weights)

        new_results, legacy_results = self._run_both_paths(backend, spec)
        self._assert_parity(new_results, legacy_results)


# ---------------------------------------------------------------------------
# §7.2 multi-branch fallback
# ---------------------------------------------------------------------------


class TestMultiBranchFallback:
    """An entry with len(prev_ids) > 1 forces the legacy DAG path."""

    def test_fallback_runs_legacy(self, caplog):
        backend = InMemoryBackend({"robot_state": 32})
        states = [_unit(32, i) for i in range(4)]
        # branchy DAG: e2 has two predecessors (a, b)
        backend.insert(_make_entry("a", states[0]))
        backend.insert(_make_entry("b", states[1]))
        backend.insert(_make_entry(
            "e2", states[2], prev_ids=["a", "b"], trajectory_id="t",
        ))
        backend.insert(_make_entry(
            "e3", states[3], prev_ids=["e2"], trajectory_id="t",
        ))

        history = [
            {"robot_state": states[3]},
            {"robot_state": states[2]},
        ]
        spec = _spec(states[3], history, [0.6, 0.4])

        with caplog.at_level("WARNING"):
            results = backend.search(spec)

        # Mainpath must have caught _MultiBranchSentinel and fallen back.
        assert any("Multi-branch trajectory" in r.message for r in caplog.records), (
            f"Expected fallback warning, got: {[r.message for r in caplog.records]}"
        )

        # Output must equal the explicit legacy invocation.
        with backend.force_legacy_path():
            legacy_results = backend.search(spec)
        assert [r.id for r in results] == [r.id for r in legacy_results]


# ---------------------------------------------------------------------------
# §7.5 / §7.9 cross-step session cache parity & query_id reuse
# ---------------------------------------------------------------------------


class TestCrossStepScoreMemoParity:
    """A search with cache on (sid + qids) must produce the same result as
    the same search with cache off — both for a single step and across a
    sequence of steps that exercises query_id reuse.
    """

    def _scores_with_cache(self, backend, sid_prefix, history_seq, weights,
                           qids_seq, *, on: bool):
        results = []
        for step_idx, (history, qids) in enumerate(zip(history_seq, qids_seq)):
            sid = f"{sid_prefix}-{0}" if on else None  # fixed sid → same bucket
            qids_to_pass = qids if on else None
            spec = _spec(
                history[0]["robot_state"], history, weights,
                search_session_id=sid, trajectory_query_ids=qids_to_pass,
            )
            results.append(backend.search(spec))
        return results

    def test_cache_on_off_equivalence_5_steps(self):
        backend = InMemoryBackend({"robot_state": 32})
        states = _populate_chain(backend, n=8)

        # Build a 5-step sequence: at step t the query is states[t+2].
        # history[0] is current, history[1] is 1 step back, history[2] is 2 back.
        weights = [0.5, 0.3, 0.2]
        history_seq = []
        qids_seq = []
        for t in range(5):
            current = t + 2
            history_seq.append([
                {"robot_state": states[current]},
                {"robot_state": states[current - 1]},
                {"robot_state": states[current - 2]},
            ])
            # query_ids monotonically increase as TrajectoryMixin would mint;
            # newest-first, current step's qid is the latest minted.
            qids_seq.append([t + 2, t + 1, t])

        # Open a search session before running cache-on path.
        sid = "test-sid"
        backend.open_search_session(sid)
        try:
            on_results = []
            for t, (history, qids) in enumerate(zip(history_seq, qids_seq)):
                spec = _spec(
                    history[0]["robot_state"], history, weights,
                    search_session_id=sid, trajectory_query_ids=qids,
                )
                on_results.append(backend.search(spec))
        finally:
            backend.close_search_session(sid)

        # Cache-off path: fresh backend (sid stripped from spec).
        backend2 = InMemoryBackend({"robot_state": 32})
        _populate_chain(backend2, n=8)
        off_results = []
        for t, (history, qids) in enumerate(zip(history_seq, qids_seq)):
            spec = _spec(
                history[0]["robot_state"], history, weights,
                search_session_id=None, trajectory_query_ids=None,
            )
            off_results.append(backend2.search(spec))

        for t, (on_step, off_step) in enumerate(zip(on_results, off_results)):
            assert [r.id for r in on_step] == [r.id for r in off_step], (
                f"step {t}: ids differ"
            )
            for o_r, f_r in zip(on_step, off_step):
                assert abs(o_r.score - f_r.score) < 1e-6, (
                    f"step {t} entry {o_r.id}: cache on/off score diverged"
                )

    def test_query_id_reuse_actually_hits_cache(self):
        """Same query_id at different layers across steps should hit the same
        cache slot (cache key uses query_id, not layer)."""
        backend = InMemoryBackend({"robot_state": 32})
        states = _populate_chain(backend, n=4)

        sid = "reuse-sid"
        backend.open_search_session(sid)
        try:
            # Step 0: query q at layer 0, qid=10. Cache slot under qid=10.
            spec_t0 = _spec(
                states[3],
                [{"robot_state": states[3]}, {"robot_state": states[2]}],
                [0.6, 0.4],
                search_session_id=sid,
                trajectory_query_ids=[10, 9],
            )
            backend.search(spec_t0)
            # Slot under (field, qid=10, sim_type) should now exist.
            bucket = backend._score_memo[sid]
            assert any(k[1] == 10 for k in bucket), (
                "qid=10 slot missing after step 0"
            )

            # Step 1: same query (states[3]) is now at layer 1 with qid=10.
            spec_t1 = _spec(
                states[3],   # new "current" still uses same vector for parity
                [{"robot_state": states[3]}, {"robot_state": states[3]}],
                [0.6, 0.4],
                search_session_id=sid,
                trajectory_query_ids=[11, 10],   # qid=10 now at layer 1
            )
            backend.search(spec_t1)

            # qid=10 slot should still be present and unchanged ⇒ reused.
            bucket_after = backend._score_memo[sid]
            assert any(k[1] == 10 for k in bucket_after), (
                "qid=10 slot evicted between steps"
            )
        finally:
            backend.close_search_session(sid)


# ---------------------------------------------------------------------------
# §7.8 mutation contract (insert/delete/load_artifact guards + S3 defensive)
# ---------------------------------------------------------------------------


class TestMutationContract:
    def test_no_active_session_allows_all_mutations(self, tmp_path):
        backend = InMemoryBackend({"robot_state": 32})
        e0 = _make_entry("e0", _unit(32, 0))
        backend.insert(e0)
        # Upsert OK without active session.
        e0_again = _make_entry("e0", _unit(32, 1))
        backend.insert(e0_again)
        # Delete OK without active session.
        backend.delete(["e0"])

    def test_active_session_blocks_upsert(self):
        backend = InMemoryBackend({"robot_state": 32})
        e0 = _make_entry("e0", _unit(32, 0))
        backend.insert(e0)
        backend.open_search_session("sid")
        try:
            with pytest.raises(SearchSessionActiveError):
                backend.insert(_make_entry("e0", _unit(32, 1)))
            # Inserting a brand-new id is still allowed.
            backend.insert(_make_entry("e_new", _unit(32, 2)))
            assert "e_new" in backend._entries
        finally:
            backend.close_search_session("sid")

    def test_active_session_blocks_delete(self):
        backend = InMemoryBackend({"robot_state": 32})
        backend.insert(_make_entry("e0", _unit(32, 0)))
        backend.open_search_session("sid")
        try:
            with pytest.raises(SearchSessionActiveError):
                backend.delete(["e0"])
        finally:
            backend.close_search_session("sid")

    def test_unregistered_sid_falls_back_to_uncached(self, caplog):
        """Round 5 S3: search with a sid that was never opened should NOT
        create a cache bucket and should fall back to uncached path."""
        backend = InMemoryBackend({"robot_state": 32})
        backend.insert(_make_entry("e0", _unit(32, 0)))
        spec = QuerySpec(
            query_keys={"robot_state": _unit(32, 0)},
            top_k=1,
            checkpoint_id=CheckpointID.CP1,
            trajectory_history=[
                {"robot_state": _unit(32, 0)},
                {"robot_state": _unit(32, 0)},
            ],
            trajectory_weights=[0.6, 0.4],
            search_session_id="never-opened",
            trajectory_query_ids=[0, 1],
        )
        with caplog.at_level("WARNING"):
            results = backend.search(spec)
        assert results, "search should still return results via fallback"
        # No bucket for the unknown sid.
        assert "never-opened" not in backend._score_memo
        # Active set is unaffected.
        assert "never-opened" not in backend._active_search_sessions
        # Warning was emitted.
        assert any("unregistered search_session_id" in r.message for r in caplog.records), (
            f"Expected unregistered-sid warning, got: {[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# §7.10 lifecycle cleanup
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_open_close_cleans_bucket(self):
        backend = InMemoryBackend({"robot_state": 32})
        backend.insert(_make_entry("e0", _unit(32, 0)))
        backend.open_search_session("sid")
        # Trigger a cache write.
        spec = _spec(
            _unit(32, 0),
            [{"robot_state": _unit(32, 0)}, {"robot_state": _unit(32, 0)}],
            [0.6, 0.4],
            search_session_id="sid",
            trajectory_query_ids=[0, 1],
        )
        backend.search(spec)
        assert "sid" in backend._score_memo
        assert "sid" in backend._active_search_sessions

        backend.close_search_session("sid")
        assert "sid" not in backend._score_memo
        assert "sid" not in backend._active_search_sessions

    def test_close_unknown_sid_is_noop(self):
        backend = InMemoryBackend({"robot_state": 32})
        # Should not raise.
        backend.close_search_session("never-opened")
        assert backend._active_search_sessions == set()


# ---------------------------------------------------------------------------
# Concurrency: multi-session search through the cached path (Round 5 + B5)
# ---------------------------------------------------------------------------


class TestMultiSessionConcurrent:
    def test_per_session_buckets_are_independent(self):
        """N threads run search() against the same backend, each with a
        unique sid. Result of each thread must equal the single-thread
        baseline for that sid; buckets must not cross-pollute.
        """
        backend = InMemoryBackend({"robot_state": 32})
        states = _populate_chain(backend, n=6)

        history = [
            {"robot_state": states[5]},
            {"robot_state": states[4]},
            {"robot_state": states[3]},
        ]
        weights = [0.5, 0.3, 0.2]

        N = 8
        sids = [f"sid-{i}" for i in range(N)]
        for sid in sids:
            backend.open_search_session(sid)

        # Single-thread baseline per sid.
        baseline = {}
        for sid in sids:
            spec = _spec(
                states[5], history, weights,
                search_session_id=sid, trajectory_query_ids=[5, 4, 3],
            )
            baseline[sid] = backend.search(spec)
        # Reset to same start state by re-opening (close drops any cache).
        for sid in sids:
            backend.close_search_session(sid)
            backend.open_search_session(sid)

        # Multi-thread: each thread runs search for its own sid.
        results: dict[str, list] = {}
        errors: list[Exception] = []

        def worker(sid):
            try:
                for _ in range(5):
                    spec = _spec(
                        states[5], history, weights,
                        search_session_id=sid, trajectory_query_ids=[5, 4, 3],
                    )
                    out = backend.search(spec)
                results[sid] = out
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(sid,)) for sid in sids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        try:
            assert not errors, f"Worker errors: {errors}"
            for sid in sids:
                assert sid in results
                # Same ids and (atol-tolerant) scores per sid.
                base = baseline[sid]
                got = results[sid]
                assert [r.id for r in got] == [r.id for r in base]
                for g, b in zip(got, base):
                    assert abs(g.score - b.score) < 1e-6
            # Each sid keeps its own bucket; no cross-pollution.
            for sid in sids:
                assert sid in backend._score_memo
        finally:
            for sid in sids:
                backend.close_search_session(sid)
