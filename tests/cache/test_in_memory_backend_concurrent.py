"""Multi-session concurrent + mutation-contract tests for InMemoryBackend (plan §7.7 + §7.8).

Plan §7.7 requires an N=8 multi-thread search loop repeated 100 times,
verifying:
  - per-session results equal the single-thread baseline,
  - cache buckets do not cross-pollute,
  - no exception is raised across the loop.

Plan §7.8 requires the mutation contract to hold across edges:
  - (b) sid registered but no search yet → guard already active,
  - (c) active session with cached data → contents not corrupted,
  - (d) close → mutation freedom restored,
  - (e) `load_artifact` blocked while sessions are active.

The single-thread parity / unregistered-sid coverage already lives in
`test_in_memory_backend_trajectory.py`; this file targets the
load-stress and edge-case surface only, so the cost stays small.
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path

import pytest
import torch

from openpi.cache.backends.in_memory_backend import (
    SearchSessionActiveError,
    InMemoryBackend,
)
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    QuerySpec,
)
from openpi.cache.types import CheckpointID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(dim: int, index: int) -> torch.Tensor:
    v = torch.zeros(dim)
    v[index] = 1.0
    return v


def _entry(eid: str, vec: torch.Tensor, *, prev: list[str] | None = None,
           step: int | None = None, traj: str | None = None) -> CacheEntry:
    return CacheEntry(
        id=eid,
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": vec.cpu().float().contiguous()},
        payload=CachePayload(action_chunk=torch.randn(50, 32)),
        prev_ids=prev or [],
        trajectory_id=traj,
        step_idx=step,
    )


def _spec(state: torch.Tensor, history: list[dict[str, torch.Tensor]],
          weights: list[float], *, sid: str | None = None,
          qids: list[int] | None = None, top_k: int = 5) -> QuerySpec:
    return QuerySpec(
        query_keys={"robot_state": state},
        top_k=top_k,
        checkpoint_id=CheckpointID.CP1,
        trajectory_history=history,
        trajectory_weights=weights,
        search_session_id=sid,
        trajectory_query_ids=qids,
    )


def _populate_chain(backend: InMemoryBackend, n: int = 6, dim: int = 32) -> list[torch.Tensor]:
    states = [_unit(dim, i) for i in range(n)]
    for i in range(n):
        backend.insert(_entry(
            f"e:{i}", states[i],
            prev=[f"e:{i - 1}"] if i > 0 else [],
            step=i, traj="traj",
        ))
    return states


# ---------------------------------------------------------------------------
# §7.7 100-loop multi-session concurrent search
# ---------------------------------------------------------------------------


def test_multi_session_concurrent_100_loops():
    """N=8 threads, each with its own sid, run 5 search() calls concurrently,
    repeated 100 times. Asserts no intermittent failure, no cross-pollution,
    per-thread results match the single-threaded baseline.
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

    # Single-thread baseline.
    baseline: dict[str, list] = {}
    for sid in sids:
        backend.open_search_session(sid)
    for sid in sids:
        baseline[sid] = backend.search(_spec(
            states[5], history, weights, sid=sid, qids=[5, 4, 3],
        ))
    for sid in sids:
        backend.close_search_session(sid)

    failures: list[tuple[int, str, str]] = []

    for loop in range(100):
        # Fresh sessions per loop — buckets must be reset cleanly.
        for sid in sids:
            backend.open_search_session(sid)

        results: dict[str, list] = {}
        errors: list[Exception] = []

        def worker(sid):
            try:
                last = None
                for _ in range(5):
                    last = backend.search(_spec(
                        states[5], history, weights, sid=sid, qids=[5, 4, 3],
                    ))
                results[sid] = last
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(sid,)) for sid in sids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            failures.append((loop, "thread_error", repr(errors)))
            for sid in sids:
                backend.close_search_session(sid)
            break

        for sid in sids:
            base = baseline[sid]
            got = results[sid]
            if [r.id for r in got] != [r.id for r in base]:
                failures.append((loop, sid, "id_mismatch"))
                continue
            if any(abs(g.score - b.score) > 1e-6 for g, b in zip(got, base)):
                failures.append((loop, sid, "score_drift"))

        # Bucket isolation: each sid keeps its own bucket.
        for sid in sids:
            assert sid in backend._score_memo, f"loop {loop} sid {sid}: bucket missing"

        for sid in sids:
            backend.close_search_session(sid)

    assert not failures, f"intermittent failures: {failures[:5]}"


# ---------------------------------------------------------------------------
# §7.8 (b) guard active immediately after open_search_session, no search yet
# ---------------------------------------------------------------------------


def test_guard_active_before_first_search():
    backend = InMemoryBackend({"robot_state": 32})
    backend.insert(_entry("e0", _unit(32, 0)))
    backend.open_search_session("sid")
    try:
        # No search has run yet → no cache bucket — but the active set is
        # still non-empty, so guard fires.
        assert "sid" not in backend._score_memo
        with pytest.raises(SearchSessionActiveError):
            backend.insert(_entry("e0", _unit(32, 1)))  # upsert
        with pytest.raises(SearchSessionActiveError):
            backend.delete(["e0"])
        # Brand-new id is still allowed.
        backend.insert(_entry("e_new", _unit(32, 2)))
        assert "e_new" in backend._entries
    finally:
        backend.close_search_session("sid")


# ---------------------------------------------------------------------------
# §7.8 (c) active session + cached data → cache contents preserved
# ---------------------------------------------------------------------------


def test_cache_contents_unchanged_under_blocked_mutation():
    backend = InMemoryBackend({"robot_state": 32})
    states = _populate_chain(backend, n=4)

    backend.open_search_session("sid")
    history = [{"robot_state": states[3]}, {"robot_state": states[2]}]
    weights = [0.7, 0.3]
    backend.search(_spec(states[3], history, weights, sid="sid", qids=[3, 2]))
    snapshot = {
        ikey: dict(slot) for ikey, slot in backend._score_memo["sid"].items()
    }

    with pytest.raises(SearchSessionActiveError):
        backend.insert(_entry("e:0", _unit(32, 9)))  # upsert blocked
    with pytest.raises(SearchSessionActiveError):
        backend.delete(["e:0"])

    # Cache bucket untouched.
    after = backend._score_memo["sid"]
    assert set(after.keys()) == set(snapshot.keys())
    for k, slot in snapshot.items():
        assert after[k] == slot, f"slot {k} changed under blocked mutation"

    backend.close_search_session("sid")


# ---------------------------------------------------------------------------
# §7.8 (d) close → mutation flows freely again
# ---------------------------------------------------------------------------


def test_close_restores_mutation_freedom():
    backend = InMemoryBackend({"robot_state": 32})
    backend.insert(_entry("e0", _unit(32, 0)))
    backend.open_search_session("sid")
    backend.close_search_session("sid")

    # All mutations now succeed, like trunk.
    backend.insert(_entry("e0", _unit(32, 1)))      # upsert
    backend.delete(["e0"])
    assert "e0" not in backend._entries


# ---------------------------------------------------------------------------
# §7.8 (e) load_artifact blocked under active session
# ---------------------------------------------------------------------------


def test_load_artifact_blocked_under_active_session(tmp_path: Path):
    backend = InMemoryBackend({"robot_state": 32})
    artifact = tmp_path / "fake.pkl"
    artifact.write_bytes(pickle.dumps({
        "key_builder_type": "robot_state",
        "checkpoint_id": "CP1",
        "vector_dims": {"robot_state": 32},
        "entries": [],
    }))

    backend.open_search_session("sid")
    try:
        with pytest.raises(SearchSessionActiveError):
            backend.load_artifact(str(artifact))
    finally:
        backend.close_search_session("sid")

    # Once closed, load_artifact succeeds.
    backend.load_artifact(str(artifact))
