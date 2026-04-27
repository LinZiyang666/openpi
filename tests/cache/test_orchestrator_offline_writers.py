"""Tests for Orchestrator B2 OfflineWriter wiring.

Covers:
- `__init__` accepts `offline_writers` and `library_stats` (default empty)
- `_build_entry_chain` runs each writer over the freshly built chain and
  merges per-entry factor dicts into payload.factors
- empty writers / None library_stats → fast path, payload.factors stays None
- writer receives the actual chain entries (sees prev_ids / next_ids)
- multiple writers' factor keys merge without overwriting each other
"""

from __future__ import annotations

import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import ThresholdJudge
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.components.write_policy import AlwaysWritePolicy
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import EpisodeRecord, StepRecord
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID

from .conftest import TestStorageSearchStrategy, _wrap_per_checkpoint


# ------------------------------------------------------------------
# Mock OfflineWriter
# ------------------------------------------------------------------


class _RecordingWriter:
    """OfflineWriter stub: records the entries it sees and emits a fixed
    factor dict per entry under the writer's prefix."""

    def __init__(self, prefix: str):
        self._prefix = prefix
        self.last_entries = None
        self.last_library_stats = None
        self.call_count = 0

    def required_payload_fields(self) -> set[str]:
        return set()

    def compute_for_episode(self, entries, library_stats):
        self.last_entries = list(entries)
        self.last_library_stats = library_stats
        self.call_count += 1
        return [
            {f"{self._prefix}_jerk": float(i), f"{self._prefix}_dir": float(-i)}
            for i in range(len(entries))
        ]


def _make_orch(
    *,
    offline_writers=(),
    library_stats=None,
    write_policy=None,
) -> tuple[CacheOrchestrator, InMemoryBackend]:
    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    kb = PlaceholderKeyBuilder()
    g = AlwaysSearchGate()
    j = ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)
    strat = TestStorageSearchStrategy(storage, top_k=1)
    return (
        CacheOrchestrator(
            storage, kb,
            gates=_wrap_per_checkpoint(g),
            judges=_wrap_per_checkpoint(j),
            search_strategies=_wrap_per_checkpoint(strat),
            timer=SystemTimer(enabled=False),
            write_policy=write_policy,
            offline_writers=offline_writers,
            library_stats=library_stats,
        ),
        backend,
    )


def _make_record(steps: int = 3) -> EpisodeRecord:
    return EpisodeRecord(
        steps=[
            StepRecord(
                query_keys={"robot_state": torch.zeros(32)},
                action_chunk=torch.zeros(50, 32),
            )
            for _ in range(steps)
        ],
        task_key="t",
        total_steps=steps,
        miss_by_checkpoint={CheckpointID.CP1: steps},
    )


def _zero_library_stats() -> LibraryStats:
    return LibraryStats(
        action_sigma=torch.ones(32),
        action_active_mask=torch.ones(32, dtype=torch.bool),
        state_sigma=torch.ones(32),
        state_active_mask=torch.ones(32, dtype=torch.bool),
    )


# ------------------------------------------------------------------
# __init__ accepts the new B2 params with sensible defaults
# ------------------------------------------------------------------


def test_init_defaults_no_writers_none_stats():
    orch, _ = _make_orch()
    assert orch._offline_writers == ()
    assert orch._library_stats is None


def test_init_accepts_writers_and_stats():
    w = _RecordingWriter("x")
    ls = _zero_library_stats()
    orch, _ = _make_orch(offline_writers=(w,), library_stats=ls)
    assert orch._offline_writers == (w,)
    assert orch._library_stats is ls


# ------------------------------------------------------------------
# _build_entry_chain merge loop
# ------------------------------------------------------------------


def test_build_entry_chain_runs_writer_and_merges_factors():
    w = _RecordingWriter("f1b_a")
    ls = _zero_library_stats()
    orch, _ = _make_orch(offline_writers=(w,), library_stats=ls)
    record = _make_record(steps=4)

    entries = orch._build_entry_chain(record)

    # Writer was invoked once with the chain
    assert w.call_count == 1
    assert len(w.last_entries) == 4
    assert w.last_library_stats is ls
    # Every entry now has factors written
    for i, entry in enumerate(entries):
        assert entry.payload.factors == {
            "f1b_a_jerk": float(i), "f1b_a_dir": float(-i),
        }


def test_build_entry_chain_writer_sees_chain_links():
    w = _RecordingWriter("f1b_a")
    ls = _zero_library_stats()
    orch, _ = _make_orch(offline_writers=(w,), library_stats=ls)
    record = _make_record(steps=3)

    orch._build_entry_chain(record)

    # entries are wired with prev_ids / next_ids by the time the writer
    # gets them — this matters for window-aware writers in production.
    es = w.last_entries
    assert es[0].prev_ids == []
    assert es[0].next_ids == [es[1].id]
    assert es[1].prev_ids == [es[0].id]
    assert es[1].next_ids == [es[2].id]
    assert es[2].prev_ids == [es[1].id]
    assert es[2].next_ids == []


def test_build_entry_chain_multiple_writers_merge_without_overwrite():
    w1 = _RecordingWriter("f1b_a")
    w2 = _RecordingWriter("f1b_t")
    ls = _zero_library_stats()
    orch, _ = _make_orch(offline_writers=(w1, w2), library_stats=ls)

    entries = orch._build_entry_chain(_make_record(steps=2))

    # Both writers' keys present on every entry; merging is union, not
    # overwrite (different prefixes don't collide).
    for entry in entries:
        keys = set(entry.payload.factors.keys())
        assert "f1b_a_jerk" in keys
        assert "f1b_a_dir"  in keys
        assert "f1b_t_jerk" in keys
        assert "f1b_t_dir"  in keys


# ------------------------------------------------------------------
# Fast path: no writers OR library_stats=None
# ------------------------------------------------------------------


def test_build_entry_chain_no_writers_skips_merge():
    orch, _ = _make_orch(offline_writers=(), library_stats=_zero_library_stats())
    entries = orch._build_entry_chain(_make_record(steps=2))
    for entry in entries:
        assert entry.payload.factors is None


def test_build_entry_chain_writers_but_no_library_stats_skips_merge():
    # If library_stats wasn't loaded, writers can't run safely (they rely
    # on library sigma). Fast-path bypass keeps behavior byte-identical
    # for old artifacts that lack the field.
    w = _RecordingWriter("f1b_a")
    orch, _ = _make_orch(offline_writers=(w,), library_stats=None)
    entries = orch._build_entry_chain(_make_record(steps=2))
    assert w.call_count == 0
    for entry in entries:
        assert entry.payload.factors is None
