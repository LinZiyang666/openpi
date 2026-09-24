"""Orchestrator under the twin component set (plan §4, §12-A/B).

The load-bearing property is byte-for-byte isolation: driving the orchestrator
with ``trace=True`` and a twin set must leave every real observable — the
returned ``CheckResult`` (minus ``trace``), the real counters and histories,
the real strategy's query history, the real backend session memo, the real
gate's verdict feedback and the real judge's state — identical to a run
without twins on the same recorded input sequence. The twin, in turn, must
have searched every step, fed its gate once per step, and never touched the
real session memo.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import AlwaysSearchGate, ScoreHysteresisGate
from openpi.cache.components.judge import HitType, ThresholdJudge
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.components.search_strategy import (
    WeightedRrfKnnStrategy,
    WeightedScoreSumKnnStrategy,
)
from openpi.cache.orchestrator import CacheOrchestrator, TwinSet
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.types import PI05_V1, CheckpointID
from tests.cache.conftest import insert_entry, make_stage1

DIM = 32
FIELD_SIM = {"robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 1.0}}}
SCORE_NORM = {"type": "per_field", "fields": {"robot_state": {"method": "exp_l2", "params": {"tau": 1.0}}}}


class RecordingGate(ScoreHysteresisGate):
    """Hysteresis gate that logs every record_verdict call."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.verdicts: list[tuple] = []
        self.calls = 0

    def __call__(self, checkpoint_id, cached_data, request_context=None):
        self.calls += 1
        return super().__call__(checkpoint_id, cached_data, request_context)

    def record_verdict(self, checkpoint_id, *, hit_type, cp1_score, winner_id, start_t, searched):
        self.verdicts.append((hit_type, cp1_score, winner_id, start_t, searched))
        return super().record_verdict(
            checkpoint_id, hit_type=hit_type, cp1_score=cp1_score, winner_id=winner_id,
            start_t=start_t, searched=searched,
        )


def _library(backend: InMemoryBackend, storage: CacheStorage, n: int = 6):
    g = torch.Generator().manual_seed(0)
    prev = None
    for i in range(n):
        state = torch.randn(1, DIM, generator=g)
        inter = {t: torch.randn(50, DIM, generator=g) for t in (0.7, 0.5, 0.3)}
        payload = CachePayload(
            action_chunk=torch.randn(50, DIM, generator=g),
            intermediates=inter,
            denoising_num_steps=10,
            schedule_id=PI05_V1.schedule_id,
        )
        entry = insert_entry(
            storage, CheckpointID.CP1, state, payload, entry_id=f"e{i}",
            step_idx=i, prev_ids=[prev] if prev else None, trajectory_id="traj",
        )
        prev = entry.id
    backend.freeze()


def _components(storage, *, strategy_kind: str, gate_kind: str, warm=True):
    kb = PlaceholderKeyBuilder()
    if gate_kind == "hysteresis":
        gate = RecordingGate(theta_low=0.5, theta_high=0.9, j=1, probe_interval=2)
    else:
        gate = AlwaysSearchGate()
    warm_tiers = [{"threshold": 0.3, "start_t": 0.5}] if warm else None
    judge = ThresholdJudge(cp1_threshold=0.9, cp3_threshold=0.95, warm_tiers=warm_tiers)
    if strategy_kind == "wss":
        strategy = WeightedScoreSumKnnStrategy(
            storage, top_k=3, fusion_weights={"robot_state": 1.0},
            field_similarity=FIELD_SIM, score_normalization=SCORE_NORM,
        )
    elif strategy_kind == "wss_depth2":
        strategy = WeightedScoreSumKnnStrategy(
            storage, top_k=3, fusion_weights={"robot_state": 1.0},
            field_similarity=FIELD_SIM, score_normalization=SCORE_NORM,
            trajectory_depth=2, trajectory_weights=[0.7, 0.3],
        )
    else:
        strategy = WeightedRrfKnnStrategy(
            storage, top_k=3, fusion_weights={"robot_state": 1.0}, field_similarity=FIELD_SIM,
        )
    return kb, gate, judge, strategy


def _orchestrator(shared_storage, *, twins: bool, strategy_kind="wss", gate_kind="hysteresis"):
    real_storage = shared_storage.per_connection_facade()
    kb, gate, judge, strategy = _components(real_storage, strategy_kind=strategy_kind, gate_kind=gate_kind)
    twin_set = None
    if twins:
        twin_storage = shared_storage.per_connection_facade()
        tkb, tgate, tjudge, tstrategy = _components(
            twin_storage, strategy_kind=strategy_kind, gate_kind=gate_kind
        )
        twin_set = TwinSet(
            key_builder=tkb, gates={CheckpointID.CP1: tgate}, judges={CheckpointID.CP1: tjudge},
            strategies={CheckpointID.CP1: tstrategy}, storage=twin_storage,
            timer=SystemTimer(enabled=False),
        )
    orch = CacheOrchestrator(
        real_storage, kb,
        gates={CheckpointID.CP1: gate}, judges={CheckpointID.CP1: judge},
        search_strategies={CheckpointID.CP1: strategy},
        timer=SystemTimer(enabled=False), trace_twins=twin_set,
    )
    return orch, (kb, gate, judge, strategy)


def _inputs(n=50, seed=1):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, DIM, generator=g)
    out = []
    for i in range(n):
        # Alternate close-to-library and far-away states so the hysteresis
        # gate actually skips on some steps.
        if i % 3 == 0:
            out.append(base + 0.01 * torch.randn(1, DIM, generator=g))
        else:
            out.append(5.0 * torch.randn(1, DIM, generator=g))
    return out


def _drive(orch, inputs, *, trace: bool):
    results = []
    orch.on_task_begin()
    orch.on_episode_start(task_key="", episode_id="1")
    for x in inputs:
        r = orch.check(CheckpointID.CP1, trace=trace, fetch_top1=True, stage1=make_stage1(x))
        orch.broadcast_action(torch.zeros(50, DIM))
        orch.clear()
        results.append(r)
    orch.on_episode_end()
    return results


def _strip(r):
    return dataclasses.replace(r, trace=None, query_keys=None)


def _memo_snapshot(backend: InMemoryBackend, sids):
    memo = getattr(backend, "_score_memo", {})
    return {sid: {k: dict(v) for k, v in memo.get(sid, {}).items()} for sid in sids}


@pytest.mark.parametrize("strategy_kind", ["wss", "wss_depth2", "rrf"])
def test_real_path_is_isolated_from_twins(strategy_kind):
    backend = InMemoryBackend({"robot_state": DIM})
    shared = CacheStorage(backend)
    _library(backend, shared)
    inputs = _inputs()

    orch_a, (kb_a, gate_a, _ja, strat_a) = _orchestrator(shared, twins=False, strategy_kind=strategy_kind)
    orch_b, (kb_b, gate_b, _jb, strat_b) = _orchestrator(shared, twins=True, strategy_kind=strategy_kind)

    # Session ids are minted per strategy instance; capture both real ids to
    # compare memo contents after mapping.
    res_a = _drive(orch_a, inputs, trace=False)
    res_b = _drive(orch_b, inputs, trace=True)

    assert [_strip(r) for r in res_a] == [_strip(r) for r in res_b]
    for ra, rb in zip(res_a, res_b):
        for k in ra.query_keys:
            assert torch.equal(ra.query_keys[k], rb.query_keys[k])
    assert gate_a.verdicts == gate_b.verdicts
    assert gate_a.calls == gate_b.calls == len(inputs)
    assert orch_a._step_counter == orch_b._step_counter
    assert orch_a._miss_by_checkpoint == orch_b._miss_by_checkpoint
    assert len(strat_a._query_history) == len(strat_b._query_history)
    assert strat_a._query_id_counter == strat_b._query_id_counter

    # Twin observations: searched every step, gate decision recorded, own feedback.
    skipped = [r for r in res_b if not r.searched]
    assert skipped, "fixture must exercise gate skips"
    for r in res_b:
        assert r.trace is not None
        assert r.trace.gate_real_should_search == r.searched
        assert r.trace.twin_results is not None
        assert r.trace.top1_entry_id == r.trace.twin_results[0].id
        assert r.trace.top1_payload is not None
        if strategy_kind == "wss":
            assert r.trace.twin_per_field is not None
            assert r.trace.twin_per_field.scores.shape == (len(r.trace.twin_results), 1)
            assert r.trace.twin_per_field.kind_by_field == {"robot_state": "normalized_layer1"}
            assert r.trace.twin_per_field.current_step_wss is not None
        elif strategy_kind == "rrf":
            assert r.trace.twin_per_field.kind_by_field == {"robot_state": "raw_neg_l2"}
            assert r.trace.twin_per_field.current_step_wss is None
        else:
            assert r.trace.twin_chain_scores is not None
    twin_gate = orch_b._twins.gates[CheckpointID.CP1]
    assert len(twin_gate.verdicts) == len(inputs)
    assert all(v[-1] is True for v in twin_gate.verdicts)  # searched=True every step


def test_twin_never_touches_real_session_memo():
    backend = InMemoryBackend({"robot_state": DIM})
    shared = CacheStorage(backend)
    _library(backend, shared)
    inputs = _inputs(n=12)
    orch_a, (_, _, _, strat_a) = _orchestrator(shared, twins=False, strategy_kind="wss_depth2")
    orch_b, (_, _, _, strat_b) = _orchestrator(shared, twins=True, strategy_kind="wss_depth2")
    orch_a.on_task_begin(); orch_a.on_episode_start(task_key="", episode_id="1")
    orch_b.on_task_begin(); orch_b.on_episode_start(task_key="", episode_id="1")
    sid_a = strat_a.get_search_session_id()
    sid_b = strat_b.get_search_session_id()
    sid_twin = orch_b._twins.strategies[CheckpointID.CP1].get_search_session_id()
    assert sid_b != sid_twin
    for x in inputs:
        orch_a.check(CheckpointID.CP1, stage1=make_stage1(x)); orch_a.clear()
        orch_b.check(CheckpointID.CP1, trace=True, stage1=make_stage1(x)); orch_b.clear()
    memo_a = _memo_snapshot(backend, [sid_a])[sid_a]
    memo_b = _memo_snapshot(backend, [sid_b])[sid_b]
    memo_t = _memo_snapshot(backend, [sid_twin])[sid_twin]
    # Same keys and same per-entry scores under the real session in both runs
    # (memo keys carry the strategy-minted query ids, identical across runs).
    assert memo_a.keys() == memo_b.keys()
    for k in memo_a:
        assert memo_a[k] == memo_b[k]
    # The twin searched on every step (the real gate skipped on some), so its
    # memo covers at least as many queries as the real one.
    assert len(memo_t) >= len(memo_b)
    orch_a.on_episode_end(); orch_b.on_episode_end()
    assert not backend._active_search_sessions


def test_trace_check_runs_twin_only():
    backend = InMemoryBackend({"robot_state": DIM})
    shared = CacheStorage(backend)
    _library(backend, shared)
    orch, (kb, gate, judge, strategy) = _orchestrator(shared, twins=True, gate_kind="always")
    orch.on_task_begin(); orch.on_episode_start(task_key="", episode_id="1")
    x = _inputs(n=1)[0]
    ct = orch.trace_check(CheckpointID.CP1, fetch_top1=False, stage1=make_stage1(x))
    assert ct is not None and ct.top1_payload is None and ct.twin_results
    # Real set untouched: no step counted, no query recorded, no key built.
    assert orch._step_counter == 0
    assert len(strategy._query_history) == 0
    assert orch._twins.state.step_counter == 1
    orch.clear()
    # And without twins the entry point is inert.
    orch2, _ = _orchestrator(shared, twins=False, gate_kind="always")
    assert orch2.trace_check(CheckpointID.CP1, stage1=make_stage1(x)) is None


def test_twin_warm_validation_failure_downgrades_only_the_twin():
    backend = InMemoryBackend({"robot_state": DIM})
    shared = CacheStorage(backend)
    # Library WITHOUT the 0.5 snapshot the warm tier asks for.
    g = torch.Generator().manual_seed(0)
    # PlaceholderKeyBuilder L2-normalizes the state; store the normalized key
    # so the query distance below is controlled.
    state = torch.nn.functional.normalize(torch.randn(1, DIM, generator=g), dim=1)
    payload = CachePayload(
        action_chunk=torch.randn(50, DIM, generator=g),
        intermediates={0.7: torch.randn(50, DIM, generator=g)},
        denoising_num_steps=10, schedule_id=PI05_V1.schedule_id,
    )
    insert_entry(shared, CheckpointID.CP1, state, payload, entry_id="only")
    backend.freeze()
    orch, _ = _orchestrator(shared, twins=True, gate_kind="always")
    orch.on_task_begin(); orch.on_episode_start(task_key="", episode_id="1")
    # A query scoring in the warm band (exp(-0.5) ~ 0.61 under exp_l2 tau=1):
    # the REAL path must raise exactly as HEAD.
    delta = torch.randn(1, DIM, generator=g)
    delta = delta - (delta * state).sum() * state  # orthogonal to the stored key
    q = state + 0.55 * delta / delta.norm()  # after re-normalization d ~ 0.48
    with pytest.raises(ValueError):
        orch.check(CheckpointID.CP1, trace=True, stage1=make_stage1(q))
    orch.clear()
    # Drive the twin alone: proposed WARM_START becomes an effective MISS with a reason.
    ct = orch.trace_check(CheckpointID.CP1, stage1=make_stage1(q))
    assert ct.twin_proposed_verdict.hit_type == HitType.WARM_START
    assert ct.twin_verdict.hit_type == HitType.MISS
    assert ct.twin_validation_error is not None
    # The twin's own gate/judge feedback ran once with the effective MISS.
    assert orch._twins.state.miss_by_checkpoint[CheckpointID.CP1] == 1
