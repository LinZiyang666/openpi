"""ThresholdJudge on CP2, the task_scoped filter knob and the frozen step-counter table."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.cp2_vlm_key_builder import CP2VlmTernaryKeyBuilder
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import ThresholdJudge
from openpi.cache.components.key_builder import PlaceholderKeyBuilder
from openpi.cache.components.search_strategy import (
    SearchContext,
    WeightedScoreSumKnnStrategy,
    _build_step_filters,
)
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import SearchResultLite
from openpi.cache.timing import SystemTimer
from openpi.cache.components.judge import HitType
from openpi.cache.types import CheckpointID


# ------------------------------------------------------------------
# Judge
# ------------------------------------------------------------------


def _res(score):
    return [SearchResultLite(id="w", score=score, checkpoint_id=CheckpointID.CP2)]


def test_threshold_judge_cp2_three_states_and_n1_contract():
    n0 = ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95, cp2_threshold=0.925)
    assert n0(_res(0.95), CheckpointID.CP2, {}).hit_type is HitType.FULL_HIT
    assert n0(_res(0.90), CheckpointID.CP2, {}).hit_type is HitType.MISS
    assert n0([], CheckpointID.CP2, {}).hit_type is HitType.MISS
    # N_hit=1 arm: FULL cut above the score range, one warm tier at start_t 0.1.
    n1 = ThresholdJudge(cp1_threshold=1.5, cp3_threshold=0.95, cp2_threshold=1.5,
                        warm_tiers=[{"threshold": 0.925, "start_t": 0.1}])
    r = n1(_res(0.999), CheckpointID.CP2, {})
    assert r.hit_type is HitType.WARM_START and r.start_t == 0.1
    assert n1(_res(0.9), CheckpointID.CP2, {}).hit_type is HitType.MISS
    # Warm tiers never apply to CP3; CP2 defaults to the CP1 cut when not given.
    assert n1(_res(0.999), CheckpointID.CP3, {}).hit_type is HitType.FULL_HIT
    dflt = ThresholdJudge(cp1_threshold=0.5)
    assert dflt(_res(0.6), CheckpointID.CP2, {}).hit_type is HitType.FULL_HIT


# ------------------------------------------------------------------
# task_scoped
# ------------------------------------------------------------------


def _ctx(task_key="t", step=3):
    return SearchContext(query_keys={}, checkpoint_id=CheckpointID.CP2, current_step=step, task_key=task_key)


def test_build_step_filters_task_scoped_six_cells():
    for sf in ("all", "exact", "window"):
        scoped = _build_step_filters(sf, 2, _ctx(), task_scoped=True)
        legacy = _build_step_filters(sf, 2, _ctx())
        assert (scoped is None) == (legacy is None)
        if scoped is not None:
            assert scoped.task_key == legacy.task_key == "t"
            assert scoped.step_range == legacy.step_range
        wide = _build_step_filters(sf, 2, _ctx(), task_scoped=False)
        if sf == "all":
            assert wide is None
        else:
            assert wide.task_key is None
            assert wide.step_range == legacy.step_range
    # No task key at all: identical with or without scoping.
    assert _build_step_filters("all", 2, _ctx(task_key=None), task_scoped=True) is None


class _SpecCapture:
    def __init__(self):
        self.specs = []

    def search(self, spec):
        self.specs.append(spec)
        return []

    def open_search_session(self, *a, **k):  # trajectory mixin hooks, unused here
        return None


def test_weighted_score_sum_strategy_honours_task_scoped():
    for scoped, expect in ((True, "t"), (False, None)):
        cap = _SpecCapture()
        strat = WeightedScoreSumKnnStrategy(cap, top_k=1, step_filter="window", step_window=1,
                                            fusion_weights={"vlm_out": 1.0}, task_scoped=scoped)
        strat.search(_ctx())
        f = cap.specs[-1].filters
        assert f is not None and f.task_key == expect and f.step_range == (2, 4)


# ------------------------------------------------------------------
# Step counter table (plan §3.3)
# ------------------------------------------------------------------


class _RecordingStrategy:
    def __init__(self):
        self.steps: list[int] = []

    def search(self, ctx):
        self.steps.append(ctx.current_step)
        return []


def _orch(cps, key_builder, dims):
    storage = CacheStorage(InMemoryBackend(dims))
    strategies = {cp: _RecordingStrategy() for cp in cps}
    orch = CacheOrchestrator(
        storage, key_builder,
        gates={cp: AlwaysSearchGate() for cp in cps},
        judges={cp: ThresholdJudge() for cp in cps},
        search_strategies=strategies,
        timer=SystemTimer(enabled=False),
    )
    orch.on_task_begin()
    return orch, strategies


def _stage1():
    return SimpleNamespace(state=torch.randn(1, 32))


def _stage2():
    return SimpleNamespace(prefix_out=torch.randn(1, 4, 16))


def test_step_counter_table_is_frozen_for_all_configs():
    """cycle-1 / cycle-2 coordinates seen by each checkpoint's search (plan §3.3)."""
    # CP1-only
    orch, s = _orch([CheckpointID.CP1], PlaceholderKeyBuilder(), {"robot_state": 32})
    for _ in range(2):
        orch.check(CheckpointID.CP1, stage1=_stage1())
    assert s[CheckpointID.CP1].steps == [0, 1]
    # CP1 + CP3
    orch, s = _orch([CheckpointID.CP1, CheckpointID.CP3], PlaceholderKeyBuilder(), {"robot_state": 32})
    for _ in range(2):
        orch.check(CheckpointID.CP1, stage1=_stage1())
        orch.check(CheckpointID.CP3, stage1=_stage1(), stage3=SimpleNamespace(action_chunk=torch.zeros(1, 50, 32)))
    assert s[CheckpointID.CP1].steps == [0, 1] and s[CheckpointID.CP3].steps == [1, 2]
    # CP3-only: the interceptor still calls the unconfigured CP1 first.
    orch, s = _orch([CheckpointID.CP3], PlaceholderKeyBuilder(), {"robot_state": 32})
    for _ in range(2):
        r = orch.check(CheckpointID.CP1, stage1=_stage1())
        assert r.hit_type is HitType.MISS
        orch.check(CheckpointID.CP3, stage1=_stage1(), stage3=SimpleNamespace(action_chunk=torch.zeros(1, 50, 32)))
    assert s[CheckpointID.CP3].steps == [1, 2]
    # CP2-only: owns the counter exactly like CP1-only.
    kb = CP2VlmTernaryKeyBuilder(seed=1, d=8, p=0.25, input_dim=64)
    orch, s = _orch([CheckpointID.CP2], kb, {"vlm_out": 8})
    assert orch.has_checkpoint(CheckpointID.CP2) and not orch.has_checkpoint(CheckpointID.CP1)
    for _ in range(2):
        r = orch.check(CheckpointID.CP2, stage2=_stage2())
        assert r.hit_type is HitType.MISS and set(r.query_keys) == {"vlm_out"}
    assert s[CheckpointID.CP2].steps == [0, 1]
    # An unconfigured CP3 probe on the CP2-only orchestrator neither searches nor counts.
    r = orch.check(CheckpointID.CP3, stage1=_stage1(), stage3=SimpleNamespace(action_chunk=torch.zeros(1, 50, 32)))
    assert r.hit_type is HitType.MISS
    orch.check(CheckpointID.CP2, stage2=_stage2())
    assert s[CheckpointID.CP2].steps == [0, 1, 2]
