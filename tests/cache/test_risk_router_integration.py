"""X15 — the risk router driven through the REAL CacheOrchestrator.check() path.

The unit tests in ``test_risk_router_judge.py`` call the judge directly, which
is exactly how a signature incompatibility hides: the Orchestrator passes
``retrieval_signals`` unconditionally, and a judge that does not accept it
raises TypeError on its very first verdict while every direct-call test still
passes. These tests therefore go through ``check()`` and the real lifecycle
broadcast, and cover the wrapper chain too.

Key dependency: ``openpi.cache.orchestrator.CacheOrchestrator``.
"""

from __future__ import annotations

import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.components.judge import HitType
from openpi.cache.components.risk_features import RiskFeatureBuilder
from openpi.cache.components.risk_router_judge import RiskRouterJudge
from openpi.cache.components.search_strategy import WeightedScoreSumKnnStrategy
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import CachePayload
from openpi.cache.timing import SystemTimer
from openpi.cache.types import CheckpointID
from tests.cache.conftest import (
    PlaceholderKeyBuilder,
    _wrap_per_checkpoint,
    insert_entry,
    make_stage1,
)


class _Risk:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def risk(self, x: torch.Tensor) -> float:
        self.calls += 1
        return self.value


def _build(risk_value: float, *, dwell: int = 1, n_entries: int = 6):
    """Real backend + storage + weighted-score-sum strategy + risk judge."""
    backend = InMemoryBackend({"robot_state": 32})
    storage = CacheStorage(backend)
    for i in range(n_entries):
        state = torch.zeros(1, 32)
        state[0, i] = 1.0
        insert_entry(
            storage, CheckpointID.CP1, state,
            CachePayload(action_chunk=torch.zeros(50, 32)),
            entry_id=f"e{i}", step_idx=i,
        )
    strategy = WeightedScoreSumKnnStrategy(
        storage,
        top_k=5,
        fusion_weights={"robot_state": 1.0},
        field_similarity={"robot_state": {"type": "cosine"}},
    )
    model = _Risk(risk_value)
    judge = RiskRouterJudge(
        feature_builder=RiskFeatureBuilder(
            task_index=0, replan_steps=5, library_replan_steps=5,
        ),
        risk_model=model,
        tau=0.5,
        dwell=dwell,
    )
    orch = CacheOrchestrator(
        storage,
        PlaceholderKeyBuilder(),
        gates=_wrap_per_checkpoint(AlwaysSearchGate()),
        judges=_wrap_per_checkpoint(judge),
        search_strategies=_wrap_per_checkpoint(strategy),
        timer=SystemTimer(enabled=False),
    )
    return orch, judge, model


def _check(orch, index: int = 0):
    state = torch.zeros(1, 32)
    state[0, index] = 1.0
    return orch.check(CheckpointID.CP1, stage1=make_stage1(state))


# ------------------------------------------------------------------
# The production call path
# ------------------------------------------------------------------


def test_verdict_survives_the_real_orchestrator_call() -> None:
    """The regression the direct-call unit tests could not see: check() passes
    ``retrieval_signals`` unconditionally."""
    orch, _, model = _build(risk_value=0.1)
    result = _check(orch)

    assert result.hit_type is HitType.FULL_HIT
    assert model.calls == 1
    orch.clear()


def test_high_risk_routes_to_teacher_through_the_real_path() -> None:
    orch, _, _ = _build(risk_value=0.9)
    assert _check(orch).hit_type is HitType.MISS
    orch.clear()


def test_diagnostics_actually_reach_the_judge_through_the_chain() -> None:
    """If the step_features seam were broken anywhere between backend, facade,
    strategy and Orchestrator, the judge would fail safe and never score."""
    orch, _, model = _build(risk_value=0.1)
    _check(orch)
    orch.clear()

    # A scored decision proves diagnostics + query_keys both arrived.
    assert model.calls == 1


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------


def test_decision_index_resets_on_episode_start() -> None:
    """The Orchestrator broadcasts ``on_episode_start``; without that hook the
    step-fraction feature and the dwell countdown leak across episodes."""
    orch, judge, _ = _build(risk_value=0.1)
    _check(orch, 0)
    _check(orch, 1)
    orch.clear()

    orch.on_episode_start()
    result = _check(orch, 2)
    assert result.router_outputs["decision_idx"] == 0
    orch.clear()


def test_dwell_does_not_leak_into_the_next_episode() -> None:
    """A teacher dwell left running would silently spend the next episode's
    first decisions on the teacher."""
    orch, judge, model = _build(risk_value=0.9, dwell=3)
    _check(orch, 0)                     # trips the gate, arms dwell
    orch.clear()

    orch.on_episode_start()
    before = model.calls
    _check(orch, 1)
    orch.clear()
    # A fresh episode re-scores rather than serving a stale dwell.
    assert model.calls == before + 1


def test_task_begin_also_resets() -> None:
    orch, _, _ = _build(risk_value=0.1)
    _check(orch, 0)
    orch.clear()

    orch.on_task_begin("some_task")
    assert _check(orch, 1).router_outputs["decision_idx"] == 0
    orch.clear()


# ------------------------------------------------------------------
# Wrapper chain
# ------------------------------------------------------------------


def test_dumping_wrapper_relays_step_features_to_a_risk_router() -> None:
    """A dump-wrapped risk router must still receive its diagnostics.

    Without the relay the inner judge sees ``step_features=None`` and fails safe
    to teacher on every single step — a silent 100%-teacher run that no error
    would reveal.
    """
    from openpi.cache.components.dumping_judge import DumpingJudge
    from openpi.cache.storage_types import StepRetrievalFeatures

    seen: dict = {}

    class _Inner:
        def __call__(self, results, checkpoint_id, cached_data, *,
                     view=None, history=None, retrieval_signals=None,
                     step_features=None):
            seen["step_features"] = step_features
            from openpi.cache.components.judge import JudgeResult
            return JudgeResult(hit_type=HitType.MISS)

    wrapper = DumpingJudge(
        inner=_Inner(), dump_normalization=None, dump_factors=[],
        dump_path="", config_id="x15_test",
    )
    features = StepRetrievalFeatures(n_results=3)
    wrapper([], CheckpointID.CP1, {}, step_features=features)

    assert seen["step_features"] is features


def test_dumping_wrapper_keeps_a_legacy_inner_call_byte_identical() -> None:
    """A legacy inner judge must not start receiving the new kwarg."""
    from openpi.cache.components.dumping_judge import DumpingJudge
    from openpi.cache.storage_types import StepRetrievalFeatures

    captured: dict = {}

    class _Legacy:
        def __call__(self, results, checkpoint_id, cached_data, *,
                     view=None, history=None, retrieval_signals=None):
            captured["kwargs_ok"] = True
            from openpi.cache.components.judge import JudgeResult
            return JudgeResult(hit_type=HitType.MISS)

    wrapper = DumpingJudge(
        inner=_Legacy(), dump_normalization=None, dump_factors=[],
        dump_path="", config_id="x15_test",
    )
    # Would raise TypeError if the wrapper forwarded step_features blindly.
    wrapper([], CheckpointID.CP1, {}, step_features=StepRetrievalFeatures(n_results=1))
    assert captured["kwargs_ok"]


def test_orchestrator_probe_admits_the_dump_wrapper() -> None:
    """The wrapper declares the kwarg so the Orchestrator's signature probe
    injects diagnostics into a dump-wrapped router at all."""
    from openpi.cache.components.dumping_judge import DumpingJudge
    from openpi.cache.components.judge import judge_accepts_kwarg

    class _Inner:
        def __call__(self, results, checkpoint_id, cached_data, *,
                     view=None, history=None, retrieval_signals=None,
                     step_features=None):
            from openpi.cache.components.judge import JudgeResult
            return JudgeResult(hit_type=HitType.MISS)

    wrapper = DumpingJudge(
        inner=_Inner(), dump_normalization=None, dump_factors=[],
        dump_path="", config_id="x15_test",
    )
    assert judge_accepts_kwarg(wrapper, "step_features")
