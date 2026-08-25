"""X15 U5 — the risk-gate verdict, its fail-safe, and its feature contract.

The gate's value depends on three things being true no matter what the
retrieval layer hands it:

* degraded input routes to teacher, never to a guessed cache replay;
* teacher dwell holds for the configured number of decisions;
* the A-tier vector reads retrieval evidence and scalar time only — never the
  trajectory history the Orchestrator injects unconditionally.

Plus the two load-time refusals that keep a wrong model or a wrong search path
from ever reaching a verdict.

Key dependencies: ``RiskRouterJudge``, ``RiskFeatureBuilder``, ``RiskModel``.
"""

from __future__ import annotations

import pytest
import torch

from openpi.cache.components.judge import HitType, judge_accepts_kwarg
from openpi.cache.components.risk_features import (
    FEATURE_DIM,
    RiskFeatureBuilder,
    feature_schema_digest,
)
from openpi.cache.components.risk_model import IsotonicMap, RiskModel, RiskNet
from openpi.cache.components.risk_router_judge import RiskRouterJudge
from openpi.cache.storage_types import (
    CacheEntry,
    CachePayload,
    SearchResultLite,
    StepRetrievalFeatures,
)
from openpi.cache.types import CheckpointID

# ------------------------------------------------------------------
# Doubles
# ------------------------------------------------------------------


class _ConstantRisk:
    """Risk model stand-in with a fixed output and a recording hook."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.seen: list[torch.Tensor] = []

    def risk(self, x: torch.Tensor) -> float:
        self.seen.append(x)
        return self.value


class _View:
    def __init__(self, entry: CacheEntry) -> None:
        self._entry = entry

    def get_entry(self, entry_id: str) -> CacheEntry:
        return self._entry

    def get(self, entry_id: str) -> CachePayload:
        return self._entry.payload


def _neighbour(step_idx: int = 4) -> CacheEntry:
    return CacheEntry(
        id="n1",
        checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.zeros(32)},
        payload=CachePayload(action_chunk=torch.zeros(50, 32)),
        step_idx=step_idx,
    )


def _results() -> list[SearchResultLite]:
    return [SearchResultLite(id="n1", score=0.9, checkpoint_id=CheckpointID.CP1)]


def _features() -> StepRetrievalFeatures:
    return StepRetrievalFeatures(
        fused_topk=(("n1", 0.9), ("n2", 0.5)),
        winner_per_field={"vision_0": 0.9, "vision_1": 0.8, "robot_state": 0.7},
        field_own_margin={"vision_0": 0.4, "vision_1": 0.3, "robot_state": 0.2},
        fused_margin=0.4,
        n_results=2,
    )


def _builder() -> RiskFeatureBuilder:
    return RiskFeatureBuilder(task_index=3, replan_steps=5, library_replan_steps=5)


def _judge(risk: float, *, tau: float = 0.5, dwell: int = 1) -> tuple[RiskRouterJudge, _ConstantRisk]:
    model = _ConstantRisk(risk)
    return (
        RiskRouterJudge(
            feature_builder=_builder(), risk_model=model, tau=tau, dwell=dwell,
        ),
        model,
    )


def _call(judge: RiskRouterJudge, **overrides):
    kwargs = {
        "view": _View(_neighbour()),
        "history": object(),      # injected unconditionally; must be ignored
        "step_features": _features(),
        "query_keys": {"robot_state": torch.ones(32)},
    }
    kwargs.update(overrides)
    return judge(_results(), CheckpointID.CP1, {"robot_state": torch.ones(32)}, **kwargs)


# ------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------


def test_low_risk_replays_the_cache_and_pins_the_winner() -> None:
    judge, _ = _judge(0.1, tau=0.5)
    verdict = _call(judge)

    assert verdict.hit_type is HitType.FULL_HIT
    assert verdict.winner_id == "n1"
    # False, not None: force the cached replay even when a hit_executor exists.
    assert verdict.hit_override is False
    assert verdict.router_outputs["arm_sampled"] == "cache"


def test_high_risk_routes_to_teacher() -> None:
    judge, _ = _judge(0.9, tau=0.5)
    verdict = _call(judge)

    assert verdict.hit_type is HitType.MISS
    # The declined candidate rides along on a MISS: teacher executes either
    # way, but the shadow labeller needs to know which cached chunk was passed
    # over, and it was already retrieved.
    assert verdict.winner_id == "n1"
    assert verdict.router_outputs["arm_sampled"] == "teacher"
    assert verdict.router_outputs["reason"] == "risk"


def test_decision_index_advances_across_verdicts() -> None:
    judge, _ = _judge(0.1)
    assert _call(judge).router_outputs["decision_idx"] == 0
    assert _call(judge).router_outputs["decision_idx"] == 1


# ------------------------------------------------------------------
# Fail-safe
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"step_features": None},
        {"query_keys": None},
    ],
    ids=["no_diagnostics", "no_query_keys"],
)
def test_missing_inputs_fail_safe_to_teacher(override) -> None:
    """A gate that cannot form an opinion must pick the expensive-but-correct
    arm; guessing cache is the one unrecoverable direction."""
    judge, model = _judge(0.0, tau=0.5)     # model would say "safe" if consulted
    verdict = _call(judge, **override)

    assert verdict.hit_type is HitType.MISS
    assert verdict.router_outputs["reason"] == "fail_safe"
    assert judge.fallback_count == 1
    assert model.seen == []                  # never even scored


def test_empty_results_fail_safe_to_teacher() -> None:
    judge, _ = _judge(0.0, tau=0.5)
    verdict = judge(
        [], CheckpointID.CP1, {},
        view=_View(_neighbour()), history=None,
        step_features=_features(), query_keys={"robot_state": torch.ones(32)},
    )
    assert verdict.hit_type is HitType.MISS
    assert verdict.router_outputs["reason"] == "fail_safe"


def test_feature_builder_failure_fails_safe_rather_than_raising() -> None:
    """A malformed neighbour must not kill the episode."""
    class _BadView:
        def get_entry(self, entry_id: str):
            raise KeyError(entry_id)

    judge, _ = _judge(0.0, tau=0.5)
    verdict = _call(judge, view=_BadView())
    assert verdict.hit_type is HitType.MISS
    assert verdict.router_outputs["reason"] == "fail_safe"


def test_non_finite_risk_fails_safe() -> None:
    judge, _ = _judge(float("nan"), tau=0.5)
    verdict = _call(judge)
    assert verdict.hit_type is HitType.MISS
    assert verdict.router_outputs["reason"] == "fail_safe"


# ------------------------------------------------------------------
# Dwell
# ------------------------------------------------------------------


def test_dwell_holds_teacher_without_rescoring() -> None:
    """Drift is not a one-step event: after switching, the gate stays on
    teacher for ``dwell`` decisions instead of flapping back."""
    judge, model = _judge(0.9, tau=0.5, dwell=3)

    first = _call(judge)
    assert first.router_outputs["reason"] == "risk"
    scored_after_first = len(model.seen)

    for _ in range(2):
        held = _call(judge)
        assert held.hit_type is HitType.MISS
        assert held.router_outputs["reason"] == "dwell"
    # Held decisions cost no model evaluation.
    assert len(model.seen) == scored_after_first


def test_dwell_one_rescore_every_decision() -> None:
    judge, model = _judge(0.9, tau=0.5, dwell=1)
    _call(judge)
    _call(judge)
    assert len(model.seen) == 2


def test_dwell_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="dwell"):
        RiskRouterJudge(
            feature_builder=_builder(), risk_model=_ConstantRisk(0.1), tau=0.5, dwell=0,
        )


# ------------------------------------------------------------------
# Feature contract
# ------------------------------------------------------------------


def test_feature_vector_has_the_frozen_width() -> None:
    judge, model = _judge(0.1)
    _call(judge)
    assert model.seen[0].numel() == FEATURE_DIM == 59


def test_history_is_never_read() -> None:
    """A-tier scope: the Orchestrator injects history unconditionally, and a
    judge that quietly consumed it would break the frozen paper scope."""
    class _ExplodingHistory:
        def __getattr__(self, name: str):
            raise AssertionError(f"history was read: {name}")

    judge, _ = _judge(0.1)
    verdict = _call(judge, history=_ExplodingHistory())
    assert verdict.hit_type is HitType.FULL_HIT


def test_two_time_axes_are_converted_before_comparison() -> None:
    """A library entry's ``step_idx`` counts ITS episode's inference cycles, so
    a library built at a different replan interval must not be compared raw."""
    same = RiskFeatureBuilder(task_index=0, replan_steps=5, library_replan_steps=5)
    wider = RiskFeatureBuilder(task_index=0, replan_steps=5, library_replan_steps=10)
    kwargs = dict(
        results=_results(), step_features=_features(),
        query_keys={"robot_state": torch.ones(32)},
        view=_View(_neighbour(step_idx=4)), decision_idx=4,
    )
    # f7 slot 0 (index 45) is the neighbour phase; it scales with the library
    # interval, which is the whole reason the two axes are converted separately.
    phase_same = same.build(**kwargs)[45]
    phase_wider = wider.build(**kwargs)[45]
    assert phase_wider == pytest.approx(2 * float(phase_same))


def test_builder_rejects_a_neighbour_without_a_step_index() -> None:
    builder = _builder()
    entry = _neighbour()
    entry.step_idx = None
    with pytest.raises(ValueError, match="step_idx"):
        builder.build(
            results=_results(), step_features=_features(),
            query_keys={"robot_state": torch.ones(32)},
            view=_View(entry), decision_idx=0,
        )


def test_neighbour_disagreement_is_visible_when_scores_are_not() -> None:
    """f8's reason for existing: neighbours can score identically and still
    disagree about the action, which no similarity threshold can see."""
    from openpi.cache.components.risk_features import RiskFeatureBuilder

    class _MultiView:
        def __init__(self, chunks):
            self._chunks = chunks

        def get_entry(self, entry_id):
            return _neighbour()

        def get(self, entry_id):
            return CachePayload(action_chunk=self._chunks[entry_id])

    results = [
        SearchResultLite(id="n1", score=0.9, checkpoint_id=CheckpointID.CP1),
        SearchResultLite(id="n2", score=0.9, checkpoint_id=CheckpointID.CP1),
    ]
    kwargs = dict(
        results=results, step_features=_features(),
        query_keys={"robot_state": torch.ones(32)}, decision_idx=0,
    )
    agree = RiskFeatureBuilder(task_index=0, replan_steps=5, library_replan_steps=5).build(
        view=_MultiView({"n1": torch.zeros(50, 32), "n2": torch.zeros(50, 32)}), **kwargs
    )
    disagree = RiskFeatureBuilder(task_index=0, replan_steps=5, library_replan_steps=5).build(
        view=_MultiView({"n1": torch.zeros(50, 32), "n2": torch.ones(50, 32)}), **kwargs
    )
    # Absolute slot: f1(5) f2(3) f3(4) f5(32) f6(1) f7(2) -> f8 at index 47.
    assert float(disagree[47]) > float(agree[47])


def test_a_wrong_library_replan_is_caught_rather_than_silently_rescaled() -> None:
    """A library_replan_steps that does not match the library would scale every
    phase feature; the guard turns that into a loud error."""
    from openpi.cache.components.risk_features import RiskFeatureBuilder

    builder = RiskFeatureBuilder(
        task_index=0, replan_steps=5, library_replan_steps=200,
    )
    with pytest.raises(ValueError, match="library_replan_steps is probably wrong"):
        builder.build(
            results=_results(), step_features=_features(),
            query_keys={"robot_state": torch.ones(32)},
            view=_View(_neighbour(step_idx=40)), decision_idx=0,
        )


def test_artifact_carries_the_provenance_needed_to_trace_a_tau(tmp_path) -> None:
    """cp_tau0 / seed / git_sha ride with the weights so a deployed threshold
    can be traced back to what produced it."""
    net = RiskNet(FEATURE_DIM, hidden=8)
    model = RiskModel(
        net, None, feature_schema_sha=feature_schema_digest(), delta=0.2,
        cp_tau0=0.61, seed=7, git_sha="abc1234",
    )
    path = str(tmp_path / "risk.pt")
    model.save(path)
    loaded = RiskModel.load(path, expected_schema_sha=feature_schema_digest())

    assert loaded.cp_tau0 == pytest.approx(0.61)
    assert loaded.seed == 7
    assert loaded.git_sha == "abc1234"


def test_judge_declares_step_features_and_not_query_keys_only() -> None:
    """The Orchestrator's signature probe is what routes diagnostics here."""
    judge, _ = _judge(0.1)
    assert judge_accepts_kwarg(judge, "step_features")


# ------------------------------------------------------------------
# Artifact contract
# ------------------------------------------------------------------


def test_model_refuses_a_mismatched_feature_schema(tmp_path) -> None:
    """Scoring through a different feature layout is confident nonsense that no
    runtime fail-safe can detect, so it is refused at load."""
    net = RiskNet(FEATURE_DIM, hidden=8)
    model = RiskModel(net, None, feature_schema_sha="stale-digest", delta=0.1)
    path = str(tmp_path / "risk.pt")
    model.save(path)

    with pytest.raises(ValueError, match="refusing to load"):
        RiskModel.load(path, expected_schema_sha=feature_schema_digest())

    reloaded = RiskModel.load(path, expected_schema_sha="stale-digest")
    assert reloaded.delta == pytest.approx(0.1)


def test_isotonic_is_monotone_and_clamped() -> None:
    iso = IsotonicMap(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.1, 0.4, 0.9]))
    assert iso(-5.0) == pytest.approx(0.1)
    assert iso(5.0) == pytest.approx(0.9)
    assert iso(0.5) == pytest.approx(0.25)
    assert iso(0.5) <= iso(1.5)


def test_risk_is_deterministic_across_calls() -> None:
    net = RiskNet(FEATURE_DIM, hidden=8)
    model = RiskModel(
        net, IsotonicMap(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])),
        feature_schema_sha=feature_schema_digest(), delta=0.1,
    )
    x = torch.arange(FEATURE_DIM, dtype=torch.float32) / FEATURE_DIM
    assert model.risk(x) == model.risk(x)
