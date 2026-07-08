"""Unit tests for FailureAwareGateJudge (TRACER Phase 3 / M2 skeleton).

Covers the sigmoid gate math, the three-state (FULL_HIT / WARM_START / MISS)
decision, the ThresholdJudge-equivalent degenerate anchor (NR3), the
retrieval_signals fail-loud contract, and numerical stability.
"""

from __future__ import annotations

import math

import pytest

from openpi.cache.components.judge import FailureAwareGateJudge, HitType
from openpi.cache.storage_types import RetrievalSignals, SearchResultLite
from openpi.cache.types import CheckpointID


def _res(score: float = 0.9, rid: str = "w0"):
    return [SearchResultLite(rid, score, CheckpointID.CP1)]


def _sig(*, margin=0.0, delta_pos=0.0, s_pos=0.0, s_neg=0.0, lambda_=0.0):
    return RetrievalSignals(
        s_pos=s_pos, s_neg=s_neg, margin=margin, delta_pos=delta_pos, lambda_=lambda_
    )


def _gate(**over) -> FailureAwareGateJudge:
    kwargs = dict(gate_betas={"b0": -0.5, "b1": 1.0, "b3": 0.0}, threshold=0.5)
    kwargs.update(over)
    return FailureAwareGateJudge(**kwargs)


@pytest.mark.parametrize(
    "margin,expected",
    [
        (0.9, HitType.FULL_HIT),  # margin 0.9 >= tau 0.5 -> g > 0.5
        (0.1, HitType.MISS),      # margin 0.1 <  tau 0.5 -> g < 0.5
        (0.5, HitType.FULL_HIT),  # boundary: g = sigmoid(0) = 0.5 >= 0.5
    ],
)
def test_gate_degenerate_threshold(margin, expected):
    """NR3: default gate (b0=-tau, b1=1, b3=0, threshold=0.5) => FULL_HIT iff margin>=tau."""
    res = _gate()(_res(), CheckpointID.CP1, {}, retrieval_signals=_sig(margin=margin))
    assert res.hit_type is expected
    if expected is HitType.FULL_HIT:
        assert res.winner_id == "w0"


def test_gate_composer_score_is_sigmoid():
    margin = 0.8
    res = _gate()(_res(), CheckpointID.CP1, {}, retrieval_signals=_sig(margin=margin))
    z = -0.5 + 1.0 * margin
    assert res.composer_score == pytest.approx(1.0 / (1.0 + math.exp(-z)))


def test_gate_empty_results_miss():
    res = _gate()([], CheckpointID.CP1, {}, retrieval_signals=_sig(margin=0.9))
    assert res.hit_type is HitType.MISS


def test_gate_none_signals_raises():
    with pytest.raises(ValueError, match="retrieval_signals"):
        _gate()(_res(), CheckpointID.CP1, {}, retrieval_signals=None)


def test_gate_three_state_warm_cp1():
    # warm band on g: full_hit at g>=0.5, warm at g>=0.3.
    gate = _gate(warm_tiers=[{"threshold": 0.3, "start_t": 0.5}])
    # margin=0.2 -> z=-0.3 -> g=sigmoid(-0.3)~0.4256 in [0.3, 0.5) -> WARM_START.
    res = gate(_res(), CheckpointID.CP1, {}, retrieval_signals=_sig(margin=0.2))
    assert res.hit_type is HitType.WARM_START
    assert res.start_t == 0.5


def test_gate_warm_tiers_not_applied_on_cp3():
    gate = _gate(warm_tiers=[{"threshold": 0.3, "start_t": 0.5}])
    # same g in [0.3, 0.5) but CP3 -> no warm band -> MISS.
    res = gate(_res(), CheckpointID.CP3, {}, retrieval_signals=_sig(margin=0.2))
    assert res.hit_type is HitType.MISS


def test_gate_delta_term_contributes():
    # b3 != 0: delta_pos drives the gate (b1=0 isolates the delta term).
    gate = FailureAwareGateJudge(gate_betas={"b0": 0.0, "b1": 0.0, "b3": 10.0}, threshold=0.5)
    hi = gate(_res(), CheckpointID.CP1, {}, retrieval_signals=_sig(delta_pos=1.0))
    lo = gate(_res(), CheckpointID.CP1, {}, retrieval_signals=_sig(delta_pos=-1.0))
    assert hi.hit_type is HitType.FULL_HIT  # z=10 -> g~1
    assert lo.hit_type is HitType.MISS      # z=-10 -> g~0


def test_gate_stable_sigmoid_extremes():
    # Large |z| must not overflow math.exp.
    gate = FailureAwareGateJudge(gate_betas={"b0": 0.0, "b1": 1000.0, "b3": 0.0}, threshold=0.5)
    hi = gate(_res(), CheckpointID.CP1, {}, retrieval_signals=_sig(margin=1.0))
    lo = gate(_res(), CheckpointID.CP1, {}, retrieval_signals=_sig(margin=-1.0))
    assert hi.composer_score == pytest.approx(1.0)
    assert lo.composer_score == pytest.approx(0.0)


def test_gate_ignores_view_history_kwargs():
    # Orchestrator injects view/history unconditionally; the gate must accept
    # and ignore them (via **kwargs) alongside retrieval_signals.
    res = _gate()(
        _res(), CheckpointID.CP1, {},
        view=None, history=None, retrieval_signals=_sig(margin=0.9),
    )
    assert res.hit_type is HitType.FULL_HIT
