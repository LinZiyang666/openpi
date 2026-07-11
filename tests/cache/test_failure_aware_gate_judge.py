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


# ----------------------------------------------------------------------
# Phase 5 — u_t kinematic activation + factor_outputs opt-in
# ----------------------------------------------------------------------

import torch  # noqa: E402

from openpi.cache.components.factors.base import HistoryView, LibraryStats  # noqa: E402
from openpi.cache.components.factors.normalization import ZScoreNormalization  # noqa: E402
from openpi.cache.components.factors.online import DirectionOnlineState  # noqa: E402
from openpi.cache.storage_types import CacheEntry, CachePayload  # noqa: E402

_UT_FACTOR = {"descriptor": "direction", "channel": "state", "past": 2, "future": 1}


class _StubView:
    """Minimal PayloadView: id->entry chain + walk_next (mirrors the factor tests)."""

    def __init__(self, entries):
        self._entries = {e.id: e for e in entries}
        self._chain = [e.id for e in entries]

    def get(self, entry_id):
        return self._entries[entry_id].payload

    def get_entry(self, entry_id):
        return self._entries[entry_id]

    def walk_next(self, entry_id, k):
        idx = self._chain.index(entry_id)
        return [self._entries[i] for i in self._chain[idx + 1 : idx + 1 + k]]


def _entry(eid, state):
    return CacheEntry(
        id=eid, checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.tensor(state, dtype=torch.float32)},
        payload=CachePayload(action_chunk=torch.zeros((1, 2), dtype=torch.float32)),
        trajectory_id="traj-1", prev_ids=[], next_ids=[],
    )


def _lib_stats():
    s = torch.ones(2, dtype=torch.float32)
    a = torch.ones(2, dtype=torch.float32)
    return LibraryStats(
        action_sigma=a, action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones(2, dtype=torch.bool),
    )


def _chain_fixture():
    """Winner w0 + forward w1 + 2-step history -> a computable direction u_t."""
    chain = [_entry("w0", [0.0, 0.0]), _entry("w1", [1.0, 1.0])]
    view = _StubView(chain)
    history = HistoryView(
        actions=[],
        states=[torch.tensor([-2.0, -2.0]), torch.tensor([-1.0, -1.0])],
    )
    results = [SearchResultLite("w0", 0.9, CheckpointID.CP1)]
    return view, history, results


def test_gate_u_t_matches_direct_factor():
    """Gate's internal u_t == the online factor's extract on the same context."""
    view, history, results = _chain_fixture()
    ls = _lib_stats()
    gate = FailureAwareGateJudge(
        gate_betas={"b0": -0.5, "b1": 1.0, "b2": 1.0, "b3": 0.0},
        u_t_factor=_UT_FACTOR, library_stats=ls, export_factor_outputs=True,
    )
    out = gate(results, CheckpointID.CP1, {}, view=view, history=history,
               retrieval_signals=_sig(margin=0.5))
    # Direct factor over an equivalent FactorContext.
    from openpi.cache.components.factors.base import FactorContext
    factor = DirectionOnlineState(windows=[{"past": 2, "future": 1}])
    ctx = FactorContext(results=results, view=view, history=history,
                        normalization=ZScoreNormalization(ls))
    (key,) = factor.descriptor_orientations.keys()
    expected = factor.extract(ctx)[key]
    assert out.factor_outputs["u_t"] == pytest.approx(expected)


def test_gate_u_t_activates_logit():
    """b2 != 0 with a non-zero u_t shifts g vs the b2=0 (margin-only) gate."""
    view, history, results = _chain_fixture()
    ls = _lib_stats()
    common = dict(u_t_factor=_UT_FACTOR, library_stats=ls)
    g_active = FailureAwareGateJudge(gate_betas={"b0": -0.5, "b1": 1.0, "b2": 5.0}, **common)(
        results, CheckpointID.CP1, {}, view=view, history=history,
        retrieval_signals=_sig(margin=0.5)).composer_score
    g_inert = FailureAwareGateJudge(gate_betas={"b0": -0.5, "b1": 1.0, "b2": 0.0}, **common)(
        results, CheckpointID.CP1, {}, view=view, history=history,
        retrieval_signals=_sig(margin=0.5)).composer_score
    assert g_active != pytest.approx(g_inert)


def test_gate_u_t_nan_degrades_to_margin_only():
    """Short history (< past) -> NaN u_t -> b2 term dropped -> == u_t_factor=None gate."""
    view, _, results = _chain_fixture()
    short_history = HistoryView(actions=[], states=[torch.tensor([-1.0, -1.0])])  # len 1 < past 2
    ls = _lib_stats()
    betas = {"b0": -0.5, "b1": 1.0, "b2": 5.0}
    g_ut = FailureAwareGateJudge(gate_betas=betas, u_t_factor=_UT_FACTOR, library_stats=ls)(
        results, CheckpointID.CP1, {}, view=view, history=short_history,
        retrieval_signals=_sig(margin=0.3)).composer_score
    g_margin = FailureAwareGateJudge(gate_betas=betas)(
        results, CheckpointID.CP1, {}, view=view, history=short_history,
        retrieval_signals=_sig(margin=0.3)).composer_score
    assert g_ut == pytest.approx(g_margin)


def test_gate_factor_outputs_default_off_wire_invariant():
    """Default (export False) -> factor_outputs is None (Phase 3 wire byte-identical)."""
    res = _gate()(_res(), CheckpointID.CP1, {}, retrieval_signals=_sig(margin=0.9))
    assert res.factor_outputs is None


def test_gate_factor_outputs_opt_in_nan_to_none():
    """export True -> dict carries schema + raw signals; NaN u_t -> None, JSON-safe."""
    import json
    view, _, results = _chain_fixture()
    short_history = HistoryView(actions=[], states=[])  # empty -> NaN u_t
    gate = FailureAwareGateJudge(
        gate_betas={"b0": -0.5, "b1": 1.0, "b2": 5.0},
        u_t_factor=_UT_FACTOR, library_stats=_lib_stats(), export_factor_outputs=True,
    )
    out = gate(results, CheckpointID.CP1, {}, view=view, history=short_history,
               retrieval_signals=_sig(margin=0.4, s_pos=0.4, s_neg=0.0, delta_pos=0.01))
    fo = out.factor_outputs
    assert fo["schema"] == "failure_gate_v1"
    assert fo["u_t"] is None  # NaN pre-converted
    assert set(fo) >= {"s_pos", "s_neg", "margin", "delta_pos", "u_t", "g"}
    json.dumps(fo, allow_nan=False)  # must not raise


def test_gate_u_t_factor_requires_library_stats():
    with pytest.raises(ValueError, match="library_stats"):
        FailureAwareGateJudge(gate_betas={"b0": -0.5, "b1": 1.0, "b2": 1.0},
                              u_t_factor=_UT_FACTOR, library_stats=None)
