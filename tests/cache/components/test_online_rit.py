"""Online RIT: signal functions, estimator, dispatch (plan §8 items 1-4)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from openpi.cache.components.judge import HitType
from openpi.cache.components.online_rit import (
    GAP_INTERPOLATED,
    SUPPORTED,
    TAIL_EXTRAPOLATED,
    UNAVAILABLE,
    ContinuationFeedback,
    ContinuationSpec,
    OnlineRiskCurves,
    OnlineRitJudge,
    continuation_disagreement,
    cut_at_pl,
    pav_nonincreasing,
    reference_update,
    tier_specs,
    weighted_quantile,
)
from openpi.cache.online_state import CurveRegistry
from openpi.cache.storage_types import CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID, groot_n15_schedule

SCHEDULE = groot_n15_schedule(8)
H, D = 16, 32
TIERS = [7, 6, 4]


def _payload() -> CachePayload:
    torch.manual_seed(0)
    inter = {SCHEDULE.snapshot_t(i): torch.randn(H, D) for i in range(1, 8)}
    return CachePayload(
        action_chunk=torch.randn(H, D),
        intermediates=inter,
        denoising_num_steps=8,
        schedule_id=SCHEDULE.schedule_id,
    )


# ------------------------------------------------------------------
# signal
# ------------------------------------------------------------------


def test_reference_update_is_snapshot_difference_times_n():
    p = _payload()
    u = reference_update(p, 0.75, SCHEDULE)
    expected = (p.intermediates[0.875] - p.intermediates[0.75]) * 8
    assert torch.allclose(u, expected)
    u_last = reference_update(p, 0.875, SCHEDULE)
    assert torch.allclose(u_last, (p.action_chunk - p.intermediates[0.875]) * 8)


def test_reference_update_raises_on_missing_neighbour():
    p = _payload()
    del p.intermediates[0.875]
    with pytest.raises(ValueError):
        reference_update(p, 0.75, SCHEDULE)
    p2 = _payload()
    p2.action_chunk = None
    with pytest.raises(ValueError):
        reference_update(p2, 0.875, SCHEDULE)


def test_disagreement_only_counts_executed_dims_and_window():
    mask = torch.zeros(D, dtype=torch.bool)
    mask[:7] = True
    scale = torch.ones(D)
    a = torch.zeros(H, D)
    b = torch.zeros(H, D)
    assert continuation_disagreement(a, b, scale, mask, 5) == 0.0
    b2 = b.clone()
    b2[:, 7:] = 100.0  # padding dims only
    b2[6:, :7] = 100.0  # beyond the executed window
    assert continuation_disagreement(a, b2, scale, mask, 5) == 0.0
    b3 = b.clone()
    b3[0, 0] = 3.0
    assert continuation_disagreement(a, b3, scale, mask, 5) == pytest.approx(3.0 / 5)
    with pytest.raises(ValueError):
        continuation_disagreement(a, b3 * float("nan"), scale, mask, 5)
    with pytest.raises(ValueError):
        continuation_disagreement(a, b3, scale, torch.zeros(D, dtype=torch.bool), 5)


# ------------------------------------------------------------------
# primitives
# ------------------------------------------------------------------


def test_weighted_quantile_matches_hand_values():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)
    assert weighted_quantile(v, w, 0.95) == 4.0
    assert weighted_quantile(v, w, 0.5) == 2.0
    w2 = np.array([10.0, 1.0, 1.0, 1.0])
    assert weighted_quantile(v, w2, 0.5) == 1.0


def test_pav_projects_to_nonincreasing():
    out = pav_nonincreasing(np.array([1.0, 3.0, 2.0, 0.5]), np.ones(4))
    assert (np.diff(out) <= 1e-12).all()
    assert out[0] == pytest.approx(2.0) and out[1] == pytest.approx(2.0)


def test_cut_at_pl_matches_rit_k_on_finite_input():
    rit_k = pytest.importorskip("exp.rit_pareto.rit_k")
    rng = np.random.default_rng(1)
    for _ in range(200):
        knots = np.sort(rng.uniform(0, 1, size=5))
        knots[0], knots[-1] = 0.0, 1.0
        q = np.sort(rng.uniform(0, 2, size=5))[::-1]
        q = q + np.linspace(0.4, 0.0, 5)  # strictly decreasing
        fit = rit_k.PLFitK(
            knots=knots,
            q={"warm": q},
            tiers=(rit_k.Tier("warm", "WARM_START", 0.3, "y"),),
            eps_total=0.02,
            n_seg_req=4,
            n_seg=4,
            alpha=0.05,
        )
        delta = float(rng.uniform(-0.5, 2.5))
        assert cut_at_pl(knots, q, delta) == pytest.approx(rit_k.cut_at(fit, "warm", delta), abs=1e-12)


def test_cut_at_pl_never_returns_nan_and_handles_plateau():
    assert cut_at_pl(np.array([0.0, 1.0]), np.array([1.0, 1.0]), 1.0) == 0.0
    assert cut_at_pl(np.array([0.0, 1.0]), np.array([2.0, 2.0]), 1.0) == math.inf
    val = cut_at_pl(np.array([0.0, 0.5, 1.0]), np.array([2.0, 1.0, 1.0]), 1.0)
    assert val == 0.5
    with pytest.raises(ValueError):
        cut_at_pl(np.array([0.0, 1.0]), np.array([math.inf, 1.0]), 1.5)


# ------------------------------------------------------------------
# estimator
# ------------------------------------------------------------------


def _curves(**kw) -> OnlineRiskCurves:
    base = dict(knots=[0.0, 0.25, 0.5, 0.75, 1.0], tier_indices=TIERS, alpha=0.05, window=128, n_min=20)
    base.update(kw)
    return OnlineRiskCurves(**base)


def _feed(c: OnlineRiskCurves, rows, tag="t"):
    for n, (s, ds) in enumerate(rows):
        fb = [ContinuationFeedback(t, float(d), "shadow") for t, d in zip(TIERS, ds)]
        c.update_batch(s, fb, (tag, n))


def test_cold_start_has_no_valid_knots_and_infinite_cuts():
    c = _curves()
    for t in TIERS:
        assert not c.valid(t).any()
        assert c.cut(t, 10.0) == (math.inf, False)
        assert c.query(t, 0.5) == (None, UNAVAILABLE)


def test_held_out_coverage_on_stationary_synthetic_data():
    rng = np.random.default_rng(3)
    c = _curves()
    rows = []
    for _ in range(3000):
        s = float(rng.uniform(0, 1))
        rows.append((s, [(1 - s) * 0.5 + 0.1 * abs(rng.standard_normal()) for _ in TIERS]))
    _feed(c, rows)
    exceed = 0
    n = 0
    for _ in range(3000):
        s = float(rng.uniform(0.05, 0.95))
        d = (1 - s) * 0.5 + 0.1 * abs(rng.standard_normal())
        q, kind = c.query(7, s)
        assert kind == SUPPORTED
        n += 1
        exceed += d > q
    assert 0.02 <= exceed / n <= 0.09


def test_support_kinds_with_gap_and_tail():
    c = _curves()
    # samples only near knots 0.25 and 0.75 -> knots 0, 0.5, 1 invalid
    rows = [(0.25, [1.0] * 3)] * 30 + [(0.75, [0.5] * 3)] * 30
    _feed(c, rows)
    assert list(c.valid(7)) == [False, True, False, True, False]
    assert c.query(7, 0.1) == (None, UNAVAILABLE)
    assert c.query(7, 0.25) == (1.0, SUPPORTED)
    q, kind = c.query(7, 0.5)
    assert kind == GAP_INTERPOLATED and q == pytest.approx(0.75)
    assert c.query(7, 0.9) == (0.5, TAIL_EXTRAPOLATED)
    theta, ok = c.cut(7, 0.6)
    assert ok and 0.25 < theta < 0.75
    assert c.cut(7, 0.4) == (math.inf, False)
    assert c.cut(7, 2.0) == (0.25, True)


def test_single_valid_knot():
    c = _curves()
    _feed(c, [(0.5, [0.3] * 3)] * 25)
    assert c.query(7, 0.5) == (0.3, SUPPORTED)
    assert c.query(7, 0.7) == (0.3, TAIL_EXTRAPOLATED)
    assert c.query(7, 0.3) == (None, UNAVAILABLE)
    assert c.cut(7, 0.3) == (0.5, True)


def test_pav_keeps_curves_nonincreasing():
    rng = np.random.default_rng(5)
    c = _curves()
    rows = [(float(rng.uniform(0, 1)), [float(rng.uniform(0, 1))] * 3) for _ in range(2000)]
    _feed(c, rows)
    q = c.q_values(7)[c.valid(7)]
    assert (np.diff(q) <= 1e-12).all()


def test_frozen_state_records_but_never_moves():
    a = _curves()
    rng = np.random.default_rng(7)
    rows = [(float(rng.uniform(0, 1)), [float(rng.uniform(0, 1))] * 3) for _ in range(300)]
    _feed(a, rows, "init")
    frozen = OnlineRiskCurves.from_snapshot(a.snapshot(), update_enabled=False)
    live = OnlineRiskCurves.from_snapshot(a.snapshot(), update_enabled=True)
    assert frozen.learning_state_sha256() == live.learning_state_sha256()
    sha0 = frozen.learning_state_sha256()
    more = [(float(rng.uniform(0, 1)), [2.0] * 3) for _ in range(100)]
    _feed(frozen, more, "x")
    _feed(live, more, "x")
    assert frozen.learning_state_sha256() == sha0
    assert frozen.revision == 0 and frozen.n_observed == 300
    assert live.learning_state_sha256() != sha0 and live.revision == 100
    assert frozen.cuts(0.5) == a.cuts(0.5)


def test_snapshot_roundtrip_gives_identical_next_state():
    a = _curves()
    rng = np.random.default_rng(9)
    rows = [(float(rng.uniform(0, 1)), [float(rng.uniform(0, 1))] * 3) for _ in range(150)]
    _feed(a, rows, "a")
    b = OnlineRiskCurves.from_snapshot(a.snapshot(), update_enabled=True, reset_counters=False)
    assert b.learning_state_sha256() == a.learning_state_sha256()
    nxt = [(0.42, [0.9, 0.8, 0.7])]
    _feed(a, nxt, "n")
    _feed(b, nxt, "n")
    assert a.learning_state_sha256() == b.learning_state_sha256()
    assert a.snapshot()["curves"] == b.snapshot()["curves"]


def test_snapshot_json_has_no_infinities():
    import json

    c = _curves()
    _feed(c, [(0.5, [0.3] * 3)] * 25)
    text = json.dumps(c.snapshot(), allow_nan=False)
    assert "Infinity" not in text and "NaN" not in text
    curves = c.snapshot()["curves"]["7"]
    assert curves["q"][0] is None and curves["valid"][0] is False


def test_tampered_snapshots_are_refused_on_every_load_path():
    import copy
    import json

    a = _curves()
    _feed(a, [(0.5, [0.1] * 3)] * 25, "a")
    snap = a.snapshot()
    OnlineRiskCurves.from_snapshot(copy.deepcopy(snap), update_enabled=False)  # intact loads
    edited = copy.deepcopy(snap)
    edited["windows"]["7"][2][0][0] = 999.0  # change a stored d, keep the recorded hashes
    edited["state_sha256"] = None  # drop the document hash: the learning hash must still catch it
    with pytest.raises(ValueError, match="learning_state_sha256"):
        OnlineRiskCurves.from_snapshot(edited, update_enabled=False)
    doc_edited = copy.deepcopy(snap)
    doc_edited["n_observed"] = 42  # outside the learning state, inside the document hash
    with pytest.raises(ValueError, match="state_sha256"):
        OnlineRiskCurves.from_snapshot(doc_edited, update_enabled=True)
    no_hash = copy.deepcopy(snap)
    del no_hash["learning_state_sha256"]
    del no_hash["state_sha256"]
    with pytest.raises(ValueError, match="no learning_state_sha256"):
        OnlineRiskCurves.from_snapshot(no_hash, update_enabled=True)
    assert json.loads(json.dumps(snap)) == snap


def test_feedback_helper_uses_the_consumed_input_and_reports_failures():
    from openpi.cache.components.online_rit import feedback_from_updates

    spec = _spec()
    t = 0.75
    # a zero vector field in the library too: every snapshot equals the chunk,
    # so the stored update is zero and a zero current update must give d = 0
    x_stored = torch.full((H, D), 1.001)  # not representable in BF16
    p = CachePayload(
        action_chunk=x_stored.clone(),
        intermediates={SCHEDULE.snapshot_t(i): x_stored.clone() for i in range(1, 8)},
        denoising_num_steps=8,
        schedule_id=SCHEDULE.schedule_id,
    )
    x_used = x_stored.to(torch.bfloat16).float()  # what a BF16 head actually consumes
    fb, reasons = feedback_from_updates(spec, p, SCHEDULE, executed=(t, x_used, x_used), side=[(0.5, x_used, x_used)])
    assert reasons == [] and [f.d for f in fb] == [0.0, 0.0]
    # differencing against the stored FP32 input would count the rounding
    u_bad = (x_used - x_stored) * 8
    assert float(u_bad.abs().max()) > 0
    fb2, reasons2 = feedback_from_updates(spec, p, SCHEDULE, executed=(t, None, x_used), side=[(0.5, x_used, x_used * float("nan"))])
    assert fb2 == [] and len(reasons2) == 2
    assert "missing_capture" in reasons2[0] and "not finite" in reasons2[1]


def test_duplicate_decision_and_bad_score_are_rejected():
    c = _curves()
    fb = [ContinuationFeedback(7, 0.1, "shadow")]
    c.update_batch(0.5, fb, ("d", 1))
    with pytest.raises(ValueError):
        c.update_batch(0.5, fb, ("d", 1))
    with pytest.raises(ValueError):
        c.update_batch(1.5, fb, ("d", 2))
    with pytest.raises(ValueError):
        ContinuationFeedback(7, float("nan"), "shadow")
    with pytest.raises(ValueError):
        ContinuationFeedback(7, 0.1, "probe")


def test_fm0_replay_leaves_unselected_knots_empty_but_cuts_may_move_left():
    """Only the selected tier is fed; knots the rule never selects stay empty."""
    rng = np.random.default_rng(11)
    c = _curves()
    delta = 0.3
    # warm-start the curve from a high-score region only
    _feed(c, [(0.9, [0.1] * 3)] * 60, "w")
    assert list(c.valid(7)) == [False, False, False, True, True]
    theta0 = c.cut(7, delta)[0]
    for n in range(300):
        s = float(rng.uniform(0, 1))
        cuts = c.cuts(delta)
        chosen = next((t for t in TIERS if s >= cuts[t]), None)
        if chosen is None:
            continue
        c.update_batch(s, [ContinuationFeedback(chosen, 0.05, "executed")], ("fm0", n))
    assert not c.valid(7)[0] and not c.valid(7)[1]  # never selected -> never filled
    assert c.cut(7, delta)[0] <= theta0  # spill-over into the left neighbour may loosen


def test_shadowed_tiers_are_reported():
    c = _curves()
    # tier 7 (cheapest) gets low d, tier 6 gets higher d at the same scores
    rows = [(float(s), [0.1, 0.9, 0.05]) for s in np.linspace(0, 1, 300)]
    _feed(c, rows)
    delta = 0.2
    shadow = c.shadowed(delta)
    assert 6 in shadow  # its cut is +inf or above the cheaper tier's
    assert 7 not in shadow


# ------------------------------------------------------------------
# judge dispatch
# ------------------------------------------------------------------


def _spec():
    tiers = tier_specs([0.875, 0.75, 0.5], SCHEDULE)
    mask = torch.zeros(D, dtype=torch.bool)
    mask[:7] = True
    return ContinuationSpec(
        tiers=tiers,
        scales={t.index: torch.ones(D) for t in tiers},
        masks={t.index: mask for t in tiers},
        h_exec=5,
        feedback_mode="fm1",
        schedule=SCHEDULE,
    )


def _judge(curves: OnlineRiskCurves, delta: float, registry=None):
    registry = registry or CurveRegistry()

    def factory():
        return curves

    key = registry.attach(yaml_id="arm", library_sha256="0" * 64, fingerprint="fp", factory=factory)
    return OnlineRitJudge(registry=registry, registry_key=key, spec=_spec(), delta=delta, yaml_id="arm"), registry, key


def _res(score, id="e1"):
    return SearchResultLite(id=id, score=score, checkpoint_id=CheckpointID.CP1)


def test_dispatch_matches_threshold_judge_order_and_never_full_hit():
    from openpi.cache.components.judge import ThresholdJudge

    c = _curves()
    rows = [(float(s), [0.9 - 0.8 * s, 0.6 - 0.55 * s, 0.3 - 0.29 * s]) for s in np.linspace(0, 1, 80)]
    _feed(c, rows)
    delta = 0.2
    judge, _, _ = _judge(c, delta)
    judge.on_episode_start(task_key="t", extra_metadata={"task_uid": "u", "attempt": 0})
    cuts = c.cuts(delta)
    ref = ThresholdJudge(
        cp1_threshold=math.inf,
        warm_tiers=[{"threshold": cuts[t], "start_t": st} for t, st in zip(TIERS, (0.875, 0.75, 0.5)) if math.isfinite(cuts[t])],
    )
    for s in np.linspace(0, 1, 101):
        mine = judge([_res(float(s))], CheckpointID.CP1, {})
        theirs = ref([_res(float(s))], CheckpointID.CP1, {})
        assert mine.hit_type is not HitType.FULL_HIT
        assert mine.hit_type == theirs.hit_type
        assert mine.start_t == theirs.start_t
        if mine.hit_type is HitType.MISS:
            assert mine.winner_id == "e1"


def test_miss_without_candidates_has_no_winner_and_no_snapshot():
    c = _curves()
    judge, _, _ = _judge(c, 0.3)
    judge.on_episode_start(task_key="t")
    r = judge([], CheckpointID.CP1, {})
    assert r.hit_type is HitType.MISS and r.winner_id is None
    assert judge.pending_snapshot is None
    diag = judge.record_continuation(CheckpointID.CP1, None, [])
    assert diag["fb"] == [] and diag["learned"] is False


def test_record_continuation_moves_the_cut_in_the_right_direction():
    c = _curves()
    _feed(c, [(float(s), [0.1] * 3) for s in np.linspace(0, 1, 300)], "w")
    delta = 0.2
    judge, reg, key = _judge(c, delta)
    judge.on_episode_start(task_key="t", extra_metadata={"task_uid": "u", "attempt": 0})
    theta_before = c.cut(7, delta)[0]
    for n in range(60):
        judge([_res(0.3)], CheckpointID.CP1, {})
        snap = judge.pending_snapshot
        assert snap is not None and snap.q_pre[7] is not None
        diag = judge.record_continuation(
            CheckpointID.CP1, snap, [ContinuationFeedback(7, 5.0, "executed")], fb_batch_size=0
        )
        assert diag["learned"] is True
        assert diag["q_pre"]["7"] == snap.q_pre[7]
    assert c.cut(7, delta)[0] > theta_before
    assert reg.describe(key)["n_updates"] == 360


def test_q_pre_is_decision_time_not_update_time():
    c = _curves()
    _feed(c, [(float(s), [0.1] * 3) for s in np.linspace(0, 1, 300)], "w")
    judge, _, _ = _judge(c, 0.5)
    judge.on_episode_start(task_key="t")
    judge([_res(0.5)], CheckpointID.CP1, {})
    snap = judge.pending_snapshot
    # mutate the shared state behind the judge's back
    _feed(c, [(0.5, [9.0] * 3)] * 40, "mut")
    assert c.query(7, 0.5)[0] > snap.q_pre[7]
    diag = judge.record_continuation(CheckpointID.CP1, snap, [ContinuationFeedback(7, 0.1, "executed")])
    assert diag["q_pre"]["7"] == snap.q_pre[7]
