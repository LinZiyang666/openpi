"""Unit tests for the round-2 tools (synthetic data; no capture roots, no inits >= 30)."""
import numpy as np
import pandas as pd
import pytest

from exp.offline_search.rounds.r09.explore_fable.round2.tools import corrector, methods


def test_phases_follow_executed_gripper_sign():
    # one episode of 8 decisions: open, open, open, close, close, close, open, open  (+1 = close)
    g = np.array([-1, -1, -1, 1, 1, 1, -1, -1], float)
    served = np.zeros((8, 10, 7), np.float32)
    served[:, :5, 6] = g[:, None]
    dec = pd.DataFrame(dict(episode_key=["e"] * 8, decision_seq=np.arange(8)))
    ph = corrector.episode_phases(dec, served)
    assert list(ph) == ["approach", "grasp", "grasp", "grasp", "carry", "release", "release", "release"]


def test_fit_head_recovers_linear_target_and_predict_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 12)).astype(np.float32)
    B = rng.standard_normal((12, 6))
    Y = (X @ B + 0.3).astype(np.float32)
    head = corrector.fit_head(X, Y, np.ones(400), n_rff=16, alpha=1e-3)
    pred = corrector.predict(head, X)
    assert pred.shape == (400, 6)
    assert np.sqrt(np.mean((pred - Y) ** 2)) < 0.05 * Y.std()
    assert methods._predict(head, X[:3]).shape == (3, 6)


def test_fit_head_episode_weights_are_normalized():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 4)).astype(np.float32)
    Y = rng.standard_normal((50, 2)).astype(np.float32)
    h1 = corrector.fit_head(X, Y, np.ones(50), n_rff=8)
    h2 = corrector.fit_head(X, Y, 7 * np.ones(50), n_rff=8)
    assert np.allclose(h1["coef"], h2["coef"], atol=1e-4)


def test_corrected_cache_parameter_validation():
    with pytest.raises(ValueError):
        methods.CorrectedCache(head_path="h.npz", blend=2.0, base_fit="b.pkl", lib="current", kref=5,
                               serving="anchor_tail", budget=1, gates="budget_only")
    m = methods.CorrectedCache(head_path="h.npz", blend=0.5, base_fit="b.pkl", lib="current", kref=5,
                               serving="anchor_tail", budget=1, gates="budget_only")
    assert m.blend == 0.5 and m.uses_nonlibrary_action


def test_grasp_miss_trigger_logic():
    g = methods.GraspMissCalls("c.pkl", {}, empty_aperture=0.001, p_uniform=0.0, max_calls=2, hold_decisions=2, burst=2)

    class Q:
        pass

    q = Q()
    q.step = 3
    hist = np.zeros((3, 10, 7), np.float32)
    hist[:, :5, 6] = 1.0                       # close command held
    q.hist_a_exec = hist
    q.raw_state = np.array([0, 0, 0, 0, 0, 0, 0.0004, -0.0004])   # aperture .0004 < .001 -> empty
    assert g.empty_grasp(q) is True
    q.raw_state = np.array([0, 0, 0, 0, 0, 0, 0.01, -0.01])         # holding an object
    assert g.empty_grasp(q) is False
    hist[-1, :5, 6] = -1.0                                           # last decision opened -> no trigger
    q.raw_state = np.array([0, 0, 0, 0, 0, 0, 0.0004, -0.0004])
    assert g.empty_grasp(q) is False
    with pytest.raises(ValueError):
        methods.GraspMissCalls("c.pkl", {}, empty_aperture=0.001, p_uniform=1.5)


def test_grasp_miss_burst_and_cap():
    g = methods.GraspMissCalls("c.pkl", {}, empty_aperture=0.001, max_calls=1, burst=2)

    class Ep:
        init = 0

    class Q:
        task_id = 0
        episode = Ep()
        step = 5
        hist_a_exec = np.ones((5, 10, 7), np.float32)
        raw_state = np.array([0, 0, 0, 0, 0, 0, 0.0, 0.0])

    q = Q()
    p1, _, d1 = g._assignment(q)
    assert p1 == 1.0 and d1["grasp_miss_trigger"] == 1.0
    p2, _, d2 = g._assignment(q)
    assert p2 == 1.0 and d2["grasp_miss_trigger"] == 0.0 and d2["grasp_miss_burst"] == 1.0
    p3, _, d3 = g._assignment(q)
    assert p3 == 0.0 and d3["n_trigger"] == 1      # cap reached: no further triggers


def test_paired_outcomes_never_read_holdout_inits(tmp_path):
    import json
    from exp.offline_search.rounds.r09.explore_fable.round2.tools import paired_r2
    arm = tmp_path / "runs" / "arm" / "client"
    arm.mkdir(parents=True)
    rows = [dict(task_uid=f"arm:eval:{t}:{i}", accepted=True, status="done", success=(i % 2 == 0)) for t in range(2) for i in (0, 25, 35, 49)]
    (arm / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = paired_r2.outcomes(tmp_path, "arm")
    assert set(s.index.get_level_values("init")) == {0, 25}       # inits 35 and 49 are never aggregated
    assert len(s) == 4


def test_grasp_miss_calls2_long_carry_trigger():
    from exp.offline_search.rounds.r09.explore_fable.round2.tools import methods
    g = methods.GraspMissCalls2(long_carry_decisions=3, corrected_fit="c.pkl", corrected_kwargs={}, empty_aperture=0.001, max_calls=2, burst=1)

    class Ep:
        init = 0

    class Q:
        task_id = 0
        episode = Ep()
        step = 3
        hist_a_exec = np.ones((3, 10, 7), np.float32)                 # closed for 3 decisions
        raw_state = np.array([0, 0, 0, 0, 0, 0, 0.01, -0.01])          # holding an object (no empty grasp)

    q = Q()
    p, _, d = g._assignment(q)
    assert p == 1.0 and d["long_carry_trigger"] == 1.0 and d["grasp_miss_trigger"] == 0.0
    q.hist_a_exec = -np.ones((3, 10, 7), np.float32)                   # open: no trigger
    p, _, d = g._assignment(q)
    assert p == 0.0 and d["long_carry_trigger"] == 0.0


def _reanchor(**kw):
    from exp.offline_search.rounds.r09.explore_fable.round2.tools import methods
    base = dict(head_path="h.npz", blend=0.0, base_fit="b.pkl", lib="current", kref=5, serving="anchor_tail", budget=1, gates="budget_only")
    g = methods.GraspMissReanchor(norm_scale=[1] * 7, norm_shift=[0] * 7, **kw, **base)
    g.H = 10
    return g


def test_reanchor_flag_and_retreat_chunk():
    g = _reanchor(empty_aperture=0.001, hold_decisions=2, max_retries=1, closed_sign=1.0)
    g.H = 10
    g._n_retry, g._last_retry_step = 0, -10 ** 6

    class Q:
        step = 3
        hist_a_exec = np.zeros((3, 10, 32), np.float32)
        raw_state = np.array([0, 0, 0, 0, 0, 0, 0.0002, -0.0002])
        rs = np.zeros(32, np.float32)
        hist_rs = np.zeros((3, 32), np.float32)

    q = Q()
    q.hist_a_exec[:, :5, 6] = 1.0                      # closed for 3 decisions
    q.hist_a_exec[:, :5, :3] = 0.2                     # moved +0.2 per control
    assert g.flag(q) == "empty"
    ch = g.retreat_chunk(q, open_gripper=True)
    assert ch.shape == (10, 32) and np.allclose(ch[:10, :3], -0.2) and np.all(ch[:, 6] == -1.0)
    g._n_retry = 1
    assert g.flag(q) is None                           # cap reached
    g2 = _reanchor(empty_aperture=0.001, closed_sign=-1.0)
    g2._n_retry, g2._last_retry_step = 0, -10 ** 6
    q.hist_a_exec[:, :5, 6] = -1.0                     # GR00T: closed is -1
    assert g2.flag(q) == "empty"
    assert np.all(g2.retreat_chunk(q, True)[:, 6] == 1.0)


def test_reanchor_union_triggers():
    g = _reanchor(empty_aperture=0.001, long_carry_decisions=3, stall_decisions=2, stall_path=0.05, closed_sign=1.0)
    g._n_retry, g._last_retry_step = 0, -10 ** 6

    class Q:
        step = 3
        hist_a_exec = np.ones((3, 10, 32), np.float32)
        raw_state = np.array([0, 0, 0, 0, 0, 0, 0.01, -0.01])      # holding
        rs = np.zeros(32, np.float32)
        hist_rs = np.zeros((3, 32), np.float32)

    q = Q()
    assert g.flag(q) == "carry"
    q.hist_a_exec[:, :5, 6] = -1.0                                 # open, not moving -> stall
    assert g.flag(q) == "stall"
    q.hist_rs[:, 0] = np.array([0, 1, 2], np.float32)              # moving -> nothing
    q.rs[0] = 3.0
    assert g.flag(q) is None
    with pytest.raises(ValueError):
        _reanchor(closed_sign=0.5)


def test_signed_trigger_handles_groot_convention():
    from exp.offline_search.rounds.r09.explore_fable.round2.tools import methods
    g = methods.GraspMissCallsSigned(closed_sign=-1.0, long_carry_decisions=30, corrected_fit="c.pkl", corrected_kwargs={}, empty_aperture=0.001)

    class Q:
        step = 2
        hist_a_exec = -np.ones((2, 10, 7), np.float32)                 # GR00T closed = -1
        raw_state = np.array([0, 0, 0, 0, 0, 0, 0.0002, -0.0002])

    assert g.empty_grasp(Q()) is True
    g1 = methods.GraspMissCallsSigned(closed_sign=1.0, corrected_fit="c.pkl", corrected_kwargs={}, empty_aperture=0.001)
    assert g1.empty_grasp(Q()) is False
