"""Round-5 gate tests: gate logic on a stub parent, veto lifting, gates-off identity, and the real frozen stacks."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exp.offline_search.closed_loop.blind import LookReason
from exp.offline_search.harness import api
from exp.offline_search.rounds.r09.explore_opus.round5.methods import (
    GATE_BUDGET, GATE_NONE, GATE_PACE, GatedNpGraspEsc3, GatedNpGraspStackGroot3, _CallGate, committed_calls)

R3C = Path("/home/weiland/trace_runs/os_closed_loop/r09_fable_r3c")


class _Parent:
    """Stub of a fitted stack: returns a scripted verdict, blind_step honours the no-progress veto like K1."""

    def __init__(self):
        self.base = SimpleNamespace(lib_step=np.arange(100))     # library row r has library step r
        self._s = {"flag": []}
        self._noprog_span = 0
        self.script = {}

    def reset(self, episode):
        self._s = {"flag": []}

    def query(self, q):
        reason, top1 = self.script.get(int(q.step), (0.0, 0))
        self._s["flag"].append(int(reason != 0))
        ex = dict(os_force_miss=float(reason != 0), os_reason=float(reason), os_flags=8.0 if reason == 4 else 0.0)
        return api.Result(np.array([top1, 1, 2]), np.zeros(3), 0.0, action=np.zeros((50, 32), np.float32), library="current",
                          extras=ex)

    def blind_step(self, bq):
        if self._noprog_span > 0:
            return LookReason(8, "noprog_span")
        return "BLIND"


class _Gated(_CallGate, _Parent):
    def __init__(self, pace=None, budget=None, force=()):
        _Parent.__init__(self)
        self._cg_init(pace, budget, force)

    def reset(self, episode):
        super().reset(episode)
        self._cg_reset(episode)

    def query(self, q):
        return self._cg_apply(super().query(q), q)


EP = SimpleNamespace(uid="x:eval:0:0", task_id=0, init=0)


def _q(step, hits, vision):
    return SimpleNamespace(step=step, episode=EP, task_id=0, hist_hit=np.array(hits[:step], np.int64),
                           hist_has_vision=np.array(vision[:step], bool))


def test_committed_calls_counts_vision_misses_only():
    q = _q(5, [1, 0, 1, 0, 1], [True, True, False, True, False])
    assert committed_calls(q) == 2
    assert committed_calls(_q(0, [], [])) == 0


def test_pace_gate_drops_only_on_pace_no_progress():
    g = _Gated(pace=2)
    g.reset(EP)
    # step 10: no-progress verdict, top1 row 9 -> lag 1 (gated); step 12: row 5 -> lag 7 (kept);
    # step 14: escalation verdict at lag 0 (never touched by gate P)
    g.script = {10: (4.0, 9), 12: (4.0, 5), 14: (91.0, 14)}
    hits, vis = [1] * 20, [True] * 20
    r = g.query(_q(10, hits, vis))
    assert r.extras["os_force_miss"] == 0.0 and r.extras["r9o5_gate"] == GATE_PACE and r.extras["r9o5_lag"] == 1.0
    assert r.extras["r9o5_gated_reason"] == 4.0 and g._s["flag"][-1] == 0
    r = g.query(_q(12, hits, vis))
    assert r.extras["os_force_miss"] == 1.0 and r.extras["os_reason"] == 4.0 and r.extras["r9o5_gate"] == GATE_NONE
    r = g.query(_q(14, hits, vis))
    assert r.extras["os_force_miss"] == 1.0 and r.extras["os_reason"] == 91.0


def test_budget_gate_counts_committed_calls_of_any_reason():
    g = _Gated(budget=2)
    g.reset(EP)
    g.script = {6: (91.0, 0), 8: (4.0, 0), 10: (93.0, 0)}
    hits = [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]      # two committed calls (steps 1, 4) before step 6
    vis = [True] * 11
    for s in (6, 8, 10):
        r = g.query(_q(s, hits, vis))
        assert r.extras["os_force_miss"] == 0.0 and r.extras["r9o5_gate"] == GATE_BUDGET
        assert r.extras["r9o5_calls_before"] == 2.0
    g2 = _Gated(budget=3)
    g2.reset(EP)
    g2.script = {6: (91.0, 0)}
    assert g2.query(_q(6, hits, vis)).extras["os_force_miss"] == 1.0


def test_veto_lifted_only_right_after_a_gated_look_and_restored():
    g = _Gated(pace=2)
    g.reset(EP)
    g.script = {10: (4.0, 9)}
    g.query(_q(10, [1] * 20, [True] * 20))
    g._noprog_span = 2
    assert g.blind_step(_q(11, [1] * 20, [True] * 20)) == "BLIND"
    assert g._noprog_span == 2                       # statistic untouched
    assert isinstance(g.blind_step(_q(13, [1] * 20, [True] * 20)), LookReason)
    g.script = {12: (4.0, 2)}                        # lag 10: call kept -> no veto lifting
    g.query(_q(12, [1] * 20, [True] * 20))
    assert isinstance(g.blind_step(_q(13, [1] * 20, [True] * 20)), LookReason)


def test_veto_lifted_for_rest_of_episode_after_budget():
    g = _Gated(budget=1)
    g.reset(EP)
    g._noprog_span = 4
    hits, vis = [1, 0, 1, 1, 1, 1, 1, 1], [True] * 8
    assert g.blind_step(_q(5, hits, vis)) == "BLIND"          # one committed call >= budget 1: pure cache
    assert g._noprog_span == 4
    assert isinstance(g.blind_step(_q(1, [1], [True])), LookReason)   # no call yet: veto stays


def test_gates_off_identity_and_new_episode_reset():
    g = _Gated()
    p = _Parent()
    g.reset(EP)
    p.reset(EP)
    for s, v in {4: (4.0, 1), 6: (91.0, 0), 8: (0.0, 3)}.items():
        g.script = p.script = {s: v}
        a, b = g.query(_q(s, [1] * 10, [True] * 10)), p.query(_q(s, [1] * 10, [True] * 10))
        for k, val in b.extras.items():
            assert a.extras[k] == val
        assert set(a.extras) - set(b.extras) == {"r9o5_lag", "r9o5_calls_before", "r9o5_gate"}
    g2 = _Gated(pace=5)
    g2.reset(EP)
    g2.script = {4: (4.0, 4)}
    g2.query(_q(4, [1] * 10, [True] * 10))
    assert g2.cg_quiet_step == 4
    other = SimpleNamespace(uid="x:eval:0:1", task_id=0, init=1)
    g2._noprog_span = 3
    q = _q(5, [1] * 10, [True] * 10)
    q.episode = other
    assert isinstance(g2.blind_step(q), LookReason)   # quiet state does not leak into another episode


def test_force_noprog_exercises_gate():
    g = _Gated(pace=1000, force=(4,))
    g.reset(EP)
    g.script = {4: (0.0, 4)}
    r = g.query(_q(4, [1] * 10, [True] * 10))
    assert r.extras["r9o5_gated_reason"] == 4.0 and r.extras["os_force_miss"] == 0.0
    g = _Gated(force=(4,))
    g.reset(EP)
    g.script = {4: (0.0, 4)}
    r = g.query(_q(4, [1] * 10, [True] * 10))
    assert r.extras["os_force_miss"] == 1.0 and r.extras["os_reason"] == 4.0


def test_bad_thresholds_rejected():
    with pytest.raises(ValueError):
        _Gated(pace=float("nan"))
    with pytest.raises(ValueError):
        _Gated(budget=2.5)
    with pytest.raises(ValueError):
        _Gated(budget=-1)


@pytest.mark.parametrize("name,cls", [("r9f3c_groot_l10_50_np_corr05", GatedNpGraspStackGroot3),
                                      ("r9f3c_pi05_l10_50_np_corr05_esc", GatedNpGraspEsc3)])
def test_real_stack_fit_matches_fable_artifact(name, cls):
    reg = R3C / "r3c_kwargs.json"
    if not reg.exists():
        pytest.skip("r3c registry missing")
    spec = json.loads(reg.read_text())[name]
    m = cls(**spec["kwargs"], pace_lag_max=2, call_budget=20)
    m.prof = api.NULL_PROFILER
    m.fit(None, SimpleNamespace(cell=spec["cell"]))
    with open(R3C / "fits" / f"{name}.pkl", "rb") as f:
        src = pickle.load(f)["method"]
    assert int(m.burst) == int(src.burst) == 0 and m.gm_burst == src.gm_burst and m.gm_max_calls == src.gm_max_calls == 0
    assert type(m.base).__name__ == type(src.base).__name__ == "CorrectedCacheJ" and m.base.blend == src.base.blend == 0.5
    np.testing.assert_array_equal(m.base.lib_step, src.base.lib_step)
    np.testing.assert_array_equal(np.asarray(m.base.act), np.asarray(src.base.act))
    for k in ("noprog_n", "prog_eps", "disabled_guards", "progress_guard", "lag_threshold", "deadline"):
        if hasattr(src, k):
            assert getattr(m, k) == getattr(src, k), k
    assert m.cg_pace_lag_max == 2.0 and m.cg_budget == 20 and m.cg_force_np == ()
    assert m.name.startswith("R9O5_gate_P2_C20__")


def test_clash_detected():
    g = _Gated(pace=2)
    keep = {k: getattr(g, k) for k in g._CG_OWN}
    g.cg_budget = 7                                    # as if a frozen judge field had overwritten it
    with pytest.raises(api.ContractError):
        g._cg_check_clash(keep)
