"""Unit tests for opus round-2 tools and serving candidates.

Run: taskset -c 2-9 env OMP_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q \
        exp/offline_search/rounds/r09/explore_opus/round2/tools/tests
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from exp.offline_search.harness import api
from exp.offline_search.rounds.r09.explore_opus.round2.tools import common, replicates, triggers
from exp.offline_search.rounds.r09.explore_opus.round2 import methods

STORE = Path("/home/weiland/trace_runs/offline_search_store")
P_FIT = Path("/home/weiland/trace_runs/os_closed_loop/r05_ptail/fits/r5t_p_l10_50_tail1uc.pkl")
NP_FIT = Path("/home/weiland/trace_runs/os_closed_loop/r08_abl/fits/r8abl_onlynp_p_l10_50.pkl")
BASE_KW = {"lib": "current", "kref": 5, "serving": "anchor_tail", "budget": 1, "gates": "budget_only"}


# ---------------------------------------------------------------- rule 1 plumbing
def test_discovery_filter_drops_holdout_before_decoding(tmp_path):
    p = tmp_path / "j.jsonl"
    rows = [{"task_uid": "a:eval:3:5", "x": 1}, {"task_uid": "a:eval:3:35", "x": 2},
            {"task_uid": "a:eval:9:29", "x": 3}, {"task_uid": "a:eval:0:30", "x": 4}]
    # the init-35 line is deliberately invalid JSON after the identity: it must never be decoded
    lines = [json.dumps(rows[0]), '{"task_uid": "a:eval:3:35", BROKEN', json.dumps(rows[2]), json.dumps(rows[3])]
    p.write_text("\n".join(lines) + "\n")
    got = list(common.iter_jsonl_discovery(p))
    assert [r["x"] for r in got] == [1, 3]


def test_forbidden_roots_refused():
    with pytest.raises(PermissionError):
        common.check_root(common.RUNS / "r09_astra_holdout" / "runs")
    with pytest.raises(PermissionError):
        common.check_root(common.RUNS / "r09_astra_holdout2")
    for name in ("r09_holdout_opus", "r09_holdout_opus_g", "r09_holdout_fable", "r09_holdout_astra_g", "r09_holdout_anything"):
        with pytest.raises(PermissionError):
            common.check_root(common.RUNS / name / "runs")


def test_parse_uid():
    assert common.parse_uid("r8_pi05_l10_50_A:eval:7:12") == (7, 12)
    with pytest.raises(ValueError):
        common.parse_uid("garbage")


# ---------------------------------------------------------------- replicate decomposition
def test_churn_counts_and_betabin():
    M = pd.DataFrame({"r1": [1, 1, 0, 0, 1], "r2": [1, 0, 0, 1, 1], "r3": [1, 1, 0, 1, 1]}, dtype=float)
    c = replicates.churn(M)
    d = {(x["a"], x["b"]): (x["a_only"], x["b_only"]) for x in c}
    assert d[("r1", "r2")] == (1, 1)
    assert d[("r1", "r3")] == (0, 1)
    a, b = replicates.betabin_fit(np.array([0, 0, 0, 7, 7, 7, 7, 3]), np.array([7] * 8))
    assert a > 0 and b > 0 and a / (a + b) > 0.5          # mass mostly at success, U-shaped (a, b < 1)
    assert a < 1 and b < 1


def test_family_mapping():
    r = SimpleNamespace(method="x:BlindAWM", fit="r5t_p_l10_50_tail1uc.pkl", flags="--os-no-shadow-native --os-blind", arm="a")
    assert replicates.family(r) == "cache"
    r.flags = "--os-no-shadow-native --os-blind --os-judge periodic:5"
    assert replicates.family(r) is None
    assert replicates.family(SimpleNamespace(method="m:PolicyEveryTen", fit=None, flags="", arm="b")) == "policy10"


# ---------------------------------------------------------------- trouble signals and simulator
def _cat(steps, eplen=50):
    n = len(steps)
    return pd.DataFrame({"step": steps, "ep_len": [eplen] * n,
                         "progress": [s / (eplen - 1) for s in steps]}, index=pd.Index(range(n), name="row"))


def _ep(seq, rows, success, n_dec, cost=None):
    rows = np.asarray(rows)[:, None]
    return triggers.Episode("t", 0, 0, success, n_dec, np.asarray(seq), rows, np.ones_like(rows, float),
                            np.full(n_dec, 0.076) if cost is None else cost)


def test_signals_lag_noprogress_eos():
    cat = _cat(list(range(50)))
    # on pace for 3 fresh looks, then stuck at library step 20 while time advances
    ep = _ep([0, 2, 4, 20, 22, 24, 26], [0, 2, 4, 20, 20, 20, 20], False, 30)
    s = triggers.signals(ep, cat)
    assert list(s["lag"]) == [0, 0, 0, 0, 2, 4, 6]
    assert list(s["np"]) == [0, 0, 0, 0, 2, 4, 6]
    assert triggers.first_trigger(ep, s, "lag", 4) == 24
    assert triggers.first_trigger(ep, s, "lag", 4, deadline=22) is None
    ep2 = _ep([0, 2, 4], [47, 48, 49], True, 5)
    assert list(triggers.signals(ep2, cat, theta=0.9)["eos"]) == [1, 2, 3]


def test_simulator_accounting():
    v, m = common.PRICE["pi05"]
    # never triggered: exact cache SR/IR
    eps = [_ep([0], [0], True, 50, np.r_[[v, 0] * 25]), _ep([0], [0], False, 104, np.r_[[v, 0] * 52])]
    s = triggers.simulate(eps, [None, None], r=0.5, model="pi05", suite="l10")
    assert s["sr"] == 0.5 and abs(s["ir"] - v / 2) < 1e-9 and s["trig_rate"] == 0
    # triggered at decision 0 with certain rescue: pure policy pattern, IR = (v+m)/2 = .5
    s = triggers.simulate([eps[1]], [0], r=1.0, model="pi05", suite="l10", rescue_dec=40)
    assert s["sr"] == 1.0 and abs(s["ir"] - 0.5) < 1e-9
    # a triggered episode the cache would have recovered uses success_r (default: r, conservative)
    s = triggers.simulate([eps[0]], [10], r=0.0, model="pi05", suite="l10", success_r=0.9)
    assert abs(s["sr"] - 0.9) < 1e-9 and s["false_alarm_rate"] == 1.0
    s = triggers.simulate([eps[0]], [10], r=0.4, model="pi05", suite="l10")
    assert abs(s["sr"] - 0.4) < 1e-9


# ---------------------------------------------------------------- serving candidates
def _lib():
    d = STORE / "library" / "pi05_l10" / "current"
    return (np.load(d / "key_v0.npy", mmap_mode="r"), np.load(d / "key_v1.npy", mmap_mode="r"),
            np.load(d / "rs.npy", mmap_mode="r"))


def _q(k0, k1, rs, row, step, ep, hist):
    return SimpleNamespace(key_v0=k0[row], key_v1=k1[row], rs=rs[row], task_id=ep.task_id, step=step, episode=ep,
                           prev_hit=True if step else None, hist_key_v0=np.array(hist[0] + [k0[row]]),
                           hist_key_v1=np.array(hist[1] + [k1[row]]))


@pytest.mark.skipif(not P_FIT.exists(), reason="prefit artifact not available")
def test_escalate_calls_real_library():
    from exp.offline_search.closed_loop.plugin import clone_method
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
    m = methods.EscalateCalls(lag_threshold=12, deadline=80, base_kwargs=BASE_KW, base_fit=str(P_FIT))
    m.prof = api.NULL_PROFILER
    m.fit(None, SimpleNamespace(cell="pi05_l10_cache"))
    with open(P_FIT, "rb") as f:
        base = FitUnpickler(f).load()["method"]
    base.prof = api.NULL_PROFILER
    k0, k1, rs = _lib()
    r = int(m.base.tasks[0].rows[0])
    ep = SimpleNamespace(uid="opus_test:eval:0:0", task_id=0, init=0)
    hist = ([], [])
    base.reset(ep)
    seen = []
    for step, row in [(0, r), (2, r + 1), (4, r + 2), (30, r + 3), (32, r + 30), (34, r + 32)]:
        q = _q(k0, k1, rs, row, step, ep, hist)
        res = m.query(q)
        ref = base.query(_q(k0, k1, rs, row, step, ep, hist))
        hist[0].append(k0[row]); hist[1].append(k1[row])
        lag = step - int(m.base.lib_step[int(res.topk[0])])
        assert res.extras["r9o_lag"] == lag
        seen.append(res.extras["os_force_miss"])
        np.testing.assert_array_equal(res.action, ref.action)      # retrieval/synthesis untouched
        np.testing.assert_array_equal(res.topk, ref.topk)
    first = next(i for i, f in enumerate(seen) if f == 1.0)
    assert seen[:first] == [0.0] * first and all(f == 1.0 for f in seen[first:])   # persistent
    # a new episode starts un-escalated
    ep2 = SimpleNamespace(uid="opus_test:eval:0:1", task_id=0, init=1)
    assert m.query(_q(k0, k1, rs, r, 0, ep2, ([], []))).extras["os_force_miss"] == 0.0
    # deadline: a large lag after the deadline never escalates
    m2 = methods.EscalateCalls(lag_threshold=12, deadline=20, base_kwargs=BASE_KW, base_fit=str(P_FIT))
    m2.prof = api.NULL_PROFILER
    m2.fit(None, SimpleNamespace(cell="pi05_l10_cache"))
    ep3 = SimpleNamespace(uid="opus_test:eval:0:2", task_id=0, init=2)
    assert m2.query(_q(k0, k1, rs, r, 0, ep3, ([], []))).extras["os_force_miss"] == 0.0
    assert m2.query(_q(k0, k1, rs, r + 1, 30, ep3, ([k0[r]], [k1[r]]))).extras["os_force_miss"] == 0.0
    # clones isolate per-episode escalation state; pickling round-trips
    c, mode = clone_method(m, strict=True)
    assert mode == "deepcopy_shared_arrays"
    c._esc_step = 5
    assert m._esc_step is None
    assert pickle.loads(pickle.dumps(m)).lag_threshold == 12.0


class _Canned:
    """Stand-in parent: returns a canned B Result (used to test the escalation hook in isolation)."""

    def query(self, q):
        return self._canned

    def reset(self, episode):
        pass


class _HookUnderTest(methods._EscalateJudge, _Canned):
    pass


def test_escalation_hook_forces_miss_only_after_trigger():
    m = object.__new__(_HookUnderTest)
    m._esc_init(12, 80)
    m.base = SimpleNamespace(lib_step=np.arange(100))
    m._s = dict(flag=[0], burst_end=0, ret_end=0)
    ep = SimpleNamespace(uid="u:eval:0:0")

    def res(top1, flags=0.0):
        return api.Result(np.array([top1] + list(range(15))), np.zeros(16), .5, action=np.zeros((10, 32)),
                          extras=dict(os_flags=flags, os_reason=0.0, os_force_miss=0.0))
    m._canned = res(10)
    out = m.query(SimpleNamespace(step=12, episode=ep))           # lag 2: unchanged verdict
    assert out.extras["os_force_miss"] == 0.0 and m._s["flag"] == [0]
    m._canned = res(10)
    out = m.query(SimpleNamespace(step=24, episode=ep))           # lag 14 >= 12: forced MISS
    assert out.extras["os_force_miss"] == 1.0 and out.extras["os_reason"] == methods.ESC_REASON
    assert m._s["flag"] == [1]
    m._s["flag"] = [0]
    m._canned = res(40)
    out = m.query(SimpleNamespace(step=40, episode=ep))           # lag 0 later: still escalated
    assert out.extras["os_force_miss"] == 1.0 and out.extras["r9o_esc_step"] == 24.0
    m._canned = res(1)
    out = m.query(SimpleNamespace(step=90, episode=SimpleNamespace(uid="u:eval:0:1")))   # new episode, after deadline
    assert out.extras["os_force_miss"] == 0.0 and out.extras["r9o_escalated"] == 0.0


@pytest.mark.skipif(not NP_FIT.exists(), reason="only-no-progress artifact not available")
def test_escalate_onlynp_fit_from_frozen_artifact():
    from exp.offline_search.closed_loop.plugin import clone_method
    m = methods.EscalateOnlyNP(onlynp_fit=str(NP_FIT), lag_threshold=12, deadline=80)
    m.prof = api.NULL_PROFILER
    m.fit(None, SimpleNamespace(cell="pi05_l10_cache"))
    api.check_method_attrs(m)
    assert m.disabled_guards == ("stuck", "terminal", "overtime") and m.name.startswith("R9O_ESC_onlynp")
    assert m.base.lib_step.shape[0] == 2640
    clone_method(m, strict=True)
    with pytest.raises(ValueError):
        methods.EscalateOnlyNP(onlynp_fit=str(NP_FIT), lag_threshold=12, deadline=80).fit(
            None, SimpleNamespace(cell="groot_l10_cache"))


def test_window_bounds_the_takeover():
    m = object.__new__(_HookUnderTest)
    m._esc_init(12, 80, window=4)
    m.base = SimpleNamespace(lib_step=np.arange(100))
    ep = SimpleNamespace(uid="u:eval:0:0")
    calls = [m._esc_update(SimpleNamespace(step=s, episode=ep), top1) for s, top1 in
             [(10, 10), (20, 2), (22, 22), (24, 24), (26, 0), (28, 0)]]
    assert calls == [False, True, True, False, False, False]   # window 4 decisions (steps 20-23) from the trigger at 20, never re-armed


def test_parameter_validation():
    with pytest.raises(ValueError):
        methods._check(12, 80, window=0)
    with pytest.raises(ValueError):
        methods._check(0, 80)
    with pytest.raises(ValueError):
        methods._check(12, -1)
    with pytest.raises(ValueError):
        methods._check(12, 80.5)


# ---------------------------------------------------------------- screen analysis on a synthetic run root
def _write_arm(run, arm, episodes):
    d = run / "runs" / arm / "client"
    d.mkdir(parents=True)
    lines = []
    for (t, i, success, kinds) in episodes:
        uid = f"{arm}:eval:{t}:{i}"
        for j, kind in enumerate(kinds):
            judge = {"b": "blind", "h": "guard_only", "m": "force:82", "n": "force:4"}[kind]
            lines.append(dict(task_uid=uid, step_idx=5 * j, hit_type="MISS" if kind in "mn" else "FULL_HIT",
                              searched=kind in "hmn", success=success, accepted=True,
                              factor_outputs={"osplug": {"os_judge": judge, "os_row": -1}}))
        lines.append(dict(_kind="client_timing", task_uid=uid, success=success, accepted=True, steps=5 * len(kinds)))
    (d / "per_step.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))


def test_screen_analysis_synthetic(tmp_path):
    from exp.offline_search.rounds.r09.explore_opus.round2.tools import screen_analysis as sa
    run = tmp_path / "r09_opus_test"
    _write_arm(run, "r9o_pi05_l10_50_cache", [(0, 20, True, "hbhb"), (0, 21, False, "hbhbhbhb"), (1, 35, True, "hb")])
    _write_arm(run, "r9o_pi05_l10_50_esc", [(0, 20, True, "hbhb"), (0, 21, True, "hbhbmbmb"), (1, 35, False, "hb")])
    c = sa.arm_summary(run, "r9o_pi05_l10_50_cache", "pi05")
    e = sa.arm_summary(run, "r9o_pi05_l10_50_esc", "pi05")
    assert len(c) == 2 and len(e) == 2                         # the init-35 pair is never read
    row = e[e.init == 21].iloc[0]
    assert row.esc_step == 4 and row.calls == 2 and row.success
    assert abs(row.cost - (0.152 * 4 + 0.848 * 2)) < 1e-9
    plus, minus, p = sa.mcnemar(e.sort_values("init").success.values, c.sort_values("init").success.values)
    assert (plus, minus) == (1, 0)


def test_screen_self_recovery_from_server_logs(tmp_path):
    """GR00T-style base arm: client logs carry no rows; the rule is evaluated from server decision logs."""
    from exp.offline_search.rounds.r09.explore_opus.round2.tools import screen_analysis as sa
    run = tmp_path / "r09_opus_test2"
    arm = "r9o_groot_l10_50_cache"
    # client: two episodes (init 20 fails, init 21 succeeds), no rows
    _write_arm(run, arm, [(0, 20, False, "hb" * 20), (0, 21, True, "hb" * 5), (0, 31, False, "hb")])
    srv = run / "runs" / arm / "server_23000"
    srv.mkdir(parents=True)
    step = triggers.catalog(("groot", "l10", 50)).step
    r0 = int(step[step == 0].index[0])          # a library row at demo step 0
    recs = []
    for d in range(0, 40, 2):                   # init 20 stays at demo step 0 -> lag grows to 38 (> 12 by d=12)
        recs.append(dict(ev="dec", uid=f"{arm}:eval:0:20", attempt=1, step=d, vision=True, top1=r0))
    for d in range(0, 10, 2):
        recs.append(dict(ev="dec", uid=f"{arm}:eval:0:21", attempt=1, step=d, vision=True, top1=r0))
    recs.append(dict(ev="dec", uid=f"{arm}:eval:0:31", attempt=1, step=30, vision=True, top1=r0))
    (srv / "decisions_x.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    res = sa.self_recovery(run, arm, ("groot", "l10", 50))
    assert res == dict(n=1, recovered=0.0)      # only init 20 crosses the rule; init 31 is never read
