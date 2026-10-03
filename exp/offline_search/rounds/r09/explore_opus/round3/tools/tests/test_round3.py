"""Unit tests for opus round-3 tools and the homing escalation."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exp.offline_search.harness import api
from exp.offline_search.rounds.r09.explore_opus.round3 import methods as m3

HERE = Path(__file__).resolve().parents[2]
P_FIT = Path("/home/weiland/trace_runs/os_closed_loop/r05_ptail/fits/r5t_p_l10_50_tail1uc.pkl")
G_FIT = Path("/home/weiland/trace_runs/os_closed_loop/r05_x/fits/r5x_g_l10_50_tail1u.pkl")
BASE_KW = {"lib": "current", "kref": 5, "serving": "anchor_tail", "budget": 1, "gates": "budget_only"}
STORE = Path("/home/weiland/trace_runs/offline_search_store")


def test_waypoints_lift_then_across_then_down():
    w = m3.waypoints([0.1, 0.2, 0.9], [0.0, 0.0, 1.0], 0.05)
    np.testing.assert_allclose(w[0], [0.1, 0.2, 1.05])
    np.testing.assert_allclose(w[1], [0.0, 0.0, 1.05])
    np.testing.assert_allclose(w[2], [0.0, 0.0, 1.0])


def test_homing_command_saturates_and_points_at_target():
    M_inv = np.linalg.inv(np.eye(3) * 0.01)
    u = m3.homing_command([0, 0, 0], [0.05, 0, 0], M_inv, 0.8)   # 5 cm in 10 controls -> .5 per control
    np.testing.assert_allclose(u, [0.5, 0, 0])
    u = m3.homing_command([0, 0, 0], [1.0, 0.5, 0], M_inv, 0.8)  # far: saturate, keep direction
    assert np.max(np.abs(u)) == pytest.approx(0.8) and u[0] / u[1] == pytest.approx(2.0)


class _Base:
    def __init__(self):
        self.lib_step = np.arange(200)
        self._anchor = None


def _machine(window=24, home_max=8):
    m = object.__new__(m3.HomingEscalation)
    m.lag_threshold, m.deadline, m.window, m.home_max, m.home_tol, m.lift, m.umax = 12.0, 80, window, home_max, .03, .05, .8
    m.base = _Base()
    m._clear()
    return m


def _q(step, pos, uid="u:eval:0:0"):
    raw = np.r_[pos, 0, 0, 0, .04, -.04]
    return SimpleNamespace(step=step, raw_state=raw, episode=SimpleNamespace(uid=uid))


def test_state_machine_trigger_homing_window_post():
    m = _machine(window=4)
    home = np.array([0.0, 0.0, 1.0])
    assert m._advance(_q(0, home), 0) == "cache"
    assert m._advance(_q(20, [0.3, 0.0, 0.9]), 18) == "cache"             # lag 2
    assert m._advance(_q(30, [0.3, 0.0, 0.9]), 10) == "homing"            # lag 20 -> trigger
    assert m._t_trig == 30 and m._wp == 0
    assert m._advance(_q(32, [0.3, 0.0, 1.05]), 10) == "homing"           # first waypoint reached
    assert m._wp == 1
    assert m._advance(_q(34, [0.0, 0.0, 1.05]), 10) == "homing" and m._wp == 2
    assert m._advance(_q(36, [0.0, 0.0, 1.01]), 10) == "window"           # home reached -> policy window
    assert m._advance(_q(38, [0.0, 0.0, 1.0]), 10) == "window"
    assert m._advance(_q(40, [0.0, 0.0, 1.0]), 10) == "post"              # 4 decisions after window start
    assert m._advance(_q(60, [0.3, 0.0, 0.9]), 0) == "post"               # never re-armed
    assert m._advance(_q(0, home, uid="u:eval:0:1"), 0) == "cache"         # new episode


def test_homing_gives_up_after_home_max_and_window_zero_goes_to_cache():
    m = _machine(window=0, home_max=2)
    m._advance(_q(0, [0, 0, 1.0]), 0)
    assert m._advance(_q(20, [0.5, 0.5, 0.5]), 0) == "homing"
    assert m._advance(_q(22, [0.5, 0.5, 0.5]), 0) == "homing"
    assert m._advance(_q(24, [0.5, 0.5, 0.5]), 0) == "post"               # 2 fresh homing decisions, no window


def test_deadline_blocks_late_trigger():
    m = _machine()
    m._advance(_q(0, [0, 0, 1.0]), 0)
    assert m._advance(_q(90, [0.3, 0, 0.9]), 0) == "cache"


def test_missing_start_pose_is_an_error():
    m = _machine()
    with pytest.raises(api.ContractError):
        m._advance(_q(10, [0, 0, 1.0]), 0)


@pytest.mark.parametrize("model,fit,cell", [("pi05", P_FIT, "pi05_l10_cache"), ("groot", G_FIT, "groot_l10_cache")])
def test_real_artifact_homing_chunk_and_identity(model, fit, cell):
    amap = HERE / "out" / f"action_map_{model}.json"
    if not (fit.exists() and amap.exists()):
        pytest.skip("artifacts not available")
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
    m = m3.HomingEscalation(lag_threshold=12, deadline=80, window=24, action_map=str(amap), base_kwargs=BASE_KW, base_fit=str(fit))
    m.prof = api.NULL_PROFILER
    m.fit(None, SimpleNamespace(cell=cell))
    with open(fit, "rb") as f:
        base = FitUnpickler(f).load()["method"]
    base.prof = api.NULL_PROFILER
    lib = STORE / "library" / f"{model}_l10" / "current"
    k0, k1, rs = (np.load(lib / n, mmap_mode="r") for n in ("key_v0.npy", "key_v1.npy", "rs.npy"))
    r = int(m.base.tasks[0].rows[0])
    ep = SimpleNamespace(uid="t:eval:0:0", task_id=0, init=0)
    base.reset(ep)
    hist = ([], [])
    home = np.array([0.0, 0.0, 1.0])
    for step, row, pos in [(0, r, home), (2, r + 1, home + [0, 0, .01]), (30, r + 2, [0.25, 0.1, 0.9]), (32, r + 3, [0.25, 0.1, 0.97])]:
        q = SimpleNamespace(key_v0=k0[row], key_v1=k1[row], rs=rs[row], task_id=0, step=step, episode=ep,
                            prev_hit=True if step else None, hist_key_v0=np.array(hist[0] + [k0[row]]),
                            hist_key_v1=np.array(hist[1] + [k1[row]]), raw_state=np.r_[pos, 0, 0, 0, .04, -.04])
        res = m.query(q)
        ref = base.query(SimpleNamespace(**{k: v for k, v in vars(q).items() if k != "raw_state"}))
        hist[0].append(k0[row]); hist[1].append(k1[row])
        if step < 30:
            np.testing.assert_array_equal(res.action, ref.action)          # untouched before the trigger
            assert res.extras["r9o3_mode"] == 0 and res.extras["os_force_miss"] == 0
        else:
            assert res.extras["r9o3_mode"] == 1 and res.extras["os_force_miss"] == 0
            assert res.action.shape == ref.action.shape and np.isfinite(res.action).all()
            wire = m._a * res.action[0, :6] + m._b
            assert np.all(np.abs(wire[3:]) < 1e-6)                            # no rotation
            assert wire[2] > 0                                                 # lifting first (z below lift height)
            assert res.action[0, 6] == (-1.0 if model == "pi05" else 1.0)     # open gripper in model space
            np.testing.assert_array_equal(m.base._anchor["action"], res.action)   # blind tail serves homing
    assert pickle.loads(pickle.dumps(m)).window == 24


def _write(run, arm, eps):
    """eps: list of (task, init, success, [(step, top1, mode)])."""
    c = run / "runs" / arm / "client"
    s = run / "runs" / arm / "server_23000"
    c.mkdir(parents=True)
    s.mkdir(parents=True)
    jl, dl = [], []
    for t, i, ok, decs in eps:
        uid = f"{arm}:eval:{t}:{i}"
        jl.append(dict(task_uid=uid, accepted=True, phase="eval", success=ok, attempt=1))
        for step, top1, mode in decs:
            ex = {} if mode is None else dict(r9o3_mode=float(mode), r9o3_t_trig=float(decs[1][0]) if mode >= 1 else -1.0)
            dl.append(dict(ev="dec", uid=uid, attempt=1, step=step, vision=True, hit=True, top1=top1, extras=ex))
    (c / "journal.jsonl").write_text("".join(json.dumps(x) + "\n" for x in jl))
    (s / "decisions_x.jsonl").write_text("".join(json.dumps(x) + "\n" for x in dl))


def test_dissect_crosstab_on_synthetic_round3_root(tmp_path):
    from exp.offline_search.rounds.r09.explore_opus.round3.tools import dissect
    from exp.offline_search.rounds.r09.explore_opus.round2.tools.triggers import catalog
    st = catalog(("pi05", "l10", 50)).step
    r0 = int(st[st == 0].index[0])
    run = tmp_path / "r09_opus_r3_test"
    # init 20: both arms fall behind pace (top1 stays at demo step 0); homing arm succeeds, cache fails
    _write(run, "r9o3_pi05_l10_50_cache", [(0, 20, False, [(0, r0, None), (30, r0, None)]), (0, 21, True, [(0, r0, None)]),
                                           (0, 40, True, [(0, r0, None)])])
    _write(run, "r9o3_pi05_l10_50_home_w24", [(0, 20, True, [(0, r0, 0), (30, r0, 1)]), (0, 21, True, [(0, r0, 0)])])
    E = dissect.episodes(run, "r9o3_pi05_l10_50_home_w24", "pi05", ("pi05", "l10", 50))
    B = dissect.episodes(run, "r9o3_pi05_l10_50_cache", "pi05", ("pi05", "l10", 50))
    assert set(B.init) == {20, 21}                              # init 40 is never read
    res, _ = dissect.attribute(E, B)
    assert res["crosstab"]["esc1_base1"] == dict(n=1, sr=1.0, base_sr=0.0, up=1, down=0)
    assert res["crosstab"]["esc0_base0"]["n"] == 1
