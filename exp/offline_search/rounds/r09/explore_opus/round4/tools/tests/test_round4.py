"""Round-4 stack tests: the corrector really acts on the escalation controller's path; escalation window intact."""
from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r09.explore_opus.round4.methods import CorrectedEscalateCalls

FABLE = Path("/home/weiland/trace_runs/os_closed_loop/r09_fable_r2/fits")
PLAIN = {"pi05": Path("/home/weiland/trace_runs/os_closed_loop/r05_ptail/fits/r5t_p_l10_50_tail1uc.pkl"),
         "groot": Path("/home/weiland/trace_runs/os_closed_loop/r05_x/fits/r5x_g_l10_50_tail1u.pkl")}
STORE = Path("/home/weiland/trace_runs/offline_search_store")


def _load(path):
    with open(path, "rb") as f:
        blob = FitUnpickler(f).load()
    blob["method"].prof = api.NULL_PROFILER
    return blob


def _stack(model, **kw):
    fit = FABLE / f"r9f2_{model}_l10_50_corr05pt.pkl"
    blob = _load(fit)
    m = CorrectedEscalateCalls(corrected_fit=str(fit), corrected_kwargs=blob["kwargs"], **kw)
    m.prof = api.NULL_PROFILER
    m.fit(None, SimpleNamespace(cell=f"{model}_l10_cache"))
    return m


def _q(model, k0, k1, rs, row, step, ep, hist):
    return SimpleNamespace(key_v0=k0[row], key_v1=k1[row], rs=rs[row], task_id=ep.task_id, step=step, episode=ep,
                           prev_hit=True if step else None, hist_key_v0=np.array(hist[0] + [k0[row]]),
                           hist_key_v1=np.array(hist[1] + [k1[row]]))


@pytest.mark.parametrize("model", ["pi05", "groot"])
def test_corrector_applied_on_stack_path_motion_only(model):
    if not (FABLE / f"r9f2_{model}_l10_50_corr05pt.pkl").exists():
        pytest.skip("fable artifact missing")
    stack = _stack(model)
    corr = _load(FABLE / f"r9f2_{model}_l10_50_corr05pt.pkl")["method"]
    plain = _load(PLAIN[model])["method"]
    lib = STORE / "library" / f"{model}_l10" / "current"
    k0, k1, rs = (np.load(lib / n, mmap_mode="r") for n in ("key_v0.npy", "key_v1.npy", "rs.npy"))
    n_diff = 0
    for task in range(10):
        r = int(stack.base.tasks[task].rows[0])
        ep = SimpleNamespace(uid=f"t:eval:{task}:0", task_id=task, init=0)
        for m in (corr, plain):
            m.reset(ep)
        hist = ([], [])
        for step, row in [(0, r), (2, r + 1)]:
            q = _q(model, k0, k1, rs, row, step, ep, hist)
            a_s = stack.query(q)
            a_c = corr.query(_q(model, k0, k1, rs, row, step, ep, hist))
            a_p = plain.query(_q(model, k0, k1, rs, row, step, ep, hist))
            hist[0].append(k0[row]); hist[1].append(k1[row])
            assert a_s.extras["os_force_miss"] == 0.0
            np.testing.assert_array_equal(a_s.action, a_c.action)                  # exactly the standalone corrector
            np.testing.assert_array_equal(a_s.topk, a_p.topk)                      # retrieval unchanged
            np.testing.assert_array_equal(a_s.action[:, 6:], a_p.action[:, 6:])    # gripper + padding untouched
            np.testing.assert_array_equal(a_s.action[10:, :6], a_p.action[10:, :6])
            d = np.abs(a_s.action[:10, :6] - a_p.action[:10, :6]).max()
            n_diff += d > 1e-6
            # the next blind decision serves the corrected tail
            np.testing.assert_array_equal(stack.base._anchor["action"], a_s.action)
        bq = SimpleNamespace(task_id=task, step=3, episode=ep, prev_hit=True, blind_age=0, executed_steps=5,
                             rs=rs[r + 1], hist_rs=np.array([rs[r], rs[r], rs[r + 1]]))
        tail = stack.blind_step(bq)
        np.testing.assert_array_equal(tail.action[:5, :7], a_s.action[5:10, :7])
    assert n_diff >= 15            # the head changes the motion channels on (almost) every query


@pytest.mark.parametrize("model", ["pi05", "groot"])
def test_forced_trigger_opens_a_24_decision_window(model):
    if not (FABLE / f"r9f2_{model}_l10_50_corr05pt.pkl").exists():
        pytest.skip("fable artifact missing")
    stack = _stack(model, force_trigger_at=(4,))
    lib = STORE / "library" / f"{model}_l10" / "current"
    k0, k1, rs = (np.load(lib / n, mmap_mode="r") for n in ("key_v0.npy", "key_v1.npy", "rs.npy"))
    r = int(stack.base.tasks[0].rows[0])
    ep = SimpleNamespace(uid="t:eval:0:0", task_id=0, init=0)
    hist, calls = ([], []), {}
    for step in range(0, 34, 2):
        row = r + min(step // 2, 5)
        q = _q(model, k0, k1, rs, row, step, ep, hist)
        q.prev_hit = True if step else None
        calls[step] = stack.query(q).extras["os_force_miss"]
        hist[0].append(k0[row]); hist[1].append(k1[row])
    assert [s for s, c in calls.items() if c == 1.0] == list(range(4, 28, 2))   # 24 decisions from the trigger
    assert pickle.loads(pickle.dumps(stack)).window == 24


def test_fit_refuses_wrong_artifact():
    fit = FABLE / "r9f2_pi05_l10_50_corr05pt.pkl"
    if not fit.exists():
        pytest.skip("fable artifact missing")
    blob = _load(fit)
    m = CorrectedEscalateCalls(corrected_fit=str(fit), corrected_kwargs=dict(blob["kwargs"], blend=1.0))
    with pytest.raises(ValueError):
        m.fit(None, SimpleNamespace(cell="pi05_l10_cache"))
    m = CorrectedEscalateCalls(corrected_fit=str(fit), corrected_kwargs=blob["kwargs"])
    with pytest.raises(ValueError):
        m.fit(None, SimpleNamespace(cell="groot_l10_cache"))


@pytest.mark.parametrize("model", ["pi05", "groot"])
def test_frozen_artifact_serves_the_fable_corrector(model):
    art = Path(__file__).resolve().parents[2] / "artifacts" / f"r9o4_{model}_l10_50_corr_esc_w24.pkl"
    if not art.exists():
        pytest.skip("frozen artifact not built yet")
    blob = _load(art)
    stack = blob["method"]
    assert blob["spec"].endswith(":CorrectedEscalateCalls") and blob["kwargs"]["window"] == 24
    assert blob["kwargs"]["lag_threshold"] == 12 and blob["kwargs"]["deadline"] == 80 and "force_trigger_at" not in blob["kwargs"]
    assert stack.force_trigger_at == () and stack.base.blend == 0.5 and not stack.base.correct_gripper
    corr = _load(FABLE / f"r9f2_{model}_l10_50_corr05pt.pkl")["method"]
    lib = STORE / "library" / f"{model}_l10" / "current"
    k0, k1, rs = (np.load(lib / n, mmap_mode="r") for n in ("key_v0.npy", "key_v1.npy", "rs.npy"))
    for task in (0, 5, 9):
        r = int(stack.base.tasks[task].rows[0])
        ep = SimpleNamespace(uid=f"f:eval:{task}:0", task_id=task, init=0)
        corr.reset(ep)
        a = stack.query(_q(model, k0, k1, rs, r, 0, ep, ([], [])))
        b = corr.query(_q(model, k0, k1, rs, r, 0, ep, ([], [])))
        np.testing.assert_array_equal(a.action, b.action)
