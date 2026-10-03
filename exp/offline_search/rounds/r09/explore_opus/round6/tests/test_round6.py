"""Round-6 tests: exhaustion trigger / bounded takeover on a stub stack, and the real fitted r3c stacks."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exp.offline_search.harness import api
from exp.offline_search.rounds.r09.explore_opus.round2.tools.triggers import catalog
from exp.offline_search.rounds.r09.explore_opus.round6.methods import (
    EXHAUST_REASON, ExhaustNpGraspEsc3, ExhaustNpGraspStack3, ExhaustNpGraspStackGroot3, _Exhaust)

R3C = Path("/home/weiland/trace_runs/os_closed_loop/r09_fable_r3c")


class _Parent:
    """Stub stack: library of one 10-row demo (rows 0..9); scripted (top1, stack verdict) per step."""

    def __init__(self):
        self.base = SimpleNamespace(lib_step=np.arange(10))
        self.C = SimpleNamespace(ep_len=np.full(10, 10), nxt=np.r_[np.arange(1, 10), -1])
        self._s = {"flag": []}
        self.script = {}
        self.fit_info = {}
        self.name = "stub"

    def reset(self, episode):
        self._s = {"flag": []}

    def query(self, q):
        top1, reason = self.script.get(int(q.step), (0, 0.0))
        self._s["flag"].append(int(reason != 0))
        ex = dict(os_force_miss=float(reason != 0), os_reason=float(reason), os_flags=0.0)
        return api.Result(np.array([top1, 1]), np.zeros(2), 0.0, action=np.zeros((50, 32), np.float32), library="current",
                          extras=ex)


class _X(_Exhaust, _Parent):
    def __init__(self, end_rows=0, dwell=2, window=6, force=()):
        _Parent.__init__(self)
        self._xt_init(end_rows, dwell, window, force)
        self._xt_fit()

    def reset(self, episode):
        super().reset(episode)
        self._xt_reset(episode)

    def query(self, q):
        return self._xt_apply(super().query(q), q)


def _ep(n=0):
    return SimpleNamespace(uid=f"x:eval:0:{n}", task_id=0, init=n)


def _run(x, script, steps, ep=None):
    ep = ep or _ep()
    x.reset(ep)
    x.script = script
    return {s: x.query(SimpleNamespace(step=s, episode=ep)).extras for s in steps}


def test_rows_to_end_from_library():
    x = _X()
    np.testing.assert_array_equal(x.xt_rte, np.arange(9, -1, -1))


def test_trigger_after_dwell_at_end_then_bounded_window():
    x = _X(end_rows=0, dwell=2, window=6)
    # demo end (row 9) first seen at step 10; still at end at step 12 -> takeover 12..17; stack verdict at 20 untouched
    script = {s: (min(max(s - 1, 0), 9), 0.0) for s in range(0, 30, 2)}
    script[20] = (9, 4.0)
    out = _run(x, script, range(0, 30, 2))
    assert out[10]["r9o6_t_end"] == 10 and out[10]["os_force_miss"] == 0.0
    assert out[12]["r9o6_trigger"] == 1.0
    for s in (12, 14, 16):
        assert out[s]["os_force_miss"] == 1.0 and out[s]["os_reason"] == EXHAUST_REASON and out[s]["r9o6_takeover"] == 1.0
    for s in (18, 22, 24):
        assert out[s]["os_force_miss"] == 0.0 and out[s]["r9o6_takeover"] == 0.0     # back to the stack, no re-trigger
    assert out[20]["os_reason"] == 4.0                                                # stack's own verdict kept
    assert x._s["flag"][-1] == 0


def test_no_trigger_when_retrieval_leaves_the_end_or_episode_ends():
    x = _X(end_rows=0, dwell=2, window=6)
    script = {0: (5, 0.0), 2: (9, 0.0), 4: (6, 0.0), 6: (7, 0.0), 8: (8, 0.0)}
    out = _run(x, script, range(0, 10, 2))
    assert all(o["os_force_miss"] == 0.0 for o in out.values())
    assert out[8]["r9o6_t_end"] == 2.0                                                # first end look remembered


def test_existing_verdict_kept_inside_window_and_reset_between_episodes():
    x = _X(end_rows=1, dwell=0, window=4)
    out = _run(x, {0: (8, 91.0), 2: (9, 0.0)}, (0, 2))
    assert out[0]["os_reason"] == 91.0 and out[0]["r9o6_takeover"] == 1.0
    assert out[2]["os_reason"] == EXHAUST_REASON
    out = _run(x, {0: (3, 0.0)}, (0,), ep=_ep(1))
    assert out[0]["r9o6_start"] == -1.0 and out[0]["os_force_miss"] == 0.0


def test_forced_trigger():
    x = _X(end_rows=0, dwell=1000, window=3, force=(4,))
    out = _run(x, {}, range(0, 10, 2))
    assert [out[s]["os_force_miss"] for s in range(0, 10, 2)] == [0.0, 0.0, 1.0, 1.0, 0.0]


def test_bad_parameters_rejected():
    with pytest.raises(ValueError):
        _X(window=0)
    with pytest.raises(ValueError):
        _X(dwell=-1)
    with pytest.raises(ValueError):
        _X(end_rows=1.5)


def test_clash_detected():
    x = _X()
    keep = {k: getattr(x, k) for k in x._XT_OWN}
    x.xt_window = 99
    with pytest.raises(api.ContractError):
        x._xt_check_clash(keep)


@pytest.mark.parametrize("name,cls,model,drop", [("r9f3c_groot_l10_50_np_corr05", ExhaustNpGraspStackGroot3, "groot", False),
                                                 ("r9f3c_pi05_l10_50_np_corr05_esc", ExhaustNpGraspEsc3, "pi05", False),
                                                 ("r9f3c_pi05_l10_50_np_corr05_esc", ExhaustNpGraspStack3, "pi05", True)])
def test_real_stack_fit_matches_fable_artifact(name, cls, model, drop):
    reg = R3C / "r3c_kwargs.json"
    if not reg.exists():
        pytest.skip("r3c registry missing")
    spec = json.loads(reg.read_text())[name]
    kw = dict(spec["kwargs"])
    if drop:
        kw.pop("lag_threshold")
        kw.pop("deadline")
    m = cls(**kw, end_rows=0, dwell=2, window=24)
    m.prof = api.NULL_PROFILER
    m.fit(None, SimpleNamespace(cell=spec["cell"]))
    with open(R3C / "fits" / f"{name}.pkl", "rb") as f:
        src = pickle.load(f)["method"]
    assert int(m.burst) == int(src.burst) == 0 and m.gm_max_calls == src.gm_max_calls == 0
    assert type(m.base).__name__ == type(src.base).__name__ == "CorrectedCacheJ" and m.base.blend == src.base.blend == 0.5
    np.testing.assert_array_equal(np.asarray(m.base.act), np.asarray(src.base.act))
    np.testing.assert_array_equal(m.base.lib_step, src.base.lib_step)
    for k in ("noprog_n", "prog_eps", "disabled_guards", "progress_guard"):
        assert getattr(m, k) == getattr(src, k), k
    if cls is ExhaustNpGraspEsc3:
        assert m.lag_threshold == 12 and m.deadline == 80          # escalation kept alongside
    else:
        assert not hasattr(m, "_esc_step")                          # no escalation (GR00T stack / pi0.5 replaced)
    cat = catalog((model, "l10", 50))
    np.testing.assert_array_equal(m.xt_rte, (cat.ep_len - 1 - cat.step).values)
    assert m.name.startswith("R9O6_exhaust_E0_D2_W24__")
