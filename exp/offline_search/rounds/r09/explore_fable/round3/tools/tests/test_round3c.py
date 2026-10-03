"""v3 stacks: attribute-collision reproduction (the r3/r3b crash), window arming, corrector active on the judge path."""
import types
from pathlib import Path

import numpy as np
import pytest

from exp.offline_search.rounds.r09.explore_fable.round3.tools import methods

ONLYNP = Path("/home/weiland/trace_runs/os_closed_loop/r08_abl/fits/r8abl_onlynp_p_l10_50.pkl")
CORR = Path("/home/weiland/trace_runs/os_closed_loop/r09_fable_r2/fits/r9f2_pi05_l10_50_corr05pt.pkl")
need = pytest.mark.skipif(not (ONLYNP.exists() and CORR.exists()), reason="frozen artifacts not available")
CTX = types.SimpleNamespace(cell="pi05_l10_cache", model="pi05")


def _fit(cls, **kw):
    m = cls(onlynp_fit=str(ONLYNP), corrected_fit=str(CORR), **kw)
    m.fit(None, CTX)
    return m


@need
def test_v2_shadowed_the_judge_burst_and_v3_does_not():
    bad = _fit(methods.NpGraspStack2, max_calls=2)          # the r3b class
    assert bad.burst == 2                                   # judge window length corrupted by the trigger's burst
    good = _fit(methods.NpGraspStack3, max_calls=2)
    assert good.burst == 0 and good.gm_burst == 2
    assert not (set(methods._GraspStack3._OWN) & set(vars(bad).keys()) - {"gm_" + k for k in ()})


@need
def test_window_arming_reproduces_p2_error_only_with_shadowed_burst():
    """k7 arms its burst/return windows after a guard firing iff self.burst > 1; the P2 mask then raises when a disabled
    guard fires. Reproduce that interaction on the fitted objects with the REAL _TriggerMask.query and a stubbed inner judge."""
    from exp.offline_search.harness import api
    from exp.offline_search.rounds.r06.p2_ablations.judge import _TriggerMask

    def run(m):
        m.reset(types.SimpleNamespace(uid="e", task_id=0, init=0))
        s = m._s
        # decision 10: the no-progress guard fires -> k7's window arming line (verbatim condition)
        step, flags = 10, 8
        if m.burst > 1:
            s["burst_end"], s["ret_end"] = step + m.burst, step + m.burst + m.ret_hold
        # decision 11: a DISABLED guard (stuck, bit 1) fires -> real P2 mask query
        inner = api.Result(np.array([0]), np.array([0.0]), 0.0, action=np.zeros((10, 32), np.float32), library="current",
                           extras={"os_flags": 1.0, "os_reason": 1.0, "os_force_miss": 1.0})
        s["flag"].append(1)
        parent = super(_TriggerMask, m)
        orig = type(parent).query if False else None
        import unittest.mock as um
        with um.patch.object(methods.TriggerCommitJudge.__mro__[methods.TriggerCommitJudge.__mro__.index(_TriggerMask) + 1], "query", return_value=inner):
            return _TriggerMask.query(m, types.SimpleNamespace(step=11))

    bad = _fit(methods.NpGraspStack2, max_calls=2)
    with pytest.raises(api.ContractError, match="burst/return windows"):
        run(bad)
    good = _fit(methods.NpGraspStack3, max_calls=2)
    out = run(good)
    assert out.extras["os_force_miss"] == 0.0 and out.extras["os_flags"] == 0.0     # disabled guard masked, no error


@need
def test_corrector_is_applied_on_the_judge_path():
    m = _fit(methods.NpGraspStack3, max_calls=0)
    base = m.base
    assert type(base).__name__ == "CorrectedCacheJ" and base.blend == 0.5
    rows = np.asarray(base.tasks[0].rows[:16])
    w = np.ones(16)
    q = types.SimpleNamespace(step=3, task_id=0, key_v0=np.zeros(32768, np.float32), key_v1=np.zeros(32768, np.float32), rs=np.zeros(32, np.float32), episode=types.SimpleNamespace(uid="e"))
    plain = methods.CorrectedCache.os_synth(base, q, rows, w)        # uncorrected kernel mean
    corrected = base.os_synth(q, rows, w)
    assert corrected.shape == plain.shape and np.abs(corrected[:10, :6] - plain[:10, :6]).max() > 1e-4
    assert np.array_equal(corrected[:, 6], plain[:, 6])             # gripper untouched
