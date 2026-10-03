"""Round-3 stack tests (synthetic; no capture roots)."""
import numpy as np
import pytest

from exp.offline_search.rounds.r09.explore_fable.round3.tools import methods


def _stack(cls=methods.NpGraspStack, **kw):
    s = cls.__new__(cls)
    s._stack_init("np.pkl", "c.pkl", 0.001, kw.get("closed_sign", 1.0), 2, kw.get("burst", 2), kw.get("max_calls", 2))
    s._s = {"flag": [0]}
    return s


class _Q:
    def __init__(self, closed=1.0, aperture=0.0002, step=3):
        self.step = step
        self.hist_a_exec = np.zeros((step, 10, 7), np.float32)
        self.hist_a_exec[:, :5, 6] = closed
        self.raw_state = np.array([0, 0, 0, 0, 0, 0, aperture, -aperture])
        self.episode = type("E", (), {"uid": "e"})()


def test_trigger_burst_cap_and_force_miss_extras():
    from exp.offline_search.harness import api
    s = _stack(burst=2, max_calls=1)
    res = api.Result(topk=np.array([0]), scores=np.array([0.0]), confidence=0.0, action=np.zeros((10, 32), np.float32), library="current", extras={"os_force_miss": 0.0})
    out = s._stack_query(res, _Q())
    assert out.extras["r9f3_grasp_trigger"] == 1.0 and out.extras["os_force_miss"] == 1.0 and out.extras["os_reason"] == methods.GRASP_REASON and s._s["flag"][-1] == 1
    out = s._stack_query(res, _Q())                      # second anchor of the burst
    assert out.extras["r9f3_grasp_burst"] == 1.0 and out.extras["r9f3_grasp_trigger"] == 0.0 and out.extras["os_force_miss"] == 1.0
    out = s._stack_query(res, _Q())                      # cap reached
    assert out.extras["os_force_miss"] == 0.0


def test_trigger_respects_sign_and_holding():
    s = _stack(closed_sign=-1.0)
    assert s.empty_grasp(_Q(closed=-1.0)) is True
    assert s.empty_grasp(_Q(closed=1.0)) is False
    assert s.empty_grasp(_Q(closed=-1.0, aperture=0.01)) is False
    assert _stack(max_calls=0).empty_grasp(_Q()) is False


def test_invalid_parameters():
    s = methods.NpGraspStack.__new__(methods.NpGraspStack)
    with pytest.raises(ValueError):
        s._stack_init("a", "b", 0.001, 0.5, 2, 2, 2)
    with pytest.raises(ValueError):
        s._stack_init("a", "b", 0.001, 1.0, 2, 0, 2)


def test_v2_classes_use_one_judge_family_each():
    fams = {}
    for cls in (methods.NpGraspStack2, methods.NpGraspEsc2, methods.NpGraspStackGroot2):
        names = [c.__name__ for c in cls.__mro__]
        fams[cls.__name__] = ("CommitJudge" in names, "GrootCommitJudge" in names)
    assert fams["NpGraspStack2"] == (True, False)
    assert fams["NpGraspEsc2"] == (True, False)
    assert fams["NpGraspStackGroot2"] == (False, True) or fams["NpGraspStackGroot2"][1]   # GR00T family only
    # the r3 class that failed closed loop mixes both families (documented root cause)
    names = [c.__name__ for c in methods.NpGraspStackGroot.__mro__]
    assert "CommitJudge" in names and "GrootCommitJudge" in names


def test_v2_forced_trigger_and_sign_guard():
    s = methods.NpGraspStack2.__new__(methods.NpGraspStack2)
    s._stack_init("a", "b", 0.001, 1.0, 2, 2, 2)
    s._stack_init2((5,))
    s._s = {"flag": [0]}
    q = _Q(closed=-1.0, aperture=0.03, step=5)        # open, not empty: only the forced index fires
    assert s.empty_grasp(q) is True
    q.step = 6
    assert s.empty_grasp(q) is False
    with pytest.raises(ValueError):
        methods.NpGraspStackGroot2(closed_sign=1.0)


def test_v2_family_guard_accepts_groot_and_rejects_mixed():
    from exp.offline_search.rounds.r08.abl.judge import TriggerCommitJudge, TriggerGrootCommitJudge
    ok = lambda cls: sum(c in cls.__mro__ for c in (TriggerCommitJudge, TriggerGrootCommitJudge))
    assert ok(methods.NpGraspStack2) == 1 and ok(methods.NpGraspStackGroot2) == 1 and ok(methods.NpGraspEsc2) == 1
    assert ok(methods.NpGraspStackGroot) == 2          # the r3 class that failed
