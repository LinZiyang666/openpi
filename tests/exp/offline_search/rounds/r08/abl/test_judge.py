"""Exhaustive masks, R6 nesting, and guard-only effects without fitting or a GPU."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from exp.offline_search.closed_loop.blind import LookReason
from exp.offline_search.harness import api
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge
from exp.offline_search.rounds.r06.p1_groot_commit.judge import GrootCommitJudge
from exp.offline_search.rounds.r06.p2_ablations.judge import TriggerCommitJudge as R6Pi
from exp.offline_search.rounds.r06.p2_ablations.judge import TriggerGrootCommitJudge as R6Groot
from exp.offline_search.rounds.r08.abl.judge import BITS, TriggerCommitJudge, TriggerGrootCommitJudge

DEPLOYED = dict(base_kwargs=dict(serving="anchor_tail", budget=1, gates="budget_only"),
                progress_guard="noprog_span", events="none", stuck_guard="vision_confirmed",
                policy_tail_gate="lifecycle", monitor="off")
CLASSES = [(TriggerCommitJudge, R6Pi, CommitJudge), (TriggerGrootCommitJudge, R6Groot, GrootCommitJudge)]


def result(flags):
    return api.Result(np.arange(16), np.arange(16, dtype=float), .5, action=np.zeros((10, 32)),
        library="current", extras=dict(os_flags=float(flags), os_reason=float((flags & -flags).bit_length()),
        os_force_miss=float(bool(flags)), stuck_n=3., overtime=2., lag=6., noprog_span=5., diagnostic=42.))


@pytest.mark.parametrize("cls,r6,parent", CLASSES)
def test_every_subset_and_flag(cls, r6, parent, monkeypatch):
    monkeypatch.setattr(parent, "query", lambda self, q: q)
    for n in range(5):
        for guards in itertools.combinations(BITS, n):
            m = cls(disabled_guards=set(guards), **DEPLOYED)
            api.check_method_attrs(m)
            mask = sum(BITS[g] for g in guards)
            assert m.disabled_mask == mask
            for flags in range(16):
                m._s = dict(burst_end=0, ret_end=0, flag=[int(bool(flags))], stuck_n=3)
                m._noprog_span = 5
                ref = result(flags)
                got = m.query(ref)
                want = flags & ~mask
                assert got.extras == {**ref.extras, "os_flags": float(want),
                    "os_reason": float((want & -want).bit_length()), "os_force_miss": float(bool(want))}
                assert all(getattr(got, field) is getattr(ref, field) for field in ("topk", "scores", "action"))
                assert got.confidence == ref.confidence and got.library == ref.library
                assert m._s == dict(burst_end=0, ret_end=0, flag=[int(bool(want))], stuck_n=3)
                assert m._noprog_span == 5
                if len(guards) == 1:
                    old = r6(disabled_guard=guards[0], **DEPLOYED)
                    old._s = dict(burst_end=0, ret_end=0, flag=[int(bool(flags))])
                    assert old.disabled_mask == m.disabled_mask
                    assert old.query(ref).extras == got.extras


@pytest.mark.parametrize("cls,r6,parent", CLASSES)
@pytest.mark.parametrize("guards", [[], ["stuck"], ["no_progress"], ["stuck", "terminal", "overtime"], list(BITS)])
def test_blind_veto_retains_progress_memo(cls, r6, parent, guards):
    m = cls(disabled_guards=guards, **DEPLOYED)
    sentinel = SimpleNamespace(action="base action")
    m.base.lifecycle_reason = lambda q: None
    m.base.blind_step = lambda q: sentinel
    m._noprog_span = 1
    got = m.blind_step(None)
    if "no_progress" in guards:
        assert got is sentinel
    else:
        assert isinstance(got, LookReason) and got.name == "noprog_span"
    assert m._noprog_span == 1
    m._noprog_span = 0
    assert m.blind_step(None) is sentinel


@pytest.mark.parametrize("cls", [TriggerCommitJudge, TriggerGrootCommitJudge])
@pytest.mark.parametrize("guards", ["stuck", ["invalid"], ["none"], ["all"], None, [["stuck"]]])
def test_invalid_guard_sets(cls, guards):
    with pytest.raises(ValueError):
        cls(disabled_guards=guards, **DEPLOYED)


@pytest.mark.parametrize("cls", [TriggerCommitJudge, TriggerGrootCommitJudge])
@pytest.mark.parametrize("field,value", [("burst", 2), ("events", "disp"), ("guards", False),
    ("monitor", "loeo_xyz99"), ("policy_tail_gate", "inherited"), ("stuck_guard", "dense"),
    ("progress_guard", "noprog_n"), ("memo_reset_after_miss", True), ("disabled_guard", "stuck")])
def test_deployed_b_restrictions(cls, field, value):
    with pytest.raises(ValueError):
        cls(disabled_guards=["stuck"], **{**DEPLOYED, field: value})


def test_duplicate_names_are_a_set():
    m = TriggerCommitJudge(disabled_guards=["overtime", "stuck", "stuck"], **DEPLOYED)
    assert m.disabled_guards == ("stuck", "overtime")
    assert m.disabled_mask == 5


def test_stuck_removal_retains_overtime(monkeypatch):
    monkeypatch.setattr(CommitJudge, "query", lambda self, q: q)
    m = TriggerCommitJudge(disabled_guards=["stuck"], **DEPLOYED)
    m._s = dict(burst_end=0, ret_end=0, flag=[1], stuck_n=3)
    got = m.query(result(1 | 4))
    assert got.extras["os_flags"] == 4 and got.extras["os_force_miss"] == 1
    assert got.extras["os_reason"] == 3 and m._s["stuck_n"] == 3
