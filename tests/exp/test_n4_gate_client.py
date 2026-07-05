"""Unit tests for the Stage 3a N4 hybrid gate client (all non-manual).

Pure state machine + fake inner client + env helpers, no server / GPU. N4 layers
a V2 cache-execution-cap injection on the embedded (unmodified) N1 V1 hysteresis;
the tests pin (a) V1 degeneration to N1 when V2 never fires, (b) V2 trigger and
run-reset semantics, (c) that a V2 injection does not perturb the N1 phase (D3).
"""

from __future__ import annotations

import pytest

from exp.gate_research.n1_gate_client import N1GateState
from exp.gate_research.n4_gate_client import (
    N4GateClient,
    N4GateState,
    make_n4_client_factory,
    n4_params_from_env,
)


# ----------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------
def drive_n1(state: N1GateState, scores):
    """Drive an N1GateState over a score sequence (score fed only on search)."""
    out = []
    for sc in scores:
        d = state.decide()
        out.append(d)
        state.observe("search", sc) if d == "search" else state.observe("skip", None)
    return out


def drive_n4(state: N4GateState, steps):
    """Drive an N4GateState. ``steps`` = list of ``(score, hit_type)`` the server
    WOULD return on a search; on a skip the machine ignores them (fresh inference,
    fh_run reset)."""
    out = []
    for sc, ht in steps:
        d = state.decide()
        out.append(d)
        if d == "search":
            state.observe("search", ht, sc)
        else:
            state.observe("skip", None, None)
    return out


# ----------------------------------------------------------------------
# construction validation (fail-fast): V1 params delegate to N1; L / include_ws new
# ----------------------------------------------------------------------
@pytest.mark.parametrize("L", [0, -1, True, 1.5, "3"])
def test_bad_L_raises(L):
    with pytest.raises(ValueError):
        N4GateState(0.5, 0.5, 1, 2, L)


@pytest.mark.parametrize("kw", [
    {"theta_low": float("nan"), "theta_high": 0.5, "j": 1, "M": 2},
    {"theta_low": 0.6, "theta_high": 0.5, "j": 1, "M": 2},  # high < low
    {"theta_low": 0.5, "theta_high": 0.5, "j": 0, "M": 2},
    {"theta_low": 0.5, "theta_high": 0.5, "j": True, "M": 2},  # bool not int
])
def test_bad_v1_params_delegate_to_n1(kw):
    with pytest.raises(ValueError):
        N4GateState(L=3, **kw)


def test_bad_include_ws_raises():
    with pytest.raises(ValueError):
        N4GateState(0.5, 0.5, 1, 2, 3, include_ws="yes")


# ----------------------------------------------------------------------
# (test 2) V1 degeneration: L too large to ever fire -> N4 == N1 step for step
# ----------------------------------------------------------------------
def test_v1_degeneration_when_L_never_triggers():
    # L is a large finite int (> trace length), never a sentinel: §3.5 requires int>=1.
    scores = [0.9, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.95, 0.2, None, 0.99, 0.3]
    for (tl, th, j, M) in [(0.5, 0.5, 2, 3), (0.4, 0.7, 1, 2), (0.5, 0.6, 3, None)]:
        n1 = N1GateState(tl, th, j, M)
        n4 = N4GateState(tl, th, j, M, L=10**9)
        # FULL_HIT every step -> fh_run grows but never reaches 10**9, V2 never fires.
        assert drive_n4(n4, [(sc, "FULL_HIT") for sc in scores]) == drive_n1(n1, scores)


# ----------------------------------------------------------------------
# (test 3) V2 pure trigger: all FULL_HIT, N1 always searching -> search x L, skip, ...
# ----------------------------------------------------------------------
def test_v2_caps_cache_execution_run():
    st = N4GateState(0.5, 0.5, 1, 2, L=3)  # high scores keep N1 searching
    got = drive_n4(st, [(0.9, "FULL_HIT")] * 9)
    assert got == ["search", "search", "search", "skip",
                   "search", "search", "search", "skip", "search"]


# ----------------------------------------------------------------------
# (test 4) a WARM_START / MISS resets the cache-execution run (include_ws=False)
# ----------------------------------------------------------------------
def test_warm_start_resets_run_by_default():
    st = N4GateState(0.5, 0.5, 1, 2, L=3, include_ws=False)
    # FH,FH,WS(reset),FH,FH,FH -> trigger deferred from step3 to step6.
    steps = [(0.9, "FULL_HIT"), (0.9, "FULL_HIT"), (0.9, "WARM_START"),
             (0.9, "FULL_HIT"), (0.9, "FULL_HIT"), (0.9, "FULL_HIT"), (0.9, "FULL_HIT")]
    got = drive_n4(st, steps)
    assert got[:6] == ["search"] * 6  # WS reset pushed the injection out
    assert got[6] == "skip"


def test_miss_resets_run():
    st = N4GateState(0.5, 0.5, 1, 2, L=2)
    # Without the MISS, FH,FH would trigger a skip at step2. The MISS at step1
    # resets fh_run, so the trigger is deferred to step4 (fh_run reaches L again).
    got = drive_n4(st, [(0.9, "FULL_HIT"), (0.9, "MISS"),
                        (0.9, "FULL_HIT"), (0.9, "FULL_HIT"), (0.9, "FULL_HIT")])
    assert got == ["search", "search", "search", "search", "skip"]


# ----------------------------------------------------------------------
# (test 5) include_ws=True counts WARM_START into the run -> earlier trigger
# ----------------------------------------------------------------------
def test_include_ws_counts_warm_start():
    st = N4GateState(0.5, 0.5, 1, 2, L=3, include_ws=True)
    steps = [(0.9, "FULL_HIT"), (0.9, "FULL_HIT"), (0.9, "WARM_START"), (0.9, "FULL_HIT")]
    got = drive_n4(st, steps)
    assert got == ["search", "search", "search", "skip"]  # WS counted -> fires at step3


# ----------------------------------------------------------------------
# (test 6, D3) a V2 injection does NOT perturb the N1 phase (since_probe frozen)
# ----------------------------------------------------------------------
def test_v2_injection_does_not_pollute_n1_phase():
    st = N4GateState(0.5, 0.5, 1, 2, L=2)  # high scores -> N1 stays searching
    drive_n4(st, [(0.9, "FULL_HIT"), (0.9, "FULL_HIT")])  # fh_run reaches 2
    # step2 is a V2 injection (fh_run>=L while N1 would search).
    assert st.decide() == "skip" and st._last_v2 is True
    st.observe("skip", None, None)
    # N1 sub-machine must be UNTOUCHED by the injection: still searching, since_probe
    # still 0 (a leaked observe("skip") would bump it to 1), low_run still 0.
    assert st._n1.searching and st._n1.since_probe == 0 and st._n1.low_run == 0
    assert st.fh_run == 0  # the injection did break the cache-execution run


# ----------------------------------------------------------------------
# (test 7) episode reset clears both V1 and V2 state
# ----------------------------------------------------------------------
def test_reset_clears_state():
    st = N4GateState(0.5, 0.5, 1, 2, L=2)
    drive_n4(st, [(0.9, "FULL_HIT"), (0.9, "FULL_HIT"), (0.9, "FULL_HIT")])
    st.reset()
    assert st.fh_run == 0 and st._last_v2 is False
    assert st._n1.searching and st._n1.low_run == 0 and st._n1.since_probe == 0
    assert st.decide() == "search"


def test_force_search_resets_v2_run():
    st = N4GateState(0.5, 0.5, 1, 2, L=2)
    drive_n4(st, [(0.9, "FULL_HIT")])
    st.force_search()
    assert st.fh_run == 0 and st._last_v2 is False and st._n1.searching


# ----------------------------------------------------------------------
# N4GateClient wrapper: authoritative searched stamp, fail-open, delegation
# ----------------------------------------------------------------------
class FakeInner:
    def __init__(self, results):
        self._results = list(results)
        self.seen_obs = []
        self.episode_starts = 0

    def infer(self, obs):
        self.seen_obs.append(obs)
        return self._results.pop(0)

    def episode_start(self, *a, **k):
        self.episode_starts += 1
        return {"__ack__": "episode_start"}

    def select_bundle(self, b):
        return ("bundle", b)


def _hm(score, ht="FULL_HIT"):
    return {"__hit_meta__": {"cp1_score": score, "hit_type": ht, "start_t": None,
                             "winner_id": None}, "actions": [0]}


def test_searched_stamp_for_v1_and_v2_skips():
    # L=2, high scores -> N1 searching; steps 0,1 search (FULL_HIT), step2 is V2 skip.
    st = N4GateState(0.5, 0.5, 1, 2, L=2)
    inner = FakeInner([_hm(0.9), _hm(0.9), _hm(0.9)])
    cli = N4GateClient(inner, st)
    r0 = cli.infer({"x": 0})
    assert inner.seen_obs[0]["__gate_decision__"] == "search"
    assert r0["__collect_meta__"] == {"searched": True}
    cli.infer({"x": 1})
    r2 = cli.infer({"x": 2})  # V2 injection -> skip
    assert inner.seen_obs[2]["__gate_decision__"] == "skip"
    assert r2["__collect_meta__"] == {"searched": False}


def test_v1_skip_stamp():
    # j=1: one low score enters skipping -> next decision is a V1 skip.
    st = N4GateState(0.5, 0.5, 1, 2, L=10**9)
    inner = FakeInner([_hm(0.1, "MISS"), _hm(0.0, "MISS")])
    cli = N4GateClient(inner, st)
    cli.infer({})            # search, low score -> skipping
    r1 = cli.infer({})       # V1 skip
    assert inner.seen_obs[1]["__gate_decision__"] == "skip"
    assert r1["__collect_meta__"] == {"searched": False}


def test_fail_open_on_nan_resets_run():
    st = N4GateState(0.5, 0.5, 1, 2, L=2)
    inner = FakeInner([_hm(0.9), _hm(float("nan"))])
    cli = N4GateClient(inner, st)
    cli.infer({})                       # search, fh_run -> 1
    cli.infer({})                       # search, NaN -> fail open
    assert st._n1.searching and st.fh_run == 0 and st.decide() == "search"


def test_fail_open_on_missing_hit_meta():
    st = N4GateState(0.5, 0.5, 1, 2, L=2)
    inner = FakeInner([{"actions": [0]}])  # no __hit_meta__
    cli = N4GateClient(inner, st)
    cli.infer({})
    assert st._n1.searching and st.fh_run == 0


def test_missing_hit_type_treated_non_cache_exec():
    # __hit_meta__ present with a valid score but no hit_type -> not anomaly (score
    # is fine), but the V2 run must NOT advance (conservative: no over-extension).
    st = N4GateState(0.5, 0.5, 1, 2, L=2)
    result = {"__hit_meta__": {"cp1_score": 0.9, "start_t": None, "winner_id": None},
              "actions": [0]}
    cli = N4GateClient(FakeInner([result]), st)
    cli.infer({})
    assert st.fh_run == 0  # hit_type missing -> not counted as cache execution


def test_episode_start_resets():
    st = N4GateState(0.5, 0.5, 1, 2, L=2)
    inner = FakeInner([_hm(0.9)])
    cli = N4GateClient(inner, st)
    cli.infer({})
    cli.episode_start(experiment="e", task="t", episode_id=0)
    assert inner.episode_starts == 1 and st.fh_run == 0 and st._n1.searching


def test_inner_exception_propagates():
    class Boom:
        def infer(self, obs):
            raise RuntimeError("boom")

    cli = N4GateClient(Boom(), N4GateState(0.5, 0.5, 1, 2, L=2))
    with pytest.raises(RuntimeError, match="boom"):
        cli.infer({})


def test_delegates_unknown_methods():
    cli = N4GateClient(FakeInner([]), N4GateState(0.5, 0.5, 1, 2, L=2))
    assert cli.select_bundle("b") == ("bundle", "b")


def test_factory_builds_wrapped_client():
    factory = make_n4_client_factory(
        {"theta_low": 0.5, "theta_high": 0.5, "j": 1, "M": 2, "L": 8},
        inner_factory=lambda server: FakeInner([]),
    )
    assert isinstance(factory(("host", 8000)), N4GateClient)


# ----------------------------------------------------------------------
# env param parsing
# ----------------------------------------------------------------------
def _env(**kw):
    base = {"N4_THETA_LOW": "0.96", "N4_THETA_HIGH": "0.97", "N4_J": "3", "N4_M": "5", "N4_L": "8"}
    base.update(kw)
    return base


def test_env_params_ok():
    assert n4_params_from_env(_env()) == {
        "theta_low": 0.96, "theta_high": 0.97, "j": 3, "M": 5, "L": 8}


def test_env_params_none_m():
    assert n4_params_from_env(_env(N4_M="none"))["M"] is None


def test_env_params_missing_L_raises():
    env = _env()
    del env["N4_L"]
    with pytest.raises(ValueError, match="N4_L"):
        n4_params_from_env(env)


def test_env_params_malformed_L_raises():
    with pytest.raises(ValueError):
        n4_params_from_env(_env(N4_L="x"))
