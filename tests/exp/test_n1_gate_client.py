"""Unit tests for the Stage 1b N1 gate client and live runner dispatch.

All non-manual: pure state machine + fake inner client + config helpers, no
server / GPU.
"""

from __future__ import annotations

import argparse
import pathlib

import pytest

from exp.gate_research.n1_gate_client import (
    N1GateClient,
    N1GateState,
    make_n1_client_factory,
    n1_params_from_env,
)
from exp.gate_research import run_n1_live


# ----------------------------------------------------------------------
# Reference decision sim (in-test, does NOT import n1_offline_scan.n1_sim)
# ----------------------------------------------------------------------
def ref_decisions(scores, theta_low, theta_high, j, M):
    """Independent replica of the offline decide/observe loop over a score
    sequence (None -> -inf). Returns the per-step decision list."""
    searching, low_run, since_probe = True, 0, 0
    out = []
    for sc in scores:
        if searching:
            d = "search"
        elif M is not None and since_probe + 1 >= M:
            d = "search"
        else:
            d = "skip"
        out.append(d)
        if d == "search":
            s = float("-inf") if sc is None else sc
            if searching:
                if s < theta_low:
                    low_run += 1
                    if low_run >= j:
                        searching, since_probe = False, 0
                else:
                    low_run = 0
            else:
                since_probe = 0
                if s >= theta_high:
                    searching, low_run = True, 0
        else:
            since_probe += 1
    return out


def drive(state: N1GateState, scores):
    """Drive N1GateState over a score sequence (feeding the score only on
    searched steps, as a live client would)."""
    out = []
    for sc in scores:
        d = state.decide()
        out.append(d)
        if d == "search":
            state.observe("search", sc)
        else:
            state.observe("skip", None)
    return out


# ----------------------------------------------------------------------
# N1GateState golden traces
# ----------------------------------------------------------------------
def test_all_high_never_skips():
    st = N1GateState(0.5, 0.5, 2, 3)
    assert drive(st, [0.9] * 10) == ["search"] * 10


def test_enter_skip_and_probe_cycle():
    # theta_low=0.5, j=2, M=3: two lows -> skip; then skip,skip,probe repeating.
    st = N1GateState(0.5, 0.5, 2, 3)
    got = drive(st, [0.1] * 8)
    assert got == ["search", "search", "skip", "skip", "search", "skip", "skip", "search"]


def test_probe_resume_on_high():
    # j=1, M=2, theta_high=0.7: one low -> skip; probe at step2 scores high -> resume.
    st = N1GateState(0.5, 0.7, 1, 2)
    got = drive(st, [0.1, 0.0, 0.9, 0.9])
    assert got == ["search", "skip", "search", "search"]


def test_none_score_is_miss_strength():
    # None on a searched step behaves like a very low score (enters skip via j).
    st = N1GateState(0.5, 0.7, 1, 2)
    got = drive(st, [None, 0.0, 0.0])
    assert got[0] == "search" and got[1] == "skip"


def test_matches_reference_on_mixed_trace():
    scores = [0.9, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.95, 0.2, None, 0.99, 0.3]
    for (tl, th, j, M) in [(0.5, 0.5, 2, 3), (0.4, 0.7, 1, 2), (0.5, 0.6, 3, None)]:
        st = N1GateState(tl, th, j, M)
        assert drive(st, scores) == ref_decisions(scores, tl, th, j, M)


def test_reset_clears_state():
    st = N1GateState(0.5, 0.5, 1, 2)
    drive(st, [0.1, 0.1, 0.1])  # push into skipping
    st.reset()
    assert st.searching and st.low_run == 0 and st.since_probe == 0
    assert st.decide() == "search"


# ----------------------------------------------------------------------
# Construction validation (fail-fast)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    {"theta_low": float("nan"), "theta_high": 0.5, "j": 1, "M": 2},
    {"theta_low": 0.5, "theta_high": float("inf"), "j": 1, "M": 2},
    {"theta_low": 0.6, "theta_high": 0.5, "j": 1, "M": 2},  # high < low
    {"theta_low": 0.5, "theta_high": 0.5, "j": 0, "M": 2},
    {"theta_low": 0.5, "theta_high": 0.5, "j": 1, "M": 0},
    {"theta_low": 0.5, "theta_high": 0.5, "j": True, "M": 2},  # bool not int
])
def test_bad_params_raise(kw):
    with pytest.raises(ValueError):
        N1GateState(**kw)


# ----------------------------------------------------------------------
# N1GateClient wrapper: exception contract, provenance, delegation
# ----------------------------------------------------------------------
class FakeInner:
    """Records obs it received; returns scripted results in order."""

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


def _hm(score):
    return {"__hit_meta__": {"cp1_score": score, "hit_type": "MISS", "start_t": None,
                             "winner_id": None}, "actions": [0]}


def test_provenance_roundtrip_and_searched_stamp():
    st = N1GateState(0.5, 0.5, 1, 2)
    inner = FakeInner([_hm(0.9), _hm(0.1), _hm(0.1)])
    cli = N1GateClient(inner, st)
    r0 = cli.infer({"x": 1})
    # step0 decision is search; obs carries it; result stamps searched=True.
    assert inner.seen_obs[0]["__gate_decision__"] == "search"
    assert r0["__collect_meta__"] == {"searched": True}
    assert "__gate_decision__" not in {"x": 1}  # caller obs not mutated
    cli.infer({"x": 2})  # 0.1 low -> enters skip (j=1)
    r2 = cli.infer({"x": 3})  # now skipping -> decision skip
    assert inner.seen_obs[2]["__gate_decision__"] == "skip"
    assert r2["__collect_meta__"] == {"searched": False}


def test_episode_start_resets():
    st = N1GateState(0.5, 0.5, 1, 2)
    inner = FakeInner([_hm(0.1)])
    cli = N1GateClient(inner, st)
    cli.infer({})  # push toward skipping
    assert not st.searching
    cli.episode_start(experiment="e", task="t", episode_id=0)
    assert inner.episode_starts == 1 and st.searching


def test_fail_open_on_nan_in_searching():
    st = N1GateState(0.5, 0.5, 1, 2)
    inner = FakeInner([_hm(float("nan"))])
    cli = N1GateClient(inner, st)
    cli.infer({})
    # anomaly on a searching step -> fail open, stay/return to full search.
    assert st.searching and st.low_run == 0 and st.decide() == "search"


def test_fail_open_on_nan_in_probe():
    # Drive into skipping; with M=2 the first stopped step is a skip and the
    # second is the probe. Feed NaN on the probe step -> must resume full search.
    st = N1GateState(0.5, 0.5, 1, 2)
    inner = FakeInner([_hm(0.1), _hm(0.0), _hm(float("nan"))])
    cli = N1GateClient(inner, st)
    cli.infer({})            # step0: search, 0.1 -> skipping
    assert not st.searching
    cli.infer({})            # step1: skip (score not read)
    assert not st.searching
    cli.infer({})            # step2: probe, NaN -> fail open
    assert st.searching and st.low_run == 0 and st.decide() == "search"


def test_fail_open_on_missing_hit_meta():
    st = N1GateState(0.5, 0.5, 1, 2)
    inner = FakeInner([{"actions": [0]}])  # no __hit_meta__
    cli = N1GateClient(inner, st)
    cli.infer({})
    assert st.searching and st.decide() == "search"


def test_inner_exception_propagates():
    class Boom:
        def infer(self, obs):
            raise RuntimeError("boom")

    cli = N1GateClient(Boom(), N1GateState(0.5, 0.5, 1, 2))
    with pytest.raises(RuntimeError, match="boom"):
        cli.infer({})


def test_delegates_unknown_methods():
    inner = FakeInner([])
    cli = N1GateClient(inner, N1GateState(0.5, 0.5, 1, 2))
    assert cli.select_bundle("b") == ("bundle", "b")


def test_factory_builds_wrapped_client():
    factory = make_n1_client_factory(
        {"theta_low": 0.5, "theta_high": 0.5, "j": 1, "M": 2},
        inner_factory=lambda server: FakeInner([]),
    )
    cli = factory(("host", 8000))
    assert isinstance(cli, N1GateClient)


# ----------------------------------------------------------------------
# env param parsing
# ----------------------------------------------------------------------
def test_env_params_ok():
    env = {"N1_THETA_LOW": "0.96", "N1_THETA_HIGH": "0.97", "N1_J": "3", "N1_M": "5"}
    assert n1_params_from_env(env) == {"theta_low": 0.96, "theta_high": 0.97, "j": 3, "M": 5}


def test_env_params_none_m():
    env = {"N1_THETA_LOW": "0.96", "N1_THETA_HIGH": "0.97", "N1_J": "3", "N1_M": "none"}
    assert n1_params_from_env(env)["M"] is None


def test_env_params_missing_raises():
    with pytest.raises(ValueError):
        n1_params_from_env({"N1_THETA_LOW": "0.96"})


def test_env_params_malformed_raises():
    with pytest.raises(ValueError):
        n1_params_from_env({"N1_THETA_LOW": "x", "N1_THETA_HIGH": "0.97", "N1_J": "3", "N1_M": "5"})


# ----------------------------------------------------------------------
# run_n1_live dispatch helpers
# ----------------------------------------------------------------------
def _ns(**kw):
    base = dict(theta_low=None, theta_high=None, j=None, M=run_n1_live.M_UNSET, matched_to=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_single_yaml(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1")
    assert run_n1_live.single_yaml(tmp_path).name == "a.yaml"
    (tmp_path / "b.yaml").write_text("y: 2")
    with pytest.raises(SystemExit):
        run_n1_live.single_yaml(tmp_path)


def test_gate_info_client_controlled(tmp_path):
    y = tmp_path / "cc.yaml"
    y.write_text("checkpoints:\n  cp1:\n    gate:\n      type: client_controlled\n")
    assert run_n1_live.gate_info(y) == {"type": "client_controlled", "cache_len": None,
                                        "inference_len": None, "lock_streak": None, "budget": None}


def test_gate_info_periodic(tmp_path):
    y = tmp_path / "p.yaml"
    y.write_text("checkpoints:\n  cp1:\n    gate:\n      type: periodic\n"
                 "      cache_len: 13\n      inference_len: 2\n")
    assert run_n1_live.gate_info(y) == {"type": "periodic", "cache_len": 13, "inference_len": 2,
                                        "lock_streak": None, "budget": None}


def test_resolve_worker_client_controlled_sets_env(monkeypatch):
    monkeypatch.delenv("N1_THETA_LOW", raising=False)
    args = _ns(theta_low=0.96, theta_high=0.97, j=3, M=5)
    mod = run_n1_live._resolve_worker_and_env({"type": "client_controlled"}, args)
    assert mod == run_n1_live.N1_WORKER_MODULE
    import os
    assert n1_params_from_env(os.environ)["j"] == 3  # env now parseable


def test_resolve_worker_client_controlled_requires_theta_and_m():
    # missing thresholds -> error
    with pytest.raises(SystemExit):
        run_n1_live._resolve_worker_and_env({"type": "client_controlled"}, _ns())
    # thresholds present but --M absent (M_UNSET) -> error
    with pytest.raises(SystemExit):
        run_n1_live._resolve_worker_and_env(
            {"type": "client_controlled"}, _ns(theta_low=0.9, theta_high=0.9, j=1))


def test_resolve_worker_client_controlled_m_none_ok(monkeypatch):
    # explicit --M none (never-probe) must run, not be treated as absent.
    monkeypatch.delenv("N1_M", raising=False)
    mod = run_n1_live._resolve_worker_and_env(
        {"type": "client_controlled"}, _ns(theta_low=0.9, theta_high=0.9, j=1, M=None))
    assert mod == run_n1_live.N1_WORKER_MODULE
    import os
    assert os.environ["N1_M"] == "none"


def test_resolve_worker_rejects_bad_params():
    # theta_high < theta_low -> N1GateState validation fails fast in the driver.
    with pytest.raises(SystemExit):
        run_n1_live._resolve_worker_and_env(
            {"type": "client_controlled"}, _ns(theta_low=0.9, theta_high=0.1, j=1, M=2))


def test_resolve_worker_periodic():
    mod = run_n1_live._resolve_worker_and_env(
        {"type": "periodic", "cache_len": 13, "inference_len": 2}, _ns())
    assert mod == run_n1_live.DEFAULT_WORKER_MODULE


def test_resolve_worker_unknown_gate():
    with pytest.raises(SystemExit):
        run_n1_live._resolve_worker_and_env({"type": "always_search"}, _ns())


def test_normalize_m():
    a = _ns(M="none")
    run_n1_live._normalize_m(a)
    assert a.M is None
    a = _ns(M="3")
    run_n1_live._normalize_m(a)
    assert a.M == 3
    a = _ns()  # flag not given -> sentinel preserved
    run_n1_live._normalize_m(a)
    assert a.M == run_n1_live.M_UNSET


def test_build_manifest_fields(tmp_path):
    y = tmp_path / "cfg.yaml"
    y.write_text("x: 1")
    args = argparse.Namespace(
        run_id="r1", task_suite="libero_spatial", point="A",
        theta_low=0.96, theta_high=0.97, j=3, M=3, matched_to=None, replan_steps=5,
        journal="j", per_step_out="p", baseline_journal="bj", baseline_gate_rows="bg",
        baseline_yaml_id=None, task_ids="0-9", eval_trials=50)
    m = run_n1_live.build_manifest(
        args, y, {"type": "client_controlled", "cache_len": None, "inference_len": None})
    assert m["gate_type"] == "client_controlled" and m["M"] == 3 and m["replan_steps"] == 5
    assert m["baseline_yaml_id"] == "cfg" and m["n_episodes"] == 500 and m["matched_to"] is None


# ----------------------------------------------------------------------
# n1 client_controlled yaml config validation (loads + builds the gate)
# ----------------------------------------------------------------------
_REPO = pathlib.Path(__file__).resolve().parents[2]
_N1_YAMLS = [
    _REPO / "exp/gate_research/config/libero_spatial/n1"
    "/cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@43__d1__fh75_ws10_quantile.yaml",
    _REPO / "exp/gate_research/config/libero_10/n1"
    "/cp1_spatial_pool_16__grid3_vision_0@56_vision_1@25_robot_state@18__d1__fh5_ws40_quantile.yaml",
]


@pytest.mark.parametrize("yaml_path", _N1_YAMLS, ids=lambda p: p.parent.parent.name)
def test_n1_yaml_validates_and_builds_client_gate(yaml_path):
    from openpi.cache.config import _build_gate, load_cache_config
    from openpi.cache.components.gate import ClientControlledGate

    cfg = load_cache_config(str(yaml_path))
    gate_cfg = cfg.checkpoints["cp1"].gate
    assert gate_cfg.type == "client_controlled"
    assert isinstance(_build_gate(gate_cfg), ClientControlledGate)
