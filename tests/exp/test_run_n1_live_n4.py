"""Unit tests for the Stage 3a N4 dispatch/provenance in run_n1_live (non-manual).

Mirrors the N1 runner-dispatch coverage in ``test_n1_gate_client.py`` /
``tests/review_tests/test_n1_live_stage1b_g2.py`` (calling ``_resolve_worker_and_env``
/ ``build_manifest`` directly): the N4 worker/env selection, ``--L`` fail-fast, the
N1/periodic backward-compatible paths, and the N4 manifest fields.
"""

from __future__ import annotations

import argparse
import os

import pytest

from exp.gate_research import run_n1_live
from exp.gate_research.n4_gate_client import n4_params_from_env


def _ns(**kw):
    """Namespace for _resolve_worker_and_env (defaults match the real argparse)."""
    base = dict(theta_low=None, theta_high=None, j=None, M=run_n1_live.M_UNSET,
                gate_family="n1", L=None, matched_to=None)
    base.update(kw)
    return argparse.Namespace(**base)


# ----------------------------------------------------------------------
# (test 16) N4 dispatch selects the N4 worker and sets N4_* env
# ----------------------------------------------------------------------
def test_resolve_worker_n4_sets_env(monkeypatch):
    for k in ("N4_THETA_LOW", "N4_THETA_HIGH", "N4_J", "N4_M", "N4_L"):
        monkeypatch.delenv(k, raising=False)
    args = _ns(gate_family="n4", theta_low=0.96, theta_high=0.97, j=3, M=5, L=8)
    mod = run_n1_live._resolve_worker_and_env({"type": "client_controlled"}, args)
    assert mod == run_n1_live.N4_WORKER_MODULE
    # env is now parseable by the N4 worker entry, with L carried through.
    assert n4_params_from_env(os.environ) == {
        "theta_low": 0.96, "theta_high": 0.97, "j": 3, "M": 5, "L": 8}


def test_resolve_worker_n4_m_none_ok(monkeypatch):
    monkeypatch.delenv("N4_M", raising=False)
    mod = run_n1_live._resolve_worker_and_env(
        {"type": "client_controlled"},
        _ns(gate_family="n4", theta_low=0.9, theta_high=0.9, j=1, M=None, L=6))
    assert mod == run_n1_live.N4_WORKER_MODULE
    assert os.environ["N4_M"] == "none"


# ----------------------------------------------------------------------
# (test 17) --L fail-fast in the driver (missing / illegal L)
# ----------------------------------------------------------------------
def test_resolve_worker_n4_requires_L():
    with pytest.raises(SystemExit, match="requires --L"):
        run_n1_live._resolve_worker_and_env(
            {"type": "client_controlled"},
            _ns(gate_family="n4", theta_low=0.9, theta_high=0.9, j=1, M=2, L=None))


@pytest.mark.parametrize("bad_L", [0, -1])
def test_resolve_worker_n4_rejects_bad_L(bad_L):
    # N4GateState validation runs in the driver -> SystemExit before spawning workers.
    with pytest.raises(SystemExit, match="invalid N4 params"):
        run_n1_live._resolve_worker_and_env(
            {"type": "client_controlled"},
            _ns(gate_family="n4", theta_low=0.9, theta_high=0.9, j=1, M=2, L=bad_L))


def test_resolve_worker_n4_still_requires_thresholds():
    # Missing thresholds is caught by the shared client_controlled guard first.
    with pytest.raises(SystemExit):
        run_n1_live._resolve_worker_and_env(
            {"type": "client_controlled"}, _ns(gate_family="n4", L=8))


# ----------------------------------------------------------------------
# (test 18) n1 / periodic paths are byte-compatible (default gate_family=n1)
# ----------------------------------------------------------------------
def test_resolve_worker_n1_default_unchanged(monkeypatch):
    monkeypatch.delenv("N1_THETA_LOW", raising=False)
    monkeypatch.delenv("N4_THETA_LOW", raising=False)
    args = _ns(theta_low=0.96, theta_high=0.97, j=3, M=5)  # gate_family default "n1"
    mod = run_n1_live._resolve_worker_and_env({"type": "client_controlled"}, args)
    assert mod == run_n1_live.N1_WORKER_MODULE
    assert os.environ["N1_THETA_LOW"] == repr(0.96)
    assert "N4_THETA_LOW" not in os.environ  # n1 path must not set N4_* env


def test_resolve_worker_periodic_unchanged():
    mod = run_n1_live._resolve_worker_and_env(
        {"type": "periodic", "cache_len": 13, "inference_len": 2}, _ns())
    assert mod == run_n1_live.DEFAULT_WORKER_MODULE


# ----------------------------------------------------------------------
# (test 19) build_manifest N4 provenance + N1/periodic compatible defaults
# ----------------------------------------------------------------------
def _manifest_args(**kw):
    base = dict(
        run_id="r", task_suite="libero_spatial", point="A",
        theta_low=0.96, theta_high=0.97, j=3, M=3, matched_to=None, replan_steps=5,
        journal="j", per_step_out="p", baseline_journal="bj", baseline_gate_rows="bg",
        baseline_yaml_id=None, task_ids="0-9", eval_trials=50)
    base.update(kw)
    return argparse.Namespace(**base)


def test_build_manifest_n4_fields(tmp_path):
    y = tmp_path / "cfg.yaml"
    y.write_text("x: 1")
    args = _manifest_args(gate_family="n4", L=8)
    m = run_n1_live.build_manifest(
        args, y, {"type": "client_controlled", "cache_len": None, "inference_len": None})
    assert m["gate_family"] == "n4" and m["L"] == 8
    assert m["gate_type"] == "client_controlled" and m["theta_low"] == 0.96


def test_build_manifest_defaults_when_fields_absent(tmp_path):
    # Older callers build args without gate_family/L; getattr keeps them working.
    y = tmp_path / "cfg.yaml"
    y.write_text("x: 1")
    args = _manifest_args()  # no gate_family / L attributes
    assert not hasattr(args, "gate_family") and not hasattr(args, "L")
    m = run_n1_live.build_manifest(
        args, y, {"type": "periodic", "cache_len": 7, "inference_len": 1})
    assert m["gate_family"] == "n1" and m["L"] is None
