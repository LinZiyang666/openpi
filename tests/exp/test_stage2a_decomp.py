"""Unit tests for Stage 2a SR-gain decomposition (all non-manual).

Covers the plan's locked contracts:
- Delta-inf decomposition identities (skip_conversion + verdict_mix == pooled
  d_inf; all three terms == episode-averaged d_inf) + hand-computed term values;
- paired SR: known 2x2 -> known chi2 + exact binomial p + Delta-SR;
- run-length stats split by success.
"""

from __future__ import annotations

import math

import pytest

from exp.gate_research import stage2a_sr_decomp as A
from exp.gate_research.stage2_common import EpisodeRec


def _ep(tid, idx, hit, searched, success, start_t=None):
    n = len(hit)
    return EpisodeRec(task_id=tid, subset_init_state_idx=idx, searched_seq=list(searched),
                      hit_type_seq=list(hit), start_t_seq=[start_t] * n,
                      cp1_score_seq=[0.9] * n, success=success)


# ---------------------------------------------------------------- delta-inf decomposition
def test_decompose_delta_inf_identity_and_terms():
    baseline = [_ep(0, 0, ["FULL_HIT", "MISS"], [True, True], True)]
    cond = [_ep(0, 0, ["FULL_HIT", "MISS"], [True, False], True)]  # 2nd step is a skip
    d = A.decompose_delta_inf(baseline, cond)
    # hand values: base_searched_inf=0.5, skip_frac=0.5, cond_searched_inf=0.0
    assert math.isclose(d["skip_conversion"], 0.25)
    assert math.isclose(d["verdict_mix"], -0.25)
    assert math.isclose(d["d_inf_pooled"], 0.0, abs_tol=1e-12)
    # pooled 2-term identity
    assert math.isclose(d["skip_conversion"] + d["verdict_mix"], d["d_inf_pooled"], abs_tol=1e-12)
    # 3-term identity against the episode-averaged total
    assert math.isclose(d["skip_conversion"] + d["verdict_mix"] + d["ep_length_residual"],
                        d["d_inf_epavg"], abs_tol=1e-12)


def test_decompose_delta_inf_pure_skip_conversion():
    # baseline all FULL_HIT (searched_inf=0); cond skips half -> pure skip conversion
    baseline = [_ep(0, 0, ["FULL_HIT", "FULL_HIT"], [True, True], True)]
    cond = [_ep(0, 0, ["FULL_HIT", "FULL_HIT"], [True, False], True)]
    d = A.decompose_delta_inf(baseline, cond)
    assert math.isclose(d["skip_conversion"], 0.5)  # 0.5 * (1 - 0)
    assert math.isclose(d["verdict_mix"], 0.0)
    assert math.isclose(d["d_inf_pooled"], 0.5)


# ---------------------------------------------------------------- paired SR
def test_paired_sr_known_2x2():
    units = [(0, 0), (0, 1), (0, 2), (0, 3)]
    base_succ = [True, True, False, False]
    cond_succ = [True, False, True, True]
    baseline = [_ep(0, i, ["FULL_HIT"], [True], s) for (_, i), s in zip(units, base_succ)]
    cond = [_ep(0, i, ["FULL_HIT"], [True], s) for (_, i), s in zip(units, cond_succ)]
    p = A.paired_sr(cond, baseline)
    assert (p["b"], p["c"]) == (1, 2)          # cond-fail/base-succeed=1, cond-succeed/base-fail=2
    assert math.isclose(p["mcnemar_chi2"], 0.0)  # (|1-2|-1)^2/3 = 0
    assert math.isclose(p["exact_p"], 1.0)       # 2*(C(3,0)+C(3,1))/8 = 1.0
    assert math.isclose(p["sr_delta_pp"], 25.0)  # 0.75 - 0.50


def test_paired_sr_fails_fast_on_unit_mismatch():
    baseline = [_ep(0, 0, ["FULL_HIT"], [True], True)]
    cond = [_ep(0, 0, ["FULL_HIT"], [True], True), _ep(0, 1, ["FULL_HIT"], [True], True)]
    with pytest.raises(ValueError):        # full-pairing fail-fast, never silent intersection
        A.paired_sr(cond, baseline)


def test_per_task_delta_sr_fails_fast_on_task_mismatch():
    baseline = [_ep(0, 0, ["FULL_HIT"], [True], True)]
    cond = [_ep(0, 0, ["FULL_HIT"], [True], True), _ep(1, 0, ["FULL_HIT"], [True], True)]
    with pytest.raises(ValueError):
        A.per_task_delta_sr(cond, baseline)


# ---------------------------------------------------------------- run-length by success
def test_run_length_stats_split_by_success():
    eps = [
        _ep(0, 0, ["FULL_HIT", "FULL_HIT", "MISS", "FULL_HIT"], [True] * 4, True),
        _ep(0, 1, ["MISS", "MISS"], [True, True], False),
    ]
    st = A.run_length_stats(eps, include_ws=False)
    assert st["success"]["n"] == 2 and st["success"]["max"] == 2   # runs [2, 1]
    assert math.isclose(st["success"]["mean"], 1.5)
    assert st["fail"]["n"] == 0                                    # no cache run in the fail ep


# ---------------------------------------------------------------- FH rate + WS
def test_searched_fh_rate_and_ws():
    eps = [_ep(0, 0, ["FULL_HIT", "WARM_START", "MISS"], [True, True, True], False, start_t=0.5)]
    assert math.isclose(A.searched_fh_rate(eps)["fh_rate"], 1 / 3)
    ws = A.ws_exec_by_success(eps)
    assert ws["fail_mean"] == 1.0 and ws["success_mean"] == 0.0
