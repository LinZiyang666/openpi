"""Tests for the frozen TRACER Phase-6 statistical protocol (plan §C)."""

import pytest

from exp.zixuan_proposal.phase6_stats import (
    N_MIN_CLUSTERS,
    N_MIN_FH,
    FullHitRow,
    claim2_verdict,
    estimable,
    exit_gate_pass,
    framework_verdict,
    holm_bonferroni,
    paired_delta_bhr_ci,
    suite_family_verdict,
)


def _rows(n_clusters: int, rows_per: int, bad_frac: float) -> list[FullHitRow]:
    """Deterministic FULL_HIT rows: `n_clusters` clusters x `rows_per` rows each,
    the first `bad_frac` fraction of each cluster's rows marked bad."""
    out: list[FullHitRow] = []
    n_bad = round(rows_per * bad_frac)
    for c in range(n_clusters):
        for i in range(rows_per):
            out.append(FullHitRow(cluster=(0, c), bad=(i < n_bad)))
    return out


# ------------------------------------------------------------------
# Estimability guard
# ------------------------------------------------------------------
def test_estimable_guard():
    # enough rows AND enough clusters
    assert estimable(_rows(N_MIN_CLUSTERS, N_MIN_FH // N_MIN_CLUSTERS + 1, 0.3))
    # too few FULL_HITs
    assert not estimable(_rows(N_MIN_CLUSTERS, 1, 0.3))
    # enough rows but too few clusters (all in one cluster)
    assert not estimable(_rows(1, N_MIN_FH + 10, 0.3))


# ------------------------------------------------------------------
# Clustered bootstrap
# ------------------------------------------------------------------
def test_bootstrap_deterministic_and_separates():
    a = _rows(40, 10, bad_frac=0.5)
    b = _rows(40, 10, bad_frac=0.05)
    r1 = paired_delta_bhr_ci(a, b)
    r2 = paired_delta_bhr_ci(a, b)
    assert r1 == r2  # seeded determinism
    assert r1["point"] < 0  # b has lower BHR
    assert r1["excludes_zero"] and r1["ci_high"] < 0


def test_bootstrap_requires_identical_cluster_sets():
    # Fully disjoint cluster sets -> mismatched -> reject.
    a = [FullHitRow(cluster=(0, 0), bad=True)] * 5
    b = [FullHitRow(cluster=(9, 9), bad=False)] * 5
    with pytest.raises(ValueError, match="mismatched cluster"):
        paired_delta_bhr_ci(a, b)


def test_bootstrap_rejects_partially_unpaired_cluster_sets():
    # A 31-cluster lane vs a 30-cluster lane must NOT mix an all-row point estimate
    # with a shared-only resample -> reject (independent G2 probe).
    a = [FullHitRow(cluster=(0, c), bad=False) for c in range(31)]
    b = [FullHitRow(cluster=(0, c), bad=True) for c in range(30)]
    with pytest.raises(ValueError, match="mismatched cluster"):
        paired_delta_bhr_ci(a, b, b_replicates=10)


# ------------------------------------------------------------------
# Verdicts
# ------------------------------------------------------------------
def test_framework_verdict():
    # insufficient power -> INSUFFICIENT regardless of separation
    assert framework_verdict(_rows(5, 5, 0.5), _rows(5, 5, 0.0))["status"] == "INSUFFICIENT"
    # estimable and b clearly better -> BHR_DOWN
    v = framework_verdict(_rows(40, 10, 0.5), _rows(40, 10, 0.02))
    assert v["status"] == "BHR_DOWN"
    # estimable but identical -> NO_CHANGE
    v2 = framework_verdict(_rows(40, 10, 0.3), _rows(40, 10, 0.3))
    assert v2["status"] == "NO_CHANGE"


def test_claim2_verdict():
    # under-powered -> INCONCLUSIVE
    assert claim2_verdict(_rows(5, 5, 0.3), _rows(5, 5, 0.3))["status"] == "INCONCLUSIVE"
    # equivalent within delta -> EQUIVALENT
    assert claim2_verdict(_rows(40, 10, 0.30), _rows(40, 10, 0.30))["status"] == "EQUIVALENT"
    # c strictly better -> SUPERIOR
    assert claim2_verdict(_rows(40, 10, 0.50), _rows(40, 10, 0.05))["status"] == "SUPERIOR"


# ------------------------------------------------------------------
# Multiplicity + joint exit gate
# ------------------------------------------------------------------
def test_holm_bonferroni_step_down():
    # smallest p passes alpha/2, next fails at alpha/1 -> one reject
    r = holm_bonferroni({"H_AB": 0.001, "H_BC": 0.5}, alpha=0.05)
    assert r == {"H_AB": True, "H_BC": False}
    # both tiny -> both reject
    r2 = holm_bonferroni({"H_AB": 0.001, "H_BC": 0.01}, alpha=0.05)
    assert r2 == {"H_AB": True, "H_BC": True}


def test_exit_gate_joint_conjuncts():
    ci = {"ci_high": -0.1}
    assert exit_gate_pass(ci, bhr_significant=True, sr_calib=0.85, sr_base=0.86, ir_calib=0.3, ir_illus=0.4)["pass"]
    # SR crash fails the gate even with BHR down
    v = exit_gate_pass(ci, bhr_significant=True, sr_calib=0.60, sr_base=0.86, ir_calib=0.3, ir_illus=0.4)
    assert not v["pass"] and not v["sr_ok"]


def test_suite_family_verdict_pass_and_sr_guard():
    a = _rows(40, 10, 0.5)
    b = _rows(40, 10, 0.02)
    c = _rows(40, 10, 0.02)
    v = suite_family_verdict(a, b, c, sr={"base": 0.86, "b": 0.85}, ir={"illus": 0.4, "b": 0.3})
    assert v["status"] == "PASS"
    assert v["holm_reject"]["H_AB"]
    # same BHR win but SR crashed -> FAIL
    v2 = suite_family_verdict(a, b, c, sr={"base": 0.86, "b": 0.55}, ir={"illus": 0.4, "b": 0.3})
    assert v2["status"] == "FAIL"


def test_family_claim2_bound_to_adjusted_bc_interval():
    # The per-suite family reports a Holm-ADJUSTED H_BC interval, and Claim-2 must be decided
    # on THAT interval (never a fresh unadjusted 95%). The adjusted interval is at least as wide
    # as the raw one, so it can never be strictly tighter.
    a = _rows(40, 10, 0.5)
    b = _rows(40, 10, 0.30)
    c = _rows(40, 10, 0.30)  # b == c -> equivalence region
    v = suite_family_verdict(a, b, c, sr={"base": 0.86, "b": 0.85}, ir={"illus": 0.4, "b": 0.3})
    raw_lo, raw_hi = v["H_BC"]["raw_ci"]
    adj_lo, adj_hi = v["H_BC"]["adjusted_ci"]
    assert adj_lo <= raw_lo + 1e-9 and adj_hi >= raw_hi - 1e-9  # adjusted no tighter than raw
    # Claim-2's reported interval is the adjusted one, not the raw 95%.
    assert v["claim2"]["ci_low"] == adj_lo and v["claim2"]["ci_high"] == adj_hi
