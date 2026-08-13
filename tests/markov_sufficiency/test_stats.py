"""Tests for the pre-registered statistical primitives.

The sample-size and power numbers are asserted against the values written into
the approved plan: if these drift, the plan's design section is no longer
backed by the code that will run it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from exp.markov_sufficiency import _stats


# ------------------------------------------------------------------
# Exact McNemar
# ------------------------------------------------------------------


def test_mcnemar_five_zero_cannot_reach_significance():
    # The arithmetic that forced the sample-size redesign: 5 net discordant
    # pairs all in one direction still gives p = 2 * 0.5^5.
    result = _stats.mcnemar_exact(5, 0)
    assert result.p_value == pytest.approx(0.0625, abs=1e-9)


def test_mcnemar_symmetric_and_empty():
    assert _stats.mcnemar_exact(3, 3).p_value == pytest.approx(1.0)
    assert _stats.mcnemar_exact(0, 0).p_value == 1.0


# ------------------------------------------------------------------
# Sample size / power (plan §3.5.1, §3.3.1c)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pi_d", "delta", "expected"),
    [
        (0.080, 0.05, 249),
        (0.120, 0.05, 374),
        (0.275, 0.05, 861),
        (0.310, 0.05, 971),
        (0.080, 0.04, 390),
        (0.275, 0.04, 1347),
    ],
)
def test_mcnemar_power_n_matches_plan_table(pi_d, delta, expected):
    assert round(_stats.mcnemar_power_n(pi_d, delta)) == expected


def test_mcnemar_power_n_unreachable_effect():
    assert math.isinf(_stats.mcnemar_power_n(0.001, 0.5))


@pytest.mark.parametrize(
    ("n1", "n0", "expected"),
    [(247, 676, 0.67), (456, 448, 0.78)],
)
def test_two_proportion_power_worst_case_matches_plan(n1, n0, expected):
    # Worst-case baseline p_bar = 0.5 with the Holm-adjusted alpha.
    got = _stats.two_proportion_power(n1, n0, delta=0.10, p_bar=0.5, alpha=0.025)
    assert got == pytest.approx(expected, abs=0.01)


# ------------------------------------------------------------------
# Multiplicity
# ------------------------------------------------------------------


def test_holm_step_down():
    assert _stats.holm([0.01, 0.04], alpha=0.05) == [True, True]
    assert _stats.holm([0.03, 0.04], alpha=0.05) == [False, False]


def test_holm_adjusted_levels_are_tighter_than_nominal():
    levels = _stats.holm_adjusted_levels([0.01, 0.20], alpha=0.05)
    # Smallest p-value gets alpha/m, the other gets alpha/1.
    assert levels[0] == pytest.approx(0.975)
    assert levels[1] == pytest.approx(0.95)
    assert min(levels) >= 0.95


# ------------------------------------------------------------------
# Cluster-level inference
# ------------------------------------------------------------------


def test_cluster_sign_permutation_detects_shift_and_ignores_noise():
    shifted = _stats.cluster_sign_permutation([0.5] * 30, n_permutations=5000, seed=1)
    assert shifted.p_value < 0.01

    rng = np.random.default_rng(0)
    null = _stats.cluster_sign_permutation(rng.normal(0, 1, 40), n_permutations=5000, seed=2)
    assert null.p_value > 0.05


def test_cluster_bootstrap_ci_brackets_the_estimate():
    values = [{"x": v} for v in np.linspace(0.0, 1.0, 60)]
    result = _stats.cluster_bootstrap_ci(
        values, lambda items: float(np.mean([i["x"] for i in items])), n_resamples=500, seed=3
    )
    assert result.low < result.estimate < result.high
    assert result.level == 0.95


def test_cluster_bootstrap_ci_respects_strata():
    items = [{"x": 0.0, "g": "a"}] * 20 + [{"x": 1.0, "g": "b"}] * 20
    strata = [i["g"] for i in items]
    result = _stats.cluster_bootstrap_ci(
        items, lambda xs: float(np.mean([i["x"] for i in xs])), n_resamples=300, seed=4, strata=strata
    )
    # Stratified resampling keeps the group balance, so the mean stays at 0.5.
    assert result.low == pytest.approx(0.5, abs=1e-9)
    assert result.high == pytest.approx(0.5, abs=1e-9)


# ------------------------------------------------------------------
# CMH
# ------------------------------------------------------------------


def test_cmh_detects_consistent_association():
    tables = [(20, 5, 5, 20), (18, 7, 6, 19)]
    assert _stats.cmh_test(tables).p_value < 0.01


def test_cmh_null_when_no_association():
    tables = [(10, 10, 10, 10), (12, 12, 12, 12)]
    assert _stats.cmh_test(tables).p_value > 0.5


# ------------------------------------------------------------------
# Regressions found in G2 round 1
# ------------------------------------------------------------------


def test_cmh_continuity_correction_is_clamped_at_zero():
    """An exactly balanced table must score 0, not 0.25/var."""
    result = _stats.cmh_test([(10, 10, 10, 10)])
    assert result.statistic == 0.0
    assert result.p_value == pytest.approx(1.0)


def test_cmh_small_imbalance_stays_below_the_correction():
    # |O - E| = 0.5 exactly: the correction cancels it, so still zero.
    assert _stats.cmh_test([(6, 4, 5, 5)]).statistic == pytest.approx(0.0, abs=1e-12)


def test_mcnemar_estimate_is_the_paired_risk_difference():
    """The denominator is the paired-episode count, not the discordant count."""
    result = _stats.mcnemar_exact(6, 2, n_pairs=100)
    assert result.estimate == pytest.approx(0.04)
    assert result.n == 100
    # Without n_pairs the degenerate all-discordant convention applies.
    assert _stats.mcnemar_exact(6, 2).estimate == pytest.approx(0.5)


def test_mcnemar_zero_discordant_with_known_pairs():
    result = _stats.mcnemar_exact(0, 0, n_pairs=200)
    assert result.estimate == 0.0
    assert result.n == 200


@pytest.mark.parametrize(("k", "n"), [(74, 100), (0, 50), (50, 50)])
def test_wilson_ci_brackets_the_rate_and_stays_in_range(k, n):
    ci = _stats.wilson_ci(k, n)
    assert 0.0 <= ci.low <= ci.estimate <= ci.high <= 1.0


def test_wilson_ci_matches_a_known_value():
    ci = _stats.wilson_ci(74, 100)
    assert ci.low == pytest.approx(0.6463, abs=1e-3)
    assert ci.high == pytest.approx(0.8160, abs=1e-3)
