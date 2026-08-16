"""Statistical primitives + decision tree for the cache-size ablation (plan §8)."""

from __future__ import annotations

import numpy as np
import pytest

from exp.ablation_study.cache_size.analysis.cache_size_decision import (
    DELTA,
    UnreachableCombination,
    classify_d,
    classify_m,
    classify_p,
    classify_q,
    decide,
)
from exp.ablation_study.cache_size.analysis.cache_size_stats import (
    cluster_bootstrap_ci,
    holm,
    not_evaluable,
    signflip_test,
)

B_FAST = 2000          # CI only -- the confirmatory tests are enumerated, not sampled
N_PATTERNS = 2 ** 10   # the whole reference distribution for a 10-cluster family


# ---------------------------------------------------------------------------
# Null calibration
# ---------------------------------------------------------------------------


def test_p_value_is_never_zero():
    """``(1 + count) / (2**n + 1)``.

    Unlike an exact randomization test, the identity pattern does not reproduce
    ``t_obs`` here -- the reference sample is centred on the null while the
    observed one need not be -- so the raw count really can be zero and a bare
    ``count / 2**n`` would hand back an invalid p.
    """
    d = [0.40] * 6 + [-0.02] * 4
    res = signflip_test(d, h0_center=DELTA, side="greater", name="t8")
    assert res.p > 0
    assert res.p >= 1 / (N_PATTERNS + 1)
    assert res.n_sign_patterns == N_PATTERNS


def test_shift_is_to_the_null_boundary_not_the_observed_mean():
    """A naive tail proportion of the *observed* distribution is a different number.

    Under H0: gap >= delta the data are shifted onto delta before resampling. If
    the implementation forgot to shift, the p-value for data centered well above
    delta would look extreme in the wrong direction.
    """
    d = [0.20] * 10  # far above delta, but with zero spread
    shifted = signflip_test(d, h0_center=DELTA, side="less", name="t7")
    # Degenerate spread -> studentized statistic is 0 everywhere -> p is ~1,
    # i.e. the test cannot confirm non-inferiority. A tail-proportion
    # implementation on the raw data would instead have returned ~0.
    assert shifted.p > 0.5


# NOTE: the boundary-size check lives in test_cache_size_prereg_gates.py, where
# it is judged by a binomial upper confidence bound at the nominal alpha. The
# bare-proportion version that used to sit here accepted twice the nominal rate.


def test_two_sided_and_one_sided_are_consistent():
    d = [0.10, 0.12, 0.09, 0.11, 0.13, 0.08, 0.10, 0.12, 0.11, 0.09]
    two = signflip_test(d, h0_center=0.0, side="two-sided", name="t6")
    right = signflip_test(d, h0_center=0.0, side="greater", name="t6r")
    assert two.p < 0.05 and right.p < 0.05
    assert right.p <= two.p + 1e-9


def test_ties_at_the_null_do_not_pin_p_to_a_lattice():
    """Flipping residuals, not raw values, is what makes ties survivable.

    The classical one-sample sign test flips ``d_t - h0``; a task tied *at the
    null* contributes nothing whichever sign it draws, so k such ties collapse
    the reference set to ``2**(n-k)`` atoms and pin ``p_min`` at ``2**(k-n)``. At
    k=2 that floor is 0.0078 -- above the strictest Holm slot 0.05/8 = 0.00625 --
    which would put that slot structurally out of reach in exactly the plateau
    regime the S4-S5 and S5-S6 comparisons are expected to land in.

    Here the flip is applied to residuals about the sample mean, which vanish
    only on an exact coincidence, so all 1,024 atoms survive and the strictest
    slot stays attainable. Ties still cost power through the variance; that is
    honest, disclosed, and not engineered away.
    """
    two_ties = [0.05] * 8 + [0.0] * 2
    three_ties = [0.05] * 7 + [0.0] * 3

    r2 = signflip_test(two_ties, h0_center=0.0, side="two-sided", name="t")
    r3 = signflip_test(three_ties, h0_center=0.0, side="two-sided", name="t")

    # The sign-test lattice would have pinned these at 2**-8 / 2**-7.
    assert r2.p not in (2 ** -8, 2 ** -7)
    assert r3.p not in (2 ** -7, 2 ** -6)
    # The strictest Holm slot is reachable in principle...
    assert 1 / (N_PATTERNS + 1) < 0.05 / 8
    # ...and these samples do clear it, which the sign test could not have.
    assert r2.p <= 0.05 / 8


def test_degenerate_patterns_take_the_signed_infinity_limit():
    """They belong in a tail, not the centre -- mapping them to 0 empties the tails.

    Pins the calibration directly: at a true null on the 1/50 grid with ceiling
    tasks, the t*=0 convention over-rejected at ~3x nominal.
    """
    rng = np.random.default_rng(11)
    probs = np.array([0.90, 0.95, 0.97, 0.98, 0.99, 1.0, 1.0, 1.0, 0.99, 0.98])
    rejects = n = 0
    for _ in range(200):
        d = rng.binomial(50, probs) / 50 - rng.binomial(50, probs) / 50
        if np.std(d, ddof=1) <= 1e-12:
            continue
        n += 1
        if signflip_test(d, h0_center=0.0, side="two-sided", name="t").p <= 0.05:
            rejects += 1
    assert n > 100, "need enough non-degenerate draws to judge"
    assert rejects / n <= 0.12, (
        f"empirical type-I {rejects / n:.3f} at a true null; the t*=0 convention "
        "measured 0.135 here"
    )


def test_the_reference_distribution_is_deterministic_and_exhaustive():
    """No seed, no B: two calls on the same data must agree bit for bit."""
    d = [0.02, -0.01, 0.0, 0.04, 0.03, -0.02, 0.01, 0.0, 0.05, 0.02]
    a = signflip_test(d, h0_center=0.0, side="two-sided", name="t")
    b = signflip_test(d, h0_center=0.0, side="two-sided", name="t")
    assert a.p == b.p and a.t_obs == b.t_obs
    assert a.n_sign_patterns == N_PATTERNS


def test_h0_cancels_so_the_two_tails_nest_exactly():
    """``t*`` is free of ``h0``, so the tails nest by construction, not by a seed.

    Under the sampled bootstrap this had to be bought with a coordinated seed,
    and even then ``p7 <= p6`` was violated ~2 in 3000 -- both times inside the
    Holm-critical band.
    """
    d = [0.09, 0.11, 0.10, 0.12, 0.08, 0.10, 0.11, 0.09, 0.13, 0.07]
    p6 = signflip_test(d, h0_center=0.0, side="two-sided", name="t6").p
    p_left = signflip_test(d, h0_center=0.0, side="less", name="tl").p
    p_right = signflip_test(d, h0_center=0.0, side="greater", name="tr").p
    assert p_right <= p6 + 1e-12
    # Complementary tails over the same 1024 atoms: counts sum to >= 1024.
    assert p_left + p_right >= 1.0


def test_zero_variance_is_not_evaluable_rather_than_p_one():
    """All clusters identical: no spread, so the super-population claim is unsupported."""
    res = signflip_test([0.05] * 10, h0_center=0.0, side="two-sided", name="t")
    assert res.evaluable is False
    assert "zero between-cluster variance" in res.note
    assert res.p == 1.0


# ---------------------------------------------------------------------------
# Holm family
# ---------------------------------------------------------------------------


def test_holm_family_is_eight_and_monotone():
    tests = [
        signflip_test([0.10, 0.12, 0.09, 0.11, 0.13, 0.08, 0.10, 0.12, 0.11, 0.09],
                       h0_center=0.0, side="two-sided", name=f"t{i}")
        for i in range(1, 9)
    ]
    res = holm(tests, alpha=0.05)
    assert len(res.adjusted) == 8
    vals = [res.adjusted[n] for n in res.order]
    assert vals == sorted(vals), "Holm adjusted p-values must be monotone"


def test_not_evaluable_keeps_family_size_but_never_rejects():
    """Degenerate tiers stay in the family (p=1) so the denominator is stable."""
    ne = not_evaluable("t8", "two-sided", 0.0, 10, "tiers identical")
    others = [
        signflip_test(
            [0.3, 0.28, 0.31, 0.29, 0.30, 0.32, 0.27, 0.30, 0.29, 0.31],
            h0_center=0.0, side="two-sided", name=f"t{i}",)
        for i in range(1, 8)
    ]
    res = holm([*others, ne], alpha=0.05)
    assert len(res.adjusted) == 8
    assert res.rejected["t8"] is False
    assert ne.evaluable is False


def test_holm_rejects_duplicate_test_names():
    """A name collision would silently shrink the family denominator."""
    a = not_evaluable("dup", "two-sided", 0.0, 10, "x")
    b = not_evaluable("dup", "two-sided", 0.0, 10, "y")
    with pytest.raises(ValueError, match="duplicate test names"):
        holm([a, b])


# ---------------------------------------------------------------------------
# CI: reporting only, with a deterministic degradation rule
# ---------------------------------------------------------------------------


def test_bca_degrades_deterministically_on_degenerate_jackknife():
    ci = cluster_bootstrap_ci([0.2] * 10, b=500, seed=5)
    assert ci.method == "percentile"
    assert "denominator" in ci.reason


def test_ci_covers_the_mean_for_a_clean_sample():
    d = [0.10, 0.12, 0.09, 0.11, 0.13, 0.08, 0.10, 0.12, 0.11, 0.09]
    ci = cluster_bootstrap_ci(d, b=B_FAST, seed=6)
    assert ci.lo < float(np.mean(d)) < ci.hi


# ---------------------------------------------------------------------------
# Decision tree: orthogonality, reachability, priority
# ---------------------------------------------------------------------------


def test_gap_ci_1pp_to_4pp_carries_both_facts():
    """The case that a one-dimensional classification loses.

    [+1pp, +4pp]: the teacher is ahead (test 6 rejects, point > 0) AND the cache
    is non-inferior (U < delta, test 7 rejects). Both must survive into the
    verdict, and the cell must be the "good enough" column.
    """
    d = classify_d(test6_rejected=True, gap_point=0.025)
    q = classify_q(test7_rejected=True, test8_rejected=False)
    assert (d, q) == ("D-teacher", "Q-pass")

    v = decide(d=d, q=q, p="P-yes", m_yes=False)
    assert v.branch == "C"
    assert "good enough" in v.headline
    assert "teacher remains statistically ahead" in v.headline


def test_cache_ahead_is_never_described_as_falling_short():
    d = classify_d(test6_rejected=True, gap_point=-0.06)
    q = classify_q(test7_rejected=True, test8_rejected=False)
    v = decide(d=d, q=q, p="P-yes", m_yes=False)
    assert v.d == "D-cache"
    assert "behind" not in v.headline
    assert any("counter-intuitive" in c for c in v.caveats)


def test_q_inconc_may_not_be_read_as_equivalence():
    v = decide(d="D-none", q="Q-inconc", p="P-inconc", m_yes=False)
    assert v.branch == "H"
    assert any("NOT evidence of equivalence" in c for c in v.caveats)


def test_branch_a_carries_index_and_descriptive_limits():
    v = decide(d="D-teacher", q="Q-fail", p="P-yes", m_yes=False)
    assert v.branch == "A"
    assert any("THIS index only" in c for c in v.caveats)
    assert any("descriptive" in c for c in v.caveats)


def test_m_yes_outranks_everything():
    v = decide(d="D-teacher", q="Q-fail", p="P-yes", m_yes=True)
    assert v.branch == "N"


@pytest.mark.parametrize("d, q", [("D-cache", "Q-fail"), ("D-cache", "Q-inconc")])
def test_unreachable_combinations_fail_loud(d, q):
    """Only the D-cache pairs are impossible: gap cannot be < 0 and > delta at once."""
    with pytest.raises(UnreachableCombination):
        decide(d=d, q=q, p="P-yes", m_yes=False)


def test_d_none_q_fail_is_reachable_and_reported_not_aborted():
    """Different tails, not an inconsistency.

    Test 8 is one-sided and test 6 two-sided, so a skewed bootstrap null can
    confirm "gap > delta" while leaving "gap != 0" unconfirmed at family level.
    A production-code search found real gap vectors doing exactly this (mean
    +0.312 -> Holm-adjusted 0.052 vs 0.026). Aborting on them would kill the
    analysis on a legitimate dataset.
    """
    v = decide(d="D-none", q="Q-fail", p="P-yes", m_yes=False)
    assert v.branch == "A"
    assert any("different tails" in c for c in v.caveats)
    assert any("not paraphrase the pair as agreement" in c for c in v.caveats)


def test_both_q_tests_rejecting_is_fatal():
    with pytest.raises(UnreachableCombination):
        classify_q(test7_rejected=True, test8_rejected=True)


def test_every_reachable_combination_lands_in_exactly_one_cell():
    reachable = [
        ("D-teacher", "Q-pass"), ("D-teacher", "Q-fail"), ("D-teacher", "Q-inconc"),
        ("D-cache", "Q-pass"),
        ("D-none", "Q-pass"), ("D-none", "Q-inconc"),
    ]
    branches = set()
    for p in ("P-yes", "P-no", "P-inconc"):
        for d, q in reachable:
            v = decide(d=d, q=q, p=p, m_yes=False)
            assert v.branch
            branches.add((p, q))
    assert len(branches) == 9, "P x Q must cover all nine cells"


def test_classification_is_driven_by_tests_not_by_ci():
    """A CI that clears delta must NOT produce Q-pass if the test did not reject."""
    q = classify_q(test7_rejected=False, test8_rejected=False)
    assert q == "Q-inconc"
    d = classify_d(test6_rejected=False, gap_point=0.09)
    assert d == "D-none"


def test_plateau_axis_thresholds():
    assert classify_p(slope5_ci_hi=0.01, slope6_ci_lo=-0.01, slope6_ci_hi=0.015) == "P-yes"
    assert classify_p(slope5_ci_hi=0.10, slope6_ci_lo=0.03, slope6_ci_hi=0.09) == "P-no"
    assert classify_p(slope5_ci_hi=0.01, slope6_ci_lo=0.005, slope6_ci_hi=0.05) == "P-inconc"


def test_m_axis_requires_a_confirmed_decrease():
    assert classify_m({"t3": True}, {"t3": -0.04}) is True
    assert classify_m({"t3": True}, {"t3": 0.04}) is False
    # CI-looking evidence without a Holm rejection is not enough.
    assert classify_m({"t3": False}, {"t3": -0.04}) is False
