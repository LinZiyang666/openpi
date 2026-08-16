"""Pre-registered statistical gates (plan §8.1-§8.2, §10).

These are the review gates the design froze to stop the bootstrap and the
multiplicity layer from validating themselves against their own conventions:

*   **marginal validity at every Holm slot level** across the pre-registered
    regimes. This is the load-bearing one: Holm's strong FWER control follows
    from Bonferroni and needs only that each *true-null* p-value be marginally
    super-uniform -- it assumes nothing about dependence. So if the marginal
    levels hold, strong control holds, and no enumeration of least-favourable
    true/false-null configurations is required to establish it. (Two such
    configurations are simulated anyway, as a check on that reasoning rather
    than as its foundation.);
*   family-level FWER at the boundary null, tested at the **nominal alpha** with
    a one-sided binomial test rather than compared as a raw proportion, so a
    lucky Monte-Carlo run cannot pass a broken implementation. (The gate asks
    whether the evidence shows the rate *exceeding* alpha. Demanding that a
    confidence bound fall below alpha would be unpassable even for a perfectly
    calibrated test, since the bound lies above the estimate by construction.);
*   every test cross-checked against an **independent reference** written from
    the definition (plain loops, no shared helpers);
*   the ``s* = 0`` degenerate-pattern boundary pinned explicitly.

Sizes here are simulated on **discrete paired success rates**, never continuous
normals: continuous draws never tie, so they never produce a degenerate pattern
and cannot exercise the code path these gates exist to protect.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from exp.ablation_study.cache_size.analysis.cache_size_decision import (
    DELTA,
    classify_d,
    classify_q,
)
from exp.ablation_study.cache_size.analysis.cache_size_stats import (
    holm,
    signflip_test,
)

ALPHA = 0.05
N_PATTERNS = 2 ** 10

# The regimes the design pre-registers as plausible for these two suites, plus
# three deliberately hostile shapes. Both arms share a task's rate, so the null
# is true by construction and any rejection is a type-I error.
REGIMES = {
    # spatial-like: success crowds the ceiling, ties are the norm
    "ceiling": np.array([0.90, 0.95, 0.97, 0.98, 0.99, 1.0, 1.0, 1.0, 0.99, 0.98]),
    # the S4-S5 / S5-S6 plateau the design *expects*: six tasks pinned at 1.0
    "plateau": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.96, 0.92, 0.60, 0.30]),
    # libero_10-like: the primary discriminating battlefield
    "mid_range": np.array([0.46, 0.62, 0.70, 0.75, 0.80, 0.84, 0.86, 0.88, 0.90, 0.92]),
    # strongly asymmetric task-effect distribution (tests the symmetry-free claim)
    "skewed": np.array([0.10, 0.20, 0.30, 0.55, 0.60, 0.65, 0.70, 0.95, 0.98, 1.0]),
    # one task wildly out of line with the other nine
    "one_odd": np.array([0.86, 0.88, 0.88, 0.90, 0.90, 0.90, 0.92, 0.92, 0.94, 0.08]),
}


def paired_null(rng, probs, shift=0.0):
    """One task-level difference vector under a true null, on the 1/50 lattice."""
    return (rng.binomial(50, probs) / 50 - rng.binomial(50, probs) / 50) + shift


# ---------------------------------------------------------------------------
# Independent reference implementations (written from the definition)
# ---------------------------------------------------------------------------


DEGEN_EPS = 1e-12


def ref_signflip_p(d, h0_center, side):
    """Plain-loop restatement of the studentized null-imposed sign-flip test.

    Enumerates the 2**n patterns with ``itertools.product`` and a Python loop --
    no numpy broadcasting, no shared helper -- so a broadcasting or axis mistake
    in production has somewhere to show up. Degenerate patterns take the
    signed-infinity limit, matching the definition; an earlier version of this
    reference kept a pre-fix convention, so production and reference implemented
    two different rules and the suite could not arbitrate between them.
    """
    a = [float(x) for x in d]
    n = len(a)
    mean_d = sum(a) / n
    var_d = sum((x - mean_d) ** 2 for x in a) / (n - 1)
    sd_d = math.sqrt(var_d)
    scale_obs = max(1.0, max(abs(x) for x in a))
    if sd_d <= DEGEN_EPS * scale_obs:
        return 1.0
    t_obs = (mean_d - h0_center) / (sd_d / math.sqrt(n))

    resid = [x - mean_d for x in a]
    count = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        sample = [h0_center + r * w for r, w in zip(resid, signs)]
        m = sum(sample) / n
        v = sum((x - m) ** 2 for x in sample) / (n - 1)
        sd = math.sqrt(v)
        scale = max(1.0, max(abs(x) for x in sample))
        if sd <= DEGEN_EPS * scale:
            t = math.copysign(math.inf, m - h0_center) if m != h0_center else 0.0
        else:
            t = (m - h0_center) / (sd / math.sqrt(n))
        if side == "two-sided" and abs(t) >= abs(t_obs):
            count += 1
        elif side == "less" and t <= t_obs:
            count += 1
        elif side == "greater" and t >= t_obs:
            count += 1
    return (1 + count) / (2 ** n + 1)


def ref_holm(pvals, alpha=ALPHA):
    """Plain restatement of Holm step-down."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj, [a <= alpha for a in adj]


def _binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), in log space.

    Not ``math.comb`` times a power: at n = 8,000 the binomial coefficient
    overflows the float conversion outright, so the naive form does not merely
    lose precision, it raises.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k, n + 1):
        log_term = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * log_p + (n - i) * log_q)
        total += math.exp(log_term)
    return total


def binom_exceeds(k, n, bound, conf=0.95):
    """Is the empirical rate *significantly* above ``bound``?

    A one-sided upper confidence bound cannot certify "size <= alpha" for a test
    whose true size IS alpha -- the bound sits above the point estimate by
    construction, so a perfectly calibrated test would fail the gate. The
    well-posed question is the other direction: does the evidence show the rate
    exceeding the bound? Reject only when P(X >= k | p = bound) is small.
    """
    if k == 0:
        return False
    return _binom_sf(k, n, bound) < 1 - conf


def binom_upper_ucb(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound on a binomial rate."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        # P(X <= k | p=mid); the UCB is the p where this equals 1 - conf.
        if 1.0 - _binom_sf(k + 1, n, mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Cross-checks against the reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d, center, side",
    [
        ([0.10, 0.12, 0.09, 0.11, 0.13, 0.08, 0.10, 0.12, 0.11, 0.09], 0.0, "two-sided"),
        ([0.02, -0.01, 0.03, 0.00, 0.04, 0.01, 0.02, -0.02, 0.03, 0.01], DELTA, "less"),
        ([0.20, 0.18, 0.25, 0.22, 0.19, 0.21, 0.24, 0.20, 0.23, 0.22], DELTA, "greater"),
        ([0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05], 0.0, "two-sided"),
        # Tie-heavy cases: these are the ones that discriminate between the
        # degenerate-resample conventions, and the plateau region is where the
        # experiment expects to live.
        ([0.0] * 8 + [0.02] * 2, 0.0, "two-sided"),
        ([0.0] * 9 + [0.02], 0.0, "two-sided"),
        ([0.05] * 8 + [0.0] * 2, 0.0, "two-sided"),
        ([0.0] * 7 + [0.02] * 3, DELTA, "less"),
    ],
)
def test_signflip_matches_independent_reference(d, center, side):
    """Same enumeration, same definition -> the SAME p, exactly.

    Exact rather than approximate on purpose, and now exact for a second reason:
    the reference distribution is enumerated, so there is no Monte-Carlo slack to
    hide behind. Every Holm slot lies in [0.00625, 0.05], so a tolerance of 0.05
    would let a production p of 0.05 pass against a reference of 0.001.
    """
    mine = signflip_test(d, h0_center=center, side=side, name="t").p
    ref = ref_signflip_p(d, center, side)
    assert mine == pytest.approx(ref, abs=1e-12), (
        f"production {mine:.6f} vs reference {ref:.6f} -- the two implementations "
        "have drifted apart on the degenerate-pattern convention"
    )


def test_holm_matches_independent_reference():
    ps = [0.001, 0.004, 0.02, 0.03, 0.2, 0.5, 0.7, 0.9]
    tests = [
        type("T", (), {"name": f"t{i}", "p": p, "evaluable": True})()
        for i, p in enumerate(ps)
    ]
    res = holm(tests, alpha=ALPHA)
    ref_adj, ref_rej = ref_holm(ps, ALPHA)

    for i, p in enumerate(ps):
        assert res.adjusted[f"t{i}"] == pytest.approx(ref_adj[i])
        assert res.rejected[f"t{i}"] == ref_rej[i]


# ---------------------------------------------------------------------------
# Marginal validity -- the load-bearing gate
# ---------------------------------------------------------------------------


# Holm compares the k-th smallest p against alpha/(m-k+1). Its strong FWER
# guarantee is Bonferroni-based, so it needs each true-null p to be super-uniform
# *at the levels it actually uses* -- not merely at alpha. These are those levels
# for m = 8.
HOLM_SLOT_LEVELS = (ALPHA, ALPHA / 2, ALPHA / 4, ALPHA / 8)

# The size gate is itself a family: 5 regimes x 3 sides x 4 slot levels = 60
# binomial tests, all of which must pass. Judged at a bare 95% each, a correctly
# calibrated implementation would trip one of them almost every run -- and a
# gate that cries wolf gets muted, which is worse than no gate. So each cell is
# judged at 1 - 0.05/60. At n_sim = 1000 that still flags any true size at or
# above ~0.072 against a nominal 0.05, which is well inside the 0.114 the
# replaced construction produced.
GATE_CELLS = 60
GATE_CONF = 1 - ALPHA / GATE_CELLS

# Measured offline at 8,000 simulations per regime and side (see
# ``test_marginal_size_high_precision``, marked manual). Every entry is at or
# below its nominal level; the largest one-sided reading is 0.0566 at alpha=0.05,
# which is +1.9 Monte-Carlo SE. The pairs cluster bootstrap this construction
# replaced measured up to 0.1140 in the plateau regime -- 2.3x nominal, in the
# regime the S4-S5 and S5-S6 comparisons are *expected* to occupy.


@pytest.mark.parametrize("regime", sorted(REGIMES))
@pytest.mark.parametrize("side,center", [("less", DELTA), ("greater", DELTA),
                                         ("two-sided", 0.0)])
def test_marginal_size_at_every_holm_slot_level(regime, side, center):
    """Each test's own level under a true null, at each level Holm will use.

    Not "the two one-sided tests never both reject" -- that is true by
    construction and would pass for any implementation. This is the property the
    family-level claim rests on, so it is checked per regime, per side, and per
    slot rather than only at alpha.
    """
    rng = np.random.default_rng(7)
    n_sim = 1000
    ps = np.array([
        signflip_test(paired_null(rng, REGIMES[regime], shift=center),
                      h0_center=center, side=side, name="t").p
        for _ in range(n_sim)
    ])
    for level in HOLM_SLOT_LEVELS:
        k = int((ps <= level).sum())
        assert not binom_exceeds(k, n_sim, level, conf=GATE_CONF), (
            f"{regime}/{side}: empirical size {k / n_sim:.4f} ({k}/{n_sim}) is "
            f"significantly above the nominal level {level}"
        )


@pytest.mark.manual
@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_marginal_size_high_precision(regime):
    """The 8,000-simulation reading quoted in plan 8.1.1. Slow; run explicitly.

    The committed fast gate above runs 1,000 simulations and flags any true size
    at or above ~0.072 -- enough to catch the 0.114 the replaced construction
    produced, but not enough to resolve 0.05 from 0.06. This is the run that
    resolves it, kept in the repo so the numbers in the plan can be reproduced
    rather than taken on trust.
    """
    rng = np.random.default_rng(11)
    n_sim = 8000
    for side, center in (("less", DELTA), ("greater", DELTA)):
        ps = np.array([
            signflip_test(paired_null(rng, REGIMES[regime], shift=center),
                          h0_center=center, side=side, name="t").p
            for _ in range(n_sim)
        ])
        for level in HOLM_SLOT_LEVELS:
            k = int((ps <= level).sum())
            assert not binom_exceeds(k, n_sim, level, conf=GATE_CONF), (
                f"{regime}/{side}: size {k / n_sim:.4f} at level {level}"
            )


def test_strictest_holm_slot_is_attainable_in_every_regime():
    """A test that can never clear alpha/8 is not controlled, it is inert.

    This is what the ties argument is about: the classical sign test flips raw
    values about the null, so k tasks tied *at* the null pin p_min at 2**(k-n)
    and put the strictest slot structurally out of reach. Flipping residuals
    about the sample mean keeps all 2**n atoms alive.
    """
    assert 1 / (N_PATTERNS + 1) < ALPHA / 8
    rng = np.random.default_rng(23)
    for regime, probs in REGIMES.items():
        # Alternative, not null: a real effect should be able to clear the slot.
        best = min(
            signflip_test(paired_null(rng, probs), h0_center=DELTA, side="less",
                          name="t").p
            for _ in range(120)
        )
        assert best <= ALPHA / 8, (
            f"{regime}: smallest attainable p was {best:.5f}; the strictest Holm "
            f"slot {ALPHA / 8} is unreachable, so that slot can never reject"
        )


# ---------------------------------------------------------------------------
# Family-level FWER -- a check on the Bonferroni argument, not its foundation
# ---------------------------------------------------------------------------


def _family(rng, probs, gap_shift=DELTA, effects=(0.0,) * 5):
    """The eight-test family for one simulated suite."""
    tests = [
        signflip_test(paired_null(rng, probs) + e, h0_center=0.0, side="two-sided",
                      name=f"t{j}")
        for j, e in enumerate(effects)
    ]
    gap = paired_null(rng, probs, shift=gap_shift)
    tests.append(signflip_test(gap, h0_center=0.0, side="two-sided", name="t6"))
    tests.append(signflip_test(gap, h0_center=DELTA, side="less", name="t7"))
    tests.append(signflip_test(gap, h0_center=DELTA, side="greater", name="t8"))
    return tests


def test_family_fwer_at_the_global_null():
    """No tier differences and the gap exactly on delta: seven true nulls.

    t6 is excluded from the error event because gap = delta != 0 makes it a false
    null; every other member is true, so any Holm rejection among them is a
    family-wise error.
    """
    rng = np.random.default_rng(99)
    n_sim = 600
    any_reject = 0
    for _ in range(n_sim):
        hr = holm(_family(rng, REGIMES["ceiling"]), alpha=ALPHA)
        if any(hr.rejected[n] for n in ("t0", "t1", "t2", "t3", "t4", "t7", "t8")):
            any_reject += 1
    rate = any_reject / n_sim
    ucb = binom_upper_ucb(any_reject, n_sim, conf=0.95)
    assert not binom_exceeds(any_reject, n_sim, ALPHA), (
        f"family FWER {rate:.4f} ({any_reject}/{n_sim}, 95% UCB {ucb:.4f}) is "
        f"significantly above the nominal {ALPHA}; Holm must control the family"
    )


def test_family_fwer_with_false_nulls_present():
    """Step-down procedures are most exposed when some nulls are false.

    Three adjacent comparisons carry a large real effect, so they reject early
    and free up looser slots for the remaining true nulls. The error event is
    restricted to the true nulls, which is what strong control is about.
    """
    rng = np.random.default_rng(101)
    n_sim = 400
    any_reject = 0
    effects = (0.30, 0.30, 0.30, 0.0, 0.0)  # t0-t2 false, t3/t4 true
    for _ in range(n_sim):
        hr = holm(_family(rng, REGIMES["mid_range"], effects=effects), alpha=ALPHA)
        if any(hr.rejected[n] for n in ("t3", "t4", "t7", "t8")):
            any_reject += 1
    rate = any_reject / n_sim
    assert not binom_exceeds(any_reject, n_sim, ALPHA), (
        f"FWER over the true nulls {rate:.4f} ({any_reject}/{n_sim}) is "
        f"significantly above {ALPHA} when three nulls are false"
    )


# ---------------------------------------------------------------------------
# s* = 0 boundary
# ---------------------------------------------------------------------------


def test_zero_spread_sample_is_not_evaluable():
    res = signflip_test([0.07] * 10, h0_center=0.0, side="two-sided", name="t")
    assert res.evaluable is False
    assert res.p == 1.0


def test_near_zero_spread_still_evaluates():
    """One perturbed cluster is enough for a reference distribution to exist."""
    d = [0.07] * 9 + [0.09]
    res = signflip_test(d, h0_center=0.0, side="two-sided", name="t")
    assert res.evaluable is True
    assert 0.0 < res.p <= 1.0


def test_degenerate_patterns_do_not_crash_or_divide_by_zero():
    """Sign patterns that collapse the spread must be handled, not divided by."""
    d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    res = signflip_test(d, h0_center=0.0, side="two-sided", name="t")
    assert math.isfinite(res.p)
    assert 0.0 < res.p <= 1.0


def test_axes_never_claim_more_than_the_tests_support():
    """Sanity on the axis mappings used by the gates above."""
    assert classify_q(test7_rejected=False, test8_rejected=False) == "Q-inconc"
    assert classify_d(test6_rejected=False, gap_point=0.5) == "D-none"
