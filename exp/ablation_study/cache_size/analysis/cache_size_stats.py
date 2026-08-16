"""Statistical primitives for the cache-size ablation (plan §8.1-§8.4).

Everything here works on **task-level paired differences** ``d_t``. The estimand
is the mean effect over a task super-population, so the inference unit is the
task (10 clusters per suite), not the episode. That choice costs a great deal of
power -- it is stated in the plan and must be reported, not hidden.

Two rules keep the family internally consistent:

*   Every confirmatory p-value comes from a **studentized null-imposed sign-flip
    (wild) test, enumerated over all 2^10 = 1,024 patterns**. Each test imposes
    its own H0 by centring, so it is calibrated under that H0, and the shared
    statistic keeps the family internally consistent -- which is what makes the
    decision tree's two `D-cache` cells genuinely unreachable. The construction
    is deterministic: no seed, no B, no Monte Carlo error in any reported number.
    It replaced a pairs cluster bootstrap that measured up to 0.114 one-sided at
    a nominal 0.05 in the tie-heavy plateau regime; see `signflip_test`.
*   Confidence intervals are for effect-size reporting only. Cell assignment is
    driven by Holm-adjusted rejections; where the two disagree near a boundary,
    the test wins and the disagreement is disclosed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Sequence

import numpy as np

Side = Literal["two-sided", "less", "greater"]

DEFAULT_B = 10_000
DEFAULT_ALPHA = 0.05

# BCa degrades to percentile when the acceleration estimate is untrustworthy.
# Frozen as a deterministic function of the data so the choice cannot be made
# after seeing both intervals.
BCA_MAX_ABS_A = 0.25
BCA_MAX_ABS_Z0 = 0.5
BCA_MIN_DENOM = 1e-12


@dataclass(frozen=True)
class TestResult:
    """One confirmatory test: its p-value and everything needed to audit it."""

    name: str
    p: float
    side: Side
    h0_center: float
    t_obs: float
    n_clusters: int
    evaluable: bool = True
    note: str = ""
    degenerate_resample_fraction: float = 0.0
    n_sign_patterns: int = 0


@dataclass
class CI:
    lo: float
    hi: float
    method: Literal["bca", "percentile"]
    z0: float = 0.0
    a: float = 0.0
    reason: str = ""


@dataclass
class HolmResult:
    adjusted: dict[str, float]
    rejected: dict[str, bool]
    alpha: float = DEFAULT_ALPHA
    order: list[str] = field(default_factory=list)


# Spread below this multiple of the data's own scale counts as "no spread".
# An absolute `<= 0` test is not enough: ten copies of 0.07 leave a residual
# std around 1e-17 rather than an exact zero, which would otherwise divide into
# a t-statistic of ~1e16 and produce a spuriously significant p-value.
_REL_SPREAD_EPS = 1e-12


def _is_degenerate(sample: np.ndarray) -> bool:
    """True when between-cluster spread is zero up to floating-point residue."""
    if sample.size < 2:
        return True
    s = float(sample.std(ddof=1))
    scale = max(1.0, float(np.abs(sample).max()))
    return s <= _REL_SPREAD_EPS * scale


def _studentized(sample: np.ndarray, center: float) -> float:
    """t-like statistic; 0 when the sample has no usable spread."""
    if _is_degenerate(sample):
        return 0.0
    n = sample.size
    s = sample.std(ddof=1)
    return float((sample.mean() - center) / (s / np.sqrt(n)))


# The confirmatory reference distribution is enumerated, not sampled: with n
# clusters there are exactly 2**n sign patterns. At n = 10 that is 1,024 --
# cheaper than any Monte Carlo run large enough to be trustworthy, and it makes
# every p-value deterministic (no seed, no B, no simulation error). The cap
# keeps the enumeration from silently turning into a memory bomb if this module
# is ever reused with a larger cluster count.
MAX_ENUMERATED_CLUSTERS = 16


@lru_cache(maxsize=None)
def _sign_patterns(n: int) -> np.ndarray:
    """All 2**n Rademacher weight vectors, as a (2**n, n) float array."""
    if n > MAX_ENUMERATED_CLUSTERS:
        raise ValueError(
            f"exhaustive sign-flip needs 2**{n} patterns, above the "
            f"{MAX_ENUMERATED_CLUSTERS}-cluster cap; this family is n=10"
        )
    bits = np.arange(2 ** n, dtype=np.uint32)[:, None]
    return 1.0 - 2.0 * ((bits >> np.arange(n, dtype=np.uint32)) & 1).astype(float)


def signflip_test(
    d: Sequence[float],
    *,
    h0_center: float,
    side: Side,
    name: str,
) -> TestResult:
    """Studentized null-imposed sign-flip (wild) test, enumerated exhaustively.

    The null is imposed by centring -- ``R = h0 + (d - mean(d)) * w`` for every
    Rademacher weight vector ``w`` -- and the statistic is studentized, which is
    what makes the test valid without assuming ``d_t`` is symmetric about the
    null (Chung & Romano 2013: studentizing a randomization statistic buys
    asymptotic validity when the randomization hypothesis itself fails).

    Two properties matter for this design, and both were measured rather than
    assumed (plan 8.1.1):

    *   **Size.** The pairs cluster bootstrap this replaces ran at up to 0.114
        one-sided -- 2.3x nominal -- in the tie-heavy plateau regime that the
        S4-S5 and S5-S6 comparisons are *expected* to land in. Across the
        pre-registered finite-sample regimes, the sign-flip construction shows
        no multiplicity-adjusted evidence of exceeding any nominal Holm slot;
        some raw point estimates are slightly above nominal. Holm control relies
        on the construction's asymptotic marginal validity, with those n=10
        simulations as design-specific support rather than an exact guarantee.
    *   **Attainability.** Flipping *residuals about the sample mean* rather than
        raw values about the null is what keeps ties harmless. A task tied at
        ``d_t == h0`` contributes nothing under the classical sign test, so k
        such ties pin ``p_min`` at ``2**(k-n)`` and can put the strictest Holm
        slot structurally out of reach; residuals are zero only on an exact
        coincidence with the sample mean. Measured ``p_min`` is 1/1025 in every
        regime, comfortably under the 0.05/8 slot.

    ``(1 + count) / (2**n + 1)`` keeps p strictly positive. The identity pattern
    does *not* reproduce ``t_obs`` here (the reference sample is centred on the
    null while the observed one need not be), so unlike an exact randomization
    test the raw count really can be zero, and a bare ``count / 2**n`` would
    hand back p = 0.
    """
    arr = np.asarray(d, dtype=float)
    n = arr.size
    if n < 2:
        raise ValueError(f"{name}: need >=2 clusters, got {n}")

    # Zero between-cluster spread degenerates the studentized statistic: every
    # sign pattern maps a constant sample onto the same constant, so there is no
    # reference distribution to compare against. With a task-super-population
    # estimand that is genuinely uninformative -- one repeated point says nothing
    # about the spread of the population -- so report it as not evaluable instead
    # of letting it masquerade as p=1 ("no effect") or p=0 ("infinitely strong").
    if _is_degenerate(arr):
        return not_evaluable(
            name, side, h0_center, n,
            f"zero between-cluster variance (all d_t == {arr[0]:.6g}, "
            f"std={arr.std(ddof=1):.3g}); the studentized sign-flip test has no "
            "reference distribution",
        )

    t_obs = _studentized(arr, h0_center)
    resid = arr - arr.mean()
    patterns = _sign_patterns(n)
    ref = h0_center + resid * patterns

    means = ref.mean(axis=1)
    sds = ref.std(axis=1, ddof=1)
    # A sign pattern with no spread: the studentized statistic genuinely diverges
    # (mean* != h0 but se* == 0), so its limit is +/-inf with the sign of
    # (mean* - h0). Mapping it to 0.0 instead would fold it into the CENTRE of
    # the reference distribution and empty the tails -- badly anti-conservative
    # in exactly the tie-heavy plateau regime this experiment expects. Dropping
    # such patterns and renormalizing does not help either: measured type-I error
    # stays at the broken value. Only the signed-infinity limit is calibrated.
    scale = np.maximum(1.0, np.abs(ref).max(axis=1))
    degenerate = sds <= _REL_SPREAD_EPS * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = np.where(
            degenerate,
            np.sign(means - h0_center) * np.inf,
            (means - h0_center) / (sds / np.sqrt(n)),
        )
    # sign() is 0 when mean* lands exactly on h0; such a pattern is genuinely
    # uninformative about either tail, so keep it at 0 rather than inventing one.
    t_star = np.nan_to_num(t_star, nan=0.0, posinf=np.inf, neginf=-np.inf)

    if side == "two-sided":
        count = int(np.sum(np.abs(t_star) >= abs(t_obs)))
    elif side == "less":
        count = int(np.sum(t_star <= t_obs))
    elif side == "greater":
        count = int(np.sum(t_star >= t_obs))
    else:
        raise ValueError(f"unknown side {side!r}")

    m = patterns.shape[0]
    return TestResult(
        name=name,
        p=(1 + count) / (m + 1),
        side=side,
        h0_center=h0_center,
        t_obs=t_obs,
        n_clusters=n,
        degenerate_resample_fraction=float(np.mean(degenerate)),
        n_sign_patterns=m,
    )


def not_evaluable(name: str, side: Side, h0_center: float, n: int, why: str) -> TestResult:
    """A comparison that cannot discriminate (e.g. two tiers selected identically).

    Kept in the family with ``p = 1`` so the Holm denominator does not silently
    change, but flagged so "not evaluable" is never read as "not different".
    """
    return TestResult(
        name=name, p=1.0, side=side, h0_center=h0_center, t_obs=0.0,
        n_clusters=n, evaluable=False, note=why,
    )


def holm(tests: Sequence[TestResult], alpha: float = DEFAULT_ALPHA) -> HolmResult:
    """Holm step-down over the pre-registered family.

    Duplicate names are rejected: results are keyed by name, so a collision
    would silently drop a member and shrink the family denominator -- exactly
    the kind of quiet multiplicity loss the pre-registration exists to prevent.
    """
    names = [t.name for t in tests]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate test names in family: {sorted(dupes)}")

    ordered = sorted(tests, key=lambda t: t.p)
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, t in enumerate(ordered):
        val = min(1.0, (m - i) * t.p)
        running = max(running, val)  # enforce monotonicity
        adjusted[t.name] = running
    return HolmResult(
        adjusted=adjusted,
        rejected={k: v <= alpha for k, v in adjusted.items()},
        alpha=alpha,
        order=[t.name for t in ordered],
    )


def cluster_bootstrap_ci(
    d: Sequence[float],
    *,
    level: float = 0.95,
    b: int = DEFAULT_B,
    seed: int = 0,
) -> CI:
    """BCa interval for the cluster mean, with a pre-registered percentile fallback."""
    arr = np.asarray(d, dtype=float)
    n = arr.size
    theta = float(arr.mean())

    rng = np.random.default_rng(seed)
    boot = arr[rng.integers(0, n, size=(b, n))].mean(axis=1)

    alpha = 1.0 - level
    lo_q, hi_q = alpha / 2, 1 - alpha / 2

    prop = float(np.mean(boot < theta))
    prop = min(max(prop, 1.0 / (b + 1)), 1.0 - 1.0 / (b + 1))
    z0 = float(_norm_ppf(prop))

    jack = np.array([np.delete(arr, i).mean() for i in range(n)])
    diff = jack.mean() - jack
    denom = float((diff ** 2).sum() ** 1.5)
    a = 0.0 if denom < BCA_MIN_DENOM else float((diff ** 3).sum() / (6 * denom))

    reasons = []
    if denom < BCA_MIN_DENOM:
        reasons.append("jackknife denominator ~0")
    if abs(a) > BCA_MAX_ABS_A:
        reasons.append(f"|a|={abs(a):.3f}>{BCA_MAX_ABS_A}")
    if abs(z0) > BCA_MAX_ABS_Z0:
        reasons.append(f"|z0|={abs(z0):.3f}>{BCA_MAX_ABS_Z0}")

    if reasons:
        lo, hi = np.quantile(boot, [lo_q, hi_q])
        return CI(float(lo), float(hi), "percentile", z0, a, "; ".join(reasons))

    def _adj(q: float) -> float:
        z = _norm_ppf(q)
        return float(_norm_cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))

    lo, hi = np.quantile(boot, [_adj(lo_q), _adj(hi_q)])
    return CI(float(lo), float(hi), "bca", z0, a)


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(q: float) -> float:
    """Inverse normal CDF via bisection -- avoids a scipy dependency."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0,1), got {q}")
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
