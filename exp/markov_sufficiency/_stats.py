"""Pre-registered statistical primitives for E1-E5.

Every routine here is a pure function returning a dataclass. The design rules
they encode come from the approved plan and are deliberately not configurable:

  * the inference unit is the episode (cluster), never the step;
  * a non-rejection is never evidence of equivalence, so verdict helpers work
    on intervals and practical bounds rather than bare p-values;
  * multiplicity is Holm, and interval-based verdicts use Holm-adjusted
    simultaneous intervals so they cannot bypass the correction.

Public interface: :func:`holm`, :func:`holm_adjusted_levels`,
:func:`cluster_sign_permutation`, :func:`cluster_bootstrap_ci`,
:func:`mcnemar_exact`, :func:`cmh_test`, :func:`mcnemar_power_n`,
:func:`two_proportion_power`.

Key dependency: numpy only -- keeping this module free of scipy makes the
offline analysis runnable in any of the project's virtualenvs.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Sequence

import numpy as np

# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TestResult:
    """A p-value plus the effect estimate it was computed from."""

    statistic: float
    p_value: float
    estimate: float
    n: int


@dataclasses.dataclass(frozen=True)
class IntervalResult:
    """A point estimate with a percentile interval and its coverage level."""

    estimate: float
    low: float
    high: float
    level: float
    n_resamples: int


# ------------------------------------------------------------------
# Multiplicity
# ------------------------------------------------------------------


def holm(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down. Returns per-hypothesis rejection flags."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break  # step-down: once one fails, all larger p-values fail too
    return reject


def holm_adjusted_levels(p_values: Sequence[float], alpha: float = 0.05) -> list[float]:
    """Return the per-comparison coverage level for Holm-adjusted intervals.

    The comparison ranked ``k`` (1-based, by ascending p-value) gets coverage
    ``1 - alpha/(m-k+1)``. Feeding these into :func:`cluster_bootstrap_ci`
    yields intervals whose verdicts agree with the Holm decision instead of
    silently reverting to uncorrected 95% intervals.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    levels = [0.0] * m
    for rank, idx in enumerate(order):
        levels[idx] = 1.0 - alpha / (m - rank)
    return levels


# ------------------------------------------------------------------
# Cluster-level inference
# ------------------------------------------------------------------


def cluster_sign_permutation(
    paired_diffs: Sequence[float],
    n_permutations: int = 100_000,
    seed: int = 0,
) -> TestResult:
    """Two-sided sign-flip permutation test on cluster-level paired differences.

    ``paired_diffs`` holds one number per episode (e.g. the difference of two
    feature sets' median residual). Sign flipping is the exchangeability
    implied by the paired design and makes no distributional assumption.
    """
    diffs = np.asarray(paired_diffs, dtype=np.float64)
    diffs = diffs[~np.isnan(diffs)]
    n = diffs.size
    if n == 0:
        return TestResult(statistic=float("nan"), p_value=float("nan"), estimate=float("nan"), n=0)

    observed = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, n))
    null = signs @ diffs / n
    # +1 in numerator and denominator: the observed assignment is itself one of
    # the equally likely sign assignments.
    p = (np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1)
    return TestResult(statistic=observed, p_value=float(p), estimate=observed, n=n)


def hodges_lehmann(values: Sequence[float]) -> float:
    """One-sample Hodges-Lehmann estimator: the median of Walsh averages.

    This is the location estimate the sign/permutation test is consistent with.
    The plain median of the differences is a different statistic -- for
    ``[0, 2, 10]`` the Walsh-average HL is 3.5 while the median is 2.0.
    """
    v = np.asarray([x for x in values if x == x], dtype=np.float64)
    if v.size == 0:
        return float("nan")
    walsh = (v[:, None] + v[None, :]) / 2.0
    return float(np.median(walsh[np.triu_indices(v.size)]))


def sign_permutation_ci(
    paired_diffs: Sequence[float],
    level: float = 0.95,
    n_permutations: int = 20_000,
    seed: int = 0,
) -> IntervalResult:
    """Interval obtained by inverting the sign-flip permutation test.

    The registered inference for E1 is a permutation test, so its interval must
    come from the same mechanism: the set of shifts ``d`` whose test on
    ``x - d`` is not rejected at ``1 - level``. A percentile bootstrap interval
    would be a different (and unregistered) procedure.
    """
    v = np.asarray([x for x in paired_diffs if x == x], dtype=np.float64)
    if v.size == 0:
        return IntervalResult(float("nan"), float("nan"), float("nan"), level, 0)

    alpha = 1.0 - level
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, v.size))

    def rejects(shift: float) -> bool:
        shifted = v - shift
        observed = float(np.mean(shifted))
        null = signs @ shifted / v.size
        p = (np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1)
        return p < alpha

    point = hodges_lehmann(v)
    span = float(np.max(np.abs(v))) + 1.0

    def edge(direction: int) -> float:
        """Bisect for the outermost shift that is still accepted."""
        near, far = point, point + direction * span
        for _ in range(40):
            mid = (near + far) / 2
            if rejects(mid):
                far = mid
            else:
                near = mid
        return far

    return IntervalResult(estimate=point, low=edge(-1), high=edge(+1), level=level, n_resamples=n_permutations)


def cluster_bootstrap_ci(
    clusters: Sequence,
    statistic: Callable[[list], float],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
    strata: Sequence | None = None,
) -> IntervalResult:
    """Percentile bootstrap with the cluster (episode) as the resampling unit.

    ``clusters`` is a sequence of arbitrary per-episode payloads; ``statistic``
    maps a resampled list of them to a scalar. Resampling whole clusters keeps
    within-episode correlation intact -- resampling steps or pairs instead
    would badly understate the variance. When ``strata`` is given (typically
    ``task_id``), resampling happens within stratum.
    """
    items = list(clusters)
    n = len(items)
    if n == 0:
        return IntervalResult(float("nan"), float("nan"), float("nan"), level, 0)

    rng = np.random.default_rng(seed)
    if strata is None:
        groups = [list(range(n))]
    else:
        buckets: dict[object, list[int]] = {}
        for i, s in enumerate(strata):
            buckets.setdefault(s, []).append(i)
        groups = list(buckets.values())

    draws = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx: list[int] = []
        for g in groups:
            idx.extend(rng.choice(g, size=len(g), replace=True).tolist())
        draws[b] = statistic([items[i] for i in idx])

    tail = (1.0 - level) / 2.0
    low, high = np.nanpercentile(draws, [100 * tail, 100 * (1 - tail)])
    return IntervalResult(
        estimate=float(statistic(items)),
        low=float(low),
        high=float(high),
        level=level,
        n_resamples=n_resamples,
    )


# ------------------------------------------------------------------
# Paired binary outcomes
# ------------------------------------------------------------------


def mcnemar_exact(b: int, c: int, n_pairs: int | None = None) -> TestResult:
    """Two-sided exact McNemar test on the discordant counts ``b`` and ``c``.

    Under the null the discordant pairs split Binomial(b+c, 1/2). With b+c=5
    and a fully one-sided split the smallest attainable p-value is 0.0625 --
    which is why sample sizes in this plan were derived rather than assumed.

    ``estimate`` is the **paired risk difference** ``(b - c) / n_pairs``, i.e.
    the SR difference the plan's effect bounds are expressed in. The
    denominator is the number of paired episodes, not the discordant count;
    dividing by ``b + c`` would inflate every effect by ``1 / pi_d``.
    ``n_pairs`` defaults to ``b + c`` only to keep the degenerate all-discordant
    case well defined, and callers with real data must pass it.
    """
    n_disc = b + c
    denom = n_disc if n_pairs is None else n_pairs
    if n_disc == 0:
        return TestResult(statistic=0.0, p_value=1.0, estimate=0.0, n=denom or 0)
    k = min(b, c)
    tail = sum(math.comb(n_disc, i) for i in range(k + 1)) / (2.0**n_disc)
    estimate = (b - c) / denom if denom else float("nan")
    return TestResult(statistic=float(b - c), p_value=float(min(1.0, 2.0 * tail)), estimate=estimate, n=denom)


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> IntervalResult:
    """Wilson score interval for a single-arm success rate."""
    if n <= 0:
        return IntervalResult(float("nan"), float("nan"), float("nan"), 1 - alpha, 0)
    z = _z_two_sided(alpha)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return IntervalResult(estimate=p, low=max(0.0, centre - half), high=min(1.0, centre + half), level=1 - alpha, n_resamples=0)


def cmh_test(tables: Sequence[tuple[int, int, int, int]]) -> TestResult:
    """Cochran-Mantel-Haenszel test over 2x2 tables ``(a, b, c, d)`` per stratum.

    Rows are the two groups, columns the binary outcome; strata are tasks. Valid
    here because the unit is the episode and the outcome is binary -- applying
    it to within-episode proportions would be pseudo-replication.
    """
    num = 0.0
    exp = 0.0
    var = 0.0
    total = 0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n == 0:
            continue
        total += n
        r1, r2 = a + b, c + d
        c1, c2 = a + c, b + d
        num += a
        exp += r1 * c1 / n
        if n > 1:
            var += (r1 * r2 * c1 * c2) / (n * n * (n - 1))
    if var <= 0:
        return TestResult(statistic=float("nan"), p_value=float("nan"), estimate=float("nan"), n=total)
    # Continuity correction must be clamped at zero: without the max(),
    # an exactly balanced table (|O - E| = 0) would score 0.25/var > 0 and
    # report a spurious association.
    corrected = max(0.0, abs(num - exp) - 0.5)
    chi = corrected**2 / var
    p = math.erfc(math.sqrt(chi / 2.0))
    return TestResult(statistic=float(chi), p_value=float(p), estimate=float(num - exp), n=total)


# ------------------------------------------------------------------
# Power / sample size
# ------------------------------------------------------------------

_Z = {0.05: 1.959963985, 0.025: 2.241402728, 0.01: 2.575829304}


def _z_two_sided(alpha: float) -> float:
    if alpha in _Z:
        return _Z[alpha]
    # Acklam-style inverse normal is overkill here; bisect the erfc instead.
    lo, hi = 0.0, 10.0
    target = alpha
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mcnemar_power_n(
    pi_d: float,
    delta: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Paired sample size for McNemar at discordance ``pi_d`` and effect ``delta``.

    ``n >= (z_{alpha/2}*sqrt(pi_d) + z_beta*sqrt(pi_d - delta^2))^2 / delta^2``.
    Returns ``inf`` when the effect is not attainable at that discordance.
    """
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if pi_d <= delta**2:
        return float("inf")
    z_a = _z_two_sided(alpha)
    z_b = _z_two_sided(2 * (1 - power))
    return (z_a * math.sqrt(pi_d) + z_b * math.sqrt(pi_d - delta**2)) ** 2 / delta**2


def two_proportion_power(
    n_group1: float,
    n_group0: float,
    delta: float,
    p_bar: float,
    alpha: float = 0.05,
) -> float:
    """Normal-approximation power for a two-proportion difference.

    ``p_bar`` is the pooled rate used for the null standard error; 0.5 is the
    most conservative choice and is what the plan's worst-case column uses.
    """
    if n_group1 <= 0 or n_group0 <= 0:
        return float("nan")
    se = math.sqrt(p_bar * (1 - p_bar) * (1 / n_group1 + 1 / n_group0))
    if se <= 0:
        return float("nan")
    return _phi(delta / se - _z_two_sided(alpha))
