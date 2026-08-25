#!/usr/bin/env python3
"""X15 statistics: init-level clustering, iso-SR interpolation, and power.

Three estimators the plan pre-registers, kept in one place so the report cannot
quietly use a different one:

* **Cluster bootstrap over inits.** The A pool has 500 unique inits but B-test
  runs 10 seeds on each of 50, so seeds within an init are correlated. Treating
  them as independent would shrink the interval by roughly sqrt(10). Resampling
  whole inits is what keeps the coverage honest.
* **iso-SR interpolation with a bracketing rule.** The headline number is the
  teacher share at which a curve reaches SR = 0.80. Three measured points do not
  always bracket that level; when they do not, the answer is "not estimable",
  never an extrapolation off the end of the curve.
* **Discordant-pair power.** McNemar's power depends on the discordant rate,
  not the marginal rates, so the pilot has to supply it before the A pool is
  ever touched.

Key dependency: the per-episode outcome rows produced by the runner; pairing
and the exact test itself live in ``paired_mcnemar.py``.
"""

from __future__ import annotations

import json
import math
import pathlib
import random
from typing import Callable, Optional, Sequence


# ------------------------------------------------------------------
# Cluster bootstrap
# ------------------------------------------------------------------


def cluster_bootstrap(
    clusters: Sequence[Sequence[float]],
    statistic: Callable[[list[float]], float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile CI for ``statistic``, resampling whole clusters.

    ``clusters`` is one sequence of observations per init. Each bootstrap draw
    samples inits with replacement and pools their observations, so an init that
    contributes 10 correlated seeds contributes them together or not at all.
    """
    clusters = [list(c) for c in clusters if len(c) > 0]
    if not clusters:
        raise ValueError("cluster_bootstrap: no non-empty clusters")

    observed = statistic([v for c in clusters for v in c])
    rng = random.Random(seed)
    n = len(clusters)
    draws: list[float] = []
    for _ in range(n_boot):
        pooled: list[float] = []
        for _ in range(n):
            pooled.extend(clusters[rng.randrange(n)])
        draws.append(statistic(pooled))
    draws.sort()

    lo = draws[max(0, int((alpha / 2) * n_boot) - 1)]
    hi = draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {
        "estimate": observed,
        "ci_low": lo,
        "ci_high": hi,
        "n_clusters": n,
        "n_obs": sum(len(c) for c in clusters),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# ------------------------------------------------------------------
# iso-SR curve
# ------------------------------------------------------------------


def isotonic_sr(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Force SR to be non-decreasing in teacher share via PAVA.

    More teacher cannot hurt in expectation, so a dip is sampling noise; fixing
    it before interpolation keeps the crossing well defined.
    """
    ordered = sorted(points)
    shares = [p[0] for p in ordered]
    values = [p[1] for p in ordered]
    weights = [1.0] * len(values)
    out_v: list[float] = []
    out_w: list[float] = []
    for v, w in zip(values, weights):
        out_v.append(v)
        out_w.append(w)
        while len(out_v) > 1 and out_v[-2] > out_v[-1]:
            total = out_w[-2] + out_w[-1]
            merged = (out_v[-2] * out_w[-2] + out_v[-1] * out_w[-1]) / total
            out_v[-2:] = [merged]
            out_w[-2:] = [total]
    flat: list[float] = []
    for v, w in zip(out_v, out_w):
        flat.extend([v] * int(w))
    return list(zip(shares, flat))


def iso_sr_share(
    points: Sequence[tuple[float, float]], target: float = 0.80
) -> Optional[float]:
    """Teacher share at which the curve reaches ``target`` SR.

    Returns None when no adjacent pair brackets the target. Refusing to
    extrapolate is pre-registered: the curve has three points and its behaviour
    past the last one is unmeasured.
    """
    curve = isotonic_sr(points)
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        lo, hi = min(y0, y1), max(y0, y1)
        if lo <= target <= hi:
            if y1 == y0:
                return x0
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return None


def iso_sr_with_ci(
    clusters_by_point: Sequence[tuple[float, Sequence[Sequence[float]]]],
    *,
    target: float = 0.80,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Bootstrap the iso-SR share, refitting the interpolation each draw.

    ``clusters_by_point`` pairs each operating point's teacher share with its
    per-init success observations. Refitting inside the loop (rather than
    perturbing a fixed curve) is what makes the interval reflect the
    interpolation's own instability. Draws that fail to bracket are counted, not
    silently dropped — a mostly-unbracketed bootstrap means the point estimate
    should not be quoted.
    """
    point_estimate = iso_sr_share(
        [(share, mean([v for c in clusters for v in c])) for share, clusters in clusters_by_point],
        target=target,
    )

    rng = random.Random(seed)
    draws: list[float] = []
    unbracketed = 0
    for _ in range(n_boot):
        resampled: list[tuple[float, float]] = []
        for share, clusters in clusters_by_point:
            clusters = [list(c) for c in clusters if len(c) > 0]
            pooled: list[float] = []
            for _ in range(len(clusters)):
                pooled.extend(clusters[rng.randrange(len(clusters))])
            resampled.append((share, mean(pooled)))
        value = iso_sr_share(resampled, target=target)
        if value is None:
            unbracketed += 1
        else:
            draws.append(value)

    if point_estimate is None or not draws:
        return {
            "estimate": None,
            "bracketed": False,
            "unbracketed_draws": unbracketed,
            "n_boot": n_boot,
        }
    draws.sort()
    return {
        "estimate": point_estimate,
        "bracketed": True,
        "ci_low": draws[max(0, int((alpha / 2) * len(draws)) - 1)],
        "ci_high": draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))],
        "unbracketed_draws": unbracketed,
        "n_boot": n_boot,
    }


# ------------------------------------------------------------------
# Power
# ------------------------------------------------------------------


def mcnemar_power(
    n_pairs: int, discordant_rate: float, effect: float, *, alpha: float = 0.05
) -> float:
    """Normal-approximation power for the exact paired test.

    Power is NOT monotone in ``discordant_rate``, which is exactly why the plan
    requires a pilot estimate of it before the A pool is touched. Lowering the
    rate concentrates a fixed absolute effect into a sharper imbalance among
    discordant pairs (helping), while shrinking how many such pairs exist
    (hurting). Above roughly ten discordant pairs the first term dominates and
    power falls as disagreement rises; below that the second term wins outright
    and power collapses regardless of effect size. Neither ``n`` nor the
    marginal rates can stand in for this number.
    """
    if not 0 < discordant_rate <= 1 or n_pairs <= 0:
        return 0.0
    m = n_pairs * discordant_rate
    if m <= 0:
        return 0.0
    # Effect expressed as an imbalance among discordant pairs.
    p_imbalance = min(0.999, max(0.001, 0.5 + effect / (2 * discordant_rate)))
    z_alpha = 1.959963984540054 if alpha == 0.05 else _z(1 - alpha / 2)
    se_null = 0.5 / math.sqrt(m)
    se_alt = math.sqrt(p_imbalance * (1 - p_imbalance) / m)
    if se_alt == 0:
        return 1.0
    z = (abs(p_imbalance - 0.5) - z_alpha * se_null) / se_alt
    return _phi(z)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _z(p: float) -> float:
    """Inverse normal CDF by bisection — adequate at this precision."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------------------------
# Pre-registration gates
# ------------------------------------------------------------------


class PreregistrationError(RuntimeError):
    """A pre-registered protocol rule was violated. Never catch this."""


def assert_no_unpaired_drop(
    n_only_a: int, n_only_b: int, *, label: str = "D1"
) -> None:
    """Hard-fail the primary comparison on ANY unpaired slot.

    ``paired_mcnemar`` warns and drops; for the primary that warning is not
    enough. A dropped slot means the two arms did not run the same list, and a
    paired p-value computed over whatever they happen to share is not the test
    that was pre-registered.
    """
    if n_only_a or n_only_b:
        raise PreregistrationError(
            f"{label}: {n_only_a} slot(s) only in arm A and {n_only_b} only in arm B. "
            "The primary requires strictly identical slot lists; investigate the "
            "run rather than testing the intersection."
        )


def assert_pool_isolation(ledger: dict, *, phase: str, reads: str) -> None:
    """Refuse a phase that reads a pool it is not allowed to see.

    Encodes the frozen four-way split as executable policy: everything that
    fits reads gradient/delta/cal, and only the final evaluation touches test
    or A. Written as a gate rather than a convention because the failure is
    invisible in the output — an over-fitted number looks exactly like a good
    one.
    """
    allowed = {
        "p0": {"gradient"},
        "g0": {"gradient"},
        "feature_selection": {"gradient"},
        "train": {"gradient"},
        "delta": {"delta"},
        "calibrate": {"cal"},
        "tau_grid": {"cal"},
        "evaluate_btest": {"test"},
        "evaluate_a": {"a"},
    }
    if phase not in allowed:
        raise PreregistrationError(f"unknown phase {phase!r}")
    if reads not in allowed[phase]:
        raise PreregistrationError(
            f"phase {phase!r} may only read {sorted(allowed[phase])}, not {reads!r}; "
            "the four-way split exists so no fitting step ever sees an evaluation pool"
        )
    if reads in ledger and phase.startswith("evaluate") is False:
        overlap = set(ledger.get(reads, [])) & set(ledger.get("test", []))
        if overlap and reads != "test":
            raise PreregistrationError(
                f"phase {phase!r} reads pool {reads!r}, which shares "
                f"{len(overlap)} init(s) with the test pool"
            )


EVALUATION_POOLS = ("test", "a")


def record_pool_touch(ledger_path: str, pool: str, *, at: str) -> dict:
    """Stamp the first time an evaluation pool is read, and refuse a rewrite.

    Written by the launcher before a single episode of that pool runs. The
    ledger is append-only for this field on purpose: if the timestamp could be
    revised, the ordering it certifies would be worthless.
    """
    path = pathlib.Path(ledger_path)
    ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    touched = ledger.setdefault("touched", {})
    if pool in touched:
        raise PreregistrationError(
            f"pool {pool!r} was already first touched at {touched[pool]!r}; "
            "a first-touch stamp is written once and never revised"
        )
    touched[pool] = at
    path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger


def freeze_parameter(ledger_path: str, key: str, value, *, at: str) -> dict:
    """Commit a pre-registered parameter, refusing to do so after the fact.

    This is the gate that actually enforces the ordering: a value cannot be
    frozen once any evaluation pool has been touched, because by then it could
    have been chosen from what the evaluation showed.
    """
    path = pathlib.Path(ledger_path)
    ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    touched = ledger.get("touched", {})
    already = {p: t for p, t in touched.items() if p in EVALUATION_POOLS}
    if already:
        raise PreregistrationError(
            f"cannot freeze {key!r} now: evaluation pool(s) {sorted(already)} were "
            f"already touched at {already}. A parameter frozen after seeing the "
            "evaluation is not pre-registered."
        )
    frozen = ledger.setdefault("frozen", {})
    if key in frozen:
        raise PreregistrationError(
            f"{key!r} is already frozen at {frozen[key].get('at')!r}; refreezing "
            "would silently replace a pre-registered value"
        )
    frozen[key] = {"value": value, "at": at}
    path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger


def assert_frozen_before(ledger: dict, key: str, *, phase: str) -> None:
    """Refuse an evaluation phase whose parameters were not frozen FIRST.

    Existence is not enough — a field can always be back-filled. This compares
    the freeze timestamp against the first-touch stamp of every evaluation pool
    and requires the freeze to strictly precede them, which is the property
    "we did not look first" actually means.
    """
    frozen = ledger.get("frozen", {})
    if key not in frozen:
        raise PreregistrationError(
            f"phase {phase!r} needs {key!r}, which is not in the ledger's frozen "
            "block; it must be committed before any evaluation pool is touched"
        )
    entry = frozen[key]
    if not isinstance(entry, dict) or "value" not in entry or "at" not in entry:
        raise PreregistrationError(
            f"frozen[{key!r}] must record both 'value' and 'at' (when it was frozen)"
        )

    touched = ledger.get("touched", {})
    relevant = {p: t for p, t in touched.items() if p in EVALUATION_POOLS}
    if not relevant:
        raise PreregistrationError(
            f"phase {phase!r} has no first-touch stamp for any evaluation pool; "
            "the launcher must record one so the freeze/touch order is auditable"
        )
    freeze_at = str(entry["at"])
    late = {p: t for p, t in relevant.items() if str(t) <= freeze_at}
    if late:
        raise PreregistrationError(
            f"{key!r} was frozen at {freeze_at!r}, at or after pool(s) {late} were "
            "first touched; the value could have been chosen from the evaluation "
            "it is supposed to be compared against"
        )
