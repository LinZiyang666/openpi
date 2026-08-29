"""Realisable Pareto frontier of a three-point operating family (protocol 3.2).

Given (cost, SR) points, the set of operating points achievable by
episode-level randomised mixtures is the convex hull; the frontier is its
UPPER concave envelope. Points strictly dominated by another point of the same
family are removed, equal-cost points keep the higher SR, and interpolation is
only defined inside the hull's actual cost support. Both AUC terms of H1/H2
use this one function so threshold, SV and S0 receive identical treatment.

This module is used ONLY by the outcome-design stage. The cost-only stage
(``cost_map``) must never import it (static source-lock test).
"""

from __future__ import annotations

import numpy as np

#: A support miss is scored at the theoretical worst value and KEPT in the
#: bootstrap distribution (protocol 4, G1R2-B5): never dropped, never clamped.
SUPPORT_MISS_AUC = -1.0


def upper_concave_hull(points) -> list[tuple[float, float]]:
    """Upper concave envelope of ``points`` = [(cost, sr), ...].

    Steps: sort by cost (ties: keep max SR), drop Pareto-dominated points
    (another point with cost <= and SR >= and at least one strict), then keep
    only the vertices of the upper concave envelope (monotone chain). Points
    below a chord of two other points are dropped by the chain.
    """
    pts = [(float(c), float(s)) for c, s in points]
    if not pts:
        raise ValueError("empty frontier")
    if any(not (np.isfinite(c) and np.isfinite(s)) for c, s in pts):
        raise ValueError("non-finite frontier point")
    # equal cost -> keep the higher SR
    best: dict[float, float] = {}
    for c, s in pts:
        best[c] = max(best.get(c, -np.inf), s)
    cand = sorted(best.items())
    # Pareto dominance removal (protocol 3.2, G1-frozen): a point leaves if
    # another point of the same family has cost <= and SR >= with at least one
    # strict inequality. An equal-SR point at higher cost is therefore dropped
    # and does NOT extend the family's support (G2R2-B3).
    kept = []
    for c, s in cand:
        dominated = any((c2 <= c and s2 >= s) and (c2 < c or s2 > s) for c2, s2 in cand)
        if not dominated:
            kept.append((c, s))
    # upper monotone chain over the kept, cost-sorted points
    hull: list[tuple[float, float]] = []
    for p in kept:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return hull


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull_support(hull) -> tuple[float, float]:
    return hull[0][0], hull[-1][0]


def covers(hull, c_lo: float, c_hi: float) -> bool:
    lo, hi = hull_support(hull)
    return lo <= c_lo and hi >= c_hi


def sr_at(hull, cost: float) -> float:
    """Linear interpolation on the hull; only valid inside its support."""
    xs = [p[0] for p in hull]
    ys = [p[1] for p in hull]
    if cost < xs[0] or cost > xs[-1]:
        raise ValueError(f"cost {cost} outside hull support [{xs[0]}, {xs[-1]}]")
    return float(np.interp(cost, xs, ys))


def auc_norm(hull, c_lo: float, c_hi: float) -> float:
    """EXACT normalised integral of the piecewise-linear hull on [c_lo, c_hi].

    The breakpoints are the interval ends plus every hull vertex strictly
    inside the interval; between consecutive breakpoints the hull is linear,
    so the trapezoid rule is exact. No grid approximation (G2R1-B3).
    """
    if not covers(hull, c_lo, c_hi):
        raise ValueError("hull does not cover the interval")
    if not c_hi > c_lo:
        raise ValueError("empty interval")
    xs = sorted({float(c_lo), float(c_hi)} | {p[0] for p in hull if c_lo < p[0] < c_hi})
    ys = [sr_at(hull, x) for x in xs]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return area / (c_hi - c_lo)


def auc_with_support(points_a, points_b, c_lo: float, c_hi: float) -> tuple[float, bool]:
    """AUC_norm(A) - AUC_norm(B) on [c_lo, c_hi], or (SUPPORT_MISS_AUC, True).

    Returns ``(value, support_miss)``. When either family's hull does not cover
    the frozen interval the replicate is scored ``SUPPORT_MISS_AUC`` and flagged;
    the caller keeps it in the distribution and separately enforces the joint
    miss-rate gate. Nothing here moves the interval.
    """
    if not (np.isfinite(c_lo) and np.isfinite(c_hi) and c_hi > c_lo):
        raise ValueError("invalid interval")
    ha = upper_concave_hull(points_a)
    hb = upper_concave_hull(points_b)
    if not covers(ha, c_lo, c_hi) or not covers(hb, c_lo, c_hi):
        return SUPPORT_MISS_AUC, True
    return auc_norm(ha, c_lo, c_hi) - auc_norm(hb, c_lo, c_hi), False


def difference_breakpoints(hull_a, hull_b, c_lo: float, c_hi: float) -> list[float]:
    """Interval ends plus every vertex of either hull strictly inside the interval."""
    if not (np.isfinite(c_lo) and np.isfinite(c_hi) and c_hi > c_lo):
        raise ValueError("invalid interval")
    inner = {p[0] for p in hull_a if c_lo < p[0] < c_hi} | {p[0] for p in hull_b if c_lo < p[0] < c_hi}
    return sorted({float(c_lo), float(c_hi)} | inner)


def frontier_difference_extrema(hull_a, hull_b, c_lo: float, c_hi: float) -> dict:
    """EXACT extrema of SR_a(c) - SR_b(c) on [c_lo, c_hi] (G2R2-B2).

    The difference of two piecewise-linear functions is piecewise-linear, so
    its minimum and maximum over the interval are attained at breakpoints:
    the interval ends and the vertices of both hulls inside it. No grid.
    """
    if not (covers(hull_a, c_lo, c_hi) and covers(hull_b, c_lo, c_hi)):
        raise ValueError("a hull does not cover the interval")
    xs = difference_breakpoints(hull_a, hull_b, c_lo, c_hi)
    diffs = [sr_at(hull_a, x) - sr_at(hull_b, x) for x in xs]
    i_min = int(np.argmin(diffs))
    i_max = int(np.argmax(diffs))
    return {"breakpoints": xs, "diffs": diffs, "min": float(diffs[i_min]), "argmin_cost": xs[i_min],
            "max": float(diffs[i_max]), "argmax_cost": xs[i_max],
            "a_dominated": bool(diffs[i_max] <= 0.0), "b_dominated": bool(diffs[i_min] >= 0.0)}
