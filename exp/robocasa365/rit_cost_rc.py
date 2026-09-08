"""RoboCasa365 cost authority for the RIT ladder, and the IR half of the fit.

Why this module exists
----------------------
``exp.rit_pareto.rit_k`` carries the LIBERO x pi0.5 cost model in module-level
constants: pi0.5's three stage latencies, a ``MISS_MS`` derived from them, and
``tier_cost``'s ``STAGE1 + STAGE2 + start_t * STAGE3``. That last formula reads
``start_t`` as *the fraction of stage 3 still to run*, which is true only for a
descending schedule. GR00T's schedule ascends: at ``start_t = 0.75`` one of four
steps remains, not three. Feeding GR00T tiers to that formula prices the k=2
ladder above the k=3 ladder and mis-addresses the whole IR grid, so the cost
half is re-derived here against the schedule's own ``remaining_steps``.

``exp.dispatch_surface.analytic_cost`` and ``rit_k`` are not edited: their
constants are recorded in the LIBERO line's frozen artifacts, and the RoboCasa
teachers are different models on a different observation geometry. This is the
same "new file, new authority" discipline the emitters follow.

What is reused, unchanged
-------------------------
The LP fit, the knot ladder, the per-tier cuts and the verdict walk come from
``rit_k`` by import. Only the cost-dependent surface -- ``tier_cost`` /
``MISS_MS`` / ``predicted_ir`` / ``attainable_range`` / ``delta_for_ir`` -- is
re-implemented, parameterised by a :class:`StageCost`.

Public interface: ``StageCost``, ``RCTier``, ``ladder``, ``predicted_ir``,
``attainable_range``, ``ir_curve``, ``delta_for_ir``, ``fit``, ``cuts_for``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi.cache.types import DenoiseSchedule
from exp.rit_pareto.rit_k import (
    EPS_TOTAL,
    IR_TOL,
    KNOT_LADDER,
    PLFitK,
    choose_knots,
    cut_at,
    cuts,
    fit_pl_quantile_k,
    predict,
    verdict_index,
)

__all__ = [
    "EPS_TOTAL", "IR_TOL", "KNOT_LADDER", "StageCost", "RCTier", "ladder", "choose_knots",
    "tier_cost", "miss_cost", "predicted_ir", "attainable_range", "ir_curve",
    "delta_for_ir", "fit", "cuts_for", "cut_at", "cuts", "predict", "verdict_index",
]

ESTIMATOR = "pl_knots_k_v1"


# ------------------------------------------------------------------
# Cost model
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StageCost:
    """Measured per-decision stage costs of one served teacher, in milliseconds.

    ``stage3_head_ms`` / ``stage3_step_ms`` are the intercept and slope of
    ``s3(k) = a + b*k`` fitted over a denoising-step ladder. Measuring only the
    full loop and dividing by the step count sets ``a`` to zero and
    systematically over-credits a warm tier, which is exactly what the k ladder
    exists to prevent; ``linear_stage3`` records when that fallback was taken.
    """

    teacher: str
    schedule: DenoiseSchedule
    stage1_ms: float
    stage2_ms: float
    stage3_head_ms: float
    stage3_step_ms: float
    provenance: str
    linear_stage3: bool = False

    def stage3_ms(self, steps: float) -> float:
        """Cost of running ``steps`` of the action-head loop (0 steps = 0)."""
        if steps <= 0:
            return 0.0
        return self.stage3_head_ms + self.stage3_step_ms * float(steps)

    @property
    def full_stage3_ms(self) -> float:
        return self.stage3_ms(self.schedule.num_steps)

    def remaining_steps(self, start_t: float) -> int:
        """Steps left after resuming at ``start_t``, from the schedule itself."""
        return int(self.schedule.remaining_steps(round(float(start_t), 4)))


def tier_cost(cost: StageCost, hit_type: str, start_t: float | None) -> float:
    """Per-decision GPU cost of one verdict under ``cost``.

    FULL_HIT pays stage 1. MISS pays all three stages with the full loop.
    WARM_START pays stages 1 and 2 plus however many steps the schedule says
    are still ahead of ``start_t`` -- direction included, which is the whole
    point of not reusing the pi0.5 formula.
    """
    if hit_type == "FULL_HIT":
        return cost.stage1_ms
    if hit_type == "MISS":
        return cost.stage1_ms + cost.stage2_ms + cost.full_stage3_ms
    if hit_type == "WARM_START":
        if start_t is None:
            raise ValueError("WARM_START needs a start_t")
        st = round(float(start_t), 4)
        if st not in cost.schedule.timestep_set:
            raise ValueError(
                f"start_t={start_t} is not a resume point of {cost.schedule.schedule_id}: "
                f"{sorted(cost.schedule.timestep_set)}"
            )
        return cost.stage1_ms + cost.stage2_ms + cost.stage3_ms(cost.remaining_steps(st))
    raise ValueError(f"unknown hit_type {hit_type!r}")


def miss_cost(cost: StageCost) -> float:
    return tier_cost(cost, "MISS", None)


# ------------------------------------------------------------------
# Tiers
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RCTier:
    """One rung: verdict, resume timestep, risk column, and its priced cost.

    ``cost_ms`` is a stored field rather than a property so the ladder can be
    validated and serialised without carrying the cost model around, and so a
    tier read back from a record prices identically to the one that emitted it.
    """

    name: str
    hit_type: str
    start_t: float | None
    y_key: str
    cost_ms: float


def ladder(cost: StageCost, warm_start_ts: list[float]) -> tuple[RCTier, ...]:
    """FULL_HIT plus one warm rung per ``start_t``, cheapest / riskiest first.

    ``warm_start_ts`` is given in the order the owner names the tiers (nearest
    to clean first); the returned ladder is re-sorted by cost so the verdict
    walk takes the cheapest admissible rung, and a duplicate or mis-ordered
    request fails here rather than at deploy time.
    """
    tiers = [RCTier("full", "FULL_HIT", None, "y_full", tier_cost(cost, "FULL_HIT", None))]
    for st in warm_start_ts:
        st = round(float(st), 4)
        rem = cost.remaining_steps(st)
        tiers.append(
            RCTier(
                name=f"warm{int(round(st * 100)):02d}",
                hit_type="WARM_START",
                start_t=st,
                y_key=f"y_rem{rem}",
                cost_ms=tier_cost(cost, "WARM_START", st),
            )
        )
    return _check(tuple(sorted(tiers, key=lambda t: t.cost_ms)), cost)


def _check(tiers: tuple[RCTier, ...], cost: StageCost) -> tuple[RCTier, ...]:
    if not tiers:
        raise ValueError("at least one tier is required")
    names = [t.name for t in tiers]
    if len(set(names)) != len(names):
        raise ValueError(f"tier names must be unique: {names}")
    costs = [t.cost_ms for t in tiers]
    if any(b <= a for a, b in zip(costs, costs[1:])):
        raise ValueError(f"tier costs must strictly increase: {list(zip(names, costs))}")
    if costs[-1] >= miss_cost(cost):
        raise ValueError(f"every tier must cost less than MISS ({miss_cost(cost):.4f} ms)")
    return tiers


# ------------------------------------------------------------------
# Fit (thin wrapper: the LP itself is rit_k's)
# ------------------------------------------------------------------


def fit(s, ys: dict[str, np.ndarray], knots, *, tiers: tuple[RCTier, ...],
        n_seg_req: int, alpha: float, eps_total: float = EPS_TOTAL) -> PLFitK:
    """The joint K-layer pinball LP, with RoboCasa-priced tiers."""
    return fit_pl_quantile_k(
        s, ys, knots, tiers=tiers, n_seg_req=n_seg_req, alpha=alpha, eps_total=eps_total
    )


def cuts_for(fit_obj: PLFitK, delta: float) -> list[float]:
    """Per-tier score cuts in ladder order (what the deployed judge compares to)."""
    table = cuts(fit_obj, delta)
    return [float(table[t.name]) for t in fit_obj.tiers]


# ------------------------------------------------------------------
# Inference ratio
# ------------------------------------------------------------------


def predicted_ir(s, thetas: list[float], tiers: tuple[RCTier, ...], cost: StageCost) -> float:
    """Analytic cost of the deployed rule on ``s``, as a percent of all-MISS."""
    if len(thetas) != len(tiers):
        raise ValueError("one cut per tier")
    idx = verdict_index(s, list(thetas))
    counts = np.bincount(idx, minlength=len(tiers) + 1)
    miss = miss_cost(cost)
    total = 0.0
    for count, unit in zip(counts.tolist(), [t.cost_ms for t in tiers] + [miss]):
        total += count * unit
    return float(100.0 * total / (len(idx) * miss))


def realized_ir(counts: dict[str, int], tiers: tuple[RCTier, ...], cost: StageCost) -> float:
    """Same ratio from observed verdict counts, keyed by tier name plus ``miss``.

    This is the number the figure's x axis uses: the calibration cohort is
    driven by the teacher, but once the ladder is deployed the trajectory
    itself changes, so the predicted ratio is an addressing label and not a
    measurement.
    """
    miss = miss_cost(cost)
    unit = {t.name: t.cost_ms for t in tiers} | {"miss": miss}
    unknown = sorted(set(counts) - set(unit))
    if unknown:
        raise ValueError(f"counts carry verdicts outside the ladder: {unknown}")
    n = sum(counts.values())
    if n <= 0:
        raise ValueError("no verdicts to price")
    total = sum(unit[k] * v for k, v in counts.items())
    return float(100.0 * total / (n * miss))


def _as_1d(name: str, arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a non-empty 1-D array of finite values")
    return a


def _endpoints(fit_obj: PLFitK) -> tuple[float, float]:
    q_all = np.concatenate([np.asarray(v, dtype=np.float64) for v in fit_obj.q.values()])
    q_min, q_max = float(q_all.min()), float(q_all.max())
    tiny = float(np.nextafter(0.0, 1.0))
    d_lo = 0.5 * q_min if q_min > tiny else tiny
    return d_lo, q_max + 1.0


def _ir_at(fit_obj: PLFitK, s: np.ndarray, delta: float, cost: StageCost) -> float:
    return predicted_ir(s, cuts_for(fit_obj, delta), fit_obj.tiers, cost)


def attainable_range(fit_obj: PLFitK, s, cost: StageCost) -> tuple[float, float]:
    """``(lowest, highest)`` predicted IR this fit can be addressed to."""
    s = _as_1d("s", s)
    d_lo, d_hi = _endpoints(fit_obj)
    return _ir_at(fit_obj, s, d_hi, cost), _ir_at(fit_obj, s, d_lo, cost)


def ir_curve(fit_obj: PLFitK, s, cost: StageCost, n: int = 200) -> list[tuple[float, float]]:
    s = _as_1d("s", s)
    d_lo, d_hi = _endpoints(fit_obj)
    return [(float(d), _ir_at(fit_obj, s, float(d), cost)) for d in np.linspace(d_lo, d_hi, int(n))]


def delta_for_ir(fit_obj: PLFitK, s, target: float, cost: StageCost, *, tol: float = IR_TOL) -> dict:
    """Nearest-attainable inverse of the predicted IR (rit_pl bisection semantics)."""
    s = _as_1d("s", s)
    d_lo, d_hi = _endpoints(fit_obj)
    ir_top, ir_bot = _ir_at(fit_obj, s, d_lo, cost), _ir_at(fit_obj, s, d_hi, cost)
    if not (ir_bot <= target <= ir_top) or target >= 100.0:
        raise ValueError(
            f"target IR {target} outside the attainable range "
            f"[{ir_bot:.4f}, {ir_top:.4f}] (and < 100)"
        )

    def pack(delta, ir, lo, ir_lo, hi, ir_hi):
        table = cuts(fit_obj, delta)
        return {
            "delta": float(delta),
            "thetas": {k: float(v) for k, v in table.items()},
            "theta_order": [t.name for t in fit_obj.tiers],
            "predicted_ir": float(ir),
            "ir_gap": float(ir - target),
            "bracket": {"delta_lo": float(lo), "ir_lo": float(ir_lo),
                        "delta_hi": float(hi), "ir_hi": float(ir_hi)},
        }

    if target == ir_top:
        return pack(d_lo, ir_top, d_lo, ir_top, d_hi, ir_bot)
    if target == ir_bot:
        return pack(d_hi, ir_bot, d_lo, ir_top, d_hi, ir_bot)
    lo, hi, ir_lo, ir_hi = d_lo, d_hi, ir_top, ir_bot
    while (hi - lo) > 1e-12 * max(1.0, abs(hi)) and not (
        abs(ir_lo - target) <= tol and abs(ir_hi - target) <= tol
    ):
        mid = 0.5 * (lo + hi)
        v = _ir_at(fit_obj, s, mid, cost)
        if v > target:
            lo, ir_lo = mid, v
        else:
            hi, ir_hi = mid, v
    if (target - ir_hi) < (ir_lo - target):
        return pack(hi, ir_hi, lo, ir_lo, hi, ir_hi)
    return pack(lo, ir_lo, lo, ir_lo, hi, ir_hi)


def common_grid(ranges: list[tuple[float, float]], n: int) -> list[float]:
    """``n`` evenly spaced IR targets inside the intersection of every range.

    The four frontiers (two teachers x two ladder depths) are addressed on one
    grid so "at the same budget, which is better" is a direct read; a curve
    whose own range is wider still gets its endpoints appended by the caller.
    """
    lo = max(r[0] for r in ranges)
    hi = min(r[1] for r in ranges)
    # The top of an attainable range is the all-MISS corner at exactly 100, and
    # the IR inverse refuses it (a rule that never uses the cache is the teacher
    # arm, which the run measures separately). Step just inside instead of
    # emitting an arm the inverse would reject.
    hi = min(hi, 100.0 - IR_TOL)
    if not (hi > lo):
        raise ValueError(f"attainable ranges do not overlap: {ranges}")
    if n < 2:
        raise ValueError("need at least two grid points")
    step = (hi - lo) / (n - 1)
    return [float(lo + i * step) for i in range(n)]
