"""K-tier RIT-PL: one delta, K nested piecewise-linear risk curves, K cuts.

Generalises ``exp.dispatch_surface.rit_pl`` (FULL / WARM@0.3) to an ordered
ladder of cache tiers -- here FULL_HIT, WARM_START@0.3, WARM_START@0.5 -- that
share one tolerance ``delta``. Tiers are ordered from the cheapest / riskiest
(FULL: the cached chunk as is) to the most expensive / safest (more re-denoising
steps); the deployed verdict takes the cheapest tier whose fitted risk at the
retrieval score is admissible, else MISS.

Fit: the same joint knot-value pinball LP as ``rit_pl.fit_pl_quantile`` --
strict monotone floor ``EPS_TOTAL`` per curve, nesting ``q_safer <= q_riskier``
at every knot, ``q >= 0`` -- laid out so that the K=2 ladder (WARM@0.3, FULL)
builds the identical LP (same variables, same rows, same order) and therefore
the identical HiGHS solution as ``rit_pl``.

Costs: the stage constants are imported from the single cost authority
(``analytic_cost``); a warm tier at canonical ``start_t`` re-runs
``start_t * num_steps`` of the stage-3 steps, so it costs
``STAGE1 + STAGE2 + start_t * STAGE3`` (``analytic_cost.unit_cost`` pins the
same formula at start_t = 0.3 and refuses other tiers; ``tier_cost`` here is
the ladder-general form and asserts parity at 0.3).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from exp.dispatch_surface.analysis.analytic_cost import (
    PINNED_START_T_WS,
    STAGE1_MS,
    STAGE2_MS,
    STAGE3_MS,
    unit_cost,
)
from exp.dispatch_surface.fit_surface import _digest_obj
from exp.dispatch_surface.rit_pl import (
    EPS_TOTAL,
    IR_MAX_GAP,
    IR_TOL,
    KNOT_LADDER,
    choose_knots,
    segment_index,
)

__all__ = [
    "EPS_TOTAL", "IR_MAX_GAP", "IR_TOL", "KNOT_LADDER", "K3_TIERS", "K2_TIERS", "MISS_MS",
    "Tier", "PLFitK", "choose_knots", "tier_cost", "fit_pl_quantile_k", "predict", "cut_at",
    "cuts", "verdict_index", "predicted_ir", "attainable_range", "ir_curve", "delta_for_ir",
    "floor_info", "fit_digests", "fit_record_fields", "fit_from_record",
]

ESTIMATOR = "pl_knots_k_v1"
MISS_MS = unit_cost("MISS", None)


# ------------------------------------------------------------------
# Tiers and costs
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Tier:
    """One cache tier of the ladder: verdict, warm start_t (None for FULL) and the
    shadow-table risk column it is calibrated on."""

    name: str
    hit_type: str
    start_t: float | None
    y_key: str

    @property
    def cost_ms(self) -> float:
        return tier_cost(self.hit_type, self.start_t)


def tier_cost(hit_type: str, start_t) -> float:
    """Per-decision GPU cost of a verdict, ladder-general.

    FULL_HIT pays stage 1; MISS pays all three stages; WARM_START at canonical
    ``start_t`` pays stages 1 and 2 plus ``start_t`` of stage 3. Matches
    ``analytic_cost.unit_cost`` exactly at the pinned 0.3 tier.
    """
    if hit_type == "FULL_HIT":
        return STAGE1_MS
    if hit_type == "MISS":
        return STAGE1_MS + STAGE2_MS + STAGE3_MS
    if hit_type == "WARM_START":
        if start_t is None:
            raise ValueError("WARM_START needs a start_t")
        st = round(float(start_t), 4)
        if not (0.0 < st < 1.0) or abs(st * 10.0 - round(st * 10.0)) > 1e-9:
            raise ValueError(f"start_t={start_t} is not a canonical denoise timestep")
        cost = STAGE1_MS + STAGE2_MS + st * STAGE3_MS
        if st == PINNED_START_T_WS and abs(cost - unit_cost("WARM_START", st)) > 1e-9:
            raise AssertionError("tier_cost diverged from analytic_cost at the pinned tier")
        return cost
    raise ValueError(f"unknown hit_type {hit_type!r}")


#: Riskiest / cheapest first; the verdict walks this order.
K3_TIERS: tuple[Tier, ...] = (
    Tier("full", "FULL_HIT", None, "y_tau10"),
    Tier("warm03", "WARM_START", 0.3, "y_tau7"),
    Tier("warm05", "WARM_START", 0.5, "y_tau5"),
)
K2_TIERS: tuple[Tier, ...] = K3_TIERS[:2]


def _check_tiers(tiers) -> tuple[Tier, ...]:
    tiers = tuple(tiers)
    if len(tiers) < 1:
        raise ValueError("at least one tier is required")
    names = [t.name for t in tiers]
    if len(set(names)) != len(names):
        raise ValueError("tier names must be unique")
    costs = [t.cost_ms for t in tiers]
    if any(b <= a for a, b in zip(costs, costs[1:])) or costs[-1] >= MISS_MS:
        raise ValueError("tiers must be ordered by strictly increasing cost, all below MISS")
    return tiers


# ------------------------------------------------------------------
# Fit
# ------------------------------------------------------------------


@dataclasses.dataclass
class PLFitK:
    knots: np.ndarray
    q: dict[str, np.ndarray]          # tier name -> knot values
    tiers: tuple[Tier, ...]           # riskiest first
    eps_total: float
    n_seg_req: int
    n_seg: int
    alpha: float


def _as_1d(name: str, arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a non-empty 1-D array of finite values")
    return a


def fit_pl_quantile_k(s, ys: dict[str, np.ndarray], knots, *, tiers=K3_TIERS, n_seg_req: int,
                      alpha: float, eps_total: float) -> PLFitK:
    """Joint K-layer pinball LP on knot values (level ``1 - alpha``).

    ``ys`` maps tier name -> risk column. Layers enter the LP from the safest
    tier to the riskiest so that the K=2 ladder reproduces ``rit_pl``'s LP
    verbatim; nesting rows ``q[safer][k] - q[riskier][k] <= 0`` follow.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    tiers = _check_tiers(tiers)
    s = _as_1d("s", s)
    knots = _as_1d("knots", knots)
    layers = [(t.name, _as_1d(t.y_key, ys[t.name])) for t in reversed(tiers)]  # safest first
    if any(len(y) != len(s) for _, y in layers):
        raise ValueError("every risk column must match s in length")
    if len(knots) < 3 or not (np.diff(knots) > 0).all():
        raise ValueError("knots must be strictly increasing with at least two segments")
    n_seg = len(knots) - 1
    if int(n_seg_req) < n_seg:
        raise ValueError("n_seg_req must be >= the number of segments actually used")
    if not (0.0 < float(alpha) <= 0.5):
        raise ValueError("alpha must lie in (0, 0.5]")
    if float(eps_total) < 0.0:
        raise ValueError("eps_total must be non-negative")

    n_knots, n, n_layers = len(knots), len(s), len(layers)
    seg = segment_index(knots, s)
    x0, x1 = knots[seg], knots[seg + 1]
    w = np.clip((s - x0) / (x1 - x0), 0.0, 1.0)
    eps = float(eps_total) / float(knots[-1] - knots[0])
    n_grid = n_layers * n_knots
    n_var = n_grid + 2 * n_layers * n
    c = np.zeros(n_var)
    r_idx: list[int] = []
    c_idx: list[int] = []
    vals: list[float] = []
    b_ub: list[float] = []
    row = 0

    def add(entries: dict[int, float], rhs: float) -> None:
        nonlocal row
        for col, val in entries.items():
            r_idx.append(row)
            c_idx.append(col)
            vals.append(val)
        b_ub.append(rhs)
        row += 1

    for layer, (_, y) in enumerate(layers):
        base = n_grid + layer * 2 * n
        c[base:base + n] = 1.0 - float(alpha)
        c[base + n:base + 2 * n] = float(alpha)
        g0 = layer * n_knots + seg
        g1 = g0 + 1
        for i in range(n):
            add({int(g0[i]): -(1.0 - w[i]), int(g1[i]): -w[i], base + i: -1.0}, -float(y[i]))
            add({int(g0[i]): (1.0 - w[i]), int(g1[i]): w[i], base + n + i: -1.0}, float(y[i]))
        for k in range(n_seg):
            add({layer * n_knots + k + 1: 1.0, layer * n_knots + k: -1.0},
                -eps * float(knots[k + 1] - knots[k]))
    for layer in range(n_layers - 1):
        for k in range(n_knots):
            add({layer * n_knots + k: 1.0, (layer + 1) * n_knots + k: -1.0}, 0.0)
    a_ub = coo_matrix((vals, (r_idx, c_idx)), shape=(row, n_var)).tocsr()
    bounds = [(0.0, None)] * n_var
    res = linprog(c, A_ub=a_ub, b_ub=np.asarray(b_ub), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"RIT-PL(K) LP failed: {res.message}")
    q = {name: res.x[layer * n_knots:(layer + 1) * n_knots].copy() for layer, (name, _) in enumerate(layers)}
    return PLFitK(knots=knots, q=q, tiers=tiers, eps_total=float(eps_total),
                  n_seg_req=int(n_seg_req), n_seg=int(n_seg), alpha=float(alpha))


# ------------------------------------------------------------------
# Curves, cuts, verdict cost
# ------------------------------------------------------------------


def predict(fit: PLFitK, s, tier: str) -> np.ndarray:
    q = np.asarray(fit.q[tier], dtype=np.float64)
    knots = np.asarray(fit.knots, dtype=np.float64)
    sc = np.clip(_as_1d("s", s), knots[0], knots[-1])
    k = segment_index(knots, sc)
    wt = (sc - knots[k]) / (knots[k + 1] - knots[k])
    return (1.0 - wt) * q[k] + wt * q[k + 1]


def cut_at(fit: PLFitK, tier: str, delta: float) -> float:
    """Smallest s whose fitted risk on ``tier`` is <= delta (``rit_pl.cut_at`` semantics)."""
    if not (float(fit.eps_total) > 0.0):
        raise ValueError("cut_at needs a strictly monotone fit (eps_total > 0)")
    if not math.isfinite(delta):
        raise ValueError("delta must be finite")
    q = np.asarray(fit.q[tier], dtype=np.float64)
    knots = np.asarray(fit.knots, dtype=np.float64)
    if q[0] <= delta:
        return float(knots[0])
    if q[-1] > delta:
        return math.inf
    idx = int(np.where(q <= delta)[0][0])
    k = idx - 1
    return float(knots[k] + (q[k] - delta) / (q[k] - q[k + 1]) * (knots[k + 1] - knots[k]))


def cuts(fit: PLFitK, delta: float) -> dict[str, float]:
    """Tier name -> cut, riskiest first; nesting (cuts nonincreasing down the ladder)
    holds by construction and is asserted, floating ties clamped."""
    out: dict[str, float] = {}
    prev = math.inf
    for t in fit.tiers:
        theta = cut_at(fit, t.name, delta)
        if theta > prev:
            if theta - prev > 1e-9:
                raise RuntimeError(f"tier nesting violated at {t.name}: cut above the riskier tier")
            theta = prev
        out[t.name] = theta
        prev = theta
    return out


def verdict_index(s, thetas: list[float]) -> np.ndarray:
    """Index of the first tier (riskiest first) admitting each s; ``len(thetas)`` = MISS."""
    s = _as_1d("s", s)
    idx = np.full(len(s), len(thetas), dtype=np.int64)
    for j in range(len(thetas) - 1, -1, -1):
        idx[s >= thetas[j]] = j
    return idx


def predicted_ir(s, thetas: list[float], tiers=K3_TIERS) -> float:
    """Analytic cost of the deployed rule on the rows of ``s``, % of always-full."""
    tiers = _check_tiers(tiers)
    if len(thetas) != len(tiers):
        raise ValueError("one cut per tier")
    idx = verdict_index(s, list(thetas))
    counts = np.bincount(idx, minlength=len(tiers) + 1)
    # Accumulate tier by tier (riskiest first, MISS last): the same summation
    # order as rit_pl.predicted_ir, so the K=2 ladder reproduces it bit for bit.
    total = 0.0
    for count, cost in zip(counts.tolist(), [t.cost_ms for t in tiers] + [MISS_MS]):
        total += count * cost
    return float(100.0 * total / (len(idx) * MISS_MS))


def _thetas(fit: PLFitK, delta: float) -> list[float]:
    return [cuts(fit, delta)[t.name] for t in fit.tiers]


def _endpoints(fit: PLFitK) -> tuple[float, float]:
    q_all = np.concatenate([np.asarray(v, dtype=np.float64) for v in fit.q.values()])
    q_min, q_max = float(q_all.min()), float(q_all.max())
    tiny = float(np.nextafter(0.0, 1.0))
    d_lo = 0.5 * q_min if q_min > tiny else tiny
    return d_lo, q_max + 1.0


def _ir_at(fit: PLFitK, s: np.ndarray, delta: float) -> float:
    return predicted_ir(s, _thetas(fit, delta), fit.tiers)


def attainable_range(fit: PLFitK, s) -> tuple[float, float]:
    s = _as_1d("s", s)
    d_lo, d_hi = _endpoints(fit)
    return _ir_at(fit, s, d_hi), _ir_at(fit, s, d_lo)


def ir_curve(fit: PLFitK, s, n: int = 200) -> list[tuple[float, float]]:
    s = _as_1d("s", s)
    d_lo, d_hi = _endpoints(fit)
    return [(float(d), _ir_at(fit, s, float(d))) for d in np.linspace(d_lo, d_hi, int(n))]


def delta_for_ir(fit: PLFitK, s, target: float, *, tol: float = IR_TOL) -> dict:
    """Nearest-attainable inverse of the predicted IR (``rit_pl.delta_for_ir`` semantics:
    bisection on the bracket, closer end wins, ties to the conservative low-delta end)."""
    s = _as_1d("s", s)
    d_lo, d_hi = _endpoints(fit)
    ir_top, ir_bot = _ir_at(fit, s, d_lo), _ir_at(fit, s, d_hi)
    if not (ir_bot <= target <= ir_top) or target >= 100.0:
        raise ValueError(f"target IR {target} outside the attainable range [{ir_bot:.4f}, {ir_top:.4f}] (and < 100)")

    def pack(delta: float, ir: float, lo: float, ir_lo: float, hi: float, ir_hi: float) -> dict:
        th = cuts(fit, delta)
        return {"delta": float(delta), "thetas": {k: float(v) for k, v in th.items()},
                "predicted_ir": float(ir), "ir_gap": float(ir - target),
                "bracket": {"delta_lo": float(lo), "ir_lo": float(ir_lo), "delta_hi": float(hi), "ir_hi": float(ir_hi)}}

    if target == ir_top:
        return pack(d_lo, ir_top, d_lo, ir_top, d_hi, ir_bot)
    if target == ir_bot:
        return pack(d_hi, ir_bot, d_lo, ir_top, d_hi, ir_bot)
    lo, hi, ir_lo, ir_hi = d_lo, d_hi, ir_top, ir_bot
    while (hi - lo) > 1e-12 * max(1.0, abs(hi)) and not (abs(ir_lo - target) <= tol and abs(ir_hi - target) <= tol):
        mid = 0.5 * (lo + hi)
        v = _ir_at(fit, s, mid)
        if v > target:
            lo, ir_lo = mid, v
        else:
            hi, ir_hi = mid, v
    if (target - ir_hi) < (ir_lo - target):
        return pack(hi, ir_hi, lo, ir_lo, hi, ir_hi)
    return pack(lo, ir_lo, lo, ir_lo, hi, ir_hi)


def floor_info(fit: PLFitK, s, delta: float) -> dict:
    """Per tier: the segment holding the cut, its fitted drop, the eps floor,
    whether the drop sits on the floor, and the segment's row share."""
    s = _as_1d("s", s)
    knots = np.asarray(fit.knots, dtype=np.float64)
    eps = float(fit.eps_total) / float(knots[-1] - knots[0])
    seg_all = segment_index(knots, s)
    out = {}
    for t in fit.tiers:
        theta = cut_at(fit, t.name, delta)
        if not math.isfinite(theta):
            out[t.name] = None
            continue
        q = np.asarray(fit.q[t.name], dtype=np.float64)
        k = int(segment_index(knots, np.asarray([theta]))[0])
        drop = float(q[k] - q[k + 1])
        floor = eps * float(knots[k + 1] - knots[k])
        out[t.name] = {"segment": k, "risk_drop": drop, "eps_floor": floor,
                       "on_eps_floor": bool(drop <= floor * (1.0 + 1e-6)),
                       "segment_share": float(np.mean(seg_all == k))}
    return out


# ------------------------------------------------------------------
# Records
# ------------------------------------------------------------------


def fit_digests(fit: PLFitK) -> dict:
    out = {"knots": _digest_obj(np.asarray(fit.knots).tolist())}
    for t in fit.tiers:
        out[f"q_{t.name}"] = _digest_obj(np.asarray(fit.q[t.name]).tolist())
    return out


def fit_record_fields(fit: PLFitK) -> dict:
    return {
        "estimator": ESTIMATOR,
        "knots": np.asarray(fit.knots, dtype=np.float64).tolist(),
        "q": {t.name: np.asarray(fit.q[t.name], dtype=np.float64).tolist() for t in fit.tiers},
        "tiers": [dataclasses.asdict(t) | {"cost_ms": t.cost_ms} for t in fit.tiers],
        "eps_total": float(fit.eps_total),
        "n_seg_req": int(fit.n_seg_req),
        "n_seg": int(fit.n_seg),
        "alpha": float(fit.alpha),
    }


def fit_from_record(rec: dict) -> PLFitK:
    tiers = tuple(Tier(t["name"], t["hit_type"], t["start_t"], t["y_key"]) for t in rec["tiers"])
    return PLFitK(knots=np.asarray(rec["knots"], dtype=np.float64),
                  q={t.name: np.asarray(rec["q"][t.name], dtype=np.float64) for t in tiers},
                  tiers=tiers, eps_total=float(rec["eps_total"]), n_seg_req=int(rec["n_seg_req"]),
                  n_seg=int(rec["n_seg"]), alpha=float(rec["alpha"]))
