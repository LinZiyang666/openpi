"""RIT-PL: piecewise-linear risk curves and inference-ratio addressing for the
Risk-Indexed Threshold (dispatch-surface line; plan
``logs/rit_pl_ir_ladder_plan.log.md``).

The frozen ladder (``fit_surface.py``) fits the conditional deviation quantile
``q_a(s)`` of each reuse tier as a step function on equal-frequency bins and
snaps every cut to a bin edge, so the reachable operating points are coarse.
Order restriction also pools adjacent bins into plateaus, and a plateau is
data, not a bin-count artefact. This module fits the same two-layer pinball
objective on knot values with linear interpolation between equal-frequency
knots, a strict slope floor (``EPS_TOTAL``) that removes plateaus, the tier
nesting ``q_warm <= q_full`` and ``q >= 0``. The cut is then the exact
crossing of the fitted curve with the tolerance ``delta`` (``cut_at``),
``delta -> theta`` is continuous and strictly monotone, and the predicted
inference ratio on the shadow table can be inverted deterministically
(``delta_for_ir``: nearest attainable point on a finite table).

Everything here is outcome-blind: only ``s``, the deviation surrogates
``y7`` / ``y10`` and the frozen unit costs enter; no rollout outcome is read.

Public interface: ``PLFit``, ``choose_knots``, ``fit_pl_quantile``,
``predict``, ``cut_at``, ``cuts``, ``predicted_ir``, ``attainable_range``,
``ir_curve``, ``delta_for_ir``, ``floor_info``, ``ecdf_quantile``,
``pl_fit_digests``, ``fit_record_fields``, ``fit_from_record``.

Key dependencies: ``scipy.optimize.linprog`` (HiGHS), ``analytic_cost``
(the single cost authority), ``fit_surface._digest_obj``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from exp.dispatch_surface.analysis.analytic_cost import PINNED_START_T_WS, unit_cost
from exp.dispatch_surface.fit_surface import _digest_obj

ESTIMATOR = "pl_knots_v1"
PROTOCOL_RIT_PL = "dispatch_surface_rit_pl_dev"
#: Requested segment counts, tried in order; every segment must hold at least
#: MIN_SEG_SAMPLES rows after knot de-duplication, else the next rung is tried.
KNOT_LADDER = (24, 12, 6)
MIN_SEG_SAMPLES = 8
#: Total forced decrease of each layer over the whole s range (in delta units),
#: spread as a per-unit-s slope floor. It removes isotonic plateaus so that the
#: crossing with any delta is unique; on a floor segment the fitted gradient IS
#: this prior (see ``floor_info``).
EPS_TOTAL = 0.02
#: Inference-ratio tolerances in percentage points of the always-full cost.
IR_TOL = 0.05
IR_MAX_GAP = 0.5
LAYERS = ("warm", "full")

_FULL_MS = unit_cost("FULL_HIT", None)
_WARM_MS = unit_cost("WARM_START", PINNED_START_T_WS)
_MISS_MS = unit_cost("MISS", None)


# ------------------------------------------------------------------
# Fit container and input contracts
# ------------------------------------------------------------------


@dataclasses.dataclass
class PLFit:
    """Two-layer piecewise-linear quantile fit on shared knots.

    ``q_warm`` / ``q_full`` are the fitted knot values of the tau=7 (warm
    start) and tau=10 (full replay) layers; ``n_seg_req`` is the ladder rung
    that was requested and ``n_seg == len(knots) - 1`` the number of segments
    actually obtained after quantile de-duplication.
    """

    knots: np.ndarray
    q_warm: np.ndarray
    q_full: np.ndarray
    eps_total: float
    n_seg_req: int
    n_seg: int
    alpha: float


def _as_1d(name: str, arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a non-empty 1-D array of finite values")
    return a


def _layer(fit: PLFit, layer: str) -> np.ndarray:
    if layer == "warm":
        return np.asarray(fit.q_warm, dtype=np.float64)
    if layer == "full":
        return np.asarray(fit.q_full, dtype=np.float64)
    raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")


def segment_index(knots: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Segment k with knots[k] <= s < knots[k+1]; s == knots[-1] joins the last segment."""
    return np.clip(np.searchsorted(knots, s, side="right") - 1, 0, len(knots) - 2)


# ------------------------------------------------------------------
# Knot ladder and the knot-value pinball LP
# ------------------------------------------------------------------


def choose_knots(s, ladder=KNOT_LADDER) -> tuple[np.ndarray, int] | None:
    """Equal-frequency knots at the first ladder rung whose every segment holds
    >= MIN_SEG_SAMPLES rows. Returns ``(knots, n_seg_req)`` so the rung that
    was actually adopted survives quantile de-duplication; ``None`` when the
    ladder is exhausted (the caller's stop-loss)."""
    s = _as_1d("s", s)
    for n_req in ladder:
        if int(n_req) < 2:
            raise ValueError("every ladder rung must request at least two segments")
        knots = np.unique(np.quantile(s, np.linspace(0.0, 1.0, int(n_req) + 1), method="linear"))
        n_seg = len(knots) - 1
        if n_seg < 2:
            continue
        counts = np.bincount(segment_index(knots, s), minlength=n_seg)
        if counts.min() >= MIN_SEG_SAMPLES:
            return knots, int(n_req)
    return None


def fit_pl_quantile(s, y7, y10, knots, *, n_seg_req: int, alpha: float, eps_total: float) -> PLFit:
    """Joint two-layer pinball LP on knot values (level ``1 - alpha``).

    Constraints: strict monotone floor ``q[k] - q[k+1] >= eps * (x[k+1] - x[k])``
    with ``eps = eps_total / (x[-1] - x[0])``; nesting ``q_warm[k] <= q_full[k]``;
    ``q >= 0``. Each row's prediction is the linear interpolation between its
    two enclosing knots, so the objective stays linear in the knot values.
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    s, y7, y10 = _as_1d("s", s), _as_1d("y7", y7), _as_1d("y10", y10)
    if not (len(s) == len(y7) == len(y10)):
        raise ValueError("s, y7 and y10 must have equal length")
    knots = _as_1d("knots", knots)
    if len(knots) < 3 or not (np.diff(knots) > 0).all():
        raise ValueError("knots must be strictly increasing with at least two segments")
    n_seg = len(knots) - 1
    if int(n_seg_req) < n_seg:
        raise ValueError("n_seg_req must be >= the number of segments actually used")
    if not (0.0 < float(alpha) <= 0.5):
        raise ValueError("alpha must lie in (0, 0.5]")
    if float(eps_total) < 0.0:
        raise ValueError("eps_total must be non-negative")

    n_knots, n = len(knots), len(s)
    seg = segment_index(knots, s)
    x0, x1 = knots[seg], knots[seg + 1]
    w = np.clip((s - x0) / (x1 - x0), 0.0, 1.0)                 # weight on the right knot
    eps = float(eps_total) / float(knots[-1] - knots[0])
    n_grid = 2 * n_knots
    n_var = n_grid + 4 * n
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

    for layer, y in enumerate((y7, y10)):
        base = n_grid + layer * 2 * n
        c[base:base + n] = 1.0 - float(alpha)
        c[base + n:base + 2 * n] = float(alpha)
        g0 = layer * n_knots + seg
        g1 = g0 + 1
        for i in range(n):
            # y - q(s) <= u  and  q(s) - y <= v  with q(s) = (1-w) q[k] + w q[k+1]
            add({int(g0[i]): -(1.0 - w[i]), int(g1[i]): -w[i], base + i: -1.0}, -float(y[i]))
            add({int(g0[i]): (1.0 - w[i]), int(g1[i]): w[i], base + n + i: -1.0}, float(y[i]))
        for k in range(n_seg):
            # q[k+1] - q[k] <= -eps * dx  (strict decrease floor)
            add({layer * n_knots + k + 1: 1.0, layer * n_knots + k: -1.0},
                -eps * float(knots[k + 1] - knots[k]))
    for k in range(n_knots):
        add({k: 1.0, n_knots + k: -1.0}, 0.0)                    # q_warm <= q_full
    a_ub = coo_matrix((vals, (r_idx, c_idx)), shape=(row, n_var)).tocsr()
    bounds = [(0.0, None)] * n_var
    res = linprog(c, A_ub=a_ub, b_ub=np.asarray(b_ub), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"RIT-PL LP failed: {res.message}")
    q = res.x[:n_grid]
    return PLFit(knots=knots, q_warm=q[:n_knots].copy(), q_full=q[n_knots:].copy(),
                 eps_total=float(eps_total), n_seg_req=int(n_seg_req), n_seg=int(n_seg),
                 alpha=float(alpha))


# ------------------------------------------------------------------
# Evaluation, exact crossings and the deployed verdict cost
# ------------------------------------------------------------------


def predict(fit: PLFit, s, layer: str) -> np.ndarray:
    """Fitted quantile curve at ``s`` (clamped to the knot range)."""
    q = _layer(fit, layer)
    knots = np.asarray(fit.knots, dtype=np.float64)
    sc = np.clip(_as_1d("s", s), knots[0], knots[-1])
    k = segment_index(knots, sc)
    w = (sc - knots[k]) / (knots[k + 1] - knots[k])
    return (1.0 - w) * q[k] + w * q[k + 1]


def cut_at(fit: PLFit, layer: str, delta: float) -> float:
    """Exact crossing of the fitted layer with ``delta``: the smallest s whose
    fitted risk is <= delta. ``knots[0]`` when the whole curve is admissible,
    ``+inf`` when no point is. Requires ``eps_total > 0`` (unique crossing)."""
    if not (float(fit.eps_total) > 0.0):
        raise ValueError("cut_at needs a strictly monotone fit (eps_total > 0)")
    if not math.isfinite(delta):
        raise ValueError("delta must be finite")
    q = _layer(fit, layer)
    knots = np.asarray(fit.knots, dtype=np.float64)
    if q[0] <= delta:
        return float(knots[0])
    if q[-1] > delta:
        return math.inf
    idx = int(np.where(q <= delta)[0][0])
    k = idx - 1
    return float(knots[k] + (q[k] - delta) / (q[k] - q[k + 1]) * (knots[k + 1] - knots[k]))


def cuts(fit: PLFit, delta: float) -> tuple[float, float]:
    """``(theta_full, theta_warm)`` at ``delta``; nesting ``theta_warm <= theta_full``
    holds by construction and is asserted (a floating tie is clamped)."""
    theta_full = cut_at(fit, "full", delta)
    theta_warm = cut_at(fit, "warm", delta)
    if theta_warm > theta_full:
        if theta_warm - theta_full > 1e-9:
            raise RuntimeError("tier nesting violated: warm cut above full cut")
        theta_warm = theta_full
    return theta_full, theta_warm


def predicted_ir(s, theta_full: float, theta_warm: float) -> float:
    """Analytic cost of the deployed rule on the rows of ``s``, in percent of the
    always-full (all-MISS) cost. Row semantics equal ``surface_verdict`` for an
    s-only artifact: ``s >= theta_full`` -> FULL_HIT, else ``s >= theta_warm``
    -> WARM_START, else MISS."""
    s = _as_1d("s", s)
    full = s >= theta_full
    warm = (~full) & (s >= theta_warm)
    n_full, n_warm = int(full.sum()), int(warm.sum())
    n_miss = len(s) - n_full - n_warm
    total = n_full * _FULL_MS + n_warm * _WARM_MS + n_miss * _MISS_MS
    return 100.0 * total / (len(s) * _MISS_MS)


def _endpoints(fit: PLFit) -> tuple[float, float]:
    """Positive delta domain endpoints: below every knot value (all-MISS when
    reachable) and above every knot value (everything admitted)."""
    q_min = float(min(np.min(fit.q_warm), np.min(fit.q_full)))
    q_max = float(max(np.max(fit.q_warm), np.max(fit.q_full)))
    tiny = float(np.nextafter(0.0, 1.0))
    d_lo = 0.5 * q_min if q_min > tiny else tiny
    return d_lo, q_max + 1.0


def _ir_at(fit: PLFit, s: np.ndarray, delta: float) -> float:
    theta_full, theta_warm = cuts(fit, delta)
    return predicted_ir(s, theta_full, theta_warm)


def attainable_range(fit: PLFit, s) -> tuple[float, float]:
    """``(IR_lo, IR_hi)`` reachable inside the positive delta domain."""
    s = _as_1d("s", s)
    d_lo, d_hi = _endpoints(fit)
    return _ir_at(fit, s, d_hi), _ir_at(fit, s, d_lo)


def ir_curve(fit: PLFit, s, n: int = 200) -> list[tuple[float, float]]:
    """``(delta, IR)`` on ``n`` equally spaced deltas over the positive domain."""
    s = _as_1d("s", s)
    d_lo, d_hi = _endpoints(fit)
    return [(float(d), _ir_at(fit, s, float(d))) for d in np.linspace(d_lo, d_hi, int(n))]


def delta_for_ir(fit: PLFit, s, target: float, *, tol: float = IR_TOL) -> dict:
    """Deterministic nearest-attainable inverse of the predicted inference ratio.

    ``IR(delta)`` is nonincreasing but a step function on a finite table, so a
    target is generally not hit exactly. The bracket ``IR(lo) > target > IR(hi)``
    is bisected until it collapses (relative width 1e-12) or both ends are
    within ``tol``; both ends are evaluated explicitly and the closer one is
    returned, ties going to the conservative (higher-cost) ``lo`` end. The
    signed miss ``ir_gap = predicted_ir - target`` is always reported.
    """
    s = _as_1d("s", s)
    d_lo, d_hi = _endpoints(fit)
    ir_top, ir_bot = _ir_at(fit, s, d_lo), _ir_at(fit, s, d_hi)
    if not (ir_bot <= target <= ir_top) or target >= 100.0:
        raise ValueError(f"target IR {target} outside the attainable range [{ir_bot:.4f}, {ir_top:.4f}] (and < 100)")

    def pack(delta: float, ir: float, lo: float, ir_lo: float, hi: float, ir_hi: float) -> dict:
        theta_full, theta_warm = cuts(fit, delta)
        return {"delta": float(delta), "theta_full": float(theta_full), "theta_warm": float(theta_warm),
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


def floor_info(fit: PLFit, s, delta: float) -> dict:
    """Per layer: the segment holding the cut, its fitted risk drop, the slope
    floor it must satisfy, whether the drop sits ON the floor (a data plateau
    tilted by the eps prior) and the segment's share of the rows."""
    s = _as_1d("s", s)
    knots = np.asarray(fit.knots, dtype=np.float64)
    eps = float(fit.eps_total) / float(knots[-1] - knots[0])
    seg_all = segment_index(knots, s)
    out = {}
    for layer in LAYERS:
        theta = cut_at(fit, layer, delta)
        if not math.isfinite(theta):
            out[layer] = None
            continue
        q = _layer(fit, layer)
        k = int(segment_index(knots, np.asarray([theta]))[0])
        drop = float(q[k] - q[k + 1])
        floor = eps * float(knots[k + 1] - knots[k])
        out[layer] = {"segment": k, "risk_drop": drop, "eps_floor": floor,
                      "on_eps_floor": bool(drop <= floor * (1.0 + 1e-6)),
                      "segment_share": float(np.mean(seg_all == k))}
    return out


def ecdf_quantile(y, delta: float) -> float:
    """Right-continuous empirical CDF of ``y`` at ``delta``."""
    return float(np.mean(_as_1d("y", y) <= delta))


# ------------------------------------------------------------------
# Record round-trip and digests
# ------------------------------------------------------------------


def pl_fit_digests(fit: PLFit) -> dict:
    """Canonical digests of the three fitted arrays (``knots``, ``q_warm``, ``q_full``)."""
    return {"knots": _digest_obj(np.asarray(fit.knots).tolist()),
            "q_warm": _digest_obj(np.asarray(fit.q_warm).tolist()),
            "q_full": _digest_obj(np.asarray(fit.q_full).tolist())}


def fit_record_fields(fit: PLFit) -> dict:
    """JSON-ready fit fields; ``fit_from_record`` inverts them exactly."""
    return {"knots": np.asarray(fit.knots, dtype=np.float64).tolist(),
            "q_warm": np.asarray(fit.q_warm, dtype=np.float64).tolist(),
            "q_full": np.asarray(fit.q_full, dtype=np.float64).tolist(),
            "eps_total": float(fit.eps_total), "n_seg_req": int(fit.n_seg_req),
            "n_seg": int(fit.n_seg), "alpha": float(fit.alpha)}


def fit_from_record(rec: dict) -> PLFit:
    """Rebuild a ``PLFit`` from ``fit_record_fields`` output (JSON round-trip exact)."""
    return PLFit(knots=np.asarray(rec["knots"], dtype=np.float64),
                 q_warm=np.asarray(rec["q_warm"], dtype=np.float64),
                 q_full=np.asarray(rec["q_full"], dtype=np.float64),
                 eps_total=float(rec["eps_total"]), n_seg_req=int(rec["n_seg_req"]),
                 n_seg=int(rec["n_seg"]), alpha=float(rec["alpha"]))
