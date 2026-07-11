"""Offline calibration of the failure-aware margin gate (TRACER Phase 5).

Mechanism cohered on a class per the ``ScoreNormalizer.fit_from_scores``
paradigm: ``MarginGateCalibrator.calibrate`` fits the dual-retrieval / gate
parameters (``margin_lambda``, gate betas ``b0``/``b2``/``b3``, warm tier
threshold) offline from a *threshold-independent* per-step signal table,
minimizing the Eq 24 objective ``L_cal = BadHitRate + c_miss*MissRate +
c_warm*WarmCost``.

Offline-only, zero runtime coupling: it produces numeric params that the exp/
driver writes into YAML, where the online strategy reads ``margin_lambda`` and
the judge reads ``gate_betas`` independently — ``lambda`` never reverse-couples
the judge's verdict responsibility.

Fixed parameterization (plan D-P6): ``b1 == 1`` (margin unit coefficient),
``threshold == 0.5`` (g-space full_hit cutoff), warm ``start_t == 0.5``. The
search sweeps ``lambda``, ``b0`` (per-lambda margin percentiles -> ``tau_hit =
-b0``), ``b2``, ``b3``, and the g-space warm threshold. A WARM_START verdict
fires only when the winner carries a warm-start snapshot at ``start_t``
(``row["winner_has_warm_snapshot"]``). The argmin is broken deterministically
(higher tau_hit -> lower lambda -> smaller |b2| -> lexicographic).

Row schema (one dict per step, from ``build_calibration_table``): ``s_pos``,
``s_neg``, ``delta_pos`` (floats), ``u_t`` (float | None), ``winner_has_warm_
snapshot`` (bool), ``episode_success`` (bool).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

# ----------------------------------------------------------------------
# Fixed constants (plan D-P6) + warm-start cost
# ----------------------------------------------------------------------

# Remaining denoise cost after a warm start at start_t. Repo convention
# ``warm_cost = 1 - 0.5*(1 - start_t)`` (start_t=0.5 -> 0.75).
_WARM_START_T = 0.5


def _warm_cost(start_t: float) -> float:
    """Remaining denoise fraction after a warm start at ``start_t``."""
    return 1.0 - 0.5 * (1.0 - start_t)


def _sigmoid(z: float) -> float:
    """Numerically stable logistic (mirrors FailureAwareGateJudge)."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolated percentile over a pre-sorted list (numpy-free)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


@dataclasses.dataclass(frozen=True)
class CalibrationGrid:
    """Explicit, finite search grid (plan D-P6). 7*8*7*3*4 = 4704 candidates."""

    lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    b0_percentiles: tuple[float, ...] = (50, 60, 70, 75, 80, 85, 90, 95)
    b2s: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    b3s: tuple[float, ...] = (0.0, 0.5, 1.0)
    # None -> no warm tier; floats are g-space WARM_START cutoffs (< threshold).
    warm_thresholds: tuple[Optional[float], ...] = (None, 0.2, 0.3, 0.4)


class MarginGateCalibrator:
    """Fit ``(margin_lambda, gate_betas, warm_tiers)`` offline; runtime-decoupled."""

    START_T: float = _WARM_START_T
    THRESHOLD: float = 0.5  # g-space full_hit cutoff (fixed)
    B1: float = 1.0         # margin unit coefficient (fixed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def calibrate(
        cls,
        rows: list[dict],
        *,
        c_miss: float,
        c_warm: float,
        grid: Optional[CalibrationGrid] = None,
    ) -> dict:
        """Return the params minimizing ``L_cal`` over ``grid``.

        Returns ``{margin_lambda, gate_betas: {b0, b1, b2, b3}, threshold,
        warm_tiers}`` — pure JSON-able numerics ready to write into YAML.
        """
        if not rows:
            raise ValueError("MarginGateCalibrator.calibrate: empty rows")
        grid = grid or CalibrationGrid()

        best_key = None
        best_params: Optional[dict] = None
        for lam in grid.lambdas:
            # margin depends on lambda -> recompute the percentile ladder per
            # lambda (plan D-P6 / G1 R2 finding 4), never at a single baseline.
            margins = sorted(r["s_pos"] - lam * r["s_neg"] for r in rows)
            b0_candidates = [-_percentile(margins, p) for p in grid.b0_percentiles]
            for b0 in b0_candidates:
                for b2 in grid.b2s:
                    for b3 in grid.b3s:
                        for warm_thr in grid.warm_thresholds:
                            l_cal = cls._score(
                                rows, lam, b0, b2, b3, warm_thr, c_miss, c_warm
                            )["L_cal"]
                            tau_hit = -b0
                            # Quantize L_cal to the plan's 1e-9 tie tolerance so
                            # candidates within |ΔL| < 1e-9 tie, then break
                            # deterministically: higher tau_hit (safer) -> lower
                            # lambda -> smaller |b2| -> lexicographic tuple.
                            key = (
                                round(l_cal, 9), -tau_hit, lam, abs(b2),
                                b0, b2, b3, (float("inf") if warm_thr is None else warm_thr),
                            )
                            if best_key is None or key < best_key:
                                best_key = key
                                best_params = {
                                    "margin_lambda": lam,
                                    "gate_betas": {"b0": b0, "b1": cls.B1, "b2": b2, "b3": b3},
                                    "threshold": cls.THRESHOLD,
                                    "warm_tiers": (
                                        [] if warm_thr is None
                                        else [{"threshold": warm_thr, "start_t": cls.START_T}]
                                    ),
                                }
        return best_params

    @classmethod
    def evaluate(cls, rows: list[dict], params: dict, *, c_miss: float, c_warm: float) -> dict:
        """L_cal + component metrics for a given params dict (for driver logging).

        Uses the same replay as ``calibrate`` so a solved params dict re-scores to
        the winning objective. Returns ``{L_cal, BadHitRate, MissRate, WarmCost,
        n_full_hit, n_warm_start, n_miss}``.
        """
        gb = params["gate_betas"]
        wt = params.get("warm_tiers") or []
        warm_thr = wt[0]["threshold"] if wt else None
        return cls._score(
            rows, params["margin_lambda"], gb["b0"], gb.get("b2", 0.0),
            gb.get("b3", 0.0), warm_thr, c_miss, c_warm,
        )

    # ------------------------------------------------------------------
    # Internal: replay a single candidate over the signal table
    # ------------------------------------------------------------------

    @classmethod
    def _score(
        cls,
        rows: list[dict],
        lam: float,
        b0: float,
        b2: float,
        b3: float,
        warm_thr: Optional[float],
        c_miss: float,
        c_warm: float,
    ) -> dict:
        n = len(rows)
        n_fh = n_fh_bad = n_ws = n_miss = 0
        for r in rows:
            margin = r["s_pos"] - lam * r["s_neg"]
            u_t = r.get("u_t")
            # None / NaN u_t drops the term -> margin-only (mirrors the gate).
            u_term = 0.0 if (u_t is None or math.isnan(u_t)) else b2 * u_t
            g = _sigmoid(b0 + cls.B1 * margin + u_term + b3 * r["delta_pos"])
            if g >= cls.THRESHOLD:
                n_fh += 1
                if not r["episode_success"]:  # bad(t) := episode failed
                    n_fh_bad += 1
            elif warm_thr is not None and g >= warm_thr and r["winner_has_warm_snapshot"]:
                n_ws += 1
            else:
                n_miss += 1
        bad_hit_rate = n_fh_bad / (n_fh + 1e-8)          # Eq 27
        miss_rate = n_miss / n
        warm_cost = n_ws * _warm_cost(cls.START_T) / n
        l_cal = bad_hit_rate + c_miss * miss_rate + c_warm * warm_cost
        return {
            "L_cal": l_cal, "BadHitRate": bad_hit_rate, "MissRate": miss_rate,
            "WarmCost": warm_cost, "n_full_hit": n_fh, "n_warm_start": n_ws, "n_miss": n_miss,
        }
