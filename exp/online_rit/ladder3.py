"""The three-warm-tier ladder and its offline fits (R' and the d-unit curves).

``exp.robocasa365.rit_cost_rc.ladder`` always prepends a FULL_HIT rung and
``exp.rit_pareto.rit_k.fit_pl_quantile_k`` always nests its layers, both of
which the online line's ladder does not have. Neither existing function is
modified; this module builds the three warm rungs itself and drives the
existing single-layer / joint LP through proxy tiers whose only job is to
pass ``rit_k``'s tier checks (strictly increasing cost below the Pi0.5 MISS
constant). Proxy costs never leave this module: real GR00T prices come from
``common.CostLedger``.

Public interface: ``warm_tiers``, ``fit_nested`` (R': D columns, ordered by
tier), ``fit_independent`` (one single-layer LP per tier, no nesting),
``curves_from_fit``, ``rprime_cuts``.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from exp.online_rit.common import SCHEDULE, TIER_COLUMNS, TIER_TS, CostLedger, tier_index_of, tier_name
from exp.rit_pareto import rit_k
from openpi.cache.components.online_rit import cut_at_pl

EPS_TOTAL = 0.02  # rit_cost_rc.EPS_TOTAL: minimum total drop of the LP curves


@dataclasses.dataclass(frozen=True)
class WarmTier:
    name: str
    start_t: float
    index: int
    remaining_steps: int
    d_column: str
    y_column: str


def warm_tiers(start_ts: tuple[float, ...] = TIER_TS) -> tuple[WarmTier, ...]:
    """Riskiest first (fewest remaining steps first), with the table columns each tier reads."""
    out = []
    prev = None
    for t in start_ts:
        t4 = round(float(t), 4)
        if prev is not None and t4 >= prev:
            raise ValueError("tiers must be strictly decreasing in start_t")
        prev = t4
        d_col, y_col = TIER_COLUMNS[t4]
        out.append(WarmTier(tier_name(t4), t4, tier_index_of(t4), SCHEDULE.remaining_steps(t4), d_col, y_col))
    return tuple(out)


def tier_costs(ledger: CostLedger, tiers: tuple[WarmTier, ...]) -> dict[str, float]:
    return {t.name: ledger.warm_ms(t.start_t) for t in tiers}


@dataclasses.dataclass(frozen=True)
class _ProxyTier:
    """What ``rit_k`` needs of a tier: a name, a y column and a proxy cost."""

    name: str
    hit_type: str
    start_t: float
    y_key: str
    cost_ms: float


def _proxies(tiers: tuple[WarmTier, ...]) -> tuple[_ProxyTier, ...]:
    k = len(tiers)
    return tuple(
        _ProxyTier(t.name, "WARM_START", t.start_t, t.y_column, (j + 1) / (k + 1) * rit_k.MISS_MS)
        for j, t in enumerate(tiers)
    )


def fit_nested(s, ys: dict[str, np.ndarray], knots, *, alpha: float, tiers: tuple[WarmTier, ...]) -> rit_k.PLFitK:
    """R': the old method's joint, nested pinball LP over the D columns."""
    return rit_k.fit_pl_quantile_k(
        s, ys, knots, tiers=_proxies(tiers), n_seg_req=len(knots) - 1, alpha=alpha, eps_total=EPS_TOTAL
    )


def fit_independent(s, ys: dict[str, np.ndarray], knots, *, alpha: float, tiers: tuple[WarmTier, ...]) -> dict[str, rit_k.PLFitK]:
    """One single-layer LP per tier (no cross-tier nesting), for the d-unit curves."""
    out = {}
    for proxy in _proxies(tiers):
        out[proxy.name] = rit_k.fit_pl_quantile_k(
            s, {proxy.name: ys[proxy.name]}, knots, tiers=(proxy,), n_seg_req=len(knots) - 1, alpha=alpha, eps_total=EPS_TOTAL
        )
    return out


def curves_from_fit(fit: rit_k.PLFitK) -> dict[str, list[float]]:
    return {name: [float(x) for x in q] for name, q in fit.q.items()}


def rprime_cuts(fit: rit_k.PLFitK, delta: float) -> dict[str, float]:
    """R' cuts at tolerance ``delta`` with the old method's nesting clamp (``rit_k.cuts``)."""
    return {k: float(v) for k, v in rit_k.cuts(fit, delta).items()}


def independent_cuts(knots: np.ndarray, curves: dict[str, list[float]], delta: float) -> dict[str, float]:
    """Per-tier crossing without nesting: the online judge's rule on frozen curves."""
    return {name: cut_at_pl(np.asarray(knots), np.asarray(q), delta) for name, q in curves.items()}


def dispatch(score: float, cuts_in_order: list[float]) -> int | None:
    """Index of the first tier (riskiest first) whose cut the score clears, else None."""
    for i, theta in enumerate(cuts_in_order):
        if math.isfinite(theta) and score >= theta:
            return i
    return None
