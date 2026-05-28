"""Solve (T_fh, T_ws) thresholds from a warmup cp1_score distribution.

Consumes the warmup per-step JSONL (rows carry ``cp1_score`` — the final
modality-aggregated + trajectory-weighted weighted_score_sum top-1 score,
collected along the real policy trajectory via a force-MISS warmup judge).

For each target (fh_ratio, ws_ratio) cell:
  - quantile method: reuses verdict phase3 ``derive_thresholds`` (descending
    quantile cuts) — top fh_ratio -> FULL_HIT, next ws_ratio -> WARM_START.
  - zscore method (owner ask): T = mu + k*sigma cuts on the same distribution.

Degenerate cells where ``T_fh - T_ws < EPS`` are SKIPPED with a warning so every
emitted YAML satisfies the strict ``warm_tiers[i].threshold < judge.threshold``
config validation (a narrow/quantized score band can collapse the two quantiles).
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from exp.verdict_factor_judge.phase3.threshold_solver import derive_thresholds

EPS = 1e-6


def load_cp1_scores(per_step_jsonl: str | Path) -> list[float]:
    """Extract finite ``cp1_score`` values from a warmup per-step JSONL."""
    scores: list[float] = []
    for raw in Path(per_step_jsonl).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rec = json.loads(line)
        s = rec.get("cp1_score")
        if s is None:
            continue
        s = float(s)
        if not math.isnan(s):
            scores.append(s)
    return scores


def distribution_spread(scores: list[float]) -> dict[str, float]:
    """Front-gate stats: if the score band is too narrow, thresholding can't
    separate the three tiers (R1). Caller decides the fail-fast cutoff."""
    if len(scores) < 2:
        return {"n": len(scores), "min": float("nan"), "max": float("nan"), "std": float("nan"), "iqr": float("nan")}
    srt = sorted(scores)
    n = len(srt)
    q1, q3 = srt[n // 4], srt[(3 * n) // 4]
    return {"n": n, "min": srt[0], "max": srt[-1], "std": statistics.pstdev(scores), "iqr": q3 - q1}


def solve_quantile(scores: list[float], fh_ratio: float, ws_ratio: float) -> tuple[float, float] | None:
    """(T_fh, T_ws) via descending quantile cuts; None if degenerate (T_fh-T_ws<EPS)."""
    t_fh, t_ws = derive_thresholds(scores, fh_ratio, ws_ratio)
    if t_fh - t_ws < EPS:
        return None
    return t_fh, t_ws


def solve_zscore(scores: list[float], k_fh: float, k_ws: float) -> tuple[float, float] | None:
    """(T_fh, T_ws) = mu + k*sigma cuts; None if degenerate or sigma~0."""
    if len(scores) < 2:
        return None
    mu = statistics.fmean(scores)
    sigma = statistics.pstdev(scores)
    if sigma < EPS:
        return None
    t_fh, t_ws = mu + k_fh * sigma, mu + k_ws * sigma
    if t_fh - t_ws < EPS:
        return None
    return t_fh, t_ws


def grid_cells(fh_ratios=(0.2, 0.3, 0.4, 0.5), ws_ratios=None, max_total=1.0) -> list[tuple[float, float]]:
    """Feasible (fh_ratio, ws_ratio) cells = cross product of the two target-fraction
    axes, keeping only cells with ``fh + ws <= max_total``.

    ``fh + ws > 1`` is impossible (can't route >100% of requests into FULL_HIT+WARM);
    ``max_total < 1`` additionally leaves a MISS margin and stays out of the degenerate
    bottom tail where the WARM cut collapses onto the FULL_HIT cut. This makes the grid
    triangular when the fh axis extends past ~0.5: high-fh cells only pair with the few
    feasible (small) ws values. ``ws_ratios`` defaults to ``fh_ratios`` (square grid,
    e.g. the legacy 16-cell ``grid_cells()``)."""
    if ws_ratios is None:
        ws_ratios = fh_ratios
    return [(fh, ws) for fh in fh_ratios for ws in ws_ratios if fh + ws <= max_total + 1e-9]
