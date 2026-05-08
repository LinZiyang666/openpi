"""Phase 3 threshold solver — derive (FH_thr, WS_thr) from warmup factor_raw.

Plan §5: warmup yaml runs ``AlwaysWarmStartJudge + DumpingJudge`` and
writes a JSONL with per-verdict ``factor_raw`` dicts. This module
replays those raw values through the recipe's Layer 3 calibration
(saturated buffer) and Layer 4 composer to produce a per-verdict score
sequence, then cuts that sequence at descending quantile boundaries to
yield the (full_hit, warm_start) thresholds for each (FH_ratio,
WS_ratio) cell of the 4x4 grid.

Public surface:

    Recipe                                         (dataclass)
    reconstruct_scores(jsonl_path, recipe)         -> list[float]
    derive_thresholds(scores, fh_ratio, ws_ratio)  -> tuple[float, float]
    solve_recipe(jsonl_path, recipe, grid)         -> dict (§5.8 schema)

CLI: not provided — call ``solve_recipe`` from the runner.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from openpi.cache.components.factors.base import CalibrationSamples
from openpi.cache.components.factors.calibrations.percentile_rolling import (
    PercentileRollingCalibration,
)
from openpi.cache.components.factors.composers import WeightedSumZeroNanComposer


# ----------------------------------------------------------------------
# Recipe descriptor
# ----------------------------------------------------------------------


@dataclass
class Recipe:
    """A Phase 3 recipe = the data the solver needs to score and threshold.

    ``declared_keys`` is the set of factor keys the recipe's composer
    weights (= keys with non-zero weight). ``orientations`` maps each
    declared key to ``"safe" | "risky" | "non_monotonic"`` and is fed to
    the composer's ``bind_orientations``. ``directions`` maps the
    non_monotonic keys to ``"high" | "low" | "range:[lo,hi]"``.

    Recipe instances are the bridge between phase3_spec (which knows the
    recipe's factor list) and the solver (which only needs the composer
    contract). Building a Recipe from a phase3_spec recipe entry is a
    pure-data transformation (no factor instantiation needed).
    """

    recipe_id: str
    declared_keys: list[str]
    orientations: dict[str, str]
    directions: dict[str, str] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Score reconstruction (plan §5.4)
# ----------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_per_key_finite_history(
    jsonl_path: Path, declared_keys: Iterable[str],
) -> dict[str, list[float]]:
    """Read warmup factor_raw column and return per-key finite-only history.

    Public helper shared by ``reconstruct_scores`` (which builds
    ``CalibrationSamples`` and lets ``bind_keys`` filter NaN internally)
    and the Phase 3 runner (which sends the buffer over the wire to
    populate the eval-side ``WarmupPool``). Returning the same source
    list under both calls is what guarantees the eval calibration
    saturated buffer matches the one the solver scored against — see
    G2 R1 B1 in the plan log.

    Order is preserved (row order); NaN / None / non-float values are
    dropped. The returned list is **uncapped** — phase3 warmup is at
    most ~420 verdicts/key, so memory and wire payload are bounded by
    construction.
    """
    keys = list(declared_keys)
    out: dict[str, list[float]] = {k: [] for k in keys}
    for row in _iter_jsonl(jsonl_path):
        raw = row.get("factor_raw") or {}
        for k in keys:
            v = raw.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                continue
            out[k].append(fv)
    return out


def reconstruct_scores(jsonl_path: Path, recipe: Recipe) -> list[float]:
    """Replay warmup verdicts through saturated-buffer calibration + composer.

    Returns one score per JSONL row, in row order. The score is the
    output of ``WeightedSumZeroNanComposer._score_only`` after each
    raw row has been passed through ``PercentileRollingCalibration``.
    The buffer is saturated from the first row onward (calibration is
    initialized from the same warmup column it then replays).

    Raises ``ValueError`` if any declared key has fewer than
    ``window_size = 50`` non-NaN samples (propagated from
    ``bind_keys``); the runner surfaces this as a recipe-level skip.
    """
    rows = list(_iter_jsonl(jsonl_path))

    # 1) Build CalibrationSamples from the warmup factor_raw column.
    per_key_history: dict[str, list[float]] = {k: [] for k in recipe.declared_keys}
    for row in rows:
        raw = row.get("factor_raw") or {}
        for k in recipe.declared_keys:
            per_key_history[k].append(float(raw.get(k, float("nan"))))
    samples = CalibrationSamples(samples=per_key_history)

    # 2) Saturate buffer (last 50 non-NaN per key). Fail-fast inside.
    cal = PercentileRollingCalibration(samples, window_size=50)
    cal.bind_keys(list(recipe.declared_keys))

    # 3) Composer with placeholder thresholds — only _score_only used.
    composer = WeightedSumZeroNanComposer(
        weights={k: 1.0 for k in recipe.declared_keys},
        full_hit_threshold=0.0,
        warm_start_threshold=0.0,
        warm_start_t=0.5,
        directions=recipe.directions or None,
    )
    composer.bind_orientations(recipe.orientations)

    # 4) Replay rows in order; cal(raw) appends-and-ranks.
    scores: list[float] = []
    for row in rows:
        calibrated = cal(row.get("factor_raw") or {})
        scores.append(composer._score_only(calibrated))
    return scores


# ----------------------------------------------------------------------
# Quantile cut math (plan §5.4 derive_thresholds)
# ----------------------------------------------------------------------


def derive_thresholds(
    scores: list[float], fh_ratio: float, ws_ratio: float,
) -> tuple[float, float]:
    """Return (FH_thr, WS_thr) from a descending-sorted score array.

    Cuts: top ``fh_ratio`` fraction goes to FULL_HIT (FH_thr = the
    last passing score), the next ``ws_ratio`` fraction goes to
    WARM_START. NaN scores are filtered first; raises if no usable
    scores remain (degenerate composer).
    """
    arr = np.asarray(
        [s for s in scores if not (s is None or math.isnan(s))],
        dtype=np.float64,
    )
    if arr.size == 0:
        raise RuntimeError("no usable scores in warmup — recipe likely degenerate")
    arr_desc = np.sort(arr)[::-1]
    n = arr_desc.size
    i_fh = max(0, min(n - 1, int(fh_ratio * n) - 1))
    i_ws = max(0, min(n - 1, int((fh_ratio + ws_ratio) * n) - 1))
    return float(arr_desc[i_fh]), float(arr_desc[i_ws])


# ----------------------------------------------------------------------
# Per-recipe driver (plan §5.8 output schema)
# ----------------------------------------------------------------------


def solve_recipe(
    jsonl_path: Path,
    recipe: Recipe,
    grid: list[tuple[float, float]],
    *,
    warmup_yaml_id: str | None = None,
) -> dict:
    """Run the full §5 pipeline on one warmup JSONL and return the §5.8 dict.

    On bind_keys / degenerate-composer failure, returns an ``error``-shaped
    dict (no ``cells`` field) so the runner can mark the recipe as ``_NA``.
    """
    base = {
        "recipe": recipe.recipe_id,
        "warmup_yaml_id": warmup_yaml_id,
        "warmup_jsonl": str(jsonl_path),
    }

    # Diagnostic: per-key non-NaN count is computed up-front so it's
    # available even on the bind_keys failure path.
    rows = list(_iter_jsonl(jsonl_path))
    per_key_non_nan: dict[str, int] = {k: 0 for k in recipe.declared_keys}
    for row in rows:
        raw = row.get("factor_raw") or {}
        for k in recipe.declared_keys:
            v = raw.get(k, float("nan"))
            if not (v is None or (isinstance(v, float) and math.isnan(v))):
                per_key_non_nan[k] += 1

    try:
        scores = reconstruct_scores(jsonl_path, recipe)
    except (ValueError, KeyError) as exc:
        return {
            **base,
            "n_warmup_rows": len(rows),
            "per_key_non_nan_count": per_key_non_nan,
            "error": str(exc),
        }

    finite = [s for s in scores if not (s is None or math.isnan(s))]
    score_stats = (
        {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
        }
        if finite
        else None
    )

    cells: dict[str, dict[str, float]] = {}
    for fh_ratio, ws_ratio in grid:
        try:
            fh_thr, ws_thr = derive_thresholds(scores, fh_ratio, ws_ratio)
        except RuntimeError as exc:
            return {
                **base,
                "n_warmup_rows": len(rows),
                "n_finite_scores": len(finite),
                "n_dropped_nan_scores": len(scores) - len(finite),
                "score_stats": score_stats,
                "per_key_non_nan_count": per_key_non_nan,
                "error": str(exc),
            }
        key = f"fh{fh_ratio}_ws{ws_ratio}"
        cells[key] = {"fh_thr": fh_thr, "ws_thr": ws_thr}

    return {
        **base,
        "n_warmup_rows": len(rows),
        "n_finite_scores": len(finite),
        "n_dropped_nan_scores": len(scores) - len(finite),
        "score_stats": score_stats,
        "per_key_non_nan_count": per_key_non_nan,
        "cells": cells,
    }
