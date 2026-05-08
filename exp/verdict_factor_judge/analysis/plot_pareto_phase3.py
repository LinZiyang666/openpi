"""Pareto-frontier plot for Phase 3 — 11 recipes x 16 (FH, WS) cells, spatial16.

============================================================================
INFERENCE_RATIO FORMULA — Phase 3 variant
============================================================================

Per-verdict inference cost (same shape as plot_pareto.py, but Phase 3 fires
``warm_start`` at ``start_t = 0.5`` (plan §9):

    miss            -> 1.0
    full_hit        -> 0.0
    warm_start@0.5  -> 1.0 - 0.5 * (1.0 - 0.5) = 0.75

Phase 2 was 0.85 (start_t=0.7); Phase 3 WS is therefore cheaper by 0.10
inference per warm verdict — direct knock-on for the Pareto plot.

Inputs:

  * ``exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl`` —
    one row per (recipe_id, fh_ratio, ws_ratio) eval cell with fields
    ``yaml_id, recipe_id, fh_ratio, ws_ratio, fh_thr, ws_thr,
    success_rate, n_eval_verdicts, n_full_hit, n_warm_start, n_miss``.
  * ``exp/random_periodic_gate/analysis/aggregate.csv`` — random / periodic
    baseline cloud (filtered to ``cfg = spatial16_w8_d4``).

Output: ``exp/verdict_factor_judge/analysis/phase3_pareto.png``.

Usage::

    MPLBACKEND=Agg uv run python -m exp.verdict_factor_judge.analysis.plot_pareto_phase3
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  headless / no Qt
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Inference-ratio model
# ---------------------------------------------------------------------------


def warm_cost(start_t: float) -> float:
    return 1.0 - 0.5 * (1.0 - start_t)


# Phase 3: composer.warm_start_t locked to 0.5 across all 11 recipes
# (plan §9). Per-verdict warm cost = 0.75.
WARM_START_T_PHASE3 = 0.50


# Phase 3 is spatial16-only — keep the constant available for parity with
# phase2 plotter but only the spatial16 row is consulted.
CFG_KEYBUILDER = "spatial16_w8_d4"

# always-WARM baselines (500 ep, 3 cfgs); only spatial16 row used here.
WARM_SR_SPATIAL16 = {0.30: 0.942, 0.50: 0.952, 0.70: 0.976}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_random_periodic(path: Path) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["cfg"] != CFG_KEYBUILDER:
                continue
            out.append(
                (float(r["mean_inference_ratio"]), float(r["success_rate"]))
            )
    return out


def _load_phase3(summary_path: Path) -> list[dict]:
    """Load Phase 3 per-yaml summary (one row per eval cell)."""
    warm_c = warm_cost(WARM_START_T_PHASE3)
    rows: list[dict] = []
    with summary_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d or d.get("n_eval_verdicts") in (None, 0):
                continue
            n = d["n_eval_verdicts"]
            inf = (d["n_full_hit"] * 0.0
                   + d["n_warm_start"] * warm_c
                   + d["n_miss"] * 1.0) / n
            rows.append({
                "recipe_id": d["recipe_id"],
                "fh_ratio": d.get("fh_ratio"),
                "ws_ratio": d.get("ws_ratio"),
                "fh_thr": d.get("fh_thr"),
                "ws_thr": d.get("ws_thr"),
                "sr": d.get("success_rate"),
                "inf": inf,
            })
    return rows


def pareto_upper_frontier(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    pts = sorted(points, key=lambda p: p[0])
    frontier: list[tuple[float, float]] = []
    best_sr = -1.0
    for inf, sr in pts:
        if sr > best_sr:
            frontier.append((inf, sr))
            best_sr = sr
    return frontier


def is_pareto_dominated(
    inf: float, sr: float, baseline_pts: list[tuple[float, float]],
) -> bool:
    for bi, bs in baseline_pts:
        if bi <= inf and bs >= sr and (bi < inf or bs > sr):
            return True
    return False


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    rp_pts = _load_random_periodic(
        repo / "exp/random_periodic_gate/analysis/aggregate.csv"
    )
    summary_path = (
        repo / "exp/verdict_factor_judge/data/phase3/per_yaml_summary.jsonl"
    )
    if not summary_path.exists():
        raise SystemExit(f"phase3 summary not found: {summary_path}")
    rows = _load_phase3(summary_path)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    fig.suptitle(
        "Phase 3 — Data-driven Threshold Sweep (spatial16, 11 recipes x 16 cells)",
        fontsize=13, fontweight="bold",
    )

    # random / periodic cloud + frontier
    rx, ry = (zip(*rp_pts) if rp_pts else ([], []))
    ax.scatter(rx, ry, s=24, c="lightgray", marker="o", alpha=0.75,
               label="random / periodic", zorder=1)
    rp_front = pareto_upper_frontier(rp_pts)
    if rp_front:
        fx, fy = zip(*rp_front)
        ax.plot(fx, fy, "-", color="gray", alpha=0.55, linewidth=1.2,
                label="r/p Pareto frontier", zorder=2)

    # always-WARM baselines
    warm_pts = [
        (warm_cost(t), WARM_SR_SPATIAL16[t]) for t in (0.30, 0.50, 0.70)
    ]
    wx = [p[0] for p in warm_pts]
    wy = [p[1] for p in warm_pts]
    ax.scatter(wx, wy, s=240, c="red", marker="*", edgecolors="darkred",
               linewidths=1.4, label="always-WARM (start_t)", zorder=4)
    for (x, y), t in zip(warm_pts, (0.30, 0.50, 0.70)):
        ax.annotate(f" t={t}", (x, y), xytext=(7, -3),
                    textcoords="offset points", fontsize=9, color="darkred")
    ax.plot(wx, wy, "--", color="red", alpha=0.4, linewidth=1.0, zorder=3)

    # Phase 3 cells colored by recipe.
    cmap = plt.get_cmap("tab20")
    recipe_ids = sorted({r["recipe_id"] for r in rows})
    color_for = {rid: cmap(i % 20) for i, rid in enumerate(recipe_ids)}
    for rid in recipe_ids:
        cells = [r for r in rows if r["recipe_id"] == rid]
        cx = [r["inf"] for r in cells]
        cy = [r["sr"] for r in cells]
        ax.scatter(cx, cy, s=45, color=color_for[rid], marker="o",
                   edgecolors="navy", linewidths=0.6, alpha=0.85,
                   label=rid, zorder=3)

    # gold-circle strict-Pareto-positives vs r/p + always-WARM
    all_base = list(rp_pts) + warm_pts
    for r in rows:
        if not is_pareto_dominated(r["inf"], r["sr"], all_base):
            ax.scatter([r["inf"]], [r["sr"]], s=170, facecolors="none",
                       edgecolors="gold", linewidths=2.2, zorder=5)

    ax.set_xlabel("inference_ratio  (lower = more cache reuse)")
    ax.set_ylabel("success_rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.6, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.95, ncol=2)

    plt.tight_layout()
    out = repo / "exp/verdict_factor_judge/analysis/phase3_pareto.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}  size: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
