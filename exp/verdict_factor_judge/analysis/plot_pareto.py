"""Pareto frontier plot — verdict_factor_judge Phase 2 Layer 1 yamls vs baselines.

============================================================================
INFERENCE_RATIO FORMULA  (project-wide convention — DO NOT change without
updating ``logs/verdict_factor_judge_experiment_plan.log.md`` §inf_ratio)
============================================================================

Per-verdict inference cost contribution:

    miss            -> 1.0
    full_hit        -> 0.0
    warm_start @ t  -> 1.0 - 0.5 * (1.0 - t)

Background:

* Flow matching convention used by openpi (see ``src/openpi/models/pi0.py``
  line 273 docstring): ``t=1`` is noise, ``t=0`` is the target distribution;
  ``dt = -1/num_steps`` integrates from t=1 down to 0.

* ``warm_start`` at ``start_t = t`` skips the noisy ``t=1 -> t`` portion of
  the path and runs the remaining ``(1 - t)`` fraction of denoising steps.

* The ``0.5`` factor in the warm-start cost = action-expert is roughly half
  the compute of full inference (KV cache reuse + no vision/language expert
  re-eval), so warm-start at ``start_t = t`` costs ``0.5 * (1 - t)`` LESS
  than full miss → ``inf = 1 - 0.5 * (1 - t)``.

Concrete values:

    start_t   warm cost
    0.7       0.85   (Phase 2 Layer 1 default; all_nan_fallback fires here)
    0.5       0.75
    0.3       0.65

Common pitfalls (do NOT do these):

    inf = 1 - start_t      # WRONG — was the original Pareto plot bug
    inf = start_t          # WRONG — second-attempt bug
    inf = 1 - 0.5*(1-t)    # CORRECT

The corresponding aggregation across n verdicts:

    inf_ratio = (n_full_hit*0.0 + n_warm*warm_cost(t) + n_miss*1.0) / n_total

============================================================================

Usage::

    MPLBACKEND=Agg uv run python -m exp.verdict_factor_judge.analysis.plot_pareto

Output: ``exp/verdict_factor_judge/analysis/phase2_layer1_pareto.png``.

Inputs (relative to repo root):

* ``exp/random_periodic_gate/analysis/aggregate.csv`` — random / periodic
  baseline points (78 rows × 3 cfg).
* ``exp/warm_start/data/state_full_<cfg>.json`` — always-WARM SR per cfg
  per start_t. Hard-coded below for clarity (values rarely change).
* ``exp/verdict_factor_judge/data/phase2_layer1/<cfg>/per_yaml_summary.jsonl``
  — Phase 2 Layer 1 per-yaml hit / warm / miss counts + success_rate.
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
    """Per-verdict inference contribution for a warm_start at the given t.

    See module docstring for derivation.
    """
    return 1.0 - 0.5 * (1.0 - start_t)


# Phase 2 Layer 1 yamls fire warm at start_t=0.7 (all_nan_fallback default
# + composer warm_start_t when T-DUAL_07). Update if a future phase mixes
# multiple start_t per yaml — current per-yaml summary doesn't carry the
# breakdown so we treat all warm as 0.7-fired.
WARM_START_T_PHASE2 = 0.70


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


CFG_TO_KEYBUILDER = {
    "clip": "clip_w7_d4",
    "max_pool": "max_pool_w3_d5",
    "spatial16": "spatial16_w8_d4",
}

# always-WARM baselines: SR per (cfg, start_t).
# Source: exp/warm_start/data/state_full_<cfg>.json (500 ep × 3 cfg).
WARM_SR = {
    "clip_w7_d4":      {0.30: 0.940, 0.50: 0.960, 0.70: 0.980},
    "max_pool_w3_d5":  {0.30: 0.946, 0.50: 0.966, 0.70: 0.964},
    "spatial16_w8_d4": {0.30: 0.942, 0.50: 0.952, 0.70: 0.976},
}


def _load_random_periodic(path: Path) -> list[dict]:
    """Load random / periodic baseline points (cfg, sr, inf)."""
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "cfg": r["cfg"],
                "sr": float(r["success_rate"]),
                "inf": float(r["mean_inference_ratio"]),
            })
    return rows


def _load_phase2_layer1(data_root: Path) -> dict[str, list[dict]]:
    """Load Phase 2 Layer 1 yaml stats per cfg, computing inf_ratio per
    the project-wide formula. Returns ``{cfg: [{stem, sr, inf, h, w, m}]}``."""
    out: dict[str, list[dict]] = {}
    warm_c = warm_cost(WARM_START_T_PHASE2)
    for cfg, kb in CFG_TO_KEYBUILDER.items():
        rows = []
        summary = data_root / cfg / "per_yaml_summary.jsonl"
        with summary.open() as fh:
            for line in fh:
                d = json.loads(line)
                n = d["n_eval_verdicts"]
                if not n:
                    continue
                hit = d["n_full_hit"]
                warm = d["n_warm_start"]
                miss = d["n_miss"]
                inf = (hit * 0.0 + warm * warm_c + miss * 1.0) / n
                rows.append({
                    "stem": d["yaml_id"].replace(f"{kb}_phase2_", ""),
                    "sr": d["success_rate"],
                    "inf": inf,
                    "h": hit / n,
                    "w": warm / n,
                    "m": miss / n,
                })
        out[cfg] = rows
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def pareto_upper_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-left Pareto frontier on (inf, sr).

    Iterate points in order of ascending inf. Keep only points that strictly
    raise the running-max sr — this gives a monotone-non-decreasing line in
    the (inf, sr) plane.
    """
    pts = sorted(points, key=lambda p: p[0])
    frontier = []
    best_sr = -1.0
    for inf, sr in pts:
        if sr > best_sr:
            frontier.append((inf, sr))
            best_sr = sr
    return frontier


def is_pareto_dominated(inf: float, sr: float, baseline_pts: list[tuple[float, float]]) -> bool:
    """Return True if any baseline point dominates (inf, sr).

    Domination = lower-or-equal inf AND higher-or-equal sr AND at least one
    of those is strict.
    """
    for bi, bs in baseline_pts:
        if bi <= inf and bs >= sr and (bi < inf or bs > sr):
            return True
    return False


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    rp = _load_random_periodic(repo / "exp/random_periodic_gate/analysis/aggregate.csv")
    p2 = _load_phase2_layer1(repo / "exp/verdict_factor_judge/data/phase2_layer1")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(
        "Phase 2 Layer 1 — Pareto Frontier  "
        "(SR vs inference_ratio,  warm cost = 1 - 0.5 * (1 - start_t))",
        fontsize=14, fontweight="bold",
    )

    for ax, cfg in zip(axes, ("clip", "max_pool", "spatial16")):
        kb = CFG_TO_KEYBUILDER[cfg]

        # random / periodic cloud + frontier
        rp_pts = [(r["inf"], r["sr"]) for r in rp if r["cfg"] == kb]
        rx, ry = (zip(*rp_pts) if rp_pts else ([], []))
        ax.scatter(rx, ry, s=24, c="lightgray", marker="o", alpha=0.75,
                   label="random / periodic", zorder=1)
        rp_front = pareto_upper_frontier(rp_pts)
        if rp_front:
            fx, fy = zip(*rp_front)
            ax.plot(fx, fy, "-", color="gray", alpha=0.55, linewidth=1.2,
                    label="r/p Pareto frontier", zorder=2)

        # always-WARM baselines
        warm_pts = [(warm_cost(t), WARM_SR[kb][t]) for t in (0.30, 0.50, 0.70)]
        wx = [p[0] for p in warm_pts]
        wy = [p[1] for p in warm_pts]
        ax.scatter(wx, wy, s=240, c="red", marker="*", edgecolors="darkred",
                   linewidths=1.4, label="always-WARM (start_t)", zorder=4)
        for (x, y), t in zip(warm_pts, (0.30, 0.50, 0.70)):
            ax.annotate(f" t={t}", (x, y), xytext=(7, -3),
                        textcoords="offset points", fontsize=9, color="darkred")
        ax.plot(wx, wy, "--", color="red", alpha=0.4, linewidth=1.0, zorder=3)

        # Phase 2 Layer 1 yamls
        rows = p2[cfg]
        px = [r["inf"] for r in rows]
        py = [r["sr"] for r in rows]
        ax.scatter(px, py, s=55, c="steelblue", marker="o",
                   edgecolors="navy", linewidths=0.8, alpha=0.85,
                   label="Phase 2 Layer 1 yamls", zorder=3)

        # gold-circle strict-Pareto-positives (vs both baseline groups combined)
        all_base = rp_pts + warm_pts
        for r in rows:
            if not is_pareto_dominated(r["inf"], r["sr"], all_base):
                ax.scatter([r["inf"]], [r["sr"]], s=170, facecolors="none",
                           edgecolors="gold", linewidths=2.2, zorder=5)
                label = (r["stem"]
                         .replace("_t_full", "")
                         .replace("w_long_risk", "LR")
                         .replace("_only", ""))
                ax.annotate(label, (r["inf"], r["sr"]), xytext=(7, 7),
                            textcoords="offset points", fontsize=7,
                            color="darkorange", fontweight="bold")

        ax.set_title(f"{cfg}  ({kb})", fontweight="bold")
        ax.set_xlabel("inference_ratio  (lower = more cache reuse)")
        if ax is axes[0]:
            ax.set_ylabel("success_rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.6, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    plt.tight_layout()
    out = repo / "exp/verdict_factor_judge/analysis/phase2_layer1_pareto.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}  size: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
