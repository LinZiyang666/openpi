"""Step 3 trade-off plot with **total** success rate on Y.

The Y axis combines Step 1a baseline (pure-cache success on the full
population) with Step 3 recovery on the failure subset:

    total_sr = baseline_sr + (1 - baseline_sr) * step3_sr

The X axis is unchanged: mean ``inference_ratio`` from Step 3 on the failure
subset (i.e. extra inference cost per Step-3 episode).  Combined with the
baseline, this makes each point a (cost-over-failures, total success rate)
trade-off; the Pareto front now reflects the full-population picture.

Baselines are read from Step 1a per-cfg result dumps:
    exp/trajectory_deviation/data/cache_eval_results_<short>.json

Usage:
    uv run python -m exp.trajectory_deviation.analysis.plot_step3_tradeoff_total
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D

STEP3_ROOT = Path("exp/trajectory_deviation/data/step3")
CFGS = ["clip_w7_d4", "spatial16_w8_d4", "max_pool_w3_d5"]
TAUS = [3, 5, 7, 10]
NS = [1, 2, 3, 5, 10]

# Step 1a pure-cache success rates, authoritative numbers from the run report:
#   clip_w7_d4      337/500 = 0.674
#   spatial16_w8_d4 346/500 = 0.692
#   max_pool_w3_d5  348/500 = 0.696
BASELINES = {
    "clip_w7_d4":      337 / 500,
    "spatial16_w8_d4": 346 / 500,
    "max_pool_w3_d5":  348 / 500,
}

TAU_COLORS = {t: cm.plasma(i / (len(TAUS) - 1)) for i, t in enumerate(TAUS)}
N_MARKERS = {1: "o", 2: "s", 3: "^", 5: "D", 10: "P"}


def load_summary() -> dict[str, list[dict]]:
    rows_by_cfg: dict[str, list[dict]] = {c: [] for c in CFGS}
    with (STEP3_ROOT / "summary.csv").open() as f:
        for r in csv.DictReader(f):
            rows_by_cfg[r["cfg"]].append(r)
    return rows_by_cfg


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Indices on (min X, max Y) Pareto front, sorted by X."""
    idxs = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    best_y = -np.inf
    front: list[int] = []
    for i in idxs:
        x, y = points[i]
        if y > best_y:
            front.append(i)
            best_y = y
    return front


def plot_cfg(ax: plt.Axes, cfg: str, rows: list[dict], baseline: float) -> None:
    points_xy: list[tuple[float, float]] = []
    scatter_args: list[tuple[float, float, int, int]] = []  # (x, y, tau, n)

    for r in rows:
        tau = int(r["tau"]); n = int(r["n"])
        step3_sr = float(r["success_rate"])
        inf_ratio = float(r["mean_inference_ratio"])
        total_sr = baseline + (1.0 - baseline) * step3_sr
        points_xy.append((inf_ratio, total_sr))
        scatter_args.append((inf_ratio, total_sr, tau, n))

    for x, y, tau, n in scatter_args:
        ax.scatter(
            x, y, s=85, color=TAU_COLORS[tau], marker=N_MARKERS[n],
            edgecolor="black", linewidth=0.6, alpha=0.95, zorder=3,
        )

    # Pareto front (min X, max Y).
    front = pareto_frontier(points_xy)
    fx = [points_xy[i][0] for i in front]
    fy = [points_xy[i][1] for i in front]
    ax.plot(fx, fy, linestyle="--", color="#555555", linewidth=1.0, zorder=2)

    # Baseline reference line (pure cache, no Step 3).
    ax.axhline(baseline, linestyle=":", color="#b23a48", linewidth=1.0, zorder=1)
    ax.text(
        0.66, baseline + 0.008, f"pure cache = {baseline*100:.1f}%",
        ha="right", va="bottom", fontsize=8, color="#b23a48",
    )

    ax.set_xlim(-0.02, 0.68)
    ax.set_ylim(0.60, 1.0)
    ax.set_xlabel("mean inference ratio (failure subset)")
    ax.set_ylabel("total success rate (full population)")
    ax.set_title(f"{cfg}  (baseline {baseline*100:.1f}%)", fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def make_legend(fig: plt.Figure) -> None:
    tau_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=TAU_COLORS[t], markeredgecolor="black", label=f"τ = {t}")
        for t in TAUS
    ]
    n_handles = [
        Line2D([0], [0], marker=N_MARKERS[n], linestyle="", markersize=8,
               markerfacecolor="lightgray", markeredgecolor="black", label=f"n = {n}")
        for n in NS
    ]
    extras = [
        Line2D([0], [0], linestyle="--", color="#555555", label="Pareto front"),
        Line2D([0], [0], linestyle=":",  color="#b23a48", label="pure-cache baseline"),
    ]
    fig.legend(
        handles=tau_handles + n_handles + extras,
        loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=len(TAUS) + len(NS) + len(extras), frameon=False, fontsize=9,
    )


def main() -> None:
    rows_by_cfg = load_summary()
    baselines = BASELINES
    for c, b in baselines.items():
        print(f"[{c}] pure-cache baseline = {b*100:.2f}%")

    fig, axes = plt.subplots(1, len(CFGS), figsize=(15, 5.4), sharey=True)
    for ax, cfg in zip(axes, CFGS):
        plot_cfg(ax, cfg, rows_by_cfg[cfg], baselines[cfg])

    fig.suptitle(
        "Step 3 per-cycle policy — total success rate vs inference cost",
        fontsize=13, y=1.00,
    )
    make_legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    out_dir = STEP3_ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step3_tradeoff_total.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
