"""Step 3 per-cycle-policy trade-off plot.

Success-rate vs inference-ratio scatter, one subplot per cfg.  Each point is
one (tau, n) cell aggregated over its ~150 episodes:

- X  : mean inference_ratio over rollouts in the cell
- Y  : success rate (successes / rollouts) in the cell
- hue: tau  (4 discrete colors)
- marker shape: n  (5 markers)

A Pareto frontier (high Y, low X) is drawn per subplot to highlight the
non-dominated cells.

Usage:
    uv run python -m exp.trajectory_deviation.analysis.plot_step3_tradeoff
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D

STEP3_ROOT = Path("exp/trajectory_deviation/data/step3")
CFGS = ["clip_w7_d4", "spatial16_w8_d4", "max_pool_w3_d5"]
TAUS = [3, 5, 7, 10]
NS = [1, 2, 3, 5, 10]

# Tau → color (sequential: small tau ≈ aggressive → warm; large tau → cool).
TAU_COLORS = {t: cm.plasma(i / (len(TAUS) - 1)) for i, t in enumerate(TAUS)}
# n → marker shape.
N_MARKERS = {1: "o", 2: "s", 3: "^", 5: "D", 10: "P"}


def aggregate_cells(results_path: Path) -> dict[tuple[int, int], tuple[float, float, int]]:
    """Return {(tau, n): (success_rate, mean_inference_ratio, count)}."""
    succ = defaultdict(list)
    inf = defaultdict(list)
    with results_path.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["tau"], r["n"])
            succ[key].append(int(r["success"]))
            inf[key].append(float(r["inference_ratio"]))
    out: dict[tuple[int, int], tuple[float, float, int]] = {}
    for key in succ:
        out[key] = (float(np.mean(succ[key])), float(np.mean(inf[key])), len(succ[key]))
    return out


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points on the (minimize X, maximize Y) Pareto frontier, sorted by X."""
    idxs = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    best_y = -np.inf
    front: list[int] = []
    for i in idxs:
        x, y = points[i]
        if y > best_y:
            front.append(i)
            best_y = y
    return front


def plot_cfg(ax: plt.Axes, cfg: str) -> None:
    cells = aggregate_cells(STEP3_ROOT / cfg / "results.jsonl")
    keys = list(cells.keys())
    points_xy: list[tuple[float, float]] = [(cells[k][1], cells[k][0]) for k in keys]

    for key, (sr, ir, _) in cells.items():
        tau, n = key
        ax.scatter(
            ir, sr,
            s=85, color=TAU_COLORS[tau], marker=N_MARKERS[n],
            edgecolor="black", linewidth=0.6, alpha=0.95, zorder=3,
        )

    # Pareto front (minimize inference, maximize success).
    front = pareto_frontier(points_xy)
    fx = [points_xy[i][0] for i in front]
    fy = [points_xy[i][1] for i in front]
    ax.plot(fx, fy, linestyle="--", color="#555555", linewidth=1.0, zorder=2, label="Pareto front")

    ax.set_xlim(-0.02, 0.68)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean inference ratio")
    ax.set_ylabel("success rate")
    ax.set_title(cfg, fontsize=11)
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
    pareto_handle = [Line2D([0], [0], linestyle="--", color="#555555", label="Pareto front")]
    fig.legend(
        handles=tau_handles + n_handles + pareto_handle,
        loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=len(TAUS) + len(NS) + 1, frameon=False, fontsize=9,
    )


def main() -> None:
    fig, axes = plt.subplots(1, len(CFGS), figsize=(15, 5.2), sharey=True)
    for ax, cfg in zip(axes, CFGS):
        plot_cfg(ax, cfg)

    fig.suptitle(
        "Step 3 per-cycle policy — success vs inference trade-off",
        fontsize=13, y=1.00,
    )
    make_legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    out_dir = STEP3_ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step3_tradeoff.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
