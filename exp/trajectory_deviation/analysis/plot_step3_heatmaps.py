"""Step 3 per-cycle-policy heatmap plots.

For each cfg, aggregates `results.jsonl` over episodes and produces one 3D
bar chart over the (tau, n) grid:

- X axis: tau  (3, 5, 7, 10)
- Y axis: n    (1, 2, 3, 5, 10)
- Z (bar height): rollout success rate on that (tau, n) cell
- Bar color: mean inference_ratio on that (tau, n) cell, mapped with the
  bwr colormap clamped to [0, 1] — 0 = pure blue, 1 = pure red.

Outputs one PNG per cfg to ``exp/trajectory_deviation/analysis/step3_plots/``.

Usage:
    uv run python -m exp.trajectory_deviation.analysis.plot_step3_heatmaps
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (register 3d projection)

STEP3_ROOT = Path("exp/trajectory_deviation/data/step3")
CFGS = ["clip_w7_d4", "spatial16_w8_d4", "max_pool_w3_d5"]
TAUS = [3, 5, 7, 10]
NS = [1, 2, 3, 5, 10]


def aggregate(results_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (success_rate, mean_inf_ratio, count) arrays shaped [len(TAUS), len(NS)]."""
    succ = defaultdict(list)
    inf = defaultdict(list)
    with results_path.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["tau"], r["n"])
            succ[key].append(int(r["success"]))
            inf[key].append(float(r["inference_ratio"]))
    sr = np.zeros((len(TAUS), len(NS)))
    ir = np.zeros((len(TAUS), len(NS)))
    cnt = np.zeros((len(TAUS), len(NS)), dtype=int)
    for i, tau in enumerate(TAUS):
        for j, n in enumerate(NS):
            key = (tau, n)
            if succ[key]:
                sr[i, j] = float(np.mean(succ[key]))
                ir[i, j] = float(np.mean(inf[key]))
                cnt[i, j] = len(succ[key])
    return sr, ir, cnt


CMAP = cm.plasma  # brighter than viridis at the low end; keeps success-rate labels readable
COLOR_VMIN = 0.0
COLOR_VMAX = 0.65  # fit actual per-cell-mean range; raises contrast at the top end


def plot_cfg(cfg: str, out_dir: Path) -> Path:
    results_path = STEP3_ROOT / cfg / "results.jsonl"
    sr, ir, _ = aggregate(results_path)

    fig = plt.figure(figsize=(9, 6.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    # Transparent panes so bars read clearly.
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((1, 1, 1, 0.0))
        pane.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
    ax.grid(True)

    xs, ys = np.meshgrid(np.arange(len(TAUS)), np.arange(len(NS)), indexing="ij")
    xs_f = xs.ravel().astype(float)
    ys_f = ys.ravel().astype(float)
    zs0 = np.zeros_like(xs_f)
    dx = dy = 0.72
    dz = sr.ravel()
    norm = Normalize(vmin=COLOR_VMIN, vmax=COLOR_VMAX)
    colors = CMAP(norm(ir.ravel()))

    ax.bar3d(
        xs_f - dx / 2, ys_f - dy / 2, zs0, dx, dy, dz,
        color=colors, shade=True, edgecolor=(0.25, 0.25, 0.25, 0.35), linewidth=0.3,
    )

    ax.set_xticks(np.arange(len(TAUS)))
    ax.set_xticklabels([str(t) for t in TAUS])
    ax.set_yticks(np.arange(len(NS)))
    ax.set_yticklabels([str(n) for n in NS])
    ax.set_xlabel(r"$\tau$ (period threshold)", labelpad=6)
    ax.set_ylabel(r"$n$ (burst length)", labelpad=6)
    ax.set_zlabel("success rate", labelpad=4)
    ax.set_zlim(0.0, 1.0)
    ax.set_zticks(np.linspace(0.0, 1.0, 6))
    # Subtle dashed z gridlines help reading bar heights without cluttering.
    ax.zaxis._axinfo["grid"].update({"color": (0.7, 0.7, 0.7, 0.5), "linewidth": 0.5, "linestyle": "--"})
    ax.view_init(elev=22, azim=-55)
    ax.set_title(
        f"Step 3 per-cycle policy — {cfg}\n"
        "bar height = success rate · color = mean inference ratio",
        fontsize=11, pad=10,
    )

    # Bold black numbers with a white stroke so they stay readable over any bar color.
    label_stroke = [pe.withStroke(linewidth=2.2, foreground="white")]
    for i in range(len(TAUS)):
        for j in range(len(NS)):
            ax.text(
                i, j, sr[i, j] + 0.03, f"{sr[i, j]:.2f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="black",
                path_effects=label_stroke,
            )

    sm = cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, aspect=18)
    cbar.set_label("inference ratio", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step3_heatmap_{cfg}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    out_dir = STEP3_ROOT / "plots"
    for cfg in CFGS:
        path = plot_cfg(cfg, out_dir)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
