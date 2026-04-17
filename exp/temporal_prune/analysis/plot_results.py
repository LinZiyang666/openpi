"""Plot temporal prune experiment results.

Three figures are produced:
  1. tp_heatmaps.{png,pdf}   — 2×2 heatmaps (reducer × weight),
     each cell shows window_size vs keep_ratio → success rate.
  2. tp_lines.{png,pdf}      — line charts: success rate vs keep_ratio,
     one line per window_size, faceted by (reducer, weight).
  3. tp_bars.{png,pdf}       — grouped bar chart: reducer on x-axis,
     bars grouped by (window, keep_ratio), faceted by weight.

Reads experiment_state.json produced by run_cache_experiments.py.
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
STATE_FILE = SCRIPT_DIR / "experiment_state.json"

OUTPUT_HEAT_PNG = SCRIPT_DIR / "tp_heatmaps.png"
OUTPUT_HEAT_PDF = SCRIPT_DIR / "tp_heatmaps.pdf"
OUTPUT_LINE_PNG = SCRIPT_DIR / "tp_lines.png"
OUTPUT_LINE_PDF = SCRIPT_DIR / "tp_lines.pdf"
OUTPUT_BAR_PNG = SCRIPT_DIR / "tp_bars.png"
OUTPUT_BAR_PDF = SCRIPT_DIR / "tp_bars.pdf"

REDUCERS = ["mean", "max"]
REDUCER_LABELS = {"mean": "mean_pool", "max": "max_pool"}
WINDOWS = [3, 4, 5, 6]
KEEP_RATIOS = [0.25, 0.5, 0.75]
KR_STRS = {"025": 0.25, "05": 0.5, "075": 0.75}
WEIGHTS = ["wa", "wb"]
WEIGHT_LABELS = {
    "wa": "WA (v0=.1 v1=.1 rs=.8)",
    "wb": "WB (v0=.5 v1=.25 rs=.25)",
}

WINDOW_COLORS = dict(zip(WINDOWS, plt.cm.viridis(np.linspace(0.15, 0.85, len(WINDOWS)))))

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

RUN_ID_PATTERN = re.compile(
    r"tp_run_\d+_(mean|max)_(\d+)w_(025|05|075)kr_(wa|wb)"
)


def load_data():
    """Return {(reducer, window, keep_ratio, weight): success_rate}."""
    with open(STATE_FILE) as f:
        experiments = json.load(f)

    data = {}
    for exp in experiments:
        if exp["status"] != "done":
            continue
        m = RUN_ID_PATTERN.match(exp["run_id"])
        if not m:
            continue
        reducer = m.group(1)
        window = int(m.group(2))
        kr = KR_STRS[m.group(3)]
        weight = m.group(4)
        data[(reducer, window, kr, weight)] = exp["success_rate"]
    return data


# ------------------------------------------------------------------
# Figure 1: Heatmaps (window × keep_ratio), faceted by reducer × weight
# ------------------------------------------------------------------


def plot_heatmaps(data):
    fig, axes = plt.subplots(
        len(REDUCERS), len(WEIGHTS),
        figsize=(10, 8),
        squeeze=False,
    )

    for row, reducer in enumerate(REDUCERS):
        for col, weight in enumerate(WEIGHTS):
            ax = axes[row, col]

            # Build matrix: rows = window (top=small), cols = keep_ratio
            matrix = np.full((len(WINDOWS), len(KEEP_RATIOS)), np.nan)
            for i, w in enumerate(WINDOWS):
                for j, kr in enumerate(KEEP_RATIOS):
                    matrix[i, j] = data.get((reducer, w, kr, weight), np.nan)

            im = ax.imshow(
                matrix, cmap="RdYlGn", vmin=0.2, vmax=0.7,
                aspect="auto",
            )

            # Annotate cells
            for i in range(len(WINDOWS)):
                for j in range(len(KEEP_RATIOS)):
                    val = matrix[i, j]
                    if np.isnan(val):
                        continue
                    text_color = "white" if val < 0.35 or val > 0.6 else "black"
                    ax.text(
                        j, i, f"{val:.0%}",
                        ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color=text_color,
                    )

            ax.set_xticks(range(len(KEEP_RATIOS)))
            ax.set_xticklabels([f"{kr}" for kr in KEEP_RATIOS], fontsize=10)
            ax.set_yticks(range(len(WINDOWS)))
            ax.set_yticklabels([str(w) for w in WINDOWS], fontsize=10)

            ax.set_title(
                f"{REDUCER_LABELS[reducer]}  ·  {WEIGHT_LABELS[weight]}",
                fontsize=11, pad=8,
            )

            if col == 0:
                ax.set_ylabel("window_size", fontsize=11)
            if row == len(REDUCERS) - 1:
                ax.set_xlabel("keep_ratio", fontsize=11)

    fig.suptitle(
        "Temporal Prune: Success Rate Heatmaps\n"
        "(window_size × keep_ratio, faceted by reducer × weight)",
        fontsize=13,
    )

    # Shared colorbar
    cbar = fig.colorbar(
        im, ax=axes, orientation="vertical",
        fraction=0.02, pad=0.04,
        format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"),
    )
    cbar.set_label("Success Rate", fontsize=10)

    fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.08, hspace=0.35, wspace=0.25)
    fig.savefig(OUTPUT_HEAT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_HEAT_PDF, bbox_inches="tight")
    print(f"Saved: {OUTPUT_HEAT_PNG}")
    print(f"Saved: {OUTPUT_HEAT_PDF}")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 2: Line chart (success rate vs keep_ratio), one line per
#           window_size, faceted by (reducer, weight)
# ------------------------------------------------------------------


def plot_lines(data):
    fig, axes = plt.subplots(
        len(REDUCERS), len(WEIGHTS),
        figsize=(12, 8),
        sharey=True, squeeze=False,
    )

    for row, reducer in enumerate(REDUCERS):
        for col, weight in enumerate(WEIGHTS):
            ax = axes[row, col]

            for w in WINDOWS:
                ys = [data.get((reducer, w, kr, weight), np.nan) for kr in KEEP_RATIOS]
                ax.plot(
                    KEEP_RATIOS, ys,
                    color=WINDOW_COLORS[w],
                    marker="o", linewidth=2, markersize=7,
                    label=f"window={w}",
                )
                # Annotate points
                for x, y in zip(KEEP_RATIOS, ys):
                    if np.isnan(y):
                        continue
                    ax.text(
                        x, y + 0.015, f"{y:.0%}",
                        ha="center", va="bottom", fontsize=8,
                        color=WINDOW_COLORS[w],
                    )

            ax.set_xticks(KEEP_RATIOS)
            ax.set_xticklabels([str(kr) for kr in KEEP_RATIOS], fontsize=10)
            ax.set_xlabel("keep_ratio", fontsize=10)
            ax.set_ylim(0, 0.8)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"{REDUCER_LABELS[reducer]}  ·  {WEIGHT_LABELS[weight]}",
                fontsize=11,
            )
            ax.legend(fontsize=9, loc="best", frameon=True)

    for ax in axes[:, 0]:
        ax.set_ylabel("Success Rate", fontsize=11)

    fig.suptitle(
        "Temporal Prune: Success Rate vs Keep Ratio\n"
        "(per window_size, faceted by reducer × weight)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_LINE_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_LINE_PDF, bbox_inches="tight")
    print(f"Saved: {OUTPUT_LINE_PNG}")
    print(f"Saved: {OUTPUT_LINE_PDF}")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 3: Grouped bar chart — all 48 runs, faceted by weight,
#           x-axis = reducer, bars grouped by (window, keep_ratio)
# ------------------------------------------------------------------


def plot_bars(data):
    combos = [(w, kr) for w in WINDOWS for kr in KEEP_RATIOS]  # 12 combos
    n_combos = len(combos)
    n_reducers = len(REDUCERS)
    bar_width = 0.065
    group_width = n_combos * bar_width

    fig, axes = plt.subplots(
        1, len(WEIGHTS), figsize=(16, 6), sharey=True,
    )

    # Color by (window, keep_ratio): hue from window, saturation from keep_ratio
    combo_colors = {}
    for w in WINDOWS:
        base = np.array(WINDOW_COLORS[w])
        for ki, kr in enumerate(KEEP_RATIOS):
            # Lighten for higher keep_ratio
            alpha = 0.5 + 0.5 * (ki / (len(KEEP_RATIOS) - 1))
            combo_colors[(w, kr)] = base * alpha + (1 - alpha) * np.array([1, 1, 1, 1])
            combo_colors[(w, kr)][3] = 1.0  # keep alpha=1

    for ax_idx, weight in enumerate(WEIGHTS):
        ax = axes[ax_idx]
        x_centers = np.arange(n_reducers)

        for c_idx, (w, kr) in enumerate(combos):
            offsets = x_centers - group_width / 2 + (c_idx + 0.5) * bar_width
            values = [data.get((r, w, kr, weight), 0) for r in REDUCERS]
            ax.bar(
                offsets, values,
                width=bar_width,
                color=combo_colors[(w, kr)],
                edgecolor="white", linewidth=0.3,
            )

        ax.set_xticks(x_centers)
        ax.set_xticklabels(
            [REDUCER_LABELS[r] for r in REDUCERS], fontsize=11,
        )
        ax.set_ylim(0, 0.8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(WEIGHT_LABELS[weight], fontsize=12)

    axes[0].set_ylabel("Success Rate", fontsize=11)

    # Legend: one entry per (window, keep_ratio)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=combo_colors[(w, kr)])
        for w, kr in combos
    ]
    legend_labels = [f"w={w} kr={kr}" for w, kr in combos]
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=6, fontsize=8, frameon=False,
        title="(window_size, keep_ratio)",
        title_fontsize=9,
    )

    fig.suptitle(
        "Temporal Prune: All Runs (grouped by window × keep_ratio)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(OUTPUT_BAR_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_BAR_PDF, bbox_inches="tight")
    print(f"Saved: {OUTPUT_BAR_PNG}")
    print(f"Saved: {OUTPUT_BAR_PDF}")
    plt.close(fig)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    data = load_data()
    if not data:
        print("No completed runs found in experiment_state.json")
        raise SystemExit(1)

    print(f"Loaded {len(data)} completed runs.\n")

    # Print summary table
    print("=" * 60)
    print("Top 10 runs by success rate:")
    print("-" * 60)
    ranked = sorted(data.items(), key=lambda x: -x[1])
    for (reducer, window, kr, weight), sr in ranked[:10]:
        print(f"  {REDUCER_LABELS[reducer]:12s}  w={window}  kr={kr:.2f}  "
              f"{weight.upper()}  →  {sr:.0%}")
    print("=" * 60)

    # Print per-dimension averages
    print("\nPer-dimension averages:")
    for dim_name, key_fn, labels in [
        ("reducer", lambda k: k[0], REDUCERS),
        ("window", lambda k: k[1], WINDOWS),
        ("keep_ratio", lambda k: k[2], KEEP_RATIOS),
        ("weight", lambda k: k[3], WEIGHTS),
    ]:
        print(f"\n  {dim_name}:")
        for label in labels:
            vals = [v for k, v in data.items() if key_fn(k) == label]
            if vals:
                print(f"    {str(label):12s}  avg={np.mean(vals):.1%}  "
                      f"min={min(vals):.0%}  max={max(vals):.0%}")
    print()

    plot_heatmaps(data)
    plot_lines(data)
    plot_bars(data)
