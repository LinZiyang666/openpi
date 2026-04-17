"""Plot phase1 experiment results as grouped bar chart.

Reads experiment_state.json, groups runs by key_builder and weight config,
and produces a bar chart with key_builder on x-axis and success rate on y-axis.
Each key_builder group has 8 bars (w1-w8) in different colors.
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
OUTPUT_PNG = SCRIPT_DIR / "phase1_results.png"
OUTPUT_PDF = SCRIPT_DIR / "phase1_results.pdf"

KEY_BUILDER_ORDER = ["a", "b1", "b2", "c", "d"]
KEY_BUILDER_LABELS = {
    "a": "cp1_mean_pool",
    "b1": "cp1_spatial_pool_16",
    "b2": "cp1_spatial_pool_64",
    "c": "cp1_max_pool",
    "d": "clip",
}
WEIGHT_IDS = [f"w{i}" for i in range(1, 9)]

# Weight descriptions: (v0, v1, prompt_emb, robot_state)
# vision_2 is always disabled
WEIGHT_LABELS = {
    "w1": "v0=1.0",
    "w2": "rs=1.0",
    "w3": "v0=.5 rs=.5",
    "w4": "v0=.25 rs=.75",
    "w5": "v0=.25 v1=.25 rs=.5",
    "w6": "v0=.15 v1=.1 rs=.75",
    "w7": "v0=.1 v1=.1 rs=.8",
    "w8": "v0=.5 v1=.25 rs=.25",
}

COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:8]

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

RUN_ID_PATTERN = re.compile(r"phase1_run_\d+_(.+)_rrf_(w\d+)")


def load_data():
    """Return {(kb, wid): success_rate} from experiment_state.json."""
    with open(STATE_FILE) as f:
        experiments = json.load(f)

    data = {}
    for exp in experiments:
        if exp["status"] != "done":
            continue
        m = RUN_ID_PATTERN.match(exp["run_id"])
        if not m:
            continue
        kb, wid = m.group(1), m.group(2)
        data[(kb, wid)] = exp["success_rate"]
    return data


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


def plot(data):
    n_kb = len(KEY_BUILDER_ORDER)
    n_w = len(WEIGHT_IDS)
    bar_width = 0.09
    group_width = n_w * bar_width

    fig, ax = plt.subplots(figsize=(14, 6))

    x_centers = np.arange(n_kb)

    for j, wid in enumerate(WEIGHT_IDS):
        offsets = x_centers - group_width / 2 + (j + 0.5) * bar_width
        values = [data.get((kb, wid), 0) for kb in KEY_BUILDER_ORDER]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            color=COLORS[j],
            label=wid,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [KEY_BUILDER_LABELS[kb] for kb in KEY_BUILDER_ORDER], fontsize=10
    )
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Phase 1: CP1 Cache Experiment Results", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------
    # Legend: weight allocation table below the chart
    # ------------------------------------------------------------------
    legend_lines = []
    for wid in WEIGHT_IDS:
        legend_lines.append(f"{wid}: {WEIGHT_LABELS[wid]}")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[j]) for j in range(n_w)
    ]
    ax.legend(
        legend_handles,
        legend_lines,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        fontsize=9,
        frameon=False,
        title="Weight Allocation (v0=vision_0, v1=vision_1, rs=robot_state)",
        title_fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    plt.close(fig)


if __name__ == "__main__":
    data = load_data()
    plot(data)
