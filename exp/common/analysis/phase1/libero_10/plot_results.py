"""Plot phase1 libero_10 experiment results as a single grouped bar chart.

The libero_10 subexperiment is spread across three batches under
``exp/common/data/phase1/libero_10/batch{1,2,3}/``:
  * batch1 + batch2 — 5 key_builders x 8 weight allocations (w1..w8), the
    vision / robot_state sweep reused from libero_spatial.
  * batch3 — 5 key_builders x 4 weight allocations (p1..p4), an additional
    prompt_emb allocation sweep.

Both axes collapse to the same ``(key_builder, weight_id) -> success_rate``
shape, so they are rendered as a single 12-bar-per-kb grouped chart.
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp.common.analysis.plot_common import (
    KEY_BUILDER_LABELS,
    KEY_BUILDER_ORDER,
    WEIGHT_IDS,
    WEIGHT_LABELS,
)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parents[2]  # exp/common/
DATA_DIR = COMMON_DIR / "data" / "phase1" / "libero_10"
BATCH_DIRS = [DATA_DIR / "batch1", DATA_DIR / "batch2", DATA_DIR / "batch3"]

OUTPUT_PNG = SCRIPT_DIR / "phase1_libero_10_results.png"
OUTPUT_PDF = SCRIPT_DIR / "phase1_libero_10_results.pdf"

# prompt_emb allocation sweep; weight ids and labels mirror
# logs/phase1_libero_10_run_commands.log.md §5.3.
PROMPT_WEIGHT_IDS = ["p1", "p2", "p3", "p4"]
PROMPT_WEIGHT_LABELS = {
    "p1": "v0=.15 v1=.15 pe=.10 rs=.60",
    "p2": "v0=.50 pe=.50",
    "p3": "v0=.33 pe=.34 rs=.33",
    "p4": "v0=.15 v1=.15 pe=.20 rs=.50",
}

ALL_WEIGHT_IDS = list(WEIGHT_IDS) + PROMPT_WEIGHT_IDS
ALL_WEIGHT_LABELS = {**WEIGHT_LABELS, **PROMPT_WEIGHT_LABELS}

# 12-color palette: tab20 gives enough distinct hues for w1..w8 + p1..p4.
ALL_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))[: len(ALL_WEIGHT_IDS)]

RUN_ID_PATTERN = re.compile(r"phase1_run_\d+_(.+)_rrf_([wp]\d+)")


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def load_all_batches():
    """Return {(kb, wid): success_rate} merged from every batch state file."""
    data = {}
    for batch_dir in BATCH_DIRS:
        state = batch_dir / "experiment_state.json"
        with open(state) as f:
            experiments = json.load(f)
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
    n_w = len(ALL_WEIGHT_IDS)
    bar_width = 0.065
    group_width = n_w * bar_width

    fig, ax = plt.subplots(figsize=(16, 6))

    x_centers = np.arange(n_kb)

    for j, wid in enumerate(ALL_WEIGHT_IDS):
        offsets = x_centers - group_width / 2 + (j + 0.5) * bar_width
        values = [data.get((kb, wid), 0) for kb in KEY_BUILDER_ORDER]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            color=ALL_COLORS[j],
            label=wid,
            edgecolor="white",
            linewidth=0.5,
        )
        # A zero-height bar is invisible; mark it with an "x" in the bar's
        # own color so the reader can still see which (kb, wid) cell the
        # zero came from.
        zero_xs = [off for off, v in zip(offsets, values) if v == 0]
        if zero_xs:
            ax.scatter(
                zero_xs,
                [0.02] * len(zero_xs),
                marker="x",
                color=ALL_COLORS[j],
                s=36,
                linewidths=1.3,
                zorder=3,
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [KEY_BUILDER_LABELS[kb] for kb in KEY_BUILDER_ORDER], fontsize=10
    )
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Phase 1 (libero_10): CP1 Cache Experiment Results", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    # Legend pairs the bar color with its weight allocation. w* covers the
    # vision/robot_state sweep; p* covers the prompt_emb sweep (pe=prompt_emb).
    legend_lines = [f"{wid}: {ALL_WEIGHT_LABELS[wid]}" for wid in ALL_WEIGHT_IDS]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=ALL_COLORS[j]) for j in range(n_w)
    ]
    ax.legend(
        legend_handles,
        legend_lines,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        fontsize=9,
        frameon=False,
        title=(
            "Weight Allocation "
            "(v0=vision_0, v1=vision_1, pe=prompt_emb, rs=robot_state)"
        ),
        title_fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])

    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    plt.close(fig)


if __name__ == "__main__":
    plot(load_all_batches())
