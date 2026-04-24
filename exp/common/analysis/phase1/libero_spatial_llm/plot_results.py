"""Plot phase1 libero_spatial_llm experiment results.

The libero_spatial_llm subexperiment adds an LLM-layer extraction dimension
on top of the libero_spatial sweep: every (key_builder, weight) cell from the
phase1 baseline is re-run for each of the first four PaliGemma language-model
layers (``extract_layer in {0,1,2,3}``).

Data is spread across six batches under
``exp/common/data/phase1/libero_spatial_llm/batch{1..6}/``:
  * batch1..batch4 — key_builders ``{a, b1, b2, c}`` x layers {0..3} x
    weights ``w1..w8`` (the vision / robot_state sweep from libero_spatial).
  * batch5..batch6 — same key_builders x layers, with weights ``p1..p4``
    (the prompt_emb sweep from libero_10).
  * batch6 additionally contains a ``prefix_mean_pool`` baseline (``e``)
    with a single ``const`` weight (``vision_0=1.0, robot_state=1.0``).

We produce a 2x2 figure, one subplot per extract layer. Each subplot lays
the four multi-weight key_builders side by side (12 bars: ``w1..w8, p1..p4``)
and appends the single-bar ``e (const)`` baseline as a fifth group.
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp.common.analysis.plot_common import WEIGHT_IDS, WEIGHT_LABELS

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parents[2]  # exp/common/
DATA_DIR = COMMON_DIR / "data" / "phase1" / "libero_spatial_llm"
BATCH_DIRS = [DATA_DIR / f"batch{i}" for i in range(1, 7)]

OUTPUT_PNG = SCRIPT_DIR / "phase1_libero_spatial_llm_results.png"
OUTPUT_PDF = SCRIPT_DIR / "phase1_libero_spatial_llm_results.pdf"

# LLM-layer subexp uses the libero_spatial key-builder letters plus `e`
# (prefix_mean_pool baseline, batch6 only, const weight only).
LLM_KB_ORDER = ["a", "b1", "b2", "c"]
# Labels are the underlying reducer names (see generate_phase1_llm_yamls.py),
# not the legacy `cp1_*` aliases — avoids confusion with the CP1 checkpoint.
LLM_KB_LABELS = {
    "a":  "per_modality_mean_pool",
    "b1": "per_modality_spatial_pool_16",
    "b2": "per_modality_spatial_pool_4",
    "c":  "per_modality_max_pool",
}
BASELINE_KB = "e"
BASELINE_KB_LABEL = "prefix_mean_pool\n(const)"
BASELINE_WEIGHT = "const"
BASELINE_WEIGHT_LABEL = "v0=1.0 rs=1.0"

LAYERS = [0, 1, 2, 3]

# prompt_emb allocation sweep; must match logs/phase1_libero_10_run_commands
# §5.3 and logs/phase1_libero_spatial_llm_run_commands §5.
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
BASELINE_COLOR = (0.35, 0.35, 0.35, 1.0)

# `kb` may be a/b1/b2/c/e; `layer` is a single digit; `weight` is
# w[1-8] / p[1-4] / const.
RUN_ID_PATTERN = re.compile(r"phase1_run_\d+_([a-z0-9]+)_l(\d+)_rrf_(\w+)")


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def load_all_batches():
    """Return {(kb, layer, weight): success_rate} merged across batches."""
    data = {}
    for batch_dir in BATCH_DIRS:
        state = batch_dir / "experiment_state.json"
        if not state.exists():
            continue
        with open(state) as f:
            experiments = json.load(f)
        for exp in experiments:
            if exp.get("status") != "done":
                continue
            m = RUN_ID_PATTERN.match(exp["run_id"])
            if not m:
                continue
            kb = m.group(1)
            layer = int(m.group(2))
            weight = m.group(3)
            data[(kb, layer, weight)] = exp["success_rate"]
    return data


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


def _plot_layer(ax, data, layer):
    """Render one layer's grouped bar chart into ``ax``."""
    n_kb = len(LLM_KB_ORDER)
    n_w = len(ALL_WEIGHT_IDS)
    bar_width = 0.07
    group_width = n_w * bar_width

    x_centers = np.arange(n_kb, dtype=float)

    for j, wid in enumerate(ALL_WEIGHT_IDS):
        offsets = x_centers - group_width / 2 + (j + 0.5) * bar_width
        values = [data.get((kb, layer, wid), 0) for kb in LLM_KB_ORDER]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            color=ALL_COLORS[j],
            label=wid,
            edgecolor="white",
            linewidth=0.4,
        )
        # A zero-height bar is invisible; mark it so readers can still tell
        # which (kb, wid) cell produced the zero.
        zero_xs = [off for off, v in zip(offsets, values) if v == 0]
        if zero_xs:
            ax.scatter(
                zero_xs,
                [0.02] * len(zero_xs),
                marker="x",
                color=ALL_COLORS[j],
                s=24,
                linewidths=1.1,
                zorder=3,
            )

    # Baseline: prefix_mean_pool (e) with const weight, one bar to the right.
    baseline_x = n_kb + 0.3
    baseline_val = data.get((BASELINE_KB, layer, BASELINE_WEIGHT), 0)
    ax.bar(
        [baseline_x],
        [baseline_val],
        width=bar_width * 4,
        color=BASELINE_COLOR,
        edgecolor="white",
        linewidth=0.4,
    )
    if baseline_val == 0:
        ax.scatter(
            [baseline_x],
            [0.02],
            marker="x",
            color=BASELINE_COLOR,
            s=36,
            linewidths=1.3,
            zorder=3,
        )

    xtick_positions = list(x_centers) + [baseline_x]
    xtick_labels = [LLM_KB_LABELS[kb] for kb in LLM_KB_ORDER] + [BASELINE_KB_LABEL]

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Success Rate", fontsize=10)
    ax.set_title(f"extract_layer = {layer}", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    # Horizontal reference line at the tallest bar of this layer, across
    # both the multi-weight kbs and the prefix_mean_pool baseline.
    all_cells = [
        (kb, wid, data.get((kb, layer, wid), 0))
        for kb in LLM_KB_ORDER
        for wid in ALL_WEIGHT_IDS
    ]
    all_cells.append(
        (BASELINE_KB, BASELINE_WEIGHT, data.get(
            (BASELINE_KB, layer, BASELINE_WEIGHT), 0)
        )
    )
    max_kb, max_wid, max_val = max(all_cells, key=lambda c: c[2])
    if max_val > 0:
        ax.axhline(
            max_val,
            color="crimson",
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            zorder=4,
        )
        ax.text(
            0.99,
            max_val + 0.015,
            f"max = {max_val:.0%}  ({max_kb}/{max_wid})",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="crimson",
        )


def plot(data):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
    for ax, layer in zip(axes.flat, LAYERS):
        _plot_layer(ax, data, layer)

    fig.suptitle(
        "Phase 1 (libero_spatial_llm): CP1 LLM-Layer Extraction Sweep",
        fontsize=14,
        y=0.995,
    )

    # Shared legend at bottom: 12 weight colors + baseline swatch.
    legend_lines = [f"{wid}: {ALL_WEIGHT_LABELS[wid]}" for wid in ALL_WEIGHT_IDS]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=ALL_COLORS[j])
        for j in range(len(ALL_WEIGHT_IDS))
    ]
    legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=BASELINE_COLOR))
    legend_lines.append(f"{BASELINE_WEIGHT} ({BASELINE_KB}): {BASELINE_WEIGHT_LABEL}")

    fig.legend(
        legend_handles,
        legend_lines,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=9,
        frameon=False,
        title=(
            "Weight Allocation "
            "(v0=vision_0, v1=vision_1, pe=prompt_emb, rs=robot_state)"
        ),
        title_fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.10, 1, 0.97])

    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    plt.close(fig)


def print_summary(data):
    """Stdout summary: best (kb, weight) per layer and best kb across layers."""
    print()
    print("=" * 72)
    print("Summary: best cell per extract_layer")
    print("=" * 72)
    for layer in LAYERS:
        cells = [
            ((kb, wid), data[(kb, layer, wid)])
            for kb in LLM_KB_ORDER
            for wid in ALL_WEIGHT_IDS
            if (kb, layer, wid) in data
        ]
        if not cells:
            print(f"  layer {layer}: no data")
            continue
        cells.sort(key=lambda kv: kv[1], reverse=True)
        top = cells[:3]
        joined = ", ".join(
            f"{kb}/{wid}={v:.0%}" for (kb, wid), v in top
        )
        print(f"  layer {layer} top-3: {joined}")

    print()
    print("=" * 72)
    print("Baseline (prefix_mean_pool, const) by layer")
    print("=" * 72)
    for layer in LAYERS:
        val = data.get((BASELINE_KB, layer, BASELINE_WEIGHT))
        if val is None:
            print(f"  layer {layer}: missing")
        else:
            print(f"  layer {layer}: {val:.0%}")


if __name__ == "__main__":
    data = load_all_batches()
    plot(data)
    print_summary(data)
