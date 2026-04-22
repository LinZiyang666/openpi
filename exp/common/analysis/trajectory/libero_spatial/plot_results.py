"""Plot trajectory experiment results.

Two figures are produced:
  1. trajectory_results.{png,pdf}        — main line chart, success rate vs.
     trajectory depth, one line per (key_builder, weight) combination.
  2. trajectory_results_facets.{png,pdf} — small multiples (one bar chart per
     trajectory depth), each in the same style as phase1/plot_results.py.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp.common.analysis.plot_common import (
    BAR_COLORS,
    KB_COLORS,
    KEY_BUILDER_LABELS,
    KEY_BUILDER_ORDER,
    ROLE_LABELS,
    ROLE_MARKERS,
    WEIGHT_LABELS,
)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parents[2]  # exp/common/
STATE_FILE = COMMON_DIR / "data" / "trajectory" / "libero_spatial" / "experiment_state.json"
PHASE1_STATE_FILE = COMMON_DIR / "data" / "phase1" / "libero_spatial" / "experiment_state.json"

OUTPUT_LINE_PNG = SCRIPT_DIR / "trajectory_results.png"
OUTPUT_LINE_PDF = SCRIPT_DIR / "trajectory_results.pdf"
OUTPUT_FACET_PNG = SCRIPT_DIR / "trajectory_results_facets.png"
OUTPUT_FACET_PDF = SCRIPT_DIR / "trajectory_results_facets.pdf"

DEPTHS = [1, 3, 4, 5, 6]  # 1 = phase1 baseline (single-step, no trajectory)

RUN_ID_PATTERN = re.compile(r"traj_d(\d+)_\d+_(.+)_rrf_(w\d+)")
PHASE1_RUN_ID_PATTERN = re.compile(r"phase1_run_\d+_(.+)_rrf_(w\d+)")


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def load_data():
    """Return {(depth, kb, wid): success_rate}.

    Trajectory runs supply depths >= 3. Phase1 runs (single-step lookup,
    trajectory_depth=1) are merged in as the depth=1 baseline, but only for
    (kb, wid) combinations that also appear in the trajectory experiment.
    """
    with open(STATE_FILE) as f:
        experiments = json.load(f)

    data = {}
    for exp in experiments:
        if exp["status"] != "done":
            continue
        m = RUN_ID_PATTERN.match(exp["run_id"])
        if not m:
            continue
        depth = int(m.group(1))
        kb = m.group(2)
        wid = m.group(3)
        data[(depth, kb, wid)] = exp["success_rate"]

    traj_pairs = {(k, w) for (_, k, w) in data}

    if PHASE1_STATE_FILE.exists():
        with open(PHASE1_STATE_FILE) as f:
            phase1 = json.load(f)
        for exp in phase1:
            if exp.get("status") != "done":
                continue
            m = PHASE1_RUN_ID_PATTERN.match(exp["run_id"])
            if not m:
                continue
            kb, wid = m.group(1), m.group(2)
            if (kb, wid) in traj_pairs:
                data[(1, kb, wid)] = exp["success_rate"]
    return data


def load_phase1_roles():
    """Per kb, classify each weight as 'top1' / 'top2' / '2nd_worst' / None
    based on phase1 ranking. Used to annotate the line plot.
    """
    roles = defaultdict(dict)
    if not PHASE1_STATE_FILE.exists():
        return roles
    with open(PHASE1_STATE_FILE) as f:
        phase1 = json.load(f)
    by_kb = defaultdict(dict)
    for exp in phase1:
        if exp.get("status") != "done":
            continue
        m = PHASE1_RUN_ID_PATTERN.match(exp["run_id"])
        if not m:
            continue
        kb, wid = m.group(1), m.group(2)
        by_kb[kb][wid] = exp["success_rate"]
    for kb, scores in by_kb.items():
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        if len(ranked) >= 1:
            roles[kb][ranked[0][0]] = "top1"
        if len(ranked) >= 2:
            roles[kb][ranked[1][0]] = "top2"
        if len(ranked) >= 2:
            asc = sorted(scores.items(), key=lambda x: x[1])
            roles[kb][asc[1][0]] = "2nd_worst"
    return roles


def kb_weight_map(data):
    """{kb: sorted list of weight ids present for that kb}."""
    return {
        kb: sorted(
            {w for (_, k, w) in data if k == kb},
            key=lambda x: int(x[1:]),
        )
        for kb in KEY_BUILDER_ORDER
    }


def all_weights_sorted(data):
    return sorted({w for (_, _, w) in data}, key=lambda x: int(x[1:]))


# ------------------------------------------------------------------
# Main figure: line chart (success rate vs. depth)
# ------------------------------------------------------------------


def plot_lines(data):
    kb_weights = kb_weight_map(data)
    used_weights = all_weights_sorted(data)
    weight_color = {w: BAR_COLORS[i] for i, w in enumerate(used_weights)}
    roles = load_phase1_roles()

    n_kb = len(KEY_BUILDER_ORDER)
    n_cols = 3
    n_rows = (n_kb + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 4.2 * n_rows), sharey=True
    )
    axes_flat = axes.flatten()

    for ax, kb in zip(axes_flat, KEY_BUILDER_ORDER):
        for wid in kb_weights[kb]:
            ys = [data.get((d, kb, wid), np.nan) for d in DEPTHS]
            role = roles.get(kb, {}).get(wid)
            marker = ROLE_MARKERS.get(role, "x")
            role_tag = ROLE_LABELS.get(role, "?")
            ax.plot(
                DEPTHS,
                ys,
                color=weight_color[wid],
                marker=marker,
                linestyle="-",
                linewidth=1.8,
                markersize=10 if marker == "*" else 7,
                label=f"{wid}  ({role_tag})",
            )
            for x, y in zip(DEPTHS, ys):
                if np.isnan(y):
                    continue
                ax.text(
                    x,
                    y + 0.015,
                    f"{y:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=weight_color[wid],
                )

        ax.set_xticks(DEPTHS)
        ax.set_xticklabels(
            [f"{d}\n(phase1)" if d == 1 else str(d) for d in DEPTHS],
            fontsize=8,
        )
        ax.set_xlabel("Trajectory Depth", fontsize=10)
        ax.set_title(KEY_BUILDER_LABELS[kb], fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0%}")
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", frameon=True)

    for ax in axes[:, 0]:
        ax.set_ylabel("Success Rate", fontsize=10)

    # Hide unused subplots.
    for ax in axes_flat[n_kb:]:
        ax.axis("off")

    fig.suptitle(
        "Trajectory: Success Rate vs. Depth (per key_builder)",
        fontsize=14,
    )

    wkey = "   ".join(f"{w}: {WEIGHT_LABELS[w]}" for w in used_weights)
    fig.text(
        0.5,
        0.01,
        f"Weight Allocation (v0=vision_0, v1=vision_1, rs=robot_state):  {wkey}",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUTPUT_LINE_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_LINE_PDF, bbox_inches="tight")
    print(f"Saved: {OUTPUT_LINE_PNG}")
    print(f"Saved: {OUTPUT_LINE_PDF}")
    plt.close(fig)


# ------------------------------------------------------------------
# Supplementary figure: small multiples (one bar chart per depth)
# ------------------------------------------------------------------


def plot_facets(data):
    kb_weights = kb_weight_map(data)
    used_weights = all_weights_sorted(data)
    color_map = {w: BAR_COLORS[i] for i, w in enumerate(used_weights)}

    n_kb = len(KEY_BUILDER_ORDER)
    max_w = max(len(v) for v in kb_weights.values())
    bar_width = 0.22
    group_width = max_w * bar_width

    n_depths = len(DEPTHS)
    n_cols = 3
    n_rows = (n_depths + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(
        n_rows, n_cols, figsize=(16, 4.5 * n_rows), sharey=True
    )
    axes = axes_grid.flatten()

    for ax, depth in zip(axes, DEPTHS):
        x_centers = np.arange(n_kb)
        for kb_idx, kb in enumerate(KEY_BUILDER_ORDER):
            weights = kb_weights[kb]
            n_w = len(weights)
            for w_idx, wid in enumerate(weights):
                offset = (
                    x_centers[kb_idx]
                    - group_width / 2
                    + (w_idx + 0.5) * bar_width
                )
                sr = data.get((depth, kb, wid))
                if sr is None:
                    continue
                ax.bar(
                    offset,
                    sr,
                    width=bar_width,
                    color=color_map[wid],
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax.text(
                    offset,
                    sr + 0.01,
                    f"{sr:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_xticks(x_centers)
        ax.set_xticklabels(
            [KEY_BUILDER_LABELS[kb] for kb in KEY_BUILDER_ORDER],
            fontsize=9,
        )
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", alpha=0.3)
        title = f"depth = {depth}" + ("  (phase1 baseline)" if depth == 1 else "")
        ax.set_title(title, fontsize=12)

    for ax in axes_grid[:, 0]:
        ax.set_ylabel("Success Rate", fontsize=11)

    for ax in axes[n_depths:]:
        ax.axis("off")

    fig.suptitle(
        "Trajectory: CP1 Cache Experiment Results (faceted by depth)",
        fontsize=14,
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color_map[w]) for w in used_weights
    ]
    legend_labels = [f"{w}: {WEIGHT_LABELS[w]}" for w in used_weights]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(used_weights),
        fontsize=9,
        frameon=False,
        title="Weight Allocation (v0=vision_0, v1=vision_1, rs=robot_state)",
        title_fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(OUTPUT_FACET_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_FACET_PDF, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FACET_PNG}")
    print(f"Saved: {OUTPUT_FACET_PDF}")
    plt.close(fig)


if __name__ == "__main__":
    data = load_data()
    plot_lines(data)
    plot_facets(data)
