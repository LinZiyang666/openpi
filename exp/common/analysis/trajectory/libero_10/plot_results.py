"""Plot trajectory libero_10 experiment results.

Mirrors :mod:`exp.common.analysis.trajectory.libero_spatial.plot_results` but
adapts to the libero_10 layout:

* Trajectory runs are split across three batches under
  ``exp/common/data/trajectory/libero_10/batch{1,2,3}/``, one batch per depth
  (4, 5, 6). Per-batch ``experiment_state.json`` is the canonical aggregate
  source; if missing, per-episode ``cache_eval_results.json`` is rolled up to
  per-run success rates.
* The depth-1 baseline is drawn from
  ``exp/common/data/phase1/libero_10/batch{1,2,3}/experiment_state.json`` (only
  the ``w*`` weights — the ``p*`` prompt_emb sweep is not part of trajectory).
* Per key_builder, libero_10 uses **4** weight allocations (top-1 / top-2 /
  top-3 / 2nd-worst from phase1), not the 3 used in libero_spatial.

Two figures are produced:

  1. ``trajectory_results.{png,pdf}``        — per-kb line chart, success rate
     vs. trajectory depth (one line per weight).
  2. ``trajectory_results_facets.{png,pdf}`` — small multiples, one bar chart
     per depth.
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

TRAJ_DATA_DIR = COMMON_DIR / "data" / "trajectory" / "libero_10"
TRAJ_CONFIG_DIR = COMMON_DIR / "config" / "trajectory" / "libero_10"
PHASE1_DATA_DIR = COMMON_DIR / "data" / "phase1" / "libero_10"

BATCHES = ["batch1", "batch2", "batch3"]
BATCH_DEPTHS = {"batch1": 4, "batch2": 5, "batch3": 6}

# 1 = phase1 baseline (single-step lookup); 4/5/6 from trajectory batches.
DEPTHS = [1, 4, 5, 6]

OUTPUT_LINE_PNG = SCRIPT_DIR / "trajectory_results.png"
OUTPUT_LINE_PDF = SCRIPT_DIR / "trajectory_results.pdf"
OUTPUT_FACET_PNG = SCRIPT_DIR / "trajectory_results_facets.png"
OUTPUT_FACET_PDF = SCRIPT_DIR / "trajectory_results_facets.pdf"

TRAJ_RUN_ID_PATTERN = re.compile(r"traj_d(\d+)_\d+_(.+)_rrf_(w\d+)")
PHASE1_RUN_ID_PATTERN = re.compile(r"phase1_run_\d+_(.+)_rrf_(w\d+)")


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def _load_state_runs(state_path: Path):
    """Yield ``(depth, kb, wid, success_rate)`` from one experiment_state.json."""
    if not state_path.exists() or state_path.stat().st_size == 0:
        return
    with open(state_path) as f:
        try:
            experiments = json.load(f)
        except json.JSONDecodeError:
            return
    for exp in experiments:
        if exp.get("status") != "done":
            continue
        m = TRAJ_RUN_ID_PATTERN.match(exp.get("run_id", ""))
        if not m:
            continue
        yield int(m.group(1)), m.group(2), m.group(3), exp["success_rate"]


def _load_episode_runs(episode_path: Path):
    """Aggregate ``cache_eval_results.json`` (per-episode rows) into per-run rates.

    Each row carries ``config_id`` (run_id) and ``success`` (bool). We average
    success per run_id, then map to (depth, kb, wid).
    """
    if not episode_path.exists() or episode_path.stat().st_size == 0:
        return
    with open(episode_path) as f:
        try:
            episodes = json.load(f)
        except json.JSONDecodeError:
            return
    if not episodes:
        return
    by_run = defaultdict(list)
    for row in episodes:
        rid = row.get("config_id")
        if rid:
            by_run[rid].append(bool(row.get("success")))
    for rid, flags in by_run.items():
        m = TRAJ_RUN_ID_PATTERN.match(rid)
        if not m:
            continue
        sr = sum(flags) / len(flags) if flags else 0.0
        yield int(m.group(1)), m.group(2), m.group(3), sr


def load_trajectory_data():
    """Return ``{(depth, kb, wid): success_rate}`` for trajectory runs.

    Prefers the canonical ``data/.../experiment_state.json``; falls back to
    aggregating per-episode results from either ``data/`` or ``config/`` —
    whichever the runner happened to write into.
    """
    data: dict[tuple[int, str, str], float] = {}
    for batch in BATCHES:
        sources = [
            TRAJ_DATA_DIR / batch / "experiment_state.json",
            TRAJ_DATA_DIR / batch / "cache_eval_results.json",
            TRAJ_CONFIG_DIR / batch / "cache_eval_results.json",
        ]
        for src in sources:
            if not src.exists() or src.stat().st_size == 0:
                continue
            loader = (
                _load_state_runs
                if src.name == "experiment_state.json"
                else _load_episode_runs
            )
            populated = False
            for depth, kb, wid, sr in loader(src):
                data.setdefault((depth, kb, wid), sr)
                populated = True
            if populated:
                break  # this batch supplied data; skip remaining sources
    return data


def load_phase1_baseline():
    """Return ``{(kb, wid): success_rate}`` for libero_10 phase1 (w* only)."""
    base: dict[tuple[str, str], float] = {}
    for batch in BATCHES:
        state_path = PHASE1_DATA_DIR / batch / "experiment_state.json"
        if not state_path.exists():
            continue
        with open(state_path) as f:
            experiments = json.load(f)
        for exp in experiments:
            if exp.get("status") != "done":
                continue
            m = PHASE1_RUN_ID_PATTERN.match(exp.get("run_id", ""))
            if not m:
                continue  # skips p* (prompt_emb sweep) — not in trajectory
            base[(m.group(1), m.group(2))] = exp["success_rate"]
    return base


def load_phase1_roles():
    """Per kb, classify weights as top-1 / top-2 / top-3 / 2nd-worst by phase1.

    libero_10 trajectory selects 4 weights per kb (vs. 3 for libero_spatial),
    so we annotate up to four roles instead of three.
    """
    roles: dict[str, dict[str, str]] = defaultdict(dict)
    by_kb: dict[str, dict[str, float]] = defaultdict(dict)
    for (kb, wid), sr in load_phase1_baseline().items():
        by_kb[kb][wid] = sr
    for kb, scores in by_kb.items():
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        for idx, label in enumerate(["top1", "top2", "top3"]):
            if len(ranked) > idx:
                roles[kb][ranked[idx][0]] = label
        if len(ranked) >= 2:
            asc = sorted(scores.items(), key=lambda x: x[1])
            roles[kb][asc[1][0]] = "2nd_worst"
    return roles


def merge_with_phase1_baseline(traj_data, baseline):
    """Add depth=1 entries for each (kb, wid) actually used in trajectory."""
    used_pairs = {(kb, wid) for (_, kb, wid) in traj_data}
    for (kb, wid), sr in baseline.items():
        if (kb, wid) in used_pairs:
            traj_data[(1, kb, wid)] = sr
    return traj_data


def kb_weight_map(data):
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
# Role labels (extends the shared phase1 top-1/top-2/2nd-worst set with top-3)
# ------------------------------------------------------------------

LIBERO10_ROLE_LABELS = {**ROLE_LABELS, "top3": "phase1 top-3"}
LIBERO10_ROLE_MARKERS = {**ROLE_MARKERS, "top3": "D"}


# ------------------------------------------------------------------
# Main figure: line chart (success rate vs. depth)
# ------------------------------------------------------------------


def plot_lines(data):
    if not data:
        print("[plot_lines] no trajectory data — skipping line chart")
        return

    kb_weights = kb_weight_map(data)
    used_weights = all_weights_sorted(data)
    weight_color = {w: BAR_COLORS[i % len(BAR_COLORS)] for i, w in enumerate(used_weights)}
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
            marker = LIBERO10_ROLE_MARKERS.get(role, "x")
            role_tag = LIBERO10_ROLE_LABELS.get(role, "?")
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

    for ax in axes_flat[n_kb:]:
        ax.axis("off")

    fig.suptitle(
        "Trajectory libero_10: Success Rate vs. Depth (per key_builder)",
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
    if not data:
        print("[plot_facets] no trajectory data — skipping facet chart")
        return

    kb_weights = kb_weight_map(data)
    used_weights = all_weights_sorted(data)
    color_map = {w: BAR_COLORS[i % len(BAR_COLORS)] for i, w in enumerate(used_weights)}

    n_kb = len(KEY_BUILDER_ORDER)
    max_w = max((len(v) for v in kb_weights.values()), default=1)
    bar_width = 0.18
    group_width = max_w * bar_width

    n_depths = len(DEPTHS)
    n_cols = 2
    n_rows = (n_depths + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(
        n_rows, n_cols, figsize=(16, 4.5 * n_rows), sharey=True
    )
    axes = axes_grid.flatten()

    for ax, depth in zip(axes, DEPTHS):
        x_centers = np.arange(n_kb)
        for kb_idx, kb in enumerate(KEY_BUILDER_ORDER):
            weights = kb_weights[kb]
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
        "Trajectory libero_10: CP1 Cache Experiment Results (faceted by depth)",
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


def _summarize(data):
    by_depth = defaultdict(int)
    for (d, _, _) in data:
        by_depth[d] += 1
    print(
        "Loaded",
        sum(by_depth.values()),
        "(depth, kb, wid) cells:",
        ", ".join(f"d={d}: {by_depth[d]}" for d in sorted(by_depth)),
    )


if __name__ == "__main__":
    traj = load_trajectory_data()
    baseline = load_phase1_baseline()
    data = merge_with_phase1_baseline(traj, baseline)
    _summarize(data)
    plot_lines(data)
    plot_facets(data)
