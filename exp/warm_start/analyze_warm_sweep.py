"""Warm Start Sweep analysis — success rate visualisation.

Reads baseline results from trajectory_deviation Step 1a and warm start
results from the warm_start_sweep experiment, then produces:

1. Grouped bar chart: B0, B1, warm t0.3/t0.5/t0.7 per key builder.
2. Line chart: success rate vs start_t, one line per key builder,
   with B0/B1 horizontal reference bands.

Usage:
    uv run python exp/warm_start/analyze_warm_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────

BUILDERS = ["max_pool", "spatial16", "clip"]
START_TS = [0.3, 0.5, 0.7]

BASELINE_FILES = {
    "max_pool": "exp/trajectory_deviation/data/cache_eval_results_maxpool.json",
    "spatial16": "exp/trajectory_deviation/data/cache_eval_results_spatial16.json",
    "clip": "exp/trajectory_deviation/data/cache_eval_results_clip.json",
}

WARM_FILES = {
    "max_pool": "exp/warm_start/data/max_pool/cache_eval_results.json",
    "spatial16": "exp/warm_start/data/spatial16/cache_eval_results.json",
    "clip": "exp/warm_start/data/clip/cache_eval_results.json",
}

WARM_CONFIG_IDS = {
    "max_pool": {0.3: "max_pool_w3_d5_warm_t0.3", 0.5: "max_pool_w3_d5_warm_t0.5", 0.7: "max_pool_w3_d5_warm_t0.7"},
    "spatial16": {0.3: "spatial16_w8_d4_warm_t0.3", 0.5: "spatial16_w8_d4_warm_t0.5", 0.7: "spatial16_w8_d4_warm_t0.7"},
    "clip": {0.3: "clip_w7_d4_warm_t0.3", 0.5: "clip_w7_d4_warm_t0.5", 0.7: "clip_w7_d4_warm_t0.7"},
}

BASELINE_INFERENCE_IDS = {
    "max_pool": "inference_max_pool_w3_d5",
    "spatial16": "inference_spatial16_w8_d4",
    "clip": "inference_clip_w7_d4",
}

BASELINE_HIT_IDS = {
    "max_pool": "max_pool_w3_d5",
    "spatial16": "spatial16_w8_d4",
    "clip": "clip_w7_d4",
}


def _success_rate(records: list[dict], config_id: str) -> float:
    subset = [r for r in records if r["config_id"] == config_id]
    if not subset:
        raise ValueError(f"No records for {config_id}")
    return sum(1 for r in subset if r["success"]) / len(subset) * 100


def load_all() -> dict:
    data = {}
    for b in BUILDERS:
        baseline = json.loads(Path(BASELINE_FILES[b]).read_text())
        warm = json.loads(Path(WARM_FILES[b]).read_text())
        data[b] = {
            "B0": _success_rate(baseline, BASELINE_INFERENCE_IDS[b]),
            "B1": _success_rate(baseline, BASELINE_HIT_IDS[b]),
        }
        for st in START_TS:
            data[b][st] = _success_rate(warm, WARM_CONFIG_IDS[b][st])
    return data


# ── Plot 1: Grouped bar chart ────────────────────────────────────────────

def plot_grouped_bar(data: dict, out: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = ["B1\n(always_hit)", "warm\nt=0.3", "warm\nt=0.5", "warm\nt=0.7", "B0\n(inference)"]
    x = np.arange(len(BUILDERS))
    n_bars = len(labels)
    width = 0.15
    offsets = np.arange(n_bars) - (n_bars - 1) / 2

    colors = ["#BDC3CB", "#7EB8DA", "#4A90C4", "#1B5E8A", "#0F3052"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        key = ["B1", 0.3, 0.5, 0.7, "B0"][i]
        vals = [data[b][key] for b in BUILDERS]
        bars = ax.bar(x + offsets[i] * width, vals, width, label=label,
                      color=color, edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5,
                    color="#333333")

    ax.set_ylabel("Success Rate (%)", fontsize=12, color="#333333")
    ax.set_title("Warm Start Sweep: Success Rate by Key Builder",
                 fontsize=14, fontweight="bold", color="#1a1a1a", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", " ") for b in BUILDERS], fontsize=11, color="#333333")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper left", fontsize=8.5, ncol=5, framealpha=0.9,
              edgecolor="#cccccc", bbox_to_anchor=(0, 1.01))
    ax.grid(axis="y", alpha=0.2, color="#999999", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", colors="#555555", length=0)

    fig.text(0.5, -0.02,
             "Note: B0 (inference) varies across key builders (98.4%–99.2%) due to model stochasticity;\n"
             "all values equally represent pure inference performance.",
             ha="center", fontsize=8, color="#777777", style="italic")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    out_dir = Path("exp/warm_start/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all()

    print("\n=== Summary ===")
    print(f"{'builder':<12} {'B0':>6} {'B1':>6} {'t0.3':>6} {'t0.5':>6} {'t0.7':>6}")
    for b in BUILDERS:
        print(f"{b:<12} {data[b]['B0']:>5.1f}% {data[b]['B1']:>5.1f}% "
              f"{data[b][0.3]:>5.1f}% {data[b][0.5]:>5.1f}% {data[b][0.7]:>5.1f}%")

    plot_grouped_bar(data, out_dir / "success_rate_sweep.png")


if __name__ == "__main__":
    main()
