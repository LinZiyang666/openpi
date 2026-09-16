"""CDF of how many steps further along the retained substitute is, per pruned entry.

Every pruned entry has a witness: the kept entry from another trajectory that
scored above the threshold and sits closer to the end of its own trajectory.
The gap ``r_removed - r_retained`` is the shortcut pruning takes at that state
(in control steps). If the gaps are mostly one or two steps, pruning removes
near-identical neighbours without shortening anything, and no speed-up can be
expected from it.

Usage:
  PYTHONPATH=. uv run python exp/ablation_study/cache_prune/analysis/plot_prune_witness_gap.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / "data" / "direct_20260912" / "cache_artifacts"
OUT = HERE / "figures"
STEM = "cache_prune_witness_gap"

INK, MUTED, GRID = "#1B2124", "#6B7780", "#E3E7EA"
FAMILIES = [
    ("libero_spatial", "rit50", "LIBERO-Spatial, small library"),
    ("libero_10", "rit50", "LIBERO-10, small library"),
    ("libero_spatial", "cs500_success", "LIBERO-Spatial, large library"),
    ("libero_10", "cs500_success", "LIBERO-10, large library"),
]
LEVELS = [("P03", "#5B8DEF", "20% removed"), ("P06", "#E0873D", "50% removed"),
          ("P09", "#C0392B", "80% removed")]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    summary = {}
    for ax, (suite, regime, title) in zip(axes.flat, FAMILIES):
        ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for point, color, label in LEVELS:
            arm = f"cache_prune_{suite}_{regime}_{point}"
            manifest = json.loads((ARTIFACTS / arm / "artifact.json").read_text())
            gaps = np.array([w["r_removed"] - w["r_retained"] for w in manifest["selection"]["witnesses"]])
            x = np.sort(gaps)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(np.concatenate([[0], x]), np.concatenate([[0], y]), where="post", color=color,
                    linewidth=1.7, zorder=3,
                    label=f"{label}: {len(gaps):,} pruned, median gap {int(np.median(gaps))}, p90 {int(np.quantile(gaps, 0.9))}")
            summary[f"{suite}/{regime}/{point}"] = {
                "pruned": int(len(gaps)), "median": float(np.median(gaps)),
                "p90": float(np.quantile(gaps, 0.9)), "share_le_2": float(np.mean(gaps <= 2)),
            }
        ax.set_xscale("log")
        ax.set_xlim(0.9, 300)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=8)
        ax.set_xlabel("steps the retained substitute is ahead of the pruned entry (log)", color=MUTED, fontsize=9.5)
        ax.set_ylabel("fraction of pruned entries", color=MUTED, fontsize=9.5)
        ax.legend(loc="lower right", frameon=False, fontsize=8.2, labelcolor=INK)
    fig.suptitle("What pruning actually removes: the step gap between a pruned entry and its substitute",
                 color=INK, fontsize=13, x=0.012, ha="left", y=0.99)
    fig.text(0.012, 0.006,
             "Gap = remaining steps of the pruned entry minus remaining steps of the kept neighbour it was "
             "judged redundant against (same task, other trajectory, retrieval score above the threshold).",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=170, bbox_inches="tight", facecolor="white")
    (OUT / f"{STEM}.json").write_text(json.dumps(summary, indent=2) + "\n")
    for key, value in summary.items():
        print(f"{key}: pruned={value['pruned']:,} median={value['median']:.0f} p90={value['p90']:.0f} "
              f"gap<=2: {value['share_le_2']:.0%}")
    print(f"wrote {OUT}/{STEM}.png / .pdf / .json")


if __name__ == "__main__":
    main()
