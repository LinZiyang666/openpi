"""CDF of control steps per episode, unpruned library against three pruning levels.

One panel per family. Every one of the 500 episodes counts; an episode that
fails runs to the suite's step cap (220 / 520), so failures pile up at the
right edge. If pruning made successes slower the curves would separate along
their whole length; if it only turns successes into failures, they overlap
until the cap and differ only in how much mass is parked there.

Usage:
  uv run python exp/ablation_study/cache_prune/analysis/plot_prune_step_cdf.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from exp.ablation_study.cache_prune.analysis.analyze_prune import family_ledgers  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "direct_20260912" / "direct"
OUT = HERE / "figures"
STEM = "cache_prune_step_cdf"

INK, MUTED, GRID = "#1B2124", "#6B7780", "#E3E7EA"
FAMILIES = [
    ("A", "libero_spatial", "rit50", "LIBERO-Spatial, small library", 220),
    ("A", "libero_10", "rit50", "LIBERO-10, small library", 520),
    ("B", "libero_spatial", "cs500_success", "LIBERO-Spatial, large library", 220),
    ("B", "libero_10", "cs500_success", "LIBERO-10, large library", 520),
]
#: Unpruned in ink; deeper pruning in progressively warmer, lighter steps of one hue.
LEVELS = [("P00", INK, 2.2, "unpruned"), ("P03", "#5B8DEF", 1.5, "20% removed"),
          ("P06", "#E0873D", 1.5, "50% removed"), ("P09", "#C0392B", 1.5, "80% removed")]


def cdf(values: np.ndarray):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return np.concatenate([[x[0]], x]), np.concatenate([[0], y])


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    for ax, (lane, suite, regime, title, cap) in zip(axes.flat, FAMILIES):
        ledgers, _ = family_ledgers(DATA / lane / f"{suite}_{regime}")
        ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.axvline(cap, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        ax.annotate(f"step cap {cap}", xy=(cap, 0.03), xytext=(-6, 0), textcoords="offset points",
                    color=MUTED, fontsize=8, ha="right")
        for point, color, width, label in LEVELS:
            arm = f"cache_prune_{suite}_{regime}_{point}"
            rows = [r for r in ledgers[arm] if "control_steps" in r]
            steps = np.array([r["control_steps"] for r in rows], dtype=float)
            success = np.mean([r["success"] for r in rows])
            x, y = cdf(steps)
            ax.step(x, y, where="post", color=color, linewidth=width, zorder=3,
                    label=f"{label}  (success {success:.0%})")
        ax.set_xlim(0, cap * 1.04)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=8)
        ax.set_xlabel("control steps in the episode", color=MUTED, fontsize=9.5)
        ax.set_ylabel("fraction of episodes finished", color=MUTED, fontsize=9.5)
        ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Episode length distribution with and without pruning (all 500 episodes per curve)",
                 color=INK, fontsize=13, x=0.012, ha="left", y=0.99)
    fig.text(0.012, 0.006,
             "A failed episode runs to the cap, so the jump at the right edge is the failure rate. "
             "Curves that overlap before the cap mean successes are not slower, only fewer.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}/{STEM}.png / .pdf")


if __name__ == "__main__":
    main()
