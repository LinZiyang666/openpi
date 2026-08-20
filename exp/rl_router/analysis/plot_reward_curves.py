#!/usr/bin/env python3
"""M6 training-side reward and success curves, one figure, four runs.

    python plot_reward_curves.py <metrics_dir> <out.png>

``metrics_dir`` holds one ``<run_id>.jsonl`` per run (the trainer's
``metrics.jsonl``, one row per batch).

Two panels rather than one, on purpose. ``R = success - lambda * cost / T_max``,
and the runs do not share a lambda: the lambda=0.05 run's reward sits about
0.025 higher than the lambda=0.5 runs **by construction**, because its cost term
is ten times smaller. A single reward panel therefore invites the reading "the
weak-penalty run learned better", which the data does not support. Plotting
success directly underneath — same units, same axis range, and the one quantity
that *is* comparable across lambda — makes that offset self-evident instead of
a caption nobody reads.

Raw per-batch values are drawn thin and pale; the heavy line is a centred
5-batch rolling mean. Both are shown because the point of the figure is that
there is no trend, and a smoother alone can manufacture one.
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# dataviz categorical slots 1-3 plus slot 7 (violet) for the fourth series,
# light mode. Validated with scripts/validate_palette.js --mode light
# --pairs all -> ALL CHECKS PASS (worst normal-vision dE 16.3, worst CVD 9.2).
#
# Why slot 7 and not slot 4: the documented palette's own note says slot 4 puts
# yellow beside orange and that pair fails the all-pairs floors, inviting the
# reader to revisit the trade "in real charts with four or more series". This is
# that chart, and the numbers confirm it -- slots 1-4 give normal-vision
# dE 13.7 (FAIL, below the 15 floor) for #eda100 vs #eb6834. Four lines on one
# axis are read pairwise, not just between neighbours, so the all-pairs list is
# the honest one here even though a line chart defaults to adjacent.
#
# The aqua slot still warns below 3:1 on the light surface, so the relief rule
# applies: every series carries a direct label at the line end.
#
# The direct label is the run's own short tag, not its lambda: three of the four
# runs share lambda=0.5, so labelling by lambda alone would not identify a line.
SERIES = [
    ("l10_ts_lam1_s0", "λ₁ seed 0", "#2a78d6"),
    ("l10_ts_lam1_s1", "λ₁ seed 1", "#eb6834"),
    ("l10_ts_lam2_s0", "λ₂ seed 0", "#1baf7a"),
    ("l10_ts_lam1_s0_knee", "λ₁ knee start", "#4a3aa7"),
]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#78776f"
SURFACE = "#fcfcfb"
GRID = "#e4e3de"
ROLL = 5
YLIM = (0.79, 0.985)


def rolling(v, w=ROLL):
    """Centred rolling mean, shrinking the window at the edges."""
    out = []
    half = w // 2
    for i in range(len(v)):
        lo, hi = max(0, i - half), min(len(v), i + half + 1)
        out.append(sum(v[lo:hi]) / (hi - lo))
    return out


def load(path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {
        "reward": [r["mean_reward"] for r in rows],
        "success": [r["mean_success"] for r in rows],
        "n": len(rows),
    }


def _spread(ends, min_gap):
    """Push overlapping label anchors apart, preserving their order.

    Line ends can land within a few thousandths of each other; drawn at their
    true y the labels overlap and the figure loses the identity encoding that
    the relief rule requires. Ordering is kept so a label never crosses a line
    it does not belong to.
    """
    order = sorted(range(len(ends)), key=lambda i: ends[i])
    out = list(ends)
    for k in range(1, len(order)):
        prev, cur = order[k - 1], order[k]
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out


def panel(ax, data, key, title, subtitle):
    smooth = {rid: rolling(data[rid][key]) for rid, _, _ in SERIES}
    anchors = _spread([smooth[rid][-1] for rid, _, _ in SERIES],
                      min_gap=(YLIM[1] - YLIM[0]) * 0.055)

    for (rid, tag, colour), y_anchor in zip(SERIES, anchors):
        d = data[rid]
        x = list(range(1, d["n"] + 1))
        ax.plot(x, d[key], color=colour, lw=1.0, alpha=0.28, zorder=2)
        ax.plot(x, smooth[rid], color=colour, lw=2.0, zorder=3, solid_capstyle="round")
        ax.annotate(tag, xy=(x[-1] + 0.6, y_anchor), xycoords="data",
                    color=colour, fontsize=9, va="center", ha="left",
                    fontweight="bold", annotation_clip=False)

    ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, pad=20)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.8,
            color=TEXT_SECONDARY, va="bottom", ha="left")
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.set_ylim(*YLIM)
    ax.set_xlim(0, 44)


def main():
    mdir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    data = {rid: load(mdir / f"{rid}.jsonl") for rid, _, _ in SERIES}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.0, 8.4), sharex=True,
        gridspec_kw={"hspace": 0.42, "left": 0.085, "right": 0.855,
                     "top": 0.815, "bottom": 0.145})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)

    fig.text(0.085, 0.962, "M6 online RL router — 4 runs × 4,000 episodes, no improvement in any",
             ha="left", va="top", fontsize=15, color=TEXT_PRIMARY, fontweight="bold")

    handles = [plt.Line2D([], [], color=c, lw=2.0) for _, _, c in SERIES]
    labels = [f"{rid}  ({tag})" for rid, tag, _ in SERIES]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.082, 0.925),
               frameon=False, fontsize=9, ncol=2, labelcolor=TEXT_SECONDARY,
               handlelength=1.6, columnspacing=2.0)

    panel(ax1, data, "reward", "Batch mean reward",
          "R = success − λ·cost/T_max.  The λ₂ run sits ~0.025 higher because its cost term is "
          "10× smaller — not because it learned more.")
    panel(ax2, data, "success", "Batch mean success rate",
          "The quantity that IS comparable across λ.  Flat at ~0.90 in all four runs, including the knee-start one.")

    ax2.set_xlabel("training batch  (100 episodes each)", fontsize=10,
                   color=TEXT_SECONDARY, labelpad=6)

    fig.text(0.085, 0.045,
             "B-pool inits, judge in sample mode — training-side numbers, not the frozen-policy "
             "A-pool evaluation (M7, not yet run).",
             fontsize=8.5, color=TEXT_MUTED, ha="left", va="top")
    fig.text(0.085, 0.022,
             "Thin line = per batch;  heavy line = centred 5-batch rolling mean.",
             fontsize=8.5, color=TEXT_MUTED, ha="left", va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
