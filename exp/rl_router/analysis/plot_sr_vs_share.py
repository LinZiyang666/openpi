#!/usr/bin/env python3
"""SR as a function of teacher share: fixed ratios vs what the router reached.

    python plot_sr_vs_share.py <sweep.json> <metrics_dir> <out.png> \
        [--cheap-arm student|cache] [--run <run_id> ...] [--bin-edges a,b,c]

This is the figure the whole ts line reduces to. Two instruments on the same
B pool, drawn in the same axes:

* the **fixed-ratio curve** -- seven constant policies (zeroed trunk, so the
  logits are exactly ``b2`` and the mixture is state-independent), each on the
  same 200 (task, init) draw;
* the **router**, binned by the teacher share it actually realised over four
  4,000-episode runs.

If state-dependence bought anything, the router's points would sit ABOVE the
curve at matched share. They do not: five of six bins are below it and the
sixth is within its interval.

Error bars are Wilson 95% intervals, which is the honest interval for a
proportion at these n -- a symmetric normal interval overstates precision near
the 0.9 range where every point here sits.

Deliberately light-mode only: this renders to a PNG embedded in a markdown
report, so there is no viewer theme to respond to.
"""
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# dataviz categorical slots 1-2, light mode. Two series only, so the palette is
# trivially inside the validated all-pairs set (slots 1-3 clear all-pairs in
# both modes). Both carry direct labels anyway -- slot 2 is above the 3:1
# contrast line but the figure leans on identity, not on the legend.
C_FIXED, C_ROUTER = "#2a78d6", "#eb6834"
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#78776f"
SURFACE, GRID = "#fcfcfb", "#e4e3de"
EP_PER_BATCH = 100
# The runs, the cheap arm and the bins are ARGUMENTS, not constants. v1 hardcoded
# the four ts runs and `student`; pointing it at the tc sweep silently plotted
# ts router points on the tc curve -- a figure that looks right and says
# something false. Anything arm-set-specific has to come from the command line.
DEFAULT_BINS = "0.15,0.25,0.30,0.35,0.40,0.45,0.60"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep"); ap.add_argument("metrics_dir"); ap.add_argument("out")
    ap.add_argument("--cheap-arm", default="student",
                    help="the arm the sweep's p is measured AGAINST (p = 1 - its share)")
    ap.add_argument("--run", action="append", required=True, help="run id; repeat")
    ap.add_argument("--bin-edges", default=DEFAULT_BINS)
    ap.add_argument("--title", default="The router never beats the constant it is standing on")
    args = ap.parse_args()

    sweep_path, mdir, out = (pathlib.Path(args.sweep), pathlib.Path(args.metrics_dir),
                             pathlib.Path(args.out))
    bin_edges = [float(x) for x in args.bin_edges.split(",")]
    sweep = [r for r in json.loads(sweep_path.read_text()) if r.get("success_rate") is not None]
    sweep.sort(key=lambda r: r["p_realized"])

    pts = []
    for rid in args.run:
        for line in (mdir / f"{rid}.jsonl").read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                pts.append((1.0 - m["arm_executed_rate"].get(args.cheap_arm, float("nan")),
                            m["mean_success"]))

    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        sel = [(t, s) for t, s in pts if lo <= t < hi]
        if not sel:
            continue
        n = len(sel) * EP_PER_BATCH
        k = round(sum(s for _, s in sel) * EP_PER_BATCH)
        bins.append((sum(t for t, _ in sel) / len(sel), k / n, *wilson(k, n), n))

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    fig.subplots_adjust(left=0.095, right=0.815, top=0.775, bottom=0.165)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = [r["p_realized"] for r in sweep]
    ys = [r["success_rate"] for r in sweep]
    lo = [y - wilson(r["successes"], r["episodes"])[0] for y, r in zip(ys, sweep)]
    hi = [wilson(r["successes"], r["episodes"])[1] - y for y, r in zip(ys, sweep)]
    ax.errorbar(xs, ys, yerr=[lo, hi], color=C_FIXED, lw=2.0, marker="o", ms=6.5,
                capsize=3, elinewidth=1.2, zorder=3, solid_capstyle="round",
                markeredgecolor=SURFACE, markeredgewidth=1.6)

    bx = [b[0] for b in bins]
    by = [b[1] for b in bins]
    blo = [b[1] - b[2] for b in bins]
    bhi = [b[3] - b[1] for b in bins]
    ax.errorbar(bx, by, yerr=[blo, bhi], color=C_ROUTER, lw=0, marker="s", ms=7.5,
                capsize=3, elinewidth=1.2, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.6)

    best = max(sweep, key=lambda r: r["p_realized"] * 0 + r["success_rate"])
    # "knee" is only the right word for an INTERIOR maximum. When the argmax sits
    # on an end point the curve is monotone and the best fixed policy is a pure
    # arm -- a corner solution, which is a different (and stronger) statement:
    # there is no mixture to find, so nothing for a router to do.
    at_edge = best is sweep[0] or best is sweep[-1]
    label = "corner (monotone curve)" if at_edge else "knee"
    ax.axvline(best["p_realized"], color=TEXT_MUTED, lw=1.0, ls=(0, (4, 4)), zorder=1)
    ax.annotate(f"{label}  p*={best['p_realized']:.2f}\n"
                f"best fixed ratio {best['success_rate']:.3f}",
                xy=(best["p_realized"] - 0.42 if at_edge and best is sweep[-1]
                    else best["p_realized"] + 0.008,
                    min(by) - 0.025), fontsize=8.6, color=TEXT_MUTED,
                va="bottom", ha="left")

    ax.annotate("fixed ratio\n(constant)", xy=(xs[-1] + 0.012, ys[-1] + 0.008),
                color=C_FIXED, fontsize=9.5, fontweight="bold", va="center", ha="left",
                annotation_clip=False)
    ax.annotate("router\n(binned)", xy=(bx[-1] + 0.012, by[-1] - 0.020),
                color=C_ROUTER, fontsize=9.5, fontweight="bold", va="center", ha="left",
                annotation_clip=False)

    fig.text(0.095, 0.955, args.title,
             ha="left", va="top", fontsize=15, color=TEXT_PRIMARY, fontweight="bold")
    fig.text(0.095, 0.898,
             f"libero_10, B pool.  teacher vs {args.cheap_arm}.  Fixed ratios: "
             f"{len(sweep)} constant policies × 200 episodes on one (task, init) draw.\n"
             f"Router: {len(pts) * EP_PER_BATCH:,} episodes over {len(args.run)} run(s), "
             "binned by the teacher share each batch actually realised.",
             ha="left", va="top", fontsize=9.2, color=TEXT_SECONDARY)

    ax.set_xlabel("teacher share  p   (fraction of control steps executed by the teacher)",
                  fontsize=10, color=TEXT_SECONDARY, labelpad=7)
    ax.set_ylabel("episode success rate", fontsize=10, color=TEXT_SECONDARY, labelpad=7)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    xmax = max(max(xs), max(bx)) + 0.06
    ylo = min(min(ys), min(by)) - 0.04
    yhi = max(max(ys), max(by)) + 0.04
    ax.set_xlim(-0.02, xmax)
    ax.set_ylim(ylo, yhi)

    fig.text(0.095, 0.075,
             "Error bars: Wilson 95%.  The two instruments' absolute levels are not perfectly "
             f"comparable — the sweep's 200 pairs\nare one subsample, the runs' "
             f"{len(pts) * EP_PER_BATCH:,} episodes are the pool mean.  The claim supported "
             "here is PARITY,\nnot \"the router is worse\".",
             fontsize=8.4, color=TEXT_MUTED, ha="left", va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
