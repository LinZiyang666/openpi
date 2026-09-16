"""The pruning curves: success, paired loss vs P00, and how failures arrive.

Top row: success rate against the fraction of entries removed, one panel per
suite, the small (rit50) and large (cs500) source libraries overlaid, with the
task-level bootstrap band. Middle row: the paired difference to the unpruned
library on the same 500 inits, with its 95% interval -- the quantity the
experiment actually estimates; the first point whose interval sits entirely
below zero is marked on both rows. Bottom row: mean control length per point,
over all 500 episodes (dashed, hollow; failures sit at the step cap and pull it
up) and over the solved episodes only (solid) -- whether a pruned library's
successes take longer than the unpruned library's.

Usage:
  uv run python exp/ablation_study/cache_prune/analysis/plot_prune_curves.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ANALYSIS = HERE / "cache_prune_analysis.json"
OUT = HERE / "figures"
STEM = "cache_prune_curves"

INK, MUTED, GRID = "#1B2124", "#6B7780", "#E3E7EA"
SERIES = {
    "rit50": ("#2E6FD9", "o", "small library (50 trajectories)"),
    "cs500_success": ("#C8641E", "s", "large library (500-init successes)"),
}
TITLE = {"libero_spatial": "LIBERO-Spatial", "libero_10": "LIBERO-10"}


def curve_rows(curve: dict) -> dict:
    points = curve["subsets"]["common_unseen"]["points"]
    x = np.array([lib["actual_rate"] * 100 for lib in curve["libraries"]])
    sr = np.array([p["sr"] for p in points]) * 100
    band = np.array([p["sr_ci95"] for p in points]) * 100
    delta = np.array([p["delta_sr"] for p in points]) * 100
    dband = np.array([p["delta_sr_ci95"] for p in points]) * 100
    steps = np.array([a["mean_control_steps_success"] if a["mean_control_steps_success"] is not None else np.nan
                      for a in curve["arms"]])
    steps_all = np.array([a["mean_control_steps"] if a["mean_control_steps"] is not None else np.nan
                          for a in curve["arms"]])
    common = [a["n_success"] for a in curve["arms"]]
    first = next((i for i, p in enumerate(points) if p["delta_sr_ci95"][1] < 0), None)
    entries = [lib["entries"] for lib in curve["libraries"]]
    return {"x": x, "sr": sr, "band": band, "delta": delta, "dband": dband,
            "first": first, "entries": entries, "steps": steps, "steps_all": steps_all,
            "common": common}


def style(ax, ylabel):
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)
    ax.set_xlim(-3, 84)
    ax.set_xticks(range(0, 81, 20))


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text())
    curves = {(c["suite"], c["regime"]): curve_rows(c) for c in analysis["curves"]}
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.6), sharex="col",
                             gridspec_kw={"height_ratios": [1.15, 1, 0.8]})
    for col, suite in enumerate(TITLE):
        top, bottom, length = axes[0, col], axes[1, col], axes[2, col]
        style(top, "success rate (%)" if col == 0 else "")
        style(bottom, "change vs unpruned library (pp)" if col == 0 else "")
        style(length, "mean control steps per episode" if col == 0 else "")
        bottom.axhline(0, color=INK, linewidth=0.9, alpha=0.6, zorder=2)
        for regime, (color, marker, label) in SERIES.items():
            r = curves[suite, regime]
            top.fill_between(r["x"], r["band"][:, 0], r["band"][:, 1], color=color, alpha=0.10, linewidth=0)
            top.plot(r["x"], r["sr"], "-", color=color, linewidth=1.8, marker=marker, markersize=5.5,
                     markeredgecolor="white", markeredgewidth=0.9, zorder=3)
            bottom.fill_between(r["x"], r["dband"][:, 0], r["dband"][:, 1], color=color, alpha=0.10, linewidth=0)
            bottom.plot(r["x"], r["delta"], "-", color=color, linewidth=1.8, marker=marker, markersize=5.5,
                        markeredgecolor="white", markeredgewidth=0.9, zorder=3)
            # Direct label at the right end: the library's full size, the number
            # that decides how far it can be pruned.
            top.annotate(f"{label.split(' (')[0]}\n{r['entries'][0]:,} entries", xy=(r["x"][0], r["sr"][0]),
                         xytext=(6, 8 if regime == "cs500_success" else -22), textcoords="offset points",
                         color=color, fontsize=8.5, ha="left")
            length.plot(r["x"], r["steps_all"], "--", color=color, linewidth=1.3, marker=marker,
                        markersize=5, markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=1.1, alpha=0.9, zorder=3)
            length.plot(r["x"], r["steps"], "-", color=color, linewidth=1.8, marker=marker, markersize=5.5,
                        markeredgecolor="white", markeredgewidth=0.9, zorder=4)
            # How many solved episodes the last success-only mean rests on.
            length.annotate(f"n={r['common'][-1]}", xy=(r["x"][-1], r["steps"][-1]),
                            xytext=(6, 5 if regime == "rit50" else -10), textcoords="offset points",
                            color=color, fontsize=7.5)
            if r["first"] is not None:
                i = r["first"]
                for ax, y in ((top, r["sr"][i]), (bottom, r["delta"][i])):
                    ax.plot([r["x"][i]], [y], marker="o", markersize=13, markerfacecolor="none",
                            markeredgecolor=color, markeredgewidth=1.4, zorder=4)
                bottom.annotate(f"first certain loss\n{r['x'][i]:.0f}% removed, {r['entries'][i]:,} left",
                                xy=(r["x"][i], r["delta"][i]), xytext=(-8, -30 if regime == "rit50" else 30),
                                textcoords="offset points", color=color, fontsize=8, ha="right")
        top.set_title(TITLE[suite], color=INK, fontsize=12, loc="left", pad=8)
        length.set_xlabel("entries removed (%)", color=MUTED, fontsize=9.5)
        top.set_ylim(0, 92)
        bottom.set_ylim(-70, 15)
        lo = min(np.nanmin(curves[suite, r]["steps"]) for r in SERIES)
        hi = max(np.nanmax(curves[suite, r]["steps_all"]) for r in SERIES)
        pad = max(4.0, 0.08 * (hi - lo))
        length.set_ylim(lo - pad, hi + pad)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="-", markersize=6,
                          markeredgecolor="white", markeredgewidth=0.9, label=lab)
               for c, m, lab in SERIES.values()]
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=11,
                              markerfacecolor="none", markeredgewidth=1.4,
                              label="first significant loss (paired 95% CI below 0)"))
    handles.append(plt.Line2D([], [], color=MUTED, linestyle="-", linewidth=1.8,
                              label="bottom row: solved episodes only"))
    handles.append(plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=1.3, marker="o",
                              markerfacecolor="white", markersize=5,
                              label="bottom row: all 500 episodes (failures at the step cap)"))
    axes[0, 1].legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Pruning a pure-cache library: success, paired loss, and episode length",
                 color=INK, fontsize=13.5, x=0.012, ha="left", y=0.99)
    fig.text(0.012, 0.006,
             "pi0.5 on LIBERO, every step answered from the cache (Stage 1 key, top-1 retrieval). "
             "500 paired inits per point (10 tasks x 50); bands are task-level paired bootstrap 95% intervals "
             "(10,000 draws). 40 arms, 20,000 episodes, 2026-09-12.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.972))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=170, bbox_inches="tight", facecolor="white")
    data = {
        "kind": "cache_prune_curves",
        "x_label": "entries removed (%)",
        "series": {
            f"{s}/{r}": {
                "removed_pct": v["x"].tolist(), "entries": v["entries"],
                "sr_pct": v["sr"].tolist(), "sr_ci95_pct": v["band"].tolist(),
                "delta_pp": v["delta"].tolist(), "delta_ci95_pp": v["dband"].tolist(),
                "first_certain_loss_index": v["first"],
                "mean_control_steps_success": [None if np.isnan(t) else float(t) for t in v["steps"]],
                "mean_control_steps_all": [None if np.isnan(t) else float(t) for t in v["steps_all"]],
                "n_success": v["common"],
            }
            for (s, r), v in curves.items()
        },
    }
    (OUT / f"{STEM}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {OUT}/{STEM}.png / .pdf / .json")


if __name__ == "__main__":
    main()
