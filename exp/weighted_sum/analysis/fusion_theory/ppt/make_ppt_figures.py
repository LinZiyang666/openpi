"""Slide-optimized figures for the 5-page 'Why z-score + tanh' collaborator deck.

One message per figure, large type, real data (libero_10 / cp1_spatial_pool_16
LOEO pools; headline numbers from expB/expE JSONs). Companion file:
slides_zscore_tanh.md.

Usage:
    uv run exp/weighted_sum/analysis/fusion_theory/ppt/make_ppt_figures.py \
        --cache-dir <scores-cache> \
        --results exp/weighted_sum/analysis/fusion_theory/results \
        --out exp/weighted_sum/analysis/fusion_theory/ppt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fusion_theory_common as C  # noqa: E402

BLUE, RED, MUTED, GREEN, VIOLET, YELLOW = (
    "#2a78d6", "#e34948", "#898781", "#008300", "#4a3aa7", "#eda100",
)
INK, SEC = "#0b0b0b", "#52514e"
PRIMARY = ("libero_10", "cp1_spatial_pool_16")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": SEC, "axes.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": SEC, "ytick.color": SEC,
    "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
    "text.color": INK, "legend.frameon": False, "pdf.fonttype": 42,
})


def density(x, lo, hi, bins=240, smooth=5):
    h, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    k = np.ones(smooth) / smooth
    h = np.convolve(h, k, mode="same")
    return 0.5 * (edges[1:] + edges[:-1]), h


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    art = C.load_artifact(*PRIMARY, Path(args.cache_dir))
    loeo = ~art.same_traj
    st = art.same_task[loeo]
    pool = {f: C.orient(art.raw[f].astype(np.float64), C.SIM_TYPE[f])[loeo] for f in C.FIELDS}
    stats = {f: (pool[f].mean(), pool[f].std()) for f in C.FIELDS}

    # ---------------- fig 1: three raw scales + the tiny signal -------------
    fig = plt.figure(figsize=(11.5, 4.0))
    ax = fig.add_axes([0.05, 0.16, 0.56, 0.74])
    xs, h = density(pool["robot_state"], -4.8, 0.2)
    ax.fill_between(xs, h / h.max(), color=MUTED, alpha=0.55, lw=0)
    for f, xoff in (("vision_0", 0), ("vision_1", 0)):
        xs, h = density(pool[f], 0.90, 1.005, bins=400, smooth=3)
        ax.fill_between(xs, h / h.max(), color=BLUE, alpha=0.9, lw=0)
    ax.annotate("robot state  (−L2 distance)\nσ ≈ 0.75", xy=(-2.0, 0.55),
                xytext=(-4.5, 0.80), fontsize=15, color=SEC)
    ax.annotate("both vision cosines\nσ ≈ 0.006  →", xy=(0.93, 0.55),
                xytext=(-1.7, 0.30), fontsize=15, color=BLUE, ha="left")
    ax.annotate("", xy=(0.93, 0.42), xytext=(-0.25, 0.38),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax.set_yticks([])
    ax.set_xlabel("raw similarity score (true units)")
    ax.set_title("Three modalities, one axis: a 300× scale spread")
    ax.set_xlim(-4.8, 1.1)

    ax2 = fig.add_axes([0.68, 0.16, 0.30, 0.74])
    xs, hc = density(pool["vision_0"][~st], 0.950, 1.002, bins=200, smooth=5)
    _, hs = density(pool["vision_0"][st], 0.950, 1.002, bins=200, smooth=5)
    m = max(hc.max(), hs.max())
    ax2.fill_between(xs, hc / m, color=MUTED, alpha=0.55, lw=0, label="vision0")
    ax2.fill_between(xs, hs / m, color=BLUE, alpha=0.60, lw=0, label="vision1")
    mu_c = pool["vision_0"][~st].mean()
    mu_s = pool["vision_0"][st].mean()
    ax2.annotate("", xy=(mu_s, 1.03), xytext=(mu_c, 1.03),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=2))
    ax2.annotate(f"Δ ≈ {mu_s - mu_c:.3f}", xy=((mu_s + mu_c) / 2, 1.06),
                 ha="center", fontsize=15, fontweight="bold")
    ax2.set_yticks([])
    ax2.set_ylim(0, 1.18)
    ax2.set_xlabel("vision cosine (zoomed ×90)")
    ax2.set_title("The task signal: 3rd decimal")
    ax2.legend(loc="lower left", fontsize=12)
    save(fig, out, "ppt_fig1_scales")

    # ---------------- fig 2: percentile censors the decision region ---------
    f = "vision_0"
    mu, sd = stats[f]
    z_all = (pool[f] - mu) / sd
    zmat = (C.orient(art.raw[f].astype(np.float64), "cosine") - mu) / sd
    zmasked = np.where(art.same_traj, -np.inf, zmat)
    ztop = -np.sort(-zmasked, axis=1)[:, :10].ravel()
    p_leg = C.fit_legacy_percentile(C.pool_random_pair(art.raw[f].astype(np.float64)), "cosine")
    zg = np.linspace(-4.2, 4.6, 600)
    xg = mu + zg * sd
    leg = np.clip((C.map_legacy(xg, "cosine") - p_leg["p5"]) / (p_leg["p95"] - p_leg["p5"]), 0, 1)
    tanh = 0.5 * (np.tanh(zg) + 1)
    z_flat = zg[np.argmax(leg >= 1.0)]

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(10.5, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 2.4], "hspace": 0.10})
    xs, h = density(z_all, -4.2, 4.6)
    a0.fill_between(xs, h / h.max(), color=MUTED, alpha=0.5, lw=0)
    xs, h = density(ztop, -4.2, 4.6, bins=120, smooth=3)
    a0.fill_between(xs, h / h.max(), color=BLUE, alpha=0.75, lw=0)
    a0.set_yticks([])
    a0.annotate("all candidates", xy=(-3.2, 0.55), color=SEC, fontsize=14)
    a0.annotate("top-10 per query\n= where the pick is decided", xy=(0.9, 0.62),
                color=BLUE, fontsize=14, fontweight="bold")
    a0.set_title("Retrieval decisions live in the far right tail")

    a1.axvspan(z_flat, 4.6, color=RED, alpha=0.10, lw=0)
    a1.plot(zg, leg, color=RED, lw=3.5, label="old: percentile clip")
    a1.plot(zg, tanh, color=BLUE, lw=3.5, label="new: ½(tanh z + 1)")
    a1.annotate("flat = every top candidate\nscores exactly 1.0 (tied)\n→ 88–99% of top-10 tied\n→ winner = storage order",
                xy=(3.15, 0.40), ha="center", color=RED, fontsize=14, fontweight="bold")
    a1.annotate("still climbing:\nnever a tie", xy=(2.4, 0.99), xytext=(-1.2, 0.86),
                color=BLUE, fontsize=14, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    a1.set_xlabel("standardized similarity  z")
    a1.set_ylabel("normalized score")
    a1.set_ylim(-0.05, 1.1)
    a1.legend(loc="center left", fontsize=13)
    save(fig, out, "ppt_fig2_censoring")

    # ---------------- fig 3: z-score = one common language ------------------
    fig = plt.figure(figsize=(11.5, 4.2))
    axl = fig.add_axes([0.04, 0.18, 0.24, 0.62])
    xs, h = density(pool["robot_state"], -4.8, 0.2)
    axl.fill_between(xs, h / h.max(), color=MUTED, alpha=0.5, lw=0)
    xs, h = density(pool["vision_0"], 0.90, 1.005, bins=400, smooth=3)
    axl.fill_between(xs, h / h.max(), color=BLUE, alpha=0.9, lw=0)
    axl.set_yticks([])
    axl.set_xticks([-4, -2, 0, 1])
    axl.set_title("raw units", fontsize=15)
    axl.set_xlabel("incomparable", fontsize=13)

    fig.text(0.30, 0.62, "z = (x−μ)/σ\nper modality,\nfitted offline", fontsize=14,
             ha="center", fontweight="bold", color=INK)
    fig.text(0.30, 0.42, "→", fontsize=34, ha="center", color=INK)

    names = {"vision_0": "vision₀", "vision_1": "vision₁", "robot_state": "state"}
    dz = {"vision_0": 1.3, "vision_1": 0.6, "robot_state": 0.7}
    for i, f in enumerate(C.FIELDS):
        axr = fig.add_axes([0.46, 0.66 - i * 0.26, 0.50, 0.22])
        mu_f, sd_f = stats[f]
        zf = (pool[f] - mu_f) / sd_f
        xs, hs = density(zf[st], -4, 4)
        m = hs.max()
        axr.fill_between(xs, hs / m, color=BLUE, alpha=0.6, lw=0)
        axr.set_yticks([])
        axr.set_xlim(-4, 4)
        axr.set_ylabel(names[f], rotation=0, ha="right", va="center", fontsize=15, color=INK)
        axr.annotate(f"same-task shift ≈ {dz[f]:.1f}σ", xy=(2.05, 0.55), fontsize=13,
                     color=BLUE, fontweight="bold")
        if i < 2:
            axr.set_xticklabels([])
        else:
            axr.set_xlabel("z  (same units for every modality → weights = importance)")
        if i == 0:
            axr.set_title("z units — signal visible & comparable", fontsize=16)
    save(fig, out, "ppt_fig3_zscore")

    # ---------------- fig 4: tanh = soft cap --------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    zg = np.linspace(-4.5, 4.5, 400)
    ax.plot(zg, 0.5 * (np.tanh(zg) + 1), color=BLUE, lw=4, label="ŝ = ½(tanh z + 1)")
    ax.plot(zg, np.clip(0.5 + zg / 3.29, 0, 1), color=RED, lw=2, ls="--", alpha=0.8,
            label="hard clip (old)")
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    ax.axhline(0.0, color=MUTED, lw=1, ls=":")
    ax.annotate("① near-linear core\nreal margins preserved", xy=(0.15, 0.54),
                xytext=(-4.3, 0.72), fontsize=14, fontweight="bold", color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    ax.annotate("② always climbing\n→ never creates ties\n(clip goes flat here)", xy=(2.9, 0.997),
                xytext=(1.15, 0.60), fontsize=14, fontweight="bold", color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax.annotate("③ hard bounds [0, 1]:\none modality can never\nbuy or veto the pick", xy=(-3.4, 0.015),
                xytext=(-3.3, 0.20), fontsize=14, fontweight="bold", color=VIOLET,
                arrowprops=dict(arrowstyle="->", color=VIOLET, lw=2))
    ax.set_xlabel("z  (how unusual the match is)")
    ax.set_ylabel("normalized score ŝ")
    ax.legend(loc="upper left", fontsize=13)
    ax.set_title("Tanh = a soft cap: keep every ranking, bound every vote")
    save(fig, out, "ppt_fig4_tanh")

    # ---------------- fig 5: results ----------------------------------------
    B = json.loads((Path(args.results) / "expB_results.json").read_text())
    combos = ["libero_spatial/cp1_spatial_pool_16", "libero_10/cp1_spatial_pool_16"]
    old = [B[c]["metrics"]["legacy@RP|w_prod"]["top1_same_task"]["mean"] for c in combos]
    new = [B[c]["metrics"]["zscore@LOEO|w_prod"]["top1_same_task"]["mean"] for c in combos]
    old_best = [B[c]["sweep"]["legacy@RP"]["max"] for c in combos]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    fig.subplots_adjust(bottom=0.22)
    xpos = np.array([0, 1.0])
    ax.bar(xpos - 0.19, old, width=0.34, color=RED, label="percentile clip (old)")
    ax.bar(xpos + 0.19, new, width=0.34, color=BLUE, label="z-score + tanh (new)")
    for i in range(2):
        ax.annotate(f"{old[i]:.2f}", xy=(xpos[i] - 0.19, old[i] + 0.02), ha="center",
                    fontsize=17, fontweight="bold", color=RED)
        ax.annotate(f"{new[i]:.2f}", xy=(xpos[i] + 0.19, new[i] + 0.02), ha="center",
                    fontsize=17, fontweight="bold", color=BLUE)
        ax.plot([xpos[i] - 0.36, xpos[i] - 0.02], [old_best[i]] * 2, color=INK, lw=2.5)
        ax.annotate("LIBERO-Spatial" if i == 0 else "LIBERO-10",
                    xy=(xpos[i], -0.20), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=16, fontweight="bold", color=INK)
    ax.annotate("— best the old scheme can reach over ALL weight settings",
                xy=(xpos[0] - 0.34, old_best[0]), xytext=(-0.44, 1.055),
                fontsize=13, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.6,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_xticks([xpos[0] - 0.19, xpos[0] + 0.19, xpos[1] - 0.19, xpos[1] + 0.19])
    ax.set_xticklabels(["old", "new"] * 2, fontsize=15)
    for t, c in zip(ax.get_xticklabels(), [RED, BLUE, RED, BLUE], strict=False):
        t.set_color(c)
        t.set_fontweight("bold")
    ax.legend(loc="center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=13)
    ax.set_ylabel("top-1 correct-task retrieval")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Same data, same weights — only the normalization changed")
    save(fig, out, "ppt_fig5_results")

    # ---------------- fig 6: ranks say nothing (schematic) ------------------
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.0, 4.0))
    fig.subplots_adjust(wspace=0.30)
    for a in (a0, a1):
        a.set_xlim(0.25, 1.05)
        a.set_ylim(-0.4, 2.2)
        a.set_yticks([])
        a.spines["left"].set_visible(False)
        a.set_xlabel("similarity score")

    # left: two races, identical ranks, very different margins
    for y, x2, tag in ((1.6, 0.965, "photo-finish #2"), (0.5, 0.72, "blown-out #2")):
        a0.axhline(y, color=MUTED, lw=1, alpha=0.4)
        a0.plot([0.98], [y], "o", color=INK, ms=13)
        a0.plot([x2], [y], "o", color=MUTED, ms=13, mfc="white", mew=2.5)
        a0.annotate("#1", xy=(1.0, y + 0.16), ha="left", fontsize=13, fontweight="bold")
        a0.annotate("#2", xy=(x2 - 0.02, y + 0.16), ha="right", fontsize=13, color=SEC,
                    fontweight="bold")
        a0.annotate(tag, xy=(x2 - 0.05, y - 0.30), ha="center", fontsize=13, color=SEC)
    a0.annotate("RRF: both #2 get the SAME score", xy=(0.63, 2.05), ha="center",
                fontsize=14.5, fontweight="bold", color=RED)
    a0.annotate("z+tanh: 0.96 vs 0.55 — margins kept", xy=(0.63, -0.02), ha="center",
                fontsize=14.5, fontweight="bold", color=BLUE)
    a0.set_title("Ranks are margin-blind")

    # right: an 'empty room' still has a #1
    rng = np.random.default_rng(3)
    xs_bad = rng.uniform(0.30, 0.55, 14)
    a1.axhline(1.05, color=MUTED, lw=1, alpha=0.4)
    a1.plot(xs_bad, np.full_like(xs_bad, 1.05), "o", color=MUTED, ms=11, alpha=0.75)
    best = xs_bad.max()
    a1.plot([best], [1.05], "o", color=INK, ms=13)
    a1.annotate("still ranked #1!", xy=(best, 1.05), xytext=(0.80, 1.55),
                fontsize=14, fontweight="bold", color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    a1.annotate("nothing in the library is similar…", xy=(0.42, 0.72), ha="center",
                fontsize=13.5, color=SEC)
    a1.annotate('RRF top score: high, "as always"', xy=(0.63, 2.05), ha="center",
                fontsize=14.5, fontweight="bold", color=RED)
    a1.annotate("z+tanh top score: LOW → MISS", xy=(0.63, -0.02),
                ha="center", fontsize=14.5, fontweight="bold", color=BLUE)
    a1.set_title('"Best in the room" ≠ actually good')
    save(fig, out, "ppt_fig6_ranks")

    # ---------------- fig 7: thresholds need meaning (real expD data) -------
    import expD_missdetect as D  # noqa: PLC0415 (sibling module, path set above)

    hit_mask = ~art.same_traj
    miss_mask = ~art.same_task
    cfgs = D.build_field_scores(art)
    w = C.W_PROD
    s = {
        ("z", "hit"): D.top1_scores(cfgs["zscore_tanh"], art, hit_mask, w),
        ("z", "miss"): D.top1_scores(cfgs["zscore_tanh"], art, miss_mask, w),
        ("r", "hit"): D.rank_perquery_top1(art, hit_mask, w),
        ("r", "miss"): D.rank_perquery_top1(art, miss_mask, w),
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.2), sharex=True,
                             gridspec_kw={"hspace": 0.16})
    rows = (("z", "z-score + tanh", BLUE), ("r", "rank fusion (RRF-style)", RED))
    for ax, (kk, name, col) in zip(axes, rows, strict=False):
        lo = 0.82
        xs, hh = density(np.clip(s[(kk, "hit")], lo, 1.0), lo, 1.001, bins=140, smooth=5)
        m = hh.max()
        ax.fill_between(xs, hh / m, color=col, alpha=0.55, lw=0,
                        label="fusion score of the top1 entry")
        ax.set_yticks([])
        ax.annotate(name, xy=(0.825, 0.82), fontsize=15, fontweight="bold", color=col)
        med_h = np.median(s[(kk, "hit")])
        med_m = np.median(s[(kk, "miss")])
        thr = 0.5 * (med_h + med_m)
        ax.axvline(thr, color=INK, lw=2.2, ls="--")
        if kk == "z":
            ax.annotate("score drops when good matches vanish\n→ threshold usable: FULL-HIT / WARM / MISS",
                        xy=(thr, 0.55), xytext=(0.838, 0.12), fontsize=13.5,
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
            ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.55), fontsize=11.5)
        else:
            ax.annotate("no room: gap ~4× narrower,\nand it shrinks as the library grows",
                        xy=(thr, 0.55), xytext=(0.838, 0.12), fontsize=13.5,
                        fontweight="bold", color=RED,
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    axes[0].set_title("Can you tell 'good match' from 'best of a bad bunch'?")
    axes[1].set_xlabel("top-1 fused score (real data, task held out for MISS)")
    save(fig, out, "ppt_fig7_threshold")


if __name__ == "__main__":
    main()
