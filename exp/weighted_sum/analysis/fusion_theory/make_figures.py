"""Generate all figures for the fusion-normalization theory report.

Reads expA/B/C/D JSONs plus the score cache and writes PNG+PDF pairs into
``figs/``. Entity-color assignment is fixed across every figure:
  zscore+tanh #2a78d6 / ecdf-rank #1baf7a / clip-band family #eda100 /
  smooth sigmoids #008300 / identity #4a3aa7 / legacy percentile #e34948.

Usage:
    uv run exp/weighted_sum/analysis/fusion_theory/make_figures.py \
        --cache-dir <scores-cache> --data exp/weighted_sum/analysis/fusion_theory/data \
        --figs exp/weighted_sum/analysis/fusion_theory/figs
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fusion_theory_common as C  # noqa: E402

BLUE, AQUA, YELLOW, GREEN, VIOLET, RED = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
)
INK, SEC, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PRIMARY = ("libero_10", "cp1_spatial_pool_16")
SECOND = ("libero_spatial", "cp1_spatial_pool_16")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": SEC,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, figs: Path, name: str) -> None:
    fig.savefig(figs / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(figs / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def band_in_oriented(p5: float, p95: float, stype: str) -> tuple[float, float]:
    """Legacy mapped-space band -> oriented raw-score band."""
    if stype == "cosine":
        return 2 * p5 - 1, 2 * p95 - 1
    # s0 = exp(-d/tau) -> d = -tau ln s0 ; oriented x = -d
    return float(C.LEGACY_TAU * np.log(max(p5, 1e-12))), float(C.LEGACY_TAU * np.log(min(p95, 1.0)))


def fig1_distributions(art, A, figs):
    key = f"{art.suite}/{art.builder}"
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
    for ax, f in zip(axes, C.FIELDS, strict=False):
        stype = C.SIM_TYPE[f]
        raw = art.raw[f].astype(np.float64)
        lo_mask = ~art.same_traj
        x = C.orient(raw, stype)[lo_mask]
        st = art.same_task[lo_mask]
        cell = A[key][f]
        lo_b, hi_b = band_in_oriented(cell["cells"]["legacy@RP"]["params"]["p5"],
                                      cell["cells"]["legacy@RP"]["params"]["p95"], stype)
        xmin, xmax = np.percentile(x, 0.02), x.max()
        pad = 0.06 * (xmax - xmin)
        bins = np.linspace(xmin - pad, xmax + pad, 120)
        ax.hist(x[~st], bins=bins, density=True, color=MUTED, alpha=0.45,
                label="cross-task", edgecolor="none")
        ax.hist(x[st], bins=bins, density=True, color=BLUE, alpha=0.55,
                label="same-task", edgecolor="none")
        ax.axvspan(hi_b, xmax + pad, color=RED, alpha=0.12, lw=0)
        ax.axvline(hi_b, color=RED, lw=1.4)
        mu = cell["pools"]["loeo_mean_oriented"]
        ax.axvline(mu, color=SEC, lw=1.0, ls=":")
        # decision region: median z of per-field top-10 (from expB, stored later)
        ax.set_title(f)
        ax.set_xlabel("oriented raw score" + (" (cos)" if stype == "cosine" else " (-L2)"))
        ax.set_yticks([])
        ax.grid(False)
        ymax = ax.get_ylim()[1]
        ax.annotate("p95 → clipped to 1.0", xy=(hi_b, ymax * 0.55), color=RED,
                    fontsize=8, rotation=90, ha="right", va="center")
        ax.annotate("μ", xy=(mu, ymax * 0.97), color=SEC, fontsize=8, ha="center")
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"LOEO serving distributions — {key}", y=1.04, fontsize=11, fontweight="bold")
    save(fig, figs, "fig1_distributions")


def fig2_transfer(art, figs):
    f = "vision_0"
    stype = C.SIM_TYPE[f]
    raw = art.raw[f].astype(np.float64)
    lo_mask = ~art.same_traj
    pool_lo = raw[lo_mask]
    pool_rp = C.pool_random_pair(raw)
    pz = C.fit_zscore(pool_lo, stype)
    pl = C.fit_legacy_percentile(pool_rp, stype)
    x_or = C.orient(raw, stype)
    z_all = (x_or - pz["mu"]) / pz["sigma"]
    # decision region: per-query top-10 z
    zm = np.where(art.same_traj, -np.inf, z_all)
    ztop = -np.sort(-zm, axis=1)[:, :10].ravel()

    zg = np.linspace(-4.2, 4.6, 800)
    xg = pz["mu"] + zg * pz["sigma"]  # oriented = cosine here
    leg_curve = np.clip((C.map_legacy(xg, stype) - pl["p5"]) / (pl["p95"] - pl["p5"]), 0, 1)
    tanh_curve = 0.5 * (np.tanh(zg) + 1)
    clip3 = np.clip(0.5 + zg / 6, 0, 1)
    srt = np.sort(C.orient(pool_lo, stype))
    ecdf_curve = np.searchsorted(srt, xg, side="right") / len(srt)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(6.4, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 3.2], "hspace": 0.08})
    ax0.hist(z_all[lo_mask], bins=140, density=True, color=MUTED, alpha=0.45, lw=0)
    ax0.hist(ztop, bins=60, density=True, color=BLUE, alpha=0.6, lw=0)
    ax0.set_yticks([])
    ax0.grid(False)
    ax0.annotate("all LOEO scores", xy=(-2.6, 0.28), color=SEC, fontsize=8)
    ax0.annotate("top-10 per query\n(decision region)", xy=(2.5, 0.28), color=BLUE, fontsize=8)
    ax0.set_title(f"Layer-1 maps vs where decisions happen — {art.suite}/{art.builder}, {f}")

    ax1.plot(zg, leg_curve, color=RED, lw=2, label="legacy percentile clip")
    ax1.plot(zg, tanh_curve, color=BLUE, lw=2, label=r"$\frac{1}{2}(\tanh z+1)$")
    ax1.plot(zg, clip3, color=YELLOW, lw=1.6, ls="--", label="hard clip ±3σ")
    ax1.plot(zg, ecdf_curve, color=AQUA, lw=1.6, ls=":", label="frozen empirical CDF")
    z_hi = np.interp(1.0 - 1e-12, leg_curve, zg)
    ax1.axvspan(z_hi, 4.6, color=RED, alpha=0.10, lw=0)
    ax1.annotate("percentile flat exactly\nwhere top-10 mass lives", xy=(3.1, 0.42),
                 color=RED, fontsize=8.5, ha="center")
    ax1.set_xlabel("standardized score z = (x − μ)/σ")
    ax1.set_ylabel("normalized score ŝ")
    ax1.set_ylim(-0.04, 1.06)
    ax1.legend(loc="upper left", fontsize=8)
    save(fig, figs, "fig2_transfer_decision_region")


def fig3_factorial(B, figs):
    cells = ["legacy@RP", "legacy@LOEO", "zscore@RP", "zscore@LOEO"]
    colors = [RED, RED, BLUE, BLUE]
    alphas = [1.0, 0.55, 0.55, 1.0]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    xpos = 0
    ticks, labels = [], []
    for suite, builder in C.COMBOS:
        key = f"{suite}/{builder}"
        for i, cell in enumerate(cells):
            m = B[key]["metrics"][f"{cell}|w_prod"]["top1_same_task"]
            ax.bar(xpos, m["mean"], width=0.82, color=colors[i], alpha=alphas[i], lw=0)
            ax.errorbar(xpos, m["mean"], yerr=[[m["mean"] - m["lo"]], [m["hi"] - m["mean"]]],
                        color=INK, lw=1.0, capsize=2)
            ax.annotate(f"{m['mean']:.2f}", xy=(xpos, m["lo"] - 0.035), ha="center",
                        fontsize=7.5, color=SEC)
            xpos += 1
        ticks.append(xpos - 2.5)
        labels.append(key.replace("cp1_", "").replace("/", "\n"))
        xpos += 1.0
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("top-1 same-task accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("2×2 factorial: normalizer family dominates, calibration pool is irrelevant")
    handles = [plt.Rectangle((0, 0), 1, 1, color=RED, alpha=1.0),
               plt.Rectangle((0, 0), 1, 1, color=RED, alpha=0.55),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.55),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=1.0)]
    ax.legend(handles, cells, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    save(fig, figs, "fig3_factorial")


FAMILY_COLOR = {
    "zscore@LOEO": BLUE, "zscore@RP": BLUE, "ecdf@LOEO": AQUA,
    "rawz@LOEO": VIOLET, "affine_clip@LOEO": YELLOW, "norm2_mix@LOEO": YELLOW,
    "logit_mix@LOEO": GREEN, "power_mix@LOEO": GREEN, "dirunify": MUTED,
    "legacy@RP": RED, "legacy@LOEO": RED,
}


def fig4_family(B, figs):
    order = ["zscore@LOEO", "ecdf@LOEO", "rawz@LOEO", "power_mix@LOEO", "logit_mix@LOEO",
             "affine_clip@LOEO", "norm2_mix@LOEO", "dirunify", "legacy@RP", "legacy@LOEO"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
    for ax, (suite, builder) in zip(axes, [SECOND, PRIMARY]):
        key = f"{suite}/{builder}"
        for yi, cname in enumerate(order):
            m = B[key]["metrics"][f"{cname}|w_prod"]["top1_same_task"]
            col = FAMILY_COLOR[cname]
            ax.plot([m["lo"], m["hi"]], [yi, yi], color=col, lw=1.6)
            ax.plot(m["mean"], yi, "o", color=col, ms=6)
        for f in C.FIELDS:
            v = B[key]["metrics"][f"single_{f}"]["top1_same_task"]["mean"]
            ax.axvline(v, color=GRID, lw=0.9, zorder=0)
            ax.annotate(f.replace("robot_state", "state").replace("vision_", "v"),
                        xy=(v, len(order) - 0.4), fontsize=7, color=MUTED,
                        ha="center", rotation=90)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("top-1 same-task accuracy (95% cluster CI)")
        ax.set_title(key, fontsize=9.5)
        ax.grid(axis="x")
    fig.suptitle("Normalizer family head-to-head (production weights); gridlines = single-field baselines",
                 y=1.03, fontsize=10.5, fontweight="bold")
    save(fig, figs, "fig4_family")


def fig5_dose_response(Cres, figs):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, (suite, builder) in zip(axes, [SECOND, PRIMARY]):
        key = f"{suite}/{builder}"
        pts = Cres[key]["C1b"]
        ks = [p["k"] for p in pts]
        vals = [p["top1_same_task"] for p in pts]
        tanh_v = Cres[key]["C1"]["tanh"]["top1_same_task"]
        mp_key = key.replace("spatial_pool_16", "mean_pool")
        mp = Cres[mp_key]["C1b"]
        ax.plot([p["k"] for p in mp], [p["top1_same_task"] for p in mp],
                color=MUTED, lw=1.1, alpha=0.7)
        ax.plot(ks, vals, color=YELLOW, lw=2, marker="o", ms=4.5, label="hard clip ±kσ")
        ax.axhline(tanh_v, color=BLUE, lw=1.6, ls="--", label="tanh (no censoring)")
        ax.set_xlabel("clip half-width k (σ units)")
        ax.set_title(key, fontsize=9.5)
        ax.annotate("mean_pool", xy=(ks[2], mp[2]["top1_same_task"] - 0.06),
                    color=MUTED, fontsize=7.5)
    axes[0].set_ylabel("top-1 same-task accuracy")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Censoring dose–response: retrieval quality vs clip band width; tanh is the k→∞ limit",
                 y=1.03, fontsize=10.5, fontweight="bold")
    save(fig, figs, "fig5_dose_response")


def fig6_influence(Cres, figs):
    key = f"{PRIMARY[0]}/{PRIMARY[1]}"
    c3 = Cres[key]["C3"]["w_prod"]
    c3b = Cres[key]["C3b"]
    deltas = c3["tanh"]["deltas"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for name, col in (("tanh", BLUE), ("hardclip3", YELLOW), ("identity", VIOLET)):
        ax0.plot(deltas, c3[name]["median_rank_of_corrupted"], color=col, lw=2,
                 marker="o", ms=4.5, label=name)
    ax0.set_yscale("log")
    ax0.set_xlabel("corruption depth δ (σ units, downward)")
    ax0.set_ylabel("median rank of corrupted candidate")
    ax0.set_title("Bounded influence: rank plateaus vs sinks")
    ax0.legend(fontsize=8.5)
    key2 = f"{SECOND[0]}/{SECOND[1]}"
    c3b2 = Cres[key2]["C3b"]
    ax1.plot(deltas, c3b2["identity"]["decoy_win_rate"], color=VIOLET, lw=2,
             marker="o", ms=4.5, label="identity (libero_spatial)")
    ax1.plot(deltas, c3b["identity"]["decoy_win_rate"], color=VIOLET, lw=2,
             marker="o", ms=4.5, alpha=0.45, label="identity (libero_10)")
    ax1.plot(deltas, c3b["tanh"]["decoy_win_rate"], color=BLUE, lw=2, marker="o",
             ms=4.5, label="tanh (both suites)")
    bounded_max = np.max(
        [c3b[n]["decoy_win_rate"] for n in c3b if n != "identity"]
        + [c3b2[n]["decoy_win_rate"] for n in c3b2 if n != "identity"], axis=0)
    ax1.plot(deltas, bounded_max, color=GREEN, lw=1.4, ls="--",
             label="max over all bounded squashes")
    ax1.set_xlabel("decoy inflation δ (σ units, upward)")
    ax1.set_ylabel("decoy wins top-1 (false hit rate)")
    ax1.set_title("Decoy attack: unbounded maps get bought")
    ax1.legend(fontsize=8.5)
    fig.suptitle(f"Single-field corruption, {key}, production weights", y=1.04,
                 fontsize=10.5, fontweight="bold")
    save(fig, figs, "fig6_influence")


def fig7_missdetect(D, figs):
    key = f"{PRIMARY[0]}/{PRIMARY[1]}"
    key2 = f"{SECOND[0]}/{SECOND[1]}"
    cfg_col = {
        "zscore_tanh": BLUE, "probit": GREEN, "hardclip3": YELLOW,
        "identity_z": VIOLET, "legacy_percentile": RED, "ecdf_frozen": AQUA,
        "rank_perquery": AQUA,
    }
    order = ["rank_perquery", "identity_z", "ecdf_frozen", "zscore_tanh",
             "probit", "hardclip3", "legacy_percentile"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for yi, cname in enumerate(order):
        for k, mk in ((key, "o"), (key2, "s")):
            a = D[k]["configs"][cname]["auc_hit_vs_miss"]
            ax0.plot(a, yi, mk, color=cfg_col[cname], ms=6,
                     mfc=cfg_col[cname] if k == key else "white")
    ax0.set_yticks(range(len(order)))
    ax0.set_yticklabels(order, fontsize=8.5)
    ax0.invert_yaxis()
    ax0.set_xlabel("AUC: hit-regime vs miss-regime top-1 score")
    ax0.set_title("Abstention separability (● libero_10, □ libero_spatial)")

    sc = D[key]["scaling"]
    ns = np.array(sc["sizes"], dtype=float)
    hit_tanh = D[key]["configs"]["zscore_tanh"]["mean_top1_hit"]
    hit_rank = D[key]["configs"]["rank_perquery"]["mean_top1_hit"]
    ax1.plot(ns, hit_tanh - np.array(sc["zscore_tanh"]), color=BLUE, lw=2, marker="o",
             ms=4, label="zscore+tanh corridor")
    ax1.plot(ns, hit_rank - np.array(sc["rank_perquery"]), color=AQUA, lw=2, marker="o",
             ms=4, label="per-query rank corridor")
    ax1.plot(ns, 1.0 - (1.0 - 1.0 / (ns + 1)), color=MUTED, lw=1.2, ls="--",
             label="uniform-max law 1/(n+1)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("miss-regime library size n")
    ax1.set_ylabel("hit − miss top-1 gap")
    ax1.set_title("Threshold corridor vs library size")
    ax1.legend(fontsize=8)
    save(fig, figs, "fig7_missdetect")


def fig8_pit(art, figs):
    f = "vision_0"
    stype = C.SIM_TYPE[f]
    raw = art.raw[f].astype(np.float64)
    pool = C.pool_loeo(raw, art.same_traj)
    x = C.orient(pool, stype)
    pz = C.fit_zscore(pool, stype)
    pl = C.fit_legacy_percentile(pool, stype)
    lo_b, hi_b = band_in_oriented(pl["p5"], pl["p95"], stype)
    xs = np.linspace(np.percentile(x, 0.02), x.max() + 0.15 * pz["sigma"], 700)
    unif_cdf = np.clip((xs - lo_b) / (hi_b - lo_b), 0, 1)
    logi_cdf = 0.5 * (np.tanh((xs - pz["mu"]) / pz["sigma"]) + 1)
    srt = np.sort(x)
    e_cdf = np.searchsorted(srt, xs, side="right") / len(srt)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.2, 4.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08})
    ax0.hist(x, bins=150, density=True, color=MUTED, alpha=0.5, lw=0)
    ax0.set_yticks([])
    ax0.grid(False)
    ax0.set_title("One normalization = one distributional model (PIT view)")
    ax1.plot(xs, unif_cdf, color=RED, lw=2, label="Uniform[p5,p95] CDF = percentile clip")
    ax1.plot(xs, logi_cdf, color=BLUE, lw=2,
             label="Logistic(μ, σ/2) CDF = ½(tanh z + 1)")
    ax1.plot(xs, e_cdf, color=AQUA, lw=1.6, ls=":", label="empirical CDF (rank)")
    ax1.set_xlabel("oriented raw score x")
    ax1.set_ylabel("ŝ = model CDF(x)")
    ax1.legend(fontsize=8, loc="upper left")
    save(fig, figs, "fig8_pit")


ORANGE = "#eb6834"  # rank-relative fusion family (RRF / Borda)
RRF_KS = [1, 5, 10, 20, 60, 240, 1000]


def fig9_rrf(E, figs):
    combos = [f"{s}/{b}" for s, b in C.COMBOS]
    short = {c: c.replace("libero_", "").replace("cp1_", "").replace("spatial_pool_16", "sp16")
             .replace("mean_pool", "mean") for c in combos}
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12.6, 3.4))

    # (a) paired delta rrf_k - zscore at production weights, with CIs
    xpos = np.arange(len(RRF_KS) + 1)
    for ci, key in enumerate(combos):
        row = E[key]["ksweep"]["w_prod"]
        ds = [row[f"rrf{k}"]["delta_vs_zscore"] for k in RRF_KS] + [row["borda"]["delta_vs_zscore"]]
        los = [row[f"rrf{k}"]["lo"] for k in RRF_KS] + [row["borda"]["lo"]]
        his = [row[f"rrf{k}"]["hi"] for k in RRF_KS] + [row["borda"]["hi"]]
        main = key == f"{PRIMARY[0]}/{PRIMARY[1]}"
        a = 1.0 if main else 0.45
        ax0.plot(xpos, ds, color=ORANGE, lw=2 if main else 1.2, marker="o", ms=4, alpha=a)
        if main:
            ax0.fill_between(xpos, los, his, color=ORANGE, alpha=0.15, lw=0)
        ax0.annotate(short[key], xy=(xpos[-1] + 0.12, ds[-1]), fontsize=7, color=ORANGE,
                     alpha=a, va="center")
    ax0.axhline(0, color=BLUE, lw=1.4, ls="--")
    ax0.annotate("zscore+tanh parity", xy=(0.1, 0.002), color=BLUE, fontsize=8)
    ax0.set_xticks(xpos)
    ax0.set_xticklabels([str(k) for k in RRF_KS] + ["∞\n(Borda)"], fontsize=8)
    ax0.set_xlabel("RRF k (reciprocal shape 1/(k+r))")
    ax0.set_ylabel("Δ top-1 same-task vs zscore+tanh")
    ax0.set_title("Reciprocal shape, fixed weights")

    # (b) per-method maxima over the 153-point weight simplex
    methods = ["zscore", "rrf60", "borda"]
    mcol = {"zscore": BLUE, "rrf60": ORANGE, "borda": ORANGE}
    for ci, key in enumerate(combos):
        for mi, m in enumerate(methods):
            v = E[key]["sweep"][m]["max"]
            mk = "o" if m != "borda" else "^"
            ax1.plot(ci + (mi - 1) * 0.18, v, mk, color=mcol[m], ms=7,
                     mfc=mcol[m] if m != "rrf60" else "white")
    ax1.set_xticks(range(len(combos)))
    ax1.set_xticklabels([short[c] for c in combos], fontsize=8)
    ax1.set_ylabel("top-1 same-task, best simplex weights")
    ax1.set_title("Per-space tuned weights: near parity")
    handles = [plt.Line2D([], [], color=BLUE, marker="o", ls="", label="zscore+tanh"),
               plt.Line2D([], [], color=ORANGE, marker="o", mfc="white", ls="", label="RRF k=60"),
               plt.Line2D([], [], color=ORANGE, marker="^", ls="", label="Borda")]
    ax1.legend(handles=handles, fontsize=8, loc="lower right")

    # (c) P(top-1 correct | fused margin quintile)
    for key in combos:
        acc = E[key]["margin"]["acc_by_margin_quintile"]
        main = "spatial" in key.split("/")[0]
        ax2.plot(range(1, 6), acc, color=BLUE, lw=2 if main else 1.2,
                 alpha=1.0 if main else 0.4, marker="o", ms=4)
        ax2.annotate(short[key], xy=(5.06, acc[-1]), fontsize=7, color=BLUE,
                     alpha=1.0 if main else 0.5, va="center")
    ax2.set_xticks(range(1, 6))
    ax2.set_xlabel("fused-margin quintile (small → large)")
    ax2.set_ylabel("P(top-1 same-task)")
    ax2.set_title("Margins carry information ranks discard")
    fig.suptitle("Rank-relative fusion vs fixed pointwise maps (expE)", y=1.04,
                 fontsize=11, fontweight="bold")
    save(fig, figs, "fig9_rrf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--figs", required=True)
    args = ap.parse_args()
    data = Path(args.data)
    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)

    A = json.loads((data / "expA_results.json").read_text())
    B = json.loads((data / "expB_results.json").read_text())
    Cres = json.loads((data / "expC_results.json").read_text())
    D = json.loads((data / "expD_results.json").read_text())

    art_p = C.load_artifact(*PRIMARY, Path(args.cache_dir))
    fig1_distributions(art_p, A, figs)
    fig2_transfer(art_p, figs)
    fig3_factorial(B, figs)
    fig4_family(B, figs)
    fig5_dose_response(Cres, figs)
    fig6_influence(Cres, figs)
    fig7_missdetect(D, figs)
    fig8_pit(art_p, figs)

    epath = data / "expE_results.json"
    if epath.exists():
        fig9_rrf(json.loads(epath.read_text()), figs)


if __name__ == "__main__":
    main()
