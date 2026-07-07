#!/usr/bin/env python3
"""Figures for the external training-free gating report
(analysis/gate_report_external.md).

Run: uv run exp/gate_research/report_figures.py
Outputs 4 PNGs (+PDFs) into exp/gate_research/analysis/figures/.
All numbers hard-coded below are transcribed from the committed analysis
reports (n1_live_final.md, stage3_n4_live.md, stage4a_n2_live.md,
stage2_fair_pareto.md); fig2 is computed from the raw Stage-0
always-search rows (exp/gate_research/data/*/gate_rows.jsonl, gitignored --
fig2 needs them present locally).
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "exp/gate_research/analysis/figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Okabe–Ito colour-blind-safe palette
C = {
    "baseline": "#000000",
    "periodic": "#E69F00",   # orange
    "hyst":     "#0072B2",   # blue
    "hybrid":   "#009E73",   # green
    "fw":       "#CC79A7",   # magenta
    "pureinf":  "#7F7F7F",   # gray
    "replay":   "#A6CEE3",   # light blue  (cached replay)
    "infer":    "#FDBF6F",   # light orange (full inference)
    "blind":    "#CAB2D6",   # light purple (blind replay)
}
MARK = {"baseline": "*", "periodic": "s", "hyst": "o", "hybrid": "^", "fw": "D", "pureinf": "X"}
LABEL = {
    "baseline": "Always-search baseline",
    "periodic": "Periodic-refresh gate",
    "hyst": "Score-hysteresis gate",
    "hybrid": "Hybrid gate (hysteresis + refresh)",
    "fw": "Follow-the-winner replay gate",
    "pureinf": "No cache (inference every step)",
}

# =====================================================================
# Master data table  (SR %, inf_ratio, skip %, net@34 ms/step, ΔSR pp)
# =====================================================================
SPATIAL_BASE = dict(sr=82.6, inf=0.287)
L10_BASE = dict(sr=77.6, inf=0.636)

# fields: (label_suffix, skip, sr, inf, net34)
DATA = {
    "spatial": {
        "baseline": [("", 0.0, 82.6, 0.287, 0.0)],
        "hyst": [("A", 12.8, 85.6, 0.283, 5.6), ("B", 20.1, 87.8, 0.293, 5.1)],
        "periodic": [("1-in-8", 10.9, 90.4, 0.280, 5.8), ("1-in-5", 18.3, 89.0, 0.369, -18.1)],
        "hybrid": [("L=6", 17.2, 92.4, 0.289, 5.4), ("L=8", 16.7, 88.8, 0.300, 2.0), ("L=12", 15.1, 85.8, 0.303, 0.5)],
        "fw": [("b=3", 24.5, 85.4, 0.269, 14.0), ("b=5", 33.2, 84.4, 0.266, 17.8), ("b=8", 42.3, 84.2, 0.260, 22.7)],
    },
    "l10": {
        "baseline": [("", 0.0, 77.6, 0.636, 0.0)],
        "hyst": [("A", 21.2, 77.0, 0.642, 5.5), ("B", 32.4, 76.2, 0.655, 5.5)],
        "periodic": [("1-in-5", 19.2, 82.2, 0.701, -12.8), ("1-in-3", 32.7, 82.4, 0.759, -25.7),
                     ("1-in-13", 7.1, 78.4, 0.663, -5.7), ("1-in-31", 2.5, 77.6, 0.650, -3.4)],
        "hybrid": [("L=6", 21.8, 81.6, 0.658, 0.8), ("L=8", 21.6, 81.0, 0.652, 2.4), ("L=12", 21.2, 78.4, 0.648, 3.7)],
        "fw": [("b=3", 10.9, 78.8, 0.621, 8.2), ("b=5", 14.0, 74.4, 0.636, 4.9), ("b=8", 19.5, 75.6, 0.606, 15.8)],
        "pureinf": [("", 0.0, 83.0, 1.0, None)],
    },
}
SUITE_TITLE = {"spatial": "LIBERO-Spatial (short-horizon)", "l10": "LIBERO-Long (long-horizon)"}

# =====================================================================
# Figure 1 — mechanism timeline (synthetic illustration)
# =====================================================================
def fig1():
    N = 30
    # ground truth of the illustrative episode:
    # steps 0-9 cache reliable, 10-15 cache broken (search would MISS), 16-29 reliable
    MISS = set(range(10, 16))

    R, I, B = "replay", "infer", "blind"

    def row_always():
        ex, se = [], []
        for t in range(N):
            ex.append(I if t in MISS else R); se.append(True)
        return ex, se

    def row_periodic(period=5):
        ex, se = [], []
        for t in range(N):
            if t % period == period - 1:
                ex.append(I); se.append(False)          # forced fresh inference, no search
            else:
                ex.append(I if t in MISS else R); se.append(True)
        return ex, se

    def row_hyst():
        ex, se = [], []
        skipping = False; probe_in = 0
        for t in range(N):
            if not skipping:
                se.append(True)
                if t in MISS:
                    ex.append(I); skipping = True; probe_in = 3   # score collapsed -> stop searching
                else:
                    ex.append(R)
            else:
                probe_in -= 1
                if probe_in == 0:
                    se.append(True)                                # probe search
                    if t in MISS:
                        ex.append(I); probe_in = 3
                    else:
                        ex.append(R); skipping = False
                else:
                    se.append(False); ex.append(I)                 # skip search, straight to inference
        return ex, se

    def row_hybrid(L=6):
        ex, se = [], []
        skipping = False; probe_in = 0; run = 0
        for t in range(N):
            if not skipping and run >= L:                          # refresh injection
                ex.append(I); se.append(False); run = 0; continue
            if not skipping:
                se.append(True)
                if t in MISS:
                    ex.append(I); skipping = True; probe_in = 3; run = 0
                else:
                    ex.append(R); run += 1
            else:
                probe_in -= 1
                if probe_in == 0:
                    se.append(True)
                    if t in MISS:
                        ex.append(I); probe_in = 3
                    else:
                        ex.append(R); skipping = False; run = 1
                else:
                    se.append(False); ex.append(I)
        return ex, se

    def row_fw(lock=3, budget=5):
        ex, se = [], []
        streak = 0; blind_left = 0
        for t in range(N):
            if blind_left > 0:
                ex.append(B); se.append(False); blind_left -= 1; streak = 0; continue
            se.append(True)
            if t in MISS:
                ex.append(I); streak = 0
            else:
                ex.append(R); streak += 1
                if streak >= lock:
                    blind_left = budget; streak = 0
        return ex, se

    rows = [
        ("Always-search\n(baseline)", *row_always()),
        ("Periodic refresh\n(1-in-5)", *row_periodic()),
        ("Score-hysteresis", *row_hyst()),
        ("Hybrid\n(hysteresis + refresh L=6)", *row_hybrid()),
        ("Follow-the-winner\n(lock 3, budget 5)", *row_fw()),
    ]
    for name, ex, se in rows:
        assert len(ex) == N and len(se) == N, name

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    fill = {R: C["replay"], I: C["infer"], B: C["blind"]}
    for r, (name, ex, se) in enumerate(rows):
        y = len(rows) - 1 - r
        n_search = sum(se); n_infer = sum(1 for e in ex if e == I); n_blind = sum(1 for e in ex if e == B)
        for t in range(N):
            ax.add_patch(Rectangle((t, y), 0.92, 0.72, facecolor=fill[ex[t]],
                                   edgecolor="white", linewidth=0.6))
            if se[t]:
                ax.plot(t + 0.46, y + 0.36, marker=".", color="black", markersize=4)
        ax.text(-0.6, y + 0.36, name, ha="right", va="center", fontsize=9.5)
        ax.text(N + 0.4, y + 0.36, f"search {n_search}  ·  infer {n_infer}"
                + (f"  ·  free {n_blind}" if n_blind else ""),
                ha="left", va="center", fontsize=8.5, color="#333333")
    # shade the cache-broken window
    ax.add_patch(Rectangle((10, -0.55), 6, len(rows) + 0.5, facecolor="#D62728",
                           alpha=0.06, edgecolor="none", zorder=0))
    ax.text(13, len(rows) - 0.02, "cache unreliable\n(retrieval would miss)", ha="center",
            va="bottom", fontsize=8.5, color="#B22222")
    ax.set_xlim(-6.5, N + 7.5); ax.set_ylim(-0.7, len(rows) + 0.55)
    ax.axis("off")
    handles = [
        Patch(facecolor=C["replay"], label="execute cached actions (no inference)"),
        Patch(facecolor=C["infer"], label="full model inference"),
        Patch(facecolor=C["blind"], label="blind replay (no search, no inference)"),
        Line2D([0], [0], marker=".", color="black", linestyle="none", label="cache search performed"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, frameon=False, fontsize=8.5)
    ax.set_title("How each gate schedules one (illustrative) episode of 30 control steps",
                 pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_mechanisms.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig1_mechanisms.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig1 done")

# =====================================================================
# Figure 2 — the free signal (real data, Stage-0 always-search rows)
# =====================================================================
def _auc(pos, neg):
    """AUC via rank statistic (pos = scores of positives)."""
    x = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(x)) + 1.0
    # average ranks for ties
    order = np.argsort(x); xs = x[order]
    ranks = np.empty_like(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty_like(ranks); r[order] = ranks
    rp = r[:len(pos)]
    return (rp.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))

def _load_pairs(path, cfg_suffix):
    """(prev cp1_score, next-step is MISS) pairs for one config."""
    eps = defaultdict(list)
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("_kind") == "episode_summary":
                continue
            if not row.get("yaml_id", "").endswith(cfg_suffix):
                continue
            if row.get("phase") != "eval":
                continue
            eps[row["task_uid"]].append((row["step_idx"], row["cp1_score"], row["hit_type"]))
    prev, nxt = [], []
    for uid, steps in eps.items():
        steps.sort()
        for (s0, sc0, h0), (s1, sc1, h1) in zip(steps, steps[1:]):
            if sc0 is None:
                sc0 = 0.0
            prev.append(sc0); nxt.append(1 if h1 == "MISS" else 0)
    return np.asarray(prev), np.asarray(nxt)

def fig2():
    cfgs = [
        ("spatial", os.path.join(ROOT, "exp/gate_research/data/libero_spatial/gate_rows.jsonl"),
         "fh75_ws10_quantile", (0.945, 0.9805)),
        ("l10", os.path.join(ROOT, "exp/gate_research/data/libero_10/gate_rows.jsonl"),
         "fh5_ws40_quantile", (0.9945, 0.99925)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, (suite, path, suffix, xlim) in zip(axes, cfgs):
        prev, nxt = _load_pairs(path, suffix)
        pos = prev[nxt == 1]; neg = prev[nxt == 0]   # pos = next step misses
        auc = _auc(-pos, -neg)                        # lower score => miss
        bins = np.linspace(xlim[0], xlim[1], 70)
        pc = np.clip(pos, xlim[0], xlim[1]); ncl = np.clip(neg, xlim[0], xlim[1])
        ax.hist(ncl, bins=bins, density=True, alpha=0.65, color=C["replay"],
                label=f"next step: cache still valid (n={len(neg):,})")
        ax.hist(pc, bins=bins, density=True, alpha=0.65, color="#D62728",
                label=f"next step: cache misses (n={len(pos):,})")
        ax.set_xlim(*xlim)
        ax.set_yticks([])
        ax.set_xlabel("retrieval score of the previous search")
        ax.set_ylabel("density (shape only)")
        ax.set_title(f"{SUITE_TITLE[suite]}   —   AUC = {auc:.3f}")
        ax.legend(frameon=False)
        ax.annotate("lower scores pooled\ninto leftmost bin", xy=(xlim[0], 0), xycoords=("data", "axes fraction"),
                    xytext=(8, 30), textcoords="offset points", fontsize=7.8, color="#666666")
        print(f"fig2 {suite}: n={len(prev):,}  AUC={auc:.4f}  pos={len(pos):,}")
    fig.suptitle("The gating signal is free: the score of the previous cache search "
                 "almost perfectly predicts whether the next lookup will miss", y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_signal.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig2_signal.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")

# =====================================================================
# Figure 3 — main result: SR vs inference ratio
# =====================================================================
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, suite in zip(axes, ["spatial", "l10"]):
        d = DATA[suite]
        base_sr, base_inf = d["baseline"][0][2], d["baseline"][0][3]
        ax.axhline(base_sr, color="gray", lw=0.8, ls=":", zorder=0)
        ax.axvline(base_inf, color="gray", lw=0.8, ls=":", zorder=0)
        for fam in ["periodic", "hyst", "hybrid", "fw", "pureinf", "baseline"]:
            if fam not in d:
                continue
            pts = d[fam]
            xs = [p[3] for p in pts]; ys = [p[2] for p in pts]
            ax.scatter(xs, ys, marker=MARK[fam], s=110 if fam == "baseline" else 62,
                       color=C[fam], label=LABEL[fam], zorder=3,
                       edgecolor="white", linewidth=0.7)
        # annotations
        if suite == "spatial":
            ann = [("Hybrid L=6", 0.289, 92.4, (12, 6)), ("Periodic 1-in-8", 0.280, 90.4, (-8, 10)),
                   ("baseline", 0.287, 82.6, (10, -12)), ("Periodic 1-in-5", 0.369, 89.0, (-14, 9))]
            ax.set_xlim(0.245, 0.395); ax.set_ylim(80.5, 94.5)
        else:
            ann = [("Hybrid L=6", 0.658, 81.6, (12, -6)), ("baseline", 0.636, 77.6, (8, -14)),
                   ("no cache", 1.0, 83.0, (-30, -14)), ("Periodic 1-in-3", 0.759, 82.4, (-6, 10))]
            ax.set_xlim(0.575, 1.035); ax.set_ylim(72.5, 85.5)
        for txt, x, y, off in ann:
            ax.annotate(txt, (x, y), textcoords="offset points", xytext=off, fontsize=8.3,
                        color="#333333")
        ax.set_xlabel("inference ratio  (fraction of full-inference compute per step)")
        ax.set_ylabel("task success rate (%)")
        ax.set_title(SUITE_TITLE[suite])
    handles, labels = axes[1].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=3, frameon=False)
    fig.suptitle("Success rate vs. inference compute — every gated point, 500 episodes each", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_main.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig3_main.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig3 done")

# =====================================================================
# Figure 4 — latency/SR trade-off quadrants + dose-response
# =====================================================================
def fig4():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # (a)(b) quadrants
    for ax, suite in zip(axes[0], ["spatial", "l10"]):
        d = DATA[suite]; base_sr = d["baseline"][0][2]
        ax.axhline(0, color="gray", lw=0.8, ls=":"); ax.axvline(0, color="gray", lw=0.8, ls=":")
        for fam in ["periodic", "hyst", "hybrid", "fw"]:
            pts = d[fam]
            xs = [p[4] for p in pts]; ys = [p[2] - base_sr for p in pts]
            ax.scatter(xs, ys, marker=MARK[fam], s=62, color=C[fam], label=LABEL[fam],
                       zorder=3, edgecolor="white", linewidth=0.7)
        ax.set_xlabel("net latency saved per step (ms, standard stack)")
        ax.set_ylabel("Δ success rate vs. baseline (pp)")
        ax.set_title(SUITE_TITLE[suite])
        ax.annotate("better", xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                    fontsize=9, color="#2CA02C",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#EAF7EA", ec="none"))
        if suite == "spatial":
            ax.annotate("Hybrid L=6", (5.4, 9.8), textcoords="offset points", xytext=(8, 2), fontsize=8.3)
            ax.annotate("Periodic 1-in-5", (-18.1, 6.4), textcoords="offset points", xytext=(4, 6), fontsize=8.3)
            ax.annotate("Follow-winner b=8", (22.7, 1.6), textcoords="offset points", xytext=(-30, -14), fontsize=8.3)
        else:
            ax.annotate("Hybrid L=6", (0.8, 4.0), textcoords="offset points", xytext=(6, 5), fontsize=8.3)
            ax.annotate("Periodic 1-in-3", (-25.7, 4.8), textcoords="offset points", xytext=(4, 6), fontsize=8.3)
            ax.annotate("Follow-winner b=8", (15.8, -2.0), textcoords="offset points", xytext=(-36, -14), fontsize=8.3)
    # (c) hybrid dose-response
    ax = axes[1][0]
    Ls = [6, 8, 12]
    for suite, ls in [("spatial", "-"), ("l10", "--")]:
        base = DATA[suite]["baseline"][0][2]
        ys = [p[2] - base for p in DATA[suite]["hybrid"]]
        ax.plot(Ls, ys, marker=MARK["hybrid"], color=C["hybrid"], ls=ls,
                label=f"{SUITE_TITLE[suite].split(' (')[0]}")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(Ls); ax.set_xlabel("refresh interval L (steps of uninterrupted cache execution)")
    ax.set_ylabel("Δ success rate vs. baseline (pp)")
    ax.set_title("Hybrid gate: more frequent refresh → higher success")
    ax.legend(frameon=False)
    # (d) follow-winner dose-response
    ax = axes[1][1]
    Bs = [3, 5, 8]
    for suite, ls in [("spatial", "-"), ("l10", "--")]:
        base = DATA[suite]["baseline"][0][2]
        ys = [p[2] - base for p in DATA[suite]["fw"]]
        ax.plot(Bs, ys, marker=MARK["fw"], color=C["fw"], ls=ls,
                label=f"{SUITE_TITLE[suite].split(' (')[0]}")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(Bs); ax.set_xlabel("blind-replay budget b (steps replayed without any check)")
    ax.set_ylabel("Δ success rate vs. baseline (pp)")
    ax.set_title("Follow-the-winner: longer blind replay → drift hurts")
    ax.legend(frameon=False)
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.035),
               ncol=4, frameon=False)
    fig.suptitle("Latency–success trade-off (top) and dose–response of the two refresh-style gates (bottom)",
                 y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_tradeoff.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig4_tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")

if __name__ == "__main__":
    fig1()
    fig3()
    fig4()
    fig2()   # slowest last (reads 158 MB of jsonl)
    print("ALL FIGURES DONE ->", OUT)
