"""Slide figure: periodic vs distance-threshold model calls (R13). Reads only R13 astra outputs."""
import csv, collections, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

A = Path(__file__).resolve().parents[1] / "astra"
OUT = Path(__file__).resolve().parent
P_C, D_C, S_C, G_C = "#2a6f97", "#d1495b", "#9a8c98", "#b8b8b8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(16.5, 5.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1], height_ratios=[1.7, 1], hspace=0.10, wspace=0.62)

# (a) one recorded episode: the budget-fed threshold saw-tooths like a clock
d = pd.read_pickle(A / "mechanism_replayed.pkl")
x = d[(d.method == "distance_feedback") & (d.target == 0.32) & (d.cell == "groot_l10_50")]
lens = x.groupby("ep").size()
ep = lens[(lens >= 30) & (lens <= 40)].index[0]
e = x[x.ep == ep].sort_values("look")
k = e.look.values
ax = fig.add_subplot(gs[0, 0])
ax.plot(k, e.u.clip(upper=1.2), color="k", lw=1.6, label="retrieval distance")
ax.step(k, e.theta, where="mid", color=D_C, lw=1.6, label="threshold (budget-fed)")
ax.set_ylabel("distance / threshold")
ax.set_xticklabels([])
ax.set_xlim(k.min() - .5, k.max() + .5)
ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
ax.set_title("(a) One episode, same states, two rules", loc="left", fontsize=12, pad=30)
ax.set_ylim(-0.5, 1.95)
ax.set_yticks([0, 0.5, 1, 1.5])
ax.text(0, 1.0, "call when distance > threshold; the threshold falls every decision without a call and jumps after one,\nso the budget ledger turns it into a timer",
        transform=ax.transAxes, fontsize=8.8, color=D_C, ha="left", va="bottom")
ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
for row, col, label in ((1, "replay_d", "threshold"), (0, "replay_p", "periodic")):
    on = e[col].values.astype(bool)
    guard = e.guard.values.astype(bool)
    ax2.scatter(k[on & ~guard], [row] * (on & ~guard).sum(), marker="|", s=420, lw=3, color=D_C if row else P_C)
    ax2.scatter(k[on & guard], [row] * (on & guard).sum(), marker="|", s=420, lw=3, color=G_C)
ax2.set_yticks([0, 1], ["periodic", "threshold\n+ feedback"])
ax2.set_ylim(-.6, 1.6)
ax2.set_xlabel("decision index within episode (GR00T, LIBERO-10, library 50, budget 0.32)")
ax2.spines["left"].set_visible(False)
ax2.tick_params(axis="y", length=0)
ax2.text(1.0, 1.02, "grey = guard call (same states in both rows)", transform=ax2.transAxes, ha="right", va="bottom", fontsize=8.5, color="#666")

# (b) gap between consecutive calls, development set, budget 0.32, all cells pooled
cnt = collections.defaultdict(collections.Counter)
for r in csv.DictReader(open(A / "final_b_gap_cdf.csv")):
    if r["target"] == "0.32" and r["kind"] == "any" and r["unit"] == "real_looks" and int(r["gap"]) > 0:
        cnt[r["method"]][min(int(r["gap"]), 6)] += int(r["count"])
ax = fig.add_subplot(gs[:, 1])
w = 0.27
for i, (m, col, lab) in enumerate((("periodic", P_C, "periodic"), ("distance_feedback", D_C, "threshold + feedback"),
                                   ("distance_static_b", S_C, "fixed threshold"))):
    tot = sum(cnt[m].values())
    mean = sum(g * c for g, c in cnt[m].items()) / tot
    ax.bar([g + (i - 1) * w for g in range(1, 7)], [cnt[m][g] / tot for g in range(1, 7)], w, color=col,
           label=f"{lab}\n(mean gap {mean:.2f})")
ax.set_xticks(range(1, 7), ["1", "2", "3", "4", "5", "6+"])
ax.set_xlabel("decisions between consecutive model calls")
ax.text(0, 1.0, "dev set, all cells pooled", transform=ax.transAxes, fontsize=8.8, color="#666", va="bottom")
ax.set_ylabel("share of calls")
ax.legend(frameon=False, fontsize=9, loc="upper right", labelspacing=0.9)
ax.set_title("(b) Call rhythm at equal budget (0.32)", loc="left", fontsize=12, pad=30)

# (c) paired success-rate difference at equal budget
rows = [r for r in csv.DictReader(open(A / "final_test_a_sr_summary.csv"))]
dev = next(csv.DictReader(open(A / "final_b_sr_summary.csv")))
names = {"groot_spatial_50": "GR00T Spatial", "groot_l10_50": "GR00T LIBERO-10",
         "pi05_spatial_50": "π0.5 Spatial", "pi05_l10_50": "π0.5 LIBERO-10", "pooled": "Test pooled (2000 pairs)"}
items = [(names[r["cell"]], float(r["difference"]), json.loads(r["ci95"]), r["cell"] == "pooled") for r in rows]
items.append(("Dev pooled (2800 pairs)", float(dev["difference"]), json.loads(dev["ci95"]), True))
ax = fig.add_subplot(gs[:, 2])
for j, (lab, diff, (lo, hi), pooled) in enumerate(items[::-1]):
    ax.errorbar(100 * diff, j, xerr=[[100 * (diff - lo)], [100 * (hi - diff)]], fmt="o" if pooled else "s",
                color="k" if pooled else "#555", ms=7 if pooled else 5, capsize=3, lw=1.6 if pooled else 1.1)
    ax.text(100 * hi + 0.4, j, f"{100*diff:+.1f}", va="center", fontsize=9)
ax.axvline(0, color="#888", lw=1, ls="--")
ax.set_yticks(range(len(items)), [i[0] for i in items[::-1]])
ax.set_xlabel("success rate: threshold − periodic (pp, 95% CI)")
ax.set_xlim(-5, 6.5)
ax.text(0, 1.0, "paired by initial state", transform=ax.transAxes, fontsize=8.8, color="#666", va="bottom")
ax.set_title("(c) Success at equal budget", loc="left", fontsize=12, pad=30)
ax.text(0.0, -0.15, "test: 4 cells × 500 paired inits; achieved budget 0.31–0.33 for both", transform=ax.transAxes, fontsize=8.5, color="#666")

fig.suptitle("Periodic vs. distance-threshold model calls: at equal inference budget, the threshold rule falls into the periodic rhythm and success is indistinguishable",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.07)
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"periodic_vs_threshold.{ext}", dpi=250, bbox_inches="tight")
print("episode", ep, "looks", len(e))
