"""Slide figure: periodic vs distance-threshold model calls (R13). Reads only R13 astra outputs."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

A = Path(__file__).resolve().parents[1] / "astra"
R11 = Path(__file__).resolve().parents[2] / "r11" / "astra" / "analysis"
OUT = Path(__file__).resolve().parent
P_C, D_C, G_C = "#2a6f97", "#d1495b", "#b8b8b8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

fig, (ax2, ax, ax3) = plt.subplots(1, 3, figsize=(19, 3.8), gridspec_kw={"width_ratios": [1.45, 1, 1], "wspace": 0.62})

# (a) one recorded episode: call positions of both rules on the same states
d = pd.read_pickle(A / "mechanism_replayed.pkl")
x = d[(d.method == "distance_feedback") & (d.target == 0.32) & (d.cell == "groot_l10_50")]
lens = x.groupby("ep").size()
ep = lens[(lens >= 30) & (lens <= 40)].index[0]
e = x[x.ep == ep].sort_values("look")
k = e.look.values
for row, col in ((1, "replay_d"), (0, "replay_p")):
    on = e[col].values.astype(bool)
    guard = e.guard.values.astype(bool)
    ax2.scatter(k[on & ~guard], [row] * (on & ~guard).sum(), marker="|", s=520, lw=3, color=D_C if row else P_C)
    ax2.scatter(k[on & guard], [row] * (on & guard).sum(), marker="|", s=520, lw=3, color=G_C)
ax2.set_yticks([0, 1], ["periodic", "threshold\n+ feedback"])
ax2.set_ylim(-.6, 1.6)
ax2.set_xlim(k.min() - .5, k.max() + .5)
ax2.set_xlabel("decision index within episode (GR00T, LIBERO-10, library 50, budget 0.32)")
ax2.spines["left"].set_visible(False)
ax2.tick_params(axis="y", length=0)
ax2.set_title("(a) Model-call positions on the same states", loc="left", fontsize=12, pad=18)
ax2.text(1.0, 1.0, "grey = guard call (same in both rows)", transform=ax2.transAxes, ha="right", va="bottom", fontsize=8.5, color="#666")

# (b) paired success-rate difference at equal budget
rows = [r for r in csv.DictReader(open(A / "final_test_a_sr_summary.csv"))]
dev = next(csv.DictReader(open(A / "final_b_sr_summary.csv")))
names = {"groot_spatial_50": "GR00T Spatial", "groot_l10_50": "GR00T LIBERO-10",
         "pi05_spatial_50": "π0.5 Spatial", "pi05_l10_50": "π0.5 LIBERO-10", "pooled": "Test pooled (2000 pairs)"}
items = [(names[r["cell"]], float(r["difference"]), json.loads(r["ci95"]), r["cell"] == "pooled") for r in rows]
items.append(("Dev pooled (2800 pairs)", float(dev["difference"]), json.loads(dev["ci95"]), True))
for j, (lab, diff, (lo, hi), pooled) in enumerate(items[::-1]):
    ax.errorbar(100 * diff, j, xerr=[[100 * (diff - lo)], [100 * (hi - diff)]], fmt="o" if pooled else "s",
                color="k" if pooled else "#555", ms=7 if pooled else 5, capsize=3, lw=1.6 if pooled else 1.1)
    ax.text(100 * hi + 0.4, j, f"{100*diff:+.1f}", va="center", fontsize=9)
ax.axvline(0, color="#888", lw=1, ls="--")
ax.set_yticks(range(len(items)), [i[0] for i in items[::-1]])
ax.set_xlabel("success rate: threshold − periodic (pp, 95% CI)")
ax.set_xlim(-5, 6.5)
ax.set_title("(b) Success at equal budget (0.32)", loc="left", fontsize=12, pad=18)
ax.text(0, 1.0, "paired by initial state; achieved budget 0.31–0.33 for both", transform=ax.transAxes, fontsize=8.5, color="#666", va="bottom")

# (c) knob accuracy: achieved IR minus the knob's target IR, one point per arm
dev = {"periodic": [], "distance": [], "periodic13": [], "feedback13": []}
for r in csv.DictReader(open(R11 / "arms.csv")):
    if r["cohort"] == "test" and r["method"] in ("periodic", "distance"):
        dev[r["method"]].append(float(r["ir"]) - float(r["target"]))
for r in csv.DictReader(open(A / "final_b_budget.csv")):
    key = {"periodic": "periodic13", "distance_feedback": "feedback13"}.get(r["method"])
    if key: dev[key].append(float(r["ir"]) - float(r["target"]))
for r in csv.DictReader(open(A / "final_test_a_arms.csv")):
    key = {"periodic": "periodic13", "distance_feedback": "feedback13"}.get(r["method"])
    if key and r["role"] == "lean": dev[key].append(float(r["ir"]) - float(r["target"]))
rows3 = [("periodic13", "periodic (R13)", P_C), ("feedback13", "threshold + budget\nfeedback (R13)", D_C),
         ("periodic", "periodic (R11)", P_C), ("distance", "threshold, offline\ncalibrated (R11)", "#9a8c98")]
ax3.axvspan(-2, 2, color="#e8eef3", zorder=0)
ax3.axvline(0, color="#888", lw=1, ls="--")
for j, (key, lab, col) in enumerate(rows3[::-1]):
    v = [100 * e for e in dev[key]]
    jit = [((i * 0.6180339) % 1 - 0.5) * 0.36 for i in range(len(v))]
    ax3.scatter(v, [j + t for t in jit], s=22, color=col, alpha=0.85, edgecolor="none")
    ok = sum(abs(e) <= 2 for e in v)
    ax3.text(4.2, j, f"mean {sum(v)/len(v)/100:+.3f}\n{ok}/{len(v)} within ±.02", va="center", fontsize=8.5)
ax3.set_yticks(range(len(rows3)), [r[1] for r in rows3[::-1]])
ax3.set_xlim(-21, 4)
ax3.set_xticks([-20, -15, -10, -5, 0], ["−.20", "−.15", "−.10", "−.05", "0"])
ax3.set_xlabel("achieved IR − knob target IR (one point per arm)")
ax3.set_title("(c) Does the knob hit its budget?", loc="left", fontsize=12, pad=18)
ax3.text(0, 1.0, "shaded = ±.02; targets .25–.45", transform=ax3.transAxes, fontsize=8.5, color="#666", va="bottom")

for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"periodic_vs_threshold.{ext}", dpi=250, bbox_inches="tight")
print("episode", ep, "looks", len(e))
