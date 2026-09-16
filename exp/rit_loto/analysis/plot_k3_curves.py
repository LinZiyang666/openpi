"""K=3 ladder curves q_a(s) per suite, x restricted to the dense cluster near s=1 (not committed).

usage: uv run python exp/rit_loto/analysis/plot_k3_curves.py
x range = [25% quantile of the LOTO scores, max]; the histogram underneath shows where the decisions are.
Solid = LOTO-all fit, dashed = shadow refit; grey band = noise floor D(ref1, ref2) median..p95;
dotted vertical = warm75 cut of the K=2 IR-70 arm addressed from the LOTO fit.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from exp.rit_loto import fit_loto as fl  # noqa: E402
from exp.rit_pareto.rit_k import predict  # noqa: E402
from exp.libero_groot.emit_rit_arms import WARM_TS  # noqa: E402
from exp.robocasa365 import rit_cost_rc as rc  # noqa: E402

F = pathlib.Path("exp/rit_loto/analysis/figures")
COL = {"full": "tab:red", "warm75": "tab:orange", "warm50": "tab:green"}
LAB = {"full": "FULL (t=1, 8 steps)", "warm75": "warm75 (t=0.75, 2 steps left)", "warm50": "warm50 (t=0.5, 4 steps left)"}
CUT_K2_IR70 = {"libero_spatial": 0.98925, "libero_10": 0.998217}

fig, axes = plt.subplots(1, 2, figsize=(19, 6.5))
for ax, suite in zip(axes, ("libero_spatial", "libero_10")):
    D = pathlib.Path(f"exp/rit_loto/data/{suite}")
    f = json.load(open(D / "fits.json")); nf = json.load(open(D / "noise_floor.json"))
    cost = fl._cost_from_json(f["cost"])
    fits = {src: fl.deserialize_fit(f[src]["fits"]["3"], cost, WARM_TS) for src in ("loto_all", "shadow")}
    s_all = np.asarray(f["loto_all"]["s_sample"], float)
    lo, hi = np.quantile(s_all, 0.25), s_all.max()
    grid = np.linspace(lo, hi, 800)
    ax2 = ax.twinx()
    ax2.hist(s_all[(s_all >= lo) & (s_all <= hi)], bins=80, range=(lo, hi), color="tab:blue", alpha=0.15)
    ax2.set_yticks([]); ax2.set_ylabel("LOTO decisions per bin", color="tab:blue", fontsize=8)
    for tier in ("full", "warm75", "warm50"):
        ax.plot(grid, predict(fits["loto_all"], grid, tier), color=COL[tier], lw=2.4, label=f"LOTO-all {LAB[tier]}")
        ax.plot(grid, predict(fits["shadow"], grid, tier), color=COL[tier], lw=1.3, ls="--", alpha=0.85, label=f"shadow {tier}")
    ax.axhspan(nf["d_ref1_ref2"]["median"], nf["d_ref1_ref2"]["p95"], color="0.5", alpha=0.15, label="noise floor D(ref1,ref2) median..p95")
    ax.axvline(CUT_K2_IR70[suite], color="k", ls=":", lw=1.2, label=f"K=2 IR70 warm75 cut = {CUT_K2_IR70[suite]:.5f}")
    ax.set_xlim(lo, hi); ax.set_ylim(7.4, 8.7)
    ax.set_xlabel("s = cp1 similarity score"); ax.set_ylabel("q_a(s): 95% quantile of D given s  (alpha=0.05)")
    # Right axis: the predicted IR of the K=3 LOTO ladder addressed at delta = this height
    # (cuts_for(delta) over the LOTO score sample, measured stage costs) -- a monotone relabelling of y.
    d_lo, d_hi = rc._endpoints(fits["loto_all"])
    dgrid = np.linspace(min(d_lo, 7.4), max(d_hi, 8.7), 400)
    irs = np.array([rc._ir_at(fits["loto_all"], s_all, float(d), cost) for d in dgrid])
    # explicit twin axis with the same y limits: tick at the smallest delta reaching each IR level
    ax3 = ax.twinx(); ax3.spines["right"].set_position(("axes", 1.07)); ax3.set_ylim(ax.get_ylim())
    ticks, labels = [], []
    for ir in (90, 80, 70, 60, 50, 40, 30, 20):
        hit = np.nonzero(irs <= ir)[0]
        if hit.size:
            ticks.append(float(dgrid[hit[0]])); labels.append(f"{ir}")
    ax3.set_yticks(ticks); ax3.set_yticklabels(labels)
    ax3.set_ylabel("predicted IR (%) when delta = this height  (LOTO K=3 ladder, measured cost)")
    for y in ticks:
        ax.axhline(y, color="0.6", lw=0.5, ls="-.", alpha=0.6)
    ax.set_title(f"{suite}  K=3 ladder,  s in [25% quantile {lo:.4f}, max {hi:.4f}]  (75% of decisions; median s = {np.median(s_all):.4f})", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="upper right")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
fig.tight_layout()
fig.savefig(F / "k3_curves.png", dpi=140); fig.savefig(F / "k3_curves.pdf")
print("wrote", F / "k3_curves.png")
