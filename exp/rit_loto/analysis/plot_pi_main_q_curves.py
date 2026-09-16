"""q curves of the pi0.5 x LIBERO RIT main experiment (K=3, not committed).

Read straight from exp/rit_pareto/data/k3_rith/<suite>/k3/export_record_rith.json (the fit is stored with knots + q).
Knots are equal-frequency (24 segments), so knot[6] is the 25% score quantile and each segment holds 1/24 of the
shadow rows; the grey bars are that equal-mass density (the shadow table itself is not in the repo).
Horizontal dashed lines: delta of the main-run arms at IR 30 / 50 / 70; dotted verticals: their thetas (cuts).
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from exp.rit_pareto.rit_k import fit_from_record, predict  # noqa: E402

F = pathlib.Path("exp/rit_loto/analysis/figures")
COL = {"full": "tab:red", "warm03": "tab:orange", "warm05": "tab:green"}
LAB = {"full": "FULL (t=1)", "warm03": "warm03 (t=0.3, 7 of 10 steps left)", "warm05": "warm05 (t=0.5, 5 of 10 steps left)"}
PRE = {"libero_spatial": "sp", "libero_10": "l10"}
ARM_IRS = (30, 50, 70)
ARM_LS = {30: (0, (6, 2)), 50: (0, (3, 2)), 70: (0, (1, 2))}

fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
for ax, suite in zip(axes, ("libero_spatial", "libero_10")):
    rec = json.load(open(f"exp/rit_pareto/data/k3_rith/{suite}/k3/export_record_rith.json"))
    r = rec["rit"]; fit = fit_from_record(r["fit"]); knots = np.asarray(r["fit"]["knots"])
    lo, hi = knots[6], knots[-1]
    grid = np.linspace(lo, hi, 800)
    ax2 = ax.twinx()
    dens = 1.0 / (24 * np.diff(knots)); ax2.bar(knots[:-1], dens, width=np.diff(knots), align="edge", color="tab:blue", alpha=0.15)
    ax2.set_yticks([]); ax2.set_ylabel("shadow score density (equal-mass knot segments)", color="tab:blue", fontsize=8)
    for t in fit.tiers:
        ax.plot(grid, predict(fit, grid, t.name), color=COL[t.name], lw=2.4, label=LAB[t.name])
    for ir in ARM_IRS:
        a = r["arms"][f"k3_{PRE[suite]}_rit_ir{ir}"]
        th = a["thetas"]
        ax.axhline(a["delta"], color="k", lw=1, ls=ARM_LS[ir], label=f"arm IR{ir}: delta={a['delta']:.3f}, thetas={ {k: (None if v is None or v == float('inf') else round(v, 4)) for k, v in th.items()} }")
        for v in th.values():
            if v is not None and np.isfinite(v) and lo <= v <= hi:
                ax.axvline(v, color="k", lw=0.8, ls=ARM_LS[ir], alpha=0.7)
    ys = np.concatenate([predict(fit, grid, t.name) for t in fit.tiers])
    ax.set_xlim(lo, hi); ax.set_ylim(ys.min() - 0.15, ys.max() + 0.4)
    ax.set_xlabel("s = cp1 similarity score"); ax.set_ylabel("q_a(s), alpha=0.05  (pi0.5 metric scale, ref_mode tau1)")
    ax.set_title(f"pi0.5 MAIN RUN  {suite}  K=3  (n_rows={rec['n_rows']}, {rec['n_episodes']} shadow ep; x in [25% quantile {lo:.4f}, max {hi:.4f}])", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="upper right")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
fig.tight_layout()
fig.savefig(F / "pi05_main_run_q_curves.png", dpi=140); fig.savefig(F / "pi05_main_run_q_curves.pdf")
print("wrote", F / "pi05_main_run_q_curves.png")
