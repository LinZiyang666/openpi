"""q curves of the MAIN GR00T x LIBERO RIT experiment (not committed).

The main run stored only knots + per-arm cuts (arm_record.json); the curves are reproduced by the same LP on the
same shadow rows with the recorded knots (fits.json['shadow'], audited: all 48 arms / 96 cuts reproduce). This plots
K=2 and K=3 per suite, x restricted to [25% quantile of the shadow scores, max]; horizontal dashed lines are the
delta of the main-run arms at IR 30 / 50 / 70, dotted verticals their cuts.
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

F = pathlib.Path("exp/rit_loto/analysis/figures")
COL = {"full": "tab:red", "warm75": "tab:orange", "warm50": "tab:green"}
LAB = {"full": "FULL (t=1)", "warm75": "warm75 (t=0.75, 2 steps left)", "warm50": "warm50 (t=0.5, 4 steps left)"}
PRE = {"libero_spatial": "sp", "libero_10": "l10"}
ARM_IRS = (30, 50, 70)
ARM_LS = {30: (0, (6, 2)), 50: (0, (3, 2)), 70: (0, (1, 2))}

fig, axes = plt.subplots(2, 2, figsize=(18, 11))
for i, suite in enumerate(("libero_spatial", "libero_10")):
    D = pathlib.Path(f"exp/rit_loto/data/{suite}")
    f = json.load(open(D / "fits.json")); cost = fl._cost_from_json(f["cost"])
    ar = json.load(open(f"exp/libero_groot/config/rit/{suite}/arm_record.json"))
    s_sh = np.asarray(f["shadow"]["s_sample"], float)
    lo, hi = np.quantile(s_sh, 0.25), s_sh.max()
    grid = np.linspace(lo, hi, 800)
    for j, k in enumerate((2, 3)):
        ax = axes[i, j]
        fit = fl.deserialize_fit(f["shadow"]["fits"][str(k)], cost, WARM_TS[: k - 1])
        ax2 = ax.twinx()
        ax2.hist(s_sh[(s_sh >= lo) & (s_sh <= hi)], bins=70, range=(lo, hi), color="tab:blue", alpha=0.15)
        ax2.set_yticks([]); ax2.set_ylabel("shadow decisions per bin", color="tab:blue", fontsize=8)
        for t in fit.tiers:
            ax.plot(grid, predict(fit, grid, t.name), color=COL[t.name], lw=2.4, label=f"{LAB[t.name]}")
        for ir in ARM_IRS:
            a = ar["arms"][f"{PRE[suite]}_rit_k{k}_ir{ir}"]
            ax.axhline(a["delta"], color="k", lw=1, ls=ARM_LS[ir], label=f"arm IR{ir}: delta={a['delta']:.3f}, cuts={[None if c is None else round(c, 4) for c in a['cuts']]}")
            for c in a["cuts"]:
                if c is not None and lo <= c <= hi:
                    ax.axvline(c, color="k", lw=0.8, ls=ARM_LS[ir], alpha=0.7)
        ax.set_xlim(lo, hi); ax.set_ylim(7.4, 8.8)
        ax.set_xlabel("s = cp1 similarity score"); ax.set_ylabel("q_a(s), alpha=0.05")
        ax.set_title(f"MAIN RUN  {suite}  K={k}  (shadow fit, n={f['shadow']['n_rows']}; x in [25% quantile {lo:.4f}, max {hi:.4f}])", fontsize=10)
        ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="upper right")
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
fig.tight_layout()
fig.savefig(F / "main_run_q_curves.png", dpi=140); fig.savefig(F / "main_run_q_curves.pdf")
print("wrote", F / "main_run_q_curves.png")
