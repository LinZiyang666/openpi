"""Single square panel: MAIN RUN (GR00T x LIBERO) libero_10 K=3 q curves (not committed).

Same content as the lower-right panel of main_run_q_curves.png.
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
ARM_IRS = (30, 50, 70)
ARM_LS = {30: (0, (6, 2)), 50: (0, (3, 2)), 70: (0, (1, 2))}
suite, k = "libero_10", 3

D = pathlib.Path(f"exp/rit_loto/data/{suite}")
f = json.load(open(D / "fits.json")); cost = fl._cost_from_json(f["cost"])
ar = json.load(open(f"exp/libero_groot/config/rit/{suite}/arm_record.json"))
s_sh = np.asarray(f["shadow"]["s_sample"], float)
lo, hi = np.quantile(s_sh, 0.25), s_sh.max()
grid = np.linspace(lo, hi, 800)
fit = fl.deserialize_fit(f["shadow"]["fits"][str(k)], cost, WARM_TS[: k - 1])

fig, ax = plt.subplots(figsize=(8, 8))
ax2 = ax.twinx()
ax2.hist(s_sh[(s_sh >= lo) & (s_sh <= hi)], bins=70, range=(lo, hi), color="tab:blue", alpha=0.15)
ax2.set_yticks([]); ax2.set_ylabel("shadow decisions per bin", color="tab:blue", fontsize=8)
for t in fit.tiers:
    ax.plot(grid, predict(fit, grid, t.name), color=COL[t.name], lw=2.4, label=LAB[t.name])
for ir in ARM_IRS:
    a = ar["arms"][f"l10_rit_k{k}_ir{ir}"]
    ax.axhline(a["delta"], color="k", lw=1, ls=ARM_LS[ir], label=f"arm IR{ir}: delta={a['delta']:.3f}, cuts={[None if c is None else round(c, 4) for c in a['cuts']]}")
    for c in a["cuts"]:
        if c is not None and lo <= c <= hi:
            ax.axvline(c, color="k", lw=0.8, ls=ARM_LS[ir], alpha=0.7)
ax.set_xlim(lo, hi); ax.set_ylim(7.5, 8.05)
ax.set_xlabel("s = cp1 similarity score"); ax.set_ylabel("q_a(s), alpha=0.05")
ax.set_title(f"MAIN RUN  {suite}  K={k}  (shadow fit, n={f['shadow']['n_rows']})\nx in [25% quantile {lo:.4f}, max {hi:.4f}]", fontsize=10)
ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="lower left")
# every main-run K=3 arm: its delta as a thin line, labelled with the target IR at the right edge
pairs = sorted((a["delta"], int(a["target_ir"])) for name, a in ar["arms"].items() if name.startswith(f"l10_rit_k{k}_ir"))
groups = []  # merge arms whose deltas sit within 0.008 of each other into one label
for d, ir in pairs:
    if groups and abs(d - groups[-1][0]) < 0.008:
        groups[-1][1].append(ir)
    else:
        groups.append([d, [ir]])
for d, irs_ in groups:
    if not (7.5 <= d <= 8.05):
        continue
    ax.axhline(d, color="0.35", lw=0.6, alpha=0.8)
    ax.text(lo + 0.03 * (hi - lo), d, "IR " + "/".join(str(v) for v in sorted(irs_)), fontsize=7.5, va="center", ha="left",
            zorder=6, clip_on=False, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
fig.tight_layout()
fig.savefig(F / "main_run_q_libero_10_k3.png", dpi=140); fig.savefig(F / "main_run_q_libero_10_k3.pdf")
print("wrote", F / "main_run_q_libero_10_k3.png")
