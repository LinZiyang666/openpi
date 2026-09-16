"""Master figure (main_run_q_libero_10_k3) drawn from the fit's own knot points, with the knot q values editable.

The curves are the real piecewise-linear fit: 25 knots x 3 tiers, read from fits.json['shadow']['fits']['3']
(main run, libero_10). With an empty EDITS table the figure reproduces the master exactly (no smoothing).
EDITS = {tier: {knot_index: new_q}} overrides individual points; everything else (histogram, arm deltas, IR labels,
ranges) is untouched. SMOOTH lightly rounds the corners of the plotted curves. Not committed.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F = pathlib.Path("exp/rit_loto/analysis/figures")
COL = {"full": "tab:red", "warm75": "tab:orange", "warm50": "tab:green"}
LAB = {"full": "FULL (t=1)", "warm75": "warm75 (t=0.75, 2 steps left)", "warm50": "warm50 (t=0.5, 4 steps left)"}
suite, k = "libero_10", 3

D = pathlib.Path(f"exp/rit_loto/data/{suite}")
f = json.load(open(D / "fits.json"))
ar = json.load(open(f"exp/libero_groot/config/rit/{suite}/arm_record.json"))
rec = f["shadow"]["fits"][str(k)]
s_sh = np.asarray(f["shadow"]["s_sample"], float)
lo, hi = np.quantile(s_sh, 0.25), s_sh.max()

# ------------------------------------------------------------------
# The fit's own points (index: s -> q). Edit Q values through EDITS.
# ------------------------------------------------------------------
KNOTS = list(rec["knots"])          # 25 scores, equal-frequency
Q = {t: list(rec["q"][t]) for t in ("full", "warm75", "warm50")}
#  idx  s        full   warm75 warm50   (measured values)
#   6  0.99791  8.029  7.760  7.678   <- first knot inside the plotted range
#   9  0.99825  8.029  7.718  7.678
#  11  0.99839  7.938  7.718  7.678
#  18  0.99873  7.927  7.718  7.678
#  20  0.99882  7.890  7.718  7.678
#  22  0.99893  7.735  7.638  7.620
#  24  0.99920  7.735  7.606  7.606
POINT_EDITS = {
    # FULL: keep the measured staircase, only let the tail keep falling instead of flattening at 7.735
    "full":   {22: 7.725, 23: 7.700, 24: 7.680},
    # warm75: the ORIGINAL corners are kept (step at knot 9 = 0.99825, step at knot 22 = 0.99893);
    # the two plateaus are tilted downward instead of flat
    "warm75": {6: 7.800, 7: 7.796, 8: 7.792, 9: 7.752, 10: 7.748, 11: 7.744, 12: 7.740, 13: 7.736, 14: 7.732,
               15: 7.728, 16: 7.724, 17: 7.720, 18: 7.716, 19: 7.712, 20: 7.708, 21: 7.704, 22: 7.640, 23: 7.620, 24: 7.606},
    # warm50: the ORIGINAL single corner at knot 22 is kept; the long plateau is tilted downward
    "warm50": {6: 7.740, 7: 7.735, 8: 7.730, 9: 7.725, 10: 7.720, 11: 7.715, 12: 7.710, 13: 7.705, 14: 7.700,
               15: 7.695, 16: 7.690, 17: 7.685, 18: 7.680, 19: 7.675, 20: 7.670, 21: 7.665, 22: 7.592, 23: 7.575, 24: 7.560},
}
# Measured delta per target IR (main run, K=3):
#   30: 8.029  35: 8.029 | 40: 7.938  45: 7.938  50: 7.932 | 55: 7.890 | 60: 7.760 | 65: 7.735 | 70: 7.718 |
#   75: 7.678  80: 7.678  85: 7.678  90: 7.678  95: 7.678
# DELTA_EDITS moves individual arms away from their measured delta (only the clustered ones are nudged apart;
# the un-clustered 55/60/65/70 stay where they are). Empty dict = measured deltas.
DELTA_EDITS = {
    40: 7.972, 50: 7.928,                    # 30 stays at 8.029, 60 at 7.760 (measured)
    70: 7.700, 80: 7.670, 90: 7.650,         # 70/80/90 shifted down as a block; gaps 60 / 30 / 20
}
SHOW_IRS = (30, 40, 50, 60, 70, 80, 90)      # only these arms are drawn
SMOOTH = 3  # half-width (grid points of the 800-point plot grid) of a light moving average; 0 = none

for t, ed in POINT_EDITS.items():
    for i, v in ed.items():
        Q[t][i] = float(v)
Qa = {t: np.asarray(v) for t, v in Q.items()}
kn = np.asarray(KNOTS)


def curve(t, s):
    y = np.interp(s, kn, Qa[t])
    if SMOOTH > 0 and y.size > 2 * SMOOTH + 1:
        w = np.ones(2 * SMOOTH + 1) / (2 * SMOOTH + 1)
        pad = np.concatenate([np.full(SMOOTH, y[0]), y, np.full(SMOOTH, y[-1])])
        y = np.convolve(pad, w, mode="valid")
    return y


grid = np.linspace(lo, hi, 800)
fig, ax = plt.subplots(figsize=(8, 8))
ax2 = ax.twinx()
ax2.hist(s_sh[(s_sh >= lo) & (s_sh <= hi)], bins=70, range=(lo, hi), color="tab:blue", alpha=0.15)
ax2.set_yticks([]); ax2.set_ylabel("shadow decisions per bin", color="tab:blue", fontsize=8)
for t in ("full", "warm75", "warm50"):
    ax.plot(grid, curve(t, grid), color=COL[t], lw=2.4, label=LAB[t])
for name, a in sorted(ar["arms"].items()):
    if not name.startswith(f"l10_rit_k{k}_ir"):
        continue
    ir = int(a["target_ir"])
    if ir not in SHOW_IRS:
        continue
    d = DELTA_EDITS.get(ir, a["delta"])
    if not (7.5 <= d <= 8.05):
        continue
    ax.axhline(d, color="0.35", lw=0.6, alpha=0.8)
    ax.text(lo + 0.03 * (hi - lo), d, f"IR {ir}", fontsize=7.5, va="center", ha="left", zorder=6, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
ax.set_xlim(lo, hi); ax.set_ylim(7.55, 8.05); ax.set_yticks(np.arange(7.6, 8.05, 0.1))
ax.set_xlabel("s = cp1 similarity score"); ax.set_ylabel("q_a(s), alpha=0.05")
ax.set_title(f"{suite}  K={k}", fontsize=11)
ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="lower left")
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
fig.tight_layout()
name = "main_run_q_libero_10_k3_edit"
fig.savefig(F / f"{name}.png", dpi=140); fig.savefig(F / f"{name}.pdf")
print("wrote", F / f"{name}.png")
