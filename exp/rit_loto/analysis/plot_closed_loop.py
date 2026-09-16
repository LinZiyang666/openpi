"""Closed-loop overlay for the LOTO verification arm (not committed; reads json/jsonl products only).

usage: uv run python exp/rit_loto/analysis/plot_closed_loop.py
Panel: warm75 tier of libero_10 -- LOTO K=2 curve (the arm's calibration), shadow K=2 curve, the closed-loop
joint refit (base rows), closed-loop D points of dispatched WARM_START decisions and counterfactual MISS rows,
binned empirical 95% quantiles of the dispatched points, the arm's cut and delta.
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

D = pathlib.Path("exp/rit_loto/data/libero_10")
F = pathlib.Path("exp/rit_loto/analysis/figures")
f = json.load(open(D / "fits.json")); v = json.load(open(D / "verify" / "verify.json")); fr = json.load(open(D / "frozen_run.json"))
cost = fl._cost_from_json(f["cost"])
k = 2; warm_ts = tuple(WARM_TS[: k - 1])
fits = {"LOTO-all K=2 (arm calibration)": fl.deserialize_fit(f["loto_all"]["fits"]["2"], cost, warm_ts),
        "shadow K=2 (refit)": fl.deserialize_fit(f["shadow"]["fits"]["2"], cost, warm_ts),
        "closed-loop joint refit (2000 base rows)": fl.deserialize_fit(v["joint_refit"]["fit"], cost, warm_ts)}
rows = [json.loads(l) for l in open(D / "verify" / "verify_rows.jsonl")]
warm = [r for r in rows if r["hit_type"] == "WARM_START"]; miss = [r for r in rows if r["hit_type"] == "MISS"]
sw = np.array([r["s"] for r in warm]); yw = np.array([r["y_rem2"] for r in warm])
sm = np.array([r["s"] for r in miss]); ym = np.array([r["y_rem2"] for r in miss])
cut = fr["arm"]["cuts"][1]; delta = fr["arm"]["delta"]
grid = np.linspace(0.9970, 0.9992, 400)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(sm, ym, s=5, color="0.7", alpha=0.5, label=f"MISS rows, counterfactual warm75 D (n={len(miss)})")
ax.scatter(sw, yw, s=6, color="tab:red", alpha=0.45, label=f"dispatched WARM_START rows, D_warm75 (n={len(warm)})")
for (name, fit), c in zip(fits.items(), ("tab:red", "tab:blue", "tab:green")):
    ax.plot(grid, predict(fit, grid, "warm75"), color=c, lw=2, label=name)
# binned empirical 95% quantile of the dispatched points (same 4 quantile bins as verify.json)
bins = v["exceedance"]["tiers"]["warm75"]["score_bins"]
for b in bins:
    m = (sw >= b["s_lo"]) & (sw <= b["s_hi"])
    if m.sum() >= 30:
        ax.hlines(np.quantile(yw[m], 0.95), b["s_lo"], b["s_hi"], color="k", lw=2.5)
        ax.text((b["s_lo"] + b["s_hi"]) / 2, 8.75 + 0.18 * (bins.index(b) % 2), f"exc {b['exceedance']:.3f} (n={b['n']})", ha="center", fontsize=7)
ax.hlines(delta, 0.9970, 0.9992, color="k", ls="--", lw=1, label=f"arm delta = {delta:.3f}")
ax.axvline(cut, color="k", ls=":", lw=1, label=f"warm75 cut = {cut:.5f}")
ax.set_xlim(0.9970, 0.9992); ax.set_ylim(5.4, 9.2)
ax.set_xlabel("s (cp1 score)"); ax.set_ylabel("D (Eq. 16, H_exec=5)")
ax.set_title("libero_10 closed loop under loto_k2_ir70 (warm75): black bars = empirical 95% quantile per score-quartile bin", fontsize=10)
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(F / "closed_loop_libero_10_warm75.png", dpi=130); fig.savefig(F / "closed_loop_libero_10_warm75.pdf")
print("wrote", F / "closed_loop_libero_10_warm75.png")
