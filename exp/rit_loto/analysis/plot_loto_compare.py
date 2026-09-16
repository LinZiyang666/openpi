"""Overlay figures for the LOTO-vs-shadow comparison (not committed; reads only the json products).

usage: uv run python exp/rit_loto/analysis/plot_loto_compare.py <suite>
Panels: (a) per-tier q_a(s): LOTO-all vs shadow, with the shadow episode-bootstrap band;
        (b) LOTO in-library vs out-of-library; (c) empirical CDF of s (psi(s)) for the three row sets.
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

suite = sys.argv[1]
D = pathlib.Path(f"exp/rit_loto/data/{suite}")
F = pathlib.Path("exp/rit_loto/analysis/figures")
f = json.load(open(D / "fits.json"))
b = json.load(open(D / "bootstrap_band.json"))
cost = fl._cost_from_json(f["cost"])
k = "3"
band = b["bands"][k]
grid = np.asarray(band["grid"])
tiers = ["full", "warm75", "warm50"]
fits = {src: fl.deserialize_fit(f[src]["fits"][k], cost) for src in ("loto_all", "loto_in_library", "loto_out_of_library", "shadow")}
supp = {src: (f[src]["s_quantiles"]["0.0"], f[src]["s_quantiles"]["1.0"]) for src in fits}


def arr(v):
    return np.asarray([np.nan if x is None else x for x in v], float)


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for j, tier in enumerate(tiers):
    ax = axes[0, j]
    tb = band["tiers"][tier]
    ax.fill_between(grid, arr(tb["lo"]), arr(tb["hi"]), color="tab:blue", alpha=0.15, label="shadow bootstrap band (95% pointwise, B=200)")
    for src, c, ls in (("shadow", "tab:blue", "-"), ("loto_all", "tab:red", "-")):
        lo, hi = supp[src]
        g = grid[(grid >= lo) & (grid <= hi)]
        ax.plot(g, predict(fits[src], g, tier), color=c, ls=ls, lw=1.8, label=f"{src} (n={f[src]['n_rows']})")
    ax.set_title(f"{suite} K=3 tier={tier}: q_a(s) LOTO-all vs shadow")
    ax.set_xlabel("s (cp1 score)"); ax.set_ylabel("q_a(s), alpha=0.05")
    ax.set_xlim(max(0.85, grid.min()), grid.max()); ax.legend(fontsize=7)
    ax = axes[1, j]
    for src, c in (("loto_out_of_library", "tab:green"), ("loto_in_library", "tab:orange"), ("shadow", "tab:blue")):
        lo, hi = supp[src]
        g = grid[(grid >= lo) & (grid <= hi)]
        ax.plot(g, predict(fits[src], g, tier), color=c, lw=1.6, label=f"{src} (n={f[src]['n_rows']})")
    ax.set_title(f"tier={tier}: in-library vs out-of-library vs shadow")
    ax.set_xlabel("s (cp1 score)"); ax.set_ylabel("q_a(s)")
    ax.set_xlim(max(0.85, grid.min()), grid.max()); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(F / f"loto_vs_shadow_{suite}.png", dpi=130)
fig.savefig(F / f"loto_vs_shadow_{suite}.pdf")

# psi(s): empirical CDF of the score in each row set
fig2, ax = plt.subplots(figsize=(6, 4))
for src, c in (("shadow", "tab:blue"), ("loto_all", "tab:red"), ("loto_in_library", "tab:orange"), ("loto_out_of_library", "tab:green")):
    s = np.sort(np.asarray(f[src]["s_sample"], float))
    ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=c, lw=1.4, label=f"{src} (sample n={len(s)})")
ax.set_xlim(0.85, 1.0); ax.set_xlabel("s"); ax.set_ylabel("empirical CDF psi(s)"); ax.set_title(f"{suite}: score marginals")
ax.legend(fontsize=7); fig2.tight_layout()
fig2.savefig(F / f"psi_{suite}.png", dpi=130)
print("wrote", F / f"loto_vs_shadow_{suite}.png", F / f"psi_{suite}.png")
