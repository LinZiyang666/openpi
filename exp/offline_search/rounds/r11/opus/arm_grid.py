"""R11 opus: frozen knob settings for the proposed arm grid, solved on the library only, plus predictions.

Knob settings (rho for random, K for periodic, K for the post-guard-tail variant, rho for random tail-2) are
solved by bisection so that the library whole-episode-out model gives owner IR = target (ir_model.simulate).
Two IR predictions are written per arm:
  * pred_IR_lib: the library model (the frozen basis of the setting; equals the target up to bisection error);
  * pred_IR_informed: the same setting re-simulated with the library guard flags rescaled so that the knob-off
    base guard rate equals the closed-loop rate implied by R10's published 3-layer IR (out/r10_check.json).
    This uses a test-set-A *aggregate* for prediction only (disclosed); no setting depends on it.

  .venv/bin/python -m exp.offline_search.rounds.r11.opus.arm_grid
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

from exp.offline_search.rounds.r11.opus import ir_model as M
from exp.offline_search.rounds.r11.opus.offline_curves import solve

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PRIORITY = {"core": 1, "levels": 2, "mid": 3, "dense": 4, "variant": 5}
# (cell, target, method, priority group)
GRID = []
for cell, levels, mid in [("pi05_l10_50", (.25, .32, .40), .32), ("groot_l10_50", (.25, .32, .40), .32),
                          ("pi05_spatial_50", (.22, .30, .40), .30), ("groot_spatial_50", (.22, .30), .30)]:
    for T in levels:
        for meth in ("random", "periodic"):
            GRID.append((cell, T, meth, "core" if T == mid else "levels"))
    GRID.append((cell, mid, "periodic_pgt1", "variant"))
for cell in ("pi05_l10_50", "groot_l10_50"):
    GRID.append((cell, .32, "random_tail2", "variant"))
for cell, levels, grp in [("groot_l10_200", (.25, .32), "mid"), ("pi05_l10_200", (.25,), "dense"),
                          ("pi05_l10_500", (.25,), "dense"), ("groot_l10_500", (.25,), "dense")]:
    for T in levels:
        for meth in ("random", "periodic"):
            GRID.append((cell, T, meth, grp))

MAKERS = {"random": (lambda x: M.Random(x), 0.0, 1.0),
          "periodic": (lambda x: M.DitheredGapCap(x), 0.0, 40.0),
          "periodic_pgt1": (lambda x: M.DitheredGapCap(x, post_guard_tail=1), 0.0, 40.0),
          "random_tail2": (lambda x: M.Random(x, tail=2), 0.0, 1.0)}
KNOB_NAME = {"random": "rho", "periodic": "K", "periodic_pgt1": "K", "random_tail2": "rho"}


def one(args):
    cell, T, meth, grp = args
    model, suite, size = cell.split("_")[0], cell.split("_")[1], int(cell.split("_")[2])
    eps = M.load_episodes(model, suite, size)
    make, lo, hi = MAKERS[meth]
    x, ir, status = solve(model, eps, make, lo, hi, T)
    sim = M.simulate(model, eps, make(x), reps=16)
    chk = {r["cell"]: r for r in json.loads((OUT / "r10_check.json").read_text())}[cell]
    fac = chk["g_cl"] / chk["g_lib"]
    eps_i = M.perturb_guard(eps, fac, seed=23)
    sim_i = M.simulate(model, eps_i, make(x), reps=16)
    base_i = M.base_stats(model, eps_i)
    stress = {f: M.simulate(model, M.perturb_guard(eps, f, seed=11), make(x), reps=16)["IR"] for f in (.75, 1.25)}
    return dict(cell=cell, target=T, method=meth, priority=PRIORITY[grp], group=grp, knob=KNOB_NAME[meth],
                setting=round(x, 4), solve_status=status, pred_IR_lib=sim["IR"], pred_f=sim["f"],
                pred_knob_share_of_eligible=sim["knob_frac_of_eligible"], pred_guard_rate=sim["g"],
                pred_IR_informed=sim_i["IR"], informed_base_IR=base_i["IR"], informed_g=base_i["g"],
                stress_IR_g075=stress[.75], stress_IR_g125=stress[1.25],
                capture=sim["capture_knob"], mean_max_cache_run=sim["mean_max_cache_run"],
                p90_max_cache_run=sim["p90_max_cache_run"])


def main():
    with mp.get_context("fork").Pool(16) as pool:
        rows = pool.map(one, GRID)
    rows.sort(key=lambda r: (r["priority"], r["cell"], r["target"], r["method"]))
    (OUT / "arm_grid.json").write_text(json.dumps(rows, indent=1) + "\n")
    L = ["| # | priority | cell-size | target IR | method | knob | setting | pred IR (library) | pred IR (informed) | "
         "stress IR g x.75 / x1.25 | anchor miss frac | knob share of eligible | capture | mean / p90 max cache run |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | {r['priority']} | {r['cell']} | {r['target']:.2f} | {r['method']} | {r['knob']} | "
                 f"{r['setting']:.4f} | {r['pred_IR_lib']:.3f} | {r['pred_IR_informed']:.3f} | "
                 f"{r['stress_IR_g075']:.3f} / {r['stress_IR_g125']:.3f} | {r['pred_f']:.3f} | "
                 f"{r['pred_knob_share_of_eligible']:.3f} | {r['capture']:.2f} | {r['mean_max_cache_run']:.2f} / "
                 f"{r['p90_max_cache_run']:.0f} |")
    (OUT / "arm_grid.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
