"""R11 opus: LOEO knob curves, target solves, stress sensitivity and offline value diagnostics. Library only.

Reads only out/anchor/*.npz (loeo_anchor_table.py). Writes out/curves.json, out/targets.json, out/OFFLINE_TABLES.md.

  taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 PYTHONPATH=.:src CUDA_VISIBLE_DEVICES= \
    .venv/bin/python -m exp.offline_search.rounds.r11.opus.offline_curves
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r11.opus import ir_model as M

OUT = Path(__file__).resolve().parent / "out"
GRID = [("pi05", "l10", 50), ("groot", "l10", 50), ("pi05", "spatial", 50), ("groot", "spatial", 50),
        ("pi05", "l10", 200), ("groot", "l10", 200), ("pi05", "l10", 500), ("groot", "l10", 500)]
CONTEXT = [("pi05", "spatial", 200), ("groot", "spatial", 200), ("pi05", "spatial", 500), ("groot", "spatial", 500)]
# proposed target owner IR levels per cell-size (arm grid); COMMON levels give comparable curves everywhere
TARGETS = {"pi05_l10_50": (.25, .32, .40), "groot_l10_50": (.25, .32, .40), "pi05_spatial_50": (.22, .30, .40),
           "groot_spatial_50": (.22, .30), "pi05_l10_200": (.25,), "groot_l10_200": (.25, .32),
           "pi05_l10_500": (.25,), "groot_l10_500": (.25,)}
COMMON = (.20, .22, .25, .28, .30, .32, .36, .40, .45)
STRESS = (0.75, 1.25)

METHODS = {
    "random": (lambda x: M.Random(x), 0.0, 1.0),                         # x = rho
    "periodic": (lambda x: M.DitheredGapCap(x), 0.0, 40.0),             # x = K (decreasing IR in K)
    "sd_total": (lambda x: M.SigmaDelta(x, scope="total"), 0.0, 1.0),   # x = F
}
VARIANTS = {
    "random_tail2": lambda x: M.Random(x, tail=2),
    "random_refr2": lambda x: M.Random(x, refractory=2),
    "random_nostep0": lambda x: M.Random(x, step0=False),
    "periodic_pgt1": lambda x: M.DitheredGapCap(x, post_guard_tail=1),
    "front1_periodic": lambda x: M.FrontLoad(1, M.DitheredGapCap(x)),
}
VARIANT_RANGE = {"periodic_pgt1": (0.0, 40.0), "front1_periodic": (0.0, 40.0)}


def solve(model, eps, make, lo, hi, target):
    """Monotone bisection; handles the decreasing-in-K periodic knob by orienting the bracket."""
    ir = lambda x: M.simulate(model, eps, make(x))["IR"]
    a, b = ir(lo), ir(hi)
    inc = b >= a
    if not inc:
        lo, hi, a, b = hi, lo, b, a
    if target <= a:
        return lo, a, "floor"
    if target >= b:
        return hi, b, "ceiling"
    for _ in range(26):
        mid = (lo + hi) / 2
        if ir(mid) < target:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2
    return x, ir(x), "ok"


def one_cell(cs):
    model, suite, size = cs
    name = M.cell_name(model, suite, size)
    eps = M.load_episodes(model, suite, size)
    base = M.base_stats(model, eps)
    base_sim = M.simulate(model, eps, M.Knob())
    stress = {str(fac): M.perturb_guard(eps, fac, seed=11) for fac in STRESS}
    res = dict(cell=name, model=model, suite=suite, size=size, base=base,
               base_diag={k: base_sim[k] for k in ("mean_max_cache_run", "p90_max_cache_run", "cache_runs_ge4_per_ep",
                                                    "capture_guard")},
               stress_base={k: M.base_stats(model, e) for k, e in stress.items()})
    # curves
    curves = {}
    for meth, (make, lo, hi) in METHODS.items():
        xs = np.linspace(lo, hi, 11) if meth != "periodic" else np.array([0, .25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8, 12])
        curves[meth] = [dict(x=float(x), **{k: v for k, v in M.simulate(model, eps, make(x)).items() if k != "knob"})
                        for x in xs]
    res["curves"] = curves
    # target solves (common levels + proposed levels)
    levels = sorted(set(COMMON) | set(TARGETS.get(name, ())))
    tgt = []
    for T in levels:
        for meth, (make, lo, hi) in METHODS.items():
            x, ir, status = solve(model, eps, make, lo, hi, T)
            sim = M.simulate(model, eps, make(x))
            row = dict(target=T, method=meth, x=x, status=status, proposed=T in TARGETS.get(name, ()),
                       cf_rho=M.solve_rho(model, base["v"], base["g"], T) if meth == "random" else None,
                       **{k: v for k, v in sim.items() if k != "knob"})
            # stress: same knob setting, guard rate scaled by .75 / 1.25 (closed-loop guard-rate mismatch)
            row["stress_IR"] = {k: M.simulate(model, e, make(x))["IR"] for k, e in stress.items()}
            tgt.append(row)
        if T in TARGETS.get(name, ()) or T in (.25, .32):
            for var, make in VARIANTS.items():
                lo, hi = VARIANT_RANGE.get(var, (0.0, 1.0))
                x, ir, status = solve(model, eps, make, lo, hi, T)
                sim = M.simulate(model, eps, make(x))
                tgt.append(dict(target=T, method=var, x=x, status=status, proposed=False,
                                **{k: v for k, v in sim.items() if k != "knob"},
                                stress_IR={k: M.simulate(model, e, make(x))["IR"] for k, e in stress.items()}))
    res["targets"] = tgt
    # value-profile diagnostics on the risk proxy (non-guard anchors)
    R = np.concatenate([e.risk for e in eps]); G = np.concatenate([e.guard for e in eps])
    idx = np.concatenate([np.arange(len(e.steps)) for e in eps])
    tn = np.concatenate([e.steps / e.n_slots for e in eps])
    mu = R[~G].mean()
    after1, after2, before1, lag1a, lag1b = [], [], [], [], []
    for e in eps:
        r, g = e.risk, e.guard
        lag1a += list(r[:-1]); lag1b += list(r[1:])
        for i in np.flatnonzero(g):
            if i + 1 < len(r) and not g[i + 1]:
                after1.append(r[i + 1])
            if i + 2 < len(r) and not g[i + 2]:
                after2.append(r[i + 2])
            if i >= 1 and not g[i - 1]:
                before1.append(r[i - 1])
    res["profile"] = dict(mean_risk_nonguard=float(mu), guard_over_mean=float(R[G].mean() / mu),
                          anchor_index_0to3=[float(R[(idx == i) & ~G].mean() / mu) for i in range(4)],
                          time_quintiles=[float(R[(tn >= a) & (tn < a + .2) & ~G].mean() / mu)
                                          for a in np.arange(0, 1, .2)],
                          lag1_autocorr=float(np.corrcoef(lag1a, lag1b)[0, 1]),
                          after_guard_1=float(np.mean(after1) / mu), after_guard_2=float(np.mean(after2) / mu),
                          before_guard_1=float(np.mean(before1) / mu))
    return res


def table(results):
    L = []
    L.append("## Base (knob off), library whole-episode-out model\n")
    L.append("| cell-size | episodes | lib success | mean slots | v | guard rate g | base IR | IR succ eps | IR failed eps | g at stress .75 / 1.25 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        b = r["base"]
        sb = r["stress_base"]
        L.append(f"| {r['cell']} | {b['n_ep']} | {b['success_frac']:.2f} | {b['mean_slots']:.1f} | {b['v']:.4f} | "
                 f"{b['g']:.3f} | {b['IR']:.3f} | {b['ir_success_eps']:.3f} | {b['ir_failed_eps']:.3f} | "
                 f"{sb['0.75']['g']:.3f} / {sb['1.25']['g']:.3f} |")
    L.append("\n## Knob curves (knob value -> anchor miss fraction f -> owner IR)\n")
    for r in results:
        L.append(f"**{r['cell']}** random rho: " + "; ".join(
            f"{c['x']:.1f}->{c['f']:.3f}/{c['IR']:.3f}" for c in r["curves"]["random"]))
        L.append("")
        L.append(f"**{r['cell']}** periodic K: " + "; ".join(
            f"{c['x']:g}->{c['f']:.3f}/{c['IR']:.3f}" for c in r["curves"]["periodic"]))
        L.append("")
    L.append("\n## Target solves (common levels): knob setting, predicted IR, stress IR at guard x.75 / x1.25\n")
    L.append("| cell-size | target | method | setting | pred IR | f | knob share of eligible | stress IR .75 | stress IR 1.25 | capture | mean max cache run | p90 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        for t in r["targets"]:
            L.append(f"| {r['cell']} | {t['target']:.2f}{'*' if t['proposed'] else ''} | {t['method']} | {t['x']:.3f} ({t['status']}) | "
                     f"{t['IR']:.3f} | {t['f']:.3f} | {t['knob_frac_of_eligible']:.3f} | {t['stress_IR']['0.75']:.3f} | "
                     f"{t['stress_IR']['1.25']:.3f} | {t['capture_knob']:.2f} | {t['mean_max_cache_run']:.2f} | "
                     f"{t['p90_max_cache_run']:.0f} |")
    L.append("\n## Risk-proxy profiles (LOEO policy-vs-corrected-cache error / mean over non-guard anchors)\n")
    L.append("| cell-size | mean err | at guard | anchors 0..3 | time quintiles | lag-1 autocorr | after guard +1 / +2 | before guard -1 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        p = r["profile"]
        L.append(f"| {r['cell']} | {p['mean_risk_nonguard']:.3f} | {p['guard_over_mean']:.2f} | "
                 f"{' '.join(f'{x:.2f}' for x in p['anchor_index_0to3'])} | {' '.join(f'{x:.2f}' for x in p['time_quintiles'])} | "
                 f"{p['lag1_autocorr']:.2f} | {p['after_guard_1']:.2f} / {p['after_guard_2']:.2f} | {p['before_guard_1']:.2f} |")
    return "\n".join(L) + "\n"


def main():
    cells = GRID + CONTEXT
    with mp.get_context("fork").Pool(min(12, len(cells))) as pool:
        results = pool.map(one_cell, cells)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "curves.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    (OUT / "OFFLINE_TABLES.md").write_text(table(results))
    print("wrote", OUT / "curves.json", OUT / "OFFLINE_TABLES.md")


if __name__ == "__main__":
    main()
