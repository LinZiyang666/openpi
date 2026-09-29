#!/usr/bin/env python3
"""Configuration C validation against the acceptance rules fixed in SELECTION.md §4 (before any C outcome).

Every arm is paired on the same 500 (task, init) pairs. Differences use a task-stratified init bootstrap (10,000 draws,
seed 20260929): within each task the 50 inits are resampled with replacement. B is the per-pair mean of its three
replicates. Pooled contrasts average the four cell differences, and each cell is resampled independently.
Noninferiority uses Q2's conservative Clopper–Pearson paired bound, `CP_lower(wins) − CP_upper(losses)`, with
margin .02. Arms that have not finished are reported as missing.
"""
import importlib.util
import os
import sys

import numpy as np
from scipy.stats import beta

spec = importlib.util.spec_from_file_location('paper_ab_mod', os.path.join(os.path.dirname(__file__), 'paper_ab.py'))
pab = importlib.util.module_from_spec(spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]
_out, sys.stdout = sys.stdout, open(os.devnull, 'w')
spec.loader.exec_module(pab)
sys.stdout, sys.argv = _out, _argv

ROOT = pab.ROOT
V = 'r06_c_validation'
SPARSE = [('pi05', 'l10_50', 'pi05_l10_50'), ('pi05', 'sp_50', 'pi05_spatial_50'),
          ('groot', 'l10_50', 'groot_l10_50'), ('groot', 'sp_50', 'groot_spatial_50')]
DENSE = [('pi05', 'l10_500', 'pi05_l10_500'), ('pi05', 'sp_500', 'pi05_spatial_500'),
         ('groot', 'l10_500', 'groot_l10_500'), ('groot', 'sp_500', 'groot_spatial_500')]
RHO = {'U30': .30, 'R30': .30, 'C30': .30, 'C45': .45, 'U18': .18, 'R18': .18, 'C18': .18}
RNG = np.random.default_rng(20260929)
DRAWS = 10000


def exists(spec_):
    run, arm = spec_.split(':')
    return os.path.exists(f'{ROOT}/{run}/runs/{arm}/summary.json') and (
        os.path.exists(f'{ROOT}/{run}/state/{arm}.DONE')
        or any(f.startswith(arm + '.manifest_') and f.endswith('.DONE') for f in os.listdir(f'{ROOT}/{run}/state')))


def arm(cell, cfg):
    if cell == 'groot_l10_50' and cfg == 'C45':
        return f'{V}:r6c_groot_l10_50_Cmax'
    return f'{V}:r6c_{cell}_{cfg}'


def l10_ref(model, cell):
    s = 'l10' if cell.startswith('l10') else 'sp'
    return f'r04_cost:r4f_p_{s}_inf_k10_L10' if model == 'pi05' else f"r05_q2:r5q2_g_{'l10' if s == 'l10' else 'spatial'}_policy_L10"


def l5_refs(model, cell):
    s = 'l10' if cell.startswith('l10') else 'sp'
    return [f'r04_cost:r4f_p_{s}_inf_s1001', f'r04_cost:r4f_p_{s}_inf_s2001'] if model == 'pi05' else []


def outcomes(spec_list):
    """Per-(task, init) success, averaged over the given runs (one run = binary)."""
    loads = [pab.load(s) for s in spec_list]
    keys = set.intersection(*[set(o) for o, _ in loads])
    return {k: sum(o[k] for o, _ in loads) / len(loads) for k in keys}, [c for _, c in loads]


def owner_ir(model, ledger):
    return pab.C1[model] * ledger['v'] + (1 - pab.C1[model]) * ledger['m']


def boot_diff(x, y):
    """Point and 95% interval of mean(x − y), task-stratified init bootstrap."""
    keys = sorted(set(x) & set(y))
    tasks = sorted({k[0] for k in keys})
    d = {t: np.array([x[k] - y[k] for k in keys if k[0] == t]) for t in tasks}
    point = float(np.mean(np.concatenate(list(d.values()))))
    sims = np.zeros(DRAWS)
    n = sum(len(v) for v in d.values())
    for t, v in d.items():
        idx = RNG.integers(0, len(v), size=(DRAWS, len(v)))
        sims += v[idx].sum(axis=1)
    sims /= n
    return point, sims


def ci(sims):
    return float(np.quantile(sims, .025)), float(np.quantile(sims, .975))


def ni_bound(x, y, alpha=.05):
    keys = set(x) & set(y)
    n = len(keys)
    w = sum(1 for k in keys if x[k] > y[k])
    l = sum(1 for k in keys if y[k] > x[k])
    lo = beta.ppf(alpha / 2, w, n - w + 1) if w > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, l + 1, n - l) if l < n else 1.0
    return float(lo - hi), w, l


def pct(v):
    return f'{100 * v:+.1f}'


def main():
    out = []
    P = out.append
    P('## Arms\n')
    P('| cell | cfg | SR | owner IR | target ρ | IR − ρ | within ±.02 |')
    P('|---|---|---|---|---|---|---|')
    have = {}
    for model, cell, name in SPARSE + DENSE:
        for cfg in (['R30', 'U30', 'C30', 'C45'] if (model, cell, name) in SPARSE else ['R18', 'U18', 'C18']):
            a = arm(name, cfg)
            if not exists(a):
                P(f'| {name} | {cfg} | missing | | | | |')
                continue
            o, c = pab.load(a)
            ir = owner_ir(model, c)
            rho = RHO[cfg]
            if a.endswith('Cmax'):
                rho = .440704194944
            have[(name, cfg)] = (a, o, ir)
            P(f"| {name} | {cfg if not a.endswith('Cmax') else 'Cmax'} | {sum(o.values()) / len(o):.3f} | {ir:.3f} | {rho:.3f} | "
              f"{ir - rho:+.3f} | {'yes' if abs(ir - rho) <= .02 else '**no**'} |")

    P('\n## 1. Placement: R.30 − U.30 (sparse cells)\n')
    P('| cell | R30 SR | U30 SR | Δ pp [95%] | ΔIR |')
    P('|---|---|---|---|---|')
    pooled = []
    irok = True
    for model, cell, name in SPARSE:
        if (name, 'R30') in have and (name, 'U30') in have:
            (ar, orr, irr), (au, ou, iru) = have[(name, 'R30')], have[(name, 'U30')]
            pt, sims = boot_diff(orr, ou)
            lo, hi = ci(sims)
            pooled.append(sims)
            irok &= abs(irr - iru) <= .01
            P(f'| {name} | {np.mean(list(orr.values())):.3f} | {np.mean(list(ou.values())):.3f} | {pct(pt)} [{pct(lo)}, {pct(hi)}] | {irr - iru:+.3f} |')
        else:
            P(f'| {name} | missing | | | |')
    if len(pooled) == 4:
        ps = np.mean(pooled, axis=0)
        lo, hi = ci(ps)
        verdict = ('**supported**' if lo > 0 else '**falsified**' if hi < 0 else 'inconclusive')
        P(f'\nPooled R30 − U30 = {pct(float(np.mean(ps)))} pp [{pct(lo)}, {pct(hi)}]; all |ΔIR| ≤ .01: {irok}. '
          f'Verdict (§4): {verdict}{"" if irok else " (IR not matched within .01: frontier comparison only)"}.')

    P('\n## 2. Stall component: C.30 vs R.30, B and B-without-no-progress (sparse cells)\n')
    P('| cell | C30 SR | R30 SR | C30 − R30 pp [95%] | B (3-run) | B−noNP | ½ of B\'s NP benefit kept? | C30 − B−noNP pp [95%] |')
    P('|---|---|---|---|---|---|---|---|')
    for model, cell, name in SPARSE:
        if (name, 'C30') not in have:
            P(f'| {name} | missing | | | | | | |')
            continue
        _, oc, _ = have[(name, 'C30')]
        A, Bs = pab.arms(model, cell)
        ob, _ = outcomes(Bs)
        nonp = f"r06_abl:r6p2_no_progress_{pab_model(model)}_{cell}"
        on, _ = outcomes([nonp])
        sb, sn, sc = np.mean(list(ob.values())), np.mean(list(on.values())), np.mean(list(oc.values()))
        ptn, simn = boot_diff(oc, on)
        lon, hin = ci(simn)
        if (name, 'R30') in have:
            ptr, simr = boot_diff(oc, have[(name, 'R30')][1])
            lor, hir = ci(simr)
            cr = f'{pct(ptr)} [{pct(lor)}, {pct(hir)}]'
            sr_ = f"{np.mean(list(have[(name, 'R30')][1].values())):.3f}"
        else:
            cr, sr_ = 'R30 missing', ''
        keep = 'n/a (NP harmful here)' if sb - sn <= 0 else ('yes' if sc - sn >= .5 * (sb - sn) else 'no')
        P(f'| {name} | {sc:.3f} | {sr_} | {cr} | {sb:.3f} | {sn:.3f} | {keep} | {pct(ptn)} [{pct(lon)}, {pct(hin)}] |')

    P('\n## 3. Reach: C.45 (Cmax on GR00T L10-50) vs pure inference, 2 pp NI (conservative CP bound)\n')
    P('| cell | C SR @ IR | L10 SR | NI lower vs L10 (pp) | L5 SR | NI lower vs L5 (pp, mean of refs) |')
    P('|---|---|---|---|---|---|')
    for model, cell, name in SPARSE:
        if (name, 'C45') not in have:
            P(f'| {name} | missing | | | | |')
            continue
        a, oc, ir = have[(name, 'C45')]
        ol, _ = outcomes([l10_ref(model, cell)])
        b10, w, l = ni_bound(oc, ol)
        refs5 = l5_refs(model, cell)
        if refs5:
            bs = []
            for r in refs5:
                o5, _ = outcomes([r])
                bs.append(ni_bound(oc, o5, alpha=.05 / len(refs5))[0])
            s5 = np.mean([np.mean(list(outcomes([r])[0].values())) for r in refs5])
            l5s = f'{s5:.3f} | {100 * np.mean(bs):+.2f}'
        else:
            l5s = 'DUAL ref (outside os_closed_loop) | see frontier_repaired'
        P(f'| {name} | {np.mean(list(oc.values())):.3f} @ {ir:.3f} | {np.mean(list(ol.values())):.3f} | {100 * b10:+.2f} ({"NI" if b10 > -.02 else "not shown"}) | {l5s} |')

    P('\n## 4. Dense libraries: C.18 vs B at B\'s cost\n')
    P('| cell | C18 SR @ IR | B SR @ IR (3-run) | C18 − B pp [95%] | lower > −2 pp | IR ≤ B |')
    P('|---|---|---|---|---|---|')
    for model, cell, name in DENSE:
        if (name, 'C18') not in have:
            P(f'| {name} | missing | | | | |')
            continue
        a, oc, ir = have[(name, 'C18')]
        A, Bs = pab.arms(model, cell)
        ob, cs = outcomes(Bs)
        bir = float(np.mean([owner_ir(model, c) for c in cs]))
        pt, sims = boot_diff(oc, ob)
        lo, hi = ci(sims)
        P(f'| {name} | {np.mean(list(oc.values())):.3f} @ {ir:.3f} | {np.mean(list(ob.values())):.3f} @ {bir:.3f} | '
          f'{pct(pt)} [{pct(lo)}, {pct(hi)}] | {"yes" if lo > -.02 else "no"} | {"yes" if ir <= bir else "no"} |')
    print('\n'.join(out))


def pab_model(model):
    return 'p' if model == 'pi05' else 'g'


if __name__ == '__main__':
    main()
