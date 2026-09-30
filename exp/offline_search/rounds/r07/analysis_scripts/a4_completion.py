"""R7 completion checks, SELECTION §9.2 (rule-4 replicate, sparse) and §9.3 (rule 4 on dense libraries).

Outcomes come from accepted client journals (R6 `common.journal`), owner IR from decision counts in each arm's
`summary.json` cost ledger (pooled over decisions). Bootstrap: task-stratified init resampling, 10,000 draws; cell c
uses seed 20260930 + c (independent across cells), pooled contrasts average the cell draws with equal weight.
§9.2 averages each cell's two replicates per (task, init) before resampling (CT_rep1+CT_rep2)/2 − (CU_rep1+CU_rep2)/2.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'r06', 'analysis_scripts'))
import common as cm  # noqa: E402

RUN = 'r07_main'
ROOT = '/home/weiland/trace_runs/os_closed_loop'
SEED, DRAWS = 20260930, 10000
SPARSE = ['pi05_l10_50', 'pi05_spatial_50', 'groot_l10_50', 'groot_spatial_50']
DENSE = ['pi05_l10_500', 'pi05_spatial_500', 'groot_l10_500', 'groot_spatial_500']


def outcomes(arm):
    return cm.journal(f'{RUN}:{arm}')


def owner_ir(arm):
    c = json.load(open(f'{ROOT}/{RUN}/runs/{arm}/summary.json'))['cost_ledger']
    cv = .152 if arm.startswith('r7_pi05') else .148
    n = c['decisions']
    return cv * c['vision_decisions'] / n + (1 - cv) * c['misses'] / n


def mean_of(arms):
    runs = [outcomes(a) for a in arms]
    keys = set.intersection(*[set(r) for r in runs])
    return {k: float(np.mean([r[k] for r in runs])) for k in keys}


def boot(x, y, seed):
    keys = sorted(set(x) & set(y))
    tasks = sorted({k[0] for k in keys})
    d = {t: np.array([x[k] - y[k] for k in keys if k[0] == t], float) for t in tasks}
    n = sum(len(v) for v in d.values())
    rng = np.random.default_rng(seed)
    sims = np.zeros(DRAWS)
    for t in tasks:
        v = d[t]
        sims += v[rng.integers(0, len(v), size=(DRAWS, len(v)))].sum(axis=1)
    return float(np.mean(np.concatenate(list(d.values())))), sims / n, len(keys)


def ci(s):
    return float(np.quantile(s, .025)), float(np.quantile(s, .975))


def contrast(cells, ct_arms, cu_arms, rho_tag):
    rows, sims = [], []
    for i, cell in enumerate(cells):
        cts, cus = ct_arms(cell), cu_arms(cell)
        if not all(os.path.exists(f'{ROOT}/{RUN}/runs/{a}/summary.json') for a in cts + cus):
            rows.append(dict(cell=cell, missing=True))
            continue
        x, y = mean_of(cts), mean_of(cus)
        p, s, n = boot(x, y, SEED + i)
        sims.append(s)
        dir_pairs = [dict(ct=a, cu=b, ct_ir=owner_ir(a), cu_ir=owner_ir(b), dIR=owner_ir(a) - owner_ir(b))
                     for a, b in zip(cts, cus)]
        rows.append(dict(cell=cell, pairs=n, ct_sr=float(np.mean(list(x.values()))), cu_sr=float(np.mean(list(y.values()))),
                         dSR=p, ci=ci(s), ir_pairs=dir_pairs,
                         ir_matched=all(abs(q['dIR']) <= .015 for q in dir_pairs)))
    out = dict(rho=rho_tag, cells=rows)
    if len(sims) == len(cells):
        ps = np.mean(sims, axis=0)
        lo, hi = ci(ps)
        matched = all(r['ir_matched'] for r in rows)
        out.update(pooled_dSR=float(np.mean([r['dSR'] for r in rows])), pooled_ci=(lo, hi), all_ir_matched=matched,
                   verdict=('supported' if lo > 0 and matched else
                            'not supported (IR not matched in a cell)' if lo > 0 else 'not supported'))
    return out


def main():
    rep = contrast(SPARSE, lambda c: [f'r7_{c}_CT30', f'r7_{c}_CT30_rep2'], lambda c: [f'r7_{c}_CU30', f'r7_{c}_CU30_rep2'], .30)
    rep2_only = contrast(SPARSE, lambda c: [f'r7_{c}_CT30_rep2'], lambda c: [f'r7_{c}_CU30_rep2'], .30)
    dense = contrast(DENSE, lambda c: [f'r7_{c}_CT18'], lambda c: [f'r7_{c}_CU18'], .18)
    res = dict(sparse_two_replicates=rep, sparse_rep2_only=rep2_only, dense=dense)
    outp = os.path.join(os.path.dirname(__file__), '..', 'analysis_r7', 'a4_completion.json')
    json.dump(res, open(outp, 'w'), indent=1)
    for name, r in res.items():
        print(f"== {name} (rho {r['rho']}): pooled {r.get('pooled_dSR', float('nan'))*100:+.2f} pp "
              f"[{', '.join(f'{v*100:+.2f}' for v in r.get('pooled_ci', (float('nan'),)*2))}] -> {r.get('verdict', 'incomplete')}")
        for c in r['cells']:
            if c.get('missing'):
                print(f"   {c['cell']}: missing")
                continue
            irs = '; '.join(f"{q['ct_ir']:.4f}/{q['cu_ir']:.4f} dIR {q['dIR']:+.4f}" for q in c['ir_pairs'])
            print(f"   {c['cell']}: CT {c['ct_sr']:.3f} CU {c['cu_sr']:.3f} d {c['dSR']*100:+.2f} "
                  f"[{c['ci'][0]*100:+.2f}, {c['ci'][1]*100:+.2f}] IR {irs} matched={c['ir_matched']}")


if __name__ == '__main__':
    main()
