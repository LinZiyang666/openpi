#!/usr/bin/env python3
"""Independent check and per-cell extraction of the final R6 frontier (ANALYSIS.md 6).

frontier_final/ was produced by ops/frontier_refresh.py; a byte-identical rerun is in analysis_r6/frontier_rerun/.
This script re-derives, with its own loaders: every eligible point's SR (accepted journal) and owner IR (cost_ledger),
the point-estimate Pareto flags per cell (pure references shared across the two library cells of a model/suite),
the NI lower bound of every configuration-C arm and of each cell's cheapest nominal-NI point versus L10 (DUAL L5
outcomes are read from frontier_final/outcomes.json), and where each C arm sits relative to the front.
Writes analysis_r6/frontier_check.json and frontier_tables.md.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402

FF = os.path.join(cm.R6, 'frontier_final')
CELLS = ['pi05_l10_50', 'pi05_l10_500', 'pi05_spatial_50', 'pi05_spatial_500',
         'groot_l10_50', 'groot_l10_500', 'groot_spatial_50', 'groot_spatial_500']


def main():
    rows = list(csv.DictReader(open(os.path.join(FF, 'frontier_points.csv'))))
    outcomes = json.load(open(os.path.join(FF, 'outcomes.json')))
    gaps = list(csv.DictReader(open(os.path.join(FF, 'cell_gaps.csv'))))
    nirows = {(r['id'], r['reference_L']): r for r in csv.DictReader(open(os.path.join(FF, 'noninferiority.csv')))}
    refs = list(csv.DictReader(open(os.path.join(FF, 'pure_references.csv'))))
    # 1. SR / IR recomputation
    checked = mism = 0
    bad = []
    for r in rows:
        if r['eligible'] != 'True' or r['id'].startswith('DUAL'):
            continue
        spec = r['id'].replace('/', ':')
        o = cm.journal(spec)
        sr_ok = abs(cm.sr(o) - float(r['SR'])) < 1e-12 and len(o) == int(r['n'])
        ir_ok = True
        if r['cost_source'].startswith('summary.cost_ledger'):
            ir_ok = abs(cm.owner_ir(spec, r['model']) - float(r['owner_IR'])) < 1e-9
        checked += 1
        if not (sr_ok and ir_ok):
            mism += 1
            bad.append(r['id'])
    # 2. Pareto per cell
    elig = [r for r in rows if r['eligible'] == 'True']
    pareto_mism = []
    tables, cpos = {}, []
    for cell in CELLS:
        model, suite = cell.split('_')[0], cell.split('_')[1]
        pts = [r for r in elig if r['cell'] == cell or r['cell'] == f'{model}_{suite}_reference']
        for r in pts:
            s, i = float(r['SR']), float(r['owner_IR'])
            dom = [q for q in pts if q is not r and float(q['SR']) >= s and float(q['owner_IR']) <= i and
                   (float(q['SR']) > s or float(q['owner_IR']) < i)]
            par = not dom
            if par != (r['pareto'] == 'True'):
                pareto_mism.append((cell, r['id']))
            r['_dom'] = sorted(dom, key=lambda q: (float(q['owner_IR']), -float(q['SR'])))
            r['_par'] = par
        front = sorted([r for r in pts if r['_par']], key=lambda q: float(q['owner_IR']))
        tables[cell] = [dict(id=r['id'], family=r['method_family'], SR=float(r['SR']), IR=float(r['owner_IR']),
                             lo=float(r['SR_lo']), hi=float(r['SR_hi'])) for r in front]
        for r in pts:
            if r['method_family'].startswith('configC') or r['method_family'] == 'episode_lottery_uniform':
                d = r['_dom'][0] if r['_dom'] else None
                cpos.append(dict(cell=cell, id=r['id'], family=r['method_family'], SR=float(r['SR']), IR=float(r['owner_IR']),
                                 pareto=r['_par'], dominated_by=(d['id'], float(d['SR']), float(d['owner_IR'])) if d else None,
                                 n_dominators=len(r['_dom'])))
    # 3. NI recomputation for C arms and the per-cell minima (vs L10 single reference; vs GR00T DUAL L5)
    def load(idx):
        if idx.startswith('DUAL'):
            return {tuple(int(x) for x in k.split(':')): bool(v) for k, v in outcomes[idx].items()}
        return cm.journal(idx.replace('/', ':'))
    refl10 = {(r['model'], r['suite']): json.loads(r['runs'])[0] for r in refs if r['L'] == '10'}
    refl5 = {(r['model'], r['suite']): json.loads(r['runs']) for r in refs if r['L'] == '5'}
    ni_check = []
    for r in elig:
        if not (r['method_family'].startswith('configC') or r['id'] in {g['minimum_NI'] for g in gaps}):
            continue
        if r['id'].startswith('REFERENCE') or r['pure'] == 'True':
            continue
        o = load(r['id'])
        m, s = r['model'], r['suite']
        own10 = cm.ni_lower(o, load(refl10[(m, s)]))[0]
        l5 = refl5[(m, s)]
        if len(l5) == 1:
            own5 = cm.ni_lower(o, load(l5[0]))[0]
        else:
            own5 = sum(cm.ni_lower(o, load(x), alpha=.05 / len(l5))[0] for x in l5) / len(l5)
        f10 = float(nirows[(r['id'], '10')]['paired_exact_lower95'])
        f5 = float(nirows[(r['id'], '5')]['paired_exact_lower95'])
        ni_check.append(dict(id=r['id'], cell=r['cell'], own10=own10, csv10=f10, own5=own5, csv5=f5,
                             sim10=float(nirows[(r['id'], '10')]['paired_exact_lower_simultaneous']),
                             sim5=float(nirows[(r['id'], '5')]['paired_exact_lower_simultaneous']),
                             agree=abs(own10 - f10) < 1e-9 and abs(own5 - f5) < 1e-9))
    out = dict(checked_points=checked, sr_ir_mismatches=mism, bad=bad, pareto_mismatches=pareto_mism,
               ni_check=ni_check, c_positions=cpos, fronts=tables, gaps=gaps)
    cm.dump('frontier_check.json', out)
    md = []
    for cell in CELLS:
        md.append(f'### {cell}\n')
        md.append('| Pareto point | family | SR [Wilson 95%] | owner IR |')
        md.append('|---|---|---|---|')
        for t in tables[cell]:
            md.append(f"| `{t['id']}` | {t['family']} | {t['SR']:.3f} [{t['lo']:.3f}, {t['hi']:.3f}] | {t['IR']:.4f} |")
        md.append('')
    with open(os.path.join(cm.OUT, 'frontier_tables.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(json.dumps(dict(checked_points=checked, sr_ir_mismatches=mism, bad=bad, pareto_mismatches=pareto_mism,
                          ni_disagree=[x for x in ni_check if not x['agree']]), indent=1))
    for c in cpos:
        print(c)


if __name__ == '__main__':
    main()
