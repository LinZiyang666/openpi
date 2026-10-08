"""Coordinator supplement to analyze.py: owner IR and extra pairings (measure only, no selection)."""
import json
from pathlib import Path

from .analyze import analyze, paired

OWNER = {'pi05': (.152, .848), 'groot': (.148, .852)}


def owner_ir(root, arm, model):
    led = json.loads((Path(root) / 'runs' / arm / 'summary.json').read_text())['cost_ledger']
    cv, cm = OWNER[model]
    return cv * led['v'] + cm * led['m'], led['decisions']


def main():
    audit = {(r['model'], r['suite'], r['size']): r
             for r in json.loads((Path(__file__).parent / 'audit.json').read_text())['records']}
    res = analyze()
    allv = {v: [] for v in ('A', 'A+C', 'G', 'GC_dist')}
    print('| cell | contrast | gains/losses | delta pp | exact p |\n|---|---|---|---|---|')
    for g in res:
        ys = {v: [y for c in g['cells'] for y in c['arms'][v]['y']] for v in allv}
        for v in allv:
            allv[v] += ys[v]
        for a, b in (('A+C', 'A'), ('G', 'A'), ('GC_dist', 'G'), ('GC_dist', 'A+C')):
            p = paired(ys[a], ys[b])
            print(f"| {g['model']} {g['suite']} | {a} vs {b} | +{p['gains']}/-{p['losses']} | "
                  f"{100 * p['delta_sr']:+.1f} | {p['p_two_sided_exact']:.3g} |")
    for a, b in (('A+C', 'A'), ('G', 'A'), ('GC_dist', 'G'), ('GC_dist', 'A+C')):
        p = paired(allv[a], allv[b])
        print(f"| all four cells (12000) | {a} vs {b} | +{p['gains']}/-{p['losses']} | "
              f"{100 * p['delta_sr']:+.1f} | {p['p_two_sided_exact']:.3g} |")
    print('\nOwner IR (decision-weighted, six sizes pooled) and SR per cell')
    print('| cell | A | A+C | G | GC_dist |\n|---|---|---|---|---|')
    for g in res:
        row = []
        for v in allv:
            num = den = 0.
            for c in g['cells']:
                r = audit[g['model'], g['suite'], c['size']]
                root, arm = (r['root'], r['arm']) if v == 'A+C' else (r['references'][v]['root'], r['references'][v]['arm'])
                ir, n = owner_ir(root, arm, g['model'])
                num += ir * n
                den += n
            row.append(f"{100 * g['pooled_arms'][v]['sr']:.1f}% / {num / den:.3f}")
        print(f"| {g['model']} {g['suite']} | " + ' | '.join(row) + ' |')
    print('\nAll four cells SR: ' + ', '.join(f"{v} {100 * sum(allv[v]) / len(allv[v]):.1f}%" for v in allv))


if __name__ == '__main__':
    main()
