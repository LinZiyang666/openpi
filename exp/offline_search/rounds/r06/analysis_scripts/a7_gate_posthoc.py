#!/usr/bin/env python3
"""POST-HOC, descriptive: do the per-cell calibration quantities available before any test outcome (the <= 10 B-val
recordings used by configuration C) order the eight banks by the value of policy calls?  Eight fixed banks, no
interval is claimed as a transferable calibration; this only sizes the 'library-level no-call gate' proposal.
Writes analysis_r6/gate_posthoc.json.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402


def main():
    cal = {r['cell']: r for r in json.load(open(os.path.join(cm.OUT, 'c_calibration.json')))}
    head = {(r['model'], r['cell']): r for r in json.load(open(os.path.join(cm.OUT, 'headline.json')))['rows']}
    carms = json.load(open(os.path.join(cm.OUT, 'c_arms.json')))['arms']
    rows = []
    for model, cell in cm.CELLS:
        name = cm.c_cell(model, cell)
        h = head[(model, cell)]
        c = cal[name]
        cfg = 'C18' if cell.endswith('500') else 'C30'
        ca = next(a for a in carms if a['model'] == model and a['cell'] == cell and a['cfg'] == cfg)
        rows.append(dict(cell=name, E_mean=c['baseline_E'], rec_sr=c['rec_success'] / c['rec_n'], A=h['A_mean'],
                         B_minus_A=h['B_mean'] - h['A_mean'], C_minus_A=ca['d_a'][0], L10_minus_A=h['l10'] - h['A_mean']))
    out = dict(rows=rows)
    for x in ('E_mean', 'rec_sr'):
        for y in ('B_minus_A', 'C_minus_A', 'L10_minus_A', 'A'):
            r = spearmanr([q[x] for q in rows], [q[y] for q in rows])
            out[f'{x}~{y}'] = float(r.statistic)
    cm.dump('gate_posthoc.json', out)
    for q in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
    print({k: round(v, 3) for k, v in out.items() if k != 'rows'})


if __name__ == '__main__':
    main()
