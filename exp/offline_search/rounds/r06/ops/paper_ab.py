#!/usr/bin/env python3
"""Recompute the R6 A/B three-replicate table (PAPER_AB.md body) from each arm's summary.json and client journal.

Journal records are taken as they stand after the exception-episode repair (closed_loop/ops/remote/purge_exc.py):
one terminal record per (task, init). Owner IR = .152 v + .848 m (pi0.5) / .148 v + .852 m (GR00T) from cost_ledger.
Pooled +/- pairs replicate k of A with replicate k of B episode by episode; exact two-sided McNemar.
"""
import json
import math
import sys

ROOT = '/home/weiland/trace_runs/os_closed_loop'
C1 = {'pi05': .152, 'groot': .148}
CELLS = [('pi05', 'l10_50', 'LIBERO-10, 50'), ('pi05', 'l10_500', 'LIBERO-10, 500'),
         ('pi05', 'sp_50', 'Spatial, 50'), ('pi05', 'sp_500', 'Spatial, 500'),
         ('groot', 'l10_50', 'LIBERO-10, 50'), ('groot', 'l10_500', 'LIBERO-10, 500'),
         ('groot', 'sp_50', 'Spatial, 50'), ('groot', 'sp_500', 'Spatial, 500')]


def arms(model, cell):
    if model == 'groot':
        a = [f'r05_x:r5x_g_{cell}_tail1u'] + [f'r06_paper:r5x_g_{cell}_tail1u_rep{k}' for k in (2, 3)]
        b = [f'r06_paper:r6p1_c10_g_{cell}'] + [f'r06_paper:r6p1_c10_g_{cell}_rep{k}' for k in (2, 3)]
    else:
        a1 = 'r04_blind:r4b3_p_sp_500_tail1uc' if cell == 'sp_500' else f'r05_ptail:r5t_p_{cell}_tail1uc'
        stem = a1.split(':')[1]
        a = [a1] + [f'r06_paper:{stem}_rep{k}' for k in (2, 3)]
        b = [f'r05_q1:r5q1_c10_p_{cell}'] + [f'r06_paper:r5q1_c10_p_{cell}_rep{k}' for k in (2, 3)]
    return a, b


def load(spec):
    run, arm = spec.split(':')
    d = f'{ROOT}/{run}/runs/{arm}'
    out = {}
    for line in open(f'{d}/client/journal.jsonl'):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if ':eval:' not in r.get('task_uid', '') or 'success' not in r or r.get('error') not in (None, ''):
            continue
        out[tuple(r['task_uid'].split(':eval:')[1].split(':')[:2])] = bool(r['success'])
    s = json.load(open(f'{d}/summary.json'))
    return out, s['cost_ledger']


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def fmt(x):
    return f'{x:.3f}'.lstrip('0') if x >= 0 else '−' + f'{-x:.3f}'.lstrip('0')


rows = []
for model, cell, label in CELLS:
    A, B = arms(model, cell)
    la, lb = [load(x) for x in A], [load(x) for x in B]
    sr = lambda o: sum(o.values()) / len(o)
    ir = lambda c: C1[model] * c['v'] + (1 - C1[model]) * c['m']
    asr, bsr = [sr(o) for o, _ in la], [sr(o) for o, _ in lb]
    air, bir = sum(ir(c) for _, c in la) / 3, sum(ir(c) for _, c in lb) / 3
    plus = minus = 0
    for (oa, _), (ob, _) in zip(la, lb):
        for k in set(oa) & set(ob):
            plus += ob[k] and not oa[k]
            minus += oa[k] and not ob[k]
    ns = [len(o) for o, _ in la + lb]
    am, bm = sum(asr) / 3, sum(bsr) / 3
    p = mcnemar(plus, minus)
    rows.append(f"| {'π0.5' if model == 'pi05' else 'GR00T'} | {label} | {' / '.join(fmt(x) for x in asr)} | {fmt(am)} | "
                f"{fmt(air)} | {' / '.join(fmt(x) for x in bsr)} | {fmt(bm)} | {fmt(bir)} | "
                f"{'+' if bm >= am else '−'}{abs(bm - am) * 100:.1f} pp | +{plus} / −{minus} | {p:.2g} |"
                + ('' if set(ns) == {500} else f' n={ns}'))
print('| model | cell | A SR (3 runs) | A mean | A IR | B SR (3 runs) | B mean | B IR | B − A | pooled +/− | p |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
print('\n'.join(rows))
