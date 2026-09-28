#!/usr/bin/env python3
"""Paired exact McNemar between closed-loop arms: pair.py <run:arm> <run:arm> [...]; first arm is the reference."""
import json, sys, math
ROOT = '/home/weiland/trace_runs/os_closed_loop'
def load(spec):
    run, arm = spec.split(':')
    out = {}
    for line in open(f'{ROOT}/{run}/runs/{arm}/client/journal.jsonl'):
        try: r = json.loads(line)
        except Exception: continue
        uid = r.get('task_uid') or ''
        if ':eval:' not in uid or 'success' not in r: continue
        if r.get('error') not in (None, ''): continue
        key = tuple(uid.split(':eval:')[1].split(':')[:2])
        out[key] = bool(r['success'])
    return out
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
ref = load(sys.argv[1]); print(f'{sys.argv[1]}: n={len(ref)} SR={sum(ref.values())/max(1,len(ref)):.3f}')
for spec in sys.argv[2:]:
    a = load(spec); keys = sorted(set(ref) & set(a))
    b = sum(1 for k in keys if a[k] and not ref[k]); c = sum(1 for k in keys if ref[k] and not a[k])
    print(f'{spec}: n={len(a)} paired={len(keys)} SR={sum(a[k] for k in keys)/max(1,len(keys)):.3f} vs ref {sum(ref[k] for k in keys)/max(1,len(keys)):.3f}  +{b}/-{c}  p={mcnemar(b,c):.4f}')
