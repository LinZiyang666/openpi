"""Simple, less overfit routing: send the k tasks where cache-only loses most to pure L10 to the policy, keep the rest
on cache (A, or B). A and B are 3-replicate means per (task, init). Choose the k tasks on one half of the inits,
evaluate on the other half (both directions), and also report in-sample. IR = mean of per-task IRs.
"""
import json, sys
import numpy as np
sys.path.insert(0, '/home/weiland/projects/openpi/exp/offline_search/rounds/r06/analysis_scripts')
sys.path.insert(0, '/home/weiland/.claude/jobs/a607dd74/tmp')
import common as cm
from per_task_oracle import per_episode_counts

out = {}
for model, cell in cm.CELLS:
    A, B = cm.ab_arms(model, cell)
    pure = cm.l10_ref(model, cell)
    c1 = cm.C1[model]

    def tables(specs):
        succ = np.zeros((10, 50)); ir = np.zeros((10, 2))
        for s in specs:
            j = cm.journal(s)
            for (t, i), v in j.items():
                succ[t, i] += v / len(specs)
            _, info, _ = per_episode_counts((s.replace(':', '/'), model))
            for t in range(10):
                for h in (0, 1):
                    v = np.array(info['half'][f'{t}:{h}']) if info and info['ok'] else None
                    ir[t, h] += (c1 * v[1] / v[0] + (1 - c1) * v[2] / v[0] if v is not None else info['ledger_ir']) / len(specs)
        return succ, ir

    SA, IA = tables(A); SB, IB = tables(B)
    SP = np.array([[cm.journal(pure)[(t, i)] for i in range(50)] for t in range(10)], float); IP = np.full((10, 2), .5)
    halves = [np.arange(50) < 25, np.arange(50) >= 25]
    res = {}
    for base_name, S, I in (('A', SA, IA), ('B', SB, IB)):
        rows = []
        for k in range(11):
            cv_sr, cv_ir = [], []
            for h in (0, 1):
                sel, ev = halves[h], halves[1 - h]
                gap = SP[:, sel].sum(1) - S[:, sel].sum(1)
                route = np.argsort(-gap, kind='stable')[:k]
                m = np.isin(np.arange(10), route)
                cv_sr.append((np.where(m[:, None], SP[:, ev], S[:, ev])).mean())
                cv_ir.append(np.where(m, IP[:, 1 - h], I[:, 1 - h]).mean())
            gap = SP.sum(1) - S.sum(1)
            route = np.argsort(-gap, kind='stable')[:k]
            m = np.isin(np.arange(10), route)
            rows.append(dict(k=k, in_SR=float(np.where(m[:, None], SP, S).mean()), in_IR=float(np.where(m, .5, I.mean(1)).mean()),
                             cv_SR=float(np.mean(cv_sr)), cv_IR=float(np.mean(cv_ir))))
        res[base_name] = rows
    out[f'{model}_{cell}'] = dict(pure_SR=float(SP.mean()), A_SR=float(SA.mean()), B_SR=float(SB.mean()),
                                  gap_by_task_A=(SP.sum(1) - SA.sum(1)).round(1).tolist(), routes=res)
    print(f'== {model} {cell}: pure {SP.mean():.3f}  A {SA.mean():.3f}  B {SB.mean():.3f}  gap(pure-A) by task {(SP.sum(1)-SA.sum(1)).round(1).tolist()}')
    for name in ('A', 'B'):
        print('  rest=' + name + ': ' + ' | '.join(f"k{r['k']} {r['cv_SR']:.3f}@{r['cv_IR']:.3f}" for r in res[name] if r['k'] in (0, 1, 2, 3, 4, 5, 10)))
json.dump(out, open('/home/weiland/.claude/jobs/a607dd74/tmp/per_task_route.json', 'w'), indent=1)
