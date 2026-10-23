"""Final SR and IR of the Q3 batches (π0.5/GR00T LIBERO: R15 full, R16 reduced) and the cache-component groups.

CPU only, read only. Outcomes come from analysis.load_job (the native R15/R16 loaders with the two
judge-name checks removed) over the q3_port_1020 QUEUE units, never from formal_a_analysis.active_jobs
(the withdrawn old-gate jobs). Selection is formal_a_analysis.choose: a cell whose complete off cohort
reaches the same-step pure A is not tuned; otherwise each task descends ir45 → ir40 → ir32 → ir25 → off
and freezes the level before its first failure (two-success tolerance). IR is the native per-decision
cost ledger (formal_a_analysis.stats). M/N counts model calls over decision heads (decisions with vision;
policy tails count in neither), from the same ledger.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

from . import analysis
from .build import INDEX, HERE
from .fit import read
from .. import formal_a as f, formal_a_analysis as fa
from ..common import dump

PAIRS = [(t, i) for t in range(10) for i in range(50)]
CS = Path(__file__).resolve().parents[1]/'cache_study_1019/CAMPAIGNS.json'


def ledger(ep, suite):
    return {(suite, str(t), i): dict(success=bool(ep.y[t, i]), n=int(ep.n[t, i]), cost_sum=float(ep.cost[t, i]),
                                     vision=int(ep.v[t, i]), misses=int(ep.m[t, i]))
            for t in range(10) for i in range(50) if ep.y[t, i] >= 0}


def load_cs(job):
    from exp.offline_search.rounds.r16 import analyze as a, conductor_analysis as ca

    class Reader(a.Reader):
        # C5 cache arms log no vision flag (c/cr: no hit flag either): every C5 decision takes a real
        # look (r14.real_look); without a judge every decision is a synthesized cache action (hit).
        # Gate-forced C5 calls (cg/crg) log no miss_k; the shard's startup row carries the served K.
        def jsonl(self, path):
            k = None
            for d in super().jsonl(path):
                if d.get('ev') == 'startup':
                    k = d.get('miss_steps')
                if d.get('ev') == 'dec' and ('hit' not in d or 'vision' not in d or 'miss_k' not in d):
                    d = dict(d, hit=d.get('hit', bool(d.get('synth'))),
                             vision=d.get('vision', bool((d.get('r14') or {}).get('real_look'))))
                    d.setdefault('miss_k', None if d['hit'] else k)
                yield d

    root = Path(job['run']); result = {}
    if not (root/'DONE.json').exists():
        return result
    group = read(root/'group.json')
    for entry in group['entries']:
        ep = ca.load_entry(group, dict(entry, pairs=PAIRS), Reader()); row = entry['row']
        result.setdefault((row['suite'], row['r10_size']), {})[row['r14_descent_level']] = ledger(ep, row['suite'])
    return result


def task(item):
    try:
        return _task(item)
    except Exception as e:  # reported per item; the other items still finish
        import traceback
        return {'__error__': traceback.format_exc()}


def _task(item):
    kind, payload = item
    with patch.object(fa, 'legacy_data', ledger):
        if kind == 'job':
            return analysis.load_job(payload)
        if kind == 'cs':
            return load_cs(payload)
        return fa.vla_references(*payload)


def stats(data, model, mode, suite):
    st = fa.stats(data, model, mode, suite)
    heads = sum(e.get('vision', 0) for e in data.values()); calls = sum(e.get('misses', 0) for e in data.values())
    decisions = sum(e.get('n', 0) for e in data.values())
    st.update(successes=sum(e['success'] for e in data.values()), level_ir=calls/heads if heads else None,
              decision_heads=heads, model_calls=calls, decisions=decisions)
    return st


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--out', type=Path, default=HERE/'RESULTS_1022.json')
    args = p.parse_args()
    units = read(INDEX)['units']
    knob = [u for u in units if u['stage'] in ('off', 'levels')]
    cs = [dict(j, model=j['model'], cs_unit=j['name']) for j in read(CS)['jobs']]
    cs += [dict(j, model=u['model'], cs_unit=u['name']) for u in units if u['stage'] == 'cache_gated' for j in u['jobs']]
    items = [('job', j) for u in knob for j in u['jobs']] + [('cs', j) for j in cs]
    items += [('ref', (m, mode)) for m in ('pi05', 'groot') for mode in ('full', 'reduced')]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        loaded = list(pool.map(task, items, chunksize=1))
    errors = [dict(kind=kind, item=payload if kind == 'ref' else payload.get('name'), error=out['__error__'])
              for (kind, payload), out in zip(items, loaded) if '__error__' in out]
    for e in errors:
        print('ERROR', e['kind'], e['item'], e['error'].strip().splitlines()[-1], flush=True)
    loaded = [{} if '__error__' in out else out for out in loaded]
    refs = {payload: out for (kind, payload), out in zip(items, loaded) if kind == 'ref'}
    data, runs = {}, {}
    for (kind, payload), out in zip(items, loaded):
        if kind != 'job':
            continue
        u = next(u for u in knob if payload in u['jobs'])
        key = u['model'], u['mode'], u['arm']
        runs.setdefault(key, []).append(dict(unit=u['name'], name=payload['name'], run=payload['run'], stage=u['stage']))
        for cell, levels in out.items():
            for level, episodes in levels.items():
                data.setdefault(key, {}).setdefault(cell, {}).setdefault(level, {}).update(episodes)
    rows = []
    for (model, mode, arm), cells in sorted(data.items()):
        pure_all = refs[model, mode]
        for suite, size in sorted(cells):
            arms, pure = cells[suite, size], pure_all.get(suite, {})
            d = fa.choose(arms, pure, f.VLA_LEVELS)
            rows.append(dict(model=model, mode=mode, arm=arm, suite=suite, size=size,
                             runs=[r for r in runs[model, mode, arm]], pure=stats(pure, model, mode, suite),
                             final=stats(d['data'], model, mode, suite), no_knob=d['no_knob'], tuned=d['tuned'],
                             complete=d['complete'], grid_complete=d['grid_complete'],
                             per_task_levels={t: x['selected'] for t, x in d['tasks'].items()}, tasks=d['tasks'],
                             measured_levels={level: stats(v, model, mode, suite) for level, v in arms.items()}))
    cs_rows = []
    for (kind, payload), out in zip(items, loaded):
        if kind != 'cs':
            continue
        pure_all = refs[payload['model'], 'reduced']
        for (suite, size), groups in sorted(out.items()):
            for group, episodes in sorted(groups.items()):
                cs_rows.append(dict(unit=payload['cs_unit'], run=payload['run'], model=payload['model'], mode='reduced',
                                    suite=suite, size=size, group=group, result=stats(episodes, payload['model'], 'reduced', suite),
                                    pure=stats(pure_all.get(suite, {}), payload['model'], 'reduced', suite)))
    dump(args.out, dict(rows=rows, cache_component_rows=cs_rows, reference='same-step pure A (formal_a_analysis.vla_references)',
                        rule='formal_a_analysis.choose: tune iff the complete off cohort has fewer successes than pure; per-task first failure, 2-success tolerance',
                        ir_definition='native R15/R16 per-decision cost ledger: sum(vision*v + miss*m)/decisions (D5 pricing, rounds/r16/cost.py)',
                        level_ir_definition='model calls / decision heads (decisions with vision; policy tails excluded)',
                        withdrawn_outcomes_used=False, load_errors=errors))
    print('Wrote', len(rows), 'knob cells and', len(cs_rows), 'cache-component cells to', args.out)


if __name__ == '__main__':
    main()
