"""New library seeds, two steps per lane (owner 10-23): knob off first, then levels only where needed.

Step 1 (fa_sd_*_o): every seed x size x suite x segment of a lane at koff, one campaign (build.define).
Step 2 (fa_sd_*_k): after step 1 completes, each cell is compared with same-step same-card pure inference
(reused, never rerun):
  LIBERO: formal_a_queue.classify (complete and knob-off successes >= pure successes -> skip);
  MT50:   the MT50 rule of knob/descent.py:50-51 (complete 2,500 and knob-off difficulty-group mean >= pure
          group mean -> skip), with the same-card same-step pure inference as the reference.
Every other cell (incomplete cells wait) gets k50..k10 with its seed's gaps, in one campaign.
References: DP/Fast-WAM/GR00T longseg_off_1022/REFERENCES.json (C5 pure, card of the knob-off run);
Cosmos formal_a PURE; pi0.5 MT50 the 4090 controls of knob/common (tpi7_0536, tpi7k2_1135).

    python -m exp.offline_search.rounds.r17.seeds_1023.workflow initialize --bundle dp_libero
    python -m exp.offline_search.rounds.r17.seeds_1023.workflow build-levels --name fa_sd_dp_r_l_k [--off-run RUN]
    python -m exp.offline_search.rounds.r17.seeds_1023.workflow {prepare,launch} --name fa_sd_dp_r_l_k
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex

from exp.offline_search.rounds.r17.longseg_1017.io_budget import dump
from exp.offline_search.rounds.r17.longseg_off_1022 import build as lob
from exp.offline_search.rounds.r17.filter_failed_knob.common import REPO, read
from . import inputs as ip
from .build import EVAL, HERE, define, name

QUEUE = HERE/'QUEUE.json'
LEVELS = tuple(ip.LEVELS)


def put(u):
    d = read(QUEUE) if QUEUE.exists() else dict(units=[])
    d['units'] = [r for r in d['units'] if r['name'] != u['name']]+[u]
    dump(QUEUE, d)


def selected(n):
    return next(r for r in read(QUEUE)['units'] if r['name'] == n)


def cells(bundle):
    """Every (seed, suite, size, segment) of a lane: the library's own segment, and C5 for long-segment models."""
    result = []
    for seed in ip.SEEDS[ip.benchmark(bundle)]:
        for (suite, size), p in sorted(ip.products(bundle, seed).items()):
            for segment in (p['native']['commit'],)+((5,) if bundle in ip.LONG else ()):
                result.append((seed, suite, size, segment))
    return result


def initialize(bundle):
    for mode in ('reduced', 'full'):
        n, levels = name(bundle, mode, 'off'), name(bundle, mode, 'levels')
        existing = {u['name'] for u in read(QUEUE)['units']} if QUEUE.exists() else set()
        common = dict(bundle=bundle, model=ip.model(bundle), benchmark=ip.benchmark(bundle), mode=mode, arm='filtered',
                      gpu='RTX 4090' if bundle == 'pi05_mw' else 'H100')
        if n not in existing:
            all_cells = cells(bundle)
            job = define(bundle, mode, 'off', all_cells, ('koff',))
            put(dict(common, name=n, stage='off', status='BUILT', episodes=job['episodes'], jobs=[job],
                     cells=[list(c) for c in all_cells]))
        if levels not in existing:
            put(dict(common, name=levels, stage='levels', status='WAITING_OFF', off=n, episodes=None, jobs=[],
                     maximum_episodes=len(selected(n)['cells'])*len(LEVELS)*(2500 if common['benchmark'] == 'mt50' else 500)))
        for verb in ('prepare', 'launch'):
            path = lob.COMMANDS/f'{verb}_{levels}.sh'
            path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'+
                            shlex.quote(str(REPO/'.venv/bin/python'))+' -m exp.offline_search.rounds.r17.seeds_1023.workflow '+
                            verb+' --name '+levels+'\n')
            path.chmod(0o755)


def reference(u, off_run):
    """Same-step same-card pure inference by suite, and its label."""
    model, bench, mode = u['model'], u['benchmark'], u['mode']
    if u['bundle'] in ip.LONG:
        key = f'{model}:{bench}:{mode}:{lob.card(off_run)}'
        if key not in read(lob.REFERENCES)['references']:
            raise ValueError(f'no same-step same-card C5 pure reference {key} in {lob.REFERENCES}')
        return lob.reference(model, bench, mode, lob.card(off_run)), key
    if model == 'cosmos':
        from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_queue as q
        return q.references('cosmos', mode), 'formal_a.PURE cosmos '+mode
    from exp.offline_search.rounds.r17.knob.test import load_controls
    refs = load_controls(EVAL, {'π0.5'}, mode, suites=('mt50',))
    return {suite: data for (label, suite), data in refs.items()}, 'knob.common MT50 π0.5 '+mode+' control (4090)'


def classify_mt50(off, pure):
    """knob/descent.py:50-51: complete, and the knob-off difficulty-group mean reaches the pure one."""
    from exp.offline_search.rounds.r17.knob import common as c
    row = dict(n=len(off), pure_n=len(pure), successes=sum(e['success'] for e in off.values()),
               pure_successes=sum(e['success'] for e in pure.values()), status='pending')
    if len(pure) == 2500 and set(off) == set(pure):
        row.update(group_mean=c.stats(off, 'DP')['group_mean'], pure_group_mean=c.stats(pure, 'DP')['group_mean'])
        row['status'] = 'skip' if row['group_mean'] >= row['pure_group_mean'] else 'tune'
    return row


def build_levels(n, off_run=None):
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_analysis as a
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_queue as q
    u = selected(n)
    if u['status'] != 'WAITING_OFF':
        return u
    off = selected(u['off'])
    off_run = Path(off_run or off['jobs'][0]['run'])
    data = a.load_r17(dict(run=str(off_run)))
    pure, key = reference(u, off_run)
    rows = []
    for seed, suite, size, segment in off['cells']:
        arms = data.get((suite, size, f'{ip.tag(seed)}c{segment}'), {})
        rule = classify_mt50 if u['benchmark'] == 'mt50' else q.classify
        rows.append(dict(seed=seed, suite=suite, size=size, segment=segment,
                         **rule(arms.get('koff', {}), pure.get(suite, {}))))
    dump(HERE/'off_decisions'/f'{n}.json', dict(rows=rows, off_run=str(off_run), reference=key, dev_outcomes_used=False))
    if any(r['status'] == 'pending' for r in rows):
        raise ValueError(f'{n}: wait for complete knob-off and reference cells: '+
                         str([(r['seed'], r['suite'], r['size'], r['segment'], r['n'], r['pure_n']) for r in rows if r['status'] == 'pending']))
    keep = [(r['seed'], r['suite'], r['size'], r['segment']) for r in rows if r['status'] == 'tune']
    if keep:
        job = define(u['bundle'], u['mode'], 'levels', keep, LEVELS, like=off['jobs'][0]['run'])
        u.update(jobs=[job], status='BUILT', episodes=job['episodes'])
    else:
        u.update(jobs=[], status='SKIP_OFF_SUFFICIENT', episodes=0)
    u.update(selected_cells=[list(c) for c in keep], off_run=str(off_run), reference=key)
    put(u)
    return u


def dispatch(n, verb):
    u = build_levels(n)
    if not u['jobs']:
        print('SKIP_OFF_SUFFICIENT', n)
        return
    root = Path(u['jobs'][0]['run'])
    if verb == 'prepare':
        from exp.offline_search.rounds.r17.filter_failed.stage import stage
        stage(root)
    else:
        from exp.offline_search.rounds.r17.filter_failed_knob.launch import main
        main(root)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('initialize', 'build-levels', 'prepare', 'launch'))
    p.add_argument('--bundle', choices=ip.BUNDLES)
    p.add_argument('--name')
    p.add_argument('--off-run', help='the knob-off run that actually served (e.g. a pod copy), default the built one')
    a = p.parse_args()
    if a.action == 'initialize':
        if not a.bundle:
            p.error('initialize needs --bundle')
        initialize(a.bundle)
    elif a.action == 'build-levels':
        build_levels(a.name, a.off_run)
    else:
        dispatch(a.name, a.action)


if __name__ == '__main__':
    main()
