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
    python -m exp.offline_search.rounds.r17.seeds_1023.workflow rounds --name fa_sd_dp_r_l_k [--off-run RUN]
    python -m exp.offline_search.rounds.r17.seeds_1023.workflow {prepare,launch} --name fa_sd_dp_r_l_k1   # ... k5

Step 2 as descent rounds (owner 10-24): round r runs level r only for the tasks that passed every earlier level,
so each task stops at its first failure; the frozen levels and the final SR/IR equal a full sweep's.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
from unittest.mock import patch

from exp.offline_search.rounds.r17.longseg_1017.io_budget import dump
from exp.offline_search.rounds.r17.longseg_off_1022 import build as lob
from exp.offline_search.rounds.r17.filter_failed_knob.common import REPO, read
from . import inputs as ip
from .build import EVAL, HERE, define, name, deployment_env

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
        registry = read(lob.REFERENCES)['references']
        if key not in registry:
            available = [k for k in registry if k.startswith(f'{model}:{bench}:{mode}:')]
            raise ValueError(f'no same-step same-card C5 pure reference {key} in {lob.REFERENCES}; '
                             f'available cards: {available} (cannot substitute another card)')
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
        job = define(u['bundle'], u['mode'], 'levels', keep, LEVELS, like=off_run)
        u.update(jobs=[job], status='BUILT', episodes=job['episodes'])
    else:
        u.update(jobs=[], status='SKIP_OFF_SUFFICIENT', episodes=0)
    u.update(selected_cells=[list(c) for c in keep], off_run=str(off_run), reference=key)
    put(u)
    return u


def tuned_cells(u, off_run):
    """Step 2's cells (classify each knob-off cell against its reference), the reference and its label."""
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_analysis as a
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_queue as q
    off = selected(u['off'])
    data = a.load_r17(dict(run=str(off_run)))
    pure, key = reference(u, off_run)
    rows = []
    for seed, suite, size, segment in off['cells']:
        arms = data.get((suite, size, f'{ip.tag(seed)}c{segment}'), {})
        rule = classify_mt50 if u['benchmark'] == 'mt50' else q.classify
        rows.append(dict(seed=seed, suite=suite, size=size, segment=segment,
                         **rule(arms.get('koff', {}), pure.get(suite, {}))))
    dump(HERE/'off_decisions'/f'{u["name"]}.json', dict(rows=rows, off_run=str(off_run), reference=key, dev_outcomes_used=False))
    if any(r['status'] == 'pending' for r in rows):
        raise ValueError(f'{u["name"]}: wait for complete knob-off and reference cells: '+
                         str([(r['seed'], r['suite'], r['size'], r['segment'], r['n'], r['pure_n']) for r in rows if r['status'] == 'pending']))
    return [(r['seed'], r['suite'], r['size'], r['segment']) for r in rows if r['status'] == 'tune'], pure, key


def rounds(n, off_run=None):
    """Register step 2 of lane n (a levels unit) as one descent round per level, <n>1..<n>5.

    Round r measures LEVELS[r-1] only for the tasks of each tuned cell that passed every earlier level (at most two
    fewer successes than the reference on the same 50 inits). A task's first failure freezes the level above it, so
    the levels below are never needed: the selection equals a full sweep's (knob/descent.descend)."""
    u = selected(n)
    for r, level in enumerate(LEVELS, 1):
        name_r = f'{n}{r}'
        existing = {x['name'] for x in read(QUEUE)['units']}
        if name_r not in existing:
            put(dict({k: u[k] for k in ('bundle', 'model', 'benchmark', 'mode', 'arm', 'gpu', 'off')},
                     name=name_r, stage='round', round=r, level=level, parent=n, status='WAITING', jobs=[], episodes=None,
                     **({'off_run': str(off_run)} if off_run else {})))
        for verb in ('prepare', 'launch'):
            path = lob.COMMANDS/f'{verb}_{name_r}.sh'
            if path.exists() and 'seeds_1023.workflow' not in path.read_text():
                continue  # already built: the wrapper now runs the defined batch directly
            path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'+
                            shlex.quote(str(REPO/'.venv/bin/python'))+' -m exp.offline_search.rounds.r17.seeds_1023.workflow '+
                            verb+' --name '+name_r+'\n')
            path.chmod(0o755)
        print('ROUND', name_r, level, flush=True)


def build_round(n):
    """Define round n from the finished earlier rounds of its lane (see rounds)."""
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_analysis as a
    u = selected(n)
    if u['status'] != 'WAITING':
        return u
    parent = selected(u['parent'])
    off_run = Path(u.get('off_run') or selected(parent['off'])['jobs'][0]['run'])
    keep, pure, key = tuned_cells(parent, off_run)
    measured = {}
    for r in range(1, u['round']):
        earlier = selected(f'{u["parent"]}{r}')
        if earlier['status'] not in ('BUILT', 'SKIP_NO_TASKS'):
            raise ValueError(f'{n}: round {r} is {earlier["status"]}')
        for job in earlier['jobs']:
            for cell, arms in a.load_r17(dict(run=job['run'])).items():
                for level, eps in arms.items():
                    measured.setdefault(cell, {}).setdefault(level, {}).update(eps)
    alive, incomplete = {}, []
    for seed, suite, size, segment in keep:
        ref = pure[suite]
        arms = measured.get((suite, size, f'{ip.tag(seed)}c{segment}'), {})
        names = sorted({k[1] for k in ref})
        for level in LEVELS[:u['round']-1]:
            survivors = []
            for task in names:
                inits = [k for k in ref if k[1] == task]
                got = [arms.get(level, {}).get(k) for k in inits]
                if len(inits) != 50 or None in got:
                    incomplete.append((seed, suite, size, segment, level, task, sum(g is not None for g in got)))
                    continue
                if sum(ref[k]['success'] for k in inits) - sum(g['success'] for g in got) <= 2:
                    survivors.append(task)
            names = survivors
        if names:
            alive[seed, suite, size, segment] = names
    if incomplete:
        raise ValueError(f'{n}: earlier rounds incomplete {incomplete[:5]} (+{max(0, len(incomplete)-5)})')
    if alive:
        job = define(u['bundle'], u['mode'], 'levels', list(alive), (u['level'],), like=off_run, unit=n, tasks=alive)
        u.update(jobs=[job], status='BUILT', episodes=job['episodes'])
    else:
        u.update(jobs=[], status='SKIP_NO_TASKS', episodes=0)
    u.update(alive=[[*cell, names] for cell, names in alive.items()], tuned_cells=[list(c) for c in keep],
             off_run=str(off_run), reference=key)
    put(u)
    return u


def dispatch(n, verb):
    u = build_round(n) if selected(n).get('stage') == 'round' else build_levels(n)
    if not u['jobs']:
        print('SKIP_OFF_SUFFICIENT', n)
        return
    root = Path(u['jobs'][0]['run'])
    env = deployment_env(read(Path(u['off_run'])/'campaign.json')) if u.get('off_run') else {}
    with patch.dict(os.environ, env):
        if verb == 'prepare':
            from exp.offline_search.rounds.r17.filter_failed.stage import stage
            stage(root)
        else:
            from exp.offline_search.rounds.r17.filter_failed_knob.launch import main
            main(root)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('initialize', 'build-levels', 'rounds', 'prepare', 'launch'))
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
    elif a.action == 'rounds':
        rounds(a.name, a.off_run)
    else:
        dispatch(a.name, a.action)


if __name__ == '__main__':
    main()
