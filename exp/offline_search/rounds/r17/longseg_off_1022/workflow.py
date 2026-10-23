"""LIBERO long segment OFF, success-filtered library: OFF batch first, then levels.

Wired as longseg_1017.workflow: after the OFF batch completes, each cell is classified against
same-step pure inference on the same GPU (formal_a_queue.classify, unchanged); only cells that do not
reach it get the k50..k10 levels. Every lane serves on the H100. Cells whose library has no C5 fit
(size 500: only C10 fits exist, and nothing is refit) are not built.

    python -m exp.offline_search.rounds.r17.longseg_off_1022.workflow initialize
    python -m exp.offline_search.rounds.r17.longseg_off_1022.workflow {build-levels,prepare,launch} --name fa_lo_dp_r_f_l_l
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex

import yaml

from .build import (COMMANDS, EVAL, H100_FW_LIBERO, HERE, LS, build, c5_refs, dp_libero_h100, dump,
                    fw_reduced_h100, on_job, read)
from exp.offline_search.rounds.r17.filter_failed_knob.common import REPO
from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_queue as q

INDEX = HERE/'FORMAL_QUEUE.json'
TOKENS = {'dp': 'dp', 'fastwam': 'fw'}
LEVELS = ('k50', 'k40', 'k30', 'k20', 'k10')
GPU = 'H100'


def name(model, mode, stage):
    return f'fa_lo_{TOKENS[model]}_{mode[0]}_f_{stage[0]}_l'


def on_name(model, mode, stage):
    return f'fa_ls_{TOKENS[model]}_{mode[0]}_f_{stage[0]}_l'


def put(u):
    d = read(INDEX) if INDEX.exists() else dict(units=[])
    d['units'] = [r for r in d['units'] if r['name'] != u['name']]+[u]
    dump(INDEX, d)


def selected(n):
    return next(r for r in read(INDEX)['units'] if r['name'] == n)


def deployment(model, mode, on_campaign):
    """(clone source, deployment edit) of the H100 resident for one lane."""
    source = on_job(on_name(model, mode, 'off'))['source']
    if model == 'fastwam' and mode == 'reduced':
        return str(H100_FW_LIBERO), fw_reduced_h100(on_campaign)
    return source, dp_libero_h100 if model == 'dp' else None


def level_gaps(source):
    """ON periodic_gap of each ladder row: the decision-clock solution that the ON level YAMLs carry."""
    solutions = read(LS/'RECALIBRATION.json')['solutions']
    out = {}
    for g in read(Path(source)/'assets/CACHE_STUDY.json')['groups']:
        level = g['group'].rsplit('_', 1)[-1]
        if level in LEVELS:
            doc = yaml.safe_load((Path(source)/'assets'/g['file']).read_text())
            out[f'{g["suite"]}/{g["library_size"]}/{g["group"]}'] = \
                solutions[doc['recipe']['fitted_digest']+':filtered:'+level]['new_gap']
    return out


def initialize():
    for model in TOKENS:
        for mode in ('reduced', 'full'):
            on = on_name(model, mode, 'off')
            j, run = on_job(on), EVAL/on
            on_campaign = read(run/'campaign.json')
            refs = c5_refs(model, 'libero')
            groups = read(run/'assets/CACHE_STUDY.json')['groups']
            stores = {(g['suite'], g['library_size']): yaml.safe_load((run/'assets'/g['file']).read_text())['recipe']['store_digest']
                      for g in groups}
            items = [(g, run/'assets'/g['file'], None) for g in groups if stores[g['suite'], g['library_size']] in refs]
            missing = sorted(cell for cell, store in stores.items() if store not in refs)
            n = name(model, mode, 'off')
            existing = {u['name']: u for u in read(INDEX)['units']} if INDEX.exists() else {}
            if n not in existing:
                src, deploy = deployment(model, mode, on_campaign)
                job = build(n, src, items, model=model, benchmark='libero', mode=mode, stage='off',
                            workers=j['workers'], selection_on=on_campaign['filter_failed_knob']['selection'],
                            deploy=deploy, on=on)
                put(dict(name=n, model=model, mode=mode, arm='filtered', stage='off', status='BUILT', gpu=GPU,
                         episodes=job['episodes'], jobs=[job], on=on, cells=job['cells'], cells_without_c5_fit=missing))
            levels = name(model, mode, 'levels')
            if levels not in existing:
                cells = selected(n)['cells']
                put(dict(name=levels, model=model, mode=mode, arm='filtered', stage='levels', status='WAITING_OFF',
                         gpu=GPU, source=j['source'], workers=j['workers'], on=on_name(model, mode, 'levels'),
                         reference_key=f'{model}:libero:{mode}:{GPU}', cells=cells, cells_without_c5_fit=missing,
                         maximum_episodes=len(cells)*len(LEVELS)*500, level_gaps=level_gaps(j['source']),
                         episodes=None, jobs=[]))
            for verb in ('prepare', 'launch'):
                path = COMMANDS/f'{verb}_{levels}.sh'
                path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'+
                    shlex.quote(str(REPO/'.venv/bin/python'))+' -m exp.offline_search.rounds.r17.longseg_off_1022.workflow '+
                    verb+' --name '+levels+'\n')
                path.chmod(0o755)


def build_levels(n):
    from .build import REFERENCES, reference
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_analysis as a
    u = selected(n)
    if u['status'] != 'WAITING_OFF':
        return u
    if u['reference_key'] not in read(REFERENCES)['references']:
        raise ValueError('register the same-step same-card C5 pure LIBERO run as '+u['reference_key']+
                         ' in longseg_off_1022/REFERENCES.json')
    off = selected(name(u['model'], u['mode'], 'off'))
    data = a.load_r17(off['jobs'][0])
    pure = reference(u['model'], 'libero', u['mode'], u['gpu'])
    rows = [dict(suite=s, size=size, **q.classify(arms.get('koff', {}), pure.get(s, {}))) for (s, size), arms in data.items()]
    dump(HERE/'off_decisions'/f'{n}.json', dict(rows=rows, reference_key=u['reference_key'], dev_outcomes_used=False))
    if len(rows) != len(u['cells']) or any(r['status'] == 'pending' for r in rows):
        raise ValueError(f'Wait for all {len(u["cells"])} complete new OFF/pure cells')
    keep = {(r['suite'], r['size']) for r in rows if r['status'] == 'tune'}
    source = Path(u['source'])
    items = [(g, source/'assets'/g['file'], u['level_gaps'][f'{g["suite"]}/{g["library_size"]}/{g["group"]}'])
             for g in read(source/'assets/CACHE_STUDY.json')['groups']
             if (g['suite'], g['library_size']) in keep and g['group'].rsplit('_', 1)[-1] in LEVELS]
    if items:
        on_campaign = read(EVAL/on_name(u['model'], u['mode'], 'off')/'campaign.json')
        src, deploy = deployment(u['model'], u['mode'], on_campaign)
        j = build(n, src, items, model=u['model'], benchmark='libero', mode=u['mode'], stage='levels',
                  workers=u['workers'], selection_on=on_campaign['filter_failed_knob']['selection'],
                  deploy=deploy, on=u['on'])
        u.update(jobs=[j], status='BUILT', episodes=j['episodes'])
    else:
        u.update(jobs=[], status='SKIP_OFF_SUFFICIENT', episodes=0)
    u['selected_cells'] = sorted(keep)
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('initialize', 'build-levels', 'prepare', 'launch'))
    p.add_argument('--name')
    a = p.parse_args()
    if a.action == 'initialize':
        initialize()
    elif a.action == 'build-levels':
        build_levels(a.name)
    else:
        dispatch(a.name, a.action)


if __name__ == '__main__':
    main()
