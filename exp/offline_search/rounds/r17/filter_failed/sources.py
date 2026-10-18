"""Explicit original Cache outcomes and the segment that actually ran each init."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

from .audit_counts import REPO, read

EVAL = Path('/data/mw_finetune/eval')
LEGACY_RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
MT50 = dict(pi05=['cspi_k2', 'cspi_k2vc'], groot=['csgr_h5', 'csgr_h6'], dp=['csdp_1504'],
            fastwam=['csfw_k3', 'csfw_k3w4', 'csfw_k3w5', 'csfw_k3w6'])
LABELS = dict(pi05='π0.5', groot='GR00T', dp='DP', fastwam='Fast-WAM', cosmos='Cosmos')


def lines(path):
    if Path(path).exists():
        with Path(path).open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def segments(run):
    result, seen = [], set()
    while run not in seen:
        seen.add(run)
        campaign = read(run / 'campaign.json')
        result.append(run)
        source = (campaign.get('cache_resume_selection') or {}).get('source') or campaign.get('resume_from')
        if not source and (run / 'CACHE_RESUME.json').exists():
            source = read(run / 'CACHE_RESUME.json').get('source')
        if not source:
            break
        run = Path(source).resolve()
    return result[::-1]


@lru_cache(None)
def r17_runs(runs):
    tasks, terminal = {}, {}
    for run in runs:
        run = Path(run)
        plan = run / ('FULL_EPISODE_PLAN.json' if (run / 'FULL_EPISODE_PLAN.json').exists() else 'EPISODE_PLAN.json')
        tasks.update((t['task_uid'], t) for t in read(plan)['tasks'])
        for r in lines(run / 'driver/journal.jsonl'):
            if not r.get('accepted') or r.get('error') or r.get('status') not in ('done', 'failed'):
                continue
            uid = r['task_uid']
            # Resume merge journals may copy ancestor rows verbatim. Keep the
            # original execution segment on equal timestamps, as analyze_cache.
            if uid not in terminal or (r.get('ts') or 0) > (terminal[uid][1].get('ts') or 0):
                terminal[uid] = (str(run), r)
    out = defaultdict(dict)
    for uid, (origin, r) in terminal.items():
        task = tasks[uid]
        ex = task['extra']
        if ex.get('cache_group') != 'c':
            continue
        suite = 'mt50' if task['experiment']=='metaworld_mt50' else task['experiment']
        key = suite, ex.get('library_size', 500)
        coordinate = ex['task_name'], task['orig_init_state_idx']
        out[key][coordinate] = dict(task=ex['task_name'], task_id=task['task_id'], init=coordinate[1],
                                   success=bool(r['success']), source=origin, uid=uid)
    return out


def baseline(cell):
    model, suite, size = cell['model'], cell['suite'], cell['size']
    if suite == 'mt50':
        runs = tuple(str(EVAL / run) for run in MT50[model])
        result = r17_runs(runs)[suite, size]
    elif cell['source']=='R17':
        report = read(REPO / 'exp/offline_search/rounds/r17/cache/CACHE_RESULTS.json')
        row = next(r for r in report['rows'] if r['model']==LABELS[model] and r['suite']==suite
                   and r['size']==size and r['group']=='c' and r['cohort']=='full')
        runs = tuple(str(p) for lane in row['lane'].split('; ') for p in segments(Path(lane)))
        result = r17_runs(tuple(dict.fromkeys(runs)))[suite, size]
    else:
        root = LEGACY_RUNS / f'r10_size_{model}'
        short = 'spatial' if suite=='libero_spatial' else 'l10'
        arm = f'r10_{model}_{short}_{size}_A'
        journal = root / 'runs' / arm / 'journal.jsonl'
        if not journal.exists():
            # R10's standard collector stores the driver file under client/.
            candidates = list((root / 'runs' / arm).glob('**/journal.jsonl'))
            if len(candidates)!=1:
                raise FileNotFoundError(f'Cannot identify original R10 journal: {root / "runs" / arm}')
            journal = candidates[0]
        result = {}
        for r in lines(journal):
            if not r.get('accepted') or r.get('error') or r.get('success') is None:
                continue
            task, init = map(int, r['task_uid'].rsplit(':', 2)[-2:])
            result[task, init] = dict(task=task, task_id=task, init=init, success=bool(r['success']),
                                     source=str(root), uid=r.get('task_uid'), arm=arm)
    expected = 2500 if suite=='mt50' else 500
    if len(result)!=expected:
        raise ValueError(f'Original Cache cell {model}/{suite}/{size}: {len(result)} != {expected}')
    return sorted(result.values(), key=lambda r:(str(r['task']), r['init']))


@lru_cache(None)
def group(source, suite, size):
    source = Path(source)
    return next(r for r in read(source / 'assets/CACHE_STUDY.json')['groups']
                if r['suite']==suite and r.get('library_size',500)==size and r['group']=='c')


def served_fit(source, row):
    return Path(source) / 'assets/cache_inputs' / row['fit_rel']
