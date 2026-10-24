"""Write SEEDS_INDEX.json from exact metadata only (CPU only; reads experiments, never changes them).

Batches: QUEUE.json (R17 knob off; descent rounds k1..k5, whose off_run is the knob-off run that served) and
vla/QUEUE.json (R15/R16 knob off, rounds k1..k4, repairs <unit>r); each run at its final continuation tip
(continued_lanes over filter_failed/CONTINUATIONS.json). Card from campaign.json: server_node local = RTX 4090,
pod_copy = H200 MIG 1g.18gb, else H100. R17 slices from assets/CACHE_STUDY.json (group, seed, library_size) and each
group YAML's adapter_binding.commit_horizon (5 = long segment off, longer = on; Cosmos LIBERO and pi0.5 MT50
default); the level is the batch's own campaign seeds_1023.levels. VLA arms from arms.json (seed/segment of the
R15 job) or group.json entries (row subset_seed/long_segment).
References (SEEDS_INDEX.example.json _instructions): DP/Fast-WAM/GR00T C5 longseg_off_1022/REFERENCES.json at the
batch card; long segment on also its C10 pure fallback, only from LONG_REFERENCES.json (long_refs.py); Cosmos
formal_a.PURE; pi0.5 MT50 knob.common MT50 controls; VLA as vla/build.references.

A lane (every line sharing its batches) is indexed only when all its batches have every planned episode accepted,
every round is defined (BUILT or SKIP_NO_TASKS) and every reference is complete on the lane's card. Other lanes are
listed under _skipped: a partial chain is an error in results_seeds, not a missing cell.

    PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.seeds_1023.make_index [--out PATH]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import tempfile

import yaml

from exp.offline_search.rounds.r17.filter_failed_knob import formal_a
from exp.offline_search.rounds.r17.filter_failed_knob.common import read, continued_lanes
from exp.offline_search.rounds.r17.filter_failed_knob.formal_a_names import resolve_reference
from exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022 import build as lo
from exp.offline_search.rounds.r17.knob import common as kc
from exp.offline_search.rounds.r17.longseg_off_1022 import build as lob, results_off as off
from . import inputs as ip, results_seeds as rs

HERE = Path(__file__).resolve().parent


def tip(run):
    tips = continued_lanes([run])
    if len(tips) != 1:
        raise ValueError(f'{run}: several continuation tips {tips}')
    return str(run) if tips[0] == Path(run).resolve() else str(tips[0])


def card(run):
    return 'RTX 4090' if read(Path(run)/'campaign.json')['server_node'] == 'local' else lob.card(run)


@lru_cache(maxsize=None)
def r17_reference(model, bench, mode, gpu, long):
    """(name, spec, problem) of a same-step pure reference for a batch on `gpu`."""
    key = f'{model}:{bench}:{mode}:{gpu}'
    if long:
        entry = read(HERE/'LONG_REFERENCES.json')['references'].get(key)
        key += ':long'
    elif model == 'cosmos':
        entry = dict(run=str(ip.EVAL/formal_a.PURE['cosmos'][mode]))
    elif model == 'pi05':
        entry = dict(run=str(ip.EVAL/(kc.MW_REDUCED_CONTROL_NAMES if mode == 'reduced' else kc.MW_CONTROL_NAMES)['π0.5'][-1]))
    else:
        entry = read(lob.REFERENCES)['references'].get(key)
    if entry is None:
        return key, None, f'no registered reference {key}' + (' in LONG_REFERENCES.json' if long else '')
    run = tip(entry['run'])
    groups = entry.get('groups') or ['control']
    if 'groups' not in entry and read(Path(run)/'campaign.json').get('cache_study'):
        groups = sorted({g['group'] for g in read(Path(run)/'assets/CACHE_STUDY.json')['groups'] if g['group'].endswith('_pure')})
    spec = dict(pipeline='R17', model=model, benchmark=bench, mode=mode, gpu=card(run), run=run, groups=groups)
    progress = off.r17_progress(run)
    waiting = f'{progress["accepted"]}/{progress["expected"]}' if progress['expected'] else progress.get('note')
    problem = (f'{key}: {Path(run).name} is on {spec["gpu"]}' if spec['gpu'] != gpu else None if progress['complete'] else
               f'等参照 {Path(run).name}（{key}，{waiting}）')
    return key, spec, problem


@lru_cache(maxsize=None)
def vla_reference(model, mode, segment):
    key = f'{model}:libero:{mode}:H100:{segment}'
    if (model, mode, segment) == ('pi05', 'full', 'on'):
        overrides = read(formal_a.ROOT/'PURE_A_OVERRIDES.json')
        jobs = [dict(zip(('run', 'arm'), resolve_reference(o['root'], o['arm'])))
                for o in (overrides[f'pi05:full:{s}'] for s in rs.suites('libero'))]
    else:
        unit = next(u for u in read(lo.INDEX)['units'] if (u['model'], u['mode'], u['stage'], u.get('segment')) ==
                    (model, mode, 'pure', 'c5' if segment == 'off' else 'c10'))
        jobs = [dict(run=j['run'], arm=j['arm']) for j in unit['jobs']]
    missing = [f'{Path(j["run"]).name}/{j["arm"]}' for j in jobs if not (Path(j['run'])/'runs'/j['arm']/'summary.json').exists()]
    spec = dict(pipeline='R15', model=model, benchmark='libero', mode=mode, gpu='H100', jobs=jobs)
    return key, spec, f'{key}: incomplete {missing}' if missing else None


def lanes():
    """(lane, pipeline, [(batch, unit, run)], problems) for every knob-off unit."""
    units = read(HERE/'QUEUE.json')['units']
    for u in units:
        if u['stage'] != 'off':
            continue
        rounds = sorted((r for r in units if r['stage'] == 'round' and r['off'] == u['name']), key=lambda r: r['round'])
        off_run = {r['off_run'] for r in rounds if r.get('off_run')} or {u['jobs'][0]['run']}
        problems = [] if len(off_run) == 1 else [f'rounds disagree on the knob-off run {off_run}']
        batches = [(Path(r).name, u['name'], r) for r in sorted(off_run)]
        for r in rounds:
            batches += [(j['name'], r['name'], j['run']) for j in r['jobs']]
        problems += [f'{r["name"]} {r["status"]}' for r in rounds if r['status'] not in ('BUILT', 'SKIP_NO_TASKS')]
        if [r['round'] for r in rounds] != list(range(1, len(ip.LEVELS)+1)):
            problems.append('descent rounds k1..k5 not all registered')
        yield u['name'].removesuffix('_o'), 'R17', batches, problems
    units = read(HERE/'vla/QUEUE.json')['units']
    for u in units:
        if u['stage'] != 'off':
            continue
        same = lambda x: all(x.get(k) == u.get(k) for k in ('model', 'mode', 'subset_seed', 'long_segment'))
        rounds = sorted((x for x in units if x['stage'] == 'round' and same(x)), key=lambda x: x['round'])
        names = {u['name']} | {x['name'] for x in rounds}
        chain = [u, *rounds, *(x for x in units if x['stage'] == 'repair' and x.get('repair_of') in names)]
        problems = [f'{x["name"]} {x["status"]}' for x in chain if x['status'] not in ('BUILT', 'SKIP_NO_TASKS')]
        if [x['round'] for x in rounds] != [1, 2, 3, 4]:
            problems.append('descent rounds k1..k4 not all registered')
        yield u['name'].removesuffix('_o'), u['pipeline'], [(j['name'], x['name'], j) for x in chain for j in x['jobs']], problems


def r17_slices(run, unit, gpu):
    level, = read(Path(run)/'campaign.json')['seeds_1023']['levels']
    slices = {}
    for g in read(Path(run)/'assets/CACHE_STUDY.json')['groups']:
        if not g['group'].endswith('_'+level):
            raise ValueError(f'{run}: group {g["group"]} in a {level} batch')
        commit = yaml.safe_load((Path(run)/'assets'/g['file']).read_text())['adapter_binding']['commit_horizon']
        segment = ('off' if commit == 5 else 'on') if unit['bundle'] in ip.LONG else 'default'
        s = slices.setdefault((g['seed'], segment), dict(
            model=unit['model'], benchmark=unit['benchmark'], mode=unit['mode'], gpu=gpu, seed=g['seed'],
            long_segment=segment, suites=list(rs.suites(unit['benchmark'])), sizes=set(), groups={}))
        s['sizes'].add(g['library_size'])
        s['groups'][g['group']] = level
    return list(slices.values())


def vla_slices(job):
    root = Path(job['run'])
    if job['pipeline'] == 'R15':
        rows = [dict(r, subset_seed=job['subset_seed'], long_segment=job['long_segment']) for r in read(root/'arms.json')]
    else:
        rows = [e['row'] for e in read(root/'group.json')['entries']]
    slices = {}
    for r in rows:
        s = slices.setdefault((r['subset_seed'], r['long_segment']), dict(
            model=job['model'], benchmark='libero', mode='full' if job['pipeline'] == 'R15' else 'reduced', gpu=job['gpu'],
            seed=r['subset_seed'], long_segment=r['long_segment'], suites=list(rs.suites('libero')), sizes=set(), arms=[]))
        s['sizes'].add(r['r10_size'])
        s['arms'].append(r['arm'])
    return list(slices.values())


def lane_entries(pipeline, batches, problems):
    """({batch: entry}, {reference: spec}, problems, [batch status])."""
    units = {u['name']: u for u in read(HERE/'QUEUE.json')['units']}
    entries, refs, status = {}, {}, []
    for name, unit, run in batches:
        if pipeline == 'R17':
            if not (Path(run)/'campaign.json').exists():
                status.append(dict(batch=name, unit=unit, run=run, complete=False, note='not built'))
                continue
            run = tip(run)
            progress = off.r17_progress(run)
            status.append(dict(batch=name, unit=unit, run=run, complete=progress['complete'],
                               accepted=progress['accepted'], expected=progress['expected']))
            if progress['complete']:
                entries[name] = dict(pipeline='R17', unit=unit, run=run, slices=r17_slices(run, units[unit], card(run)))
        else:
            progress = off.vla_progress(dict(jobs=[run]))
            status.append(dict(batch=name, unit=unit, run=run['run'], complete=progress['complete'],
                               accepted=progress['accepted'], expected=progress['expected']))
            if progress['complete']:
                entries[name] = dict(pipeline=pipeline, unit=unit, run=run['run'], slices=vla_slices(run))
    problems = problems + [f'{s["batch"]}: ' + (s.get('note') or f'{s["accepted"]}/{s["expected"]} accepted')
                           for s in status if not s['complete']]
    gpus = {s['gpu'] for e in entries.values() for s in e['slices']}
    if len(gpus) > 1:
        problems.append(f'batches on different cards {sorted(gpus)}')
    for e in entries.values():
        for s in e['slices']:
            s['sizes'] = sorted(s['sizes'])
            if pipeline != 'R17':
                kinds = [('reference', vla_reference(s['model'], s['mode'], s['long_segment']))]
            else:
                kinds = [('reference', r17_reference(s['model'], s['benchmark'], s['mode'], s['gpu'], False))]
                if s['long_segment'] == 'on':
                    kinds.append(('long_reference', r17_reference(s['model'], s['benchmark'], s['mode'], s['gpu'], True)))
            for kind, (key, spec, problem) in kinds:
                s[kind] = key
                if problem:
                    problems.append(problem)
                if spec:
                    refs[key] = spec
    return entries, refs, list(dict.fromkeys(problems)), status


def build():
    index = dict(version=1, generated_by='seeds_1023/make_index.py', generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 references={}, batches={}, _skipped=[])
    for lane, pipeline, batches, problems in lanes():
        entries, refs, problems, status = lane_entries(pipeline, batches, problems)
        if problems:
            index['_skipped'].append(dict(lane=lane, problems=problems, batches=status))
        else:
            index['batches'].update(entries)
            index['references'].update(refs)
    return index


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=HERE/'SEEDS_INDEX.json')
    a = p.parse_args(argv)
    index = build()
    with tempfile.TemporaryDirectory() as tmp:  # absolute paths only, so the check is location independent
        rs.write(Path(tmp)/'SEEDS_INDEX.json', index)
        rs.read_index(Path(tmp)/'SEEDS_INDEX.json')
    rs.write(a.out, index)
    for name, b in index['batches'].items():
        print('INDEXED', name, b['run'], ' '.join(f'{s["seed"]}/{s["long_segment"]}' for s in b['slices']), flush=True)
    for s in index['_skipped']:
        done = sum(b['complete'] for b in s['batches'])
        print('SKIPPED', s['lane'], f'{done}/{len(s["batches"])} batches finished;', '; '.join(s['problems']), flush=True)
    print(f'Wrote {a.out}: {len(index["batches"])} batches, {len(index["references"])} references, '
          f'{len(index["_skipped"])} lanes skipped', flush=True)


if __name__ == '__main__':
    main()
