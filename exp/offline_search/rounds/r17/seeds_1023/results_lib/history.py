"""Read frozen summaries, and independently replay their recorded existing runs."""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

from . import native as n

R17 = Path(__file__).resolve().parents[2]
SOURCES = (R17/'longseg_off_1022/RESULTS_OFF.json', R17/'longseg_1017/RESULTS_1022.json',
           R17/'filter_failed_knob/q3_port_1020/RESULTS_1022.json')


def line_of(row, segment):
    benchmark = row.get('benchmark', 'libero')
    return dict(model=row['model'], benchmark=benchmark, mode=row.get('mode', row.get('step_mode')),
                long_segment='default' if (row['model'], benchmark) in (('cosmos', 'libero'), ('pi05', 'mt50')) else segment,
                seed=8 if benchmark == 'mt50' else 20261002, gpu=row.get('gpu'))


def normalized(row, source, segment):
    line = line_of(row, segment)
    if 'off' in row and 'detail' in row:
        summary, raw = row['off'], row['detail']
        detail = raw.get('filtered', raw)
        line['gpu'] = raw.get('gpu')
    elif 'final' in row:
        summary, detail = n.off.summary_vla(row), row
        # The original Q3 summary omits gpu; these are its formal-A references.
        line['gpu'] = row.get('gpu') or ('H100' if (line['model'], line['mode']) == ('pi05', 'full') else 'RTX 4090')
    else:
        detail = row['filtered']
        if 'step_mode' in row:  # unaffected Cosmos uses formal_a_analysis's row schema
            detail = dict(detail, native_pure_target=row['reference'])
            detail['tolerance_all_tasks'] = n.off.vla_meets(detail)
        summary = n.off.summary_r17({'filtered': detail})
    detail = {k: detail[k] for k in ('tasks', 'measured_levels') if k in detail}
    return dict(line, suite=row['suite'], size=row['size'], summary=summary, detail=detail,
                source=str(source), status='complete' if summary.get('n') == (2500 if line['benchmark'] == 'mt50' else 500)
                and all(v is not None for v in summary['per_task_levels'].values()) else 'pending')


def read_existing(*, replay=False):
    rows, specs = [], {}
    for source, segment in zip(SOURCES, ('off', 'on', 'on')):
        document = n.off.read(source)
        for raw in document['rows']:
            if raw.get('arm', 'filtered') != 'filtered':
                continue
            row = normalized(raw, source, segment)
            rows.append(row)
            if replay:
                line_key = n.key(row)
                if line_key not in specs:
                    detail = raw.get('detail', raw)
                    specs[line_key] = dict(line={k: row[k] for k in (*n.DIMS, 'gpu')}, source=str(source),
                                           raw=detail, expected=[])
                # Only compact expectations are sent to worker processes.
                specs[line_key]['expected'].append(dict(suite=row['suite'], size=row['size'],
                                                        summary=row['summary'], chain=chain_signature(row['detail'])))
        del document
    return rows, list(specs.values())


def chain_signature(detail):
    return {t: dict(selected=d.get('selected'), first_failure=d.get('first_failure'),
                    incomplete=d.get('incomplete'),
                    levels={l: {k: e.get(k) for k in ('n', 'successes', 'passes')} for l, e in d['levels'].items()})
            for t, d in detail['tasks'].items()}


def replay(spec):
    """Only existing summaries, recorded runs, queues and native readers; no rebuilds."""
    line, raw = spec['line'], spec['raw']
    model, mode = line['model'], line['mode']
    vla = line['benchmark'] == 'libero' and model in ('pi05', 'groot')
    data, long_refs = {}, None
    if vla:
        from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import analysis, results
        from exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022 import build as vb
        with patch.object(n.fa, 'legacy_data', results.ledger):
            refs = vb.references(model, mode)[0] if line['long_segment'] == 'off' else n.fa.vla_references(model, mode)
            for run in raw['runs']:
                job = dict(run, model=model, mode=mode, pipeline='R15' if mode == 'full' else 'R16')
                progress = (n.off.vla_progress(dict(jobs=[dict(job, episodes=500*len(n.off.read(Path(job['run'])/'arms.json')))]))
                            if mode == 'full' else None)
                if progress is not None and not progress['complete']:
                    raise ValueError(f'Incomplete historical VLA run: {progress}')
                if mode == 'reduced' and not (Path(job['run'])/'DONE.json').exists():
                    raise ValueError(f'Incomplete historical VLA run: {job["run"]}')
                for cell, arms in analysis.load_job(job).items():
                    for level, episodes in arms.items():
                        n.merge(data.setdefault(cell, {}).setdefault(level, {}), episodes, r17=False)
    elif model == 'cosmos':
        from exp.offline_search.rounds.r17.filter_failed_knob import formal_a as f
        from exp.offline_search.rounds.r17.knob.test import load_controls
        jobs = [j for j in n.fa.active_jobs() if (j['model'], j['mode'], j['arm']) == (model, mode, 'filtered')]
        for job in jobs:
            for cell, arms in n.fa.load_r17(job).items():
                for level, episodes in arms.items():
                    n.merge(data.setdefault(cell, {}).setdefault(level, {}), episodes, r17=True)
        refs = {s: v for (_, s), v in load_controls(f.EVAL, {f.LABELS[model]}, mode,
                                                   [f.LABELS[model]+'='+f.PURE[model][mode]]).items()}
    else:
        detail = raw['filtered']
        for run in detail['runs']:
            if not n.off.r17_progress(run['run'])['complete']:
                raise ValueError(f'Incomplete historical R17 run: {run["run"]}')
            for (_, suite, size), arms in n.an.lane_data(run['run']).items():
                for level, episodes in arms.items():
                    n.merge(data.setdefault((suite, size), {}).setdefault(level, {}), episodes, r17=True)
        entries = detail.get('reference_runs') or {'pure': {'run': detail['reference_run']}}
        reference_key = f'{model}:{line["benchmark"]}:{mode}:{line["gpu"]}'
        registry = {reference_key: entries['pure']}
        if line['long_segment'] == 'on':
            registry[reference_key+':long'] = entries['pure_long']
        with n.models(), patch.object(n.an, 'registry', lambda: registry):
            refs = n.an.reference(model, line['benchmark'], mode, line['gpu'])
            if line['long_segment'] == 'on':
                long_refs = n.an.reference(model, line['benchmark'], mode, line['gpu'], long=True)
    actual = {(r['suite'], r['size']): r for r in n.calculate(line, data, refs, long_refs)}
    differences = []
    expected_cells = {(r['suite'], r['size']) for r in spec['expected']}
    for cell in sorted(actual.keys()-expected_cells):
        differences.append(dict(cell=cell, field='row', expected='absent', actual='extra'))
    for expected in spec['expected']:
        cell = expected['suite'], expected['size']
        row = actual.get(cell)
        if row is None:
            differences.append(dict(cell=cell, field='row', expected='present', actual='missing'))
            continue
        compare(expected['summary'], row['summary'], 'summary', cell, differences)
        compare(expected['chain'], chain_signature(row['detail']), 'chain', cell, differences)
    return dict(line=line, source=spec['source'], checked=len(spec['expected']), differences=differences)


def compare(expected, actual, field, cell, differences):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for k, value in expected.items():
            compare(value, actual.get(k), field+'.'+str(k), cell, differences)
    elif isinstance(expected, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12):
            differences.append(dict(cell=cell, field=field, expected=expected, actual=actual))
    elif expected != actual:
        differences.append(dict(cell=cell, field=field, expected=expected, actual=actual))
