"""Seed-aware routing around the existing outcome, selection and cost readers.

No IR formula lives here. R17 uses analyze_cache.load_lane (the reader beneath
results_off.r17_lane); VLA uses the same R15/R16 readers and ledger as results_off.
Only the explicit index determines which seed/segment receives an arm/group.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from unittest.mock import patch

from exp.offline_search.rounds.r17.longseg_off_1022 import results_off as off

an, c, fa = off.an, off.c, off.fa
LABELS = dict(an.LABELS, pi05='π0.5')
DIMS = ('model', 'benchmark', 'mode', 'long_segment', 'seed')


def key(line):
    return tuple(line[k] for k in DIMS)


@contextmanager
def models():
    # The existing 4090 profile selector supports MT50's pi05 profile; its
    # presentation label table predates this lane. No pricing rule is changed.
    with patch.dict(an.LABELS, LABELS):
        yield


def merge(target, incoming, *, r17):
    if r17:
        c.merge(target, incoming)
    else:
        overlap = target.keys() & incoming.keys()
        if any(target[k] != incoming[k] for k in overlap):
            raise ValueError('Conflicting VLA episodes; index must name the final run only')
        target.update(incoming)


def cohort(data, suite, planned=None):
    """Whole 50-init tasks: every task of the suite, or exactly the planned ones (a descent-round arm)."""
    tasks = {k[1] for k in data}
    return ((len(tasks) == c.task_count(suite) if planned is None else tasks == {str(t) for t in planned}) and
            set(data) == {(suite, t, i) for t in tasks for i in range(50)})


def r17_evidence(root):
    """Retain load_knob's TEST-pool/init checks before routing new group names."""
    sources = c.cache.Sources()
    for segment, _ in c.cache.resume_segments(root, sources):
        plan = segment/'FULL_EPISODE_PLAN.json'
        if not plan.exists():
            plan = segment/'EPISODE_PLAN.json'
        for task in off.read(plan)['tasks']:
            suite, _, init = c.cache.canonical(task)
            if suite not in c.SUITES:
                continue
            if task['extra'].get('init_pool', 'A') != 'A' or not 0 <= init < 50:
                raise ValueError(f'{root}: expected TEST A, init 0..49: {task["task_uid"]}')
    return c.cache.load_lane(root, c.cache.receipt_snapshot(root), sources, c.SIZES, suites=c.SUITES)


def load_batch(item):
    """One process per physical run, even when it contains four seed/segment slices."""
    name, batch = item
    root = Path(batch['run'])
    slices = batch['slices']
    outputs = [dict(line=s, data={}) for s in slices]
    pipeline = batch['pipeline']
    if pipeline == 'R17':
        progress = off.r17_progress(root)
        if not progress['complete']:
            raise ValueError(f'{name}: incomplete batch: {progress}')
        loaded = r17_evidence(root)
        seen = [set() for _ in slices]
        for (model, size, group, suite), result in loaded.items():
            if group == 'control':
                continue
            matches = [i for i, s in enumerate(slices) if group in s['groups'] and
                       suite in s['suites'] and size in s['sizes']]
            if len(matches) != 1:
                raise ValueError(f'{name}: unmapped/ambiguous group {(suite, size, group)}')
            i = matches[0]; s = slices[i]
            if model != LABELS[s['model']]:
                raise ValueError(f'{name}: model mismatch: {model}, {s["model"]}')
            level = s['groups'][group]
            if level not in off.ORDER:
                raise ValueError(f'{name}: invalid knob level {level}')
            seen[i].add(group)
            arms = outputs[i]['data'].setdefault((suite, size), {})
            if level in arms:
                raise ValueError(f'{name}: multiple groups map to {(key(s), suite, size, level)}')
            arms[level] = result['data']
        for s, used in zip(slices, seen):
            if used != set(s['groups']):
                raise ValueError(f'{name}: missing groups {set(s["groups"])-used}')
    else:
        from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import analysis, results
        a, ca = analysis.a, analysis.ca
        entries = off.read(root/'arms.json') if pipeline == 'R15' else off.read(root/'group.json')['entries']
        record = off.read(root/'ROOT.json') if pipeline == 'R15' else off.read(root/'group.json')
        if pipeline == 'R16' and not (root/'DONE.json').exists():
            raise ValueError(f'{name}: missing DONE.json')
        reader = a.Reader(); seen = [set() for _ in slices]
        for entry in entries:
            row = entry if pipeline == 'R15' else entry['row']
            arm = row['arm']
            matches = [i for i, s in enumerate(slices) if arm in s['arms']]
            if len(matches) != 1:
                raise ValueError(f'{name}: unmapped/ambiguous arm {arm}')
            i = matches[0]; s = slices[i]; seen[i].add(arm)
            if row.get('init_pool') != 'A' or row.get('dev'):
                raise ValueError(f'{name}/{arm}: expected TEST A')
            long = '--os-blind' in row['plugin_args']
            if long != (s['long_segment'] == 'on'):
                raise ValueError(f'{name}/{arm}: segment flags disagree with index')
            if pipeline == 'R15':
                if not (root/'runs'/arm/'summary.json').exists():
                    raise ValueError(f'{name}/{arm}: incomplete arm')
                k = 1 if s['mode'] == 'reduced' else 10 if s['model'] == 'pi05' else 8
                ep = a.load(record, arm, results.PAIRS, 'A', k, reader)
                level, planned = row['r14_descent_level'], row.get('tasks')
            else:
                ep = ca.load_entry(record, dict(entry, pairs=results.PAIRS), reader)
                level, planned = entry['formal_grid_level'], sorted({t for t, _ in entry['pairs']})
            if level not in off.f.VLA_LEVELS:
                raise ValueError(f'{name}/{arm}: invalid knob level {level}')
            suite, size = row['suite'], row['r10_size']
            if suite not in s['suites'] or size not in s['sizes']:
                raise ValueError(f'{name}/{arm}: unexpected suite/size')
            data = results.ledger(ep, suite)
            if not cohort(data, suite, planned):
                raise ValueError(f'{name}/{arm}: incomplete cohort ({len(data)}/{50*len(planned or range(10))})')
            merge(outputs[i]['data'].setdefault((suite, size), {}).setdefault(level, {}), data, r17=False)
        for s, used in zip(slices, seen):
            if used != set(s['arms']):
                raise ValueError(f'{name}: missing arms {set(s["arms"])-used}')
    return name, outputs


def load_reference(spec):
    """Explicit pure runs/arms. Same-card and same-mode assertions are in the index."""
    result = {}
    if spec['pipeline'] == 'R17':
        root = Path(spec['run'])
        progress = off.r17_progress(root)
        if not progress['complete']:
            raise ValueError(f'Incomplete reference: {progress}')
        loaded = r17_evidence(root)
        groups = set(spec['groups'])  # "control" for a standalone control; exact pure group otherwise
        seen = set()
        for (model, _, group, suite), row in loaded.items():
            if group not in groups:
                continue
            if model != LABELS[spec['model']]:
                raise ValueError(f'{root}: reference model mismatch')
            seen.add(group)
            data = {k: dict(v, pure=True) for k, v in row['data'].items()}
            merge(result.setdefault(suite, {}), data, r17=True)
        if seen != groups:
            raise ValueError(f'{root}: missing pure groups {groups-seen}')
    else:
        from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import analysis, results
        a = analysis.a
        reader = a.Reader()
        k = 1 if spec['mode'] == 'reduced' else 10 if spec['model'] == 'pi05' else 8
        for job in spec['jobs']:
            root, arm = Path(job['run']), job['arm']
            if not (root/'runs'/arm/'summary.json').exists():
                raise ValueError(f'Incomplete reference: {root}/{arm}')
            row = next(r for r in off.read(root/'arms.json') if r['arm'] == arm)
            if row.get('init_pool') != 'A' or row.get('dev'):
                raise ValueError(f'{root}/{arm}: reference is not TEST A')
            ep = a.load(off.read(root/'ROOT.json'), arm, results.PAIRS, 'A', k, reader)
            merge(result.setdefault(row['suite'], {}), results.ledger(ep, row['suite']), r17=False)
    for suite, data in result.items():
        if not cohort(data, suite):
            raise ValueError(f'Incomplete pure cohort {suite}: {len(data)}')
    return result


def calculate(line, data, refs, long_refs=None):
    """The filtered-arm bodies of results_off / longseg_1017 / q3_port results."""
    from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import results
    model, mode, gpu = line['model'], line['mode'], line['gpu']
    vla = line['benchmark'] == 'libero' and model in ('pi05', 'groot')
    selections = {}
    with models():
        if line['benchmark'] == 'mt50':
            # analyze_mt50 writes diagnostics. Keep them temporary, never in an old
            # study, and never at a seed-less persistent analysis/evidence key.
            with tempfile.TemporaryDirectory(prefix=f'seeds_{model}_{line["seed"]}_{line["long_segment"]}_') as tmp:
                native = off.descent.analyze_mt50(
                    {(LABELS[model], s, n): v for (s, n), v in data.items()},
                    {(LABELS[model], s): v for s, v in refs.items()}, Path(tmp), mode,
                    prices=lambda label, suite: an.prices(label, suite, gpu), summary=an.decision_summary)
            for row in native['cells']:
                chosen = {}
                for task, d in row['tasks'].items():
                    pool = refs.get(row['suite'], {}) if d['selected'] == 'pure' else data[row['suite'], row['size']].get(d['selected'], {})
                    chosen.update({k: v for k, v in pool.items() if k[1] == task})
                selections[row['suite'], row['size']] = row, chosen
        else:
            for cell, arms in data.items():
                d = fa.choose(arms, refs.get(cell[0], {}), off.f.VLA_LEVELS if vla else off.ORDER)
                selections[cell] = d, d['data']
        rows = []
        for (suite, size), (selection, chosen) in sorted(selections.items()):
            target = refs.get(suite, {})
            if not cohort(target, suite):
                raise ValueError(f'{key(line)} {suite}: missing/incomplete pure reference')
            measured = data[suite, size]
            off_level = 'off' if vla else 'koff'
            if set(measured.get(off_level, {})) != set(target):
                raise ValueError(f'{key(line)} {suite}/{size}: incomplete initial off cohort')
            if any(t.get('incomplete') or t.get('selected') is None for t in selection['tasks'].values()):
                raise ValueError(f'{key(line)} {suite}/{size}: incomplete descent chain')
            if vla:
                detail = dict(final=results.stats(chosen, model, mode, suite), pure=results.stats(target, model, mode, suite),
                              no_knob=selection['no_knob'], tasks=selection['tasks'],
                              per_task_levels={t: d['selected'] for t, d in selection['tasks'].items()},
                              measured_levels={l: results.stats(v, model, mode, suite) for l, v in measured.items()})
                summary = off.summary_vla(detail)
            else:
                fallback = (long_refs or {}).get(suite, {}) if line['long_segment'] == 'on' else target
                if line['long_segment'] == 'on' and not cohort(fallback, suite):
                    raise ValueError(f'{key(line)} {suite}: missing/incomplete long pure reference')
                detail, chosen = an.whole_segment_result(selection, chosen, target, fallback, model, mode, suite, gpu)
                detail.pop('native_length_fallback_diagnostic', None)
                detail.update(native_pure_target=an.stats(target, model, mode, suite, gpu),
                              measured_levels={l: an.stats(measured.get(l, {}), model, mode, suite, gpu) for l in off.ORDER})
                summary = off.summary_r17({'filtered': detail})
            if not cohort(chosen, suite):
                raise ValueError(f'{key(line)} {suite}/{size}: incomplete final cohort')
            rows.append(dict(line, suite=suite, size=size, summary=summary, detail=detail))
        return rows
