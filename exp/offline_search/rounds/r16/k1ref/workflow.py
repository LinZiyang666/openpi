"""CPU preparation/analysis; runtime is an explicit, separate coordinator command."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from .. import analyze as a, conductor as c, conductor_analysis as ca
from .. import retest as rt, retest_analysis as ra
from .. import retest_superseding as rs
from ..prepare import HERE as R16, CELLS as ALL_CELLS, parts, atomic, write_json
from ..cost import pricing

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('R16_K1REF_OUTPUT', CODE)).resolve()
SOURCE_REPO = Path('/home/weiland/projects/openpi')
SERVER_REPO = CODE.parents[4]
# Frozen R16 locks name their original absolute source paths. In the immutable
# worktree those source checks must read the serving bytes, including the three
# restored R16 files. Outcome/fit/identity paths keep their original locations.
if SERVER_REPO != SOURCE_REPO:
    _Reader = a.Reader

    class SnapshotReader(_Reader):
        def read(self, path):
            path = Path(path).resolve()
            if path.suffix == '.py' and path.is_relative_to(SOURCE_REPO) and not path.is_relative_to(ROOT):
                path = SERVER_REPO / path.relative_to(SOURCE_REPO)
            return super().read(path)

    a.Reader = SnapshotReader
CELLS = tuple(cell for cell in ALL_CELLS if parts(cell)[3] == 1)
ORDER = ('c10', 'ir45', 'ir40', 'ir32', 'ir25', 'off')
FIELDS = ('y', 'n', 'v', 'm', 'cost')


def now():
    return datetime.now(timezone.utc).isoformat()


def key(cell, task, level):
    return f'{cell}/{task}/{level}'


def fresh(path):
    a.require(not Path(path).exists(), f'Fresh output required: {path}')


def seal(path, names):
    write_json(Path(path) / 'SEAL.json', {n: c.sha(Path(path) / n) for n in names})


def read_sealed(path, name, reader=None):
    reader = reader or a.Reader()
    digests = reader.json(Path(path) / 'SEAL.json')
    raw = reader.read(Path(path) / name)
    a.require(a.sha(raw) == digests[name], f'Changed sealed {name}')
    return json.loads(raw)


def take(ep, t, **audit):
    return {**{f: getattr(ep, f)[t].tolist() for f in FIELDS}, 'audit': {**ep.audit, **audit}}


def put(records, cell, task, level, record):
    """Keep the earliest identical arm/UID measurement, retaining later provenance."""
    k = key(cell, task, level)
    if k in records:
        records[k].setdefault('also_measured', []).append(record['audit'])
    else:
        records[k] = record


def done(group, reader):
    receipt = reader.json(Path(group['root']) / 'DONE.json')
    a.require(receipt['group_sha256'] == group['group_sha256'], 'Incomplete group')


def verify_code(reader):
    rs.verify(reader)
    lock = reader.json(ROOT / 'CODE_LOCK.json')
    for name, digest in lock.items():
        a.require(a.sha(reader.read(CODE / name)) == digest, f'Changed k1ref code: {name}')


def bootstrap():
    """Audit the existing results once; subsequent phases consume the sealed cache."""
    out = ROOT / 'inventory'; fresh(out)
    reader = a.Reader(); design = rs.verify(reader)
    old = reader.json(R16 / 'dev_results_conductor/dev.json')
    freeze = reader.json(R16 / 'dev_results_conductor/TABLE_FREEZE.json')
    for name, digest in freeze['sha256'].items():
        a.require(a.sha(reader.read(R16 / 'dev_results_conductor' / name)) == digest, 'Changed R16 dev freeze')
    dev = {}; references = {}; sources = {}; originals = {}; original_groups = {}; test = {}; pure = {}
    # The frozen dev result includes exactly the accepted whole cohorts. Retain
    # its input ledger; no inferred below-failure observations enter this cache.
    for cell in CELLS:
        source = design['cells'][cell]; stats = old['cells'][cell]['task_stats']
        fallback = source['fallback']; root = Path(fallback['root'])
        fallback_row = reader.json(root / 'arms.json')[0]
        a.require(old['cells'][cell]['audits']['c10']['arm'] == fallback_row['arm'], 'Wrong paired B reference family')
        manifest = reader.json(fallback_row['manifest'])
        a.require((manifest if isinstance(manifest, list) else manifest['pairs']) == source['pairs'], 'Unpaired B reference')
        references[cell] = [int(stats[str(t)]['c10']['successes']) for t in range(10)]
        for t in range(10):
            for level, row in stats[str(t)].items():
                a.require(row['n'] == 30 and type(row['successes']) is int, 'Invalid frozen dev counts')
                dev[key(cell, t, level)] = row['successes']
    for path in reader.json(R16 / 'conductor_dev_groups.json')['groups']:
        if '_k1_' not in Path(path).parent.name: continue
        group = c.read_group(path, reader); done(group, reader)
        for entry in group['entries']:
            row = entry['row']; cell = row['r16_cell']; level = row['r14_descent_level']
            if cell in CELLS:
                sources[key(cell, 0, level)] = entry
    old_levels, _, old_freeze = rt.frozen_tables(R16 / 'dev_results_conductor')
    for path in reader.json(R16 / 'conductor_test_groups.json')['groups']:
        if '_k1_' not in Path(path).parent.name: continue
        group = c.read_group(path, reader); done(group, reader)
        original_groups[group['model']] = path
        for entry in group['entries']:
            cell = entry['row']['r16_cell']; originals[cell] = entry
            rt.check_original(entry, old_levels, design, old_freeze, reader)
            print('Audit original A:', cell, flush=True)
            ep = ca.load_entry(group, entry, reader, freeze=old_freeze)
            a.require(np.all(ep.y >= 0), 'Incomplete original A')
            for t in range(10):
                put(test, cell, t, old_levels[cell][str(t)], take(ep, t, original_arm=entry['row']['arm'], generation='original'))
    # Load all three generations, including v1 measurements superseded by v3.
    for generation in ('v1', 'v2', 'v3'):
        plan = rt.read_plan(R16 / f'retest_{generation}/plan.json', reader)
        for path in plan['groups']:
            if '_k1_' not in Path(path).parent.name: continue
            from ..retest_runner import checked_group
            group = checked_group(Path(path), reader); done(group, reader)
            for entry in group['entries']:
                if entry['passive']: continue
                ep = ra.load_retest(group, entry, reader, plan['selected_at_utc'])
                cell = entry['row']['r16_cell']
                for t in ep.audit['completed_tasks']:
                    put(test, cell, t, entry['row']['kwargs']['task_levels'][str(t)],
                        take(ep, t, original_arm=entry['original_arm'], generation=generation))
    for rec in design['phase1']:
        m, s, _, k = parts(rec['cell'])
        if k != 1: continue
        print('Audit paired pure A:', m, s, flush=True)
        ep = a.load(rec, rec['arms'][0], rt.PAIRS, 'A', 1, reader)
        pure[f'{m}_{s}'] = c.pack(ep, {})
    reader.stable()
    state = dict(schema='r16.k1ref.inventory.v1', cells=list(CELLS), dev=dev, references=references,
                 dev_sources=sources, originals=originals, original_groups=original_groups, test=test, pure=pure,
                 old_levels={cell: old_levels[cell] for cell in CELLS},
                 inputs=reader.inputs, frozen_dev_input_ledger=old['inputs'], created_at_utc=now())
    write_json(out / 'inventory.json', state); seal(out, ['inventory.json'])
    report = dev_selection(state)
    a.require(report['resolved_changes'] == 31 and len(report['missing']) == 19, '31 / 19 source audit differs')
    write_json(ROOT / 'BASELINE_AUDIT.json', report)
    print('BASELINE: 31 resolved changes / 19 missing dev cohorts', flush=True)


def inventory():
    return read_sealed(ROOT / 'inventory', 'inventory.json')


def dev_pass(successes, reference):
    a.require(type(successes) is int and type(reference) is int and 0 <= successes <= 30 and 0 <= reference <= 30,
              'Dev comparisons require integer successes out of 30')
    return 100 * (successes - reference) >= -5 * 30


def dev_selection(state, *, skip_cells=()):
    levels = {}; missing = []; decisions = {}; changes = []
    for cell in CELLS:
        levels[cell] = {}
        if cell in skip_cells:
            levels[cell] = {str(t): 'off' for t in range(10)}
            decisions[cell] = dict(rule='full_system', tuned=False)
            continue
        for t in range(10):
            selected = 'c10'; visited = []; failure = None; unresolved = None
            for level in ORDER[1:]:
                k = key(cell, t, level)
                if k not in state['dev']:
                    unresolved = level; missing.append(dict(cell=cell, task=t, level=level)); break
                success = state['dev'][k]; passes = dev_pass(success, state['references'][cell][t])
                visited.append(dict(level=level, successes=success, reference_successes=state['references'][cell][t], passes=passes))
                if not passes: failure = level; break
                selected = level
            levels[cell][str(t)] = selected
            decisions[f'{cell}/{t}'] = dict(visited=visited, first_failure=failure, next_missing=unresolved)
            if not unresolved and selected != state['old_levels'][cell][str(t)]:
                changes.append(dict(cell=cell, task=t, old=state['old_levels'][cell][str(t)], level=selected))
    return dict(levels=levels, missing=missing, decisions=decisions, resolved_changes=len(changes), changes=changes,
                next_episodes=30 * len(missing), fully_resolved=not missing, untuned_cells=sorted(skip_cells))


def operational_dev_selection(state):
    rules, missing = off_decisions(state)
    a.require(not missing, 'Complete knob-off A before dev supplement/freeze')
    return dev_selection(state, skip_cells=[cell for cell, r in rules.items() if r['rule'] == 'full_system'])


class IdentityView(ra.IdentityView):
    """Same R16 UID join, extended to B and fixed-level dev methods."""
    def jsonl(self, path):
        entry = self.entry; arm = entry['row']['arm']; original = entry['original_arm']
        selected = set(map(tuple, entry['dispatch_pairs']))
        for raw in self.reader.jsonl(path):
            row = copy.deepcopy(raw)
            for field, belongs in (('task_uid', row.get('yaml_id') == arm), ('uid', row.get('ev') == 'dec')):
                if field not in row or not belongs: continue
                uid = row[field]; pair = a.uid_pair(uid)
                a.require(uid == c.make_task_uid(original, 'eval', *pair) and pair in selected, 'Wrong original UID/unscheduled identity')
                row[field] = c.make_task_uid(arm, 'eval', *pair)
            if row.get('ev') == 'dec':
                level = entry.get('levels', {}).get(str(row['task_id']), entry['row']['r14_descent_level'])
                if 'task_levels' in entry['row']['kwargs'] and (row['vision'] or 'descent_level' in row['r14']):
                    a.require(row['r14'].get('descent_level') == level, 'Wrong measured level')
                if level == 'c10': a.require(not row['vision'] or not row['hit'], 'Pure C10 cache hit')
            yield row


def load_sparse(group, entry, reader):
    tasks = sorted({t for t, _ in entry['dispatch_pairs']})
    view = copy.deepcopy(entry); view['pairs'] = entry['template_pairs']
    ep = ca.load_entry(group, view, IdentityView(reader, entry), freeze={'frozen_at_utc': group['selected_at_utc']})
    a.require(ep.audit['completed_tasks'] == tasks, 'Incomplete/extra whole cohorts')
    return ep


def collect():
    state = inventory(); reader = a.Reader()
    for path in sorted(ROOT.glob('runs/*/plan.json')):
        plan = read_plan(path, reader)
        if plan['phase'] == 'smoke': continue
        for group_path in plan['groups']:
            group = checked_group(group_path, reader); done(group, reader)
            names = {e['row']['arm'] for e in group['entries'] if e['dispatch_pairs']}
            for row in reader.jsonl(Path(group['root']) / 'journal.jsonl'):
                if row.get('accepted') and not row.get('error') and row['status'] in ('done', 'failed'):
                    a.require(row['yaml_id'] in names, 'Accepted unscheduled arm')
            for entry in group['entries']:
                if not entry['dispatch_pairs']: continue
                ep = load_sparse(group, entry, reader); cell = entry['row']['r16_cell']
                for t in ep.audit['completed_tasks']:
                    level = entry['levels'][str(t)]
                    if plan['phase'] == 'dev':
                        k = key(cell, t, level); a.require(k not in state['dev'], 'Dev cohort was already measured')
                        state['dev'][k] = int(ep.y[t].sum())
                    else:
                        a.require(key(cell, t, level) not in state['test'], 'A cohort was already measured')
                        put(state['test'], cell, t, level, take(ep, t, original_arm=entry['original_arm'], generation=plan['name']))
    reader.stable(); state['supplement_inputs'] = reader.inputs
    return state


def emit(name, phase, requests, state, *, previous=None):
    a.require(name.replace('_', '').isalnum() and phase in ('dev', 'test', 'smoke'), 'Invalid generation/phase')
    out = ROOT / 'runs' / name; fresh(out); groups = []
    a.require(len({(r['cell'], r['task'], r['level'] if phase == 'smoke' else '') for r in requests}) == len(requests),
              'Duplicate task in generation')
    reader = a.Reader(); design = a.verify(reader)
    for model in ('pi05', 'groot'):
        selected = [r for r in requests if parts(r['cell'])[0] == model]
        if not selected: continue
        group_root = out / f'r16_k1ref_{model}_{name}'; entries = []
        # Original groups supply the pool digests and fixed R16 runtime metadata.
        original_path = state['original_groups'][model]
        group = c.read_group(original_path, reader)
        buckets = {}
        for r in selected:
            a.require(r['cell'] in CELLS and type(r['task']) is int and 0 <= r['task'] < 10 and r['level'] in ORDER,
                      'Invalid cell/task/level')
            bucket = (r['cell'], r['level']) if phase in ('dev', 'smoke') else (r['cell'], '')
            buckets.setdefault(bucket, []).append(r)
        for (cell, fixed), rows in sorted(buckets.items()):
            tasks = sorted(r['task'] for r in rows)
            if phase in ('dev', 'smoke'):
                source = state['dev_sources'][key(cell, 0, fixed)]
                e = copy.deepcopy(source); e['original_arm'] = e['row']['arm']
                e['template_pairs'] = e['pairs']; e['levels'] = {str(t): fixed for t in range(10)}
                e['dispatch_pairs'] = [p for p in e['pairs'] if p[0] in tasks]
                if phase == 'smoke': e['dispatch_pairs'] = e['dispatch_pairs'][:2]
                e['pairs'] = e['dispatch_pairs']; e.pop('reuse', None)
            else:
                source = state['originals'][cell]; table = dict(state.get('serving_levels', state['old_levels'])[cell])
                table.update({str(r['task']): r['level'] for r in rows})
                e = rt.build_entry(source, group_root, f'r16k1r_{cell}_{name}', table, tasks, design, reader)
                e['template_pairs'] = rt.PAIRS; e['levels'] = table
            entries.append(e)
        # Both suite checkpoints are required by the unchanged launcher. Dormant
        # methods have no episodes and cannot enter the outcome inventory.
        for suite in ('libero_10', 'libero_spatial'):
            if not any(e['row']['suite'] == suite for e in entries):
                e = copy.deepcopy(next(e for e in group['entries'] if e['row']['suite'] == suite))
                e.update(original_arm=e['row']['arm'], template_pairs=e['pairs'], pairs=[], dispatch_pairs=[], levels={})
                entries.append(e)
        group.update(root=str(group_root), phase='dev' if phase in ('dev', 'smoke') else 'test', entries=entries,
                     references=None, k1ref=True, k1ref_plan=str(out / 'plan.json'), selected_at_utc=now(),
                     original_episode_uids=True, yaml_id_is_original_arm=False)
        from exp.offline_search.closed_loop.devset import pool_record
        group['pool_digests'] = {s: pool_record(s, 'B' if group['phase'] == 'dev' else 'A')[1]['per_task_digests']
                                 for s in ('libero_10', 'libero_spatial')}
        group['definitions_sha256'] = {str(p): c.sha(p) for p in sorted(group_root.rglob('*')) if p.is_file()}
        # Dev definitions still live at their immutable original locations.
        for entry in entries:
            for p in (entry['row']['yaml'], entry['row']['manifest'], str(rt.fit_path(entry['row']))):
                group['definitions_sha256'][p] = c.sha(p)
        group.pop('group_sha256', None); group['group_sha256'] = a.sha(c.canonical(group))
        write_json(group_root / 'group.json', group)
        write_json(group_root / 'ROOT.json', dict(group_sha256=group['group_sha256'], definitions_sha256=group['definitions_sha256']))
        groups.append(str(group_root / 'group.json'))
    reader.stable()
    plan = dict(schema='r16.k1ref.plan.v1', name=name, phase=phase, requests=requests, groups=groups,
                group_sha256={p: c.read_group(p)['group_sha256'] for p in groups},
                episodes=sum(len(e['dispatch_pairs']) for p in groups for e in c.read_group(p)['entries']),
                inventory_sha256=c.sha(ROOT / 'inventory/inventory.json'), code_lock_sha256=c.sha(ROOT / 'CODE_LOCK.json'),
                previous_analysis=str(previous) if previous else None,
                previous_sha256=c.sha(previous) if previous else None, inputs=reader.inputs)
    write_json(out / 'plan.json', plan)
    atomic(out / 'queue.txt', ''.join(f'@local {p}\n' for p in groups).encode()); seal(out, ['plan.json', 'queue.txt'])
    print(f'{name}: {len(requests)} tasks / {plan["episodes"]} episodes; {out / "queue.txt"}', flush=True)
    return plan


def read_plan(path, reader=None):
    reader = reader or a.Reader(); path = Path(path)
    plan = read_sealed(path.parent, path.name, reader)
    a.require(plan['inventory_sha256'] == a.sha(reader.read(ROOT / 'inventory/inventory.json')) and
              plan['code_lock_sha256'] == a.sha(reader.read(ROOT / 'CODE_LOCK.json')), 'Changed inventory/code lock')
    if plan['previous_analysis']:
        a.require(a.sha(reader.read(plan['previous_analysis'])) == plan['previous_sha256'], 'Changed previous analysis')
    return plan


def checked_group(path, reader=None):
    reader = reader or a.Reader(); group = c.read_group(path, reader)
    a.require(group.get('k1ref') and group['k'] == 1, 'Only k1ref groups are allowed')
    plan = read_plan(group['k1ref_plan'], reader)
    a.require(plan['group_sha256'].get(str(Path(path).resolve())) == group['group_sha256'], 'Unregistered group')
    actual = []
    for e in group['entries']:
        a.require(e['pairs'] == e['dispatch_pairs'], 'Changed sparse dispatch')
        for t in sorted({t for t, _ in e['dispatch_pairs']}):
            expected = [p for p in e['template_pairs'] if p[0] == t]
            if plan['phase'] == 'smoke': expected = expected[:2]
            a.require([p for p in e['dispatch_pairs'] if p[0] == t] == expected, 'Incomplete cohort definition')
            actual.append(dict(cell=e['row']['r16_cell'], task=t, level=e['levels'][str(t)]))
    a.require(sorted(actual, key=lambda r: (r['cell'], r['task'])) ==
              sorted([r for r in plan['requests'] if parts(r['cell'])[0] == group['model']],
                     key=lambda r: (r['cell'], r['task'])), 'Plan/dispatch mismatch')
    return group


def check_smoke():
    plan = read_plan(ROOT / 'runs/smoke/plan.json'); counts = {}
    for path in plan['groups']:
        group = checked_group(path); done(group, a.Reader())
        expected = {c.make_task_uid(e['original_arm'], 'eval', *p)
                    for e in group['entries'] for p in e['dispatch_pairs']}
        accepted = c.completed_rows(Path(group['root']) / 'journal.jsonl')
        a.require(set(accepted) == expected and len(expected) == 8, 'Smoke needs 2 accepted episodes per arm')
        progress = json.loads((Path(group['root']) / 'progress.json').read_bytes())
        a.require(progress['group_sha256'] == group['group_sha256'] and progress['worker_census'], 'Missing attested smoke workers')
        counts[group['model']] = len(accepted)
    a.require(counts == {'pi05': 8, 'groot': 8}, 'Both model smokes required')
    print('SMOKE_OK: 16 accepted episodes; inspect startup/memory/attestation as in HANDBACK.md')


def pure_for(state, cell):
    m, s, _, _ = parts(cell)
    return c.unpack(state['pure'][f'{m}_{s}'])


def off_decisions(state):
    rows = {}; missing = []
    for cell in CELLS:
        records = [state['test'].get(key(cell, t, 'off')) for t in range(10)]
        missing.extend(dict(cell=cell, task=t, level='off') for t, r in enumerate(records) if r is None)
        if all(r is not None for r in records):
            success = sum(sum(r['y']) for r in records); reference = int(pure_for(state, cell).y.sum())
            rows[cell] = dict(successes=success, reference_successes=reference,
                              sr=success/500, baseline_sr=reference/500,
                              ir=sum(sum(r['cost']) for r in records)/sum(sum(r['n']) for r in records),
                              rule='full_system' if success >= reference else 'knob')
    return rows, missing


def frozen():
    levels = read_sealed(ROOT / 'freeze', 'levels.json')
    record = read_sealed(ROOT / 'freeze', 'TABLE_FREEZE.json')
    a.require(record['inventory_sha256'] == c.sha(ROOT / 'inventory/inventory.json') and
              record['code_lock_sha256'] == c.sha(ROOT / 'CODE_LOCK.json'), 'Changed freeze inventory/code')
    for path, info in record['supplement_inputs'].items():
        a.require(c.sha(path) == info['sha256'], f'Changed frozen supplement: {path}')
    a.require(set(levels) == set(CELLS), 'Need all 16 frozen tables')
    return levels


def raise_level(level, deficit):
    a.require(level in ORDER and type(deficit) is int, 'Integer deficit and known level required')
    if deficit <= 2 or level == 'c10': return level
    return 'c10' if deficit >= 5 else ORDER[ORDER.index(level) - 1]


def resolve_test(state, initial, *, allow_raise=True):
    rules, off_missing = off_decisions(state)
    a.require(not off_missing, 'Complete knob-off A before tuning')
    levels = copy.deepcopy(initial); missing = []; trail = []; finished = True
    for cell in CELLS:
        for t in range(10):
            level = 'off' if rules[cell]['rule'] == 'full_system' else levels[cell][str(t)]
            while True:
                record = state['test'].get(key(cell, t, level))
                if record is None:
                    missing.append(dict(cell=cell, task=t, level=level)); finished = False; break
                deficit = int(pure_for(state, cell).y[t].sum()) - sum(record['y'])
                new = level if rules[cell]['rule'] == 'full_system' else raise_level(level, deficit)
                trail.append(dict(cell=cell, task=t, level=level, successes=sum(record['y']), deficit=deficit, next_level=new))
                if new == level: break
                if not allow_raise:
                    finished = False; break
                level = new
            levels[cell][str(t)] = level
    return dict(levels=levels, missing=missing, trail=trail, complete=finished, rules=rules,
                next_episodes=50 * len(missing))


def matrix(state, cell, levels):
    records = [state['test'][key(cell, t, levels[str(t)])] for t in range(10)]
    return a.Episodes(**{f: np.asarray([r[f] for r in records]) for f in FIELDS},
                      audit={str(t): r['audit'] for t, r in enumerate(records)})


def calculate(arms, draws=5000):
    """Reuse R16's estimator, then give its single k1 reference a truthful label."""
    result = a.calculate_test(arms, draws=draws)
    for group in (result['cells'], result['pooled'], result['pooled_by_k']):
        for row in group.values():
            row['arms']['k1_pure'] = row['arms'].pop('full_c10')
            row['arms'].pop('full_descent')
            old_key = 'task_vs_full_c10' if 'task_vs_full_c10' in row['comparisons'] else 'full_c10'
            row['comparisons'] = {'task_vs_k1_pure': row['comparisons'][old_key]}
    return result


def analysis(state, selection, draws=5000):
    a.require(selection['complete'], 'Final analysis requires every task pass or c10')
    all_arms = {}; before = {}
    for cell in CELLS:
        reference = pure_for(state, cell); task = matrix(state, cell, selection['levels'][cell])
        # The existing estimator has two reference slots. Both intentionally
        # contain the same paired k1 pure reference; no full-step reference enters.
        all_arms[cell] = dict(task=task, full_c10=reference, full_descent=reference)
        before[cell] = dict(task=matrix(state, cell, state['old_levels'][cell]), full_c10=reference, full_descent=reference)
    result = calculate(all_arms, draws=draws)
    result['before'] = calculate(before, draws=draws)
    result['selection'] = selection
    result['reference'] = 'phase-1 k1 pure, same A task/init identities'
    result['post_test_selected'] = True
    result['measurement_only'] = False
    result['full_system_rule'] = selection['rules']
    result['merged_episodes'] = {cell: c.pack(arms['task'], {}) for cell, arms in all_arms.items()}
    result['summary'] = [dict(model=parts(cell)[0], suite='libero_10' if parts(cell)[1] == 'l10' else 'libero_spatial',
                              size=parts(cell)[2], sr=arms['task'].stats()['sr'], baseline_sr=arms['full_c10'].stats()['sr'],
                              diff_pp=100*(arms['task'].stats()['sr']-arms['full_c10'].stats()['sr']),
                              ir=arms['task'].stats()['ir'], pure_ir=arms['full_c10'].stats()['ir'], rule=selection['rules'][cell]['rule'])
                         for cell, arms in all_arms.items()]
    return result


def write_analysis(out, result):
    fresh(out); write_json(out / 'retest.json', result); write_json(out / 'summary.json', result['summary'])
    write_json(out / 'levels.json', result['selection']['levels'])
    lines = ['# R16 k1-reference retest', '', 'Post-test selection; descriptive paired intervals do not account for selection.', '',
             '| cell | before SR | after SR | before IR | after IR | rule |', '|---|---:|---:|---:|---:|---|']
    for cell in CELLS:
        old = result['before']['cells'][cell]['arms']['task']; new = result['cells'][cell]['arms']['task']
        lines.append(f'| {cell} | {old["sr"]:.4f} | {new["sr"]:.4f} | {old["ir"]:.6f} | {new["ir"]:.6f} | {result["selection"]["rules"][cell]["rule"]} |')
    lines += ['', '| cell | off SR | k1 pure SR | off IR | decision |', '|---|---:|---:|---:|---|']
    for cell, row in result['full_system_rule'].items():
        lines.append(f'| {cell} | {row["sr"]:.4f} | {row["baseline_sr"]:.4f} | {row["ir"]:.6f} | {row["rule"]} |')
    lines += ['', '| task | final level | SR | deficit |', '|---|---|---:|---:|']
    for cell in CELLS:
        ep = result['merged_episodes'][cell]
        for t in range(10):
            last = next(r for r in reversed(result['selection']['trail']) if (r['cell'], r['task']) == (cell, t))
            lines.append(f'| {cell} / {t} | {result["selection"]["levels"][cell][str(t)]} | {sum(ep["y"][t])/50:.4f} | {last["deficit"]} |')
    lines += ['', a.markdown(result, 'k1 pure paired comparison').replace('Measure only.', 'Post-test selection; descriptive intervals.'),
              '', 'D5 pricing; 5,000 shared suite/task/init bootstrap draws, seed 20261005. Repeated references are weights, not independent trials.']
    atomic(out / 'RETEST.md', ('\n'.join(lines) + '\n').encode())
    seal(out, ['retest.json', 'summary.json', 'levels.json', 'RETEST.md'])


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='action', required=True)
    sub.add_parser('bootstrap'); sub.add_parser('verify'); sub.add_parser('smoke'); sub.add_parser('check-smoke')
    sub.add_parser('off'); sub.add_parser('freeze')
    p = sub.add_parser('dev'); p.add_argument('--generation', type=int, required=True)
    p = sub.add_parser('test'); p.add_argument('--generation', type=int, default=0); p.add_argument('--previous', type=Path)
    p = sub.add_parser('analyze'); p.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.action == 'bootstrap': bootstrap(); return
    reader = a.Reader(); verify_code(reader); reader.stable()
    if args.action == 'verify':
        inventory(); print('K1REF_CODE_AND_INVENTORY_OK'); return
    if args.action == 'check-smoke': check_smoke(); return
    if args.action == 'smoke':
        state = inventory()
        # Two levels × two suites × two models; 2 episodes per arm, 4 workers.
        req = [dict(cell=f'{m}_{s}_50_k1', task=0, level=l) for m in ('pi05', 'groot') for s in ('l10', 'spatial') for l in ('c10', 'ir45')]
        emit('smoke', 'smoke', req, state); return
    state = collect()
    if args.action == 'off':
        _, missing = off_decisions(state)
        state['serving_levels'] = {cell: {str(t): 'off' for t in range(10)} for cell in CELLS}
        emit('off', 'test', missing, state); return
    if args.action == 'dev':
        selection = operational_dev_selection(state)
        if not selection['missing']: print('DEV_COMPLETE; freeze next'); return
        emit(f'dev{args.generation:02d}', 'dev', selection['missing'], state); return
    if args.action == 'freeze':
        selection = operational_dev_selection(state); a.require(selection['fully_resolved'], 'Missing dev cohorts; emit next wave')
        out = ROOT / 'freeze'; fresh(out)
        write_json(out / 'levels.json', selection['levels']); write_json(out / 'dev.json', selection)
        write_json(out / 'TABLE_FREEZE.json', dict(frozen_at_utc=now(), selection_reference='paired same-k pure B family',
                   inventory_sha256=c.sha(ROOT / 'inventory/inventory.json'), code_lock_sha256=c.sha(ROOT / 'CODE_LOCK.json'),
                   supplement_inputs=state['supplement_inputs']))
        seal(out, ['levels.json', 'dev.json', 'TABLE_FREEZE.json']); return
    if args.action == 'test':
        a.require(args.generation >= 0 and bool(args.previous) == (args.generation > 0), 'Raise generation requires previous analysis')
        if args.previous:
            previous = read_sealed(args.previous.parent, args.previous.name)
            a.require(previous['generation'] == args.generation - 1 and not previous['selection']['complete'], 'Wrong previous generation')
            a.require(previous['inventory_sha256'] == c.sha(ROOT / 'inventory/inventory.json'), 'Wrong previous inventory')
            frozen()
            initial = previous['selection']['levels']
        else: initial = frozen()
        selection = resolve_test(state, initial, allow_raise=args.generation > 0)
        state['serving_levels'] = selection['levels']
        out = ROOT / 'analyses' / f'g{args.generation:02d}'; fresh(out)
        write_json(out / 'analysis.json', dict(generation=args.generation, selection=selection,
                   inventory_sha256=c.sha(ROOT / 'inventory/inventory.json'), inputs=state['supplement_inputs']))
        seal(out, ['analysis.json'])
        emit(f'test{args.generation:02d}', 'test', selection['missing'], state, previous=out / 'analysis.json')
        print('READY_FOR_FINAL_ANALYSIS' if selection['complete'] else f'After queue completes, emit generation {args.generation+1} from {out / "analysis.json"}')
        return
    selection = resolve_test(state, frozen()); write_analysis(args.output, analysis(state, selection))


if __name__ == '__main__': main()
