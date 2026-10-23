"""Owner-selected, post-test retests. Emission reads definitions, never outcomes.

New roots only; original frozen tables, fits, groups and results stay immutable.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re

from . import analyze as a
from . import conductor as c
from .prepare import HERE, TRACE, CELLS, parts, frozen_tables, atomic, write_json

PAIRS = [[t, i] for t in range(10) for i in range(50)]
LEVELS = ('off', 'ir25', 'ir32', 'ir40', 'ir45', 'c10')
LOCK = HERE / 'RETEST_LOCK.json'
BASELINES = ('DESIGN_LOCK.json', 'CONDUCTOR_LOCK.json', 'design.json',
             'PROTOCOL.md', 'protocol.json', 'COORDINATOR_DECISIONS.md',
             'CONDUCTOR_AMENDMENT.json')
SELECTION = 'Owner-selected after observing test A; exploratory, not the original frozen confirmatory result'


def verify(reader):
    design = a.verify(reader)
    lock = reader.json(LOCK)
    a.require(lock['schema'] == 'r16.retest.lock.v1', 'Wrong retest lock')
    return design


def adjusted_tables(levels, changes):
    a.require(isinstance(changes, list) and changes, 'Need a nonempty JSON changes list')
    result = copy.deepcopy(levels); seen = set()
    for change in changes:
        a.require(isinstance(change, dict) and set(change) == {'cell', 'task', 'from', 'to'},
                  'Changes require exactly cell/task/from/to')
        cell, task, old, new = (change[k] for k in ('cell', 'task', 'from', 'to'))
        a.require(isinstance(cell, str) and cell in CELLS and cell in levels, 'Unknown cell')
        a.require(type(task) is int and 0 <= task < 10, 'Task must be an integer 0..9')
        a.require((cell, task) not in seen, 'Duplicate changed task'); seen.add((cell, task))
        a.require(old == levels[cell][str(task)], 'from disagrees with the frozen table')
        # The owner-authored file is authoritative, including the explicitly
        # allowed off level. Do not infer replacements or veto a valid target.
        a.require(new in LEVELS and old in LEVELS and new != old,
                  'to must be a different existing level (c10 is the same-k top fallback)')
        result[cell][str(task)] = new
    return result


def fit_path(row):
    args = row['plugin_args']
    return Path(args[args.index('--os-fit-artifact') + 1])


def check_original(entry, levels, design, freeze, reader):
    row = entry['row']; cell = row['r16_cell']
    caps = [design['cells'][cell]['periodic_caps'][levels[cell][str(t)]] for t in range(10)]
    table = reader.json(Path(entry['source_root']) / 'descent_table.json')
    a.require(entry['freeze'] == freeze and table['freeze'] == freeze and
              table['levels'] == levels[cell] and table['caps'] == caps and
              table['uniform'] is None and row['kwargs']['task_levels'] == levels[cell] and
              row['kwargs']['task_caps'] == caps, 'Wrong original frozen serving table')
    a.require(entry['pairs'] == PAIRS and row['init_pool'] == 'A' and row['recording'] is False,
              'Wrong original test identities/pool')
    return caps


def adjusted_blob(raw, row, levels, caps):
    """Change only the frozen method's table; never fit or reselect lower layers."""
    blob = pickle.loads(raw)
    a.require(blob['spec'] == row['method'] and blob['kwargs'] == row['kwargs'], 'Source fit mismatch')
    method = blob['method']
    a.require(method.task_levels == row['kwargs']['task_levels'] and
              list(method.knob_settings['K']) == row['kwargs']['task_caps'] and
              not getattr(method, 'leave_own_episode_out', False), 'Wrong test fit table/LOEO')
    method.set_table(levels, caps)
    method.r14_fit.update(levels=dict(levels), K=list(caps))
    blob['kwargs'].update(task_levels=dict(levels), task_caps=list(caps))
    for field in ('r14_manifest', 'descent_manifest'):
        blob[field] = copy.deepcopy(method.r14_fit)
    return pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)


def build_entry(source, group_root, name, table, tasks, design, reader, *, passive=False):
    row = copy.deepcopy(source['row']); cell = row['r16_cell']; original = row['arm']
    caps = [design['cells'][cell]['periodic_caps'][table[str(t)]] for t in range(10)]
    root = group_root / 'definitions' / name
    raw = reader.read(fit_path(row))
    a.require(a.sha(raw) == source['fit_sha256'], 'Changed original fitted bytes')
    if not passive:
        raw = adjusted_blob(raw, row, table, caps)
        row['kwargs'].update(task_levels=dict(table), task_caps=caps)
    fit = root / 'fit.pkl'; atomic(fit, raw)
    row['plugin_args'][row['plugin_args'].index('--os-fit-artifact') + 1] = str(fit)
    for field, name_on_disk in (('yaml', 'config.yaml'), ('manifest', 'eval500.json')):
        path = root / name_on_disk; atomic(path, reader.read(row[field])); row[field] = str(path)
    row.update(arm=name, episodes=50 * len(tasks))
    selected = [[t, i] for t in sorted(tasks) for i in range(50)]
    write_json(root / 'selection.json', dict(pairs=selected, original_arm=original, init_pool='A'))
    write_json(root / 'arms.json', [row])
    write_json(root / 'descent_table.json', dict(levels=table, caps=caps, k=parts(cell)[3],
               freeze=source['freeze'], uniform=None, off_tasks=[t for t in range(10) if table[str(t)] == 'off'],
               original_arm=original, post_test_selected=True, passive=passive))
    definitions = {str(p.relative_to(root)): c.sha(p) for p in sorted(root.iterdir()) if p.is_file()}
    write_json(root / 'ROOT.json', dict(definitions_sha256=definitions, fit_sha256={str(fit): c.sha(fit)},
               recording=False, fleet_binding=False, init_pool='A', episodes=50 * len(tasks)))
    return dict(row=row, source_root=str(root), source_seal_sha256=c.sha(root / 'ROOT.json'),
                fit_sha256=c.sha(fit), yaml_sha256=c.sha(row['yaml']), pairs=selected,
                dispatch_pairs=selected, original_arm=original, passive=passive,
                original_fit_sha256=source['fit_sha256'], freeze=source['freeze'])


def emit(changes_path, *, generation='v1', tables=HERE / 'dev_results_conductor',
         index=HERE / 'conductor_test_groups.json', output=None, trace=TRACE):
    a.require(bool(re.fullmatch(r'[A-Za-z0-9_]+', generation)), 'Invalid generation')
    output = Path(output or HERE / f'retest_{generation}').resolve(); trace = Path(trace).resolve()
    reader = a.Reader(); design = verify(reader)
    levels, _, freeze = frozen_tables(Path(tables))
    # Record all freeze bytes explicitly, without opening A outcomes.
    reader.read(Path(tables) / 'TABLE_FREEZE.json')
    for name in freeze['sha256']: reader.read(Path(tables) / name)
    changes_raw = reader.read(changes_path); changes = json.loads(changes_raw)
    adjusted = adjusted_tables(levels, changes)
    keys = sorted({(parts(d['cell'])[0], parts(d['cell'])[3]) for d in changes})
    targets = [trace / f'r16_conductor_retest_{m}_k{k}_{generation}' for m, k in keys]
    a.require(not output.exists() and not any(p.exists() for p in targets), 'Fresh retest generation/output required')
    original_index = reader.json(index); sources = {}
    # Index names identify groups; never read unrelated (possibly still running) roots.
    for m, k in keys:
        matches = [Path(p) for p in original_index['groups']
                   if re.fullmatch(rf'r16_conductor_test_{m}_k{k}_[A-Za-z0-9_]+', Path(p).parent.name)]
        a.require(len(matches) == 1, 'Missing/ambiguous original test group')
        g = c.read_group(matches[0], reader)
        a.require(g['phase'] == 'test' and (g['model'], g['k']) == (m, k), 'Wrong original test group')
        sources[m, k] = (matches[0], g)
    # Validate every requested source before creating any new root.
    for change in changes:
        cell = change['cell']; key = (parts(cell)[0], parts(cell)[3])
        entries = [e for e in sources[key][1]['entries'] if e['row']['r16_cell'] == cell]
        a.require(len(entries) == 1, 'Original cell must have exactly one frozen task arm')
        check_original(entries[0], levels, design, freeze, reader)
        raw = reader.read(fit_path(entries[0]['row']))
        caps = [design['cells'][cell]['periodic_caps'][adjusted[cell][str(t)]] for t in range(10)]
        adjusted_blob(raw, entries[0]['row'], adjusted[cell], caps)
    reader.stable(); groups = []
    selected_at = datetime.now(timezone.utc).isoformat()
    selection_record = dict(changes=changes, freeze=freeze, generation=generation, selected_at_utc=selected_at)
    for (m, k), root in zip(keys, targets, strict=True):
        original_path, original = sources[m, k]
        group = copy.deepcopy(original); entries = []
        cells = sorted({d['cell'] for d in changes if (parts(d['cell'])[0], parts(d['cell'])[3]) == (m, k)})
        for cell in cells:
            source = next(e for e in original['entries'] if e['row']['r16_cell'] == cell)
            tasks = [d['task'] for d in changes if d['cell'] == cell]
            entries.append(build_entry(source, root, f'r16rt_{cell}_{generation}', adjusted[cell], tasks, design, reader))
        # The unchanged launcher requires both checkpoint suites, even for a one-suite subset.
        # One dormant original fit fills a missing suite; it dispatches ZERO episodes.
        for suite in ('libero_10', 'libero_spatial'):
            if not any(e['row']['suite'] == suite for e in entries):
                source = next(e for e in original['entries'] if e['row']['suite'] == suite)
                check_original(source, levels, design, freeze, reader)
                entries.append(build_entry(source, root, source['row']['arm'], levels[source['row']['r16_cell']], [],
                                           design, reader, passive=True))
        group.update(root=str(root), entries=entries, references=None, retest=True,
                     original_group=str(original_path), original_group_sha256=original['group_sha256'],
                     retest_plan=str(output / 'plan.json'), retest_lock_sha256=c.sha(LOCK),
                     retest_selection=selection_record,
                     yaml_id_is_original_arm=False, original_episode_uids=True)
        group['definitions_sha256'] = {str(p): c.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
        group.pop('group_sha256', None); group['group_sha256'] = a.sha(c.canonical(group))
        write_json(root / 'group.json', group)
        write_json(root / 'ROOT.json', dict(schema=group['schema'], group_sha256=group['group_sha256'],
                   definitions_sha256={**group['definitions_sha256'], str(root / 'group.json'): c.sha(root / 'group.json')},
                   recording=False, fleet_binding=False))
        groups.append(str(root / 'group.json'))
    reader.stable()
    atomic(output / 'changes.json', changes_raw)
    write_json(output / 'levels.json', adjusted)
    plan = dict(schema='r16.retest.plan.v1', generation=generation, selection=SELECTION,
                selected_at_utc=selected_at, groups=groups, original_index=str(Path(index).resolve()),
                group_sha256={p: json.loads(Path(p).read_bytes())['group_sha256'] for p in groups},
                tables=str(Path(tables).resolve()), freeze=freeze, changes=changes, changes_sha256=a.sha(changes_raw),
                levels_sha256=c.sha(output / 'levels.json'), retest_lock_sha256=c.sha(LOCK),
                definitions_only=True, test_a_outcome_reads=0, original_episode_uids=True,
                episodes=50 * len(changes), inputs=reader.inputs)
    write_json(output / 'plan.json', plan)
    write_json(output / 'PLAN_FREEZE.json', dict(schema='r16.retest.selection_freeze.v1', post_test_selected=True,
               sha256={name: c.sha(output / name) for name in ('plan.json', 'changes.json', 'levels.json')}))
    atomic(output / 'queue_retest.txt', ''.join(f'@local {p}\n' for p in groups).encode())
    print(f'Emitted {len(groups)} new groups / {plan["episodes"]} episodes; queue: {output / "queue_retest.txt"}')
    return plan


def read_plan(path, reader):
    plan = reader.json(path)
    root = Path(path).parent
    seal = reader.json(root / 'PLAN_FREEZE.json')
    a.require(seal['schema'] == 'r16.retest.selection_freeze.v1' and seal['post_test_selected'], 'Wrong retest selection seal')
    for name, digest in seal['sha256'].items():
        a.require(a.sha(reader.read(root / name)) == digest, 'Changed emitted retest selection')
    a.require(plan['schema'] == 'r16.retest.plan.v1' and plan['retest_lock_sha256'] == a.sha(reader.read(LOCK)),
              'Changed retest plan/lock')
    changes_raw = reader.read(root / 'changes.json')
    a.require(a.sha(changes_raw) == plan['changes_sha256'] and json.loads(changes_raw) == plan['changes'], 'Changed owner changes')
    levels, _, freeze = frozen_tables(Path(plan['tables']))
    a.require(freeze == plan['freeze'] and reader.json(root / 'levels.json') == adjusted_tables(levels, plan['changes']) and
              a.sha(reader.read(root / 'levels.json')) == plan['levels_sha256'], 'Changed adjusted/frozen tables')
    for p, info in plan['inputs'].items():
        a.require(a.sha(reader.read(p)) == info['sha256'], f'Changed retest source: {p}')
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    sub.add_parser('verify')
    e = sub.add_parser('emit'); e.add_argument('--changes', type=Path, required=True)
    e.add_argument('--generation', default='v1'); e.add_argument('--tables', type=Path, default=HERE / 'dev_results_conductor')
    e.add_argument('--original-index', type=Path, default=HERE / 'conductor_test_groups.json')
    e.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.action == 'emit':
        emit(args.changes, generation=args.generation, tables=args.tables, index=args.original_index, output=args.output)
    else:
        reader = a.Reader(); verify(reader); reader.stable(); print('RETEST_LOCK_OK; no outcome reads')


if __name__ == '__main__': main()
