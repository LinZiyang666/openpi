"""Explicit seven-pair v3 correction, over the unchanged sealed retest runtime.

Validation and emission read definitions only. Analysis accepts only the owner
listed v3 -> v1 overlaps, and retains both measurements in the report.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np

from . import analyze as a, conductor as c, conductor_analysis as ca
from . import retest as rt, retest_analysis as ra, retest_combined as cm
from .retest_runner import checked_group

LOCK = rt.HERE / 'RETEST_SUPERSEDING_LOCK.json'


def verify(reader):
    design = cm.verify(reader)
    lock = reader.json(LOCK)
    a.require(lock['schema'] == 'r16.retest.superseding.lock.v1' and
              lock['combined_lock_sha256'] == a.sha(reader.read(cm.LOCK)), 'Changed superseding retest lock')
    for path, digest in {**lock['source_sha256'], **lock['owner_selection_sha256']}.items():
        a.require(a.sha(reader.read(path)) == digest, f'Changed superseding source/selection: {path}')
    return design


def normalize(changes, generation):
    a.require(generation == 'v3', 'This correction requires generation v3')
    a.require(isinstance(changes, list) and len(changes) == 7, 'Need exactly seven superseding changes')
    for d in changes:
        a.require(isinstance(d, dict) and set(d) == {'cell', 'task', 'from', 'to', 'supersedes'},
                  'Superseding changes require cell/task/from/to/supersedes')
        a.require(d['supersedes'] == 'v1' and d['to'] == 'c10', 'Only v3 c10 superseding v1 is authorized')
    return [{k: d[k] for k in ('cell', 'task', 'from', 'to')} for d in changes]


def check_authority(raw, reader):
    lock = reader.json(LOCK)
    a.require(a.sha(raw) in lock['owner_selection_sha256'].values(), 'Selection differs from the authorized seven-pair file')


def compose(plans, levels):
    """Resolve explicit pair-level supersession independently of argument order."""
    a.require(bool(plans), 'Need at least one retest plan')
    anchor = plans[0]; generations = set(); groups = set(); pairs = {}; declarations = {}
    for plan in plans:
        generation = plan['generation']
        a.require(generation not in generations, 'Duplicate retest generation'); generations.add(generation)
        a.require(plan['freeze'] == anchor['freeze'] and plan['retest_lock_sha256'] == anchor['retest_lock_sha256'] and
                  all(Path(plan[k]).resolve() == Path(anchor[k]).resolve() for k in ('tables', 'original_index')),
                  'Conflicting generation freeze/original pairing')
        rt.adjusted_tables(levels, plan['changes'])
        for path in plan['groups']:
            path = str(Path(path).resolve())
            a.require(path not in groups, 'Repeated group across retest generations'); groups.add(path)
        if 'superseding_changes' in plan:
            owner = plan['superseding_changes']
            a.require(normalize(owner, generation) == plan['changes'], 'Supersession differs from emitted changes')
            declarations.update({(d['cell'], d['task']): d for d in owner})
        for d in plan['changes']:
            pairs.setdefault((d['cell'], d['task']), {})[generation] = d
    final = []; owners = {}; superseded = {}
    for key, versions in sorted(pairs.items()):
        declaration = declarations.get(key)
        if declaration is None:
            a.require(len(versions) == 1, f'Overlapping retest task without explicit supersession: {key}')
            generation = next(iter(versions))
        else:
            a.require(set(versions) == {'v1', 'v3'}, f'Missing superseded v1 or other overlap: {key}')
            old, new = versions['v1'], versions['v3']
            a.require(old['to'] == 'ir45' and new['to'] == 'c10' and old['from'] == new['from'],
                      f'Superseded pair must be frozen-from -> v1 ir45 -> v3 c10: {key}')
            generation = 'v3'; superseded[key] = 'v1'
        owners[key] = generation; final.append(copy.deepcopy(versions[generation]))
    return rt.adjusted_tables(levels, final), final, owners, superseded


def read_selections(paths, reader):
    paths = sorted(Path(p).resolve() for p in paths)
    a.require(len(paths) == len(set(paths)), 'Duplicate retest plan path')
    plans = [rt.read_plan(path, reader) for path in paths]
    for path, plan in zip(paths, plans, strict=True):
        if 'superseding_changes' not in plan: continue
        raw = reader.read(path.parent / 'owner_changes.json')
        check_authority(raw, reader)
        a.require(json.loads(raw) == plan['superseding_changes'] and a.sha(raw) == plan['owner_changes_sha256'] and
                  plan['superseding_lock_sha256'] == a.sha(reader.read(LOCK)), 'Changed owner supersession seal')
        for previous, digest in plan['against_plan_sha256'].items():
            a.require(a.sha(reader.read(previous)) == digest, 'Changed superseded/previous plan')
            a.require(Path(previous).resolve() in paths, 'Include every against-plan in combined analysis')
    a.require(bool(plans), 'Need at least one retest plan')
    levels, _, freeze = rt.frozen_tables(Path(plans[0]['tables']))
    adjusted, changes, owners, superseded = compose(plans, levels)
    return paths, plans, levels, freeze, adjusted, changes, owners, superseded


def preflight(changes_path, against_paths, *, generation='v3', tables=None, index=None, output=None, trace=rt.TRACE):
    reader = a.Reader(); design = verify(reader)
    paths, previous, _, _, _, _, _, _ = read_selections(against_paths, reader)
    tables = Path(tables or previous[0]['tables']).resolve(); index = Path(index or previous[0]['original_index']).resolve()
    output = Path(output or rt.HERE / f'retest_{generation}').resolve(); trace = Path(trace).resolve()
    levels, _, freeze = rt.frozen_tables(tables)
    reader.read(tables / 'TABLE_FREEZE.json')
    for name in freeze['sha256']: reader.read(tables / name)
    raw = reader.read(changes_path); check_authority(raw, reader)
    owner = json.loads(raw); changes = normalize(owner, generation)
    adjusted = rt.adjusted_tables(levels, changes)
    keys = sorted({(rt.parts(d['cell'])[0], rt.parts(d['cell'])[3]) for d in changes})
    targets = [trace / f'r16_conductor_retest_{m}_k{k}_{generation}' for m, k in keys]
    candidate = dict(generation=generation, tables=str(tables), original_index=str(index), freeze=freeze,
                     retest_lock_sha256=a.sha(reader.read(rt.LOCK)), changes=changes, superseding_changes=owner,
                     groups=[str(root / 'group.json') for root in targets])
    _, combined, _, superseded = compose([*previous, candidate], levels)
    a.require(not output.exists() and not any(root.exists() for root in targets), 'Fresh retest generation/output required')
    original_index = reader.json(index); sources = {}
    for m, k in keys:
        matches = [Path(p) for p in original_index['groups'] if
                   re.fullmatch(rf'r16_conductor_test_{m}_k{k}_[A-Za-z0-9_]+', Path(p).parent.name)]
        a.require(len(matches) == 1, 'Missing/ambiguous original test group')
        group = c.read_group(matches[0], reader)
        a.require(group['phase'] == 'test' and (group['model'], group['k']) == (m, k), 'Wrong original test group')
        sources[m, k] = matches[0], group
        requested = {d['cell'] for d in changes if (rt.parts(d['cell'])[0], rt.parts(d['cell'])[3]) == (m, k)}
        entries = []
        for cell in requested:
            matches_cell = [e for e in group['entries'] if e['row']['r16_cell'] == cell]
            a.require(len(matches_cell) == 1, 'Original cell needs one frozen arm')
            entries.append((matches_cell[0], False))
        for suite in ('libero_10', 'libero_spatial'):
            if not any(e['row']['suite'] == suite for e, _ in entries):
                entries.append((next(e for e in group['entries'] if e['row']['suite'] == suite), True))
        for entry, passive in entries:
            cell = entry['row']['r16_cell']; rt.check_original(entry, levels, design, freeze, reader)
            fitted = reader.read(rt.fit_path(entry['row']))
            a.require(a.sha(fitted) == entry['fit_sha256'], 'Changed original fitted bytes')
            if not passive:
                caps = [design['cells'][cell]['periodic_caps'][adjusted[cell][str(t)]] for t in range(10)]
                rt.adjusted_blob(fitted, entry['row'], adjusted[cell], caps)
    reader.stable()
    report = dict(schema='r16.retest.superseding_preflight.v1', generation=generation, changes=owner,
                  additional_tasks=len(changes), additional_episodes=50 * len(changes), combined_tasks=len(combined),
                  superseded_tasks=len(superseded), planned_groups=candidate['groups'], planned_queue=str(output / 'queue_retest.txt'),
                  against_plans=list(map(str, paths)), definitions_and_fits_verified=True, original_episode_uids=True,
                  files_written=0, roots_created=0, test_a_outcome_reads=0, inputs=reader.inputs)
    return report, reader, design, levels, freeze, adjusted, changes, raw, sources, targets, output, tables, index, paths


def validate(changes_path, against_paths, **kwargs):
    return preflight(changes_path, against_paths, **kwargs)[0]


def emit(changes_path, against_paths, *, generation='v3', **kwargs):
    (report, reader, design, levels, freeze, adjusted, changes, owner_raw, sources,
     targets, output, tables, index, paths) = preflight(changes_path, against_paths, generation=generation, **kwargs)
    selected_at = datetime.now(timezone.utc).isoformat(); groups = []
    selection = dict(changes=changes, freeze=freeze, generation=generation, selected_at_utc=selected_at)
    for root in targets:
        key = next(key for key in sources if root.name == f'r16_conductor_retest_{key[0]}_k{key[1]}_{generation}')
        original_path, original = sources[key]; group = copy.deepcopy(original); entries = []
        cells = sorted({d['cell'] for d in changes if (rt.parts(d['cell'])[0], rt.parts(d['cell'])[3]) == key})
        for cell in cells:
            source = next(e for e in original['entries'] if e['row']['r16_cell'] == cell)
            tasks = [d['task'] for d in changes if d['cell'] == cell]
            entries.append(rt.build_entry(source, root, f'r16rt_{cell}_{generation}', adjusted[cell], tasks, design, reader))
        for suite in ('libero_10', 'libero_spatial'):
            if not any(e['row']['suite'] == suite for e in entries):
                source = next(e for e in original['entries'] if e['row']['suite'] == suite)
                entries.append(rt.build_entry(source, root, source['row']['arm'], levels[source['row']['r16_cell']], [],
                                              design, reader, passive=True))
        group.update(root=str(root), entries=entries, references=None, retest=True,
                     original_group=str(original_path), original_group_sha256=original['group_sha256'],
                     retest_plan=str(output / 'plan.json'), retest_lock_sha256=c.sha(rt.LOCK), retest_selection=selection,
                     yaml_id_is_original_arm=False, original_episode_uids=True)
        group['definitions_sha256'] = {str(p): c.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
        group.pop('group_sha256', None); group['group_sha256'] = a.sha(c.canonical(group))
        rt.write_json(root / 'group.json', group)
        rt.write_json(root / 'ROOT.json', dict(schema=group['schema'], group_sha256=group['group_sha256'],
                      definitions_sha256={**group['definitions_sha256'], str(root / 'group.json'): c.sha(root / 'group.json')},
                      recording=False, fleet_binding=False))
        groups.append(str(root / 'group.json'))
    reader.stable()
    # The legacy runner consumes its original four-field selection. The complete
    # owner file and explicit pair-level supersession are sealed alongside it.
    rt.write_json(output / 'changes.json', changes); rt.atomic(output / 'owner_changes.json', owner_raw)
    rt.write_json(output / 'levels.json', adjusted)
    plan = dict(schema='r16.retest.plan.v1', generation=generation, selection=rt.SELECTION,
                selected_at_utc=selected_at, groups=groups, original_index=str(index),
                group_sha256={p: json.loads(Path(p).read_bytes())['group_sha256'] for p in groups},
                tables=str(tables), freeze=freeze, changes=changes, changes_sha256=c.sha(output / 'changes.json'),
                levels_sha256=c.sha(output / 'levels.json'), retest_lock_sha256=c.sha(rt.LOCK),
                superseding_changes=json.loads(owner_raw), owner_changes_sha256=a.sha(owner_raw),
                superseding_lock_sha256=c.sha(LOCK), against_plan_sha256={str(p): a.sha(reader.read(p)) for p in paths},
                definitions_only=True, test_a_outcome_reads=0, original_episode_uids=True,
                episodes=50 * len(changes), inputs=reader.inputs)
    rt.write_json(output / 'plan.json', plan)
    rt.write_json(output / 'PLAN_FREEZE.json', dict(schema='r16.retest.selection_freeze.v1', post_test_selected=True,
                  sha256={name: c.sha(output / name) for name in ('plan.json', 'changes.json', 'owner_changes.json', 'levels.json')}))
    rt.atomic(output / 'queue_retest.txt', ''.join(f'@local {p}\n' for p in groups).encode())
    print(f'Emitted {len(groups)} v3 groups / {plan["episodes"]} episodes; queue: {output / "queue_retest.txt"}')
    return plan


def task_reference(ep, task, generation, level, entry, group):
    fields = {f: getattr(ep, f)[task:task+1] for f in ('y', 'n', 'v', 'm', 'cost')}
    stats = a.Episodes(**fields, audit={}).stats()
    return dict(generation=generation, level=level, arm=entry['row']['arm'], group_sha256=group['group_sha256'],
                result=stats, episodes={f: value[0].tolist() for f, value in fields.items()})


def markdown(result):
    text = cm.markdown(result).replace('# R16 combined additive retests', '# R16 combined retests with explicit v3 supersession')
    text += '\n## Superseded measurements retained for reference\n\n'
    text += '| changed task | final generation / level / SR | superseded generation / level / SR |\n|---|---|---|\n'
    for d in result['changed_tasks']:
        for old in d['superseded_results']:
            text += (f'| {d["cell"]} / {d["task"]} | {d["generation"]} / {d["to"]} / {d["new"]["sr"]:.4f} | '
                     f'{old["generation"]} / {old["level"]} / {old["result"]["sr"]:.4f} |\n')
    return text


def apply(plan_paths, output):
    output = Path(output)
    a.require(not output.exists(), 'Use a fresh combined output; never overwrite results')
    reader = a.Reader(); design = verify(reader)
    paths, plans, levels, freeze, adjusted, changes, owners, superseded = read_selections(plan_paths, reader)
    retest_groups = []
    # Definitions, supersession and DONE receipts must validate before outcomes.
    for path, plan in zip(paths, plans, strict=True):
        for group_path in plan['groups']:
            group = checked_group(Path(group_path), reader)
            a.require(Path(group['retest_plan']).resolve() == path, 'Wrong generation plan binding')
            done = reader.json(Path(group['root']) / 'DONE.json')
            a.require(done['group_sha256'] == group['group_sha256'], 'Retest group must be complete')
            retest_groups.append((plan, group))
    before = {}; originals = {}; original_arms = {}
    for group in ca.group_index(plans[0]['original_index'], reader):
        a.require(group['phase'] == 'test', 'Wrong original phase')
        originals[group['model'], group['k']] = group['group_sha256']
        for entry in group['entries']:
            row = entry['row']; cell = row['r16_cell']; model = rt.parts(cell)[0]
            a.require(cell not in before, 'Duplicate original cell')
            rt.check_original(entry, levels, design, freeze, reader)
            ep = ca.load_entry(group, entry, reader, freeze=freeze)
            a.require(np.all(ep.y >= 0), 'Original cell needs all 500 outcomes')
            arms = dict(task=ep); original_arms[cell] = row['arm']
            for key, source in (('full_c10', 'full_test_c10'), ('full_descent', 'full_test_descent')):
                ref = design['cells'][cell][source]
                arms[key] = a.load(ref, ref['arm'], rt.PAIRS, 'A', 10 if model == 'pi05' else 8, reader)
            before[cell] = arms
    a.require(set(before) == set(rt.CELLS), 'Need all 32 original cells for pooling')
    after = {cell: dict(arms) for cell, arms in before.items()}; replaced = set(); measured = set(); references = {}
    audits = {cell: [] for cell in rt.CELLS}
    provenance = {cell: {str(t): dict(generation='original', level=levels[cell][str(t)], arm=original_arms[cell])
                         for t in range(10)} for cell in rt.CELLS}
    for plan, group in retest_groups:
        generation = plan['generation']
        a.require(group['original_group_sha256'] == originals[group['model'], group['k']], 'Wrong original pairing group')
        names = {e['row']['arm'] for e in group['entries'] if not e['passive']}
        for row in reader.jsonl(Path(group['root']) / 'journal.jsonl'):
            if row.get('accepted') and not row.get('error') and row['status'] in ('done', 'failed'):
                a.require(row['yaml_id'] in names, 'Accepted unscheduled retest arm')
        for entry in group['entries']:
            if entry['passive']: continue
            cell = entry['row']['r16_cell']; tasks = sorted({t for t, i in entry['dispatch_pairs']})
            ep = ra.load_retest(group, entry, reader, plan['selected_at_utc'])
            active = []
            for t in tasks:
                key = cell, t
                a.require((generation, cell, t) not in measured, 'Repeated measured retest task')
                measured.add((generation, cell, t))
                reference = task_reference(ep, t, generation, entry['row']['kwargs']['task_levels'][str(t)], entry, group)
                if owners.get(key) == generation:
                    a.require(key not in replaced, 'Repeated final retest task')
                    active.append(t); replaced.add(key); provenance[cell][str(t)] = {
                        k: reference[k] for k in ('generation', 'level', 'arm', 'group_sha256')}
                else:
                    a.require(superseded.get(key) == generation, 'Unauthorized superseded measurement')
                    references[key] = reference
            if active:
                # Split mixed v1 cells: superseded tasks must never enter the final arrays.
                view = a.Episodes(**{f: getattr(ep, f) for f in ('y', 'n', 'v', 'm', 'cost')},
                                  audit={**ep.audit, 'completed_tasks': active})
                after[cell]['task'] = ra.merge_cohorts(after[cell]['task'], view, active)
            audits[cell].append(dict(generation=generation, group_sha256=group['group_sha256'], audit=ep.audit,
                                     tasks=tasks, final_tasks=active, superseded_tasks=[t for t in tasks if t not in active]))
    expected = {(p['generation'], d['cell'], d['task']) for p in plans for d in p['changes']}
    a.require(measured == expected and replaced == set(owners) and set(references) == set(superseded),
              'Missing/extra final or superseded cohorts')
    for cell in rt.CELLS:
        for t in range(10):
            if (cell, t) in references: provenance[cell][str(t)]['superseded_results'] = [references[cell, t]]
        after[cell]['task'] = a.Episodes(**{f: getattr(after[cell]['task'], f) for f in ('y', 'n', 'v', 'm', 'cost')},
            audit=dict(original=before[cell]['task'].audit, retest_generations=audits[cell],
                       replaced_tasks=sorted(t for c_, t in replaced if c_ == cell),
                       unchanged_tasks=[t for t in range(10) if (cell, t) not in replaced]))
    result = ra.calculate(before, after, changes)
    result.update(schema='r16.retest.superseding.result.v1', levels=adjusted, task_provenance=provenance,
                  superseded_tasks=len(superseded),
                  generations=[dict(generation=p['generation'], plan=str(path), plan_sha256=a.sha(reader.read(path)),
                                    changed_tasks=len(p['changes']), groups=p['group_sha256'])
                               for path, p in zip(paths, plans, strict=True)],
                  audits={cell: dict(before=before[cell]['task'].audit, after=after[cell]['task'].audit) for cell in rt.CELLS},
                  merged_episodes={cell: {f: getattr(arms['task'], f).tolist() for f in ('y', 'n', 'v', 'm', 'cost')}
                                   for cell, arms in after.items()})
    for d in result['changed_tasks']:
        key = d['cell'], d['task']; d['generation'] = owners[key]
        d['superseded_results'] = [references[key]] if key in references else []
    reader.stable(); rt.frozen_tables(Path(plans[0]['tables'])); result['inputs'] = reader.inputs
    rt.write_json(output / 'levels.json', adjusted); rt.write_json(output / 'retest.json', result)
    rt.atomic(output / 'RETEST.md', markdown(result).encode())
    print(f'Combined {len(plans)} generations / {len(changes)} final tasks / {len(superseded)} superseded: {output}')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='action', required=True)
    for action in ('validate', 'emit'):
        parser = sub.add_parser(action)
        parser.add_argument('--changes', type=Path, required=True)
        parser.add_argument('--against-plan', action='append', type=Path, required=True)
        parser.add_argument('--generation', default='v3'); parser.add_argument('--tables', type=Path)
        parser.add_argument('--original-index', type=Path); parser.add_argument('--output', type=Path)
    parser = sub.add_parser('analyze'); parser.add_argument('--plan', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.action == 'analyze': apply(args.plan, args.output)
    else:
        function = validate if args.action == 'validate' else emit
        report = function(args.changes, args.against_plan, generation=args.generation, tables=args.tables,
                          index=args.original_index, output=args.output)
        if args.action == 'validate': print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=2))


if __name__ == '__main__': main()

