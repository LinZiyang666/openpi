"""Additive retest generations, with read-only preflight and combined analysis.

The sealed v1 emitter/runner/loaders/estimators are imported without editing
them. Only `analyze` reads outcomes; `validate` never emits or launches anything.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

import numpy as np

from . import analyze as a, conductor as c, conductor_analysis as ca
from . import retest as rt, retest_analysis as ra
from .retest_runner import checked_group

LOCK = rt.HERE / 'RETEST_COMBINED_LOCK.json'


def verify(reader):
    design = rt.verify(reader)
    lock = reader.json(LOCK)
    a.require(lock['schema'] == 'r16.retest.combined.lock.v1', 'Wrong combined retest schema')
    return design


def compose(plans, levels):
    """Union disjoint selections against ONE original freeze, never last-wins."""
    a.require(bool(plans), 'Need at least one retest plan')
    anchor = plans[0]; generations = set(); groups = set(); seen = {}; changes = []
    for plan in plans:
        generation = plan['generation']
        a.require(generation not in generations, 'Duplicate retest generation')
        generations.add(generation)
        a.require(plan['freeze'] == anchor['freeze'] and
                  plan['retest_lock_sha256'] == anchor['retest_lock_sha256'] and
                  all(Path(plan[key]).resolve() == Path(anchor[key]).resolve()
                      for key in ('tables', 'original_index')), 'Conflicting generation freeze/original pairing')
        rt.adjusted_tables(levels, plan['changes'])  # Includes exact frozen `from` validation.
        for path in plan['groups']:
            path = str(Path(path).resolve())
            a.require(path not in groups, 'Repeated group across retest generations'); groups.add(path)
        for change in plan['changes']:
            key = (change['cell'], change['task'])
            a.require(key not in seen, f'Overlapping retest task {key}: {seen.get(key)} / {generation}')
            seen[key] = generation; changes.append(copy.deepcopy(change))
    return rt.adjusted_tables(levels, changes), changes, seen


def read_selections(paths, reader):
    paths = [Path(p).resolve() for p in paths]
    a.require(len(paths) == len(set(paths)), 'Duplicate retest plan path')
    plans = [rt.read_plan(path, reader) for path in paths]
    a.require(bool(plans), 'Need at least one retest plan')
    levels, _, freeze = rt.frozen_tables(Path(plans[0]['tables']))
    adjusted, changes, owners = compose(plans, levels)
    return paths, plans, levels, freeze, adjusted, changes, owners


def validate(changes_path, against_paths, *, generation='v2', tables=None, index=None, output=None, trace=rt.TRACE):
    """Preflight existing emitter inputs WITHOUT creating any file/directory."""
    a.require(bool(re.fullmatch(r'[A-Za-z0-9_]+', generation)), 'Invalid generation')
    reader = a.Reader(); design = verify(reader)
    paths, previous, _, _, _, _, _ = read_selections(against_paths, reader)
    tables = Path(tables or previous[0]['tables']).resolve()
    index = Path(index or previous[0]['original_index']).resolve()
    levels, _, freeze = rt.frozen_tables(tables)
    changes_raw = reader.read(changes_path); changes = json.loads(changes_raw)
    adjusted = rt.adjusted_tables(levels, changes)
    keys = sorted({(rt.parts(d['cell'])[0], rt.parts(d['cell'])[3]) for d in changes})
    targets = [Path(trace).resolve() / f'r16_conductor_retest_{m}_k{k}_{generation}' for m, k in keys]
    output = Path(output or rt.HERE / f'retest_{generation}').resolve()
    candidate = dict(generation=generation, tables=str(tables), original_index=str(index), freeze=freeze,
                     retest_lock_sha256=a.sha(reader.read(rt.LOCK)), changes=changes,
                     groups=[str(root / 'group.json') for root in targets])
    _, combined, owners = compose([*previous, candidate], levels)
    a.require(not output.exists() and not any(root.exists() for root in targets), 'Fresh retest generation/output required')
    original_index = reader.json(index)
    for model, k in keys:
        matches = [Path(p) for p in original_index['groups'] if
                   re.fullmatch(rf'r16_conductor_test_{model}_k{k}_[A-Za-z0-9_]+', Path(p).parent.name)]
        a.require(len(matches) == 1, 'Missing/ambiguous original test group')
        group = c.read_group(matches[0], reader)
        a.require(group['phase'] == 'test' and (group['model'], group['k']) == (model, k), 'Wrong original test group')
        requested = {d['cell'] for d in changes if (rt.parts(d['cell'])[0], rt.parts(d['cell'])[3]) == (model, k)}
        for cell in requested:
            entries = [e for e in group['entries'] if e['row']['r16_cell'] == cell]
            a.require(len(entries) == 1, 'Original cell needs one frozen arm')
            entry = entries[0]; rt.check_original(entry, levels, design, freeze, reader)
            raw = reader.read(rt.fit_path(entry['row']))
            a.require(a.sha(raw) == entry['fit_sha256'], 'Changed original fitted bytes')
            caps = [design['cells'][cell]['periodic_caps'][adjusted[cell][str(t)]] for t in range(10)]
            rt.adjusted_blob(raw, entry['row'], adjusted[cell], caps)  # RAM only; set_table, never fit.
        # Match the unchanged emitter's dormant-suite preflight too.
        for suite in ('libero_10', 'libero_spatial'):
            if any(e['row']['r16_cell'] in requested and e['row']['suite'] == suite for e in group['entries']): continue
            entry = next(e for e in group['entries'] if e['row']['suite'] == suite)
            rt.check_original(entry, levels, design, freeze, reader)
            a.require(a.sha(reader.read(rt.fit_path(entry['row']))) == entry['fit_sha256'], 'Changed dormant suite fit')
    reader.stable()
    return dict(schema='r16.retest.additive_preflight.v1', generation=generation, changes=changes,
                changes_sha256=a.sha(changes_raw), additional_tasks=len(changes), additional_episodes=50 * len(changes),
                combined_tasks=len(combined), against_plans=list(map(str, paths)), planned_groups=candidate['groups'],
                planned_queue=str(output / 'queue_retest.txt'), definitions_and_fits_verified=True,
                original_episode_uids=True, files_written=0, roots_created=0, test_a_outcome_reads=0,
                inputs=reader.inputs)


def markdown(result):
    lines = ['# R16 combined additive retests', '', rt.SELECTION + '.', '',
             '| cell / pool | before SR | after SR | before IR | after IR |', '|---|---:|---:|---:|---:|']
    for section in ('cells', 'pooled', 'pooled_by_k'):
        for key, group in result['before'][section].items():
            old = group['arms']['task']; new = result['after'][section][key]['arms']['task']
            lines.append(f'| {key} | {old["sr"]:.4f} | {new["sr"]:.4f} | {old["ir"]:.6f} | {new["ir"]:.6f} |')
    lines += ['', '| changed task | generation | old level / SR | new level / SR |', '|---|---|---|---|']
    for d in result['changed_tasks']:
        lines.append(f'| {d["cell"]} / {d["task"]} | {d["generation"]} | '
                     f'{d["from"]} / {d["old"]["sr"]:.4f} | {d["to"]} / {d["new"]["sr"]:.4f} |')
    lines += ['', 'Combined table: `levels.json`. Counts, provenance, input hashes and all paired CIs: `retest.json`.', '',
              '## Before: original frozen test', '', a.markdown(result['before'], 'test A'), '',
              '## After: combined post-test selection', '',
              a.markdown(result['after'], 'combined adjusted A').replace('Measure only.', 'Post-test owner selection; exploratory.')]
    return '\n'.join(lines) + '\n'


def apply(plan_paths, output):
    output = Path(output)
    a.require(not output.exists(), 'Use a fresh combined output; never overwrite results')
    reader = a.Reader(); design = verify(reader)
    paths, plans, levels, freeze, adjusted, changes, owners = read_selections(plan_paths, reader)
    # Reject overlap/pairing conflict above, and incomplete groups here, BEFORE any A outcome read.
    retest_groups = []
    for path, plan in zip(paths, plans, strict=True):
        for group_path in plan['groups']:
            group = checked_group(Path(group_path), reader)
            a.require(Path(group['retest_plan']).resolve() == path, 'Wrong generation plan binding')
            done = reader.json(Path(group['root']) / 'DONE.json')
            a.require(done['group_sha256'] == group['group_sha256'], 'Retest group must be complete')
            retest_groups.append((plan, group))
    original_groups = ca.group_index(plans[0]['original_index'], reader)
    before = {}; originals = {}; original_arms = {}
    for group in original_groups:
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
    after = {cell: dict(arms) for cell, arms in before.items()}; replaced = set()
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
            a.require(all((cell, t) not in replaced and owners.get((cell, t)) == generation for t in tasks),
                      'Overlapping or misassigned measured retest task')
            ep = ra.load_retest(group, entry, reader, plan['selected_at_utc'])
            # Merge into the accumulated cell, never back into original and erase v1.
            after[cell]['task'] = ra.merge_cohorts(after[cell]['task'], ep, tasks)
            audits[cell].append(dict(generation=generation, group_sha256=group['group_sha256'], audit=ep.audit, tasks=tasks))
            for t in tasks:
                provenance[cell][str(t)] = dict(generation=generation, level=adjusted[cell][str(t)],
                                               arm=entry['row']['arm'], group_sha256=group['group_sha256'])
            replaced.update((cell, t) for t in tasks)
    a.require(replaced == set(owners), 'Missing/extra combined replacements')
    for cell in rt.CELLS:
        after[cell]['task'] = a.Episodes(**{f: getattr(after[cell]['task'], f) for f in ('y', 'n', 'v', 'm', 'cost')},
            audit=dict(original=before[cell]['task'].audit, retest_generations=audits[cell],
                       replaced_tasks=sorted(t for c, t in replaced if c == cell),
                       unchanged_tasks=[t for t in range(10) if (cell, t) not in replaced]))
    result = ra.calculate(before, after, changes)
    result.update(schema='r16.retest.combined.result.v1', levels=adjusted, task_provenance=provenance,
                  generations=[dict(generation=p['generation'], plan=str(path), plan_sha256=a.sha(reader.read(path)),
                                    changed_tasks=len(p['changes']), groups=p['group_sha256'])
                               for path, p in zip(paths, plans, strict=True)],
                  audits={cell: dict(before=before[cell]['task'].audit, after=after[cell]['task'].audit) for cell in rt.CELLS},
                  merged_episodes={cell: {f: getattr(arms['task'], f).tolist() for f in ('y', 'n', 'v', 'm', 'cost')}
                                   for cell, arms in after.items()})
    for change in result['changed_tasks']: change['generation'] = owners[change['cell'], change['task']]
    reader.stable(); rt.frozen_tables(Path(plans[0]['tables'])); result['inputs'] = reader.inputs
    rt.write_json(output / 'levels.json', adjusted)
    rt.write_json(output / 'retest.json', result)
    rt.atomic(output / 'RETEST.md', markdown(result).encode())
    print(f'Combined {len(plans)} generations / {len(changes)} tasks: {output}')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='action', required=True)
    v = sub.add_parser('validate', help='Read-only emission preflight; creates no roots or files')
    v.add_argument('--changes', type=Path, required=True); v.add_argument('--against-plan', action='append', type=Path, required=True)
    v.add_argument('--generation', default='v2'); v.add_argument('--tables', type=Path); v.add_argument('--original-index', type=Path)
    v.add_argument('--output', type=Path)
    e = sub.add_parser('analyze'); e.add_argument('--plan', action='append', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.action == 'validate':
        report = validate(args.changes, args.against_plan, generation=args.generation, tables=args.tables,
                          index=args.original_index, output=args.output)
        # Full per-input hashes are available to API callers; keep CLI output compact.
        print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=2))
    else: apply(args.plan, args.output)


if __name__ == '__main__': main()
