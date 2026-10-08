"""Replace selected whole A task cohorts, reusing the frozen R16 estimators.

Only the explicit analyze command opens outcomes, after original groups finish.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from . import analyze as a
from . import conductor_analysis as ca
from . import retest as rt
from .retest_runner import checked_group
from .prepare import HERE, CELLS, parts, frozen_tables, atomic, write_json


class IdentityView:
    """Validate real old UIDs, then rename their prefix for the existing loader.

    This is an in-memory join only: no input bytes, outcomes, decisions or counts
    are manufactured. Reader hashes the actual bytes before this projection.
    """
    def __init__(self, reader, entry):
        self.reader = reader; self.entry = entry
        self.shards = reader.shards

    def __getattr__(self, name): return getattr(self.reader, name)

    def jsonl(self, path):
        entry = self.entry; arm = entry['row']['arm']; original = entry['original_arm']
        selected = set(map(tuple, entry['dispatch_pairs']))
        for raw in self.reader.jsonl(path):
            row = dict(raw)
            # Journals contain all arms; a decision shard belongs to one arm.
            for field, belongs in (('task_uid', row.get('yaml_id') == arm), ('uid', row.get('ev') == 'dec')):
                if field not in row or not belongs: continue
                uid = row[field]; pair = a.uid_pair(uid)
                a.require(uid == ca.make_task_uid(original, 'eval', *pair) and pair in selected,
                          'Wrong retest original UID or unscheduled task/init')
                row[field] = ca.make_task_uid(arm, 'eval', *pair)
            if row.get('ev') == 'dec':
                t = row['task_id']; level = entry['row']['kwargs']['task_levels'][str(t)]
                # The unchanged blind/tail emitter records fit identity but has
                # no fresh query diagnostic. Real vision queries carry the level.
                if row['vision'] or 'descent_level' in row['r14']:
                    a.require(row['r14'].get('descent_level') == level, 'Wrong actual retest descent level')
                if level == 'c10': a.require(not row['vision'] or not row['hit'], 'Pure retest C10 vision HIT')
            yield row


def load_retest(group, entry, reader, selected_at):
    tasks = sorted({t for t, i in entry['dispatch_pairs']})
    a.require(tasks and not entry['passive'], 'No retest tasks')
    # Embed the measured subset in the original 10×50 identity template; all
    # unmeasured rows remain explicitly unknown, then are replaced from original A.
    view_entry = copy.deepcopy(entry); view_entry['pairs'] = rt.PAIRS
    result = ca.load_entry(group, view_entry, IdentityView(reader, entry), freeze={'frozen_at_utc': selected_at})
    a.require(result.audit['completed_tasks'] == tasks and all(np.all(result.y[t] == -1) for t in range(10) if t not in tasks),
              'Retest requires exactly the selected 50-episode cohorts')
    result.audit.update(original_arm=entry['original_arm'], original_episode_uids=True,
                        identity_view='Validated original UID; new prefix used only to join the unchanged loader',
                        post_test_selected=True)
    return result


def merge_cohorts(original, retest, tasks):
    a.require(original.y.shape == retest.y.shape == (10, 50) and np.all(original.y >= 0), 'Incomplete original A')
    tasks = sorted(tasks)
    a.require(tasks and retest.audit['completed_tasks'] == tasks, 'Incomplete/extra retest tasks')
    fields = {}
    for field in ('y', 'n', 'v', 'm', 'cost'):
        arr = getattr(original, field).copy(); arr[tasks] = getattr(retest, field)[tasks]
        fields[field] = arr
    a.require(np.all(fields['y'] >= 0), 'Unmeasured adjusted outcome')
    return a.Episodes(**fields, audit=dict(original=original.audit, retest=retest.audit, replaced_tasks=tasks,
                                          unchanged_tasks=[t for t in range(10) if t not in tasks]))


def calculate(before, after, changes, *, draws=5000):
    # Same shared suite/task/init resampling, families, intervals and denominators.
    old = a.calculate_test(before, draws=draws); new = a.calculate_test(after, draws=draws)
    vs_old = a.calculate_test({cell: dict(task=arms['task'], full_c10=before[cell]['task'], full_descent=before[cell]['task'])
                              for cell, arms in after.items()}, draws=draws)
    rows = []
    for change in changes:
        cell = change['cell']; t = change['task']; o = before[cell]['task']; n = after[cell]['task']
        def task_stats(ep):
            return a.Episodes(**{f: getattr(ep, f)[t:t+1] for f in ('y', 'n', 'v', 'm', 'cost')}, audit={}).stats()
        rows.append(dict(**change, old=task_stats(o), new=task_stats(n)))
    return dict(schema='r16.retest.result.v1', selection=rt.SELECTION, original_frozen_result_unchanged=True,
                measurement_only=False, post_test_selected=True, before=old, after=new, changed_tasks=rows,
                after_vs_before=dict(
                    cells={c: d['comparisons']['task_vs_full_c10'] for c, d in vs_old['cells'].items()},
                    pooled={c: d['comparisons']['full_c10'] for c, d in vs_old['pooled'].items()},
                    pooled_by_k={c: d['comparisons']['full_c10'] for c, d in vs_old['pooled_by_k'].items()}))


def apply(plan_path, output):
    output = Path(output)
    a.require(not output.exists(), 'Use a fresh analysis output; never overwrite results')
    reader = a.Reader(); design = rt.verify(reader); plan = rt.read_plan(plan_path, reader)
    levels, _, freeze = frozen_tables(Path(plan['tables']))
    original_groups = ca.group_index(plan['original_index'], reader)  # All four must be DONE.
    before = {}; originals = {}
    for group in original_groups:
        a.require(group['phase'] == 'test', 'Wrong original phase')
        originals[group['model'], group['k']] = group['group_sha256']
        for entry in group['entries']:
            row = entry['row']; cell = row['r16_cell']; m, s, size, k = parts(cell)
            a.require(cell not in before, 'Duplicate original cell')
            rt.check_original(entry, levels, design, freeze, reader)
            ep = ca.load_entry(group, entry, reader, freeze=freeze)
            a.require(np.all(ep.y >= 0), 'Original cell needs all 500 outcomes')
            arms = dict(task=ep)
            for key, source in (('full_c10', 'full_test_c10'), ('full_descent', 'full_test_descent')):
                ref = design['cells'][cell][source]
                arms[key] = a.load(ref, ref['arm'], rt.PAIRS, 'A', 10 if m == 'pi05' else 8, reader)
            before[cell] = arms
    a.require(set(before) == set(CELLS), 'Need all 32 original cells for pooled analysis')
    after = {c: dict(arms) for c, arms in before.items()}; replaced = set()
    adjusted = reader.json(Path(plan_path).parent / 'levels.json')
    for path in plan['groups']:
        group = checked_group(Path(path), reader)
        a.require(group['original_group_sha256'] == originals[group['model'], group['k']], 'Wrong original pairing group')
        done = reader.json(Path(group['root']) / 'DONE.json')
        a.require(done['group_sha256'] == group['group_sha256'], 'Retest group must be complete')
        names = {e['row']['arm'] for e in group['entries'] if not e['passive']}
        # A dormant suite must not have accepted work, and no unknown arm may enter the merge.
        for row in reader.jsonl(Path(group['root']) / 'journal.jsonl'):
            if row.get('accepted') and not row.get('error') and row['status'] in ('done', 'failed'):
                a.require(row['yaml_id'] in names, 'Accepted unscheduled retest arm')
        for entry in group['entries']:
            if entry['passive']: continue
            cell = entry['row']['r16_cell']; tasks = sorted({t for t, i in entry['dispatch_pairs']})
            a.require(entry['row']['kwargs']['task_levels'] == adjusted[cell], 'Wrong adjusted serving table')
            a.require(not any((cell, t) in replaced for t in tasks), 'Repeated retest task')
            ep = load_retest(group, entry, reader, plan['selected_at_utc'])
            after[cell]['task'] = merge_cohorts(before[cell]['task'], ep, tasks)
            replaced.update((cell, t) for t in tasks)
    a.require(replaced == {(d['cell'], d['task']) for d in plan['changes']}, 'Missing/extra replacements')
    result = calculate(before, after, plan['changes'])
    result['audits'] = {c: dict(before=before[c]['task'].audit, after=after[c]['task'].audit) for c in CELLS}
    # Preserve actual per-episode arrays/counts so the replacement is reviewable.
    result['merged_episodes'] = {c: {f: getattr(arms['task'], f).tolist() for f in ('y', 'n', 'v', 'm', 'cost')}
                                 for c, arms in after.items()}
    reader.stable(); frozen_tables(Path(plan['tables'])); result['inputs'] = reader.inputs
    result['plan'] = str(Path(plan_path).resolve())
    write_json(output / 'retest.json', result)
    lines = ['# R16 owner-selected retest', '', rt.SELECTION + '.', '',
             '| cell | before SR | after SR | before IR | after IR |', '|---|---:|---:|---:|---:|']
    for cell in CELLS:
        o = result['before']['cells'][cell]['arms']['task']; n = result['after']['cells'][cell]['arms']['task']
        lines.append(f'| {cell} | {o["sr"]:.4f} | {n["sr"]:.4f} | {o["ir"]:.6f} | {n["ir"]:.6f} |')
    lines += ['', '| size × k | before SR | after SR | before IR | after IR |', '|---|---:|---:|---:|---:|']
    for key, group in result['before']['pooled'].items():
        o = group['arms']['task']; n = result['after']['pooled'][key]['arms']['task']
        lines.append(f'| {key} | {o["sr"]:.4f} | {n["sr"]:.4f} | {o["ir"]:.6f} | {n["ir"]:.6f} |')
    lines += ['', '| changed task | old level / SR | new level / SR |', '|---|---|---|']
    for d in result['changed_tasks']:
        lines.append(f'| {d["cell"]} / {d["task"]} | {d["from"]} / {d["old"]["sr"]:.4f} | {d["to"]} / {d["new"]["sr"]:.4f} |')
    lines += ['', 'Paired reference comparisons, CIs, before/after comparisons and every merged V/M/N count: `retest.json`.', '',
              '## Before: original frozen test', '', a.markdown(result['before'], 'test A'), '',
              '## After: post-test owner selection', '',
              a.markdown(result['after'], 'adjusted A').replace('Measure only.', 'Post-test owner selection; exploratory.')]
    atomic(output / 'RETEST.md', ('\n'.join(lines) + '\n').encode())
    print(f'Wrote {output / "retest.json"} and {output / "RETEST.md"}')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); apply(args.plan, args.output)


if __name__ == '__main__': main()
