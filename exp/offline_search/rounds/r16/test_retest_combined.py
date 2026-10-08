"""Synthetic outcomes only; additive generation scratch stays outside the repo."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from openpi.conductor.task import ServerEndpoint, make_task_uid
from exp.offline_search.closed_loop.plugin import ep_seed
from . import retest_combined as cm, retest as rt, retest_analysis as ra
from . import conductor as c, conductor_analysis as ca, analyze as a
from .retest_runner import checked_group, RetestStrategy
from .test_retest import sources, levels_fixture
from .test_r16 import episode, jl


CELL = 'pi05_l10_100_k1'


def change(cell=CELL, task=4, target='ir45'):
    return dict(cell=cell, task=task, **{'from': 'ir32', 'to': target})


def simple_plan(generation, changes):
    return dict(generation=generation, changes=changes, freeze={'same': True},
                tables='/tmp/combined_tables', original_index='/tmp/combined_original.json',
                retest_lock_sha256='same', groups=[f'/tmp/group_{generation}/group.json'])


def fixture(tmp_path, monkeypatch, first=None, second=None):
    levels, design, freeze, tables, index = sources(tmp_path, monkeypatch)
    monkeypatch.setattr(cm, 'verify', lambda reader: design)
    first = first or [change(task=4, target='c10')]
    second = second or [change(task=7)]
    plans = []
    for generation, changes in (('v1', first), ('v2', second)):
        changes_path = tmp_path / f'changes_{generation}.json'; c.write_json(changes_path, changes)
        plan = rt.emit(changes_path, generation=generation, tables=tables, index=index,
                       output=tmp_path / generation, trace=tmp_path / 'new')
        plans.append(plan)
    return levels, design, freeze, tables, index, plans


def snapshot(root):
    return {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_compose_disjoint_tasks_in_same_cell_keeps_both_tables_and_inputs():
    levels = levels_fixture(); plans = [simple_plan('v1', [change(task=4, target='c10')]), simple_plan('v2', [change(task=7)])]
    original = copy.deepcopy((levels, plans)); table, changes, owners = cm.compose(plans, levels)
    assert table[CELL]['4'] == 'c10' and table[CELL]['7'] == 'ir45'
    assert all(table[CELL][str(t)] == 'ir32' for t in range(10) if t not in (4, 7))
    assert owners == {(CELL, 4): 'v1', (CELL, 7): 'v2'} and len(changes) == 2
    assert (levels, plans) == original
    reverse, _, _ = cm.compose(plans[::-1], levels); assert reverse == table


@pytest.mark.parametrize('bad', ['overlap_same', 'overlap_conflicting', 'generation', 'group', 'freeze', 'tables', 'index', 'lock', 'from'])
def test_compose_refuses_overlap_and_conflicting_generations(bad):
    plans = [simple_plan('v1', [change(task=4)]), simple_plan('v2', [change(task=7)])]
    if bad.startswith('overlap'):
        plans[1]['changes'] = [change(task=4, target='c10' if bad == 'overlap_conflicting' else 'ir45')]
    elif bad == 'generation': plans[1]['generation'] = 'v1'
    elif bad == 'group': plans[1]['groups'] = plans[0]['groups']
    elif bad == 'freeze': plans[1]['freeze'] = {'different': True}
    elif bad == 'tables': plans[1]['tables'] = '/tmp/other_tables'
    elif bad == 'index': plans[1]['original_index'] = '/tmp/other_index'
    elif bad == 'lock': plans[1]['retest_lock_sha256'] = 'different'
    elif bad == 'from': plans[1]['changes'][0]['from'] = 'off'
    with pytest.raises(ValueError): cm.compose(plans, levels_fixture())


def test_v2_emits_only_seven_disjoint_pairs_old_uids_and_keeps_v1_bytes(tmp_path, monkeypatch):
    first = [change(task=4, target='c10')]
    second = [change(cell, task) for cell, task in (
        ('pi05_l10_100_k1', 0), ('pi05_l10_50_k1', 8), ('pi05_spatial_500_k1', 9),
        ('pi05_l10_50_k2', 7), ('groot_l10_200_k1', 7), ('groot_l10_50_k1', 0), ('groot_l10_50_k2', 4))]
    levels, design, freeze, tables, index = sources(tmp_path, monkeypatch)
    monkeypatch.setattr(cm, 'verify', lambda reader: design)
    path1 = tmp_path / 'changes_v1.json'; c.write_json(path1, first)
    v1 = rt.emit(path1, tables=tables, index=index, output=tmp_path / 'v1', trace=tmp_path / 'new')
    prior = snapshot(tmp_path); path2 = tmp_path / 'changes_v2.json'; c.write_json(path2, second)
    class NoOutcomes(a.Reader):
        def jsonl(self, path): raise AssertionError('Dry run opened outcomes')
    monkeypatch.setattr(a, 'Reader', NoOutcomes)
    report = cm.validate(path2, [tmp_path / 'v1/plan.json'], output=tmp_path / 'v2', trace=tmp_path / 'new')
    assert report['additional_tasks'] == 7 and report['additional_episodes'] == 350 and report['combined_tasks'] == 8
    assert report['files_written'] == report['roots_created'] == report['test_a_outcome_reads'] == 0
    assert not (tmp_path / 'v2').exists()
    assert all(Path(p).read_bytes() == data for p, data in prior.items())
    assert all(not Path(p).parent.exists() for p in report['planned_groups'])
    v2 = rt.emit(path2, generation='v2', tables=tables, index=index, output=tmp_path / 'v2', trace=tmp_path / 'new')
    actual = []
    for path in v2['groups']:
        group = checked_group(Path(path)); strategy = RetestStrategy(group)
        graph = strategy.plan([], {'virtual': ServerEndpoint('synthetic', 23101)})
        for stage in graph.stages.values():
            for ep in stage.episodes:
                entry = strategy.entries[ep.yaml_id]; cell = entry['row']['r16_cell']
                assert ep.yaml_id == ep.bundle_id == f'r16rt_{cell}_v2'
                original = make_task_uid(entry['original_arm'], 'eval', ep.task_id, ep.episode_idx)
                assert ep.task_uid == original and ep_seed(0, ep.task_uid) == ep_seed(0, original)
                assert ep.orig_init_state_idx == ep.episode_idx and ep.extra['init_pool'] == 'A'
                assert ep.extra['num_trials_per_task'] == 50
                actual.append((cell, ep.task_id, ep.episode_idx))
    assert len(v2['groups']) == 4 and len(actual) == 350
    assert sorted(actual) == sorted((d['cell'], d['task'], i) for d in second for i in range(50))
    assert (tmp_path / 'v2/queue_retest.txt').read_text().splitlines() == ['@local ' + p for p in v2['groups']]
    assert all(Path(p).read_bytes() == data for p, data in prior.items())


def test_overlap_rejected_before_outcomes_or_output(tmp_path, monkeypatch):
    *_, plans = fixture(tmp_path, monkeypatch, [change(task=4)], [change(task=4, target='c10')])
    monkeypatch.setattr(ca, 'group_index', lambda *args: (_ for _ in ()).throw(AssertionError('Opened originals')))
    out = tmp_path / 'combined'
    with pytest.raises(ValueError, match='Overlapping'):
        cm.apply([tmp_path / 'v1/plan.json', tmp_path / 'v2/plan.json'], out)
    assert not out.exists()


def test_duplicate_plan_rejected(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Duplicate retest plan'):
        cm.apply([tmp_path / 'v1/plan.json', tmp_path / 'v1/plan.json'], tmp_path / 'combined')


def write_outcomes(plan, failures=0, miss_count=1):
    for path in plan['groups']:
        group = checked_group(Path(path)); root = Path(group['root']); journal = []
        c.write_json(root / 'DONE.json', dict(group_sha256=group['group_sha256']))
        c.write_json(root / 'launch.json', dict(server_host='local', worker_host='timan107', STAGE1_ONLY=0,
                                               group_sha256=group['group_sha256'], smoke_episodes=None))
        for entry in group['entries']:
            if entry['passive']: continue
            row = entry['row']; arm = row['arm']
            decisions = [dict(ev='startup', ts=datetime.now(timezone.utc).timestamp() + 1, method_spec=row['method'],
                              kwargs=row['kwargs'], miss_steps=group['k'], argv=row['plugin_args'])]
            for t, i in entry['dispatch_pairs']:
                uid = make_task_uid(entry['original_arm'], 'eval', t, i)
                journal.append(dict(task_uid=uid, yaml_id=arm, accepted=True, status='done' if i >= failures else 'failed',
                                    success=i >= failures, attempt=2, error=None))
                audit = dict(fit_sha256=entry['fit_sha256'], descent_level=row['kwargs']['task_levels'][str(t)])
                base = dict(ev='dec', uid=uid, task_id=t, init=i, arm=arm, yaml_id=arm, bundle=arm, r14=audit)
                # Rejected attempt evidence deliberately has different counts.
                decisions.append(dict(base, step=0, attempt=1, hit=False, vision=True, miss_k=group['k']))
                for j in range(miss_count):
                    decisions.extend((dict(base, step=2*j, attempt=2, hit=False, vision=True, miss_k=group['k']),
                                      dict(base, step=2*j+1, attempt=2, hit=True, vision=False, miss_k=None, source='policy_tail')))
            jl(root / 'runs' / arm / 'server_0/decisions_synthetic.jsonl', decisions)
        jl(root / 'journal.jsonl', journal)


def stub_originals(design, monkeypatch):
    before = {}; original_loader = ca.load_entry
    for cell in rt.CELLS:
        model, _, _, k = rt.parts(cell)
        y = np.ones((10, 50), int); y[:, :3] = 0  # Exactly -6 pp vs full pure inference.
        before[cell] = dict(task=episode(y, model=model, k=k),
                            full_c10=episode(np.ones_like(y), model=model, k=10 if model == 'pi05' else 8),
                            full_descent=episode(np.ones_like(y), model=model, k=10 if model == 'pi05' else 8))
        design['cells'][cell]['full_test_c10'] = dict(arm='synthetic_c10', cell=cell)
        design['cells'][cell]['full_test_descent'] = dict(arm='synthetic_descent', cell=cell)
    def load_entry(group, entry, reader, **kwargs):
        if group.get('retest'): return original_loader(group, entry, reader, **kwargs)
        return before[entry['row']['r16_cell']]['task']
    monkeypatch.setattr(ca, 'load_entry', load_entry)
    monkeypatch.setattr(a, 'load', lambda ref, *args, **kwargs:
                        before[ref['cell']]['full_c10' if ref['arm'] == 'synthetic_c10' else 'full_descent'])
    calc = ra.calculate; monkeypatch.setattr(ra, 'calculate', lambda b, n, d: calc(b, n, d, draws=20))
    return before


def done_originals(index):
    for path in json.loads(index.read_bytes())['groups']:
        group = c.read_group(path)
        c.write_json(Path(group['root']) / 'DONE.json', dict(group_sha256=group['group_sha256']))


def test_combined_end_to_end_same_cell_counts_tables_reports_and_order(tmp_path, monkeypatch):
    first = [change(task=4, target='c10'), change('groot_spatial_500_k2', 1)]
    second = [change(task=7), change('groot_spatial_500_k2', 4)]
    levels, design, _, _, index, plans = fixture(tmp_path, monkeypatch, first, second)
    before = stub_originals(design, monkeypatch); done_originals(index)
    write_outcomes(plans[0], failures=1, miss_count=1); write_outcomes(plans[1], failures=2, miss_count=3)
    prior = snapshot(tmp_path); paths = [tmp_path / 'v1/plan.json', tmp_path / 'v2/plan.json']
    result = cm.apply(paths, tmp_path / 'combined'); after = {cell: dict(arms) for cell, arms in before.items()}
    for cell in rt.CELLS:
        after[cell]['task'] = a.Episodes(**{f: np.asarray(result['merged_episodes'][cell][f]) for f in ('y','n','v','m','cost')}, audit={})
    assert result['before'] == a.calculate_test(before, draws=20)
    assert result['after'] == a.calculate_test(after, draws=20)
    assert len(result['after']['cells']) == 32 and len(result['after']['pooled']) == 8 and len(result['after']['pooled_by_k']) == 2
    assert len(result['changed_tasks']) == 4
    for cell, tasks in ((CELL, (4,7)), ('groot_spatial_500_k2', (1,4))):
        merged = after[cell]['task']
        assert merged.y[tasks[0]].sum() == 49 and merged.y[tasks[1]].sum() == 48
        assert merged.n[tasks[0]].sum() == 100 and merged.n[tasks[1]].sum() == 300
        assert merged.v[tasks[0]].sum() == merged.m[tasks[0]].sum() == 50
        assert merged.v[tasks[1]].sum() == merged.m[tasks[1]].sum() == 150
        assert result['task_provenance'][cell][str(tasks[0])]['generation'] == 'v1'
        assert result['task_provenance'][cell][str(tasks[1])]['generation'] == 'v2'
        for field in ('y','n','v','m','cost'):
            keep = [t for t in range(10) if t not in tasks]
            assert np.array_equal(getattr(merged, field)[keep], getattr(before[cell]['task'], field)[keep])
    assert result['levels'][CELL]['4'] == 'c10' and result['levels'][CELL]['7'] == 'ir45'
    assert all(d['old']['successes'] == 47 and d['old']['sr'] == .94 for d in result['changed_tasks'])
    assert all(Path(path).read_bytes() == data for path, data in prior.items())
    assert json.loads((tmp_path / 'combined/levels.json').read_bytes()) == result['levels']
    text = (tmp_path / 'combined/RETEST.md').read_text()
    assert 'k1_across_sizes' in text and 'task_vs_full_c10' in text and 'task_vs_full_descent' in text
    reverse = cm.apply(paths[::-1], tmp_path / 'reverse')
    for key in ('after', 'levels', 'merged_episodes', 'task_provenance', 'after_vs_before'): assert reverse[key] == result[key]
    with pytest.raises(ValueError, match='fresh combined'): cm.apply(paths, tmp_path / 'combined')


def test_single_generation_is_numerically_identical_to_v1_analysis(tmp_path, monkeypatch):
    levels, design, _, _, index, plans = fixture(tmp_path, monkeypatch)
    stub_originals(design, monkeypatch); done_originals(index); write_outcomes(plans[0])
    monkeypatch.setattr(ra, 'frozen_tables', rt.frozen_tables)
    old = ra.apply(tmp_path / 'v1/plan.json', tmp_path / 'legacy_results')
    new = cm.apply([tmp_path / 'v1/plan.json'], tmp_path / 'combined')
    for key in ('before','after','after_vs_before','merged_episodes'): assert old[key] == new[key]


def test_incomplete_v2_refused_before_any_original_outcomes(tmp_path, monkeypatch):
    *_, plans = fixture(tmp_path, monkeypatch); write_outcomes(plans[0])
    monkeypatch.setattr(ca, 'group_index', lambda *args: (_ for _ in ()).throw(AssertionError('Read original outcomes')))
    with pytest.raises(FileNotFoundError):
        cm.apply([tmp_path / 'v1/plan.json', tmp_path / 'v2/plan.json'], tmp_path / 'combined')
    assert not (tmp_path / 'combined').exists()


def test_combined_lock_preserves_legacy_and_rejects_source_tamper():
    cm.verify(a.Reader())  # Metadata only; no outcome reads.
    class Tamper(a.Reader):
        def read(self, path):
            data = super().read(path)
            return data + b'\n# changed' if Path(path).resolve() == self.target else data
    for name in ('retest_combined.py', 'retest_analysis.py', 'conductor_analysis.py', 'cost.py'):
        reader = Tamper(); reader.target = rt.HERE / name
        with pytest.raises(ValueError): cm.verify(reader)
