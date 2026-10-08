"""CPU-only v3 dispatch and supersession proofs; all outcomes are synthetic."""
import copy
import itertools
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from openpi.conductor.task import ServerEndpoint, make_task_uid
from exp.offline_search.closed_loop.plugin import ep_seed
from . import retest_superseding as sp, retest as rt, retest_combined as cm, retest_analysis as ra
from . import analyze as a, conductor as c, conductor_analysis as ca
from .retest_runner import checked_group, RetestStrategy
from .test_retest import sources, levels_fixture
from .test_retest_combined import simple_plan, snapshot, write_outcomes, stub_originals, done_originals


PAIRS = [('groot_l10_100_k2', 5), ('groot_l10_50_k1', 5), ('groot_l10_50_k2', 1),
         ('groot_l10_50_k2', 5), ('pi05_l10_500_k1', 4), ('pi05_l10_500_k2', 0), ('pi05_l10_50_k1', 6)]
CELL = PAIRS[0][0]


def owner_changes():
    return [dict(cell=cell, task=task, **{'from': 'ir32', 'to': 'c10', 'supersedes': 'v1'}) for cell, task in PAIRS]


def pure_plans():
    owner = owner_changes()
    first = [dict(cell=d['cell'], task=d['task'], **{'from': d['from'], 'to': 'ir45'}) for d in owner]
    first.append(dict(cell=CELL, task=7, **{'from': 'ir32', 'to': 'ir45'}))
    second = [dict(cell=CELL, task=8, **{'from': 'ir32', 'to': 'ir40'})]
    third = sp.normalize(owner, 'v3')
    plans = [simple_plan('v1', first), simple_plan('v2', second), simple_plan('v3', third)]
    plans[2]['superseding_changes'] = owner
    return plans


def fixture(tmp_path, monkeypatch, *, emit_v3=True):
    levels, design, freeze, tables, index = sources(tmp_path, monkeypatch)
    monkeypatch.setattr(sp, 'verify', lambda reader: design)
    monkeypatch.setattr(cm, 'verify', lambda reader: design)
    plans = []
    for plan in pure_plans()[:2]:
        path = tmp_path / f'changes_{plan["generation"]}.json'; c.write_json(path, plan['changes'])
        plans.append(rt.emit(path, generation=plan['generation'], tables=tables, index=index,
                             output=tmp_path / plan['generation'], trace=tmp_path / 'new'))
    owner = tmp_path / 'owner.json'; c.write_json(owner, owner_changes())
    lock = tmp_path / 'superseding_lock.json'
    c.write_json(lock, dict(owner_selection_sha256={str(owner): c.sha(owner)}))
    monkeypatch.setattr(sp, 'LOCK', lock)
    against = [tmp_path / 'v1/plan.json', tmp_path / 'v2/plan.json']
    if emit_v3:
        plans.append(sp.emit(owner, against, output=tmp_path / 'v3', trace=tmp_path / 'new'))
    return levels, design, freeze, tables, index, plans, owner, against


def test_composition_final_tables_no_double_count_and_all_argument_orders():
    plans = pure_plans(); original = copy.deepcopy(plans); levels = levels_fixture()
    expected = sp.compose(plans, levels)
    adjusted, final, owners, superseded = expected
    assert len(final) == 9 and len(superseded) == 7
    assert all(owners[pair] == 'v3' and superseded[pair] == 'v1' for pair in PAIRS)
    assert all(adjusted[cell][str(task)] == 'c10' for cell, task in PAIRS)
    assert owners[CELL, 7] == 'v1' and owners[CELL, 8] == 'v2'
    for order in itertools.permutations(plans): assert sp.compose(order, levels) == expected
    assert plans == original and levels == levels_fixture()


@pytest.mark.parametrize('bad', ['undeclared', 'extra_overlap', 'source_missing', 'source_target', 'source_from',
                                 'target', 'supersedes', 'generation', 'count', 'extra_field', 'duplicate',
                                 'metadata_mismatch', 'freeze', 'tables', 'index', 'lock', 'group', 'same_generation'])
def test_rejects_every_unapproved_overlap_or_conflict(bad):
    plans = pure_plans(); third = plans[2]
    if bad == 'undeclared': third.pop('superseding_changes')
    elif bad == 'extra_overlap': plans[1]['changes'].append(copy.deepcopy(plans[0]['changes'][0]))
    elif bad == 'source_missing': plans = plans[1:]
    elif bad == 'source_target': plans[0]['changes'][0]['to'] = 'ir40'
    elif bad == 'source_from': plans[0]['changes'][0]['from'] = 'off'
    elif bad == 'target': third['changes'][0]['to'] = third['superseding_changes'][0]['to'] = 'ir45'
    elif bad == 'supersedes': third['superseding_changes'][0]['supersedes'] = 'v2'
    elif bad == 'generation': third['generation'] = 'v0'
    elif bad == 'count': third['changes'].pop(); third['superseding_changes'].pop()
    elif bad == 'extra_field': third['superseding_changes'][0]['extra'] = True
    elif bad == 'duplicate': third['changes'][1] = copy.deepcopy(third['changes'][0])
    elif bad == 'metadata_mismatch': third['changes'][0]['to'] = 'ir40'
    elif bad == 'freeze': third['freeze'] = {'different': True}
    elif bad == 'tables': third['tables'] = '/tmp/other'
    elif bad == 'index': third['original_index'] = '/tmp/other'
    elif bad == 'lock': third['retest_lock_sha256'] = 'different'
    elif bad == 'group': third['groups'] = plans[0]['groups']
    elif bad == 'same_generation': plans[1]['generation'] = 'v1'
    with pytest.raises(ValueError): sp.compose(plans, levels_fixture())


def test_preflight_and_emission_exact_seven_pairs_preserve_prior_bytes_uids_seeds_and_fits(tmp_path, monkeypatch):
    levels, design, _, _, _, plans, owner, against = fixture(tmp_path, monkeypatch, emit_v3=False)
    prior = snapshot(tmp_path)
    class NoOutcomes(a.Reader):
        def jsonl(self, path): raise AssertionError('Emission/preflight opened outcomes')
    monkeypatch.setattr(a, 'Reader', NoOutcomes)
    report = sp.validate(owner, against, output=tmp_path / 'v3', trace=tmp_path / 'new')
    assert report['additional_tasks'] == report['superseded_tasks'] == 7
    assert report['additional_episodes'] == 350 and report['combined_tasks'] == 9
    assert report['files_written'] == report['roots_created'] == report['test_a_outcome_reads'] == 0
    assert not (tmp_path / 'v3').exists() and all(not Path(p).exists() for p in report['planned_groups'])
    v3 = sp.emit(owner, against, output=tmp_path / 'v3', trace=tmp_path / 'new')
    assert (tmp_path / 'v3/owner_changes.json').read_bytes() == owner.read_bytes()
    rt.read_plan(tmp_path / 'v3/plan.json', a.Reader())  # Unchanged runner's plan contract.
    actual = []
    for path in v3['groups']:
        group = checked_group(path); strategy = RetestStrategy(group)
        graph = strategy.plan([], {'virtual': ServerEndpoint('synthetic', 23101)})
        assert Path(group['root']).name == f'r16_conductor_retest_{group["model"]}_k{group["k"]}_v3'
        for stage in graph.stages.values():
            for ep in stage.episodes:
                entry = strategy.entries[ep.yaml_id]; cell = entry['row']['r16_cell']
                uid = make_task_uid(entry['original_arm'], 'eval', ep.task_id, ep.episode_idx)
                assert ep.task_uid == uid and ep_seed(0, ep.task_uid) == ep_seed(0, uid)
                assert ep.yaml_id == ep.bundle_id == f'r16rt_{cell}_v3'
                assert ep.orig_init_state_idx == ep.episode_idx and ep.extra['init_pool'] == 'A'
                assert ep.extra['num_trials_per_task'] == 50
                actual.append((cell, ep.task_id, ep.episode_idx))
        for entry in group['entries']:
            cell = entry['row']['r16_cell']
            original = pickle.loads((tmp_path / ('original_' + cell) / 'fit.pkl').read_bytes())
            new = pickle.loads(rt.fit_path(entry['row']).read_bytes())
            assert np.array_equal(original['method'].lower, new['method'].lower)
            assert original['method'].guard == new['method'].guard
            assert Path(entry['row']['manifest']).read_bytes() == (tmp_path / ('original_' + cell) / 'eval500.json').read_bytes()
            for task in range(10):
                assert new['method'].task_levels[str(task)] == ('c10' if (cell, task) in PAIRS else 'ir32')
                if (cell, task) in PAIRS: assert new['method'].knob_settings['K'][task] == 0
    assert len(v3['groups']) == 4 and len(actual) == 350
    assert sorted(actual) == sorted((cell, task, i) for cell, task in PAIRS for i in range(50))
    assert (tmp_path / 'v3/queue_retest.txt').read_text().splitlines() == ['@local ' + p for p in v3['groups']]
    assert all(Path(p).read_bytes() == data for p, data in prior.items())
    with pytest.raises(ValueError, match='Fresh'): sp.emit(owner, against, output=tmp_path / 'v3', trace=tmp_path / 'new')


def test_other_seven_pair_file_is_not_authorized(tmp_path, monkeypatch):
    *_, owner, against = fixture(tmp_path, monkeypatch, emit_v3=False)
    wrong = owner_changes(); wrong[0]['task'] = 7
    path = tmp_path / 'wrong.json'; c.write_json(path, wrong)
    with pytest.raises(ValueError, match='authorized seven-pair'):
        sp.validate(path, against, output=tmp_path / 'v3', trace=tmp_path / 'new')
    assert not (tmp_path / 'v3').exists()


def test_end_to_end_winning_arrays_v1_reference_counts_reports_and_original_estimators(tmp_path, monkeypatch):
    levels, design, _, _, index, plans, owner, against = fixture(tmp_path, monkeypatch)
    before = stub_originals(design, monkeypatch); done_originals(index)
    write_outcomes(plans[0], failures=1, miss_count=1)
    write_outcomes(plans[1], failures=2, miss_count=2)
    write_outcomes(plans[2], failures=5, miss_count=3)
    prior = snapshot(tmp_path); paths = [*against, tmp_path / 'v3/plan.json']
    result = sp.apply(paths, tmp_path / 'combined')
    after = {cell: dict(arms) for cell, arms in before.items()}
    for cell in rt.CELLS:
        after[cell]['task'] = a.Episodes(**{f: np.asarray(result['merged_episodes'][cell][f])
                                          for f in ('y', 'n', 'v', 'm', 'cost')}, audit={})
    assert result['before'] == a.calculate_test(before, draws=20)
    assert result['after'] == a.calculate_test(after, draws=20)
    expected = ra.calculate(before, after, [d for d in sp.compose(plans, levels)[1]])
    assert result['after_vs_before'] == expected['after_vs_before']
    assert len(result['changed_tasks']) == 9 and result['superseded_tasks'] == 7
    for d in result['changed_tasks']:
        key = d['cell'], d['task']; ep = after[d['cell']]['task']; task = d['task']
        if key in PAIRS:
            assert d['generation'] == 'v3' and d['to'] == 'c10'
            assert d['new']['successes'] == 45 and d['new']['sr'] == .9
            old = d['superseded_results'][0]
            assert old['generation'] == 'v1' and old['level'] == 'ir45' and old['result']['successes'] == 49
            assert sum(old['episodes']['n']) == 100 and sum(old['episodes']['v']) == sum(old['episodes']['m']) == 50
            assert ep.n[task].sum() == 300 and ep.v[task].sum() == ep.m[task].sum() == 150
            assert result['task_provenance'][d['cell']][str(task)]['superseded_results'] == d['superseded_results']
        else:
            assert d['superseded_results'] == []
            assert d['new']['successes'] == (49 if d['generation'] == 'v1' else 48)
    changed = set(sp.compose(plans, levels)[2])
    for cell in rt.CELLS:
        keep = [t for t in range(10) if (cell, t) not in changed]
        for f in ('y', 'n', 'v', 'm', 'cost'):
            assert np.array_equal(getattr(after[cell]['task'], f)[keep], getattr(before[cell]['task'], f)[keep])
    text = (tmp_path / 'combined/RETEST.md').read_text()
    assert 'v3 / c10 / 0.9000' in text and 'v1 / ir45 / 0.9800' in text
    assert 'task_vs_full_c10' in text and 'task_vs_full_descent' in text and 'k1_across_sizes' in text
    reverse = sp.apply(paths[::-1], tmp_path / 'reverse')
    for k in ('before', 'after', 'after_vs_before', 'levels', 'changed_tasks', 'task_provenance', 'merged_episodes'):
        assert result[k] == reverse[k]
    assert all(Path(p).read_bytes() == data for p, data in prior.items())
    with pytest.raises(ValueError, match='fresh combined'): sp.apply(paths, tmp_path / 'combined')


@pytest.mark.parametrize('bad', ['missing_done', 'duplicate_plan', 'missing_previous', 'owner_tamper'])
def test_analysis_rejects_invalid_inputs_before_any_original_outcome_read(tmp_path, monkeypatch, bad):
    *_, plans, owner, against = fixture(tmp_path, monkeypatch)
    paths = [*against, tmp_path / 'v3/plan.json']
    if bad != 'missing_done':
        for p in plans: write_outcomes(p)
    if bad == 'duplicate_plan': paths.append(paths[0])
    elif bad == 'missing_previous': paths = paths[1:]
    elif bad == 'owner_tamper': (tmp_path / 'v3/owner_changes.json').write_text('[]')
    monkeypatch.setattr(ca, 'group_index', lambda *args: (_ for _ in ()).throw(AssertionError('Opened original outcomes')))
    with pytest.raises((ValueError, FileNotFoundError)): sp.apply(paths, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_disjoint_legacy_analysis_remains_numerically_identical(tmp_path, monkeypatch):
    _, design, _, _, index, plans, _, against = fixture(tmp_path, monkeypatch, emit_v3=False)
    stub_originals(design, monkeypatch); done_originals(index)
    for p in plans: write_outcomes(p)
    old = cm.apply(against, tmp_path / 'old'); new = sp.apply(against, tmp_path / 'new_results')
    for k in ('before', 'after', 'after_vs_before', 'levels', 'merged_episodes'):
        assert old[k] == new[k]


def test_superseding_lock_checks_new_and_sealed_legacy_sources():
    sp.verify(a.Reader())
    class Tamper(a.Reader):
        def read(self, path):
            data = super().read(path)
            return data + b'\n# tamper' if Path(path).resolve() == self.target else data
    for path in (Path(sp.__file__), rt.HERE / 'retest_combined.py', rt.HERE / 'retest.py',
                 rt.HERE / 'cost.py', rt.HERE / 'retest_runner.py'):
        reader = Tamper(); reader.target = path
        with pytest.raises(ValueError): sp.verify(reader)
