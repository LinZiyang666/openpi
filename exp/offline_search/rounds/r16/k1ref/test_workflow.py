"""CPU-only proofs: no listeners, workers, model construction or network calls."""
import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpi.conductor.task import ServerEndpoint, EpisodeResult
from openpi.conductor.protocol import result_to_wire, task_from_wire
from .. import analyze as a, conductor as c, retest as rt
from ..test_retest import sources
from ..test_r16 import synthetic, jl
from . import workflow as w, runner as run, snapshot


def episode(successes=50, *, model='pi05', scale=1):
    y = np.ones((10, 50), dtype=int); y[:, successes:] = 0
    n = np.full((10, 50), 2 * scale, dtype=int); v = n // 2; m = v.copy()
    p = w.pricing(model, 1)
    return a.Episodes(y, n, v, m, p['vision'] * v + p['miss'] * m, {})


def state_fixture():
    state = dict(dev={}, references={}, old_levels={}, test={}, pure={})
    for cell in w.CELLS:
        model, suite, _, _ = w.parts(cell)
        state['references'][cell] = [29] * 10
        state['old_levels'][cell] = {str(t): 'ir32' for t in range(10)}
        state['pure'][f'{model}_{suite}'] = c.pack(episode(model=model), {})
        for t in range(10):
            for level in w.ORDER:
                state['dev'][w.key(cell, t, level)] = 29
                state['test'][w.key(cell, t, level)] = w.take(episode(model=model), t)
    return state


@pytest.mark.parametrize('successes,reference,passed', [(28,29,True),(27,29,False),(0,1,True),(0,2,False),(30,30,True)])
def test_dev_integer_boundary(successes, reference, passed):
    assert w.dev_pass(successes, reference) is passed


@pytest.mark.parametrize('deficit', [0, 1, 2, 3, 4, 5, 6, 50, -1])
@pytest.mark.parametrize('level', w.ORDER)
def test_raise_integer_boundaries_and_exactly_one_level(level, deficit):
    expected = level if deficit <= 2 or level == 'c10' else 'c10' if deficit >= 5 else w.ORDER[w.ORDER.index(level)-1]
    assert w.raise_level(level, deficit) == expected


def test_dev_missing_one_at_a_time_and_first_failure_ignores_lower_outcomes():
    state = state_fixture(); cell = w.CELLS[0]
    del state['dev'][w.key(cell, 4, 'ir32')]
    first = w.dev_selection(state)
    assert first['missing'] == [dict(cell=cell, task=4, level='ir32')]
    state['dev'][w.key(cell, 4, 'ir32')] = 27
    second = w.dev_selection(state)
    assert not second['missing'] and second['levels'][cell]['4'] == 'ir40'
    assert second['decisions'][f'{cell}/4']['first_failure'] == 'ir32'


def test_real_frozen_source_reproduces_31_19_and_570_next_episodes():
    old = json.loads((w.R16 / 'dev_results_conductor/dev.json').read_bytes())
    state = dict(dev={}, references={}, old_levels={})
    for cell in w.CELLS:
        stats = old['cells'][cell]['task_stats']
        state['old_levels'][cell] = old['cells'][cell]['tolerances']['5']['levels']
        state['references'][cell] = [stats[str(t)]['c10']['successes'] for t in range(10)]
        for t, levels in stats.items():
            for level, row in levels.items(): state['dev'][w.key(cell, int(t), level)] = row['successes']
    result = w.dev_selection(state)
    assert result['resolved_changes'] == 31 and len(result['missing']) == 19 and result['next_episodes'] == 570


def test_full_system_integer_tie_bypasses_bad_task_and_reuses_off():
    state = state_fixture(); cell = w.CELLS[0]
    # Pure and off each total 490, but off task 0 has a deficit of ten.
    pure = episode(); pure.y[1, :10] = 0
    m, s, _, _ = w.parts(cell); state['pure'][f'{m}_{s}'] = c.pack(pure, {})
    state['test'][w.key(cell, 0, 'off')] = w.take(episode(40), 0)
    selection = w.resolve_test(state, state['old_levels'])
    assert selection['rules'][cell]['rule'] == 'full_system'
    assert set(selection['levels'][cell].values()) == {'off'} and selection['complete']
    del state['dev'][w.key(cell, 0, 'ir45')]
    operational = w.operational_dev_selection(state)
    assert cell in operational['untuned_cells'] and not operational['missing']


def test_raise_reuses_intermediate_measurement_without_dispatch():
    state = state_fixture(); cell = w.CELLS[0]
    state['test'][w.key(cell, 0, 'off')] = w.take(episode(40), 0)  # force knob
    state['old_levels'][cell]['0'] = 'ir25'
    state['test'][w.key(cell, 0, 'ir25')] = w.take(episode(47), 0)
    state['test'][w.key(cell, 0, 'ir32')] = w.take(episode(46), 0)
    del state['test'][w.key(cell, 0, 'ir40')]
    initial = w.resolve_test(state, state['old_levels'], allow_raise=False)
    assert not initial['missing'] and not initial['complete'] and initial['levels'][cell]['0'] == 'ir25'
    raised = w.resolve_test(state, initial['levels'])
    assert raised['missing'] == [dict(cell=cell, task=0, level='ir40')]
    state['test'][w.key(cell, 0, 'ir40')] = w.take(episode(45), 0)
    final = w.resolve_test(state, raised['levels'])
    assert final['complete'] and final['levels'][cell]['0'] == 'c10'


def emitted(tmp_path, monkeypatch):
    levels, design, freeze, tables, index = sources(tmp_path / 'sources', monkeypatch)
    monkeypatch.setattr(w, 'ROOT', tmp_path / 'new')
    monkeypatch.setattr(w.a, 'verify', lambda reader: design)
    rt.write_json(w.ROOT / 'inventory/inventory.json', {})
    rt.write_json(w.ROOT / 'CODE_LOCK.json', {})
    groups = [c.read_group(p) for p in json.loads(index.read_bytes())['groups']]
    original_groups = {g['model']: str(Path(g['root'])/'group.json') for g in groups if g['k'] == 1}
    originals = {e['row']['r16_cell']: e for g in groups if g['k'] == 1 for e in g['entries']}
    state = dict(old_levels=levels, originals=originals, original_groups=original_groups)
    req = [dict(cell='groot_spatial_200_k1', task=4, level='off'),
           dict(cell='pi05_l10_100_k1', task=7, level='c10')]
    return w.emit('test00', 'test', req, state), state, req


def driver(group, servers=2):
    strategy = run.SparseStrategy(group)
    return run.RetestDriver(strategy, {f'a:{i}': ['libero_10','libero_spatial'] for i in range(1,servers+1)},
        yaml_weights={e['row']['arm']: len(e['dispatch_pairs']) for e in group['entries']},
        servers=[ServerEndpoint('a', i) for i in range(1,servers+1)], journal_path=str(Path(group['root'])/'journal.jsonl'),
        ctl_factory=lambda ep: NS())


def test_sparse_emit_preserves_exact_uid_and_resume_fences(tmp_path, monkeypatch):
    plan, state, req = emitted(tmp_path, monkeypatch)
    assert plan['episodes'] == 100
    for path in plan['groups']:
        group = w.checked_group(path); d = driver(group); d.drive_stages_once()
        graph = d._graph
        assert sum(len(s.episodes) for s in graph.stages.values()) == 50
        for stage in graph.stages.values():
            e = d._strategy.entries[stage.yaml_id]
            for ep in stage.episodes:
                assert ep.task_uid == c.make_task_uid(e['original_arm'], 'eval', ep.task_id, ep.episode_idx)
                assert ep.bundle_id == ep.yaml_id and ep.extra['init_pool'] == 'A'
        _, wire = d.handle_pull('a:1'); held = task_from_wire(wire['task'])
        resumed = driver(group, servers=3); resumed.drive_stages_once()
        _, wire = resumed.handle_pull('a:1'); retry = task_from_wire(wire['task'])
        assert held.task_uid == retry.task_uid and retry.attempt > held.attempt
        resumed.handle_result(result_to_wire(EpisodeResult(held.task_uid,False,5,attempt=held.attempt)))
        assert held.task_uid not in c.completed_rows(Path(group['root'])/'journal.jsonl')
        resumed.handle_result(result_to_wire(EpisodeResult(retry.task_uid,True,5,attempt=retry.attempt)))
        assert retry.task_uid in c.completed_rows(Path(group['root'])/'journal.jsonl')
    with pytest.raises(ValueError, match='Fresh output'): w.emit('test00','test',req,state)


def test_group_plan_and_fit_tamper_rejected(tmp_path, monkeypatch):
    plan, _, _ = emitted(tmp_path, monkeypatch)
    p = Path(plan['groups'][0]); group = w.checked_group(p)
    rt.fit_path(group['entries'][0]['row']).write_bytes(b'tampered')
    with pytest.raises(ValueError): w.checked_group(p)


@pytest.mark.parametrize('loeo', [False, True])
@pytest.mark.parametrize('bad', [None, 'uid', 'k', 'missing', 'fit'])
def test_sparse_B_loader_preserves_cohort_and_loeo(tmp_path, loeo, bad):
    source, original, pairs, old_run = synthetic(tmp_path/'old', pool='B', k=1, loeo=loeo)
    row = json.loads((Path(source['root'])/'arms.json').read_bytes())[0]
    name = original + '_new'; row['arm'] = name
    group = dict(root=str(tmp_path/'new'), model='pi05', k=1, phase='dev', group_sha256='fixture',
                 selected_at_utc='2000-01-01T00:00:00+00:00')
    selected = [[t,i] for t,i in pairs if t == 4]
    entry = dict(row=row, original_arm=original, pairs=selected, dispatch_pairs=selected, template_pairs=pairs,
                 levels={str(t):'ir45' for t in range(10)}, fit_sha256=source['fit_sha256'][str(rt.fit_path(row))],
                 pool_contract=json.loads((Path(source['root'])/'dev_contracts.json').read_bytes())[original])
    root=Path(group['root']);rt.write_json(root/'launch.json',dict(server_host='local',worker_host='timan107',
        STAGE1_ONLY=0,group_sha256='fixture',smoke_episodes=None))
    journal=[dict(r,yaml_id=name) for r in a.Reader().jsonl(old_run/'client/journal.jsonl') if a.uid_pair(r['task_uid'])[0]==4]
    jl(root/'journal.jsonl',journal);rows=[]
    for r in a.Reader().jsonl(old_run/'server_synthetic/decisions_0.jsonl'):
        if r['ev']=='startup':rows.append(r)
        elif r['task_id']==4:
            r.update(arm=name,yaml_id=name,bundle=name)
            if bad=='uid':r['uid']=r['uid'].replace(original,name)
            elif bad=='k' and not r['hit']:r['miss_k']=2
            elif bad=='fit':r['r14']['fit_sha256']='wrong'
            if bad!='missing' or r['init']!=49:rows.append(r)
    jl(root/'runs'/name/'server_0/decisions_fake.jsonl',rows)
    if bad:
        with pytest.raises(ValueError):w.load_sparse(group,entry,a.Reader())
    else:
        ep=w.load_sparse(group,entry,a.Reader())
        assert ep.audit['completed_tasks']==[4] and ep.y[4].sum()==30 and ep.n[4].sum()==60
        assert np.all(ep.y[np.arange(10)!=4]==-1) and ep.m.sum()==ep.v.sum()==30


def test_original_estimators_and_d5_denominators_in_final_summary(tmp_path):
    state = state_fixture(); selection = w.resolve_test(state, state['old_levels'])
    result = w.analysis(state, selection, draws=25)
    assert len(result['summary']) == 16 and len(result['merged_episodes']) == 16
    for row in result['summary']:
        assert row['sr'] == row['baseline_sr'] == 1. and row['diff_pp'] == 0.
        assert row['rule'] == 'full_system' and row['ir'] == pytest.approx(row['pure_ir'])
    arms = {cell: dict(task=w.matrix(state,cell,selection['levels'][cell]), full_c10=w.pure_for(state,cell),
                       full_descent=w.pure_for(state,cell)) for cell in w.CELLS}
    expected = w.calculate(arms, draws=25)
    assert result['cells'] == expected['cells'] and result['pooled'] == expected['pooled']
    w.write_analysis(tmp_path/'result',result)
    assert len(json.loads((tmp_path/'result/summary.json').read_text())) == 16


@pytest.mark.parametrize('model,count', [('pi05',3),('groot',4)])
def test_ports_refuse_live_23179_and_enforce_layout(model,count):
    group = dict(model=model,root='/new/group')
    args = NS(port=23159,ports=None,groot_servers_per_suite=None)
    run.validate_ports(args,group); assert args.ports == list(range(23150,23150+count))
    args.port=23179
    with pytest.raises(ValueError, match='Fast-WAM'): run.validate_ports(args,group)
    args.port=23159; args.ports=[23170+i for i in range(count)]
    with pytest.raises(ValueError): run.validate_ports(args,group)


def test_snapshot_verifies_all_manifest_entries_and_rejects_tamper(tmp_path):
    p=tmp_path/'source.py';p.write_text('frozen')
    (tmp_path/'R16_SNAPSHOT.sha256').write_text(f'{snapshot.sha(p)}  {p}\n')
    snapshot.verify(tmp_path);p.write_text('changed')
    with pytest.raises(ValueError):snapshot.verify(tmp_path)
