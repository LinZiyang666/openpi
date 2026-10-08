"""CPU synthetic-only retest tests. All emitted roots/outcomes live in /tmp."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpi.conductor.protocol import result_to_wire, task_from_wire
from openpi.conductor.task import EpisodeResult, ServerEndpoint, make_task_uid
from exp.offline_search.closed_loop.plugin import ep_seed
from . import retest as rt, retest_runner as rr, retest_analysis as ra
from . import conductor as c, conductor_analysis as ca, analyze as a
from .cost import pricing
from .test_r16 import episode, jl, synthetic


class FrozenMethod:
    """A synthetic serialized fit, with a fit() tripwire and lower-layer bytes."""
    def __init__(self, levels, caps):
        self.lower = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.guard = b'unchanged frozen guard'; self.r14_fit = {'source_sha256': 'parent'}
        self.leave_own_episode_out = False; self.set_table(levels, caps)

    def fit(self, *args): raise AssertionError('Refit forbidden')

    def set_table(self, levels, caps):
        self.task_levels = dict(levels); self.knob_settings = dict(K=list(caps))


def levels_fixture(): return {cell: {str(t): 'ir32' for t in range(10)} for cell in rt.CELLS}


def sources(tmp_path, monkeypatch):
    """32 fake definition-only cells, four sealed groups; no real outcomes opened."""
    levels = levels_fixture(); freeze = dict(frozen_at_utc='2026-10-07T00:00:00+00:00', sha256={'levels.json': 'fixture'})
    tables = tmp_path / 'tables'; c.write_json(tables / 'TABLE_FREEZE.json', freeze); c.write_json(tables / 'levels.json', levels)
    design = dict(cells={cell: dict(periodic_caps=dict(off=0., ir25=25., ir32=32., ir40=40., ir45=45., c10=0.))
                             for cell in rt.CELLS})
    monkeypatch.setattr(rt, 'verify', lambda reader: design)
    monkeypatch.setattr(rt, 'frozen_tables', lambda directory: (copy.deepcopy(levels), {c: None for c in rt.CELLS}, freeze))
    lock = tmp_path / 'lock.json'; c.write_json(lock, dict(synthetic=True)); monkeypatch.setattr(rt, 'LOCK', lock)
    groups = []
    for model in ('pi05', 'groot'):
        for k in (1, 2):
            root = tmp_path / f'r16_conductor_test_{model}_k{k}_v1'; entries = []
            for cell in rt.CELLS:
                if (rt.parts(cell)[0], rt.parts(cell)[3]) != (model, k): continue
                source = tmp_path / ('original_' + cell); name = 'r16t_' + cell + '_task_descent'
                fit = source / 'fit.pkl'; caps = [32.] * 10; kwargs = dict(library='frozen', episode_subset=[1], task_levels=levels[cell], task_caps=caps)
                blob = dict(spec='synthetic:Method', kwargs=kwargs, method=FrozenMethod(levels[cell], caps))
                rt.atomic(fit, pickle.dumps(blob))
                yaml = source / 'config.yaml'; rt.atomic(yaml, f'miss:\n  num_steps: {k}\n'.encode())
                manifest = source / 'eval500.json'; c.write_json(manifest, rt.PAIRS)
                row = dict(arm=name, r16_cell=cell, model=model, cell=model+'_l10_cache',
                           suite='libero_10' if rt.parts(cell)[1]=='l10' else 'libero_spatial',
                           init_pool='A', dev=False, recording=False, method='synthetic:Method', kwargs=kwargs,
                           r14_descent_level='task_descent', replan_steps=5, full_model=True,
                           yaml=str(yaml), manifest=str(manifest),
                           plugin_args=['--os-fit-artifact',str(fit),'--os-blind','--os-policy-tail',
                                        '--os-policy-tail-blocks','1','--os-rollout-seed','2026101401','--os-judge','guard_only'])
                c.write_json(source / 'arms.json', [row])
                c.write_json(source / 'descent_table.json', dict(freeze=freeze, levels=levels[cell], caps=caps, uniform=None))
                definitions={str(p.relative_to(source)):c.sha(p) for p in source.iterdir()}
                c.write_json(source / 'ROOT.json', dict(definitions_sha256=definitions))
                entries.append(dict(row=row, source_root=str(source), source_seal_sha256=c.sha(source/'ROOT.json'),
                                    fit_sha256=c.sha(fit), yaml_sha256=c.sha(yaml), pairs=rt.PAIRS, freeze=freeze))
            group=dict(schema='r16.conductor.group.v1',root=str(root),model=model,k=k,phase='test',entries=entries,
                       references=None,definitions_sha256={},recording=False,horizon=5,simulator_seed=c.SEED,
                       **{field:c.sha(rt.HERE/name) for field,name in
                          (('design_sha256','design.json'),('protocol_sha256','PROTOCOL.md'),('amendment_sha256','CONDUCTOR_AMENDMENT.json'))})
            group['group_sha256']=a.sha(c.canonical(group));c.write_json(root/'group.json',group);groups.append(str(root/'group.json'))
    index=tmp_path/'index.json';c.write_json(index,dict(groups=groups))
    return levels, design, freeze, tables, index


def emit_fixture(tmp_path, monkeypatch, changes):
    fixture=sources(tmp_path,monkeypatch); path=tmp_path/'changes.json';c.write_json(path,changes)
    plan=rt.emit(path,tables=fixture[3],index=fixture[4],output=tmp_path/'plan',trace=tmp_path/'new')
    return plan, fixture


@pytest.mark.parametrize('subset', ['one','both_suites','all'])
def test_emitter_any_subset_preserves_sources_and_only_dispatches_changes(tmp_path,monkeypatch,subset):
    cells=rt.CELLS if subset=='all' else (('pi05_l10_100_k1',) if subset=='one' else ('groot_l10_50_k2','groot_spatial_500_k2'))
    changes=[dict(cell=cell,task=t,**{'from':'ir32','to':'c10' if t==4 else 'ir45'}) for cell in cells for t in (4,7)]
    plan, (levels,design,freeze,tables,index)=emit_fixture(tmp_path,monkeypatch,changes)
    assert plan['test_a_outcome_reads']==0 and plan['episodes']==100*len(cells)
    actual=[]
    for path in plan['groups']:
        group=rr.checked_group(Path(path)); assert group['phase']=='test' and group['k'] in (1,2)
        strategy=rr.RetestStrategy(group)
        graph=strategy.plan([],{'virtual':ServerEndpoint('host',23101)})
        for stage in graph.stages.values():
            for ep in stage.episodes:
                entry=strategy.entries[ep.yaml_id];cell=entry['row']['r16_cell']
                assert ep.task_uid==make_task_uid(entry['original_arm'],'eval',ep.task_id,ep.episode_idx)
                assert ep_seed(0,ep.task_uid)==ep_seed(0,make_task_uid('r16t_'+cell+'_task_descent','eval',ep.task_id,ep.episode_idx))
                assert ep.yaml_id==ep.bundle_id==f'r16rt_{cell}_v1' and ep.orig_init_state_idx==ep.episode_idx
                assert ep.extra['init_pool']=='A' and ep.extra['num_trials_per_task']==50
                actual.append((cell,ep.task_id,ep.episode_idx))
        for entry in group['entries']:
            cell=entry['row']['r16_cell'];before=pickle.loads((tmp_path/('original_'+cell)/'fit.pkl').read_bytes())
            after=pickle.loads(rt.fit_path(entry['row']).read_bytes())
            assert np.array_equal(before['method'].lower,after['method'].lower)
            assert before['method'].guard==after['method'].guard
            if entry['passive']: assert not entry['dispatch_pairs'] and before['kwargs']==after['kwargs']
            else:
                for t in range(10):assert after['method'].task_levels[str(t)]==('c10' if t==4 else 'ir45' if t==7 else 'ir32')
                assert after['method'].knob_settings['K'][4]==0.
            original_yaml=tmp_path/('original_'+cell)/'config.yaml'
            assert Path(entry['row']['yaml']).read_bytes()==original_yaml.read_bytes()
    assert sorted(actual)==sorted((d['cell'],d['task'],i) for d in changes for i in range(50))
    assert json.loads((tables/'levels.json').read_bytes())==levels
    queue=(tmp_path/'plan/queue_retest.txt').read_text().splitlines()
    assert queue==['@local '+p for p in plan['groups']]


@pytest.mark.parametrize('bad', ['from','same','cell','task','bool','duplicate','extra','empty','level'])
def test_change_validation(bad):
    row=dict(cell='pi05_l10_100_k1',task=4,**{'from':'ir32','to':'ir40'});changes=[row]
    if bad=='from':row['from']='off'
    elif bad=='same':row['to']='ir32'
    elif bad=='cell':row['cell']='pi05_l10_999_k1'
    elif bad=='task':row['task']=10
    elif bad=='bool':row['task']=True
    elif bad=='duplicate':changes.append(dict(row))
    elif bad=='extra':row['ignored']=1
    elif bad=='empty':changes=[]
    elif bad=='level':row['to']='full_c10'
    with pytest.raises(ValueError):rt.adjusted_tables(levels_fixture(),changes)


@pytest.mark.parametrize('target', ['off','ir25','ir40','ir45','c10'])
def test_every_distinct_owner_target_is_accepted(target):
    changes=[dict(cell='pi05_l10_100_k1',task=4,**{'from':'ir32','to':target})]
    adjusted=rt.adjusted_tables(levels_fixture(),changes)
    assert adjusted['pi05_l10_100_k1']['4']==target


@pytest.mark.parametrize('target,cap',[('off',0.),('ir25',25.)])
def test_explicit_off_and_lower_targets_are_emitted_without_dev_descent(tmp_path,monkeypatch,target,cap):
    plan,_=emit_fixture(tmp_path,monkeypatch,[dict(cell='groot_spatial_200_k2',task=4,**{'from':'ir32','to':target})])
    group=rr.checked_group(Path(plan['groups'][0]));entry=next(e for e in group['entries'] if not e['passive'])
    blob=pickle.loads(rt.fit_path(entry['row']).read_bytes())
    assert blob['method'].task_levels['4']==target and blob['method'].knob_settings['K'][4]==cap
    assert all(blob['method'].task_levels[str(t)]=='ir32' for t in range(10) if t!=4)
    assert len(entry['pairs'])==50


def make_driver(group, root, *, servers=2):
    strategy=rr.RetestStrategy(group);endpoints={f'a:{p}':['libero_10','libero_spatial'] for p in range(1,servers+1)}
    return rr.RetestDriver(strategy,endpoints,yaml_weights={e['row']['arm']:len(e['dispatch_pairs']) for e in group['entries']},
                          servers=[ServerEndpoint('a',p) for p in range(1,servers+1)],
                          journal_path=str(root/'journal.jsonl'),ctl_factory=lambda ep:NS())


def test_retest_original_uid_resume_new_count_attempt_fencing_and_arm_logs(tmp_path,monkeypatch):
    plan,_=emit_fixture(tmp_path,monkeypatch,[dict(cell='pi05_l10_100_k1',task=4,**{'from':'ir32','to':'c10'})])
    group=c.read_group(plan['groups'][0]);root=Path(group['root']);d=make_driver(group,root);d.drive_stages_once()
    def pull(driver):
        _,wire=driver.handle_pull('a:1');return task_from_wire(wire['task'])
    done=pull(d);held=pull(d)
    d.handle_result(result_to_wire(EpisodeResult(done.task_uid,True,5,attempt=done.attempt,
                                                 per_step_rows=[dict(step_idx=0)])))
    records=c.completed_rows(root/'journal.jsonl');assert records[done.task_uid]['yaml_id']==done.yaml_id
    assert d._per_step_rows[0]['task_uid']==done.task_uid and d._per_step_rows[0]['yaml_id']==done.yaml_id
    resumed=make_driver(group,root,servers=3);resumed.drive_stages_once();retry=pull(resumed)
    assert retry.task_uid==held.task_uid and retry.attempt>held.attempt
    resumed.handle_result(result_to_wire(EpisodeResult(held.task_uid,False,5,attempt=held.attempt)))
    assert held.task_uid not in c.completed_rows(root/'journal.jsonl')
    resumed.handle_result(result_to_wire(EpisodeResult(retry.task_uid,True,5,attempt=retry.attempt)))
    records=c.completed_rows(root/'journal.jsonl');assert len(records)==2 and records[retry.task_uid]['yaml_id']==retry.yaml_id
    assert all(make_task_uid('r16t_pi05_l10_100_k1_task_descent','eval',4,i) in resumed._scheduler._ep_index for i in range(50))


def sparse_outcomes(tmp_path, model='pi05', k=1, task=4):
    seal,original,pairs,run=synthetic(tmp_path/'original',model=model,pool='A',k=k)
    row=json.loads((Path(seal['root'])/'arms.json').read_bytes())[0]
    name='r16rt_'+model+'_l10_100_k'+str(k)+'_v1';row.update(arm=name,r14_descent_level='task_descent',
        kwargs=dict(task_levels={str(t):'c10' if t==task else 'ir32' for t in range(10)}),r16_cell=model+'_l10_100_k'+str(k))
    group=dict(root=str(tmp_path/'retest'),phase='test',model=model,k=k,group_sha256='synthetic')
    root=Path(group['root']);c.write_json(root/'launch.json',dict(server_host='local',worker_host='timan107',STAGE1_ONLY=0,
                                                               group_sha256='synthetic',smoke_episodes=None))
    selected=[[task,i] for i in range(50)]
    entry=dict(row=row,pairs=selected,dispatch_pairs=selected,original_arm=original,passive=False,
               fit_sha256=seal['fit_sha256'][str(rt.fit_path(row))])
    journal=[dict(r,yaml_id=name) for r in a.Reader().jsonl(run/'client/journal.jsonl') if a.uid_pair(r['task_uid'])[0]==task]
    jl(root/'journal.jsonl',journal)
    rows=list(a.Reader().jsonl(run/'server_synthetic/decisions_0.jsonl'));kept=[]
    for r in rows:
        if r['ev']=='startup':r['kwargs']=row['kwargs'];kept.append(r)
        elif r['task_id']==task:
            r.update(arm=name,yaml_id=name,bundle=name)
            if r['vision']:r['r14']['descent_level']='c10'
            kept.append(r)
    path=root/'runs'/name/'server_0/decisions_synthetic.jsonl';jl(path,kept)
    return group,entry,path


@pytest.mark.parametrize('model,k',[('pi05',1),('pi05',2),('groot',1),('groot',2)])
def test_sparse_load_uses_real_uid_accepted_attempt_cost_and_only_selected_cohort(tmp_path,model,k):
    group,entry,path=sparse_outcomes(tmp_path,model,k)
    ep=ra.load_retest(group,entry,a.Reader(),'2026-10-07T00:00:00+00:00')
    assert ep.y[4].sum()==50 and np.all(ep.y[np.arange(10)!=4]==-1)
    assert ep.n[4].sum()==100 and ep.v[4].sum()==ep.m[4].sum()==50
    price=pricing(model,k)
    assert ep.cost[4].sum()==pytest.approx(50*(price['vision']+price['miss']))
    assert ep.audit['original_episode_uids']
    original=episode(np.zeros((10,50),int),model=model,k=k);merged=ra.merge_cohorts(original,ep,[4])
    for field in ('y','n','v','m','cost'):
        assert np.array_equal(getattr(merged,field)[4],getattr(ep,field)[4])
        assert np.array_equal(getattr(merged,field)[np.arange(10)!=4],getattr(original,field)[np.arange(10)!=4])


@pytest.mark.parametrize('bad',['uid','missing','extra','level','pure_hit','k','freeze','fit'])
def test_sparse_loader_rejects_invalid_retest_evidence(tmp_path,bad):
    group,entry,path=sparse_outcomes(tmp_path);rows=list(a.Reader().jsonl(path))
    if bad=='missing':rows=[r for r in rows if r.get('init')!=49]
    else:
        for r in rows:
            if r['ev']!='dec':continue
            if bad=='uid':r['uid']=r['uid'].replace(entry['original_arm'],entry['row']['arm'])
            elif bad=='extra':r['uid']=make_task_uid(entry['original_arm'],'eval',3,r['init'])
            elif bad=='level' and r['vision']:r['r14']['descent_level']='ir32'
            elif bad=='pure_hit' and r['vision']:r['hit']=True
            elif bad=='k' and not r['hit']:r['miss_k']=2
            elif bad=='fit':r['r14']['fit_sha256']='wrong'
    jl(path,rows)
    with pytest.raises(ValueError):ra.load_retest(group,entry,a.Reader(),'2100-01-01T00:00:00+00:00' if bad=='freeze' else '2026-10-07T00:00:00+00:00')


def test_merge_and_all_32_estimators_equal_existing_conductor_test_analysis():
    before={};after={};changes=[]
    for cell in rt.CELLS:
        m,s,size,k=rt.parts(cell);y=np.ones((10,50),int);y[4,:10]=0
        before[cell]=dict(task=episode(y,model=m,k=k),full_c10=episode(np.ones_like(y),model=m,k=10 if m=='pi05' else 8),
                          full_descent=episode(np.ones_like(y),model=m,k=10 if m=='pi05' else 8))
        after[cell]=dict(before[cell]);retest=episode(np.ones_like(y),model=m,k=k,scale=2)
        retest.y[np.arange(10)!=4]=-1;retest.audit['completed_tasks']=[4]
        after[cell]['task']=ra.merge_cohorts(before[cell]['task'],retest,[4])
        changes.append(dict(cell=cell,task=4,**{'from':'ir32','to':'c10'}))
    result=ra.calculate(before,after,changes,draws=30)
    assert result['before']==a.calculate_test(before,draws=30) and result['after']==a.calculate_test(after,draws=30)
    assert len(result['after']['pooled'])==8 and len(result['changed_tasks'])==32
    for key,stats in result['after']['pooled'].items():
        assert stats['arms']['task']['sr']==1. and result['after_vs_before']['pooled'][key]['delta']==pytest.approx(.02)
        # Cost/N ratio, not an unweighted mean of episode or task IRs.
        group=[cell for cell in rt.CELLS if 'size'+str(rt.parts(cell)[2])+'_k'+str(rt.parts(cell)[3])==key]
        cost=sum(after[cell]['task'].cost.sum() for cell in group);n=sum(after[cell]['task'].n.sum() for cell in group)
        assert stats['arms']['task']['ir']==pytest.approx(cost/n)
    assert result['post_test_selected'] and result['measurement_only'] is False


def test_synthetic_emit_run_outputs_merge_end_to_end(tmp_path,monkeypatch):
    changes=[dict(cell=cell,task=t,**{'from':'ir32','to':'c10' if t==4 else 'ir40'})
             for cell,t in (('pi05_spatial_100_k1',4),('groot_l10_500_k2',7))]
    plan,(levels,design,freeze,tables,index)=emit_fixture(tmp_path,monkeypatch,changes)
    monkeypatch.setattr(ra,'frozen_tables',rt.frozen_tables)
    # Original/reference outcomes are synthetic matrices, not existing run data.
    old_loader=ca.load_entry
    def synthetic_original(group,entry,reader,**kwargs):
        if group.get('retest'):return old_loader(group,entry,reader,**kwargs)
        row=entry['row'];return episode(np.zeros((10,50),int),model=row['model'],k=group['k'])
    monkeypatch.setattr(ca,'load_entry',synthetic_original)
    monkeypatch.setattr(a,'load',lambda ref,arm,pairs,pool,k,reader:episode(np.ones((10,50),int),model='groot' if k==8 else 'pi05',k=k))
    for source in design['cells'].values():
        source['full_test_c10']=source['full_test_descent']=dict(arm='synthetic')
    for path in json.loads(index.read_bytes())['groups']:
        g=c.read_group(path);c.write_json(Path(g['root'])/'DONE.json',dict(group_sha256=g['group_sha256']))
    for path in plan['groups']:
        g=rr.checked_group(Path(path));root=Path(g['root']);journal=[]
        c.write_json(root/'launch.json',dict(server_host='local',worker_host='timan107',STAGE1_ONLY=0,
                                           group_sha256=g['group_sha256'],smoke_episodes=None))
        c.write_json(root/'DONE.json',dict(group_sha256=g['group_sha256']))
        for entry in g['entries']:
            if entry['passive']:continue
            row=entry['row'];name=row['arm'];decisions=[dict(ev='startup',ts=datetime.now(timezone.utc).timestamp()+1,
                method_spec=row['method'],kwargs=row['kwargs'],miss_steps=g['k'],argv=row['plugin_args'])]
            for t,i in entry['dispatch_pairs']:
                uid=make_task_uid(entry['original_arm'],'eval',t,i)
                journal.append(dict(task_uid=uid,yaml_id=name,accepted=True,status='done',success=True,attempt=1,error=None))
                audit=dict(fit_sha256=entry['fit_sha256'],descent_level=row['kwargs']['task_levels'][str(t)])
                base=dict(ev='dec',uid=uid,task_id=t,init=i,attempt=1,arm=name,yaml_id=name,bundle=name,r14=audit)
                decisions += [dict(base,step=0,hit=False,vision=True,miss_k=g['k']),
                              dict(base,step=1,hit=True,vision=False,miss_k=None,source='policy_tail')]
            jl(root/'runs'/name/'server_0/decisions_fake.jsonl',decisions)
        jl(root/'journal.jsonl',journal)
    calculate=ra.calculate;monkeypatch.setattr(ra,'calculate',lambda b,n,d:calculate(b,n,d,draws=20))
    result=ra.apply(tmp_path/'plan/plan.json',tmp_path/'analysis')
    assert len(result['after']['cells'])==32 and len(result['after']['pooled'])==8
    for d in changes:
        cell=d['cell'];t=d['task'];assert result['after']['cells'][cell]['arms']['task']['sr']==.1
        merged=result['merged_episodes'][cell];assert sum(merged['y'][t])==50 and sum(merged['m'][t])==50
        assert all(sum(merged['y'][j])==0 for j in range(10) if j!=t)
    assert 'after' in (tmp_path/'analysis/RETEST.md').read_text().lower()
    with pytest.raises(ValueError,match='fresh analysis'):ra.apply(tmp_path/'plan/plan.json',tmp_path/'analysis')


def test_retest_lock_rejects_new_code_and_measurement_changes():
    rt.verify(a.Reader())  # Definitions only; no outcome reads.
    class Tamper(a.Reader):
        def read(self,path):
            raw=super().read(path)
            return raw+b'\n# changed' if Path(path).resolve()==self.target else raw
    for name in ('retest.py','conductor_analysis.py','cost.py'):
        reader=Tamper();reader.target=rt.HERE/name
        with pytest.raises(ValueError):rt.verify(reader)


@pytest.mark.parametrize('bad',['plan','table','changes','group','fit'])
def test_emitted_retest_selection_and_group_tamper_rejected(tmp_path,monkeypatch,bad):
    plan,_=emit_fixture(tmp_path,monkeypatch,[dict(cell='pi05_l10_100_k1',task=4,**{'from':'ir32','to':'c10'})])
    path=Path(plan['groups'][0]);group=c.read_group(path)
    if bad=='plan':
        plan['selected_at_utc']='2020-01-01T00:00:00+00:00';c.write_json(tmp_path/'plan/plan.json',plan)
    elif bad=='table':
        adjusted=json.loads((tmp_path/'plan/levels.json').read_bytes());adjusted['pi05_l10_100_k1']['0']='off'
        c.write_json(tmp_path/'plan/levels.json',adjusted)
    elif bad=='changes':c.write_json(tmp_path/'plan/changes.json',[])
    elif bad=='fit':rt.fit_path(group['entries'][0]['row']).write_bytes(b'altered fit')
    elif bad=='group':
        group['entries'][0]['row']['arm']='wrong_new_arm';group.pop('group_sha256');group['group_sha256']=a.sha(c.canonical(group))
        c.write_json(path,group)
    with pytest.raises(ValueError):rr.checked_group(path)


def test_runner_cli_adapter_is_controller_local_and_restores_originals(tmp_path,monkeypatch):
    plan,_=emit_fixture(tmp_path,monkeypatch,[dict(cell='groot_spatial_200_k2',task=4,**{'from':'ir32','to':'ir45'})])
    original=(c.R16Strategy,c.R16Driver,rr.runner.read_group);seen=[]
    def run(args):
        assert args.groot_servers_per_suite==3 and args.workers==4
        assert c.R16Strategy is rr.RetestStrategy and c.R16Driver is rr.RetestDriver
        group=rr.runner.read_group(args.group)
        specs=rr.runner.server_specs(group,[23170,23171,23172,23173,23174,23175],1,groot_servers_per_suite=3)
        assert [s['suite'] for s in specs]==['libero_10']*3+['libero_spatial']*3
        assert all(s['registry']==str(Path(group['root'])/'group.json') for s in specs)
        seen.append(group)
    monkeypatch.setattr(rr.runner,'run',run)
    monkeypatch.setattr(sys,'argv',['retest_runner','run','--group',plan['groups'][0],
                                 '--agent-island','/scratch/zixuans8/openpi_r16_conductor_synthetic',
                                 '--workers','4','--groot-servers-per-suite','3'])
    rr.main()
    assert len(seen)==1 and (c.R16Strategy,c.R16Driver,rr.runner.read_group)==original
