"""Synthetic outcomes only; all scratch is supplied by pytest outside the repo."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from . import analyze as a, prepare as p
from .cost import pricing
from exp.offline_search.rounds.r04.k4_eval.cost_ledger import ledger
from exp.offline_search.rounds.r15.descent.analyze import descend, paired


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data)+'\n')


def jl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def synthetic(root,model='pi05',pool='B',k=1,loeo=False):
    width = 30 if pool=='B' else 50
    expected = [(t,i) for t in range(10) for i in range(20,50) if pool=='B'] if pool=='B' else [(t,i) for t in range(10) for i in range(50)]
    name = 'r15d_r16_synthetic' if pool=='B' else 'r16t_synthetic'
    fit = root/'fit.bin'; root.mkdir(parents=True); fit.write_bytes(b'synthetic fit')
    digest = a.sha(fit.read_bytes()); manifest=root/('dev_selection.json' if pool=='B' else 'eval500.json')
    dump(manifest,dict(init_pool=pool,dev=True,pairs=expected) if pool=='B' else expected)
    config=root/'config.yaml';config.write_text(f'miss:\n  num_steps: {k}\n')
    row = dict(arm=name,model=model,dev=pool=='B',init_pool=pool,episodes=width*10,replan_steps=5,
               full_model=True,recording=False,manifest=str(manifest),yaml=str(config),method='synthetic:Method',kwargs={},
               r14_descent_level='ir45',r15_leave_own_episode_out=loeo,
               plugin_args=['--os-fit-artifact',str(fit),'--os-blind','--os-policy-tail','--os-policy-tail-blocks','1',
                            '--os-rollout-seed','2026101401','--os-judge','guard_only'])
    dump(root/'arms.json',[row]);run=root/'runs'/name
    run.mkdir(parents=True);(run/'manifest.json').write_bytes(manifest.read_bytes())
    dump(run/'selection.json',dict(init_pool=pool,manifest=str(run/'manifest.json')))
    if pool=='B':
        contract=dict(init_pool='B',dev=True,fit_sha256=digest,
            manifest_sha256=a.sha(json.dumps(expected,separators=(',',':')).encode()),
            subset=dict(excluded_inits_by_task={str(t):list(range(50 if loeo else 20)) for t in range(10)}),
            **(dict(leave_own_episode_out=True) if loeo else {}))
        dump(run/'pool_contract.json',contract);dump(root/'dev_contracts.json',{name:contract})
    start=dict(ev='startup',ts=datetime.now(timezone.utc).timestamp(),argv=row['plugin_args'],
               kwargs={},method_spec=row['method'],miss_steps=k)
    journal=[];decisions=[]
    for t,i in expected:
        uid=f'{name}:eval:{t}:{i}'
        journal.append(dict(task_uid=uid,init_pool=pool,accepted=True,status='done',success=True,attempt=2,error=None))
        audit=dict(fit_sha256=digest)
        if loeo:
            audit.update(leave_own_episode_out=True,query_pool='B',query_init=i,excluded_episode=t*50+i,
                         donor_episode_ids=[9999],candidate_episode_ids=[9999],parent_donor_rows=[10000])
        old=dict(ev='dec',uid=uid,task_id=t,init=i,step=0,attempt=1,hit=False,vision=True,miss_k=k,r14=audit)
        decisions.extend([old,dict(old,attempt=2),dict(old,attempt=2,step=1,hit=True,vision=False,
                    source='policy_tail',miss_k=None)])
    journal.append(dict(journal[0],attempt=3,error='rejected synthetic retry',success=False))
    decisions.append(copy.deepcopy(decisions[1]))
    jl(run/'client/journal.jsonl',journal);jl(run/'server_synthetic/decisions_0.jsonl',[start,*decisions])
    accepted=[d for d in decisions[:-1] if d['attempt']==2]
    l=ledger(dict(model=model,yaml_patch=dict(miss=dict(num_steps=k)),client_overrides=dict(replan_steps=5)),accepted,[start],table=root/'missing_cost_table.json')
    dump(run/'summary.json',dict(init_pool=pool,arm=name,model=model,complete=width*10,success=width*10,sr=1.,
        journal_rows=len(journal),error_rows=1,retried=width*10,
        per_task={str(t):dict(n=width,success=width) for t in range(10)},cost_ledger=l))
    seal=dict(root=str(root),definitions_sha256={str(x.relative_to(root)):a.sha(x.read_bytes())
             for x in (root/'arms.json',manifest,config,root/'dev_contracts.json') if x.exists()},fit_sha256={str(fit):digest})
    dump(root/'ROOT.json',seal)
    return seal,name,expected,run


@pytest.mark.parametrize('model,pool,k,loeo',[
    ('pi05','A',1,False),('pi05','B',2,False),('groot','A',2,False),('groot','B',1,True),
    ('pi05','B',10,False),('groot','A',8,False)])
def test_loader_pricing_attempts_and_tail(tmp_path,model,pool,k,loeo):
    seal,name,expected,run = synthetic(tmp_path/'root',model,pool,k,loeo)
    ep=a.load(seal,name,expected,pool,k,a.Reader())
    assert ep.y.sum()==len(expected) and ep.n.sum()==len(expected)*2
    assert ep.m.sum()==ep.v.sum()==len(expected)
    prices=pricing(model,k)
    assert ep.stats()['ir']==pytest.approx((prices['vision']+prices['miss'])/2)
    assert ep.audit['duplicate_decisions']==1 and ep.audit['rejected_attempt_decisions']==len(expected)


@pytest.mark.parametrize('bad', ['k','missing','fit','loeo','startup','counts','pool','tail','identity'])
def test_loader_rejects_invalid_records(tmp_path,bad):
    seal,name,expected,run=synthetic(tmp_path/'root',loeo=True)
    path=run/'server_synthetic/decisions_0.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
    if bad=='k': rows[2]['miss_k']=2
    elif bad=='missing': rows=[r for r in rows if r.get('uid')!=f'{name}:eval:0:20']
    elif bad=='fit': rows[2]['r14']['fit_sha256']='0'*64
    elif bad=='loeo': rows[2]['r14']['donor_episode_ids']=[rows[2]['r14']['excluded_episode']]
    elif bad=='startup': rows[0]['miss_steps']=2
    elif bad=='pool': rows[2]['init_pool']='A'
    elif bad=='tail': rows[3]['miss_k']=1
    elif bad=='identity': rows[2]['init']=21
    elif bad=='counts':
        summary=json.loads((run/'summary.json').read_text());summary['cost_ledger']['misses']-=1;dump(run/'summary.json',summary)
    # Change all duplicate rows for the affected identity: last-row semantics are intentional.
    if bad in ('k','fit','loeo','identity'):
        for r in rows:
            if r.get('uid')==rows[2]['uid'] and r.get('attempt')==2 and r.get('step')==0:
                r.update(copy.deepcopy(rows[2]))
    jl(path,rows)
    with pytest.raises(ValueError): a.load(seal,name,expected,'B',1,a.Reader())


def episode(y,model='pi05',k=1,scale=1):
    n=np.ones_like(y)*10*scale;v=n//2;m=n//4;price=pricing(model,k)
    return a.Episodes(y,n,v,m,price['vision']*v+price['miss']*m,{})


def test_full_reference_first_failure_and_actual_fallback():
    full=episode(np.ones((10,30),int),k=10)
    arms={l:episode(full.y.copy()) for l in a.LEVEL_ORDER}
    arms['c10'].y[0,:10]=0 # k-pure can be worse; never becomes tolerance reference.
    arms['ir45'].y[0,0]=0 # equality in 5pp net budget passes.
    arms['ir40'].y[0,:2]=0 # first failure, even though lower levels recover.
    result=a.dev_cell('pi05_l10_50_k1',arms,full,draws=50)
    assert result['tolerances']['5']['levels']['0']=='ir45'
    assert result['tolerances']['0']['levels']['0']=='c10'
    assert result['tolerances']['0']['levels']==result['tolerances']['2']['levels']
    assert result['tolerances']['0']['selected']['successes']==290
    assert result['tolerances']['5']['task_decisions']['0']['passing_below_first_failure']==['ir32','ir25','off']
    assert result['tolerances']['5']['uniform']=='off'
    stats={l:paired(arms[l].y[0],full.y[0]) for l in a.LEVEL_ORDER}
    assert descend(stats,5)['level']==result['tolerances']['5']['task_decisions']['0']['level']
    assert result['tolerances']['0']['task_decisions']['0']['passes']['c10']


def test_pool_cost_ratio_and_paired_cluster_draws():
    all_arms={}
    rng=np.random.default_rng(5)
    shared={s:rng.integers(0,2,(10,50)) for s in ('l10','spatial')}
    for cell in p.CELLS:
        m,s,n,k=p.parts(cell);default=10 if m=='pi05' else 8
        all_arms[cell]=dict(task=episode(shared[s].copy(),m,k,scale=1 if m=='pi05' else 4),
                           full_c10=episode(shared[s].copy(),m,default),full_descent=episode(shared[s].copy(),m,default,scale=2))
    result=a.calculate_test(all_arms,draws=100)
    assert len(result['cells'])==32 and len(result['pooled'])==8
    assert len(result['pooled_by_k'])==2
    assert result['pooled_by_k']['k1_across_sizes']['original_suite_task_init_clusters']==1000
    assert result['pooled_by_k']['k1_across_sizes']['comparisons']['full_c10']['bootstrap_ci95']==[0.,0.]
    for key,row in result['pooled'].items():
        assert row['episode_outcomes']==2000 and row['original_suite_task_init_clusters']==1000
        assert row['comparisons']['full_c10']['bootstrap_ci95']==[0.,0.]
    group=[e['task'] for c,e in all_arms.items() if p.parts(c)[2:]==(50,1)]
    correct=sum(e.cost.sum() for e in group)/sum(e.n.sum() for e in group)
    assert result['pooled']['size50_k1']['arms']['task']['ir']==pytest.approx(correct)
    assert correct!=pytest.approx(np.mean([e.stats()['ir'] for e in group]))


def test_source_pricing():
    assert pricing('pi05',1)['miss']==pytest.approx(.4538,abs=.0002)
    assert pricing('pi05',2)['miss']==pytest.approx(.4976,abs=.0002)
    assert pricing('groot',1)['miss']==pytest.approx(.2583,abs=.0002)
    assert pricing('groot',2)['miss']==pytest.approx(.3431,abs=.0002)
    assert pricing('pi05',10)['miss']==.848 and pricing('groot',8)['miss']==.852


def test_emitter_refuses_before_freeze(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'TRACE',tmp_path/'trace')
    with pytest.raises(FileNotFoundError): p.emit_test(tmp_path/'no_tables')
    assert not (tmp_path/'trace').exists()


def test_frozen_table_hash_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'HERE',tmp_path/'r16');p.HERE.mkdir()
    dump(p.HERE/'design.json',dict(synthetic=True));out=tmp_path/'tables'
    levels={c:{str(t):'off' for t in range(10)} for c in p.CELLS};uniform={c:'off' for c in p.CELLS}
    dump(out/'levels.json',levels);dump(out/'uniform.json',uniform)
    dump(out/'TABLE_FREEZE.json',dict(design_sha256=a.sha((p.HERE/'design.json').read_bytes()),
         sha256={n:a.sha((out/n).read_bytes()) for n in ('levels.json','uniform.json')}))
    assert p.frozen_tables(out)[0]==levels
    levels[p.CELLS[0]]['0']='c10';dump(out/'levels.json',levels)
    with pytest.raises(ValueError,match='Changed frozen'):p.frozen_tables(out)


def test_reader_detects_changes(tmp_path):
    path=tmp_path/'data';path.write_bytes(b'first');reader=a.Reader();reader.read(path)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):reader.stable()


def test_test_start_must_follow_freeze(tmp_path):
    seal,name,expected,run=synthetic(tmp_path/'root',pool='A')
    freeze=dict(frozen_at_utc='2099-01-01T00:00:00+00:00')
    with pytest.raises(ValueError,match='before table freeze'):
        a.load(seal,name,expected,'A',1,a.Reader(),freeze=freeze)


def test_runtime_batcher_mechanism():
    from .replay import batchers,boundaries
    evidence=batchers()
    assert len(evidence)==6
    for model in ('pi05','groot'):
        for k in (1,2,10 if model=='pi05' else 8): boundaries(model,k)


def test_one_step_schedule_is_miss_only_and_defaults_equal():
    from .replay import schedule_default_proof
    from openpi.cache.types import DenoiseSchedule, DIRECTION_ASC, DIRECTION_DESC, groot_n15_schedule, schedule_from_id
    assert schedule_default_proof()['k1_warm_start_refused']
    one=groot_n15_schedule(1)
    assert schedule_from_id(one.schedule_id)==one and one.timesteps==()
    for n in (0,-1,False):
        with pytest.raises(ValueError):groot_n15_schedule(n)
    with pytest.raises(ValueError):DenoiseSchedule('pi05_v1',1,DIRECTION_DESC)
    with pytest.raises(ValueError):DenoiseSchedule('other',1,DIRECTION_ASC)


@pytest.mark.parametrize('k',[1,2,8])
def test_real_groot_cp1_interceptor_runs_requested_head_steps(k):
    import torch
    from tests.cache.groot.test_groot_stage3 import _model_with_flow_head
    from tests.cache.groot.test_groot_interceptor import _StubPolicy,_StubOrchestrator,_obs
    from openpi.cache.groot.staged import GrootStagedRunner
    from openpi.cache.groot.interceptor import GrootCacheInterceptor
    from openpi.cache.components.judge import HitType
    model=_model_with_flow_head(k);runner=GrootStagedRunner(model,verify_upstream=False)
    calls=[];encoder=model.action_head.action_encoder
    def counted(*args):calls.append(True);return encoder(*args)
    model.action_head.action_encoder=counted
    icpt=GrootCacheInterceptor(_StubPolicy(model),runner,orchestrator=_StubOrchestrator(HitType.MISS),miss_num_steps=k)
    out=icpt.get_action(_obs())
    assert len(calls)==k and out['__hit_meta__']['miss_nfe']==k


def test_freeze_and_emitter_on_synthetic_tables_only(tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    work=tmp_path/'r16';work.mkdir();tables=work/'dev_results'
    trace=tmp_path/'trace';trace.mkdir();template=work/'eval500.json'
    dump(template,[(t,i) for t in range(10) for i in range(50)])
    cells={c:dict(tolerances={'5':dict(levels={str(t):'off' for t in range(10)},uniform='off')}) for c in p.CELLS}
    dump(tables/'dev.json',dict(cells=cells,inputs={}))
    (tables/'DESCENT.md').write_text('synthetic only\n')
    for tol in (0,2,5):
        for name,field in [('levels','levels'),('uniform','uniform')]:
            dump(tables/f'{name}_tol{tol}pp.json',{c:r['tolerances']['5'][field] for c,r in cells.items()})
    for name in ('levels','uniform'):(tables/f'{name}.json').write_bytes((tables/f'{name}_tol5pp.json').read_bytes())
    design=dict(cells={c:dict(full_test_c10=dict(row=dict(manifest=str(template)))) for c in p.CELLS})
    dump(work/'design.json',design)
    monkeypatch.setattr(a,'HERE',work);monkeypatch.setattr(p,'HERE',work);monkeypatch.setattr(p,'TRACE',trace)
    monkeypatch.setattr(a,'verify',lambda r:design)
    a.freeze_tables(tables)
    # Use synthetic fitted artifacts; the existing arm emitter, hashes and A contracts stay real.
    adapter=NS(descent_blob=lambda cell,table:dict(spec='synthetic:Method',kwargs=dict(library='current',
                   episode_subset=None,task_levels=table,task_caps=[0.]*10),method=NS(knob_settings={'K':[0.]*10})),
               arm_spec=p.r15.arm_spec,frozen=lambda *args,**kwargs:dict(sha256='synthetic_frozen_sha'))
    monkeypatch.setattr(p,'prior',lambda c:adapter)
    p.emit_test(tables)
    records=json.loads((work/'test_roots.json').read_text())
    assert len(records)==32 and len((work/'queue_test.txt').read_text().splitlines())==32
    for rec in records:
        row=json.loads((Path(rec['root'])/'arms.json').read_text())[0]
        assert row['r16_k']==p.parts(rec['cell'])[3] and not row['dev'] and row['episodes']==500
        table=json.loads((Path(rec['root'])/'descent_table.json').read_text())
        assert table['freeze']==json.loads((tables/'TABLE_FREEZE.json').read_text())
    with pytest.raises(ValueError,match='Tables are frozen'):a.apply_dev(tables)
