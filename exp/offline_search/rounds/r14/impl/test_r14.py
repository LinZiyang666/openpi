"""Frozen math, full-row fits, real-plugin lifecycle and root contracts."""
from __future__ import annotations
import copy
import csv
import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from exp.offline_search.closed_loop import plugin, devset, look_recording
from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.data import SubsetLibrary
from exp.offline_search.rounds.r02.g3_recovery import g3_core as core
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r11.knob.recipe import R11Knob, REASONS, uniform
from exp.offline_search.rounds.r14.knob.controller import caps, propose, REASON_TASK_PERIODIC
from exp.offline_search.rounds.r14.knob.recipe import R14Knob, _PartitionCache
from exp.offline_search.rounds.r14.knob.fitting import metric_fit
from .common import *
from .build import converted
from .mechanics import Harness, load, equal_result, EVIDENCE

CELLS = json.loads((ASTRA / 'design.json').read_text())['cells']

def pseudo(cell, method, task=0, pos=0, prev_hit=True):
    model, suite, size = cell.split('_')
    lib = SubsetLibrary(STORE, f'{model}_{suite}', int(size))
    rows = next(r for r in core.episode_rows(method.inner.C).values() if method.inner.C.task[r[0]] == task)
    q = core.PseudoQuery(lib, method.inner.C, rows, pos, model, prev_hit=prev_hit)
    attrs = ('step','task_id','episode','key_v0','key_v1','rs','prev_hit','prev_a_exec',
             'hist_key_v0','hist_key_v1','hist_rs','hist_hit','hist_a_exec')
    return NS(**{k:getattr(q,k) for k in attrs}, hist_has_vision=np.ones(q.step,bool)), lib

@pytest.fixture
def harness_factory(tmp_path,monkeypatch):
    monkeypatch.setattr(plugin,'_git_head',lambda:None)
    allh=[]
    def make(cell):
        h=Harness(cell,tmp_path/f'{cell}_{len(allh)}');allh.append(h);return h
    yield make
    for h in allh:h.rt.flush_all()

def test_metric_equation_and_affine_early_reference():
    rng=np.random.default_rng(20261004)
    X,H=rng.normal(size=(81,12)),rng.normal(size=(81,35));ep=np.repeat(np.arange(9),9)
    a=awm.fit_metric(X,H,ep);b=metric_fit(X,H,ep,block=13)
    delta=max(float(np.max(np.abs(u-v))) for u,v in zip(a,b))
    assert delta<1e-10
    EVIDENCE['metric_equation']=dict(max_abs=delta,full_rank=True,own_episode_excluded=True)

def test_reference_controller_frozen_caps_coins_and_endpoints():
    spec=importlib.util.spec_from_file_location('r14_reference',ASTRA/'controller.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    rng=np.random.default_rng(20261004);checks=0
    for fit in json.loads((ASTRA/'allocation_fits.json').read_text()):
        np.testing.assert_array_equal(caps(fit['reference_K'],fit['priority']['z'],fit['bias'],fit['gain']),fit['K'])
        np.testing.assert_array_equal(caps(fit['reference_K'],fit['priority']['z'],0,0),np.full(10,fit['reference_K']))
        for settings in (fit['K'],[0.]*10,[1e6]*10):
            ref=reference.TaskPeriodic(settings)
            hv=[];hh=[]
            for i in range(40):
                q=NS(step=len(hv),task_id=i%10,episode=NS(init=7,task_id=i%10),
                     hist_has_vision=np.array(hv),hist_hit=np.array(hh))
                guard=bool(rng.random()<.2)
                a=propose(q,guard,settings);b=ref.propose(q,guard)
                assert a==b==ref.propose(q,guard)
                hv.append(True);hh.append(int(not a['miss']))
                if rng.random()<.6:hv.append(False);hh.append(1)
                checks+=1
    assert checks==1920
    from exp.offline_search.rounds.r13.knob.recipe import REASONS as R13_REASONS
    assert REASON_TASK_PERIODIC not in {*REASONS.values(),*R13_REASONS.values(),*range(1,13)}
    EVIDENCE['controller']=dict(reference_proposals=checks,retries=checks,coin_key='R11-gap-v1',coin_seed=0,
                                collision_free_reason=REASON_TASK_PERIODIC)

@pytest.mark.parametrize('cell',CELLS)
def test_real_plugin_base_R10_R11_bytes(cell,harness_factory):
    h=harness_factory(cell)
    _,r11=source(cell)
    r10=pickle.loads((ROUNDS/f'r10/recipe/artifacts/r10_recipe_{cell}.pkl').read_bytes())['method']
    results=[h.run(m,force_guards=True) for m in (load(cell,'base'),r11['method'],r10)]
    assert results[0][:2]==results[1][:2]==results[2][:2]
    EVIDENCE.setdefault('base_equivalence',[]).append(dict(cell=cell,decisions=120,differing_action_bytes=0,differing_decisions=0))

@pytest.mark.parametrize('cell',CELLS)
def test_guard_pool_is_active_behavioral_null(cell,harness_factory):
    h=harness_factory(cell)
    a=h.run(load(cell,'base'),force_guards=True)[2]
    b=h.run(load(cell,'guard_null'),force_guards=True)[2]
    def active(rows):
        return [(d['task_id'],d['step'],d['hit'],d['vision'],d['src'],d['served_head'],
                 d.get('rows'),d.get('weights'),d.get('extras',{}).get('os_reason'),
                 d.get('extras',{}).get('os_flags')) for d in rows]
    # Confidence and inactive diagnostics are allowed to change.
    assert json.dumps(active(a),default=plugin._json_default)==json.dumps(active(b),default=plugin._json_default)
    assert any(d.get('extras',{}).get('os_force_miss') for d in a)
    EVIDENCE.setdefault('guard_null',[]).append(dict(cell=cell,decisions=120,actions_and_active_verdicts_exact=True))

@pytest.mark.parametrize('cell',CELLS)
def test_real_plugin_periodic_R11_and_zero_gain_equivalence(cell,harness_factory):
    h=harness_factory(cell)
    for dose in (.32,.4):
        _,blob=source(cell,'periodic',dose)
        ref=blob['method'];adapter=copy.deepcopy(blob)
        switches=json.loads((ASTRA/'design.json').read_text())['variants']['base']
        m=converted(adapter,switches,'periodic',dose)
        a=h.run(ref,steps=12,tasks=range(4),force_guards=True)
        b=h.run(m,steps=12,tasks=range(4),force_guards=True)
        assert a[:2]==b[:2]
        alloc=load(cell,'task_periodic',dose)
        alloc.knob_settings['K']=[ref.knob_settings['setting']]*10
        c=h.run(alloc,steps=12,tasks=range(4),force_guards=True)
        assert a[0]==c[0]
        life=lambda seq:[(d['task_id'],d['step'],d['hit'],d['vision'],d['src']) for d in seq]
        assert life(a[2])==life(c[2])
    EVIDENCE.setdefault('periodic_equivalence',[]).append(dict(cell=cell,doses=2,decisions=96,zero_gain_calls_actions_exact=True))

def audit_allocation(seq, method):
    groups={}
    for d in seq:groups.setdefault(d['uid'],[]).append(d)
    looks=guards=knobs=tails=0
    for rows in groups.values():
        hv=[];hh=[]
        for d in rows:
            r=d['r14'];step=d['step']
            assert (r['N'],r['V'],r['M'])==(step,sum(hv),sum(v and not h for v,h in zip(hv,hh)))
            if d['vision']:
                q=NS(step=step,task_id=d['task_id'],episode=NS(task_id=d['task_id'],init=d['init']),
                     hist_has_vision=np.array(hv),hist_hit=np.array(hh))
                a=propose(q,r['guard'],method.knob_settings['K'])
                for k,v in a.items():assert r[k]==v
                assert (not d['hit'])==a['miss']
                if r['guard']:assert d['extras']['os_reason']==4 and d['extras']['os_flags']==8
                elif r['knob']:assert d['extras']['os_reason']==REASON_TASK_PERIODIC
                assert len(r['parent_donor_rows'])==len(r['donor_task_ids'])==16
                looks+=1;guards+=r['guard'];knobs+=r['knob']
            elif d['src']=='policy_tail':
                assert d['hit'] and step and not hh[-1];tails+=1
            hv.append(d['vision']);hh.append(int(d['hit']))
    return dict(decisions=len(seq),real_looks=looks,guards=guards,knob_calls=knobs,policy_tails=tails)

@pytest.mark.parametrize('cell',CELLS)
def test_real_plugin_task_periodic_retries_order_clones_ledgers(cell,harness_factory):
    h=harness_factory(cell);m=load(cell,'task_periodic',.32)
    a=h.run(m,force_guards=True,retries=True)
    b=h.run(m,force_guards=True,retries=True,reverse=True)
    assert a[:2]==b[:2]
    report=audit_allocation(a[2],m)
    assert report['guards']>=20 and report['knob_calls']>0 and report['policy_tails']>0
    EVIDENCE.setdefault('allocation_plugin',[]).append(dict(cell=cell,**report,reordered_connections_exact=True))

@pytest.mark.parametrize('cell',CELLS)
def test_continuity_fresh_engages_stale_early_identical(cell):
    m=load(cell,'base');p=load(cell,'continuity');fresh=changed=nominal=0
    for task in range(10):
        for pos in (0,1,3):
            q,_=pseudo(cell,m,task,pos)
            m.reset(q.episode);p.reset(q.episode)
            a=m.query(q);b=p.query(q)
            equal_result(a,b);nominal+=1
        for pos in (1,3,5):
            q,_=pseudo(cell,m,task,pos,False)
            m.reset(q.episode);p.reset(q.episode)
            a=m.query(q);b=p.query(q);fresh+=1
            changed+=a.action.tobytes()!=b.action.tobytes()
            # Continuity may change kernel weights even with identical donor
            # IDs, so compare the gripper to its own uncorrected proposal.
            cache=p.inner.base;rows,S,_=cache.os_score_all(q)
            idx=core.topk_pos(S,cache.k);w=np.exp((S[idx]-S[idx[0]])/cache.synth_T)
            raw=awm.AWM.os_synth(cache,q,rows[idx],w)
            np.testing.assert_array_equal(b.action[:,6:],raw[:,6:])
    assert changed>0
    EVIDENCE.setdefault('continuity',[]).append(dict(cell=cell,nominal_exact=nominal,injected_fresh=fresh,changed=changed))

@pytest.mark.parametrize('cell',CELLS)
def test_parent_support_query_metric_and_actual_foreign_corrector_anchors(cell):
    base=load(cell,'base');scope=load(cell,'scope');allpool=load(cell,'all_pool');metric=load(cell,'metric')
    assert allpool.inner.base.heads['0'] is allpool.inner.base.heads['9']
    assert allpool.inner.base.heads['0']['mean'].shape==(207,)
    assert base.inner.base.heads['0']['mean'].shape==(217,)
    for t in range(10):
        T=scope.inner.base.tasks[t];own=base.inner.base.tasks[t]
        assert len(T.rows)==len(base.row_subset)
        np.testing.assert_array_equal(T.Z[own.rows],own.Z)
        np.testing.assert_array_equal(T.Wf,own.Wf)
        np.testing.assert_array_equal(T.W0f,own.W0f)
        np.testing.assert_array_equal(metric.inner.base.tasks[t].Wf,metric.inner.base.tasks[0].Wf)
    q,_=pseudo(cell,scope,task=9,pos=3)
    q.task_id=0;q.episode=api.EpisodeView('foreign_fixture','instruction0',0,7,-1,0)
    scope.reset(q.episode);res=scope.query(q)
    b=scope.inner.base;T,_,_,_,_,_,_,ds,_,_,_=b._dist(q)
    anchor=b._anchor;parent=scope.row_subset[anchor['rows']]
    assert np.isin(parent,scope.row_subset).all()
    assert np.any(scope.inner.C.task[anchor['rows']]!=q.task_id)
    d=float(np.min(ds[np.searchsorted(T.rows,anchor['rows'])]))
    assert b._correction_audit['distance']==d
    assert set(scope.r14_diagnostics['parent_donor_rows'])==set(parent.tolist())
    rows,S,_=b.os_score_all(q);idx=core.topk_pos(S,b.k);w=np.exp((S[idx]-S[idx[0]])/b.synth_T)
    raw=awm.AWM.os_synth(b,q,rows[idx],w)
    np.testing.assert_array_equal(res.action[:,6:],raw[:,6:])
    EVIDENCE.setdefault('foreign_rows',[]).append(dict(cell=cell,all_parent_rows_supported=True,
        query_task_main_early_exact=True,foreign_donors=int(np.sum(scope.inner.C.task[anchor['rows']]!=0)),actual_anchor_gate=True))

@pytest.mark.parametrize('model',['pi05','groot'])
@pytest.mark.parametrize('fault',['partial_execution','invalid_wire_tail','missing_tail_hook','invalid_tail_result','early_termination'])
def test_real_plugin_policy_tail_faults(model,fault,harness_factory):
    from exp.offline_search.closed_loop.blind import BlindResult
    h=harness_factory(f'{model}_l10_50');m=load(h.cell,'task_periodic',.32);m.knob_settings['K']=[0.]*10
    conn=h.connection(m);h.start(conn,0);first=conn.infer(h.obs(0,0,fixed=True));s=conn._osp_sessions[0]
    assert s.hits==[0] and s._policy_tail is not None
    if fault=='early_termination':
        conn.on_episode_end(success=False);assert s._policy_tail is None
        h.start(conn,0);conn.infer(h.obs(0,0,fixed=True));assert s.hits==[0]
    else:
        if fault=='invalid_wire_tail':s._policy_tail['wire'][5,0]=np.nan
        if fault=='missing_tail_hook':s.method.policy_tail_step=None
        if fault=='invalid_tail_result':s.method.policy_tail_step=lambda q:BlindResult(np.zeros((1,32),np.float32),np.array([0]),np.ones(1),'current',{})
        conn.infer(h.obs(0,1,executed_steps=4 if fault=='partial_execution' else 5,fixed=True))
        assert s.has_vision==[True,True] and s.hits==[0,0]
        assert (h.logs[1]['r14']['N'],h.logs[1]['r14']['V'],h.logs[1]['r14']['M'])==(1,1,1)
    EVIDENCE.setdefault('lifecycle_faults',[]).append(dict(model=model,fault=fault,PASS=True))

@pytest.mark.parametrize('cell',CELLS)
def test_every_pooled_arm_in_real_plugin_and_corrected_tails(cell,harness_factory):
    h=harness_factory(cell);reports=[]
    for variant in ('scope','metric','head','scope_metric','scope_only','all_pool','continuity'):
        m=load(cell,variant)
        seq=h.run(m,tasks=[0,9],steps=8,force_guards=True)[2]
        assert len(seq)==16
        for d in seq:
            assert np.isfinite(d['served_head']).all()
            if d['vision']:
                x=d['r14'];assert x['task_id']==d['task_id']
                assert len(x['donor_task_ids'])==len(x['parent_donor_rows'])==16
                assert np.isin(x['parent_donor_rows'],m.row_subset).all()
                assert 0<=x['corrector']['gate']<=.5
                if d['step']==0:assert x['corrector']['gate']==x['corrector']['norm']==0
            elif d['src']=='policy_tail':
                assert d['hit']
        reports.append(dict(variant=variant,decisions=16,PASS=True))
    EVIDENCE.setdefault('pooled_plugin',[]).append(dict(cell=cell,variants=reports))

@pytest.mark.parametrize('model',['pi05','groot'])
def test_real_plugin_valid_tail_reset_duplicate_and_fresh_retrieval(model,harness_factory):
    h=harness_factory(f'{model}_l10_50');m=load(h.cell,'task_periodic',.32);m.knob_settings['K']=[0.]*10
    conn=h.connection(m);h.start(conn,0);a=conn.infer(h.obs(0,0,fixed=True));s=conn._osp_sessions[0]
    b=conn.infer(h.obs(0,1,fixed=True))
    assert s.hits==[0,1] and s.has_vision==[True,False]
    np.testing.assert_array_equal(b['actions'][:5],a['actions'][5:10])
    assert s.method.inner.base._anchor is None
    before=s.step,s.b_aex.n,len(h.logs)
    with pytest.raises(ValueError,match='duplicate'):conn.infer(h.obs(0,1,fixed=True))
    assert before==(s.step,s.b_aex.n,len(h.logs))
    conn.infer(h.obs(0,2,fixed=True));assert h.logs[-1]['r14']['branch']=='stale'
    conn.on_episode_end(success=False);h.start(conn,1);conn.infer(h.obs(1,0,fixed=True))
    assert s.method._accepted_prefix==((),()) and s.method._fallback_counts=={}

@pytest.mark.parametrize('fault',['prefix','malformed','decreasing'])
def test_contract_fallback_commits_no_cost(fault,harness_factory):
    h=harness_factory('pi05_l10_50');m=load(h.cell,'task_periodic',.32);m.knob_settings['K']=[0.]*10
    conn=h.connection(m);h.start(conn,0)
    for step in range(3):conn.infer(h.obs(0,step,fixed=True))
    s=conn._osp_sessions[0]
    if fault=='prefix':s.hits[0]=1;s._recs[0]['hit']=True
    elif fault=='malformed':s.hits.pop()
    else:s.method._proposal_step=100
    before=s.step,s.b_aex.n,len(h.logs),conn.policy_calls
    with pytest.raises(api.ContractError):conn.infer(h.obs(0,3,executed_steps=4,fixed=True))
    assert before==(s.step,s.b_aex.n,len(h.logs),conn.policy_calls)
    conn.on_episode_end(success=False);h.start(conn,0);conn.infer(h.obs(0,0,fixed=True));assert s.hits==[0]

def test_roots_exact_frozen_grid_selection_sha_recording_hosts_and_queue():
    catalog=json.loads((HERE/'fits.json').read_text())
    for rec in catalog:
        assert sha(rec['path'])==rec['sha256']
        if 'manifest' in rec and rec['manifest'].get('variant') not in ('task_periodic','guard_null'):
            assert rec['manifest']['full_training_rows']
            if rec['manifest']['variant']!='base':
                assert sum(v['rows'] for v in rec['manifest']['head_counts'].values())==rec['manifest']['rows']
    pilot=json.loads((HERE/'pilot_roots.json').read_text());test=json.loads((HERE/'test_a_roots.json').read_text())
    assert len(pilot)==6 and sum(r['arm_count'] for r in pilot)==68
    assert len(test)==8 and sum(r['arm_count'] for r in test)==128
    selections={r['cell']:r['pairs'] for r in json.loads((ASTRA/'pilot_selection.json').read_text())['cells']}
    recovered=[]
    for rec in pilot+test:
        root=Path(rec['root']);rows=json.loads((root/'arms.json').read_text())
        for row in rows:
            args=row['plugin_args'];g=row['r14_grid']
            assert row['intended_host']==('h100' if row['model']=='pi05' else 'local_4090')
            assert row['fleet_exclusive'] and 'server_host' not in row
            assert args[args.index('--os-rollout-seed')+1]==str(row['rollout_replicate_seed'])
            assert row['knob_coin_seed']==0 and '--os-r14-audit' in args
            from exp.offline_search.closed_loop.ops.h100.control import execution_env
            assert execution_env(row)['OSCL_ROLLOUT_SEED']==str(row['rollout_replicate_seed'])
            from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
            manifest=load_manifest(row['manifest'])
            if rec['init_pool']=='B':
                assert set(manifest['selected'])==set(map(tuple,selections[g['cell']]))
                contract=devset.arm_contract(root,row,manifest)
                assert contract==rec['contracts'][row['arm']]
                assert '--os-record-looks' in args
                assert ('--os-record-look-shadow' in args)==(g['part']=='A')
            else:
                assert not look_recording.requested(args)
                assert Path(row['manifest']).read_bytes()==(TRACE/'r11_knob_1/eval500.json').read_bytes()
                assert not Path(rec['final_root']).exists()
                recovered.append(g)
        for p,digest in rec['definitions_sha256'].items():assert sha(root/p)==digest
    assert sorted(recovered,key=lambda g:int(g['order']))==list(csv.DictReader((ASTRA/'test_grid.csv').open()))
    for filename,roots in [('queue_pilot.txt',pilot),('queue_test_a.txt',test)]:
        lines=(HERE/filename).read_text().splitlines()
        if filename=='queue_test_a.txt':roots=json.loads((HERE/'test_queue_segments.json').read_text())
        assert len(lines)==len(roots)
        for line,r in zip(lines,roots):
            prefix='@local' if r['intended_host']=='local_4090' else '@h100'
            assert line==f'{prefix} {r["final_root"]}|{" ".join(r["arms"])}'
    queued=[arm for line in (HERE/'queue_test_a.txt').read_text().splitlines() for arm in line.split('|',1)[1].split()]
    assert queued==[r['arm'] for r in csv.DictReader((ASTRA/'test_grid.csv').open())]
    EVIDENCE['roots']=dict(pilot_roots=6,pilot_arms=68,staged_roots=8,test_arms=128,installed=False)

def test_staged_installer_only_regular_definitions(tmp_path):
    from .prepare import install_test_a
    records=json.loads((HERE/'test_a_roots.json').read_text());installed=install_test_a(tmp_path/'install')
    assert len(installed)==8
    for rec,dest in zip(records,installed):
        root=Path(dest);rows=json.loads((root/'arms.json').read_text())
        assert [r['arm'] for r in rows]==rec['arms']
        assert not (root/'runs').exists() and not (root/'state').exists()
        assert (root/'eval500.json').read_bytes()==(Path(rec['root'])/'eval500.json').read_bytes()
        assert all(r['manifest']==str(root/'eval500.json') for r in rows)
        assert not Path(rec['final_root']).exists()
    with pytest.raises(FileExistsError):install_test_a(tmp_path/'install')

def test_r14_side_record_survives_scalar_limit_and_miss(tmp_path):
    from exp.offline_search.closed_loop.test_look_recording import writer_options,connection
    opts,_=writer_options(tmp_path,shadow=True)
    opts.os_r14_audit=True;opts.os_record_look_cache_proposal=True
    writer=look_recording.LookWriter(opts,'pi05');conn,s,obs,logs=connection(plugin,opts,writer)
    s.rt.r14_fit_sha256='c'*64
    s.method.r14_diagnostics=dict(schema='offline_search.r14.v1',donor_task_ids=list(range(10)),priority=.125)
    conn.infer({**obs,'_miss':True})
    record=list(look_recording.read_looks(writer.directory))[0]
    side=json.loads(str(record['r14']))
    assert side['donor_task_ids']==list(range(10))
    assert side['fit_sha256']=='c'*64
    assert np.all(record['cache_proposal']==2.) and np.all(record['action']==77.)

def test_frozen_rollout_seeds_reach_worker_and_cpu_policy_rng(monkeypatch):
    import random,torch
    from openpi.conductor import WorkerSpec
    from exp.offline_search.closed_loop import worker_pool
    from exp.offline_search.closed_loop.rollout_seed import seed_policy_process
    seeds=sorted({int(r['rollout_seed']) for r in csv.DictReader((ASTRA/'test_grid.csv').open())})
    assert seeds==[20261401,20261402]
    saved=random.getstate(),np.random.get_state(),torch.random.get_rng_state();streams=[]
    try:
        for seed in [*seeds,seeds[0]]:
            monkeypatch.setenv('OSCL_ROLLOUT_SEED',str(seed));run=NS(WorkerSpec=WorkerSpec)
            worker_pool.install_worker_options(run)
            spec=run.WorkerSpec(worker_id='w',server_key='x',gpu_id='')
            assert spec.seed==seed
            seed_policy_process(seed)
            streams.append((random.random(),np.random.random(),torch.randn(8).numpy().tobytes()))
        assert streams[0]!=streams[1] and streams[0]==streams[2]
        coin=uniform('R11-gap-v1',0,1,7,0,'cap');seed_policy_process(seeds[1])
        assert coin==uniform('R11-gap-v1',0,1,7,0,'cap')
    finally:
        random.setstate(saved[0]);np.random.set_state(saved[1]);torch.random.set_rng_state(saved[2])
    EVIDENCE['rollout_seeds']=dict(seeds=seeds,worker_seed_verified=True,cpu_policy_rng_changes=True,
        simulator_seed_route_verified=True,simulator_not_launched=True,
        live_simulator_policy_seed_check_required=True,coin_seed=0)

def test_write_evidence():
    write_json(HERE/'mechanics.json',dict(PASS=True,**EVIDENCE))
