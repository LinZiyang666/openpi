"""SPEC mechanics, frozen fits/roots, opt-in recording and replicate seed gates."""
import copy
import csv
import json
import pickle
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from exp.offline_search.closed_loop import plugin, look_recording, devset, worker_pool
from exp.offline_search.closed_loop.ops.h100 import control
from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe
from exp.offline_search.rounds.r11.knob.recipe import R11Knob, REASONS as R11_REASONS, uniform
from exp.offline_search.rounds.r11.knob.tests import BaseStub, query
from exp.offline_search.rounds.r13.knob.recipe import R13Knob, REASONS
from .prepare import HERE, ASTRA, R13, TRACE, key, source_artifact
from .mechanics import Harness, EVIDENCE, load, audit_feedback, equal_result

CELLS = [f'{m}_{s}_{n}' for m in ('pi05','groot') for s,nn in (('l10',(50,200,500)),('spatial',(50,))) for n in nn]


def toy(method='distance_feedback', *, model='pi05', target=.32, bias=.5):
    m=R13Knob(method=method,target=target)
    m.inner=BaseStub();m.inner.base=NS(_dist=lambda q:(None,)*7+(np.array([1.,2.],np.float64),))
    m.row_subset=np.array([42])
    cfg=dict(lo=0.,hi=2.,width=1.,bias=bias) if method=='distance_feedback' else dict(threshold=1.)
    m.configure(cfg,model);m.reset(NS(uid='ep',init=7))
    return m


def test_reference_arithmetic_all_frozen_configs_irregular_committed_ledgers():
    from exp.offline_search.rounds.r13.astra.controller import DistanceFeedback, Ledger
    rng=np.random.default_rng(20261004);proposals=retries=0
    for c in json.loads((ASTRA/'configs.json').read_text()):
        if c['method']!='distance_feedback':continue
        m=toy(model=c['model'],target=c['target']);m.configure({k:c[k] for k in ('lo','hi','width','bias')},c['model'])
        ref=DistanceFeedback(**{k:c[k] for k in ('model','target','lo','hi','width','bias')})
        hv=[];hits=[]
        for j in range(100):
            step=len(hv);d=float(np.exp(rng.normal(2,1)));guard=bool(rng.random()<.2)
            m.inner.guard=guard;m.inner.base._dist=lambda q,d=d:(None,)*7+(np.array([d,d+100]),)
            q=query(step,hv,hits);a=m.query(q);x=a.extras
            b=ref.propose(step=step,distance=d,guard=guard,ledger=Ledger(step,sum(hv),sum(v and not h for v,h in zip(hv,hits))))
            assert b['miss']==bool(x['os_force_miss']) and b['knob']==bool(x['os_r13_knob_call'])
            assert b['threshold']==x['os_r13_theta'] and b['debt']==x['os_r13_D'] and b['score']==x['os_r13_u']
            before=m.inner.n;equal_result(a,m.query(q));assert m.inner.n==before
            hv.append(True);hits.append(int(not b['miss']))
            if rng.random()<.65:hv.append(False);hits.append(1)
            proposals+=1;retries+=1
    assert proposals==retries==3200
    EVIDENCE['reference_arithmetic']=dict(proposals=proposals,retries=retries,configs=32,exact_float64=True)


@pytest.mark.parametrize('distance',[-1.,np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('guard',[False,True])
def test_nonfinite_conservative_and_guard_reason_preserved(distance,guard):
    m=toy();m.inner.guard=guard;m.inner.base._dist=lambda q:(None,)*7+(np.array([distance,distance]),)
    a=m.query(query());assert a.extras['os_force_miss']==1
    assert a.extras['os_r13_nonfinite']==1 and a.extras['os_r13_knob_call']==int(not guard)
    assert a.extras['os_reason']==(4 if guard else 69)
    if guard:assert a.extras['os_flags']==8


def test_strict_ties_unclipped_theta_and_static_raw_score():
    m=toy(bias=float(np.log1p(1.)/2))
    a=m.query(query());assert a.extras['os_force_miss']==0  # exact tie
    for bias,call in [(-1.,1),(1.,0),(2.,0)]:
        m=toy(bias=bias);a=m.query(query());assert a.extras['os_force_miss']==call
        assert a.extras['os_r13_theta']==bias
    m=toy('distance_static_b');a=m.query(query());assert a.extras['os_force_miss']==0
    m.reset(NS(uid='e',init=7));m.inner.base._dist=lambda q:(None,)*7+(np.array([1.0001,100.]),)
    a=m.query(query());assert a.extras['os_reason']==70 and a.extras['os_r13_raw_distance']==1.0001


def test_contract_errors_reset_clone_isolation_and_no_future_tail_credit():
    m=toy();m.query(query())
    m.query(query(3,[True,True,False],[0,0,1]))
    assert m._ledger_checkpoint==(3,2,2)  # extra fallback look and two actual calls
    with pytest.raises(api.ContractError):m.query(query(2,[True,False],[0,1]))
    with pytest.raises(api.ContractError):m.query(query(4,[True,False,False,False],[0,1,1,1]))
    with pytest.raises(api.ContractError):m.query(query(4,[True,True,False,False],[1,0,1,1]))
    with pytest.raises(api.ContractError):toy().query(query(2,[True],[0]))
    with pytest.raises(api.ContractError):toy().query(query(1,[False],[0]))
    b,_=plugin.clone_method(m,strict=True);b.reset(NS(uid='new',init=7))
    assert m._ledger_checkpoint==(3,2,2) and b._ledger_checkpoint==(0,0,0)
    assert b._proposal is None and b._proposal_step==-1
    b.query(query());assert m.inner.n!=b.inner.n


def test_reason_collision_and_audit_priority():
    assert set(REASONS.values()).isdisjoint(R11_REASONS.values())
    x=toy().query(query()).extras
    assert list(x)[:15]==['os_r13_raw_distance','os_r13_u','os_r13_theta','os_r13_D',
                         'os_r13_target','os_r13_width','os_r13_bias','os_r13_N','os_r13_V','os_r13_M',
                         'os_r13_guard','os_r13_knob_call','os_r13_nonfinite','os_r13_saturated_low','os_r13_saturated_high']


@pytest.fixture
def harness_factory(tmp_path,monkeypatch):
    monkeypatch.setattr(plugin,'_git_head',lambda:None)
    harnesses=[]
    def make(cell,pure_c5=False):
        h=Harness(cell,tmp_path/f'plugin_{cell}_{len(harnesses)}',pure_c5=pure_c5);harnesses.append(h);return h
    yield make
    for h in harnesses:h.rt.flush_all()


@pytest.mark.parametrize('cell',CELLS)
def test_real_plugin_off_R13_R11_R10_byte_identical_recorded_B(cell,harness_factory):
    h=harness_factory(cell)
    original=load(cell,'off');r13=copy.deepcopy(original);r13.__class__=R13Knob
    r13._ledger_checkpoint=(0,0,0);r13._accepted_prefix=((),())
    source=R13.parent/f'r10/recipe/artifacts/r10_recipe_{cell}.pkl'
    r10=pickle.loads(source.read_bytes())['method']
    results=[h.run(m,force_guards=True) for m in (original,r13,r10)]
    assert results[0][:2]==results[1][:2]==results[2][:2]
    EVIDENCE.setdefault('off_equivalence',[]).append(dict(cell=cell,decisions=120,differing_actions=0,differing_decisions=0))


@pytest.mark.parametrize('cell',CELLS)
def test_real_plugin_reference_accepted_ledger_retries_order_and_clones(cell,harness_factory):
    h=harness_factory(cell);m=load(cell,'distance_feedback',.45)
    a=h.run(m,force_guards=True,retries=True)
    b=h.run(m,force_guards=True,retries=True,reverse=True)
    assert a[:2]==b[:2]
    report=audit_feedback(a[2],h.model)
    assert report['guards']>=20 and report['knob_calls']>0 and report['policy_tails']>0
    EVIDENCE.setdefault('feedback_plugin',[]).append(dict(cell=cell,**report,reordered_independent_connections=True))


@pytest.mark.parametrize('model',['pi05','groot'])
@pytest.mark.parametrize('fault',['partial_execution','invalid_wire_tail','missing_tail_hook','invalid_tail_result','early_termination'])
def test_real_plugin_policy_tail_fallback_and_early_termination(model,fault,harness_factory):
    from exp.offline_search.closed_loop.blind import BlindResult
    h=harness_factory(f'{model}_l10_50');m=load(h.cell,'distance_feedback',.45)
    m.knob_settings['bias']=-16.  # mechanics all-call, never stored as an experimental setting
    conn=h.connection(m);h.start(conn,0);conn.infer(h.obs(0,0,fixed=True));s=conn._osp_sessions[0]
    assert s.hits==[0] and s._policy_tail is not None and s.method.inner._policy_gate_anchor is not None
    if fault=='early_termination':
        conn.on_episode_end(success=False);assert s._policy_tail is None
        assert s.method._ledger_checkpoint==(0,0,0)
        h.start(conn,0);conn.infer(h.obs(0,0,fixed=True));assert s.hits==[0]
    else:
        if fault=='invalid_wire_tail':s._policy_tail['wire'][5,0]=np.nan
        if fault=='missing_tail_hook':s.method.policy_tail_step=None
        if fault=='invalid_tail_result':
            s.method.policy_tail_step=lambda q:BlindResult(np.zeros((1,32),np.float32),np.array([0]),np.ones(1),'current',{})
        conn.infer(h.obs(0,1,executed_steps=4 if fault=='partial_execution' else 5,fixed=True))
        assert s.has_vision==[True,True] and s.hits==[0,0]
        conn.infer(h.obs(0,2,fixed=True))
        report=audit_feedback(h.logs,model,dict(m.knob_settings,model=model,target=.45));assert report['reference_proposals']>=2
        assert h.logs[1]['extras']['os_r13_N']==h.logs[1]['extras']['os_r13_V']==h.logs[1]['extras']['os_r13_M']==1
    EVIDENCE.setdefault('lifecycle_faults',[]).append(dict(model=model,fault=fault,pass_gate=True))


@pytest.mark.parametrize('model',['pi05','groot'])
def test_real_plugin_tail_is_policy_then_fresh_stale_retrieval_and_duplicate_refused(model,harness_factory):
    h=harness_factory(f'{model}_l10_50');m=load(h.cell,'pure_c10');conn=h.connection(m);h.start(conn,0)
    first=conn.infer(h.obs(0,0,fixed=True));s=conn._osp_sessions[0]
    second=conn.infer(h.obs(0,1,fixed=True))
    assert s.hits==[0,1] and s.has_vision==[True,False]
    assert np.array_equal(second['actions'][:5],first['actions'][5:10])
    assert s.method.inner.base._anchor is None  # discarded cache tail never served
    before=(s.step,s.b_aex.n,len(s.has_vision),conn.orch._step_counter)
    with pytest.raises(ValueError,match='duplicate'):conn.infer(h.obs(0,1,fixed=True))
    assert before==(s.step,s.b_aex.n,len(s.has_vision),conn.orch._step_counter)
    conn.infer(h.obs(0,2,fixed=True))
    assert s.has_vision==[True,False,True] and s.method.inner._policy_gate_anchor['step']==2


@pytest.mark.parametrize('cell',CELLS)
def test_periodic_unchanged_and_pure_c10_K0_endpoint(cell,harness_factory):
    h=harness_factory(cell)
    for target in (.25,.32,.40,.45):
        m=load(cell,'periodic',target);r13=copy.deepcopy(m);r13.__class__=R13Knob
        r13._ledger_checkpoint=(0,0,0);r13._accepted_prefix=((),())
        a=h.run(m,tasks=range(2),steps=8);b=h.run(r13,tasks=range(2),steps=8)
        assert a[:2]==b[:2]
    pure=load(cell,'pure_c10');ref=copy.deepcopy(pure);ref.__class__=R11Knob
    assert pure.knob_settings['setting']==0
    assert h.run(pure,tasks=range(2),steps=8)[:2]==h.run(ref,tasks=range(2),steps=8)[:2]
    EVIDENCE.setdefault('periodic_equivalence',[]).append(dict(cell=cell,doses=4,decisions=64,pure_c10_K0_decisions=16))


@pytest.mark.parametrize('model',['pi05','groot'])
def test_pure_c10_shared_across_sizes_and_pure_c5_every_five(model,harness_factory):
    h=harness_factory(f'{model}_l10_50')
    results=[h.run(load(f'{model}_l10_{size}','pure_c10'),steps=8,tasks=range(10)) for size in (50,200,500)]
    # Donor diagnostics necessarily vary by library. Compare executed actions,
    # actual call/vision lifecycle and source: these alone define a pure control.
    assert results[0][0]==results[1][0]==results[2][0]
    lifecycle=lambda logs:[(r['task_id'],r['step'],r['hit'],r['vision'],r['src'],r.get('look_reason')) for r in logs]
    assert lifecycle(results[0][2])==lifecycle(results[1][2])==lifecycle(results[2][2])
    h5=harness_factory(f'{model}_l10_50',pure_c5=True)
    a=h5.run(load(f'{model}_l10_50','pure_c5'),steps=8,tasks=range(2))
    assert all(d['vision'] and not d['hit'] and d['src']=='policy' for d in a[2])
    EVIDENCE.setdefault('pure_controls',[]).append(dict(model=model,shared_C10_sizes=[50,200,500],C10_decisions=80,C5_decisions=16,C5_IR=1.))


@pytest.mark.parametrize('cell',[f'{m}_{s}_50' for m in ('pi05','groot') for s in ('l10','spatial')])
@pytest.mark.parametrize('target',[.32,.40])
def test_real_plugin_static_frozen_threshold_guard_or_strict_raw_distance(cell,target,harness_factory):
    h=harness_factory(cell);m=load(cell,'distance_static_b',target)
    seq=h.run(m,tasks=range(2),steps=12,force_guards=True)[2]
    V=M=0;uid=None
    for d in sorted(seq,key=lambda d:(d['uid'],d['step'])):
        if d['uid']!=uid:V=M=0;uid=d['uid']
        if d['vision']:
            x=d['extras'];assert (x['os_r13_N'],x['os_r13_V'],x['os_r13_M'])==(d['step'],V,M)
            guard=bool(x['os_r13_guard']);trigger=x['os_r13_raw_distance']>m.knob_settings['threshold']
            assert (not d['hit'])==bool(guard or trigger)
            assert x['os_r13_knob_call']==int(trigger and not guard)
            if guard:assert x['os_reason']==4 and x['os_flags']==8
            elif trigger:assert x['os_reason']==70
            V+=1;M+=not d['hit']
    EVIDENCE.setdefault('static_plugin',[]).append(dict(cell=cell,target=target,decisions=24,threshold_exact=True))


@pytest.mark.parametrize('fault',['rewritten_prefix','nonmonotone_ledger','accepted_buffer_mismatch'])
def test_real_plugin_history_contract_error_uses_ordinary_error_path_without_commit(fault,harness_factory):
    h=harness_factory('pi05_l10_50');conn=h.connection(load(h.cell,'distance_feedback',.45))
    h.start(conn,0);conn.infer(h.obs(0,0,fixed=True));conn.infer(h.obs(0,1,fixed=True))
    s=conn._osp_sessions[0];m=s.method
    if fault=='rewritten_prefix':
        # Establish a checkpoint that includes accepted slots, then mutate it.
        conn.infer(h.obs(0,2,fixed=True));s.hits[0]=1-s.hits[0]
        s._recs[0]['hit']=bool(s.hits[0])
        step=3
    elif fault=='nonmonotone_ledger':
        m._ledger_checkpoint=(2,2,2);step=2
    else:
        s.hits.pop();step=2
    before=(s.step,s.b_aex.n,len(s.hits),len(h.logs),conn.policy_calls)
    with pytest.raises(api.ContractError):conn.infer(h.obs(0,step,executed_steps=4,fixed=True))
    assert before==(s.step,s.b_aex.n,len(s.hits),len(h.logs),conn.policy_calls)
    assert s._policy_tail is None
    # The ordinary error path returns no action and charges no committed cost.
    # A lifecycle reset starts a clean episode rather than fabricating history.
    conn.on_episode_end(success=False);h.start(conn,0);conn.infer(h.obs(0,0,fixed=True))
    assert s.method._ledger_checkpoint==(0,0,0)
    EVIDENCE.setdefault('history_contract_errors',[]).append(dict(fault=fault,no_action_or_cost_committed=True,reset_recovers=True))


def test_seed_reaches_WorkerSpec_and_changes_policy_rng_without_knob_coin(monkeypatch):
    import random,torch
    from openpi.conductor import WorkerSpec
    from exp.offline_search.closed_loop.rollout_seed import seed_policy_process
    streams=[];states=(random.getstate(),np.random.get_state(),torch.random.get_rng_state())
    try:
        for seed in (2026100400,2026100401,2026100400):
            monkeypatch.setenv('OSCL_ROLLOUT_SEED',str(seed));run=NS(WorkerSpec=WorkerSpec)
            worker_pool.install_worker_options(run)
            spec=run.WorkerSpec(worker_id='w',server_key='x',gpu_id='')
            assert spec.seed==seed
            record=seed_policy_process(seed);assert record['seed']==seed
            streams.append((random.random(),np.random.random(),torch.randn(8).numpy().tobytes()))
        assert streams[0]!=streams[1] and streams[0]==streams[2]
        a=uniform('R11-gap-v1',0,1,7,0,'cap');seed_policy_process(2026100402)
        assert a==uniform('R11-gap-v1',0,1,7,0,'cap')
    finally:
        random.setstate(states[0]);np.random.set_state(states[1]);torch.random.set_rng_state(states[2])
    EVIDENCE['replicate_seed']=dict(worker_spec_seed_verified=True,policy_rng_changes=True,repeatable_seed=True,
                                  knob_coin_seed=0,simulator_render_not_launched=True,
                                  caveat='process RNG remains warmup and concurrent request order dependent')


def test_recording_opt_in_raw_geometry_cache_proposal_survives_miss(tmp_path):
    from exp.offline_search.closed_loop.test_look_recording import writer_options,connection
    opts,_=writer_options(tmp_path,shadow=True)
    opts.os_record_look_raw_distance=opts.os_record_look_cache_proposal=True
    writer=look_recording.LookWriter(opts,'pi05');conn,s,obs,_=connection(plugin,opts,writer)
    s.method.inner=NS(base=NS(_dist=lambda q:(None,)*7+(np.array([7.,8.]),)))
    conn.infer({**obs,'_miss':True})
    row=list(look_recording.read_looks(writer.directory))[0]
    assert row['raw_distance'].dtype==np.float64 and row['raw_distance']==7.
    assert row['distances'][0]==1. and np.all(row['cache_proposal']==2.)
    assert np.all(row['action']==77.) and row['cache_proposal'].shape==(10,32)
    assert np.all(row['shadow_policy']==77.)
    EVIDENCE['recording']=dict(raw_distance_float64=True,miss_cache_before_replacement=True,default_off=True)


@pytest.mark.parametrize('model',['pi05','groot'])
def test_real_plugin_shadow_recording_corrected_proposal_on_guard_and_knob_misses(model,harness_factory,tmp_path):
    from .prepare import write_json
    h=harness_factory(f'{model}_l10_50')
    root=TRACE/f'r13_pilot_{h.cell}'
    row=next(r for r in json.loads((root/'arms.json').read_text())
             if r['r13_method']=='distance_feedback' and r['target_ir']==.32 and r['r13_grid']['shadow'])
    manifest=load_manifest_safe(row['manifest']);contract=devset.arm_contract(root,row,manifest)
    contract.update(schema=look_recording.SCHEMA,cell=row['cell'],method=row['method'],
                    recording_arm_dir=str(tmp_path),pairs=list(map(list,manifest['selected'])))
    path=tmp_path/'look_contract.json';write_json(path,contract)
    opts=h.rt.opts
    opts.os_log_dir=str(tmp_path/'recorded')
    opts.os_look_contract=str(path);opts.os_look_contract_sha256=devset.sha(path)
    opts.os_record_looks=opts.os_record_look_shadow=True
    opts.os_record_look_raw_distance=opts.os_record_look_cache_proposal=True
    writer=look_recording.LookWriter(opts,model);h.rt.looks=writer
    conn=h.connection(load(h.cell,'distance_feedback',.32),force_guards=True)
    task,init=next(iter(manifest['selected']))
    conn.on_episode_start(task=h.lib.meta['tasks'][str(task)],episode_id=init,
                          extra_metadata=dict(task_uid=f'{row["arm"]}:eval:{task}:{init}',task_id=task,
                                              orig_init_state_idx=init,run_id=look_recording.token(contract)))
    proposals={}
    for step in range(16):
        conn.infer(h.obs(task,step,init=init));s=conn._osp_sessions[0]
        if s.has_vision[-1]:
            proposals[step]=s.method._proposal.action[:10].copy()
    records=list(look_recording.read_looks(writer.directory))
    assert len(records)==len(proposals)
    guards=knobs=0
    for r in records:
        step=int(r['step']);ex=r['guard_state']['extras']
        assert r['cache_proposal'].tobytes()==proposals[step].tobytes()
        assert float(r['raw_distance'])==ex['os_r13_raw_distance']
        assert r['cache_proposal'].shape==(10,32)
        if not r['hit']:
            assert np.array_equal(r['shadow_policy'],r['action'])
            guards+=bool(ex['os_r13_guard']);knobs+=bool(ex['os_r13_knob_call'])
    assert guards>0
    EVIDENCE.setdefault('real_recording',[]).append(dict(model=model,looks=len(records),guards=guards,knob_misses=knobs,
                                                      corrected_proposal_bytes_exact=True,raw_distance_exact=True))


def test_frozen_roots_selection_sha_exact_settings_host_queue_order_and_seeds():
    frozen=json.loads((HERE/'protected_before.json').read_text())
    for path,digest in frozen.items():
        if '/astra/' in path:assert devset.sha(path)==digest
    catalog=json.loads((HERE/'fits.json').read_text())
    for c in json.loads((ASTRA/'configs.json').read_text()):
        rec=next(r for r in catalog if r['config_key']==c['config_key'])
        assert devset.sha(rec['path'])==rec['sha256']
        m=pickle.loads(Path(rec['path']).read_bytes())['method']
        fields=('lo','hi','width','bias') if c['method']=='distance_feedback' else ('setting',)
        for field in fields:assert m.knob_settings[field]==c[field]
    pilot=json.loads((HERE/'pilot_roots.json').read_text());tests=json.loads((HERE/'test_a_roots.json').read_text())
    assert len(pilot)==6 and sum(r['arm_count'] for r in pilot)==64
    assert len(tests)==16 and sum(r['arm_count'] for r in tests)==256
    used=json.loads((ASTRA/'used_b_identities.json').read_text());sel=json.loads((ASTRA/'validation_selection.json').read_text())
    for r in pilot+tests:
        root=Path(r['root']);rows=json.loads((root/'arms.json').read_text())
        for row in rows:
            args=row['plugin_args'];seed=row['rollout_replicate_seed']
            assert args[args.index('--os-rollout-seed')+1]==str(seed)
            assert control.execution_env(row)['OSCL_ROLLOUT_SEED']==str(seed)
            assert row['knob_coin_seed']==0 and 'server_host' not in row
            assert row['intended_host']==('local_4090' if row['model']=='groot' else 'h100')
            manifest=load_manifest_safe(row['manifest'])
            if r['init_pool']=='B':
                assert '--os-record-look-raw-distance' in args
                expected=sel['pairs_by_cell'][row['r13_grid']['cell']]
                assert set(manifest['selected'])==set(map(tuple,expected))
                old={tuple(p) for k,pp in used.items() if k.endswith('_'+row['suite_short']) for p in pp}
                assert not old&set(manifest['selected'])
                contract=devset.arm_contract(root,row,manifest)
                assert contract==json.loads((root/'dev_contracts.json').read_text())[row['arm']]
                assert ('--os-record-look-cache-proposal' in args)==row['r13_grid']['shadow']
            else:
                assert not look_recording.requested(args)
                assert Path(row['manifest']).read_bytes()==(TRACE/'r11_knob_1/eval500.json').read_bytes()
                assert not Path(r['final_root']).exists()
    grid=list(csv.DictReader((ASTRA/'test_grid.csv').open()))
    flattened=[row['r13_grid'] for r in tests for row in json.loads((Path(r['root'])/'arms.json').read_text())]
    assert flattened==grid
    for filename,roots in [('queue_pilot.txt',pilot),('queue_test_a.txt',tests)]:
        lines=(HERE/filename).read_text().splitlines();assert len(lines)==len(roots)
        for line,r in zip(lines,roots,strict=True):
            prefix='@local' if r['intended_host']=='local_4090' else '@h100'
            assert line==f'{prefix} {r["final_root"]}|{" ".join(r["arms"])}'
    EVIDENCE['roots']=dict(pilot_roots=6,pilot_arms=64,staged_A_roots=16,staged_A_arms=256,
                          zero_unknown_missing_identities=True,fresh_B_disjoint=True,frozen_A_order_exact=True)


def load_manifest_safe(path):
    from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
    return load_manifest(path)


def test_staged_installer_only_copies_definitions_and_preserves_frozen_order(tmp_path):
    from .prepare import install_test_a
    records=json.loads((HERE/'test_a_roots.json').read_text())
    installed=install_test_a(tmp_path/'install')
    assert len(installed)==16
    for rec,target in zip(records,installed,strict=True):
        source=Path(rec['root']);root=Path(target)
        rows=json.loads((root/'arms.json').read_text())
        assert [r['arm'] for r in rows]==rec['arms']
        assert (root/'eval500.json').read_bytes()==(source/'eval500.json').read_bytes()
        assert not (root/'runs').exists() and not (root/'state').exists()
        for row in rows:
            assert row['manifest']==str(root/'eval500.json')
            name=Path(row['yaml']).name
            assert Path(row['yaml']).read_bytes()==(source/'config'/name).read_bytes()
        assert not Path(rec['final_root']).exists()
    with pytest.raises(FileExistsError):install_test_a(tmp_path/'install')
    EVIDENCE['installer']=dict(staged_roots=16,order_and_configs_unchanged=True,final_trace_roots_not_installed=True)


def test_write_mechanics_evidence():
    # Explicitly listed last by the verification command; all preceding checks
    # also have independent pytest results, so this is not the pass criterion.
    from .prepare import write_json
    write_json(HERE/'mechanics.json',dict(PASS=True,**EVIDENCE))
