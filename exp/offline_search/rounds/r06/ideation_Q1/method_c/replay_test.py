"""Real plugin/orchestrator, CPU fake policy, recorded observations/actions.

Dry-run calibrations enter only through a scoped in-memory test patch. No test
fit pickle is written. No server, simulator, network or model is started.

SELECTION §7b checks: every fresh anchor's logged p/call/cooldown/extra LOOK is
re-derived from the logged stall state and nominal p with the shared rule, and
the realized trajectory of each episode must be one path of budget.py's
cadence DAG built from the same recorded keys (same stall states, cooldowns,
LOOK eligibility), i.e. modeled and deployed behaviour agree anchor by anchor.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from .common import HERE,OUT,STORE,sources,load_base,write_json,output_path
from .methods import CalibratedRescue
from .budget import replay_cadence
from .stall_bridge import STATE, call_probability, scheduled_look

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True)
    ap.add_argument('--case',choices=['A','floor','uniform','R','stall','R45','stall30'],required=True)
    ap.add_argument('--reverse',action='store_true');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--dryrun',default='dryrun_v2b',help='dry-run calibration directory name under each cell')
    a=ap.parse_args();out=output_path(a.out);out.mkdir(parents=True,exist_ok=False)
    from exp.offline_search.closed_loop import plugin,selftest
    from exp.offline_search.closed_loop.blind import policy_tail_chunk
    from exp.offline_search.harness import store
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator
    plugin._git_head=lambda:None  # no indirect git call
    source=sources()[a.cell];model,suite,size=a.cell.split('_');cell=f'{model}_{suite}_cache'
    calpath=OUT/a.cell/a.dryrun/'calibration.json';cal=json.loads(calpath.read_text())
    assert cal['status']=='DRYRUN_TEST_INITS'
    if a.case=='A':
        spec=source['method'];kwargs=source['kwargs'];cls,_=plugin.load_method_class(spec)
        def fit(method,lib,ctx):method.__dict__.update(load_base(source)[0].__dict__)
    else:
        spec='exp.offline_search.rounds.r06.ideation_Q1.method_c.methods:CalibratedRescue';cls=CalibratedRescue
        rho=cal['solutions']['no_stall']['R']['0.18']['floor'] if a.case=='floor' else .45 if a.case=='R45' else .30 if a.case=='stall30' else .18
        kwargs=dict(rho=rho,placement='uniform' if a.case=='uniform' else 'R',calibration_path=str(calpath),
            stall_model_path=str(Path('/tmp/q3_stall_fits')/a.cell.replace('spatial','sp')) if a.case in ('stall','stall30') else None,
            random_seed=26092903,randomization_key='R6-C-v2/'+a.cell,cooldown_scope='stall')
        def fit(method,lib,ctx):method._load(ctx.cell,allow_dryrun=True)
    argv=['--os-method',spec,'--os-kwargs',json.dumps(kwargs),'--os-cell',cell,'--os-root',str(STORE),
        '--os-log-dir',str(out),'--os-tag','cpu_test','--os-no-shadow-native','--os-blind',
        '--os-policy-tail','--os-policy-tail-blocks','1','--os-judge','guard_only']
    opts,rest=plugin.parse_cli(argv);assert not rest
    with patch.object(cls,'fit',fit):rt=plugin.install(opts,model)
    assert not rt.shadow_native and not rt.randomized and rt.gpu is None
    cfg=cc.load_cache_config(str(HERE.parents[5]/f'exp/trace_dual/config/tr_{model}_{"sp" if suite=="spatial" else "l10"}_cache.yaml'))
    shared=cc.build_shared_storage(cfg);qc=store.QueryCell(STORE,cell)
    class Fake(selftest.FakePolicy):
        calls=0
        def stage1(self,obs):Fake.calls+=1
        def infer(self,obs):self.stage1(obs);return super().infer(obs)
        def _osp_prepare_blind(self,obs):return np.array(qc.rs[int(obs['_row'])],copy=True),None
        def _osp_blind_output(self,action,state):return {'actions':action.copy()}
    def factory(_base,bundle_id='default'):
        comps=cc.build_per_connection_components(cfg,shared,quiet=True);kb=selftest.FakeKB(qc)
        for s in plugin._TLS.new_sessions:s.kb=kb
        orch=CacheOrchestrator(storage=comps['storage'],key_builder=kb,gates=comps['gates'],
            judges=comps['judges'],search_strategies=comps['search_strategies'],timer=comps['timer'],
            write_policy=comps.get('write_policy'),offline_writers=comps.get('offline_writers',()),library_stats=comps.get('library_stats'))
        return Fake(orch,kb,np.asarray(qc.a_inf,np.float32))
    conn=plugin._wrap_factory(factory)(None,'0');session=conn._osp_sessions[0]
    # Record what the deployed tracker observes (fresh anchors only), per episode.
    current={'ep':None};observed=collections.defaultdict(list)
    if a.case!='A':
        method=session.method;base_tracker=method.tracker_class
        class RecordingTracker(base_tracker):
            def observe(self,key,control_index):
                observed[current['ep']].append((int(control_index),int(key.step),np.array(key.key_v0,copy=True),
                    np.array(key.key_v1,copy=True),np.array(key.rs,copy=True)))
                return super().observe(key,control_index)
        method.tracker_class=RecordingTracker
    selected=[i for i,e in enumerate(qc.episodes) if e['task_id'] in (0,1) and e['init'] in (0,1)]
    assert len(selected)==4
    if a.reverse:selected.reverse()
    rows=[];eps=[];steps=[];acts=[];vision=[];hits=[];img=np.zeros((2,2,3),np.uint8)
    for ei in selected:
        e=qc.episodes[ei];uid=('renamed/' if a.reverse else '')+e['uid'];current['ep']=ei
        conn.on_episode_start(task=e['task'],episode_id=e['init'],extra_metadata=dict(task_uid=uid,task_id=e['task_id'],orig_init_state_idx=e['init'],attempt=1))
        for step,r in enumerate(range(e['start'],e['end'])):
            obs={'observation/state':np.asarray(qc.raw_state[r],np.float64),'prompt':e['task'],'_row':r,
                'observation/image':img,'observation/wrist_image':img,'__extra__':{'decision_id':step,'executed_steps':5}}
            action=np.asarray(conn.infer(obs)['actions'],np.float32)
            assert action.shape==(rt.H,32) and np.array_equal(session.b_aex.a[step],action)
            rows.append(r);eps.append(ei);steps.append(step);acts.append(action)
            vision.append(bool(session.has_vision[-1]));hits.append(bool(session.hits[-1]))
        conn.on_episode_end(success=False) # placeholder terminal event, never an SR estimate
    decs=[json.loads(s) for s in rt.dec_path.read_text().splitlines() if '"ev": "dec"' in s]
    assert len(decs)==len(acts) and Fake.calls==sum(vision)
    tails=0;extra_looks=0;last_call={};last_cooldown={};consecutive_lottery=0;cooldowns=0
    amb=collections.Counter();anchors_by_ep=collections.defaultdict(list)
    for j,d in enumerate(decs):
        ex=d.get('extras',{})
        if not hits[j]:
            assert vision[j] and d['src']=='policy' and np.array_equal(acts[j],qc.a_inf[rows[j]])
            if j+1<len(acts) and eps[j+1]==eps[j]:
                assert decs[j+1]['src']=='policy_tail'
                assert np.array_equal(acts[j+1],policy_tail_chunk(acts[j],5)) # all 32 cols, gripper sign included
                assert not vision[j+1];tails+=1
                if j+2<len(acts) and eps[j+2]==eps[j]:assert vision[j+2]
        if vision[j] and 'os_c_p' in ex:
            assert ex['os_c_fresh']==1 and (ex['os_c_coin']<ex['os_c_p'])==(not hits[j])
            k=int(ex['os_c_anchor'])
            cooled=eps[j] in last_cooldown and k==last_cooldown[eps[j]]+1
            assert ex['os_c_cooldown']==float(cooled)
            if cooled:
                assert hits[j] and ex['os_c_p']==0;cooldowns+=1
            if not hits[j]:
                if eps[j] in last_call and k==last_call[eps[j]]+1:consecutive_lottery+=1
                last_call[eps[j]]=k
                if ex['os_c_stall_call']:last_cooldown[eps[j]]=k
            if ex['os_c_extra_look']:
                extra_looks+=1
                if j+1<len(acts) and eps[j+1]==eps[j]:assert vision[j+1]
            # §7b, re-derived from the log with the shared rule.
            state=next(k for k,v in STATE.items() if v==ex['os_c_stall_state'])
            assert ex['os_c_p']==call_probability(state,bool(ex['os_c_cooldown']),ex['os_c_nominal_p'])
            assert ex['os_c_call']==float(ex['os_c_coin']<ex['os_c_p'])
            assert not (ex['os_c_call'] and ex['os_c_extra_look'])
            assert ex['os_reason']==(63 if ex['os_c_stall_call'] else 62 if ex['os_c_call'] else 0)
            if state=='slow_ambiguous':
                amb['anchors']+=1;amb['p_positive']+=ex['os_c_p']>0;amb['lottery_calls']+=ex['os_c_call']>0
                amb['extra_LOOKs']+=ex['os_c_extra_look']>0
                if not ex['os_c_cooldown']:assert ex['os_c_p']==ex['os_c_nominal_p']
            anchors_by_ep[eps[j]].append((steps[j],ex))
        if not vision[j] and 'os_c_p' in ex:
            assert ex['os_c_fresh']==0 and ex['os_c_p']==0 and ex['os_c_call']==0
    # Deployed trajectory == one path of the modeled cadence DAG (same recorded keys).
    dag=collections.Counter()
    if a.case!='A':
        for ei in selected:
            e=qc.episodes[ei];ep_rows=list(range(e['start'],e['end']))
            keys=[SimpleNamespace(key_v0=np.array(qc.key_v0[r]),key_v1=np.array(qc.key_v1[r]),rs=np.array(qc.rs[r]),step=s)
                  for s,r in enumerate(ep_rows)]
            for control,step,v0,v1,rs in observed[ei]:
                assert control==5*step
                k=keys[step]
                for x,y in [(v0,k.key_v0),(v1,k.key_v1),(rs,k.rs)]:
                    assert np.array_equal(np.asarray(x,np.float32),np.asarray(y,np.float32))
            tree=replay_cadence(dict(uid=e['uid'],task=int(e['task_id']),init=int(e['init']),weight=1.,block_controls=5,commit_controls=10,
                decisions=[dict(step=s,control_index=5*s,Ehat=float('nan'),key=k) for s,k in enumerate(keys)]),
                method.stall_model,base_tracker,'stall')
            path=anchors_by_ep[ei];assert len(path)==len(observed[ei])
            i=0
            for step,ex in path:
                n=tree['nodes'][i]
                assert n['step']==step and STATE[n['state']]==ex['os_c_stall_state'] and n['cooled']==bool(ex['os_c_cooldown'])
                call=bool(ex['os_c_call'])
                assert call_probability(n['state'],n['cooled'],ex['os_c_nominal_p'])==ex['os_c_p']
                assert scheduled_look(n['look_eligible'],call)==bool(ex['os_c_extra_look'])
                i=n['call'] if call else n['nocall'];assert i is not None
                dag['anchors']+=1
            assert i==-1,'deployed episode ended before the modeled path'
            dag['episodes']+=1;dag['dag_nodes']+=len(tree['nodes'])
    np.savez(out/'served.npz',row=rows,ep=eps,step=steps,action=acts,vision=vision,hit=hits,source=[d['src'] for d in decs])
    if a.case=='R45':assert consecutive_lottery>0
    report=dict(status='PASS',controller_version='R6-C-v2',ambiguous_rule=cal.get('ambiguous_rule'),calibration=str(calpath),
        cell=a.cell,case=a.case,reverse=a.reverse,episodes=4,decisions=len(acts),
        consecutive_lottery_calls=consecutive_lottery,stall_cooldowns=cooldowns,ambiguous=dict(amb),dag_path_agreement=dict(dag),
        anchors=sum(vision),calls=sum(not h for h in hits),tails=tails,extra_LOOKs=extra_looks,
        IR=(.152 if model=='pi05' else .148)*sum(vision)/len(acts)+( .848 if model=='pi05' else .852)*sum(not h for h in hits)/len(acts),
        sources=dict(collections.Counter(d['src'] for d in decs)),action_sha256=hashlib.sha256(np.asarray(acts).tobytes()).hexdigest(),
        no_model_no_simulator=True,no_test_fit_pickle=True)
    write_json(out/'report.json',report);print(json.dumps(report))

if __name__=='__main__':main()
