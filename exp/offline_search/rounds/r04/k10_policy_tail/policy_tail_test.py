"""CPU integration: actual pi05 output transforms, L=10 client queue, edge vetoes.

No model weights, inference server, network, or simulator is constructed.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
import numpy as np
from launch_test import load


def main(a):
    plugin=load(a.source)
    from exp.offline_search.closed_loop import selftest
    from exp.offline_search.closed_loop.blind import BlindResult, LookReason, policy_tail_chunk
    from exp.offline_search.harness import store
    from exp.offline_search.rounds.r04.k4_eval.cost_ledger import ledger
    from openpi import transforms
    from openpi.training import config, checkpoints
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    kw=dict(base_kwargs=dict(lib='current',kref=5,serving='anchor_tail',budget=1,gates='budget_only'),guards=False,ncal=64)
    opts,_=plugin.parse_cli(['--os-method','exp.offline_search.rounds.r04.k10_policy_tail.judge:PolicyTailJudge','--os-kwargs',json.dumps(kw),'--os-cell','pi05_l10_cache','--os-root','/home/weiland/trace_runs/offline_search_store','--os-log-dir',str(out),'--os-tag','tail','--os-log-inputs','--os-no-shadow-native','--os-blind','--os-policy-tail','--os-judge','threshold:inf'])
    rt=plugin.install(opts,'pi05');qc=store.QueryCell(opts.os_root,opts.os_cell)
    cfg=cc.load_cache_config('exp/trace_dual/config/tr_pi05_l10_cache.yaml');shared=cc.build_shared_storage(cfg)
    pcfg=config.get_config('pi05_libero');data=pcfg.data.create(pcfg.assets_dirs,pcfg.model)
    checkpoint=Path('/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch')
    try: stats=checkpoints.load_norm_stats(checkpoint/'assets',data.asset_id)
    except FileNotFoundError:
        checkpoint=Path(pcfg.assets_dirs)
        stats=checkpoints.load_norm_stats(checkpoint,data.asset_id)
    output_transform=transforms.compose([*data.model_transforms.outputs,transforms.Unnormalize(stats,use_quantiles=data.use_quantile_norm),*data.data_transforms.outputs])
    checks=[];active=peak=0;lock=threading.Lock()
    class Fake(selftest.FakePolicy):
        stateful=False
        def stage1(self,obs):
            nonlocal active,peak
            assert not rt.decision_lock._is_owned()
            with lock:active+=1;peak=max(peak,active)
            time.sleep(.01)
            with lock:active-=1
        def wire(self,action,state):
            result=output_transform({'state':np.array(state),'actions':np.array(action)})
            if self.stateful:
                result=transforms.AbsoluteActions([True]*6+[False])({'state':np.array(state),'actions':result['actions'].copy()})
            return {'actions':np.asarray(result['actions'])}
        def infer(self,obs):
            self.stage1(obs);result=super().infer(obs)
            result['actions']=self.wire(result['actions'],obs['observation/state'])['actions']
            return result
        def _osp_prepare_blind(self,obs):return np.array(qc.rs[obs['_row']],copy=True),obs['observation/state']
        def _osp_blind_output(self,action,state):return self.wire(action,state)
    def factory(_base,bundle_id='default'):
        comps=cc.build_per_connection_components(cfg,shared,quiet=True);kb=selftest.FakeKB(qc)
        for s in plugin._TLS.new_sessions:s.kb=kb
        orch=CacheOrchestrator(storage=comps['storage'],key_builder=kb,gates=comps['gates'],judges=comps['judges'],search_strategies=comps['search_strategies'],timer=comps['timer'],write_policy=comps.get('write_policy'),offline_writers=comps.get('offline_writers',()),library_stats=comps.get('library_stats'))
        return Fake(orch,kb,np.asarray(qc.a_inf))
    conns=[plugin._wrap_factory(factory)(None,str(i)) for i in range(8)]
    eps=[qc.episodes[i] for i in selftest.pick_episodes(qc,8)]
    seq=0
    def start(i,ei=0,uid=None):
        nonlocal seq
        seq+=1;e=eps[ei]
        conns[i].on_episode_start(task=e['task'],episode_id=e['init'],extra_metadata={'task_uid':uid or f'c{i}:e{seq}','task_id':e['task_id'],'orig_init_state_idx':e['init']})
    def obs(step,ei=0,executed=5):
        e=eps[ei];r=e['start']+step
        return {'observation/state':np.asarray(qc.raw_state[r],np.float64),'prompt':e['task'],'_row':r,'__extra__':{'decision_id':step,'executed_steps':executed}}
    for i in range(8):start(i,i)
    def drive(i):
        conn=conns[i];s=conn._osp_sessions[0]
        for step in range(0,10,2):
            miss=conn.infer(obs(step,i));tail=conn.infer(obs(step+1,i))
            assert miss['hit_type']=='MISS' and tail['hit_type']=='FULL_HIT'
            assert tail['__hit_meta__']['source']=='policy_tail'
            assert s.has_vision[-2:]==[True,False] and s.hits[-2:]==[0,1]
            assert np.array_equal(s.b_aex.a[step+1],policy_tail_chunk(qc.a_inf[eps[i]['start']+step]))
            assert np.array_equal(tail['actions'],policy_tail_chunk(miss['actions']))
            assert tail['actions'].dtype==miss['actions'].dtype
            assert np.concatenate((miss['actions'][:5],tail['actions'][:5])).tobytes()==miss['actions'][:10].tobytes()
            assert np.array_equal(tail['actions'][5:],np.broadcast_to(miss['actions'][-1],tail['actions'][5:].shape))
            normalized=np.asarray(qc.a_inf[eps[i]['start']+step],np.float32)
            assert s.b_aex.a[step+1,:5].tobytes()==normalized[5:10].tobytes()
            assert np.array_equal(s.b_aex.a[step+1,5:],np.broadcast_to(normalized[-1],(5,32)))
            # Exact client path (examples/libero/main.py): extend [:L], pop,
            # env.step(action.tolist()). Compare all ten controls, not only heads.
            l10=deque(miss['actions'][:10]);l5=deque(miss['actions'][:5]);l5.extend(tail['actions'][:5])
            for _ in range(10):assert l10.popleft().tolist()==l5.popleft().tolist()
            assert s.step==s.b_aex.n==conn.orch._step_counter==step+2
        conn.on_episode_end(success=False)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(drive,range(8)))
    assert peak>=2
    checks.append('40 policy tails: 400 real-transform L10 controls byte-exact across 8 connections')
    conn=conns[0];s=conn._osp_sessions[0]
    # Changing the observation state cannot change the saved wire tail.
    conn._osp_inner.stateful=True
    start(0);miss=conn.infer(obs(0));next_obs=obs(1);next_obs['observation/state']=next_obs['observation/state']+10
    tail=conn.infer(next_obs)
    assert np.array_equal(tail['actions'],policy_tail_chunk(miss['actions']))
    assert not np.array_equal(tail['actions'],conn._osp_inner.wire(s.b_aex.a[1],next_obs['observation/state'])['actions'])
    checks.append('state-dependent AbsoluteActions transform uses original wire response')
    # A buggy permissive method cannot serve a second blind decision.
    original=s.method.blind_step
    s.method.blind_step=lambda q:BlindResult(np.zeros((10,32),np.float32),np.array([0],np.int64),np.ones(1,np.float32),'current',{})
    conn.infer(obs(2));assert s.has_vision[-1] and s._look_reason==6
    s.method.blind_step=original
    checks.append('second consecutive blind forced to vision before method')
    for count in (0,4,6,10):
        start(0);conn.infer(obs(0));conn.infer(obs(1,executed=count))
        assert s.has_vision[-1] and s._look_reason==6
        checks.append(f'executed_steps={count} forces vision')
    # Default client contract (no audit field) still executes five controls.
    start(0);m=conn.infer(obs(0));o=obs(1);o.pop('__extra__');t=conn.infer(o)
    assert np.array_equal(t['actions'],policy_tail_chunk(m['actions']))
    checks.append('absent execution audit follows documented L5 contract')
    # End exactly after a MISS, then reuse identical external UID: internal epoch wins.
    start(0,uid='same');conn.infer(obs(0));conn.on_episode_end(success=True)
    start(0,uid='same');conn.infer(obs(0));assert s.has_vision==[True] and s._look_reason==6
    checks.append('MISS at last decision; reset even with same external uid')
    saved=s._policy_tail;start(0,1);conn.infer(obs(0,1));s._policy_tail=saved
    conn.infer(obs(1,1));assert s.has_vision[-1] and s._look_reason==6
    checks.append('MISS snapshot from other episode refused')
    start(0);conn.infer(obs(0));conn.infer(obs(1,1));assert s.step==1 and s.has_vision==[True]
    checks.append('implicit task change begins with vision')
    for label,change,code in [
        ('no-progress span veto',lambda m:setattr(m,'_noprog_span',2),8),
        ('zero budget veto',lambda m:setattr(m.base,'budget',0),1),
        ('gripper gate veto',lambda m:(setattr(m.base,'gates','all'),setattr(m.base,'blind_event',np.ones_like(m.base.blind_event))),2),
    ]:
        start(0);conn.infer(obs(0));saved_method,_=plugin.clone_method(s.method)
        s.method.guards=True;change(s.method);conn.infer(obs(1))
        assert s.has_vision[-1] and s._look_reason==code,(label,s._look_reason)
        s.method=saved_method;checks.append(label)
    start(0);conn.infer(obs(0));s.burst_left=1;conn.infer(obs(1))
    assert s.has_vision[-1] and s._look_reason==8;checks.append('plugin judge burst veto')
    start(0);conn.infer(obs(0));old=rt.judge;rt.judge=plugin.JudgeSpec.parse('periodic:1');conn.infer(obs(1));rt.judge=old
    assert s.has_vision[-1] and s._look_reason==7;checks.append('global periodic due wins')
    start(0);conn.infer(obs(0));s.method.policy_tail_step=None;conn.infer(obs(1));del s.method.policy_tail_step
    assert s.has_vision[-1] and s._look_reason==8;checks.append('optional hook absent forces vision')
    start(0);conn.infer(obs(0));hook=s.method.policy_tail_step
    def wrong(q):
        r=hook(q);return replace(r,action=r.action+1)
    s.method.policy_tail_step=wrong;conn.infer(obs(1));del s.method.policy_tail_step
    assert s.has_vision[-1] and s._look_reason==8;checks.append('hook cannot substitute another action')
    start(0);conn.infer(obs(0));prepare=conn._osp_adapter.prepare
    conn._osp_adapter.prepare=lambda o:(np.full_like(qc.rs[0],np.nan),o['observation/state'])
    o=obs(1);s.set_obs(o);s._decision_index=rt.decision_count
    before=(s.step,s.b_aex.n,rt.decision_count)
    assert plugin._try_blind(s,conn._osp_adapter,o) is None and s._look_reason==6
    assert before==(s.step,s.b_aex.n,rt.decision_count)
    conn._osp_adapter.prepare=prepare;checks.append('invalid state forces vision without commit')
    for c in conns:c.on_episode_end(success=False)
    rows=[json.loads(x) for x in rt.dec_path.read_text().splitlines()];decs=[r for r in rows if r['ev']=='dec'];tails=[r for r in decs if r['src']=='policy_tail']
    starts=[r for r in rows if r['ev']=='startup']
    bill=ledger({'model':'pi05','suite':'l10','cost_ledger':True},tails,starts)
    assert bill['ir_per_five_controls']==0 and all(r['hit'] and not r['vision'] and r['miss_k'] is None and r['s1_ms'] is None and r['s23_ms'] is None for r in tails)
    # K7 proof on a known MISS -> tail -> vision gap: no blind keys, span counts 2.
    from exp.offline_search.rounds.r04.k10_policy_tail.judge import PolicyTailJudge
    from types import SimpleNamespace as NS
    m=PolicyTailJudge(**kw);m.model='pi05';m.m_thr=1.;m.c_thr=.95;m.M0={0:np.zeros(2,np.float32)};m.M1={0:np.zeros(2,np.float32)}
    q=NS(step=2,task_id=0,rs=np.zeros(32,np.float32),hist_rs=np.zeros((2,32),np.float32),key_v0=np.array([1,0],np.float32),key_v1=np.array([1,0],np.float32),hist_key_v0=np.array([[1,0],[np.nan,np.nan]],np.float32),hist_key_v1=np.array([[1,0],[np.nan,np.nan]],np.float32),hist_has_vision=np.array([True,False]),prev_hit=True)
    assert m.confirmed_stuck(q)==2
    m._vision_progress=[(0,0.,10)];m.C=NS(prog=np.array([0.]),ep_len=np.array([11]));assert m._progress(q,0)==2
    checks.append('K7 confirmed stuck and no-progress span count the policy blind gap')
    report=dict(PASS=True,checks=checks,decisions=len(decs),policy_tails=len(tails),l10_controls=400,concurrent_connections=8,peak_stage1=peak,tail_ledger=bill,checkpoint=str(checkpoint))
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='installed');p.add_argument('--out',required=True);main(p.parse_args())
