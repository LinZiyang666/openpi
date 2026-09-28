"""CycleTail integration, real CPU wire transforms, lifecycle and queue equality."""
import argparse
from collections import deque
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'regression'))
from launch_test import load


def main(a):
    plugin=load(a.source)
    from exp.offline_search.closed_loop import selftest,verify_logs
    from exp.offline_search.closed_loop.blind import BlindResult,LookReason,policy_tail_chunk
    from exp.offline_search.harness import store
    from exp.offline_search.rounds.r04.k4_eval.cost_ledger import ledger
    from cpu_transforms import action_transform,wire
    from openpi.cache.orchestrator import CacheOrchestrator
    import openpi.cache.config as cc
    import torch
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    kw=dict(lib='current' if a.scale==50 else 'big',kref=5 if a.scale==50 else 8,cycle_k=4,tail_blocks=a.blocks)
    opts,_=plugin.parse_cli(['--os-method','exp.offline_search.rounds.r05.q2_groot.judge:CycleTail','--os-kwargs',json.dumps(kw),'--os-cell',f'groot_{a.suite}_cache','--os-root','/home/weiland/trace_runs/offline_search_store','--os-log-dir',str(out),'--os-tag','q2','--os-log-inputs','--os-no-shadow-native','--os-blind','--os-policy-tail','--os-policy-tail-blocks',str(a.blocks),'--os-judge','guard_only'])
    rt=plugin.install(opts,'groot');qc=store.QueryCell(opts.os_root,f'groot_{a.suite}_inf')
    cfg=cc.load_cache_config(f"exp/trace_dual/config/tr_groot_{'sp' if a.suite=='spatial' else 'l10'}_cache.yaml")
    shared=cc.build_shared_storage(cfg);transform,metadata=action_transform(a.suite)
    # Exercise production output adapter, with real inverse transforms, no model.
    cpu=object.__new__(plugin._BlindAdapter);cpu.fake=False
    cpu.s=SimpleNamespace(rt=SimpleNamespace(model='groot'))
    cpu.policy=SimpleNamespace(_policy=SimpleNamespace(unapply_transforms=transform.unapply))
    counts=dict(actual_chunks=0,queue_controls=0,policy_tails=0,cache_tails=0)
    for row in np.linspace(0,len(qc.a_inf)-1,256,dtype=int):
        chunk=np.asarray(qc.a_inf[row]);original=wire(transform,chunk)
        assert np.array_equal(cpu.output(chunk,None)['actions'],original)
        for blocks in (1,2):
            length=5*(blocks+1);client=deque(original[:length]);server=deque(original[:5])
            for h in range(1,blocks+1):
                shifted=policy_tail_chunk(original,5*h)
                assert shifted.dtype==original.dtype
                assert shifted[:5].tobytes()==original[5*h:5*h+5].tobytes()
                server.extend(shifted[:5])
            while client:
                assert client.popleft().tobytes()==server.popleft().tobytes()
                counts['queue_controls']+=1
        counts['actual_chunks']+=1
    sign=np.zeros((16,32),np.float32);sign[:,6]=np.resize([-1.,0.,1.],16)
    signed=cpu.output(sign,None)['actions'][:,6]
    assert np.array_equal(signed,-np.sign(sign[:,6]))
    checks=['256 actual H16 chunks: exact L10/L15 client queues','production GR00T output adapter and reversed gripper sign']
    # Production prepare() path: exact CPU cast/mask, including non-contiguous mask.
    for dtype in (torch.float32,torch.bfloat16):
        norm=np.arange(64,dtype=np.float64).reshape(1,1,64)/3
        mask=np.zeros((1,1,64),bool);mask[0,0,[0,2,4,6,8,10,12,14]]=True
        cpu.policy._policy.apply_transforms=lambda _:dict(state=norm,state_mask=mask)
        cpu.policy._runner=SimpleNamespace(_model=SimpleNamespace(action_head=SimpleNamespace(dtype=dtype)))
        cpu.s.kb=SimpleNamespace(_state_index=torch.as_tensor(mask[0,0]))
        testobs={'observation/state':np.zeros(8),'observation/image':np.zeros((256,256,3),np.uint8),'observation/wrist_image':np.zeros((256,256,3),np.uint8),'prompt':'test'}
        rs,_=cpu.prepare(testobs)
        assert rs.tobytes()==torch.as_tensor(norm).to(dtype)[0,-1,mask[0,0]].float().numpy().tobytes()
        cpu.s.kb._state_index=~cpu.s.kb._state_index
        try:cpu.prepare(testobs)
        except ValueError:pass
        else:raise AssertionError('changed state mask accepted')
    checks.append('production prepare float32/bfloat16 cast, masked state, changing-mask refusal')
    class Fake(selftest.FakePolicy):
        stateful=False
        def stage1(self,obs):
            assert not rt.decision_lock._is_owned()
            if obs.get('_expect_blind'):raise AssertionError('vision on tail')
        def wire(self,action,state):
            result=cpu.output(action,None)
            if self.stateful:result['actions'][:,:6]+=state[:6]
            return result
        def infer(self,obs):
            self.stage1(obs);res=super().infer(obs)
            res['actions']=self.wire(res['actions'],obs['observation/state'])['actions'];return res
        def _osp_prepare_blind(self,obs):return np.array(qc.rs[obs['_row']],copy=True),obs['observation/state']
        def _osp_blind_output(self,action,state):return self.wire(action,state)
    def factory(_base,bundle_id='default'):
        comps=cc.build_per_connection_components(cfg,shared,quiet=True);kb=selftest.FakeKB(qc)
        for s in plugin._TLS.new_sessions:s.kb=kb
        orch=CacheOrchestrator(storage=comps['storage'],key_builder=kb,gates=comps['gates'],judges=comps['judges'],search_strategies=comps['search_strategies'],timer=comps['timer'],write_policy=comps.get('write_policy'),offline_writers=comps.get('offline_writers',()),library_stats=comps.get('library_stats'))
        return Fake(orch,kb,np.asarray(qc.a_inf))
    conn=plugin._wrap_factory(factory)(None,'0');s=conn._osp_sessions[0]
    episodes=[qc.episodes[i] for i in selftest.pick_episodes(qc,8)];sequence=0
    def start(ei=0,uid=None):
        nonlocal sequence
        sequence+=1;e=episodes[ei]
        conn.on_episode_start(task=e['task'],episode_id=e['init'],extra_metadata={'task_uid':uid or f'q2e{sequence}','task_id':e['task_id'],'orig_init_state_idx':e['init']})
    def obs(step,ei=0,executed=5):
        e=episodes[ei];r=e['start']+min(step,e['end']-e['start']-1)
        return {'observation/state':np.asarray(qc.raw_state[r],np.float64),'prompt':e['task'],'_row':r,'__extra__':{'decision_id':step,'executed_steps':executed}}
    start()
    for anchor in range(9):
        step=anchor*(a.blocks+1);head=conn.infer(obs(step));normalized=s.b_aex.a[step].copy()
        assert s.has_vision[-1] and s.hits[-1]==int(anchor%4!=0)
        assert s.method._anchor_count==anchor+1
        for h in range(1,a.blocks+1):
            request=obs(step+h);request['_expect_blind']=True
            tail=conn.infer(request);kind='policy_tails' if anchor%4==0 else 'cache_tails';counts[kind]+=1
            assert s.hits[-1]==1 and not s.has_vision[-1]
            assert tail['actions'][:5].tobytes()==head['actions'][h*5:h*5+5].tobytes()
            assert s.b_aex.a[step+h,:5,:7].tobytes()==normalized[h*5:h*5+5,:7].tobytes()
            assert np.isnan(s.b_v0.a[step+h]).all() and np.isnan(s.b_v1.a[step+h]).all()
            assert s.step==s.b_aex.n==conn.orch._step_counter==step+h+1
            assert s.method._anchor_count==anchor+1
    conn.on_episode_end(success=False)
    # Verify standard schedule BEFORE injecting deliberately bad method/state fixtures.
    assert verify_logs.main(['--log-dir',str(out),'--tag','q2','--work',str(out/'verify')])==0
    checks.append('nine real anchors: MISS 0/4/8, source-own cache and policy tails, no fabricated visual keys')
    start(uid='reused');conn.infer(obs(0));conn.on_episode_end(success=True)
    start(uid='reused');conn.infer(obs(0));assert s.hits==[0] and s.has_vision==[True]
    checks.append('MISS at last decision / identical external UID reset')
    saved=s._policy_tail;start(1);conn.infer(obs(0,1));s._policy_tail=saved
    conn.infer(obs(1,1));assert s.has_vision[-1] and s._look_reason==6
    checks.append('stale snapshot from another internal episode rejected')
    start();conn.infer(obs(0));conn.infer(obs(1,1));assert s.step==1 and s.hits==[0]
    checks.append('implicit task change resets anchor clock and forces first MISS')
    start();conn.infer(obs(0));conn.infer(obs(0,1));assert s.step==1 and s.hits==[0]
    checks.append('implicit task change permits a reset decision id zero')
    for count in (0,4,6,10,15):
        start();conn.infer(obs(0));conn.infer(obs(1,executed=count));assert s.has_vision[-1] and s._look_reason==6
    checks.append('partial/non-five execution audit requests vision')
    start();conn.infer(obs(0));ob=obs(1);ob.pop('__extra__');tail=conn.infer(ob);assert not s.has_vision[-1]
    checks.append('absent execution audit uses configured L5 contract')
    start();head=conn.infer(obs(0))
    for step in range(a.blocks+1):
        snapshot=(s.step,s.b_aex.n,rt.decision_count,s.method._anchor_count,copy.deepcopy(s._policy_tail))
        try:conn.infer(obs(step))
        except ValueError as exc:assert 'duplicate' in str(exc)
        else:raise AssertionError('duplicate committed')
        assert snapshot[:4]==(s.step,s.b_aex.n,rt.decision_count,s.method._anchor_count)
        if snapshot[4] is not None:assert np.array_equal(snapshot[4]['wire'],s._policy_tail['wire'])
        if step<a.blocks:conn.infer(obs(step+1))
    checks.append('duplicate MISS/first-tail/second-tail ids preserve cursor and reserve/commit nothing')
    # Wrong normalized tail cannot become an accepted original-wire response.
    start();conn.infer(obs(0));hook=s.method.policy_tail_step
    def bad(q):
        result=hook(q);return replace(result,action=np.zeros((16,32),np.float32))
    s.method.policy_tail_step=bad;conn.infer(obs(1));assert s.has_vision[-1] and s._look_reason==8
    del s.method.policy_tail_step
    checks.append('wrong normalized policy tail refused before broadcast')
    start();head=conn.infer(obs(0));conn._osp_inner.stateful=True
    for step in range(1,a.blocks+1):
        ob=obs(step);ob['observation/state']=ob['observation/state']+10
        tail=conn.infer(ob);assert tail['actions'][:5].tobytes()==head['actions'][step*5:step*5+5].tobytes()
    conn._osp_inner.stateful=False
    checks.append('original policy wire retained across state-dependent transform changes')
    # An over-permissive hook cannot execute padding at offset 15.
    good=s.method.blind_step
    s.method.blind_step=lambda q:BlindResult(np.zeros((16,32),np.float32),np.array([0],np.int64),np.ones(1,np.float32),'current',{})
    conn.infer(obs(a.blocks+1));assert s.has_vision[-1] and s._look_reason==6
    del s.method.blind_step
    checks.append('exhausted policy source requires vision even with permissive method')
    # Admission failure clears saved response; a duplicate failure above did not.
    start();conn.infer(obs(0));prepare=conn._osp_adapter.prepare
    def fail(_):raise RuntimeError('injected admitted prepare failure')
    conn._osp_adapter.prepare=fail
    reserved=rt.decision_count
    try:conn.infer(obs(1))
    except RuntimeError:pass
    else:raise AssertionError('expected failure')
    assert s._policy_tail is None and rt.decision_count==reserved+1
    conn._osp_adapter.prepare=prepare
    start();conn.infer(obs(0));conn.on_episode_end(success=False)
    checks.append('admitted failure discards cursor; next episode starts with MISS')
    dec=[json.loads(l) for l in rt.dec_path.read_text().splitlines() if json.loads(l).get('ev')=='dec']
    tails=[r for r in dec if r['src']=='policy_tail']
    starts=[json.loads(l) for l in rt.dec_path.read_text().splitlines() if json.loads(l).get('ev')=='startup']
    cost=ledger(dict(model='groot',suite=a.suite,cost_ledger=True),tails,starts)
    assert cost['vision_decisions']==0 and cost['misses']==0 and cost['total_cost']==0
    checks.append('policy tails cost zero using explicit vision/hit ledger')
    report=dict(PASS=True,suite=a.suite,scale=a.scale,blocks=a.blocks,checks=checks,counts=counts,decisions=len(dec),all_policy_tails=len(tails),metadata=str(metadata),tail_ledger=cost)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='installed');p.add_argument('--suite',default='spatial');p.add_argument('--scale',type=int,default=50);p.add_argument('--blocks',type=int,default=2);p.add_argument('--out',required=True)
    main(p.parse_args())
