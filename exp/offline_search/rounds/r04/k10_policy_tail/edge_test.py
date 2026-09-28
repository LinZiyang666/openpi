"""Deterministic races: real sessions/orchestrators, explicit synchronization events."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import threading
import time
import types
import numpy as np
from concurrency_test import load


def main(a):
    plugin=load(a.source)
    from exp.offline_search.closed_loop import selftest
    from exp.offline_search.harness import store
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    opts,_=plugin.parse_cli(['--os-method','exp.offline_search.closed_loop.probe:ProbeBlind','--os-cell','pi05_spatial_cache','--os-root','/home/weiland/trace_runs/offline_search_store','--os-log-dir',str(out),'--os-tag','edge','--os-log-inputs','--os-blind','--os-judge','guard_only'])
    rt=plugin.install(opts,'pi05');qc=store.QueryCell(opts.os_root,opts.os_cell)
    cfg=cc.load_cache_config('exp/trace_dual/config/tr_pi05_sp_cache.yaml');shared=cc.build_shared_storage(cfg)
    class Fake(selftest.FakePolicy):
        def stage1(self,obs):
            assert not rt.decision_lock._is_owned()
            if obs.get('_raise'): raise RuntimeError('injected stage failure')
            if '_block' in obs:
                entered,release=obs['_block'];entered.set();assert release.wait(10)
        def infer(self,obs): self.stage1(obs);return super().infer(obs)
        def _osp_prepare_blind(self,obs): return np.array(qc.rs[obs['_row']],copy=True),None
        def _osp_blind_output(self,action,state): return {'actions':action.copy()}
    def factory(_base,bundle_id='default'):
        comps=cc.build_per_connection_components(cfg,shared,quiet=True);kb=selftest.FakeKB(qc)
        for s in plugin._TLS.new_sessions:s.kb=kb
        orch=CacheOrchestrator(storage=comps['storage'],key_builder=kb,gates=comps['gates'],judges=comps['judges'],search_strategies=comps['search_strategies'],timer=comps['timer'],write_policy=comps.get('write_policy'),offline_writers=comps.get('offline_writers',()),library_stats=comps.get('library_stats'))
        return Fake(orch,kb,np.asarray(qc.a_inf))
    conns=[plugin._wrap_factory(factory)(None,str(i)) for i in range(8)]
    eps=[qc.episodes[i] for i in selftest.pick_episodes(qc,2)]
    def start(i,ep=0,uid=None):
        e=eps[ep];conns[i].on_episode_start(task=e['task'],episode_id=e['init'],extra_metadata={'task_uid':uid or f'c{i}','task_id':e['task_id'],'orig_init_state_idx':e['init']})
    def obs(step,ep=0,**kw):
        e=eps[ep];row=e['start']+step
        return {'observation/state':np.asarray(qc.raw_state[row],np.float64),'prompt':e['task'],'_row':row,'__extra__':{'decision_id':step,'executed_steps':5},**kw}
    def rejected(future,needle):
        try: future.result(timeout=10)
        except (ValueError,RuntimeError) as exc: assert needle in str(exc)
        else:raise AssertionError('expected rejection')
    for i in range(8): start(i)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda c:c.infer(obs(0)),conns))
        for step in (1,2,3): conns[0].infer(obs(step))
        entered,release=threading.Event(),threading.Event()
        request=obs(4,_block=(entered,release))
        miss=pool.submit(conns[0].infer,request);assert entered.wait(10)
        duplicate=pool.submit(conns[0].infer,request)
        # Other connections complete blind and lifecycle decisions while c0 is blocked.
        blind=pool.submit(conns[1].infer,obs(1));assert blind.result(timeout=5)['__hit_meta__']['source']=='cache_blind'
        start(2,1,'reset-c2');conns[2].infer(obs(0,1))
        changed=obs(0,1);changed['__extra__']['decision_id']=100
        conns[3].infer(changed)  # implicit prompt/task change, no episode_start
        assert conns[3]._osp_sessions[0].ep.task==eps[1]['task']
        assert conns[2]._osp_sessions[0].step==conns[3]._osp_sessions[0].step==1
        assert not duplicate.done()
        count=rt.decision_count
        rejected(pool.submit(conns[4].infer,obs(0)),'duplicate decision_id')
        assert rt.decision_count==count
        release.set();assert miss.result(timeout=10)['hit_type']=='MISS'
        rejected(duplicate,'duplicate decision_id')
        assert rt.decision_count==count
        assert conns[0]._osp_sessions[0].hits==[1,1,1,1,0]
        conns[0].infer(obs(5));assert conns[0]._osp_sessions[0].has_vision[-1]
        # Same-connection lifecycle must wait until infer AND after_infer complete.
        entered,release=threading.Event(),threading.Event()
        request=obs(1,_block=(entered,release));request['__extra__']['executed_steps']=4
        running=pool.submit(conns[5].infer,request);assert entered.wait(10)
        resetting=pool.submit(start,5,1,'race-reset-c5')
        conns[6].infer(obs(1));assert not resetting.done()
        release.set();running.result(timeout=10);resetting.result(timeout=10)
        conns[5].infer(obs(0,1));s=conns[5]._osp_sessions[0]
        assert s.step==1 and s.method.n==1 and s.method.blind_calls==0
        # A failed stage consumes its reserved index, but cannot poison any peer.
        request=obs(1,_raise=True);request['__extra__']['executed_steps']=4
        before=rt.decision_count
        rejected(pool.submit(conns[7].infer,request),'injected stage failure')
        assert rt.decision_count==before+1
        assert conns[7]._osp_sessions[0].step==1
        conns[6].infer(obs(2));conns[7].infer(obs(1))
        # Exception after input-history mutation: only that connection needs reset.
        s=conns[7]._osp_sessions[0];query=s.method.query
        def broken(q):raise RuntimeError('injected query failure')
        s.method.query=broken
        request=obs(2);request['__extra__']['executed_steps']=4
        rejected(pool.submit(conns[7].infer,request),'injected query failure')
        s.method.query=query;start(7,1,'after-failure-c7');conns[7].infer(obs(0,1))
        conns[6].infer(obs(3))
    for c in conns:c.on_episode_end(success=False)
    rows=[json.loads(l) for l in rt.dec_path.read_text().splitlines()]
    dec=[r for r in rows if r['ev']=='dec'];indices=[r['decision_index'] for r in dec]
    assert len(indices)==len(set(indices)) and len(dec)==rt.decision_count-2
    for i in range(8):
        seq=[r['decision_index'] for r in dec if r['conn']==i];assert seq==sorted(seq)
    # Eight writers; deliberately force short writes, with yields between fragments.
    real_os=plugin.os;write=real_os.write
    def short_write(fd,data):
        n=write(fd,data[:17]);time.sleep(.00001);return n
    plugin.os=types.SimpleNamespace(**{**vars(real_os),'write':short_write})
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i:rt.emit({'ev':'writer_probe','i':i,'payload':str(i)*4096}),range(32)))
    finally:plugin.os=real_os
    rows=[json.loads(l) for l in rt.dec_path.read_text().splitlines()]
    probes=[r for r in rows if r['ev']=='writer_probe'];assert len(probes)==32
    assert sorted(r['i'] for r in probes)==list(range(32))
    assert all(r['payload']==str(r['i'])*4096 for r in probes)
    # The controller's tau/read/decide/push must be one linearizable transaction.
    ctrl=plugin.QuantileController(.7,20,.5);rt.ctrl=ctrl
    rt.judge=plugin.JudgeSpec.parse('quantile:0.7:20:0.5')
    audit=[];push=ctrl.push
    def tracked_push(v):
        assert rt.decision_lock._is_owned()
        audit.append((local.i,v));push(v)
    ctrl.push=tracked_push
    results={};local=threading.local()
    original_tau=ctrl.tau
    def tau():
        value=original_tau();time.sleep(.001);return value
    ctrl.tau=tau
    def verdict(i):
        s=conns[i%8]._osp_sessions[0]
        with s.lock:
            s.hits=[];s.burst_left=0
            local.i=i
            conf=(i%13)/13
            value=s._verdict(1,conf,{})
            results[i]=(conf,value)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verdict,range(64)))
    ref=plugin.QuantileController(.7,20,.5)
    for i,_ in audit:
        conf,r=results[i];assert r['tau']==ref.tau() and r['hit']==(conf>=ref.tau());ref.push(conf)
    # Strict R4 cloning cannot silently share nested mutable method containers.
    class Uncopyable:
        def __deepcopy__(self,memo):raise TypeError('uncopyable')
    try:plugin.clone_method(Uncopyable(),strict=True)
    except RuntimeError:pass
    else:raise AssertionError('unsafe clone accepted')
    report=dict(PASS=True,connections=8,decisions=len(dec),reserved=rt.decision_count,failed_reservation_gaps=2,duplicate_rejections=2,peer_progress_during_blocked_miss=True,same_connection_duplicate_serialized=True,lifecycle_race=True,explicit_reset_and_implicit_task_change=True,exception_isolation=True,jsonl_short_write_rows=32,quantile_transactions=64,strict_clone_rejected=True)
    (out/'edge.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='dev');p.add_argument('--out',required=True);main(p.parse_args())
