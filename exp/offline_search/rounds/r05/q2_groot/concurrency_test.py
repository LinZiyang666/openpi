"""Real plugin/orchestrator, recorded keys/actions, eight concurrent sleeping policies.

Threaded reservation order is replayed by the pre-K6 plugin in a fresh process.
This controls the only allowed ambiguity in periodic:k: global arrival order.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

BASE=Path(__file__).resolve().parent
PREFIX=['taskset','-c','30-33,74-77','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',sys.executable]


sys.path.insert(0,str(BASE/'regression'))
from launch_test import load


def configs():
    result={}
    for suite in ('spatial','l10'):
        for scale in (50,500):
            for blocks in (1,2):
                key=f'{suite}_{scale}_G{5*(blocks+1)}'
                result[key]=dict(cell=f'groot_{suite}_cache',judge='guard_only',blind=True,blocks=blocks,
                    method='exp.offline_search.rounds.r05.q2_groot.judge:CycleTail',
                    kwargs=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8,cycle_k=4,tail_blocks=blocks))
    return result


def digest(x):
    a=np.asarray(x)
    return hashlib.sha256(a.tobytes()).hexdigest()


def clean(x):
    if isinstance(x,dict):
        return {k:clean(v) for k,v in x.items() if not(k.endswith(('_ms','_us')) or k in ('ts','t_start','t_end'))}
    if isinstance(x,list): return [clean(v) for v in x]
    return x


def worker(a):
    plugin=load(a.source)
    from exp.offline_search.closed_loop import selftest,verify_logs
    from exp.offline_search.harness import store
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator
    c=configs()[a.config]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    args=['--os-method',c['method'],'--os-kwargs',json.dumps(c['kwargs']),'--os-cell',c['cell'],'--os-root','/home/weiland/trace_runs/offline_search_store','--os-log-dir',str(out),'--os-tag','concurrency','--os-log-inputs','--os-log-r4','--os-judge',c['judge']]
    if c['blind']: args+=['--os-blind','--os-policy-tail','--os-policy-tail-blocks',str(c['blocks'])]
    if c.get('fit'): args+=['--os-fit-artifact',c['fit']]
    if c.get('replicate'): args+=['--os-rand-seed','20260927','--os-rand-replicate',str(c['replicate'])]
    opts,_=plugin.parse_cli(args);rt=plugin.install(opts,c['cell'].split('_')[0])
    reservations=[]
    class ReservationAudit:
        def __init__(self,inner):self.inner=inner;self.local=threading.local()
        def __enter__(self):
            self.inner.acquire()
            stack=getattr(self.local,'stack',[]);stack.append(rt.decision_count);self.local.stack=stack
            return self
        def __exit__(self,*exc):
            previous=self.local.stack.pop()
            if rt.decision_count!=previous:
                assert rt.decision_count==previous+1
                reservations.append(previous)
            self.inner.release()
        def _is_owned(self):return self.inner._is_owned()
    rt.decision_lock=ReservationAudit(rt.decision_lock)
    cfg=cc.load_cache_config(f"exp/trace_dual/config/tr_{rt.model}_{'l10' if rt.suite=='l10' else 'sp'}_cache.yaml")
    shared=cc.build_shared_storage(cfg);qc=store.QueryCell(opts.os_root,opts.os_cell)
    lock=threading.Lock();active=0;peak=0;windows=[]
    class Fake(selftest.FakePolicy):
        def stage1(self,obs):
            nonlocal active,peak
            if a.source!='before':
                assert not rt.decision_lock._is_owned(), 'global lock held across stage1'
            with lock: active+=1;peak=max(peak,active)
            try: time.sleep(.060)
            finally:
                with lock: active-=1
        def infer(self,obs):
            if a.source!='before': assert not rt.decision_lock._is_owned()
            self.stage1(obs)
            t=time.perf_counter()
            result=super().infer(obs)
            # Full-model stand-in delay on a MISS, while the connection remains busy.
            if result['hit_type']=='MISS': time.sleep(.025)
            with lock: windows.append((self.cid,t,time.perf_counter(),result['hit_type']))
            return result
        def _osp_prepare_blind(self,obs): return np.array(qc.rs[obs['_row']],copy=True),None
        def _osp_blind_output(self,action,state): return {'actions':action.copy()}
    def factory(_base,bundle_id='default'):
        comps=cc.build_per_connection_components(cfg,shared,quiet=True);kb=selftest.FakeKB(qc)
        for s in plugin._TLS.new_sessions: s.kb=kb
        orch=CacheOrchestrator(storage=comps['storage'],key_builder=kb,gates=comps['gates'],judges=comps['judges'],search_strategies=comps['search_strategies'],timer=comps['timer'],write_policy=comps.get('write_policy'),offline_writers=comps.get('offline_writers',()),library_stats=comps.get('library_stats'))
        f=Fake(orch,kb,np.asarray(qc.a_inf));f.cid=int(bundle_id);return f
    conns=[plugin._wrap_factory(factory)(None,str(i)) for i in range(8)]
    # Several different tasks, with unequal reset times so resets overlap live peers.
    selected=[qc.episodes[i] for i in selftest.pick_episodes(qc,8)]
    events=[]
    for i in range(8):
        seq=[]
        for episode,n in enumerate((10+i%3,12)):
            for step in range(n): seq.append([i,episode,step])
        events.append(seq)
    records={};started={};actions=[]
    def execute(event):
        i,episode,step=event;conn=conns[i];s=conn._osp_sessions[0]
        e=selected[(i+episode)%8]
        if step==0:
            if episode: conn.on_episode_end(success=False)
            conn.on_episode_start(task=e['task'],episode_id=e['init'],extra_metadata={'task_uid':f'c{i}:e{episode}','task_id':e['task_id'],'orig_init_state_idx':e['init']})
        # Repeated observations are a real guard trigger, without modifying any
        # method/verdict: exercise first AND third K5 landmarks in a short run.
        row=e['start']+(min(step,3) if rt.randomized else step)
        obs={'observation/state':np.asarray(qc.raw_state[row],np.float64),'prompt':e['task'],'_row':row,'__extra__':{'decision_id':step,'executed_steps':5}}
        output=conn.infer(obs)
        assert s.step==step+1==s.b_aex.n==len(s.hits)==len(s.has_vision)==conn.orch._step_counter
        assert len(conn.orch._state_history)==len(conn.orch._action_history)==step+1
        assert np.array_equal(s.b_aex.a[step],output['actions'])
        assert np.array_equal(s.b_rs.a[step],qc.rs[row])
        snap={name:digest(getattr(s,name).a[:getattr(s,name).n]) for name in ('b_v0','b_v1','b_rs','b_raw','b_aex')}
        snap.update(hits=list(s.hits),vision=list(s.has_vision),stage1_calls=s.stage1_calls,step=s.step,blind_age=s.blind_age,orch_states=digest(np.asarray([np.asarray(v) for v in conn.orch._state_history])),orch_actions=digest(np.asarray([np.asarray(v) for v in conn.orch._action_history])))
        if c['method'].endswith(':ProbeBlind'): snap.update(n=s.method.n,blind_calls=s.method.blind_calls)
        snap.update(anchor_count=s.method._anchor_count,anchor=clean(s.method._anchor) if s.method._anchor is not None else None,policy_cursor=None if s._policy_tail is None else s._policy_tail['cursor'])
        if snap['anchor'] is not None:
            snap['anchor']={k:(digest(v) if isinstance(v,np.ndarray) else v) for k,v in snap['anchor'].items()}
        if rt.randomized: snap.update(opportunities=s.randomization.opportunities,exposed=s.randomization.exposed)
        record=dict(event=event,index=s._decision_index,action=digest(output['actions']),verdict=output['hit_type'],history=snap,wire=clean({k:v for k,v in output.items() if k!='actions'}))
        records[tuple(event)]=record
    start=time.perf_counter()
    if a.mode=='threaded':
        barrier=threading.Barrier(8)
        def drive(i):
            barrier.wait()
            for event in events[i]: execute(event)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(drive,range(8)))
    else:
        schedule=json.loads(Path(a.schedule).read_text())['schedule']
        for event in schedule: execute(event)
    elapsed=time.perf_counter()-start
    for conn in conns: conn.on_episode_end(success=False)
    rows=[json.loads(l) for l in rt.dec_path.read_text().splitlines()]
    dec=[r for r in rows if r['ev']=='dec']
    assert len(dec)==len(records)==rt.decision_count
    assert reservations==list(range(rt.decision_count)), 'reservation order is not monotonic'
    assert sorted(r['decision_index'] for r in dec)==list(range(len(dec)))
    for i in range(8):
        indices=[r['decision_index'] for r in dec if r['conn']==i]
        assert indices==sorted(indices) and len(indices)==len(set(indices))
    assert all(r['stage1_calls']==records[(r['conn'],int(r['uid'].split('e')[-1]),r['step'])]['history']['stage1_calls'] for r in dec)
    verify_rc=verify_logs.main(['--log-dir',str(out),'--tag','concurrency','--work',str(out/'verify'),'--out',str(out/'verify.json')])
    assert verify_rc==0
    eligible={t:sum(r.get('eligible',False) and r['assigned_treatment']==t for r in dec) for t in ('CALL','CACHE')}
    if rt.randomized: assert all(eligible.values()),eligible
    result=dict(PASS=True,config=a.config,mode=a.mode,source=a.source,seconds=elapsed,peak_inference=peak,decisions=len(dec),vision=sum(r['vision'] for r in dec),miss=sum(not r['hit'] for r in dec),blind=sum(not r['vision'] for r in dec),policy_tails=sum(r['src']=='policy_tail' for r in dec),eligible=eligible,records=sorted(records.values(),key=lambda r:r['event']),rows=sorted([clean(r) for r in dec],key=lambda r:(r['conn'],r['uid'],r['step'])),schedule=[r['event'] for r in sorted(records.values(),key=lambda r:r['index'])])
    (out/'concurrency.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','rows','schedule')}),flush=True)


def matrix(a):
    import subprocess
    names=list(configs()) if a.config=='all' else a.config.split(',')
    result=[]
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    for name in names:
        reports={}
        for mode,source in [('threaded',a.source),('serialized',a.source)]:
            out=root/name/mode;out.parent.mkdir(exist_ok=True)
            cmd=PREFIX+[str(Path(__file__).resolve()),'--worker','--source',source,'--config',name,'--mode',mode,'--out',str(out)]
            if mode=='serialized': cmd+=['--schedule',str(root/name/'threaded/concurrency.json')]
            with (root/name/(mode+'.log')).open('w') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            reports[mode]=json.loads((out/'concurrency.json').read_text())
        t,b=reports['threaded'],reports['serialized']
        assert t['records']==b['records'],f'{name}: actions/verdicts/history mismatch'
        assert t['rows']==b['rows'],f'{name}: decision row mismatch'
        speedup=b['seconds']/t['seconds']
        assert t['peak_inference']>=4 and b['peak_inference']==1
        assert speedup>=3.0,(name,speedup)
        row=dict(config=name,PASS=True,policy_tails=t['policy_tails'],connections=8,decisions=t['decisions'],vision=t['vision'],blind=t['blind'],miss=t['miss'],eligible=t['eligible'],serialized_s=b['seconds'],threaded_s=t['seconds'],speedup=speedup,peak=t['peak_inference'],all_rows_actions_verdicts_histories_equal=True)
        result.append(row);print(json.dumps(row),flush=True)
        (root/'summary.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='dev');p.add_argument('--config',default='all');p.add_argument('--out',required=True);p.add_argument('--mode',default='threaded');p.add_argument('--schedule');p.add_argument('--worker',action='store_true');a=p.parse_args()
    worker(a) if a.worker else matrix(a)
