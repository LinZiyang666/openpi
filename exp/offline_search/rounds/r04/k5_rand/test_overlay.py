"""Assignment, lifecycle and real-orchestrator injection checks, CPU only."""
import argparse
from collections import Counter
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from exp.offline_search.rounds.r04.k5_rand.overlay import assignment,RandomizedLandmark
BASE=Path(__file__).resolve().parent
PREFIX=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',sys.executable]
SEED=20260927

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',default='installed'); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.source!='installed':
        for name in ('plugin','verify_logs','selftest'):
            spec=importlib.util.spec_from_file_location('exp.offline_search.closed_loop.'+name,BASE/a.source/(name+'.py'))
            mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
            if name=='plugin': mod.REPO=BASE.parents[4]
    from exp.offline_search.closed_loop import plugin,selftest
    pairs=[(t,i) for t in range(10) for i in range(50)]
    truth=[assignment(SEED,t,i,1) for t,i in pairs]
    counts=Counter((r['landmark_class'],r['assigned_treatment']) for r in truth)
    for t,i in reversed(pairs):
        x,y=assignment(SEED,t,i,1),assignment(SEED,t,i,2)
        assert x['landmark_class']==y['landmark_class'] and x['assigned_treatment']!=y['assigned_treatment']
    code="import json; from exp.offline_search.rounds.r04.k5_rand.overlay import assignment; print(json.dumps([assignment(20260927,t,i,1) for t in range(10) for i in range(50)]))"
    for hs in ('1','793','random'):
        got=json.loads(subprocess.check_output(PREFIX+['-c',code],env={**os.environ,'PYTHONHASHSEED':hs}))
        assert got==truth
    shuffled=list(reversed(pairs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda p:assignment(SEED,*p,1),shuffled))==list(reversed(truth))
    base=['--os-method','exp/offline_search/rounds/r03/h3_judge/judge.py:MixedJudge','--os-kwargs',json.dumps(dict(base='exp/offline_search/rounds/r02/g1_awm/awm.py:AWM',base_kwargs=dict(lib='current',kref=5),guards=True,events='none')),
          '--os-cell','pi05_l10_cache','--os-root','/home/weiland/trace_runs/offline_search_store','--os-log-dir',str(a.out),'--os-tag','injection','--os-judge','guard_only']
    rejected=0
    for extra in (['--os-rand-seed','0'],['--os-rand-replicate','1'],['--os-rand-seed','0','--os-rand-replicate','1','--os-blind'],
                  ['--os-rand-seed','0','--os-rand-replicate','1','--os-judge-cap','1'],
                  ['--os-rand-seed','0','--os-rand-replicate','1','--os-judge-step0','hit'],
                  ['--os-rand-seed','0','--os-rand-replicate','1','--os-judge-burst','2']):
        try: plugin.parse_cli(base+extra)
        except SystemExit: rejected+=1
        else: raise AssertionError(extra)
    # Boundary bins are exact; step-zero previous gripper is unknown, never invented.
    ov=RandomizedLandmark(SEED,0,0,1)
    pa=np.zeros((5,7),np.float32); pa[4,6]=-.2
    _,rec=ov.apply(0,False,-.3,{'top1_prog':.5,'stuck_n':2},None)
    assert rec['context']==dict(progress='>=0.5',confidence='<=-0.3',gripper=None,stall_age='0-9')
    _,rec=ov.apply(10,True,-.299,{'top1_prog':.499},pa)
    assert rec['context']==dict(progress='<0.5',confidence='>-0.3',gripper='open',stall_age='>=10')
    opts,_=plugin.parse_cli(base+['--os-rand-seed',str(SEED),'--os-rand-replicate','1','--os-fit-artifact','/home/weiland/trace_runs/os_closed_loop/r03_mx/fits/r3mx_p_l10_g.pkl'])
    rt=plugin.install(opts,'pi05')
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator
    from exp.offline_search.harness import store,api
    qc=store.QueryCell(opts.os_root,opts.os_cell)
    cfg=cc.load_cache_config('exp/trace_dual/config/tr_pi05_l10_cache.yaml'); shared=cc.build_shared_storage(cfg)
    class Fake(selftest.FakePolicy):
        def stage1(self,obs): pass
        def infer(self,obs): self.stage1(obs); return super().infer(obs)
        def _osp_prepare_blind(self,obs): return None
        def _osp_blind_output(self,action,state): return {'actions':action}
    def factory(_base,bundle_id='default'):
        comps=cc.build_per_connection_components(cfg,shared,quiet=True); kb=selftest.FakeKB(qc)
        for s in plugin._TLS.new_sessions: s.kb=kb
        orch=CacheOrchestrator(storage=comps['storage'],key_builder=kb,gates=comps['gates'],judges=comps['judges'],search_strategies=comps['search_strategies'],timer=comps['timer'],write_policy=comps.get('write_policy'),offline_writers=comps.get('offline_writers',()),library_stats=comps.get('library_stats'))
        return Fake(orch,kb,np.asarray(qc.a_inf))
    selected=[]
    for lm,tr in ((1,'CALL'),(1,'CACHE'),(3,'CALL'),(3,'CACHE')):
        selected.append(next(r for r in truth if r['landmark_class']==lm and r['assigned_treatment']==tr))
    conns=[plugin._wrap_factory(factory)(None,str(i)) for i in range(4)]
    expected=[[] for _ in conns]; actions=[[] for _ in conns]; checks=0; elig=0
    for i,c in enumerate(conns):
        s=c._osp_sessions[0]
        original=s.method.query
        def query(q,i=i,original=original):
            nonlocal checks
            assert list(q.hist_hit)==expected[i]
            assert q.prev_hit==(bool(expected[i][-1]) if expected[i] else None)
            if actions[i]: assert np.array_equal(q.prev_a_exec,actions[i][-1])
            checks+=1
            res=original(q)
            # Injection tests step zero and each guard reason, while preserving ordinary proposal.
            ex=dict(res.extras); force=q.step%2==0
            ex.update(os_force_miss=float(force),os_reason=float(q.step//2%4+1) if force else 0.)
            return api.Result(topk=res.topk,scores=res.scores,confidence=res.confidence,action=res.action,library=res.library,extras=ex)
        s.method.query=query
    exposure_count=[]
    for restart,nsteps in enumerate((12,2,12)):
        for i,c in enumerate(conns):
            r=selected[i]; e=next(e for e in qc.episodes if e['task_id']==r['task_id'])
            expected[i]=[]; actions[i]=[]
            c.on_episode_start(task=e['task'],episode_id=9999,extra_metadata={'task_id':r['task_id'],'orig_init_state_idx':r['init'],'task_uid':f'injected:{i}:{restart}','attempt':restart+1})
        for step in range(nsteps):
            # Reverse connection arrival on alternate rounds.
            for i in (range(4) if restart%2==0 else reversed(range(4))):
                c=conns[i]; r=selected[i]; s=c._osp_sessions[0]
                e=next(e for e in qc.episodes if e['task_id']==r['task_id']); row=e['start']+step
                result=c.infer({'observation/state':np.asarray(qc.raw_state[row],np.float64),'_row':row})
                hit=not(step%2==0)
                intervention=(step==2*(r['landmark_class']-1))
                if intervention: hit=r['assigned_treatment']=='CACHE'; elig+=1
                expected[i].append(int(hit)); actions[i].append(result['actions'])
                assert result['hit_type']==('FULL_HIT' if hit else 'MISS')
                assert s.hits==expected[i] and s.b_aex.n==step+1
                assert np.array_equal(s.b_aex.a[step],result['actions'])
                if not hit: assert np.array_equal(result['actions'],qc.a_inf[row])
        for c in conns:
            s=c._osp_sessions[0]; exposure_count.append(s.randomization.exposed)
            c.on_episode_end(success=False)
    # Missing identity must fail before method query; no connection/order fallback.
    c=conns[0]; c.on_episode_start(task=e['task'],episode_id=0)
    try: c.infer({'observation/state':np.asarray(qc.raw_state[e['start']],np.float64),'_row':e['start']})
    except ValueError as exc: assert 'orig_init_state_idx' in str(exc)
    else: raise AssertionError('missing init accepted')
    # Avoid emitting a failed empty episode during process cleanup.
    c._osp_sessions[0].ep=None
    rows=[json.loads(l) for l in rt.dec_path.read_text().splitlines()]
    dec=[r for r in rows if r['ev']=='dec']
    assert len(dec)==104 and sum(r['eligible'] for r in dec)==10
    assert sum(exposure_count)==10 and elig==10
    assert all(r['served_head'] is not None and r['vision'] for r in dec)
    assert all(r['exec_ok'] for r in dec if r['hit'])
    for r in dec:
        if r['eligible']: assert r['opportunity_index']==r['landmark_class'] and r['baseline_verdict']=='MISS'
        else: assert r['baseline_verdict']==r['actual_verdict']
    result=dict(PASS=True,assignment_pairs=500,cross_process_checks=3,replicate_complements=500,
                balance={f'{k[0]}_{k[1]}':v for k,v in counts.items()},invalid_options_rejected=rejected,
                concurrent_connections=4,episodes=12,decisions=len(dec),truthful_history_checks=checks,
                exposed=10,unexposed=2,missing_init_rejected=True)
    (BASE/'results'/f'overlay_{a.source}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__': main()
