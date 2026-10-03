"""Freeze 100-pair eval-only arms, validate real library queries; NEVER launch."""
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.closed_loop.blind import policy_tail_chunk,BlindResult,LookReason
from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from ...round2.inference import predict
from ..methods import TransitionRecovery
from ..inference import Monitor
from .data import HERE,RUN,CELLS,library,dump,sha

OUT=Path('/home/weiland/trace_runs/os_closed_loop/r09_astra_round2b')
MODULE='exp.offline_search.rounds.r09.explore_astra.round2b.methods:TransitionRecovery'


def source_fit(src):
    args=src['plugin_args'];return args[args.index('--os-fit-artifact')+1]


def spec(src,name,method=None,kwargs=None,path=None):
    args=[x for x in src['plugin_args'] if x!='--os-debug']
    if path:
        args[args.index('--os-fit-artifact')+1]=str(path)
        args+=['--os-policy-tail','--os-policy-tail-blocks','1','--os-judge','guard_only']
    out=dict(name=name,model=src['model'],suite=src['suite_short'],mode='plugin',method=method or src['method'],
        kwargs=src['kwargs'] if kwargs is None else kwargs,plugin_args=args,
        full_model=True if path else bool(src.get('full_model',False)),cost_ledger=True,
        client_overrides=src['client_overrides'],manifest=str(OUT/'manifests/eval100.json'))
    if src.get('server_env'):out['server_env']=src['server_env']
    return out


def verify(method,cell):
    lib=library(cell);keys=[np.load(lib/f'key_v{i}.npy',mmap_mode='r') for i in [0,1]]
    states=np.load(lib/'rs.npy',mmap_mode='r');queries=0;tails=0;policytails=0;errors=[]
    for forced in [False,True]:
        tasks=[0] if forced else range(10)
        for task in tasks:
            ep=SimpleNamespace(uid=f'r9r2b_library_probe:{task}:{forced}',task_id=task,init=20)
            method.reset(ep);reference,_=clone_method(method.base,strict=True);reference.reset(ep)
            monitor=Monitor();history=[];hits=[];vision=[];statehist=[];k0hist=[];k1hist=[]
            if forced:
                if method.response=='random3':method.gate.probability=1.
                else:method.gate.threshold=-1e6
            for j in range(40 if forced else 3):
                step=2*j;row=int(method.base.tasks[task].rows[min(j,len(method.base.tasks[task].rows)-1)])
                rs=states[row]
                q=SimpleNamespace(task_id=task,step=step,episode=ep,key_v0=keys[0][row],key_v1=keys[1][row],rs=rs,
                    prev_hit=bool(hits[-1]) if hits else None,hist_rs=np.array(statehist),
                    hist_key_v0=np.array(k0hist),hist_key_v1=np.array(k1hist),hist_has_vision=np.array(vision),
                    hist_hit=np.array(hits),hist_a_exec=np.array(history))
                ref=reference.query(q);a=reference._anchor
                vis=np.r_[reference.B0T@q.key_v0-reference.muB0,reference.B1T@q.key_v1-reference.muB1]
                x=monitor.observe(vis,rs,ref.action,a['rows'],a['weights'],method.phase[a['rows']],step,ref.extras['d1_rel'])
                expected=float(predict(method.head,x[None])[0,0])
                got=method.query(q);queries+=1
                np.testing.assert_array_equal(got.action,ref.action)
                np.testing.assert_allclose(got.extras['r9b_score'],expected,atol=1e-6,rtol=1e-6)
                errors.append(abs(got.extras['r9b_score']-expected))
                call=bool(got.extras['os_force_miss'])
                served=got.action.copy()
                if call:served[:,:6]+=.0123  # synthetic full-policy chunk to distinguish source
                history.append(served);hits.append(0 if call else 1);vision.append(True);statehist.append(rs)
                k0hist.append(q.key_v0);k1hist.append(q.key_v1)
                if call:
                    before=len(method.monitor.history);used=method.gate.used;method.invalidate_anchor()
                    assert len(method.monitor.history)==before and method.gate.used==used
                bq=SimpleNamespace(task_id=task,step=step+1,episode=ep,rs=rs,raw_state=rs,prev_hit=not call,
                    prev_a_exec=served,hist_rs=np.array(statehist),hist_a_exec=np.array(history),hist_hit=np.array(hits),
                    hist_has_vision=np.array(vision),blind_age=0,executed_steps=5)
                tail=method.policy_tail_step(bq) if call else method.blind_step(bq)
                assert isinstance(tail,BlindResult)
                # Cache blind tails intentionally zero unused padding channels.
                np.testing.assert_array_equal(tail.action[:5,:7],served[5:10,:7]);tails+=1
                if call:
                    np.testing.assert_array_equal(tail.action,policy_tail_chunk(served));policytails+=1
                    assert isinstance(method.policy_tail_step(bq),LookReason)  # consume exactly once
                history.append(tail.action);hits.append(1);vision.append(False);statehist.append(rs)
                k0hist.append(q.key_v0);k1hist.append(q.key_v1)
            if forced:assert 0<method.gate.used<=12
            clone,_=clone_method(method,strict=True);old_used=method.gate.used
            clone.reset(ep);assert method.gate.used==old_used and len(method.monitor.history)>0
    method.reset(ep);assert method.gate.used==0 and len(method.monitor.history)==0
    xx=np.zeros((1,len(method.head['mean'])),np.float32);ms=[]
    for _ in range(100):
        start=time.perf_counter_ns();predict(method.head,xx);ms.append((time.perf_counter_ns()-start)/1e6)
    return dict(queries=queries,tail_checks=tails,policy_tail_checks=policytails,score_max_error=max(errors),
        clone_isolation=True,reset=True,cap=True,predict_ms_p50=float(np.median(ms)))


def main():
    sources=json.loads((RUN/'arms.json').read_text());arms=[];frozen=[];checks=[]
    dump(OUT/'manifests/eval100.json',dict(role='EVAL_ONLY_DISJOINT_FIT_0_19',
        selected=[dict(task=t,init=i) for t in range(10) for i in range(20,30)]))
    for src in sources:
        if src['suite_short']!='l10':continue
        variant=src['r8']['variant']
        if variant=='P10':arms.append(spec(src,src['arm'].replace('r8_','r9r2b_astra_')));continue
        if variant!='A':continue
        cell=f"{src['model']}_l10_{src['r8']['library_size']}"
        if cell not in CELLS:continue
        arms.append(spec(src,f'r9r2b_astra_{cell}_cache'))
        gate=json.loads((HERE/'artifacts'/f'{cell}_response.json').read_text())
        for response in ['burst1','burst3','latch','random3']:
            kw=dict(base_fit=source_fit(src),base_spec=src['method'],base_kwargs=src['kwargs'],
                head_path=str(HERE/'artifacts'/f'{cell}_monitor.npz'),phase_path=str(HERE/'artifacts'/f'{cell}_phase.npy'),
                threshold=gate['threshold'],response=response,random_probability=gate['random_probability'],random_seed=20261002)
            method=TransitionRecovery(**kw);method.prof=api.NULL_PROFILER;method.fit(None,SimpleNamespace(cell=src['cell']))
            name=f'r9r2b_astra_{cell}_{response}';check=verify(method,cell);checks.append(dict(arm=name,**check))
            path=HERE/'artifacts'/f'{name}.pkl'
            with open(source_fit(src),'rb') as f:base=FitUnpickler(f).load()
            blob=dict(method=method,registered=base.get('registered',{}),spec=MODULE,kwargs=kw,cell=src['cell'],fit_s=0,
                provenance=dict(fit_inits=list(range(20)),eval_inits=list(range(20,30)),
                    head_sha256=sha(kw['head_path']),phase_sha256=sha(kw['phase_path']),
                    method_sha256=sha(HERE/'methods.py'),inference_sha256=sha(HERE/'inference.py')))
            with path.open('wb') as f:pickle.dump(blob,f,protocol=4)
            arms.append(spec(src,name,MODULE,kw,path))
            frozen.append(dict(arm=name,kwargs=kw,artifact=str(path),artifact_sha256=sha(path),
                head_sha256=sha(kw['head_path']),phase_sha256=sha(kw['phase_path'])))
            print(name,check,flush=True)
    dump(HERE/'confirmation_specs.json',arms)
    dump(HERE/'results/deployment_checks.json',checks)
    dump(HERE/'FROZEN_CANDIDATES.json',dict(fit_inits=list(range(20)),eval_inits=list(range(20,30)),
        no_task_switches=True,no_predicate_inputs=True,candidates=frozen))
    emit(['--run-root',str(OUT),'--spec',str(HERE/'confirmation_specs.json')])


if __name__=='__main__':main()
