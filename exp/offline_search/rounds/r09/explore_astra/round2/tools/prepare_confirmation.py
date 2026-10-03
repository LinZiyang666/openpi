"""Emit frozen models/arms on EVAL inits 20..29 only. No remote launch."""
import copy
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .data import HERE, RUN, STORE, dump, sha
from ..methods import SharedResidual
from ..inference import predict, inputs, correct
from exp.offline_search.harness import api
from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.closed_loop.ops.emit_arms import main as emit

OUT=Path('/home/weiland/trace_runs/os_closed_loop/r09_astra_round2_eval')
MODULE='exp.offline_search.rounds.r09.explore_astra.round2.methods:SharedResidual'


def source_fit(src):
    args=src['plugin_args'];return args[args.index('--os-fit-artifact')+1]


def spec(src,name,method=None,kwargs=None,path=None):
    args=[x for x in src['plugin_args'] if x!='--os-debug']
    if path:args[args.index('--os-fit-artifact')+1]=str(path)
    out=dict(name=name,model=src['model'],suite=src['suite_short'],mode='plugin',
        method=method or src['method'],kwargs=src['kwargs'] if kwargs is None else kwargs,
        plugin_args=args,full_model=bool(src.get('full_model',False)),cost_ledger=True,
        client_overrides=src['client_overrides'],manifest=str(OUT/'manifests/eval100.json'))
    if src.get('server_env'):out['server_env']=src['server_env']
    return out


def verify(method,cell):
    cache=method._cache();model,suite,_=cell.split('_')
    lib=STORE/'library'/f'{model}_{suite}'/cache.cand_name
    k0=np.load(lib/'key_v0.npy',mmap_mode='r');k1=np.load(lib/'key_v1.npy',mmap_mode='r');rs=np.load(lib/'rs.npy',mmap_mode='r')
    errors=[];times=[]
    for task in range(10):
        rows=cache.tasks[task].rows
        ep=SimpleNamespace(uid=f'r9r2_library_probe:{task}',task_id=task,init=20)
        reference,_=clone_method(method.base,strict=True);reference.reset(ep)
        method.reset(ep)
        for j,step in enumerate([0,2]):
            row=int(rows[min(j,len(rows)-1)])
            q=SimpleNamespace(key_v0=k0[row],key_v1=k1[row],rs=rs[row],task_id=task,step=step,episode=ep,prev_hit=True,
                hist_key_v0=np.array([k0[row]]),hist_key_v1=np.array([k1[row]]),hist_has_vision=np.array([False]),
                hist_rs=np.array([rs[row]]))
            mode='full' if step==0 else 'wrist_only'
            if method.wrist:method.set_camera_mode(mode);reference.set_camera_mode(mode)
            original=reference.query(q).action
            vis1=cache.B1T@k1[row]-cache.muB1
            visual=vis1 if method.wrist else np.r_[cache.B0T@k0[row]-cache.muB0,vis1]
            x=inputs(visual,rs[row],original,step,method._previous,method.history)
            expected=correct(method.head,x,original[None],method.blend,method.gripper,method.gripper_threshold)[0]
            start=time.perf_counter();got=method.query(q);times.append((time.perf_counter()-start)*1000)
            np.testing.assert_allclose(got.action,expected,atol=1e-6,rtol=1e-6)
            if not method.gripper:np.testing.assert_array_equal(got.action[:,6],original[:,6])
            if method.blend==0:np.testing.assert_array_equal(got.action[:,:6],original[:,:6])
            np.testing.assert_array_equal(got.action[:,7:],original[:,7:])
            bq=SimpleNamespace(task_id=task,step=step+1,episode=ep,prev_hit=True,blind_age=0,executed_steps=5,
                rs=rs[row],hist_rs=np.array([rs[row]]),prev_a_exec=got.action[:5])
            tail=method.blind_step(bq)
            np.testing.assert_array_equal(tail.action[:5,:7],got.action[5:10,:7])
            clone,_=clone_method(method,strict=True);clone.invalidate_anchor()
            assert method._cache()._anchor is not None and method._previous is not None
            errors.append(float(np.max(np.abs(got.action-expected))))
        method.reset(ep);assert method._previous is None
    # Zero correction exact identity, including gripper and neighbors.
    zero,_=clone_method(method,strict=True);zero.blend=0;zero.gripper=False
    zero.reset(ep);reference.reset(ep)
    if method.wrist:zero.set_camera_mode('full');reference.set_camera_mode('full')
    q.step=0
    a=zero.query(q);b=reference.query(q)
    np.testing.assert_array_equal(a.action,b.action);np.testing.assert_array_equal(a.topk,b.topk)
    x=np.zeros((1,len(method.head['mean'])),np.float32);head_ms=[]
    for _ in range(100):
        start=time.perf_counter_ns();predict(method.head,x);head_ms.append((time.perf_counter_ns()-start)/1e6)
    method.invalidate_anchor()
    return dict(queries=len(errors),max_parity_error=max(errors),correct_tail=True,
        zero_identity=True,clone_isolation=True,reset_history=True,
        head_ms_p50=float(np.median(head_ms)),head_ms_p95=float(np.quantile(head_ms,.95)),
        query_ms_p50=float(np.median(times)))


def main():
    sources=json.loads((RUN/'arms.json').read_text());arms=[];checks=[];frozen=[]
    dump(OUT/'manifests/eval100.json',dict(role='EVAL_DISJOINT_FROM_FIT_0_19',selected=[dict(task=t,init=i) for t in range(10) for i in range(20,30)]))
    for src in sources:
        variant=src['r8']['variant']
        if variant not in ['A','W10','P10']:continue
        if variant=='P10':
            arms.append(spec(src,src['arm'].replace('r8_','r9r2_astra_')));continue
        cell=f"{src['model']}_{src['suite_short']}_{src['r8']['library_size']}"
        wrist=variant=='W10'
        arms.append(spec(src,f'r9r2_astra_{cell}_'+('wrist_cache' if wrist else 'cache')))
        candidates=[('wrist',.5,False)] if wrist else [('motion',.5,False),('joint',.5,True),('grip',0.,True)]
        head=HERE/'artifacts'/f"{cell}_{'wrist' if wrist else 'temporal'}.npz"
        for suffix,blend,grip in candidates:
            kw=dict(base_fit=source_fit(src),base_spec=src['method'],base_kwargs=src['kwargs'],
                head_path=str(head),history=not wrist,blend=blend,gripper=grip,gripper_threshold=.8,wrist=wrist)
            method=SharedResidual(**kw);method.prof=api.NULL_PROFILER;method.fit(None,SimpleNamespace(cell=src['cell']))
            name=f'r9r2_astra_{cell}_{suffix}'
            check=verify(method,src['cell']);checks.append(dict(arm=name,**check))
            path=HERE/'artifacts'/f'{name}.pkl'
            with open(source_fit(src),'rb') as f:base=FitUnpickler(f).load()
            blob=dict(method=method,registered=base.get('registered',{}),spec=MODULE,kwargs=kw,cell=src['cell'],fit_s=0,
                provenance=dict(fit_inits=list(range(20)),evaluation_inits=list(range(20,30)),
                    head_sha256=sha(head),method_sha256=sha(HERE/'methods.py'),inference_sha256=sha(HERE/'inference.py')))
            with path.open('wb') as f:pickle.dump(blob,f,protocol=4)
            arms.append(spec(src,name,MODULE,kw,path))
            frozen.append(dict(arm=name,kwargs=kw,artifact=str(path),artifact_sha256=sha(path),head_sha256=sha(head)))
            print(name,check,flush=True)
    dump(HERE/'confirmation_specs.json',arms)
    dump(HERE/'results/deployment_checks.json',checks)
    dump(HERE/'FROZEN_CANDIDATES.json',dict(fit_inits=list(range(20)),evaluation_inits=list(range(20,30)),
        no_task_switches=True,candidates=frozen))
    emit(['--run-root',str(OUT),'--spec',str(HERE/'confirmation_specs.json')])


if __name__=='__main__':main()
