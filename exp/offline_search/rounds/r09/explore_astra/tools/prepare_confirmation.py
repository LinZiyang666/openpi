"""Freeze reviewable discovery confirmation arms and test real cache/tail contracts.

Only local artifacts/specs are emitted. No server, worker, sync or remote call.
"""
import copy
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .common import HERE,RUN,STORE,dump,sha
from ..methods import ResidualCache,EpisodePolicyLottery,TenStepMetricCache
from ..inference import predict
from exp.offline_search.harness import api
from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.closed_loop.ops.emit_arms import main as emit

OUT=Path('/home/weiland/trace_runs/os_closed_loop/r09_astra_confirmation')
MODULE='exp.offline_search.rounds.r09.explore_astra.methods'


def publish(name,method,spec,kwargs,source):
    path=HERE/'artifacts'/f'{name}.pkl'
    blob=dict(method=method,registered=source.get('registered',{}),spec=spec,kwargs=kwargs,cell=source['cell'],
              fit_s=0,provenance=dict(training_split='R8 discovery inits 0..29',
                                     method_sha256=sha(HERE/'methods.py'),head_sha256=sha(HERE/'inference.py')))
    with path.open('wb') as f:pickle.dump(blob,f,protocol=4)
    return str(path)


def arm_spec(source,name,method=None,kwargs=None,artifact=None):
    row=dict(name=name,model=source['model'],suite=source['suite_short'],mode='plugin',
             method=method or source['method'],kwargs=kwargs if kwargs is not None else source['kwargs'],
             full_model=bool(source.get('full_model',False)),cost_ledger=True,
             client_overrides=source['client_overrides'],manifest=str(OUT/'manifests/discovery300.json'))
    args=list(source['plugin_args'])
    # Use standard capture mode; copied kwargs/cadence unchanged.
    for flag in ['--os-debug']:
        if flag in args:args.remove(flag)
    if artifact:
        j=args.index('--os-fit-artifact');args[j+1]=artifact
    row['plugin_args']=args
    return row


def verify(method,base,cell):
    """Real library queries: zero-blend identity; gripper invariant; correct blind tail."""
    model,suite,_=cell.split('_')
    lib=STORE/'library'/f'{model}_{suite}'/base.cand_name
    k0=np.load(lib/'key_v0.npy',mmap_mode='r');k1=np.load(lib/'key_v1.npy',mmap_mode='r');rs=np.load(lib/'rs.npy',mmap_mode='r')
    maximum=0.;latencies=[]
    for task in range(10):
        r=int(base.tasks[task].rows[0]);ep=SimpleNamespace(uid=f'r09_local:eval:{task}:0',task_id=task,init=0)
        q=SimpleNamespace(key_v0=k0[r],key_v1=k1[r],rs=rs[r],task_id=task,step=0,episode=ep,prev_hit=None)
        zero,_=clone_method(method,strict=True);zero.blend=0;zero.reset(ep)
        original,_=clone_method(base,strict=True);original.reset(ep)
        a=zero.query(q);b=original.query(q)
        np.testing.assert_array_equal(a.action,b.action);np.testing.assert_array_equal(a.topk,b.topk)
        method.reset(ep);start=time.perf_counter();c=method.query(q);latencies.append(time.perf_counter()-start)
        np.testing.assert_array_equal(c.action[:,6],b.action[:,6])
        np.testing.assert_array_equal(c.action[:,7:],b.action[:,7:])
        bq=SimpleNamespace(task_id=task,step=1,episode=ep,prev_hit=True,blind_age=0,executed_steps=5,rs=rs[r],hist_rs=np.array([rs[r]]))
        tail=method.blind_step(bq)
        np.testing.assert_array_equal(tail.action[:5,:7],c.action[5:10,:7])
        # Clones own their mutable anchor, share only frozen fitted arrays.
        clone,_=clone_method(method,strict=True)
        clone.invalidate_anchor()
        assert method.base._anchor is not None
        maximum=max(maximum,float(np.max(np.abs(c.action-b.action))))
    # CPU head-only microbenchmark; re-projection costs are measured separately.
    x=np.zeros((1,207),np.float32)
    # n_features is 128+8+70+1 = 207.
    m=method.models[0]
    for _ in range(10):predict(m,x)
    times=[]
    for _ in range(300):
        start=time.perf_counter_ns();predict(m,x);times.append((time.perf_counter_ns()-start)/1e6)
    return dict(tasks=10,zero_blend_identity=True,gripper_padding_unchanged=True,corrected_tail_exact=True,
                clone_state_isolation=True,max_action_correction=maximum,
                head_cpu_ms_p50=float(np.median(times)),head_cpu_ms_p95=float(np.quantile(times,.95)),
                full_query_cpu_ms_median=float(np.median(latencies)*1000))


def main():
    specs=json.loads((RUN/'arms.json').read_text());arms=[];checks=[]
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'manifests/discovery300.json',dict(role='DISCOVERY_ONLY',selected=[dict(task=t,init=i) for t in range(10) for i in range(30)]))
    for src in specs:
        if src['r8']['variant']!='A':continue
        cell=f"{src['model']}_{src['suite_short']}_{src['r8']['library_size']}"
        with open(src['r8']['source']['artifact'],'rb') as f:blob=FitUnpickler(f).load()
        kwargs=dict(base_fit=src['r8']['source']['artifact'],base_kwargs=src['kwargs'],
                    student_path=str(HERE/'artifacts'/f"{src['arm']}_student.npz"),blend=.5)
        method=ResidualCache(**kwargs);method.prof=api.NULL_PROFILER;method.fit(None,SimpleNamespace(cell=src['cell']))
        check=verify(method,blob['method'],src['cell']);check['cell']=cell;checks.append(check)
        method.base.invalidate_anchor()
        name=f'r9_astra_{cell}_residual_half'
        artifact=publish(name,method,MODULE+':ResidualCache',kwargs,blob)
        arms.append(arm_spec(src,name,MODULE+':ResidualCache',kwargs,artifact))
        arms.append(arm_spec(src,f'r9_astra_{cell}_cache'))
        if src['r8']['library_size']==50:
            metric_kwargs=dict(base_fit=src['r8']['source']['artifact'],base_kwargs=src['kwargs'])
            metric=TenStepMetricCache(**metric_kwargs);metric.prof=api.NULL_PROFILER
            metric.fit(None,SimpleNamespace(cell=src['cell']))
            metric_name=f'r9_astra_{cell}_metric10'
            metric_path=publish(metric_name,metric,MODULE+':TenStepMetricCache',metric_kwargs,blob)
            arms.append(arm_spec(src,metric_name,MODULE+':TenStepMetricCache',metric_kwargs,metric_path))
        # Episode policy lottery is a comparator, not the primary recommendation.
        rr=json.loads((HERE/'results/routing.json').read_text())
        rule=next(r for r in rr if r['cell']==cell and r['menu']=='binary' and r['prior']==12 and r['tolerance']==.01)
        kw=dict(base_fit=src['r8']['source']['artifact'],base_kwargs=src['kwargs'],
                task_probabilities=[p[0] for p in rule['full_probabilities']],random_seed=20261001,
                coin_domain='R9/astra/episode/'+cell)
        lottery=EpisodePolicyLottery(**kw);lottery.prof=api.NULL_PROFILER;lottery.fit(None,SimpleNamespace(cell=src['cell']))
        name=f'r9_astra_{cell}_episode'
        artifact=publish(name,lottery,MODULE+':EpisodePolicyLottery',kw,blob)
        row=arm_spec(src,name,MODULE+':EpisodePolicyLottery',kw,artifact);row['full_model']=True
        row['plugin_args']+=['--os-judge','guard_only','--os-policy-tail','--os-policy-tail-blocks','1']
        arms.append(row)
        print(cell,check,flush=True)
    for src in specs:
        v=src['r8']['variant']
        if v=='P10' or (src['model']=='pi05' and src['suite_short']=='spatial' and src['r8']['library_size']==500 and v in ['W10','SW']):
            name=src['arm'].replace('r8_','r9_astra_')
            arms.append(arm_spec(src,name))
    dump(HERE/'confirmation_specs.json',arms)
    dump(HERE/'results/deployment_checks.json',checks)
    emit(['--run-root',str(OUT),'--spec',str(HERE/'confirmation_specs.json')])


if __name__=='__main__':main()
