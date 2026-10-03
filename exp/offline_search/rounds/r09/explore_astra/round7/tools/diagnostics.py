"""Frozen candidate strata and local CPU correction timing; no refitting."""
import json,time,os
from types import SimpleNamespace
import numpy as np
from .data import *
from .numeric import predict,confidence
from .test_tools import load
from exp.offline_search.rounds.r09.explore_astra.round6.tools.learn import library,load_control,control_prediction


def main():
    rows=json.loads((RUN/'arms.json').read_text()); result={}
    for cell in CELLS:
        d=annotated(cell,'A','eval'); lib=library(cell); ctrl=control_prediction(load_control(cell),d)
        ce=((ctrl-d['teacher'][:,:,:6])**2).mean((1,2)); result[cell]={}
        for variant in ('capacity','confidence'):
            b=load(artifact(next(r for r in rows if r['arm']==f'r9a7_{cell}_{variant}')))['method'].base
            out=d['base'][:,:,:6]+predict(b.ar7_head,b.ar7_table,d['x'],d['rows'],d['weights'],lib,b.ar7_meta)
            err=((out-d['teacher'][:,:,:6])**2).mean((1,2))
            def group(take): return dict(n=int(take.sum()),mse=float(err[take].mean()),delta=float((err-ce)[take].mean()))
            q=SimpleNamespace(key_v0=np.zeros(b.B0T.shape[1],np.float32),key_v1=np.zeros(b.B1T.shape[1],np.float32),rs=d['x'][0,128:136],step=int(d['seq'][0]))
            a=np.einsum('k,kha->ha',d['weights'][0],lib[d['rows'][0]]).astype(np.float32)
            for _ in range(30): b._ar7_action(q,a,d['rows'][0],d['weights'][0])
            times=[]
            for _ in range(300):
                start=time.perf_counter_ns();b._ar7_action(q,a,d['rows'][0],d['weights'][0]);times.append((time.perf_counter_ns()-start)/1e6)
            strength=.5+b.ar7_meta['confidence_gain']*confidence(lib,d['rows'],d['weights'],b.ar7_meta['dispersion_scale'])
            result[cell][variant]=dict(by_task={str(t):group(d['task']==t) for t in range(10)},
                by_phase={p:group(d['phase']==p) for p in np.unique(d['phase'])},
                by_grip_event={str(v):group(d['grip_event']==v) for v in (False,True)},
                strength_quantiles=np.quantile(strength,[0,.1,.5,.9,1]),
                latency_ms=dict(median=float(np.median(times)),p95=float(np.quantile(times,.95)),repetitions=300,synthetic_visual_keys=True),
                tensor_bytes=sum(v.nbytes for v in b.ar7_head.values())+b.ar7_table.nbytes)
    dump(HERE/'results/candidate_diagnostics.json',result)
    dump(HERE/'results/execution_environment.json',dict(affinity=sorted(os.sched_getaffinity(0)),
        threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES')))
    print(json.dumps({c:{v:r['latency_ms'] for v,r in x.items()} for c,x in result.items()}))

if __name__=='__main__': main()
