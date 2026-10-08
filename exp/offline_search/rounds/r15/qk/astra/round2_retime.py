"""Controlled CPU timing of already exported weights; never reserializes models."""
import argparse
import hashlib
import json
import os
import pickle
import time
import numpy as np
import torch
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,load_cell,sha,dump


def run(size):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert os.environ.get('OPENBLAS_NUM_THREADS')=='1' and os.environ.get('MKL_NUM_THREADS')=='1'
    torch.set_num_interop_threads(1)
    cell=f'pi05_l10_{size}';out=HERE/'round2'/'pilot'/cell
    model_hash=sha(out/'student.ts');model=torch.jit.load(str(out/'student.ts'),map_location='cpu').eval()
    _,_,lib,rows,_=load_cell('l10',size)
    with np.load(DERIVED/cell/'train.npz') as data:
        tr={k:data[k] for k in ('rs','task_id')}
    metric=pickle.load((out/'metric.pkl').open('rb'));images=[np.load(lib/f'tok/img{c}.npy',mmap_mode='r') for c in range(2)]
    examples=[]
    for task in range(10):
        ids=np.flatnonzero(tr['task_id']==task);chosen=ids[np.linspace(0,len(ids)-1,40,dtype=int)]
        assert len(np.unique(chosen))==40
        raw=[np.stack([np.array(a[int(rows[i])]) for a in images]) for i in chosen]
        examples.append((task,chosen,raw))
    timings=[]
    for threads in (1,4):
        torch.set_num_threads(threads);samples=[];feature_samples=[]
        with torch.inference_mode():
            for task,chosen,raws in examples:
                wf=metric[task][0]['wf'];shift=metric[task][0]['shift']
                def call(raw,index,record):
                    start=time.perf_counter_ns()
                    image=torch.from_numpy(raw[:,::4,::4].transpose(0,3,1,2).copy()).float()[None]/255.
                    state=tr['rs'][index:index+1,:8].copy()
                    p=model(image,torch.from_numpy(state),torch.tensor([task],dtype=torch.long)).numpy()
                    feature_end=time.perf_counter_ns()
                    code=np.c_[p,state]@wf-shift
                    end=time.perf_counter_ns()
                    if record:samples.append((end-start)/1e6);feature_samples.append((feature_end-start)/1e6)
                    return code
                for n in range(20):call(raws[n],int(chosen[n]),False)
                for raw,index in zip(raws,chosen):
                    assert np.isfinite(call(raw,int(index),True)).all()
        timings.append({'cpu_threads':threads,'torch_threads':threads,'numpy_blas_threads_env':1,'n':len(samples),
            'distinct_inputs':400,'median_ms':float(np.median(samples)),'q1_ms':float(np.quantile(samples,.25)),
            'q3_ms':float(np.quantile(samples,.75)),'p95_ms':float(np.quantile(samples,.95)),
            'features_only_median_ms':float(np.median(feature_samples)),'samples_ms':samples})
    previous=out/'latency_initial_busy.json'
    if not previous.exists():previous.write_bytes((out/'latency.json').read_bytes())
    record={'input':'400 distinct own-library pairs of raw224 RGB -> nearest56 -> float -> frozen student -> same main AWM CPU code; common synthesis excluded',
        'cpu_only':True,'tasks':list(range(10)),'warmups_per_task':20,'process_affinity':sorted(os.sched_getaffinity(0)),
        'timings':timings,'initial_busy_timing_retained':str(previous),'model_sha256':model_hash,
        'torch_and_numpy_blas_threads_controlled':True}
    dump(out/'latency.json',record)
    spec=json.loads((out/'encoder_config.json').read_text());spec['epsilon']=.152*timings[0]['median_ms']/10.26
    spec['cost_basis']='round2 controlled Torch+BLAS one-thread raw-image-to-AWM-code median over400 distinct inputs; pi05 graph tower10.26ms cv=.152'
    spec.pop('representation_sha256',None);spec['representation_sha256']=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    dump(out/'encoder_config.json',spec)
    manifest=json.loads((out/'manifest.json').read_text())
    for f in ('encoder_config.json','latency.json'):manifest['files'][f]=sha(out/f)
    manifest['controlled_cpu_latency']=True;manifest['representation_sha256']=spec['representation_sha256'];dump(out/'manifest.json',manifest)
    assert sha(out/'student.ts')==model_hash
    print('CONTROLLED_LATENCY',size,[(x['cpu_threads'],x['median_ms'],x['p95_ms']) for x in timings],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,required=True);a=ap.parse_args();run(a.size)
