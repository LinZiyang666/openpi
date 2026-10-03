"""Extra-data baseline: nearest-neighbor memory of discovery policy shadows.

Same init folds and SAME labels as student; frozen library Mahalanobis metric.
This checks whether the result needs a learned residual head or just more memory.
"""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .common import RUN,HERE,DERIVED,load_compact,dump
from .student import dataset
from .shadows import errors,episode_mean
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler


def one(spec):
    name=spec['arm'];d=dataset(load_compact(name));datasets={'cache':d}
    for v in ['IP','CU']:
        if (DERIVED/'compact'/f'{name[:-1]+v}.npz').exists():datasets[v]=dataset(load_compact(name[:-1]+v))
    with open(spec['r8']['source']['artifact'],'rb') as f:base=FitUnpickler(f).load()['method']
    outputs={key:{m:value['base'].copy() for m in ['memory_teacher_half','memory_residual_half']} for key,value in datasets.items()}
    for task in range(10):
        T=base.tasks[task]
        for fold in range(5):
            tr=(d['task']==task)&(d['init']%5!=fold)
            codes=d['x'][tr,:136]@T.Wf-T.shift
            target=d['teacher'][tr,:,:6]
            residual=target-d['base'][tr,:,:6]
            for key,z in datasets.items():
                te=np.flatnonzero((z['task']==task)&(z['init']%5==fold))
                for lo in range(0,len(te),256):
                    ii=te[lo:lo+256];q=z['x'][ii,:136]@T.Wf-T.shift
                    distance=np.sqrt(np.maximum((q*q).sum(1)[:,None]+(codes*codes).sum(1)[None]-2*q@codes.T,0))
                    nn=np.argsort(distance,axis=1)[:,:16]
                    ds=np.take_along_axis(distance,nn,1)
                    w=_kernel_w(ds-ds[:,:1],8);w/=w.sum(1,keepdims=True)
                    mean=np.einsum('nk,nkha->nha',w,target[nn])
                    correction=np.einsum('nk,nkha->nha',w,residual[nn])
                    outputs[key]['memory_teacher_half'][ii,:,:6]=.5*(z['base'][ii,:,:6]+mean)
                    outputs[key]['memory_residual_half'][ii,:,:6]+=.5*correction
    result={}
    for key,z in datasets.items():
        base_error=episode_mean(errors(z['base'],z['teacher'])[0],z['task'],z['init']).mean()
        result[key]={}
        for method,a in outputs[key].items():
            err=episode_mean(errors(a,z['teacher'])[0],z['task'],z['init']).mean()
            result[key][method]=dict(mse=float(err),relative_mse=float(err/base_error-1))
    return dict(arm=name,results=result)


def main():
    specs=[s for s in json.loads((RUN/'arms.json').read_text()) if s['r8']['variant']=='A']
    out=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,s) for s in specs]):
            r=f.result();out.append(r);print(json.dumps(r),flush=True)
    dump(HERE/'results/shadow_memory.json',out)


if __name__=='__main__':main()
