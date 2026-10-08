"""Library-only fresh-fallback stress of the task-indexed continuity scale.

This conditional diagnostic uses fixed full-library representation and excludes
the query episode from donors. It is not episode-held-out metric validation.
"""
from common import *
fence()
import pickle
import numpy as np
from partition import anchors,load
from exp.offline_search.rounds.r02.g1_awm.awm import _masked_min,_kernel_w
from concurrent.futures import ThreadPoolExecutor

def one(cell):
    model,suite,n=cell.split('_');C=load(model,suite,int(n))
    m=pickle.load((R10/'recipe/artifacts'/f'r10_recipe_{cell}.pkl').open('rb'))['method'];b=m.inner.base
    head=(C['act'][:,:5].astype(float)/b.sig.astype(float)).reshape(-1,35)
    tail=(C['act'][:,5:10].astype(float)/b.sig.astype(float)).reshape(-1,35)
    pooled=float(np.median(_masked_min(tail,head,C['ep'],C['ep'],rms_dim=35)))+1e-6
    rows=[]
    for t in range(10):
        T=b.tasks[t];qr=anchors(C,T.rows);qr=qr[C['step'][qr]>0];cr=T.rows
        z=C['X'][qr]@T.Wf-T.shift
        D=np.sqrt(np.maximum(T.z2[None]-2*z@T.Z.T+(z*z).sum(1)[:,None],0))
        prev=qr-1;assert np.all(C['ep'][prev]==C['ep'][qr])
        h=tail[prev].astype(np.float32)
        cd=np.sqrt(np.maximum(T.h2[None]-2*h@T.HD.T+(h*h).sum(1)[:,None],0)/35)
        own=C['ep'][qr,None]==C['ep'][cr][None];D[own]=np.nan
        med=np.nanmedian(D,axis=1)+1e-12;D[own]=np.inf
        result={}
        for label,scale in [('task',T.s_c),('pool',pooled)]:
            dt=D/med[:,None]+b.lam_c*cd/scale
            ix=np.argsort(dt,axis=1,kind='stable')[:,:16];d=np.take_along_axis(dt,ix,1).astype(float)
            w=_kernel_w(d-d[:,:1],b.kref);w=(w/w.sum(1,keepdims=True)).astype(np.float32)
            a=np.einsum('qk,qktc->qtc',w,C['act'][cr[ix]],optimize=False)
            result[label]=(cr[ix[:,0]],np.mean(((a[:,:,:6]-C['act'][qr,:,:6])/b.sig[:6])**2,axis=(1,2)))
        for e in np.unique(C['ep'][qr]):
            k=C['ep'][qr]==e
            rows.append(dict(cell=cell,task=t,ep=int(e),queries=int(k.sum()),task_scale=T.s_c,pool_scale=pooled,
               task_mse=float(result['task'][1][k].mean()),pool_mse=float(result['pool'][1][k].mean()),
               top1_changed=float((result['task'][0][k]!=result['pool'][0][k]).mean())))
    csvout(HERE/f'continuity_{cell}.csv',rows);print(cell,'fresh queries',sum(r['queries'] for r in rows),flush=True)
    return dict(cell=cell,pooled_scale=pooled,task_scales=[b.tasks[t].s_c for t in range(10)],
        queries=sum(r['queries'] for r in rows),task_mse=float(np.mean([r['task_mse'] for r in rows])),
        pool_mse=float(np.mean([r['pool_mse'] for r in rows])),top1_changed=float(np.mean([r['top1_changed'] for r in rows])))

with ThreadPoolExecutor(4) as ex:results=list(ex.map(one,CELLS))
dump(HERE/'continuity_summary.json',results);provenance('continuity')
