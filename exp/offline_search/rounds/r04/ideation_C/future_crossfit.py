"""Hold out entire library episodes from label-dependent metric fitting and retrieval.
PCA basis remains the deployed library's fixed unsupervised basis. Not a loop proxy.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from exp.offline_search.rounds.r04.ideation_C.library_diagnostic import OUT,ROOT,load_library,dsq
from exp.offline_search.harness.store import LibraryView
from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric

def work(args):
    model,suite,scale=args;lib,x=load_library(*args)
    sig=np.std(LibraryView(ROOT,f'{model}_{suite}').action[:,:5,:7],axis=(0,1))
    h=(lib.action[:,:5,:7]/sig).reshape(lib.L,35)
    nx=np.asarray(lib.next);i1=np.where(nx>=0,nx,np.arange(lib.L));i2=np.where(nx[i1]>=0,nx[i1],i1)
    targets={'head':h,'future':np.c_[h,np.sqrt(.5)*h[i1],.5*h[i2]]}
    out=[]
    for task in lib.tasks():
        r=np.asarray(lib.rows_of_task(task)); ep=lib.episode[r]
        epi=np.searchsorted(np.unique(ep),ep)
        for fold in range(5):
            tr=r[epi%5!=fold];q=r[epi%5==fold]
            if not len(q):continue
            for key,label in targets.items():
                mu,sd,w=fit_metric(x[tr],label[tr],lib.episode[tr])
                ztr=((x[tr]-mu)/sd)@w;zq=((x[q]-mu)/sd)@w
                d=np.sqrt(dsq(zq,ztr));kk=min(16,len(tr));kr=5 if scale==50 else 8
                j=np.argpartition(d,kk-1,axis=1)[:,:kk];dd=np.take_along_axis(d,j,axis=1)
                o=np.argsort(dd,axis=1);j=np.take_along_axis(j,o,axis=1);dd=np.take_along_axis(dd,o,axis=1)
                wt=np.exp(-((dd-dd[:,:1])/np.maximum(dd[:,kr-1:kr]-dd[:,:1],1e-6))**2);wt/=wt.sum(1)[:,None]
                sel=tr[j];head=np.einsum('nk,nkd->nd',wt,h[sel]);future=np.einsum('nk,nkd->nd',wt,h[i1[sel]])
                val=(nx[q]>=0)&(nx[i1[q]]>=0)
                out.append(dict(task=int(task),fold=fold,label=key,n=len(q),n_future=int(val.sum()),
                    head_rms=float(np.sqrt(np.mean((head-h[q])**2,axis=1)).mean()),
                    next_rms=float(np.sqrt(np.mean((future-h[i1[q]])**2,axis=1))[val].mean()),
                    progress_far=float((wt*(np.abs(lib.progress[q,None]-lib.progress[sel])>.25)).sum(1).mean())))
    result=dict(model=model,suite=suite,scale=scale,rows=out)
    (OUT/f'crossfit_{model}_{suite}_{scale}.json').write_text(json.dumps(result,indent=2))
    summary={}
    for label in targets:
        rr=[r for r in out if r['label']==label]
        summary[label]={k:sum(r[k]*r['n_future' if k=='next_rms' else 'n'] for r in rr)/sum(r['n_future' if k=='next_rms' else 'n'] for r in rr) for k in ['head_rms','next_rms','progress_far']}
    print(model,suite,scale,json.dumps(summary),flush=True)
    return summary

if __name__=='__main__':
    jobs=[(m,s,n) for m in ['pi05','groot'] for s in ['spatial','l10'] for n in [50,500]]
    with ProcessPoolExecutor(max_workers=8) as pool:list(pool.map(work,jobs))
