"""Read-only library diagnostics. All writes remain beside this script (<50 MB)."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from exp.offline_search.harness.store import LibraryView
from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/weiland/trace_runs/offline_search_store')

def load_library(model, suite, scale):
    key=f'{model}_{suite}'
    name='current' if scale==50 else ('bpool_cs' if model=='pi05' else 'bpool_all')
    lib=LibraryView(ROOT,key,name)
    pc=ROOT/'derived'/('r02/g1_awm/pca_current' if scale==50 else 'r01/f4_vision/pca')/key
    if scale==500: pc=pc/name
    x=np.concatenate([np.load(pc/f/'proj.npy',mmap_mode='r')[:,:64] for f in ['v0','v1']]+[lib.rs[:,:8]],axis=1).astype(float)
    return lib,x

def phase_labels(lib):
    # Cumulative executed gripper transitions, including within-chunk transitions. Offline diagnostic ONLY.
    phase=np.zeros(lib.L,int)
    for ep in lib.episodes:
        lo,hi=ep['start'],ep['end']
        s=np.sign(lib.action[lo:hi,:5,6]).reshape(-1)
        switches=np.r_[0,np.cumsum(s[1:]!=s[:-1])].reshape(-1,5)
        phase[lo:hi]=switches[:,0]
    return phase

def dsq(a,b):
    return np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None]-2*a@b.T,0)

def summary(v):
    v=np.asarray(v)
    return dict(n=int(v.size),mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)))

def work(args):
    model,suite,scale=args
    lib,x=load_library(*args)
    current=LibraryView(ROOT,f'{model}_{suite}','current')
    sig=np.std(current.action[:,:5,:7],axis=(0,1)).astype(float)
    hd=(lib.action[:,:5,:7]/sig).reshape(lib.L,35)
    tail=(lib.action[:,5:10,:7]/sig).reshape(lib.L,35)
    nx=np.asarray(lib.next)
    n1=np.where(nx>=0,nx,np.arange(lib.L))
    n2=np.where(nx[n1]>=0,nx[n1],n1)
    phase=phase_labels(lib)
    # No query history, endpoint labels, or progress used online: these are library-only labels.
    targets={'head':hd,'tail':np.c_[hd,tail*np.sqrt(.5)],
             'future':np.c_[hd,hd[n1]*np.sqrt(.5),hd[n2]*.5]}
    rows=[]
    saved={}
    for task in lib.tasks():
        ix=np.asarray(lib.rows_of_task(task)); n=len(ix); ep=lib.episode[ix]
        labels={k:v[ix] for k,v in targets.items()}
        same=ep[:,None]==ep[None,:]
        top={}
        for k,h in labels.items():
            d=dsq(h,h); d[same]=np.inf
            top[k]=np.argpartition(d,2,axis=1)[:,:3]
        for k,pp in top.items():
            p=ix[pp]; ok=(nx[ix]>=0)&(nx[n1[ix]]>=0)
            now=np.sqrt(np.mean((hd[ix,None]-hd[p])**2,axis=-1))
            nxt=np.sqrt(np.mean((hd[n1[ix],None]-hd[n1[p]])**2,axis=-1))
            prog=np.abs(lib.progress[ix,None]-lib.progress[p])
            pd=phase[ix,None]!=phase[p]
            tm=(nx[ix,None]<0)!=(nx[p]<0)
            metrics=dict(model=model,suite=suite,scale=scale,task=int(task),label=k,n=n,
                pair_head_rms=float(now.mean()),pair_next_rms=float(nxt[ok].mean()),
                pair_phase_mismatch=float(pd.mean()),pair_progress_far=float((prog>.25).mean()),
                pair_terminal_mismatch=float(tm.mean()))
            # Full-library metric fit, leave-own-episode-out retrieval (not held-out metric fit).
            mean,std,w=fit_metric(x[ix],labels[k],ep)
            z=((x[ix]-mean)/std)@w
            d=np.sqrt(dsq(z,z)); d[same]=np.inf
            kk=min(16,int((~same).sum(1).min()))
            j=np.argpartition(d,kk-1,axis=1)[:,:kk]
            dd=np.take_along_axis(d,j,axis=1); order=np.argsort(dd,axis=1)
            j=np.take_along_axis(j,order,axis=1); dd=np.take_along_axis(dd,order,axis=1)
            kr=5 if scale==50 else 8
            wgt=np.exp(-((dd-dd[:,:1])/np.maximum(dd[:,kr-1:kr]-dd[:,:1],1e-6))**2)
            wgt/=wgt.sum(1)[:,None]
            sel=ix[j]
            metrics['loeo_phase_mismatch']=float((wgt*(phase[ix,None]!=phase[sel])).sum(1).mean())
            metrics['loeo_progress_far']=float((wgt*(np.abs(lib.progress[ix,None]-lib.progress[sel])>.25)).sum(1).mean())
            metrics['loeo_head_rms']=float(np.sqrt(np.mean((np.einsum('nk,nkd->nd',wgt,hd[sel])-hd[ix])**2,axis=1)).mean())
            metrics['loeo_next_rms']=float(np.sqrt(np.mean((np.einsum('nk,nkd->nd',wgt,hd[n1[sel]])-hd[n1[ix]])**2,axis=1))[ok].mean())
            rows.append(metrics)
            if k=='head':
                saved[f't{task}_rows']=ix; saved[f't{task}_mean']=mean
                saved[f't{task}_std']=std; saved[f't{task}_w']=w; saved[f't{task}_z']=z.astype(np.float32)
        # Repeated physical phase flips vary, so a hard event-count gate may be dangerous.
    np.savez_compressed(OUT/f'codes_{model}_{suite}_{scale}.npz',**saved)
    np.savez_compressed(OUT/f'phases_{model}_{suite}_{scale}.npz',phase=phase)
    result=dict(model=model,suite=suite,scale=scale,L=lib.L,episodes=len(lib.episodes),
        failed_episodes=sum(not e['success'] for e in lib.episodes),
        phases=summary([phase[e['end']-1] for e in lib.episodes]),rows=rows)
    (OUT/f'library_{model}_{suite}_{scale}.json').write_text(json.dumps(result,indent=2))
    return {k:v for k,v in result.items() if k!='rows'}

if __name__=='__main__':
    jobs=[(m,s,n) for m in ['pi05','groot'] for s in ['spatial','l10'] for n in [50,500]]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for r in pool.map(work,jobs): print(json.dumps(r),flush=True)
