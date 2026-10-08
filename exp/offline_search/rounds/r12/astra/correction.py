"""Outer-held-out heads for full, wrist and delayed-context retrieval.

Diagnostic: all heads train on steps>=2, same task/episode-balanced R10 ridge,
main metric. Distance scale is training LOEO, not deployed five-fold scale.
This checks integration risk; it is not an exact deployed-recipe SR proxy.
"""
from .common import *
install()
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .retrieval import Metric, retrieve, lag
from exp.offline_search.rounds.r10.train import fit_head, anchor_weights
from exp.offline_search.rounds.r10.recipe.recipe import predict

def feat(X, A, step, task, sigma):
    return np.c_[X, (A/sigma).reshape(len(A), 70), np.minimum(step,120)/120,
                  np.eye(10)[task]].astype(np.float32)

def wm(x, ep):
    ix=np.argsort(x, kind='stable'); w=ew(ep)[ix]
    return float(x[ix[np.searchsorted(np.cumsum(w), w.sum()/2)]])

def run(cell,size,dynamic=False):
    start=time.monotonic(); C,_=load(cell,size)
    P=C['P']; past=lag(C,2)
    Xs=dict(full=np.c_[P,C['rs']],wrist=np.c_[P[:,64:],C['rs']],
            memory2=np.c_[P[past,:64],P[:,64:],C['rs']])
    if dynamic:
        delta=C['rs']-C['rs'][past]
        Xs['full_delta']=np.c_[Xs['full'],delta]
        Xs['wrist']=np.c_[Xs['wrist'],delta]
        Xs['memory2']=np.c_[Xs['memory2'],delta]
    output=[]; saved={}; metas=[]
    for fold in range(5):
        tr=np.flatnonzero(C['fold']!=fold)
        sig=np.maximum(C['act'][tr,:5].std((0,1)),1e-6).astype(np.float32); C['sigma']=sig
        cache={v:{} for v in Xs}; dscale={}
        # All task training distances pooled, preserving equal episode weights.
        for v,X in Xs.items():
            ds=[]; es=[]
            for t in range(10):
                cr=tr[C['task'][tr]==t]
                ar=cr[C['step'][cr]>=2]
                qr=np.flatnonzero((C['fold']==fold)&(C['task']==t)&(C['step']>=2)&(C['step']%2==0))
                m=Metric(X[cr],C['act'][cr],C['ep'][cr],sig)
                rtr=retrieve(m,X[ar],ar,cr,C); rva=retrieve(m,X[qr],qr,cr,C)
                ds.append(rtr['distance']); es.append(C['ep'][ar])
                y=((C['act'][ar,:,:6]-rtr['action'][:,:,:6])/sig[:6]).reshape(len(ar),60)
                h=fit_head(feat(X[ar],rtr['action'],C['step'][ar],C['task'][ar],sig),y,anchor_weights(ar,C['ep'][ar]))
                cache[v][t]=(ar,qr,rtr,rva,h)
            dscale[v]=wm(np.concatenate(ds),np.concatenate(es))
        for t in range(10):
            qr=cache['full'][t][1]; results={}
            for v,X in Xs.items():
                _,_,_,r,h=cache[v][t]
                strength=.5*np.clip((2-r['distance']/dscale[v])/1.25,0,1)
                f=feat(X[qr],r['action'],C['step'][qr],C['task'][qr],sig)
                corr=predict(h,f).reshape(len(qr),10,6)*sig[:6]
                a=r['action'].copy(); a[:,:,:6]+=strength[:,None,None]*corr
                results[v]=a
                # Feeding stale v0 into the original head: a known integration hazard.
                if v=='memory2' and not dynamic:
                    hf=cache['full'][t][-1]
                    ab=r['action'].copy()
                    ab[:,:,:6]+=strength[:,None,None]*predict(hf,f).reshape(len(qr),10,6)*sig[:6]
                    results['memory2_oldhead']=ab
            for v,a in results.items():
                source='memory2' if v=='memory2_oldhead' else v
                r=cache[source][t][3]
                stats=dict(action=a,
                    motion_mse=np.mean(((a[:,:,:6]-C['act'][qr,:,:6])/sig[:6])**2,(1,2)),
                    raw_mse=np.mean(((r['action'][:,:,:6]-C['act'][qr,:,:6])/sig[:6])**2,(1,2)),
                    full_delta_mse=np.mean(((a[:,:,:6]-results['full'][:,:,:6])/sig[:6])**2,(1,2)),
                    grip_flip_full=np.mean((a[:,:,6]>=0)!=(results['full'][:,:,6]>=0),1),
                    distance=r['distance'],variance=r['variance'],gripvar=r['gripvar'],
                    rows=r['rows'],weights=r['weights'])
                for k,val in stats.items():saved.setdefault(v,{}).setdefault(k,[]).append(val)
            metas.append({k:C[k][qr] for k in ('row','ep','fold','task','step')})
        print(f'head {cell}_{size} fold={fold} secs={time.monotonic()-start:.1f}',flush=True)
    dest=HERE/('correction_dynamic' if dynamic else 'correction')/f'{cell}_{size}'; dest.mkdir(parents=True,exist_ok=True)
    meta={k:np.concatenate([m[k] for m in metas]) for k in metas[0]}
    np.savez_compressed(dest/'meta.npz',**meta)
    for v,vals in saved.items():
        a={k:np.concatenate(x) for k,x in vals.items()};np.savez_compressed(dest/f'{v}.npz',**a)
        output.append(dict(variant=v,**{k:mean(a[k],meta['ep']) for k in ('motion_mse','raw_mse','full_delta_mse','grip_flip_full')}))
    result=dict(timestamp=now(),cell=cell,size=size,dynamic=dynamic,rows=len(meta['ep']),variants=output,seconds=time.monotonic()-start,
                caveats=__doc__)
    dump(dest/'summary.json',result);dump(dest/'reads.json',sorted(READS));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sizes',nargs='+',type=int,default=[50,200,500]);ap.add_argument('--dynamic',action='store_true');args=ap.parse_args()
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(run,c,s,args.dynamic) for s in args.sizes for c in ('pi05_l10','pi05_spatial','groot_l10','groot_spatial')]
        for f in fs: print(json.dumps(f.result()),flush=True)
