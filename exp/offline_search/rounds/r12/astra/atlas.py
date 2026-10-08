"""Pre-encoding half-safety atlas, with labels fitted only inside each outer fold.

The camera planner sees the previous full query, current state, and training
donors' two-row successors. It never sees the current query's wrist or v0 key.
The inner labels use training-library LOEO retrieval (not heldout query labels).
"""
from .common import *
install()
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from .retrieval import Metric,retrieve,lag

def run(cell,size):
    start=time.monotonic();C,lib=load(cell,size);P,rs=C['P'],C['rs'];p2=lag(C,2)
    ep_len=np.asarray(lib.ep_len)
    X=np.c_[P,rs];Y=np.c_[P[p2,:64],P[:,64:],rs,rs-rs[p2]]
    nxt=np.full(len(P),-1,int)
    r=np.flatnonzero(C['prev']>=0);nxt[C['prev'][r]]=r
    n2=np.where(nxt>=0,nxt[np.maximum(nxt,0)],-1)
    accum=[]
    for f in range(5):
        tr=np.flatnonzero(C['fold']!=f);sig=np.maximum(C['act'][tr,:5].std((0,1)),1e-6);C['sigma']=sig
        scale=np.maximum(rs[tr].std(0),1e-6)
        for t in range(10):
            cr=tr[C['task'][tr]==t]
            # Skip step0 and the first candidate; no fabricated early-metric replay.
            qr=np.flatnonzero((C['fold']==f)&(C['task']==t)&(C['step']>=4)&(C['step']%2==0))
            ar=cr[C['step'][cr]>=2]
            fm=Metric(X[cr],C['act'][cr],C['ep'][cr],sig)
            hm=Metric(Y[cr],C['act'][cr],C['ep'][cr],sig)
            rf=retrieve(fm,X[ar],ar,cr,C);rh=retrieve(hm,Y[ar],ar,cr,C)
            rp=retrieve(fm,X[p2[ar]],p2[ar],cr,C)
            qf=retrieve(fm,X[qr],qr,cr,C);qh=retrieve(hm,Y[qr],qr,cr,C)
            prev=retrieve(fm,X[p2[qr]],p2[qr],cr,C)
            score=np.full(len(P),np.inf);same=np.zeros(len(P),bool)
            score[ar]=np.sqrt(np.mean(((rf['action'][:,:,:6]-rh['action'][:,:,:6])/sig[:6])**2,(1,2)))
            same[ar]=((rf['action'][:,:,6]>=0)==(rh['action'][:,:,6]>=0)).all(1)
            progress_same=np.zeros(len(P),bool)
            previous_prog=C['progress'][rp['rows'][:,0]]
            af=(C['progress'][rf['rows'][:,0]]-previous_prog)*np.maximum(ep_len[rf['rows'][:,0]]-1,1)>.5
            ah=(C['progress'][rh['rows'][:,0]]-previous_prog)*np.maximum(ep_len[rh['rows'][:,0]]-1,1)>.5
            progress_same[ar]=(af==ah)&(C['step'][ar]>=4)
            succ=n2[prev['rows']];ok=succ>=0;ss=np.maximum(succ,0)
            assert np.isin(C['ep'][ss[ok]],C['ep'][cr]).all()
            delta=rs[ss]-rs[prev['rows']]
            mass=(prev['weights']*ok).sum(1)
            w=prev['weights']*ok/np.maximum(mass[:,None],1e-9)
            expected=np.einsum('qk,qkd->qd',w,delta)
            state_error=np.sqrt(np.mean(((rs[qr]-rs[p2[qr]]-expected)/scale)**2,1))
            vals={k:C[k][qr] for k in ('row','ep','step','fold','task')}
            vals.update(full_mse=np.mean(((qf['action'][:,:,:6]-C['act'][qr,:,:6])/sig[:6])**2,(1,2)),
                        half_mse=np.mean(((qh['action'][:,:,:6]-C['act'][qr,:,:6])/sig[:6])**2,(1,2)),
                        full_delta=np.mean(((qf['action'][:,:,:6]-qh['action'][:,:,:6])/sig[:6])**2,(1,2)),
                        grip_flip=np.mean((qf['action'][:,:,6]>=0)!=(qh['action'][:,:,6]>=0),1),
                        state_error=state_error,support_mass=mass,
                        full_top=qf['rows'][:,0],half_top=qh['rows'][:,0],previous_top=prev['rows'][:,0])
            for threshold in (.1,.15,.2):
                vals[f'safe_mass_{threshold}']=(prev['weights']*ok*(score[ss]<=threshold)*same[ss]).sum(1)
                vals[f'safe_mass_guard_{threshold}']=(prev['weights']*ok*(score[ss]<=threshold)*same[ss]*progress_same[ss]).sum(1)
            accum.append(vals)
        print(f'atlas {cell}_{size} fold={f} secs={time.monotonic()-start:.1f}',flush=True)
    a={k:np.concatenate([x[k] for x in accum]) for k in accum[0]};summ=[]
    for threshold in (.1,.15,.2):
        for required in (.5,.75,.9):
            for valve in (.1,.2,float('inf')):
                accept=(a[f'safe_mass_{threshold}']>=required)&(a['support_mass']>=.99)&(a['state_error']<=valve)
                selected=np.zeros(len(accept),bool)
                for ep in np.unique(a['ep']):
                    rr=np.flatnonzero(a['ep']==ep);rr=rr[np.argsort(a['step'][rr])];last_full=2
                    for i in rr:
                        if a['step'][i]-last_full==2 and accept[i]:selected[i]=True
                        else:last_full=int(a['step'][i])
                d=np.where(selected,a['half_mse']-a['full_mse'],0)
                summ.append(dict(threshold=threshold,required_mass=required,state_valve=None if not np.isfinite(valve) else valve,
                    half_share=mean(selected,a['ep']),mse_delta=mean(d,a['ep']),relative_change=mean(d,a['ep'])/mean(a['full_mse'],a['ep']),
                    full_delta=mean(np.where(selected,a['full_delta'],0),a['ep']),grip_flip=mean(np.where(selected,a['grip_flip'],0),a['ep'])))
    dest=HERE/'atlas'/f'{cell}_{size}';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'rows.npz',**a)
    result=dict(timestamp=now(),cell=cell,size=size,rows=len(a['ep']),candidates=summ,seconds=time.monotonic()-start)
    dump(dest/'summary.json',result);dump(dest/'reads.json',sorted(READS));return result

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(run,c,s) for s in (50,200,500) for c in ('pi05_l10','pi05_spatial','groot_l10','groot_spatial')]
        for f in fs:print(json.dumps(f.result()),flush=True)
