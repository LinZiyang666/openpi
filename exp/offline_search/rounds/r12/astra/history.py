"""Does correctly represented camera memory need a short refresh interval?"""
from .common import *
install()
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from .retrieval import Metric,retrieve,lag

def run(cell,size):
    start=time.monotonic();C,_=load(cell,size);P,rs=C['P'],C['rs'];p2=lag(C,2);p4=lag(C,4)
    initial=np.empty(len(P),int)
    for e in np.unique(C['ep']):
        rr=np.flatnonzero(C['ep']==e);initial[rr]=rr[np.argmin(C['step'][rr])]
    Xs=dict(memory_init=np.c_[P[initial,:64],P[:,64:],rs],
            memory4=np.c_[P[p4,:64],P[:,64:],rs],
            memory2_state=np.c_[P[p2,:64],P[:,64:],rs,rs[p2]],
            memory4_state=np.c_[P[p4,:64],P[:,64:],rs,rs[p4]],
            memory2_both=np.c_[P[p2],P[:,64:],rs,rs[p2]],
            memory2_delta=np.c_[P[p2,:64],P[:,64:],rs,rs-rs[p2]])
    output=[]
    for name,X in Xs.items():
        vals=[];eps=[];deltas=[]
        for f in range(5):
            tr=np.flatnonzero(C['fold']!=f);sig=np.maximum(C['act'][tr,:5].std((0,1)),1e-6);C['sigma']=sig
            for t in range(10):
                cr=tr[C['task'][tr]==t];qr=np.flatnonzero((C['fold']==f)&(C['task']==t)&(C['step']>=2)&(C['step']%2==0))
                m=Metric(X[cr],C['act'][cr],C['ep'][cr],sig);r=retrieve(m,X[qr],qr,cr,C)
                vals.append(np.mean(((r['action'][:,:,:6]-C['act'][qr,:,:6])/sig[:6])**2,(1,2)));eps.append(C['ep'][qr])
        output.append(dict(variant=name,motion_mse=mean(np.concatenate(vals),np.concatenate(eps))))
        print(f'history {cell}_{size} {name} secs={time.monotonic()-start:.1f}',flush=True)
    result=dict(timestamp=now(),cell=cell,size=size,variants=output,seconds=time.monotonic()-start)
    d=HERE/'history'/f'{cell}_{size}';dump(d/'summary.json',result);dump(d/'reads.json',sorted(READS));return result

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(run,c,s) for s in (50,200,500) for c in ('pi05_l10','pi05_spatial','groot_l10','groot_spatial')]
        for f in fs:print(json.dumps(f.result()),flush=True)
