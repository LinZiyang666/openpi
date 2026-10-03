"""Bootstrap routing's whole cross-fitting procedure, refitting every LP.

Matched controller outcomes and original init-fold identity travel together.
Includes training instability; remains exploratory after comparing menu families.
"""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .common import HERE,SEED,load_compact,dump
from .routing import optimize


def one(row,draws=1000):
    data=[load_compact(name) for name in row['arms']]
    y,c,n=[np.stack([z[k] for z in data],axis=-1) for k in ['success','cost','decisions']]
    rng=np.random.default_rng(SEED);values=[]
    for draw in range(draws):
        weights=np.stack([np.bincount(rng.integers(0,30,30),minlength=30) for _ in range(10)])
        probs=np.zeros_like(y)
        for fold in range(5):
            train=weights*(np.arange(30)[None,:]%5!=fold)
            p,_=optimize(y,c,n,prior=12,tolerance=.01,weights=train)
            probs[:,np.arange(30)%5==fold]=p[:,None,:]
        success=(y*probs).sum(-1)
        delta=((success-y[:,:,0])*weights).sum()/300
        ir=(c*probs*weights[:,:,None]).sum()/(n*probs*weights[:,:,None]).sum()
        values.append((delta,ir,(success*weights).sum()/300))
    values=np.asarray(values)
    return dict(cell=row['cell'],draws=draws,point=row['oof'],
                delta_interval=np.quantile(values[:,0],[.025,.975]).tolist(),
                ir_interval=np.quantile(values[:,1],[.025,.975]).tolist(),
                sr_interval=np.quantile(values[:,2],[.025,.975]).tolist(),
                warning='Refits rules, but neither new task nor new policy random seed uncertainty; not a confirmatory NI test')


def main():
    rows=[r for r in json.loads((HERE/'results/routing.json').read_text())
          if r['menu']=='binary' and r['prior']==12 and r['tolerance']==.01]
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,r) for r in rows]):
            r=f.result();results.append(r);print(json.dumps(r),flush=True)
    dump(HERE/'results/routing_uncertainty.json',results)


if __name__=='__main__':main()
