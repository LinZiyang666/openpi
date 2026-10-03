"""Cheap perception distillation: 143 inputs, NO third-camera features.

Fit on 0..19 W10 live paths plus A wrist-counterfactual queries. Evaluate ONLY
20..29, separately on both paths. W10 first observation is full as deployed.
Full-camera shadow retrieval is only a same-state diagnostic benchmark.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from .data import HERE, CELLS, load, dataset, subset, masks, temporal, dump
from .learn import fit
from .shared_head import metrics
from ..inference import correct


def wrist_dataset(cell,variant):
    d=dataset(cell,variant); z=load(cell,variant)
    take=z['vision'] & np.isfinite(z['shadow_look_cache_chunk']).all((1,2)) & np.isfinite(z['state_norm']).all(1)
    d['full_cache']=z['shadow_look_cache_chunk'][take]
    if variant=='A':
        d['base']=z['camera_shadow_wrist_cache_chunk'][take].copy()
        d['base'][d['seq']==0]=d['full_cache'][d['seq']==0]
    # Only wrist PCA + proprioception + ACTUAL wrist cache action + elapsed.
    d['x']=np.c_[d['x'][:,64:128],d['state_norm'],d['base'].reshape(len(d['base']),-1),np.minimum(d['seq'],120)/120]
    return d


def one(cell):
    ds={v:wrist_dataset(cell,v) for v in ['A','W10']}
    tr={v:subset(d,masks(d['init'])[0]) for v,d in ds.items()}
    te={v:subset(d,masks(d['init'])[1]) for v,d in ds.items()}
    x=np.concatenate([d['x'] for d in tr.values()]);targets=[];episodes=[]
    for k,d in enumerate(tr.values()):
        y=d['teacher'].copy();y[:,:,:6]-=d['base'][:,:,:6];y[:,:,6]=np.where(y[:,:,6]>=0,1.,-1.)
        targets.append(y.reshape(len(y),-1));episodes.append(d['episode']+k*300)
    model=fit(x,np.concatenate(targets),np.concatenate(episodes))
    path=HERE/'artifacts'/f'{cell}_wrist.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**model)
    out=dict(cell=cell,fit_inits=list(range(20)),eval_inits=list(range(20,30)),
        artifact=str(path),features=143,fit_anchors=len(x),fit_arm_episodes=400,paths={})
    for v,d in te.items():
        out['paths'][v]={}
        for name,blend,grip in [('half',.5,False),('half_grip',.5,True),('full',1.,False)]:
            out['paths'][v][name]=metrics(correct(model,d['x'],d['base'],blend,grip),d)
        out['paths'][v]['full_camera_cache']=metrics(d['full_cache'],d)
        z=load(cell,v)
        out['paths'][v]['factual_sr']=float(z['success'][:,20:30].mean())
        out['paths'][v]['factual_ir']=float(z['cost'][:,20:30].sum()/z['decisions'][:,20:30].sum())
    dump(HERE/'results'/f'wrist_{cell}.json',out)
    print(cell,{v:{k:round(m['relative_mse'],3) for k,m in r.items() if isinstance(m,dict)} for v,r in out['paths'].items()},flush=True)
    return out


def main():
    out=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,c) for c in CELLS if c.startswith('pi05')]):out.append(f.result())
    dump(HERE/'results/wrist_head.json',sorted(out,key=lambda x:x['cell']))


if __name__=='__main__':main()
