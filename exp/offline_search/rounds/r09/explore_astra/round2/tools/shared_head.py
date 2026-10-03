"""Fixed task-agnostic ablations; fit inits 0..19, evaluate ONLY 20..29.

No hyperparameter selection on the evaluation split. Models are frozen at fit
time. Motion residual targets and gripper-sign targets share one multioutput
ridge head. Gripper changes use the fixed .8 confidence threshold, not outcomes.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import numpy as np
from .data import HERE, CELLS, dataset, subset, masks, features, epmeans, interval, dump
from .learn import fit
from ..inference import correct, predict


CONFIGS = dict(cache=dict(paths=['A'],history=False,random_features=768),
    mixed=dict(paths=['A','CU','IP'],history=False,random_features=768),
    temporal=dict(paths=['A','CU','IP'],history=True,random_features=768),
    linear=dict(paths=['A','CU','IP'],history=False,random_features=0))


def metrics(a, d):
    bm = ((d['base'][:,:,:6]-d['teacher'][:,:,:6])**2).mean((1,2))
    am = ((a[:,:,:6]-d['teacher'][:,:,:6])**2).mean((1,2))
    bg = ((d['base'][:,:,6]>=0)!=(d['teacher'][:,:,6]>=0)).mean(1)
    ag = ((a[:,:,6]>=0)!=(d['teacher'][:,:,6]>=0)).mean(1)
    episodes,b = epmeans(bm,d); _,e = epmeans(am,d)
    _,gb=epmeans(bg,d);_,ga=epmeans(ag,d)
    changed=(a[:,:,6]>=0)!=(d['base'][:,:,6]>=0)
    fixed=changed & ((a[:,:,6]>=0)==(d['teacher'][:,:,6]>=0))
    first=np.zeros(len(a),bool)
    for ep in np.unique(d['episode']):
        ix=np.flatnonzero((d['episode']==ep)&changed.any(1))
        if len(ix):first[ix[0]]=True
    out=dict(n=len(a), episodes=len(episodes), base_mse=float(b.mean()), mse=float(e.mean()),
        relative_mse=float(e.mean()/b.mean()-1), mse_difference=interval(e-b,episodes),
        base_gripper_error=float(gb.mean()),gripper_error=float(ga.mean()),
        gripper_difference=interval(ga-gb,episodes), gripper_changes=int(changed.sum()),
        changed_correct=int(fixed.sum()),changed_wrong=int((changed & ~fixed).sum()),
        gripper_change_precision=float(fixed.sum()/changed.sum()) if changed.any() else None)
    out.update(changed_anchors=int(changed.any(1).sum()),changed_episodes=int(first.sum()),
        first_change_precision=float(fixed[first].sum()/changed[first].sum()) if first.any() else None,
        first_change_control_median=float(np.median(d['seq'][first]*5)) if first.any() else None,
        gripper_better_episodes=int((ga<gb-1e-10).sum()),gripper_worse_episodes=int((ga>gb+1e-10).sum()))
    strata={'early':d['seq']<6,'late':d['seq']>=40,'gripper_agrees':bg==0,
        'gripper_disagrees':bg>0,'success':d['success']>0,'failure':d['success']==0,
        'moving_teacher':np.sqrt((d['teacher'][:,:,:6]**2).mean((1,2)))>.1}
    out['strata']={}
    for name,mask in strata.items():
        if not mask.any(): continue
        dd=subset(d,mask);_,bb=epmeans(bm[mask],dd);_,aa=epmeans(am[mask],dd)
        out['strata'][name]=dict(n=int(mask.sum()),episodes=len(bb),base_mse=float(bb.mean()),
            mse=float(aa.mean()),relative_mse=float(aa.mean()/max(bb.mean(),1e-12)-1))
    return out


def diagnose(d):
    # Descriptive only: these outcomes are NEVER model inputs or deployment gates.
    g=((d['base'][:,:,6]>=0)!=(d['teacher'][:,:,6]>=0)).mean(1)
    _,ge=epmeans(g,d)
    eps=np.unique(d['episode']); y=np.array([d['success'][d['episode']==e][0] for e in eps])
    # Ten controls of coherent opposition to the teacher, rather than a lone bit.
    persistent=g>=.8
    return dict(episodes=len(eps),success_rate=float(y.mean()),gripper_error=float(ge.mean()),
        success_gripper_error=float(ge[y>0].mean()) if np.any(y>0) else None,
        failure_gripper_error=float(ge[y==0].mean()) if np.any(y==0) else None,
        persistent_disagreement_episode_fraction=float(np.mean([persistent[d['episode']==e].any() for e in eps])),
        persistent_disagreement_anchor_fraction=float(persistent.mean()))


def one(cell):
    start=time.monotonic()
    data={v:dataset(cell,v) for v in ['A','CU','IP']}
    data={k:v for k,v in data.items() if v is not None}
    train={k:subset(v,masks(v['init'])[0]) for k,v in data.items()}
    test={k:subset(v,masks(v['init'])[1]) for k,v in data.items()}
    out=dict(cell=cell,split=dict(fit=list(range(20)),eval=list(range(20,30))),
        input_identity='No task/init ID, task heads, task thresholds or task scales',
        diagnostics={k:diagnose(v) for k,v in test.items()},models={})
    for name,config in CONFIGS.items():
        ds=[train[v] for v in config['paths'] if v in train]
        x=np.concatenate([features(d,config['history']) for d in ds])
        target=[]
        for d in ds:
            y=d['teacher'].copy(); y[:,:,:6]-=d['base'][:,:,:6]
            y[:,:,6]=np.where(d['teacher'][:,:,6]>=0,1.,-1.)
            target.append(y.reshape(len(y),-1))
        ep=np.concatenate([d['episode']+k*300 for k,d in enumerate(ds)])
        model=fit(x,np.concatenate(target),ep,random_features=config['random_features'])
        path=HERE/'artifacts'/f'{cell}_{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**model)
        result=dict(config=config,fit_anchors=len(x),fit_arm_episodes=len(np.unique(ep)),artifact=str(path),paths={})
        for label,d in test.items():
            xx=features(d,config['history']); result['paths'][label]={}
            for variant,blend,grip in [('half',.5,False),('full',1.,False),('half_grip',.5,True),('grip_only',0.,True)]:
                pred=correct(model,xx,d['base'],blend,grip)
                result['paths'][label][variant]=metrics(pred,d)
        out['models'][name]=result
    out['seconds']=time.monotonic()-start
    dump(HERE/'results'/f'shared_{cell}.json',out)
    print(cell,{name:{p:round(r['half']['relative_mse'],3) for p,r in m['paths'].items()} for name,m in out['models'].items()},flush=True)
    return out


def main():
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,c) for c in CELLS]): results.append(f.result())
    dump(HERE/'results/shared_head.json',sorted(results,key=lambda r:r['cell']))


if __name__=='__main__': main()
