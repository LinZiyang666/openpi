"""Cross-fit cheap motion residuals against policy shadows (not SR prediction).

Per-task ridge on deployed PCA keys, robot state, cached action and elapsed time.
All trajectories with the same init are in the same fold, INCLUDING transfer arms.
Gripper stays exactly the cache's command. Training episode weights are balanced.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.linalg import solve
from .common import HERE, DERIVED, SEED, load_compact, dump
from .shadows import errors, episode_mean
from ..inference import predict


def dataset(z):
    take=z['vision'] & np.isfinite(z['shadow_look_cache_chunk']).all((1,2)) & np.isfinite(z['state_norm']).all(1)
    x=np.concatenate([z['shadow_look_keys_pca_third'],z['shadow_look_keys_pca_wrist'],
                      z['state_norm'],z['shadow_look_cache_chunk'].reshape(len(take),-1),
                      np.minimum(z['seq'][:,None],120)/120],axis=1)[take]
    return dict(x=x,base=z['shadow_look_cache_chunk'][take],teacher=z['policy_shadow_chunk'][take],
                task=z['task'][take],init=z['init'][take],seq=z['seq'][take])


def fit_residual(x, residual, init, nonlinear=False):
    mean=x.mean(0);std=np.maximum(x.std(0),1e-4)
    xx=np.clip((x-mean)/std,-8,8)
    rng=np.random.default_rng(SEED)
    w=rng.normal(size=(x.shape[1],384))/np.sqrt(x.shape[1]) if nonlinear else None
    bias=rng.uniform(0,2*np.pi,384) if nonlinear else None
    features=np.c_[xx, np.cos(xx@w+bias)*np.sqrt(2)] if nonlinear else xx
    counts=np.bincount(init,minlength=30)
    weight=1/np.maximum(counts[init],1);weight*=len(weight)/weight.sum()
    target=residual.reshape(len(x),-1)
    fm=np.average(features,axis=0,weights=weight)
    ym=np.average(target,axis=0,weights=weight)
    centered=features-fm
    gram=centered.T@(centered*weight[:,None])+100.*np.eye(features.shape[1])
    coef=solve(gram,centered.T@((target-ym)*weight[:,None]),assume_a='pos')
    return dict(mean=mean,std=std,w=w,bias=bias,coef=coef.T,intercept=ym-fm@coef)


def one(meta):
    start=time.monotonic()
    name=meta['arm']
    datasets={'cache_path':dataset(load_compact(name))}
    ip=name[:-1]+'IP'
    if (DERIVED/'compact'/f'{ip}.npz').exists():
        datasets['random_call_path']=dataset(load_compact(ip))
    train=datasets['cache_path']
    outputs={label:{method:d['base'].copy() for method in ['linear','rff','rff_half']} for label,d in datasets.items()}
    full_models={}
    for task in range(10):
        for fold in range(5):
            tr=(train['task']==task)&(train['init']%5!=fold)
            for nonlinear in [False,True]:
                model=fit_residual(train['x'][tr],train['teacher'][tr,:,:6]-train['base'][tr,:,:6],train['init'][tr],nonlinear)
                key='rff' if nonlinear else 'linear'
                for label,d in datasets.items():
                    te=(d['task']==task)&(d['init']%5==fold)
                    residual=predict(model,d['x'][te])
                    outputs[label][key][te,:,:6]+=residual
                    if nonlinear:
                        outputs[label]['rff_half'][te,:,:6]+=.5*residual
        tr=train['task']==task
        full_models[task]=fit_residual(train['x'][tr],train['teacher'][tr,:,:6]-train['base'][tr,:,:6],train['init'][tr],True)
    out=dict(arm=name,fold='init modulo 5, shared across paths',training='A fresh observations only; policy shadows; all successes/failures',
             alpha=100.,random_features=384,gripper='unchanged',paths={})
    for label,d in datasets.items():
        mb,gb=errors(d['base'],d['teacher'])
        eb=episode_mean(mb,d['task'],d['init'])
        rows={}
        for method,action in outputs[label].items():
            m,g=errors(action,d['teacher'])
            e=episode_mean(m,d['task'],d['init'])
            rows[method]=dict(mse=float(e.mean()),relative_mse=float(e.mean()/eb.mean()-1),
                              task_relative=(e.mean(1)/eb.mean(1)-1).tolist(),
                              episode_win_fraction=float((e<eb).mean()),
                              late_relative_mse=float(episode_mean(m,d['task'],d['init'],d['seq']>=10).mean()/
                                                      episode_mean(mb,d['task'],d['init'],d['seq']>=10).mean()-1))
        out['paths'][label]=dict(n=len(d['task']),base_mse=float(eb.mean()),methods=rows)
        np.savez_compressed(DERIVED/'compact'/f'{name}_student_{label}_evaluation.npz',
                            task=d['task'],init=d['init'],base_error=mb,
                            **{method+'_error':errors(action,d['teacher'])[0] for method,action in outputs[label].items()})
    # Pure numerical serving artifact; no training observations are embedded.
    save={}
    for t,model in full_models.items():
        for key,value in model.items():
            save[f't{t}_{key}']=value.astype(np.float32)
    path=HERE/'artifacts'/f'{name}_student.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**save)
    out.update(artifact=str(path),seconds=time.monotonic()-start)
    return out


def main():
    metas=[json.loads(p.read_text()) for p in (DERIVED/'compact').glob('*.json')]
    metas=[m for m in metas if m['variant']=='A']
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for fut in as_completed([pool.submit(one,m) for m in metas]):
            r=fut.result();results.append(r);print(json.dumps(r),flush=True)
    dump(HERE/'results/student.json',results)


if __name__=='__main__':
    main()
