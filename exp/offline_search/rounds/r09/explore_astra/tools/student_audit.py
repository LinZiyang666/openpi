"""Ablations, independent path transfer, movement-only and matched-noise audits."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from .common import HERE, DERIVED, load_compact, dump, paired_bootstrap
from .student import dataset, fit_residual, predict
from .shadows import errors, episode_mean


def summary(action,d):
    mb,_=errors(d['base'],d['teacher'])
    m,_=errors(action,d['teacher'])
    masks=dict(all=np.ones(len(m),bool),first_30_controls=d['seq']<6,
               moving_teacher=np.sqrt((d['teacher'][:,:,:6]**2).mean((1,2)))>.1,
               success=d['success'],failure=~d['success'])
    out={}
    for label,mask in masks.items():
        b=episode_mean(mb,d['task'],d['init'],mask)
        e=episode_mean(m,d['task'],d['init'],mask)
        valid=np.isfinite(b)&np.isfinite(e)
        out[label]=dict(episodes=int(valid.sum()),n=int(mask.sum()),base_mse=float(b[valid].mean()),
                       mse=float(e[valid].mean()),relative_mse=float(e[valid].mean()/b[valid].mean()-1))
        if label=='all':
            out[label]['paired_mse_difference']=paired_bootstrap(e-b,draws=3000)
    return out


def load_data(arm):
    z=load_compact(arm);d=dataset(z)
    d['success']=z['success'][d['task'],d['init']].astype(bool)
    return d


def one(meta):
    name=meta['arm'];train=load_data(name)
    datasets={'cache':train}
    for variant in ['IP','CU','W10']:
        path=DERIVED/'compact'/f'{name[:-1]+variant}.npz'
        if path.exists():
            datasets[variant]=load_data(name[:-1]+variant)
    predictions={label:{method:d['base'].copy() for method in
                  ['zero_motion','task_teacher_mean','scalar_shrink','no_vision_rff_half','student_half','student_full']}
                 for label,d in datasets.items()}
    for task in range(10):
        for fold in range(5):
            tr=(train['task']==task)&(train['init']%5!=fold)
            residual=train['teacher'][tr,:,:6]-train['base'][tr,:,:6]
            full=fit_residual(train['x'][tr],residual,train['init'][tr],True)
            no_vision=fit_residual(train['x'][tr,128:],residual,train['init'][tr],True)
            mean=train['teacher'][tr,:,:6].mean(0)
            a=train['base'][tr,:,:6];b=train['teacher'][tr,:,:6]
            scale=np.clip((a*b).sum()/np.maximum((a*a).sum(),1e-9),0,2)
            for label,d in datasets.items():
                te=(d['task']==task)&(d['init']%5==fold)
                predictions[label]['zero_motion'][te,:,:6]=0
                predictions[label]['task_teacher_mean'][te,:,:6]=mean
                predictions[label]['scalar_shrink'][te,:,:6]*=scale
                predictions[label]['no_vision_rff_half'][te,:,:6]+=.5*predict(no_vision,d['x'][te,128:])
                correction=predict(full,d['x'][te])
                predictions[label]['student_half'][te,:,:6]+=.5*correction
                predictions[label]['student_full'][te,:,:6]+=correction
    result=dict(arm=name,paths={label:{method:summary(a,datasets[label]) for method,a in methods.items()}
                                for label,methods in predictions.items()})
    print(name,{label:{m:round(s['all']['relative_mse'],3) for m,s in methods.items()} for label,methods in result['paths'].items()},flush=True)
    return result


def main():
    metas=[json.loads(p.read_text()) for p in (DERIVED/'compact').glob('*.json')]
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,m) for m in metas if m['variant']=='A']):
            results.append(f.result())
    dump(HERE/'results/student_audit.json',results)


if __name__=='__main__':
    main()
