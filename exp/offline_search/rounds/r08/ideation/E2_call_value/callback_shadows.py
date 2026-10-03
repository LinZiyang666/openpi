"""Run supplied shadow tools, plus same-state tail/fresh versus policy comparison."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from callback_calls import RUN, OUT, DiscoveryArm, frozen_r7, save
from exp.offline_search.debug.tools.decision import common as C
from exp.offline_search.debug.tools.decision import divergence, follow_vs_look


def one(name):
    assert (RUN/'state'/f'{name}.AUG_DONE').exists(), 'Only augmentation-complete arms'
    start=time.monotonic()
    print('start',name,flush=True)
    arm=DiscoveryArm(name)
    frozen_r7(arm)
    arm.stage_column='r7_stage'
    target=OUT/'shadows'/name
    target.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(bootstraps=1000,seed=20261001)
    div=divergence.analyze(arm,args)
    div_summary={k:v for k,v in div.items() if k!='tables'}
    pd.DataFrame(div['tables']['curves']).to_csv(target/'divergence_curves.csv',index=False)
    pd.DataFrame(div['tables']['noise_floor']).drop(columns=['stage_mass'],errors='ignore').to_parquet(target/'noise_floor.parquet',index=False)
    offsets=pd.DataFrame(div['tables']['offsets'])
    compact=offsets[offsets.executed_offset.eq(True)].drop(columns=['stage_mass'],errors='ignore')
    compact.to_parquet(target/'divergence_applied_offsets.parquet',index=False)
    del div,offsets,compact
    print('divergence',name,round(time.monotonic()-start,1),flush=True)
    follow=follow_vs_look.analyze(arm,args)
    follow_summary={k:v for k,v in follow.items() if k!='tables'}
    pd.DataFrame(follow['tables']['curves']).to_csv(target/'follow_curves.csv',index=False)
    pd.DataFrame(follow['tables']['decisions']).drop(columns=['stage_mass'],errors='ignore').to_parquet(target/'follow_decisions.parquet',index=False)
    # Profile geometric difference and its direction relative to a shared teacher.
    ds,eps=C.inputs(arm)
    ids=ds.decision_id.tolist()
    live=C.arrays(arm,['served_chunk','cache_chunk'],ids)
    fresh=C.pick(C.augmentation(arm,'shadow_look',ids),'cache_chunk')
    policy=C.pick(C.augmentation(arm,'policy_shadow',ids),'chunk')
    dim,scale,grip,threshold,units=C.dimensions(arm,live['served_chunk'].shape[-1])
    motion=[v for v in dim if v!=grip]
    records=[]
    factual_checks=[]
    for i,r in enumerate(ds.to_dict('records')):
        n=int(r['n_applied'])
        if r['vision'] and 'cache_chunk' in live:
            cache=live['cache_chunk'][i][:n,:]
            other=fresh[i][:n,:]
            if np.isfinite(cache[:,dim]).all() and np.isfinite(other[:,dim]).all():
                factual_checks.append(dict(decision_id=r['decision_id'],max_abs=float(np.max(np.abs(cache[:,dim]-other[:,dim]))),
                                           motion_rms=float(np.sqrt(np.mean(((cache[:,motion]-other[:,motion])/scale[motion])**2)))))
        if r['vision'] or r['src']=='policy_tail':
            continue
        tail=live['served_chunk'][i][:n,:]
        look=fresh[i][:n,:]
        teacher=policy[i][:n,:]
        if not all(np.isfinite(x[:,dim]).all() for x in (tail,look,teacher)):
            continue
        rms=lambda x,d:float(np.sqrt(np.mean(((x[:,d]-teacher[:,d])/scale[d])**2)))
        rec=dict(C.identity(r),arm=name,src=r['src'],age=r['blind_age_controls'],n_applied=n,
                 tail_rms=rms(tail,dim),fresh_rms=rms(look,dim),
                 tail_motion_rms=rms(tail,motion),fresh_motion_rms=rms(look,motion),units=units)
        rec.pop('stage_mass',None)
        rec.update(improvement=rec['tail_rms']-rec['fresh_rms'],
                   motion_improvement=rec['tail_motion_rms']-rec['fresh_motion_rms'])
        records.append(rec)
    pd.DataFrame(records).to_parquet(target/'look_policy.parquet',index=False)
    pd.DataFrame(factual_checks).to_parquet(target/'fresh_anchor_checks.parquet',index=False)
    summary=dict(arm=name,seconds=time.monotonic()-start,discovery_episodes=len(eps),decisions=len(ds),
                 divergence=div_summary,follow=follow_summary,blind_comparisons=len(records),
                 fresh_anchor_diagnostics=dict(n=len(factual_checks),exact=sum(r['max_abs']==0 for r in factual_checks),
                     median_motion_rms=float(np.median([r['motion_rms'] for r in factual_checks])) if factual_checks else None)
    )
    save(target/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--arms',nargs='+',required=True)
    a=p.parse_args()
    for name in a.arms:
        if not (OUT/'shadows'/name/'summary.json').exists():
            one(name)
