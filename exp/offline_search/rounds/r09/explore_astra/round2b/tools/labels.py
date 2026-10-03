"""Causal-time predicate labels, plus explicitly future offline targets.

Controls contain AFTER-state. Anchor state is the last control strictly BEFORE
the current decision, including initial settle. Never use current action effects
as pre-treatment inputs. Predicates are privileged LABELS, never serving inputs.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from .data import HERE,DERIVED,CELLS,source,safe_episode,dump


def targets(predicates, decision_seq, is_settle, anchors, success):
    if not np.isfinite(predicates).all():raise ValueError('Missing predicates')
    p=predicates>.5
    live=np.flatnonzero(~is_settle)
    if len(live)==0:raise ValueError('No policy controls')
    first=int(live[0])
    if first==0:raise ValueError('Initial settle predicate required')
    initial=p[first-1].copy()
    # New goal = first-ever satisfaction of a goal initially false.
    ever=np.logical_or.accumulate(p[first:],axis=0)|initial
    novel=(ever&~initial).sum(1)
    steps=np.arange(len(live))
    if not np.array_equal(live,np.arange(first,len(p))):raise ValueError('Noncontiguous live controls')
    before=np.searchsorted(decision_seq[first:],anchors,side='left')-1
    prior=np.maximum(before,0)
    count=novel[prior].copy();count[before<0]=0
    events=np.flatnonzero(np.diff(np.r_[0,novel])>0)
    last=np.full(len(anchors),-1,int)
    for j,b in enumerate(before):
        e=events[events<=b]
        if len(e):last[j]=int(e[-1])
    age=np.where(last>=0,before-last,before+1)
    current=p[np.maximum(first+before,first-1)].sum(1)
    out=dict(new_goals=count,goal_age=age,current_goals=current,
        n_goals=np.full(len(anchors),p.shape[1]),
        prior_controls=before+1,remaining_controls=len(live)-before-1)
    for horizon in [30,50,100]:
        end=np.minimum(before+horizon,len(live)-1)
        progressed=novel[end]>count
        terminal=(end==len(live)-1)&bool(success)
        out[f'progress{horizon}']=(progressed|terminal).astype(float)
        out[f'valid{horizon}']=(before+horizon<len(live))|bool(success)
    out['late_stall']=(count>0)&(age>=60)&(current<p.shape[1])
    out['late_trap100']=out['late_stall']&(out['progress100']==0)
    return out


def one(cell,variant):
    d=source(cell,variant)
    if d is None:return None
    out={};episodes=[]
    for ep in np.unique(d['episode']):
        take=np.flatnonzero(d['episode']==ep);t,i=int(d['task'][take[0]]),int(d['init'][take[0]])
        meta,c=safe_episode(cell,variant,t,i,d['decision_id'][take])
        if bool(meta['success'])!=bool(d['success'][take[0]]):raise ValueError('accepted outcome mismatch')
        y=targets(c['predicates'],c['decision_seq'],c['is_settle'],d['seq'][take],meta['success'])
        for k,v in y.items():
            if k not in out:out[k]=np.empty(len(d['seq']),v.dtype)
            out[k][take]=v
        episodes.append(dict(task=t,init=i,success=bool(meta['success']),
            new_goals=int(y['new_goals'].max()),late_stall_anchors=int(y['late_stall'].sum()),
            task_uid=meta['task_uid'],source_stat_sha256=meta['source_stat_sha256']))
    DERIVED.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(DERIVED/f'{cell}_{variant}_labels.npz',
        task=d['task'],init=d['init'],decision_id=d['decision_id'],**out)
    # Reporting populations remain separated by init; never evaluate fit on fit.
    summary=dict(cell=cell,variant=variant,fit_inits=list(range(20)),eval_inits=list(range(20,30)),
        episodes=len(episodes),anchors=len(d['seq']),records=episodes)
    dump(HERE/'results'/f'labels_{cell}_{variant}.json',summary)
    print(cell,variant,'labelled',len(episodes),len(d['seq']),flush=True)
    return {k:v for k,v in summary.items() if k!='records'}


def main():
    result=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(one,c,v) for c in CELLS for v in ['A','CU','IP']]
        for f in as_completed(fs):
            r=f.result()
            if r:result.append(r)
    dump(HERE/'results/label_population.json',result)


if __name__=='__main__':main()
