"""Per-suite paired SR plus task-cluster sensitivity and censored first divergence.

Inputs are normalized accepted outcomes / aligned decision traces. Missing
observations/actions are reported unavailable, never inferred from SR or timing.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from artifacts import write_json
from paired_sr import paired_interval
from paired_identity import check as check_identity

KEY=('model','suite','arm','task_id','init_idx','env_seed','policy_seed')


def indexed(rows,trace=False,clock='decision_step'):
    result={}
    for row in rows:
        key=tuple(row[k] for k in KEY)+(tuple([row[clock]]) if trace else ())
        if key in result:raise ValueError('duplicate paired identity')
        if not trace and row.get('accepted') is not True:raise ValueError('unaccepted episode cannot enter SR')
        result[key]=row
    return result


def sr(left,right,resamples=10000):
    bindings=check_identity(left,right)
    a,b=indexed(left),indexed(right)
    if set(a)!=set(b):raise ValueError('unmatched outcomes; missing/failed episodes must be resolved')
    groups=defaultdict(list)
    for key in sorted(a):groups[key[:3]].append(key)
    result=[]
    for group,keys in groups.items():
        report=paired_interval([a[k]['success'] for k in keys],[b[k]['success'] for k in keys])
        clusters=defaultdict(list)
        for k in keys:clusters[k[3]].append(int(a[k]['success'])-int(b[k]['success']))
        totals=np.array([sum(v) for v in clusters.values()]);counts=np.array([len(v) for v in clusters.values()])
        rng=np.random.default_rng(1017)
        draws=rng.integers(0,len(totals),size=(resamples,len(totals)))
        samples=totals[draws].sum(axis=1)/counts[draws].sum(axis=1)
        report.update(group=list(group),task_clusters=len(totals),
            scientific_binding=next(r for r in bindings if r['group']==list(group)),
            cluster_bootstrap_CI95=np.quantile(samples,[.025,.975]).tolist(),
            cluster_caution='sensitivity only; few tasks and zero discordance can make bootstrap degenerate; exact CI remains required')
        report['paired_CI_within_margin']=report['equivalent']
        report['equivalent']=report['equivalent'] and report['scientific_binding']['qualified']
        report['equivalence_scope']='paired SR with matching scientific identities; layers (a)/(b) remain separately required'
        result.append(report)
    return result


def divergence(left,right,action_atol,state_atol,clock='decision_step'):
    if clock not in ('decision_step','env_step'):raise ValueError('unknown alignment clock')
    bindings=check_identity(left,right)
    a,b=indexed(left,True,clock),indexed(right,True,clock)
    episodes=sorted({k[:-1] for k in a}|{k[:-1] for k in b})
    report=[]
    for episode in episodes:
        ka={k[-1] for k in a if k[:-1]==episode};kb={k[-1] for k in b if k[:-1]==episode}
        common=sorted(ka&kb)
        row=dict(episode=list(episode),first_action=None,first_state=None,first_cache=None,first_images=None,
                 observed_steps=len(common),unmatched_steps=sorted(ka^kb),
                 last_common_step=max(common) if common else None,unavailable={})
        for field,threshold,label in [('actions',action_atol,'action'),('state',state_atol,'state'),('cache_decision',None,'cache'),
                                     ('camera_sha256',None,'images')]:
            missing=[]
            for step in common:
                x,y=a[episode+(step,)],b[episode+(step,)]
                if x.get('env_step')!=y.get('env_step'):
                    raise ValueError('decision clocks map to different execution steps')
                if field not in x or field not in y:
                    missing.append(step);continue
                if threshold is None:different=x[field]!=y[field]
                else:
                    u,v=np.asarray(x[field],float),np.asarray(y[field],float)
                    if u.shape!=v.shape or not np.isfinite(u).all() or not np.isfinite(v).all():raise ValueError('invalid trace tensor')
                    different=bool(np.max(np.abs(u-v),initial=0)>threshold)
                if different and row['first_'+label] is None:row['first_'+label]=step
            row['unavailable'][label]=missing
            row[label+'_right_censored']=bool(common) and not missing and not row['unmatched_steps'] and row['first_'+label] is None
            first=row['first_'+label]
            row['first_'+label+'_is_exact']=first is not None and common[0]==0 and not any(
                step<first for step in missing+row['unmatched_steps'])
        report.append(row)
    return dict(episodes=report,action_atol=action_atol,state_atol=state_atol,alignment_clock=clock,scientific_bindings=bindings,
        observation_scope='first Action = first differing returned full chunk at an observed query; not necessarily first executed control. State = observed wire proprio, not hidden simulator state. Images = exact camera-byte hash mismatch.',
        warning='native long-segment tails repeat decision_step: use env_step for full delivery traces; missing steps are not right-censored equivalence',
        exact_first_action_distribution=dict(sorted(__import__('collections').Counter(
            str(r['first_action']) for r in report if r['first_action_is_exact']).items())))


def read(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['sr','divergence','controls']);p.add_argument('left');p.add_argument('right')
    p.add_argument('--output',required=True);p.add_argument('--action-atol',type=float,default=.001)
    p.add_argument('--state-atol',type=float,default=1e-6)
    p.add_argument('--trace-clock',choices=('decision_step','env_step'),default='env_step')
    p.add_argument('--outcomes-left');p.add_argument('--outcomes-right')
    a=p.parse_args();l,r=read(a.left),read(a.right)
    if a.mode=='controls':
        from executed_controls import compare
        if bool(a.outcomes_left)!=bool(a.outcomes_right):p.error('both paired outcome populations required')
        population=read(a.outcomes_left)if a.outcomes_left else None
        if population is not None:
            other=read(a.outcomes_right)
            if set(indexed(population))!=set(indexed(other)):raise ValueError('unmatched execution populations')
            check_identity(population,other)
        result=compare(l,r,a.action_atol,population)
    else:result=sr(l,r)if a.mode=='sr'else divergence(l,r,a.action_atol,a.state_atol,a.trace_clock)
    write_json(a.output,result)
