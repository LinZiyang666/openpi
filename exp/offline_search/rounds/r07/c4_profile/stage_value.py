"""Supported causal call excursions under the recorded R6 lottery continuation."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import common as C


def score(z, p, episode_probability=1., supported_doses=1, target='natural'):
    """Two randomization levels, never use (pi*p) as a Bernoulli propensity.

    Natural-dose mixture: the realized dose distribution is the target, so the
    dose weight cancels. Equal supported-dose target: g(d)/pi(d), g(d)=1/K.
    The subsequent CACHE propensity is 1-p conditional on that same dose.
    """
    if not 0 < p < 1:
        raise ValueError('forced/end-point anchor has no local treatment support')
    if not 0 < episode_probability <= 1 or supported_doses < 1:
        raise ValueError('invalid episode-dose propensity')
    dose_weight = 1. if target=='natural' else 1./(supported_doses*episode_probability)
    return dose_weight*(z/p-(1-z)/(1-p)), dose_weight


def observed_support(interval, treatments):
    """Keep the HT point diagnostic, withhold resampled CIs missing either arm."""
    present=set(treatments)
    if present!={0,1}:
        return {**interval,'lo':None,'hi':None,
                'interval_status':'withheld_missing_observed_call_or_cache'}
    return {**interval,'interval_status':'reported_exploratory'}


def candidates(run_root, arm=None):
    if arm:
        return [Path(run_root)/'runs'/v for v in arm]
    return sorted(p for p in (Path(run_root)/'runs').iterdir() if p.is_dir() and ('_U18' in p.name or '_U30' in p.name or '_risk_rho' in p.name))


def audit_arm(root, cell, reps):
    episodes,audit=C.accepted_arm(root)
    base,lib,manifest=C.bank(cell);table=C.stage_table(lib,manifest,base)
    data=[];unavailable=forced=0
    supported_doses=defaultdict(set)
    for ep in episodes:
        anchors=[]
        decisions=ep['decisions']
        for d in decisions:
            if not d['vision']:continue
            x=d.get('extras',{})
            if 'os_q2_p_call' not in x:raise ValueError('lottery telemetry missing')
            p=float(x['os_q2_p_call']);z=int(not d['hit'])
            pi=float(x['os_q2_episode_propensity']);dose=float(x['os_q2_episode_dose'])
            if p!=dose or not 0 < pi <= 1 or z!=int(float(x['os_q2_coin'])<p):
                raise ValueError('invalid conditional dose/coin/treatment')
            if not 0<p<1:forced+=1;continue
            supported_doses[ep['task']].add(dose)
            rows,w=d.get('rows',[]),d.get('weights',[])
            if len(rows)!=base.k or len(w)!=base.k:
                unavailable+=1;continue
            if d['lib']!=lib.dir.name or np.any(lib['task_id'][rows]!=ep['task']):
                raise ValueError('library/row task mismatch')
            s=int(d['step']);future=decisions[s:]
            cost=sum((.152 if cell.startswith('pi05_') else .148)*int(v['vision']) + (.848 if cell.startswith('pi05_') else .852)*int(not v['hit']) for v in future)
            anchors.append(dict(task=ep['task'],init=ep['init'],uid=ep['uid'],step=s,Y=ep['Y'],
                stage=C.stage_label(table,rows,w),p=p,Z=z,pi=pi,dose=dose,
                future_decisions=len(future),future_calls=sum(not v['hit'] for v in future),future_cost=cost))
        data.extend(anchors)
    # Recover the support menu from the frozen fitted budget, not the observed dose census.
    armrows=json.loads((root.parents[1]/'arms.json').read_text())
    spec=next(r for r in armrows if r.get('arm',r.get('name'))==root.name)
    fit=Path(spec['plugin_args'][spec['plugin_args'].index('--os-fit-artifact')+1])
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
    with fit.open('rb') as f: method=FitUnpickler(f).load()['method']
    from exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.budget import DOSES
    K={int(t):sum(0<float(d)<1 and float(w)>0 for d,w in zip(DOSES,b['weights'])) for t,b in method.q2_budget['tasks'].items()}
    for d in data:
        weights=method.q2_budget['tasks'][d['task']]['weights']
        index=next(i for i,v in enumerate(DOSES) if float(v)==d['dose'])
        if not np.isclose(weights[index],d['pi'],rtol=0,atol=1e-12):raise ValueError('dose probability differs from frozen fit')
    stages=sorted({r['stage'] for r in data})
    results=[]
    for estimand in ['first_entry','occupancy']:
        for stage in stages:
            g=[r for r in data if r['stage']==stage]
            if estimand=='first_entry':
                first={}
                for r in g:first.setdefault(r['uid'],r)
                g=list(first.values())
            for target in ['natural','equal_supported_doses']:
                # Cross-fit outcome centering on the other init parity within task.
                values=[];cost_values=[];call_values=[];dose_weights=[]
                for r in g:
                    other=[e['Y'] for e in episodes if e['task']==r['task'] and e['init']%2!=r['init']%2]
                    b=float(np.mean(other)) if other else 0.
                    s,w=score(r['Z'],r['p'],r['pi'],K[r['task']],target)
                    values.append(s*(r['Y']-b));cost_values.append(s*r['future_cost']);call_values.append(s*r['future_calls']);dose_weights.append(w)
                # Sum supported anchor scores per episode. Dividing by realized
                # stage occupancy would condition on a post-treatment quantity.
                target_episodes=[e for e in episodes if target=='natural' or K[e['task']]>0]
                def episode_scores(v):
                    sums=defaultdict(float)
                    for r,x in zip(g,v):sums[r['uid']]+=x
                    return [sums[e['uid']] for e in target_episodes]
                es=episode_scores(values)
                treatment=[r['Z'] for r in g]
                interval=observed_support(C.task_bootstrap(target_episodes,es,reps,alpha=.05/max(len(stages),1)),treatment)
                costs=observed_support(C.task_bootstrap(target_episodes,episode_scores(cost_values),reps),treatment)
                calls=observed_support(C.task_bootstrap(target_episodes,episode_scores(call_values),reps),treatment)
                treated=np.asarray([w*r['Z']/r['p'] for r,w in zip(g,dose_weights)])
                control=np.asarray([w*(1-r['Z'])/(1-r['p']) for r,w in zip(g,dose_weights)])
                ess=lambda w:float(w.sum()**2/(w@w)) if w@w else 0.
                results.append(dict(stage=stage,estimand=estimand,dose_target=target,anchors=len(g),
                    calls=sum(r['Z'] for r in g),propensity_min=min(r['p'] for r in g),propensity_max=max(r['p'] for r in g),
                    episode_probability_min=min(r['pi'] for r in g),effect=interval,
                    downstream_cost_effect=costs,downstream_call_effect=calls,
                    ESS_call=ess(treated),ESS_cache=ess(control),
                    unsupported_equal_dose_tasks=[t for t,k in K.items() if k==0],
                    held_task_estimates={str(t):float(np.mean([v for r,v in zip(target_episodes,es) if r['task']!=t])) for t in sorted({r['task'] for r in g})},
                    init_fold_estimates={str(f):float(np.mean([v for r,v in zip(target_episodes,es) if r['init']%2==f])) for f in [0,1]}))
    audit.update(arm=root.name,cell=cell,accepted_decisions=sum(len(e['decisions']) for e in episodes),
        supported_anchors=len(data),forced_or_endpoint=forced,unavailable_full_kernel=unavailable,
        supported_dose_counts=K,SR_descriptive=np.mean([e['Y'] for e in episodes]))
    return dict(audit=audit,results=results),data


def main():
    p=C.parser(__doc__)
    p.add_argument('--run-roots',nargs='+',type=Path,default=[C.RUNS/'r06_c_validation',C.RUNS/'r06_frontier'])
    p.add_argument('--arms',nargs='+');p.add_argument('--reps',type=int,default=2000)
    a=p.parse_args();reports=[];rows=[]
    for run in a.run_roots:
        for root in candidates(run,a.arms):
            cell=root.name.removeprefix('r6c_').removeprefix('q2_').split('_risk_rho')[0].rsplit('_U',1)[0]
            if cell not in C.cells(a.cells):
                # Frontier names need their authoritative arm-row library-size metadata.
                pts=list(__import__('csv').DictReader(open('exp/offline_search/rounds/r06/frontier_final/frontier_points.csv')))
                match=next((r for r in pts if r['run']==run.name and r['arm']==root.name),None)
                if match is None or match['cell'] not in C.cells(a.cells):continue
                cell=match['cell']
            report, rr=audit_arm(root,cell,a.reps);reports.append(report)
            rows.extend(dict(run=run.name,arm=root.name,cell=cell,**r) for r in rr)
            print(root.name,report['audit'],flush=True)
    if not reports:raise ValueError('no matching lottery arms')
    C.write(a.out/'stage_value.json',dict(schema='r7.c4.value.v1',reports=reports,
        caveat='Local supported call excursions under each original dose/continuation, not an entire stage-routing policy. Pre-coin frozen stages; first entry is first supported encounter, with zero contribution for unreached/endpoint episodes. Occupancy sums episode anchor scores: a derivative of a common stage probability shift, not a per-call ATE. No conditioning on realized stage count. Two-level assignment uses g(d)/pi(d) times Z/p-(1-Z)/(1-p); natural dose mixture cancels pi. Equal supported-dose target excludes endpoint doses at level one. Stage intervals use Bonferroni within arm/estimand/target; held-task/fold checks are diagnostics. No serving table fitted to outcomes. Untried camera/cadence packages unsupported.'))
    C.csv_write(a.out/'stage_value_anchors.csv',rows)


if __name__=='__main__':main()
