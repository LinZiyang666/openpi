"""Paired Cache row-removal analysis, with empty tasks scored as failures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from .audit_counts import HERE, read
from .build import cell_id
from .sources import lines,segments,r17_runs


def means(values, mt50):
    if not values:return dict(n=0,task_mean=None,group_mean=None)
    per=defaultdict(list)
    for task, success in values:per[str(task)].append(int(success))
    per_task={task:sum(v)/len(v) for task,v in per.items()}
    task_mean=sum(per_task.values())/len(per_task)
    group_mean=None;group_counts={};group_means={}
    if mt50:
        from exp.metaworld.tasks import DIFFICULTY
        groups=defaultdict(list)
        for task,value in per_task.items():groups[DIFFICULTY[task]].append(value)
        group_mean=sum(sum(v)/len(v) for v in groups.values())/len(groups)
        group_counts={str(k):len(v) for k,v in groups.items()}
        group_means={str(k):sum(v)/len(v) for k,v in groups.items()}
    return dict(n=sum(map(len,per.values())),task_mean=task_mean,group_mean=group_mean,per_task=per_task,
                difficulty_group_task_counts=group_counts,difficulty_group_means=group_means)


def score(cell, baseline, observed, *, rerun=None, rerun_required=(), control_sources=None):
    empty=set(map(str,cell['empty_tasks']))
    mt50=cell['suite']=='mt50'
    historical={(str(r['task']),r['init']):bool(r['success']) for r in baseline}
    old=dict(historical)
    required=set(rerun_required)
    rerun=rerun or {}
    baseline_complete=required<=rerun.keys()
    for key in required:old[key]=rerun.get(key)
    wanted={key for key in old if key[0] not in empty}
    filtered_complete=wanted<=observed.keys()
    complete=filtered_complete and baseline_complete
    primary=dict(observed)
    for key in old:
        if key[0] in empty:primary[key]=False
    secondary={key:value for key,value in primary.items() if key in wanted}
    def comparison(keys, got):
        original=means([(k[0],old[k]) for k in keys],mt50) if baseline_complete else None
        historical_sr=means([(k[0],historical[k]) for k in keys],mt50)
        filtered=means([(k[0],got[k]) for k in keys],mt50) if filtered_complete else None
        difference=({name:filtered[name]-original[name] for name in ('task_mean','group_mean')
                     if filtered[name] is not None} if complete else None)
        wins=sum(bool(got[k]) and not old[k] for k in keys) if complete else None
        losses=sum(not got[k] and old[k] for k in keys) if complete else None
        return dict(unfiltered=original,historical_unfiltered=historical_sr,filtered=filtered,
                    difference=difference,gains=wins,losses=losses)
    pairs=[dict(task=k[0],init=k[1],unfiltered=old[k],filtered=primary.get(k),
                historical_unfiltered=historical[k],
                difference=int(primary[k])-int(old[k]) if k in primary and old[k] is not None else None,
                automatic_failure=k[0] in empty) for k in sorted(old)]
    origins={(str(r['task']),r['init']):r['source'] for r in baseline}
    for pair in pairs:
        origin=origins[pair['task'],pair['init']]
        pair['historical_unfiltered_source']=origin
        pair['unfiltered_source']=(control_sources or {}).get(origin,origin)
    return dict(id=cell_id(cell),model=cell['model'],suite=cell['suite'],size=cell['size'],
                collected=cell['collected'],kept=cell['kept'],removed_failures=cell['failed'],
                empty_tasks=cell['empty_tasks'],automatic_failures=50*len(empty),
                planned_simulations=len(wanted),observed_simulations=len(wanted&observed.keys()),
                baseline_kind='rerun_control' if required else 'historical',
                rerun_required=len(required),rerun_observed=len(required&rerun.keys()),
                status='COMPLETE' if complete else 'PENDING',
                primary=comparison(set(old),primary),secondary=comparison(wanted,secondary)),pairs


def accepted(path):
    result={}
    for row in lines(path):
        if row.get('accepted') and not row.get('error') and row.get('success') is not None:
            uid=row['task_uid']
            if uid not in result or (row.get('ts') or 0)>(result[uid].get('ts') or 0):result[uid]=row
    return result


def collect(jobs, extra_runs=()):
    observed=defaultdict(dict)
    for job in jobs:
        root=Path(job['run'])
        if job['domain']=='r10_libero':
            for arm in job['arms']:
                short=arm.split('_')[2];size=int(arm.split('_')[3][1:])
                suite='libero_spatial' if short=='spatial' else 'libero_10'
                for uid,r in accepted(root/'runs'/arm/'client/journal.jsonl').items():
                    task,init=map(int,uid.rsplit(':',2)[-2:])
                    observed[f"{job['model']}/{suite}/{size}"][str(task),init]=bool(r['success'])
            continue
        merged=r17_runs(tuple(str(p) for p in segments(root)))
        for (suite,size),values in merged.items():
            observed[f"{job['model']}/{suite}/{size}"].update({key:r['success'] for key,r in values.items()})
    for root in extra_runs:
        campaign=read(Path(root)/'campaign.json')
        added=collect([dict(run=str(root),domain=campaign['domain'],model=campaign['model'])])
        for key,values in added.items():observed[key].update(values)
    return observed


def continuations(jobs):
    path=HERE/'CONTINUATIONS.json'
    if not path.exists():return []
    seen={str(Path(j['run']).resolve()) for j in jobs};result=[]
    pending=list(read(path)['jobs'])
    while pending:
        found=[j for j in pending if str(Path(j['source']).resolve()) in seen]
        if not found:break
        for job in found:
            pending.remove(job);root=str(Path(job['run']).resolve())
            if root not in seen:result.append(root);seen.add(root)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'analysis')
    p.add_argument('--r17-run',type=Path,action='append',default=[],help='Additional continuation campaigns; later supplied results take precedence')
    p.add_argument('--control-run',type=Path,action='append',default=[],help='Additional unfiltered-control continuations')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cells=read(HERE/'CELLS.json')['cells'];baseline=read(HERE/'BASELINE.json')
    jobs=read(HERE/'CAMPAIGNS.json')['jobs']
    controls=[j for j in jobs if j.get('arm')=='unfiltered_control']
    filtered=[j for j in jobs if j.get('arm')!='unfiltered_control']
    observed=collect(filtered,[*continuations(filtered),*args.r17_run])
    control_observed=collect(controls,[*continuations(controls),*args.control_run])
    control_sources={j['source']:j['run'] for j in controls}
    rows=[]
    with (args.output/'paired_by_init.jsonl').open('w') as pairs:
        for cell in cells:
            key=cell_id(cell)
            required={(str(r['task']),r['init']) for r in baseline[key] if r['source'] in control_sources}
            row,paired=score(cell,baseline[key],observed[key],rerun=control_observed[key],
                             rerun_required=required,control_sources=control_sources)
            rows.append(row)
            for pair in paired:pairs.write(json.dumps(dict(cell=row['id'],**pair))+'\n')
    (args.output/'RESULTS.json').write_text(json.dumps(dict(method='original size-N fit; successful candidate episodes only',
        baseline_policy='matched-environment rerun controls where required; historical baseline retained separately',
        unfiltered_control_jobs=[j['name'] for j in controls],rows=rows),indent=2)+'\n')
    flat=[]
    for r in rows:
        f={k:r[k] for k in ('model','suite','size','collected','kept','removed_failures','status','planned_simulations','observed_simulations','automatic_failures')}
        f['empty_tasks']=';'.join(map(str,r['empty_tasks']))
        f.update(baseline_kind=r['baseline_kind'],rerun_required=r['rerun_required'],rerun_observed=r['rerun_observed'])
        for scope in ('primary','secondary'):
            for name in ('task_mean','group_mean'):
                for arm in ('unfiltered','historical_unfiltered','filtered','difference'):
                    f[f'{scope}_{arm}_{name}']=(r[scope][arm] or {}).get(name)
        flat.append(f)
    with (args.output/'RESULTS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    def fmt(value):return 'pending' if value is None else f'{100*value:.2f}%'
    text=['# Frozen-fit failed-episode removal','',
          'Primary includes empty tasks as failures. Secondary uses the same non-empty tasks in both arms. MT50 reports equal-weight difficulty-group and task means. Missing simulated outcomes remain pending.','',
          '| Model / suite / size | Kept | Primary filtered / unfiltered task SR | Historical unfiltered SR | MT50 group SR filtered / unfiltered | Secondary filtered / unfiltered task SR | Empty tasks |',
          '|---|---:|---|---|---|---|---|']
    for r in rows:
        a=r['primary'];b=r['secondary'];af=a['filtered'] or {};bf=b['filtered'] or {}
        au=a['unfiltered'] or {};bu=b['unfiltered'] or {}
        group=f"{fmt(af.get('group_mean'))} / {fmt(au.get('group_mean'))}" if r['suite']=='mt50' else '—'
        text.append(f"| {r['id']} | {r['kept']} | {fmt(af.get('task_mean'))} / {fmt(au.get('task_mean'))} | {fmt(a['historical_unfiltered']['task_mean'])} | {group} | {fmt(bf.get('task_mean'))} / {fmt(bu.get('task_mean'))} | {', '.join(map(str,r['empty_tasks']))} |")
    (args.output/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(cells=len(rows),complete=sum(r['status']=='COMPLETE' for r in rows),output=str(args.output))))


if __name__=='__main__':main()
