"""Preserve each resumed MT50 episode's actual runtime and worker topology."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .common import HERE, EVAL, read, dump
from exp.offline_search.rounds.r17.filter_failed.sources import segments, lines


def audit(source):
    plans={};accepted={}
    for segment in segments(Path(source)):
        path=segment/('FULL_EPISODE_PLAN.json' if (segment/'FULL_EPISODE_PLAN.json').exists() else 'EPISODE_PLAN.json')
        plans.update((t['task_uid'],t) for t in read(path)['tasks'])
        for row in lines(segment/'driver/journal.jsonl'):
            if not row.get('accepted') or row.get('error') or row.get('status') not in ('done','failed'):continue
            uid=row['task_uid'];stamp=row.get('ts') or 0
            if uid not in accepted or stamp>accepted[uid][1]:accepted[uid]=(segment,stamp)
    grouped=defaultdict(list)
    for uid,(source,_) in accepted.items():grouped[str(source)].append(plans[uid])
    if sum(map(len,grouped.values()))!=90000:raise ValueError('Source full π0.5 knob lane is not complete: stop rather than infer an origin')
    return grouped


def main():
    from .build import build
    path=HERE/'CAMPAIGNS.json';manifest=read(path)
    old=next((j for j in manifest['jobs'] if j['name']=='ffk1015_pi05_mt50_full_descent'),None)
    if old is None:
        print('Source-origin split already published');return
    grouped=audit(old['source'])
    dump(HERE/'MT50_PI05_FULL_ORIGINS.json',dict(source=old['source'],episodes=90000,
        groups=[dict(source=s,episodes=len(rows)) for s,rows in grouped.items()]))
    new=[]
    for i,(source,records) in enumerate(grouped.items(),1):
        name=f'ffk1015_pi05_mt50_full_descent_{i:02d}'
        root=EVAL/name
        row=read(root/'FFK_DEFINITION.json') if root.exists() else build(source,name,records=records)
        if row['episodes']!=len(records):raise ValueError('Origin split plan differs')
        new.append(dict(row,mode='full',stage='descent'))
        print(name,len(records),source,flush=True)
    jobs=[]
    for j in manifest['jobs']:
        jobs.extend(new if j['name']==old['name'] else [j])
    dump(path,dict(jobs=jobs))
    # Retain the superseded, never-launched draft. It is not in any queue.
    original=Path(old['run'])
    if original.exists():
        if (original/'LAUNCH.json').exists():raise ValueError('Draft was launched')
        original.rename(original.with_name(original.name+'.DRAFT_WHOLE_LANE_NOT_FOR_LAUNCH'))


if __name__=='__main__':main()
