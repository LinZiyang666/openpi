"""Read-only formal acceptance, including native legacy journal continuation."""
from collections import Counter, defaultdict, deque
import csv
from pathlib import Path

from .compare import rows, wait_io
from .node import atomic, read
from .policy import GIB


def curve(scopes, output, *, rate_scopes=()):
    samples=[];accepted=[]
    for scope in scopes:
        for name in ('AUTOSCALE_DECISIONS.jsonl','AUTOSCALE_SAMPLES.jsonl'):
            path=scope/name
            if path.exists():samples += [(r,scope) for r in rows(path)]
    for scope in dict.fromkeys([*scopes,*rate_scopes]):
        path=scope/'AUTOSCALE_ACCEPTED.jsonl'
        if path.exists():accepted += list(rows(path))
    accepted.sort(key=lambda r:r['time']); samples.sort(key=lambda r:r[0]['time'])
    pending=iter(accepted);head=next(pending,None);windows=defaultdict(deque);totals=Counter();result=[]
    for r,scope in samples:
        if r['node']=='global':continue
        at=r['time'];node=r['node']
        while head is not None and head['time']<=at:
            windows[head['node']].append(head['time']);totals[head['node']]+=1;head=next(pending,None)
        while windows[node] and windows[node][0]<=at-60:windows[node].popleft()
        counts=r['counts'];signals=r['signals']
        result.append(dict(time=at,node=node,scope=str(scope),active=counts.get('active'),
            target=counts.get('target'),draining=counts.get('draining',0),delta=r['delta'],reason=r['reason'],
            accepted_episodes_per_min=len(windows[node]),accepted_total=totals[node],window_seconds=60,
            minimum_available_gib=(signals['minimum_available']/GIB if signals.get('minimum_available') is not None else None)))
    columns=['time','node','scope','active','target','draining','delta','reason',
             'accepted_episodes_per_min','accepted_total','window_seconds','minimum_available_gib']
    with output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(result)
    return result,accepted


def audit(name, output):
    from .formal_dynamic import jobs, legacy_plan
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    expected=Counter();accepted=Counter();errors=[];scopes=[];closed=True;no_probe_conflicts=True;new_episodes=0
    for j in jobs(name):
        wait_io();root=Path(j['run']);meta=read(root/'FORMAL_DYNAMIC.json')
        if not meta:raise ValueError('no private formal preparation receipt: '+str(root))
        if j['pipeline']=='R17':
            tasks=read(root/'EPISODE_PLAN.json')['tasks'];journals=[root/'driver/journal.jsonl']
            expected.update(t['task_uid'] for t in tasks);new_episodes+=len(tasks)
            scopes += [root/'workers'/n for n in meta['desired_maxima']]
            receipt=read(root/'RECEIPT.json',{})
            closed &= receipt.get('status')=='PASS' and receipt.get('all_owned_closed') is True
            no_probe_conflicts &= not read(root/'driver/rejected_worker_probes.json',[])
        else:
            arms=read(root/'arms.json');expected.update(t['task_uid'] for t in legacy_plan(root,arms))
            journals=[]
            for row in arms:
                journal=root/'runs'/row['arm']/'client/journal.jsonl';journals.append(journal)
                closed &= any((root/'state').glob(row['arm']+'*.DONE'))
                prior=any(n.startswith(row['arm']+'.') for n in meta['retained_done_arms'])
                if not prior:scopes.append(journal.parent/'autoscale');new_episodes+=500
        for journal in journals:
            if not journal.exists():continue
            for row in rows(journal):
                if row.get('accepted') is not False and row.get('status') in ('done','failed'):
                    accepted[row['task_uid']]+=1
                    if row.get('error'):errors.append(dict(task_uid=row['task_uid'],error=row['error']))
    evidence=[];events=[];minima=[]
    for scope in scopes:
        evidence.append(read(scope/'AUTOSCALE_KERNEL.json',{}))
        for n in ('AUTOSCALE_DECISIONS.jsonl','AUTOSCALE_SAMPLES.jsonl'):
            p=scope/n
            if p.exists():
                for r in rows(p):
                    events.append(r);v=r['signals'].get('minimum_available',r['signals'].get('available'))
                    if v is not None:minima.append(v)
    recycled=sum(r['reason']=='generation_exit' and bool(r['signals'].get('complete')) for r in events)
    controller_scopes=[]
    for j in jobs(name):
        root=Path(j['run'])
        if j['pipeline']=='R17':controller_scopes.append(root)
    csv_rows,rate_events=curve(scopes,output/'AUTOSCALE_ACTIVE.csv',rate_scopes=controller_scopes)
    checks=dict(every_uid_once=bool(expected) and accepted==expected and all(v==1 for v in accepted.values()),
        accepted_errors_zero=not errors,native_closed_success=bool(closed),no_probe_conflicts=bool(no_probe_conflicts),
        recycled_generations=recycled>=2 if new_episodes>=64 else True,
        minimum_available_recorded=bool(minima),above_critical=bool(minima) and min(minima)>=10*GIB,
        zero_oom=bool(evidence) and all(e.get('status')=='PASS' and e.get('coverage') is True and not e.get('oom_lines') for e in evidence),
        every_count_change_has_reason=bool(events) and all(r.get('reason') for r in events),
        per_node_active_and_rate=bool(csv_rows) and bool(rate_events))
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',name=name,checks=checks,
        expected=sum(expected.values()),accepted=sum(accepted.values()),unique_accepted=len(accepted),
        missing=len(expected.keys()-accepted.keys()),errors=errors,recycled_generations=recycled,
        minimum_available_gib=min(minima)/GIB if minima else None,
        kernel_evidence=evidence,kernel_window='new dynamic executions only; retained OFF outcomes are not re-certified',
        rate_window_seconds=60,curve=str(output/'AUTOSCALE_ACTIVE.csv'))
    atomic(output/'AUTOSCALE_ACCEPTANCE.json',result);return result
