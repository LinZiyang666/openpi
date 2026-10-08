"""Hash-audited reuse of completed B runs, including one legacy resume log gap.

Never fabricate server decisions. Pure C10's client receipts identify N and
V=M exactly: every searched decision is a MISS, every blind HIT is its one
policy tail. Missing historical miss_k rows remain an explicit audit limitation.
"""
import json
from pathlib import Path

import numpy as np

from . import analyze as a
from .cost import pricing


def load_completed(record,name,expected,pool,k,reader):
    try:
        return a.load(record,name,expected,pool,k,reader)
    except ValueError as error:
        if str(error)!='Noncontiguous decision steps' or pool!='B':raise
    root=Path(record['root']);row=next(r for r in reader.json(root/'arms.json') if r['arm']==name)
    a.require(row['r14_descent_level']=='c10' and row.get('r16_k')==k,'Log-gap recovery is pure R16 C10 only')
    run=root/'runs'/name;pairs=sorted(map(tuple,expected));positions={p:i for i,p in enumerate(pairs)}
    accepted={}
    for r in reader.jsonl(run/'client/journal.jsonl'):
        if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error'):accepted[r['task_uid']]=r
    server={}
    for path in sorted(run.glob('server_*/decisions_*.jsonl')):
        for d in reader.jsonl(path):
            if d.get('ev')=='dec' and d['uid'] in accepted and int(d.get('attempt',1) or 1)==accepted[d['uid']]['attempt']:
                server[d['uid'],d['step']]=d
    receipts={}
    for r in reader.jsonl(run/'client/per_step.jsonl'):
        if r.get('_kind'):continue
        uid=r.get('task_uid')
        if uid not in accepted or not r.get('accepted') or r.get('attempt')!=accepted[uid]['attempt']:continue
        if r.get('run_id')!=accepted[uid].get('run_id'):continue
        a.require(r['yaml_id']==name and (r['task_id'],r['orig_init_state_idx'])==a.uid_pair(uid),'Wrong receipt identity')
        a.require(type(r['step_idx']) is int and r['step_idx']>=0 and r['step_idx']%5==0,'Wrong receipt horizon')
        receipts[uid,r['step_idx']//5]=r
    y=np.empty(300,dtype=np.int64);n=np.zeros(300,dtype=np.int64);v=n.copy();m=n.copy();missing=[]
    logged_n=logged_v=logged_m=0
    for uid,outcome in accepted.items():
        j=positions[a.uid_pair(uid)];y[j]=outcome['success']
        rows=sorted((r for (u,s),r in receipts.items() if u==uid),key=lambda r:r['step_idx'])
        a.require(rows and [r['step_idx'] for r in rows]==list(range(0,5*len(rows),5)),'Missing/noncontiguous client receipts')
        decisions=sorted((d for (u,s),d in server.items() if u==uid),key=lambda d:d['step'])
        a.require(not decisions or len(decisions)==len(rows),'Partial historical server log cannot be reconstructed')
        if not decisions:missing.append(uid)
        for index,r in enumerate(rows):
            hit=r['hit_type']=='FULL_HIT'
            a.require(r['hit_type'] in ('MISS','FULL_HIT') and r['searched'] is (not hit),'Pure C10 receipt is ambiguous')
            diag=(r.get('factor_outputs') or {}).get('osplug',{})
            a.require(diag.get('os_hit') is hit,'Wrong C10 hit receipt')
            prefix='osplug:blind:' if hit else 'osplug:synth:'
            a.require(r['winner_id'].startswith(prefix+name+'_'),'Wrong C10 winner/arm')
            if hit:a.require(index>0 and rows[index-1]['hit_type']=='MISS' and diag['os_judge']=='blind','Not a single policy tail')
            n[j]+=1;v[j]+=not hit;m[j]+=not hit
            if decisions:
                d=decisions[index]
                a.require(d['step']==index and d['hit'] is hit and d['vision'] is (not hit),'Server/client receipts disagree')
                logged_n+=1;logged_v+=d['vision'];logged_m+=not d['hit']
    summary=reader.json(run/'summary.json');ledger=summary['cost_ledger']
    a.require([logged_n,logged_v,logged_m]==[ledger[x] for x in ('decisions','vision_decisions','misses')],
              'Historical subset ledger does not match retained server logs')
    a.require(ledger['controls']==5*logged_n and ledger['k_per_miss']['mean']==k,'Historical K/controls mismatch')
    a.require(missing,'Recovery needs an actual historical gap')
    price=pricing(row['model'],k)
    audit=dict(root=str(root),arm=name,k=k,pricing=price,complete_episodes=300,
        historical_log_gap=dict(episodes=len(missing),uids=sorted(missing),
            cost_source='Accepted client receipts for all 300; crosschecked every retained server decision',
            live_k_independently_audited_episodes=300-len(missing),
            missing_episode_k_binding='Sealed arm YAML and launch; overwritten miss_k rows are unavailable'),
        original_partial_ledger=dict(decisions=logged_n,vision=logged_v,misses=logged_m),
        no_fabricated_server_decisions=True,leave_own_episode_out=False)
    return a.Episodes(y.reshape(10,30),n.reshape(10,30),v.reshape(10,30),m.reshape(10,30),
                      (price['vision']*v+price['miss']*m).reshape(10,30),audit)
