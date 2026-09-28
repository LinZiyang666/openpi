"""Replay actual successful MISS arrival counts against K9's fixed capacities.
No visual dedup is possible from JSON. Append counts are before visual dedup.
"""
from pathlib import Path
from collections import Counter
import json, math
import numpy as np
OUT=Path(__file__).resolve().parent
memory=json.loads((OUT/'memory.json').read_text());audit=json.loads((OUT/'log_audit.json').read_text())
results=[]
for r in audit:
    if r['root']!='r04_k7' or not r['arm'].endswith('tail1ug'):continue
    cell='_'.join(r['meta']['cell'].split('_')[:2]);scale=500 if 'bpool_cs' in r['libnames'] else 50
    mem=next(x for x in memory if x['cell']==cell and x['scale']==scale)
    data=np.load(OUT/'usage'/f'{r["root"]}__{r["arm"]}.npz');info=data['miss_info'];uids=data['miss_uid']
    num=Counter(uids[info[:,3]==1]);tasks={u:int(i[0]) for u,i in zip(uids,info)}
    journal=Path('/home/weiland/trace_runs/os_closed_loop')/r['root']/'runs'/r['arm']/'client/journal.jsonl'
    done={}
    for line in journal.open():
        j=json.loads(line)
        if j.get('accepted') and j.get('status') in ('done','failed') and not j.get('error'):done[j['task_uid']]=j
    counts=np.zeros(10,int);first_uniform=None;first_ragged=None;path=[]
    for n,u in enumerate(sorted(done,key=lambda x:done[x].get('ts',0)),1):
        if num[u]:counts[tasks[u]]+=num[u]
        used=np.asarray(mem['task_rows'])+counts
        if first_uniform is None and (used>mem['uniform_capacity_rows']//10).any():first_uniform=n
        if first_ragged is None and (used>np.asarray(mem['task_capacities'])).any():first_ragged=n
        if n in (50,100,250,500):path.append(dict(episodes=n,successful_misses=counts.tolist()))
    # Capacity for this measured 500-episode arrival envelope, rounded up 64; forecast, not worst-case bound.
    capacity=[math.ceil(x/64)*64 for x in (np.asarray(mem['task_rows'])+counts)]
    out=dict(arm=r['arm'],cell=cell,scale=scale,initial_task_rows=mem['task_rows'],successful_miss_by_task=counts.tolist(),
             uniform_task_capacity=mem['uniform_capacity_rows']//10,first_uniform_overflow_episode=first_uniform,
             first_ragged_overflow_episode=first_ragged,capacity_covering_observed500=capacity,trajectory=path)
    results.append(out);print(json.dumps(out))
(OUT/'capacity_growth.json').write_text(json.dumps(results,indent=2))
