import json, sys, collections
from pathlib import Path
R=Path('/home/weiland/trace_runs/os_closed_loop/r08_main/runs')
arms=json.load(open('/home/weiland/trace_runs/os_closed_loop/r08_main/arms.json'))
tab={}
for a in arms:
    name=a['arm']; v=a['r8']['variant']
    if v not in ('A','P10','CU','CT','IP','SW','W10','FL','SF1','O5a','O5b','A5','SHIFT','W5'): continue
    j=R/name/'client'/'journal.jsonl'
    per=collections.defaultdict(lambda:[0,0])
    seen=set()
    for line in open(j):
        r=json.loads(line)
        if not r.get('accepted') or r.get('status') not in ('done','failed') or r.get('error'): continue
        uid=r['task_uid']
        if uid in seen: continue
        seen.add(uid)
        t=int(uid.split(':')[-2]); init=int(uid.split(':')[-1])
        per[t][0]+=1; per[t][1]+=int(bool(r.get('success')))
    tab[name]=(a['model'],a['suite_short'],a['r8'].get('library_size'),v,{t:(per[t][1],per[t][0]) for t in sorted(per)})
json.dump(tab,open('/tmp/fable_pertask.json','w'))
for name,(m,s,l,v,per) in sorted(tab.items(), key=lambda kv:(kv[1][0],kv[1][1],kv[1][2] or 0,kv[1][3])):
    row=' '.join(f"{per[t][0]:2d}" for t in sorted(per))
    tot=sum(per[t][0] for t in per); n=sum(per[t][1] for t in per)
    print(f"{m:5s} {s:3s} {str(l):4s} {v:5s} {tot/n:.3f} | {row}")
