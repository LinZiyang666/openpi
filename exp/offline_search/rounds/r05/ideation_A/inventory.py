"""Read-only data/schema inventory for R5 execution-horizon diagnostics."""
from pathlib import Path
import json
from collections import Counter
OUT=Path('exp/offline_search/rounds/r05/ideation_A')
RUN=Path('/home/weiland/trace_runs/os_closed_loop')
store=Path('/dev/shm/offline_search_store')
if not store.exists(): store=Path('/home/weiland/trace_runs/offline_search_store')
out={'store':str(store),'arms':[]}
for root in sorted(RUN.glob('r04_*')):
    af=root/'arms.json'
    if not af.exists():continue
    aa=json.loads(af.read_text()); aa=aa if isinstance(aa,list) else aa.get('arms',[])
    for a in aa:
        name=a.get('arm',a.get('name')); rd=root/'runs'/name
        if not rd.exists():continue
        info={'run':root.name,'name':name,'meta':a}
        jp=rd/'client/journal.jsonl'
        rows=[]
        if jp.exists():
            for line in jp.open():
                try:rows.append(json.loads(line))
                except ValueError:pass
        acc={x['task_uid']:x for x in rows if x.get('accepted') and x.get('status') in ('done','failed') and not x.get('error')}
        info.update(episodes=len(acc),successes=sum(x.get('success',False) for x in acc.values()))
        info['server_files']=[{'path':str(p),'bytes':p.stat().st_size} for p in sorted(rd.glob('server_*/*')) if p.is_file()]
        ds=sorted(rd.glob('server_*/decisions_*.jsonl'))
        if ds:
            with ds[0].open() as f:
                for line in f:
                    x=json.loads(line)
                    if x.get('ev')=='startup':info['startup']=x
                    if x.get('ev')=='dec':info['example_dec']=x;break
        out['arms'].append(info)
(OUT/'inventory.json').write_text(json.dumps(out,indent=2))
for x in out['arms']:
    if any(t in x['name'] for t in ('inf','tail','ph2','csl')):
        d=x.get('example_dec',{})
        print(x['run'],x['name'],x['successes'],x['episodes'],'keys='+','.join(d.keys()))
