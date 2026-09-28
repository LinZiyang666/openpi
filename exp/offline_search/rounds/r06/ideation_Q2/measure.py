"""Read-only closed-loop audit. Run from repository root with the documented CPU mask."""
from pathlib import Path
import ast, collections, csv, datetime, hashlib, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np

OUT = Path(__file__).resolve().parent
CL = Path('/home/weiland/trace_runs/os_closed_loop')
STORE = Path('/home/weiland/trace_runs/offline_search_store')

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=1))

def journal(path):
    acc = {}
    for line in path.open():
        x = json.loads(line)
        if x.get('accepted') and x.get('status') in ('done','failed') and not x.get('error'):
            key = tuple(map(int,x['task_uid'].split(':')[-2:]))
            assert key not in acc, (path,key)
            acc[key] = x
    return acc

def scan(item):
    run, arm, spec, marker = item
    p=CL/run/'runs'/arm; sm=p/'summary.json'; s=json.loads(sm.read_text()); acc=journal(p/'client/journal.jsonl')
    assert len(acc)==s['complete'] and sum(x['success'] for x in acc.values())==s['success']
    model=s['model']; suite='l10' if s['suite'] in ('l10','libero_10') else 'spatial'
    kw=s.get('kwargs') or {}; base=kw.get('base_kwargs',kw)
    lib=base.get('library',base.get('lib','big'))
    lib={'big':'500','current':'50','bpool_cs':'500','bpool_all':'500','demo100':'100','demo200':'200','demo300':'300'}.get(lib,lib)
    if run.startswith('r02_'):lib=run[5:]
    if 'b0' in arm:lib='50'
    ledger=s.get('cost_ledger') or {}; L=ledger.get('l_per_request',{}).get('mean',spec.get('replan_steps',5))
    pure=s.get('mode') in ('inference','infer','full_inference') or 'inf_' in arm or 'inferL' in arm or 'policy_L' in arm
    if pure:lib='-'
    seen={}; duplicates=0; conflicts=set(); files=[]
    for f in sorted(p.glob('server_*/decisions_*.jsonl')):
        files.append({'path':str(f),'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns})
        for line in f.open():
            d=json.loads(line)
            if d.get('ev')!='dec':continue
            uid=d.get('uid',''); bits=uid.split(':')
            if len(bits)<2:continue
            key=tuple(map(int,bits[-2:])); a=acc.get(key)
            if a is None or d.get('attempt',1)!=a.get('attempt',1):continue
            k=(*key,int(d['step']))
            value=(int(d.get('vision',True)),int(not d.get('hit',not pure)),d.get('src'),d.get('top1'),d.get('conf'),d.get('ok',True))
            if k in seen:
                duplicates+=1
                if seen[k]!=value:conflicts.add(key)
            seen[k]=value
    ep=[]; incomplete=[]
    groups=collections.defaultdict(list)
    for (t,i,step),v in seen.items():groups[t,i].append((step,v))
    for (t,i),a in sorted(acc.items()):
        ds=sorted(groups[t,i]); n=len(ds)
        if ds and [d[0] for d in ds]!=list(range(n)):incomplete.append([t,i])
        ep.append(dict(task=t,init=i,Y=int(a['success']),N=n,V=sum(d[1][0] for d in ds),M=sum(d[1][1] for d in ds)))
    N=sum(e['N'] for e in ep);V=sum(e['V'] for e in ep);M=sum(e['M'] for e in ep)
    # Some historical native policy servers have no plugin decision log.
    raw_available=bool(N)
    if not N:
        N=ledger.get('decisions',s.get('client_decisions',0));V=N;M=N if pure else (s.get('mixed') or {}).get('misses',0)
    mode=next(iter(ledger.get('stage1_modes',{'full':1})))
    c1=.152 if model=='pi05' else .148
    v=V/N if N else None;m=M/N if N else None
    conflict=bool(conflicts or incomplete)
    ledger_match=not ledger or (N==ledger['decisions'] and V==ledger['vision_decisions'] and M==ledger['misses'])
    summary_match=N==s.get('client_decisions',N)
    family='other'
    if pure:family='policy_L'+str(int(L))
    elif run=='r04_k5':family='randomized_guard'
    elif 'G10' in arm:family='G10'
    elif 'c10' in arm:family='B'
    elif run=='r05_demo_curve':family='A_frozen' if 'frozen' in arm else 'A'
    elif (kw.get('serving')=='anchor_tail' and kw.get('budget')==1) or ('tail1uc' in arm):family='A'
    elif 'per' in arm and (run in ('r03_mx','r04_frontier','r04_cost')):family='periodic'
    elif 'h50' in arm or 'h70' in arm:family='quantile'
    elif run=='r04_rep' or (run in ('r03_mx','r04_frontier') and '_g' in arm):family='guard'
    elif 'cl2' in arm:family='CL2'
    cost_valid=not conflict and ledger_match and summary_match
    out=dict(run=run,arm=arm,cell=f'{model}_{suite}',lib=str(lib),family=family,n=len(acc),sr=s['success']/len(acc),
             N=N,V=V,M=M,L=L,v=v,m=m,m5=m*5/L if m is not None else None,
             ir=(c1*v+(1-c1)*m)*5/L if N else None,miss_per_ep=M/len(acc),stage1_mode=mode,
             cost_valid=cost_valid,duplicates=duplicates,conflicts=sorted(conflicts),incomplete=incomplete,
             ledger_match=ledger_match,summary_match=summary_match,raw_available=raw_available,
             summary=str(sm),marker=marker,kwargs=kw,spec=spec,episodes=ep,sources=files)
    out['hashes']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [sm,p/'client/journal.jsonl']}
    return out

def main():
    items=[]; excluded=[]
    for run in sorted(CL.iterdir()):
        if 'smoke' in run.name or run.name=='r03_pilot':continue
        specs={s['arm']:s for s in json.loads((run/'arms.json').read_text())} if (run/'arms.json').exists() else {}
        for sm in sorted((run/'runs').glob('*/summary.json')):
            arm=sm.parent.name;done=list((run/'state').glob(arm+'.*DONE'))
            if not done or (run/'state'/(arm+'.SKIPPED')).exists():excluded.append([run.name,arm,'not completed']);continue
            s=json.loads(sm.read_text())
            if s['complete'] not in (250,500):excluded.append([run.name,arm,'not full evaluation']);continue
            items.append((run.name,arm,specs.get(arm,{}),str(done[0])))
    # Preserve a fixed snapshot: later running R6 experiments must not alter a reproduction.
    if (OUT/'snapshot.json').exists():
        saved=json.loads((OUT/'snapshot.json').read_text());items=saved['items']
    else:
        dump('snapshot.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),items=items,excluded=excluded))
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[]
        for r in pool.map(scan,items):
            rows.append(r); print(r['arm'],r['sr'],round(r['m'] or 0,4),r['cost_valid'],flush=True)
    # Original pure inference episodes for all four cells, independently joined from journals.
    for model in ['pi05','groot']:
        for suite,short in [('l10','l10'),('spatial','sp')]:
            arm=f'tr_{model}_{short}_inf';p=Path('/home/weiland/trace_runs/dual_20260923/runs')/arm/'client/journal.jsonl'
            a=journal(p); qp=STORE/'queries'/f'{model}_{suite}_inf'; ee=json.loads((qp/'episodes.json').read_text())
            nrows=np.load(qp/'ep.npy',mmap_mode='r'); nn=collections.Counter(nrows.tolist());ep=[]
            for j,e in enumerate(ee):
                t=int(e['task_id']);i=int(e['init']); assert bool(e['success'])==bool(a[t,i]['success'])
                n=nn[j];ep.append(dict(task=t,init=i,Y=int(e['success']),N=n,V=n,M=n))
            N=sum(e['N'] for e in ep)
            rows.append(dict(run='DUAL',arm=arm,cell=f'{model}_{suite}',lib='-',family='policy_L5',n=len(ep),sr=np.mean([e['Y'] for e in ep]),N=N,V=N,M=N,L=5,v=1.,m=1.,m5=1.,ir=1.,miss_per_ep=N/len(ep),stage1_mode='full',cost_valid=True,raw_available=True,episodes=ep,summary=str(p),hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()}))
    dump('arms.json',rows)
    fields=['run','arm','cell','lib','family','n','sr','N','V','M','L','v','m','m5','ir','miss_per_ep','cost_valid','stage1_mode','summary']
    with (OUT/'arms.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    with (OUT/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['arm','cell','lib','task','init','Y','N','V','M']);w.writeheader()
        for r in rows:
            for e in r['episodes']:w.writerow(dict(arm=r['arm'],cell=r['cell'],lib=r['lib'],**e))
    print('TOTAL',len(rows),'invalid cost',[(r['arm'],len(r.get('conflicts',[])),r.get('ledger_match')) for r in rows if not r['cost_valid']],flush=True)

if __name__=='__main__':main()
