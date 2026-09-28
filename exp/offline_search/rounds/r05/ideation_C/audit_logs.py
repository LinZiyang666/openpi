"""Read-only, byte-snapshotted log audit. Run with the CPU/env prefix in RUN.md.

Selection coverage is of REPORTED members (usually ten, sometimes sixteen),
never a claim of exact full-kernel or closed-loop equivalence.
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import datetime, json, hashlib
import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
ROOT = Path('/home/weiland/trace_runs/offline_search_store')

def lines(p, size):
    with Path(p).open('rb') as f:
        while f.tell() < size:
            b = f.readline()
            if f.tell() > size or not b.endswith(b'\n'): break
            try: yield json.loads(b)
            except (ValueError, UnicodeDecodeError): continue

def audit(job):
    root, arm, meta, files, journal = job
    accepted = {}
    if journal:
        for d in lines(*journal):
            if d.get('accepted') and d.get('status') in ('done','failed') and not d.get('error'):
                accepted[d['task_uid']] = d
    fields = Counter(); lens = Counter(); evs = Counter(); raw = 0; misses = 0
    input_paths = list((RUNS/root/'runs'/arm).rglob('*.npz'))
    chosen = {}; duplicates = 0; conflicts = set(); growth_fields = Counter()
    for p, size in files:
        for d in lines(p, size):
            evs[str(d.get('ev'))] += 1
            if d.get('ev') != 'dec': continue
            raw += 1
            for k in ('key_v0','key_v1','robot_state','a_exec','served_head','rows','weights'):
                if d.get(k) is not None: fields[k] += 1
            lens[str(len(d.get('topk') or []))] += 1
            if not d.get('hit',True):
                misses += 1
                for k in ('key_v0','key_v1','robot_state','a_exec','served_head'):
                    if d.get(k) is not None: growth_fields[k] += 1
            uid=d.get('uid'); acc=accepted.get(uid)
            if acc is None or int(d.get('attempt') or 1) != int(acc.get('attempt') or 1) or not d.get('ok',True): continue
            key=(uid,int(d['step']))
            compact=(d.get('top1'), d.get('hit',True), d.get('vision',True), d.get('lib'))
            if key in chosen:
                duplicates += 1
                if chosen[key][0] != compact: conflicts.add(uid)
                continue
            members=d.get('rows') or d.get('topk') or []
            chosen[key]=(compact, members, int(d.get('task_id',-1)), int(d.get('init',-1)),
                         int((d.get('extras') or {}).get('os_reason',0)),
                         d.get('a_exec'), d.get('robot_state'),
                         int((d.get('extras') or {}).get('stuck_n',0)))
    chosen={k:v for k,v in chosen.items() if k[0] not in conflicts}
    keys=sorted(chosen); n=len(keys)
    top=np.full((n,16),-1,np.int32); info=np.zeros((n,10),np.int32)
    libnames=sorted({str(v[0][3]) for v in chosen.values()}); counts=Counter()
    epstats=defaultdict(Counter); mh=[]; mr=[]; mi=[]; mu=[]
    for i,(uid,step) in enumerate(keys):
        c, rr, task, init, reason, head, rs, stuck = chosen[(uid,step)]
        top[i,:min(len(rr),16)]=np.asarray(rr[:16],np.int32)
        success=bool(accepted[uid].get('success'))
        info[i]=[task,init,step,int(success),int(c[1]),int(c[2]),int(c[0] if c[0] is not None else -1),reason,libnames.index(str(c[3])),stuck]
        es=epstats[uid];es.update(N=1,V=int(c[2]),M=int(not c[1]),S=int(success))
        es['task']=task; es['init']=init
        counts['success_N' if success else 'failed_N']+=1
        if not c[1]:
            counts['success_M' if success else 'failed_M']+=1
            if head is not None:
                h=np.asarray(head,dtype=np.float32)
                counts['miss_head_rows_'+str(len(h))]+=1
                if h.shape == (5,7):
                    mh.append(h); mr.append(np.asarray(rs[:8],np.float32) if rs is not None else np.full(8,np.nan,np.float32))
                    mi.append(info[i].copy());mu.append(uid)
        if c[1]: counts['hit_reported_'+str(len(rr))]+=1
    growth=[]; ordered=sorted(epstats,key=lambda u:accepted[u].get('ts',0))
    sums=Counter()
    for j,u in enumerate(ordered,1):
        e=epstats[u];s=bool(accepted[u].get('success'))
        sums['M_all']+=e['M'];sums['M_success']+=e['M']*s
        sums['success_episodes']+=int(s)
        if j in (50,100,250,500) or j==len(ordered): growth.append(dict(episodes=j,**sums))
    np.savez_compressed(OUT/'usage'/f'{root}__{arm}.npz',top=top,info=info,
                        miss_head=np.asarray(mh,np.float32).reshape(-1,5,7),miss_rs=np.asarray(mr,np.float32).reshape(-1,8),
                        miss_info=np.asarray(mi,np.int32).reshape(-1,10),miss_uid=np.asarray(mu))
    sr=sum(bool(accepted[u].get('success')) for u in epstats)/len(epstats) if epstats else None
    result=dict(root=root,arm=arm,meta=meta,raw_decisions=raw,raw_miss=misses,fields=dict(fields),miss_fields=dict(growth_fields),
                topk_lengths=dict(lens),input_npz=[str(p) for p in input_paths],accepted_episodes=len(accepted),
                represented_episodes=len(epstats),conflicting_uids=sorted(conflicts),duplicate_decisions=duplicates,
                decisions=n,SR=sr,V=int(info[:,5].sum()),M=int((1-info[:,4]).sum()),counts=dict(counts),libnames=libnames,
                growth_yield=growth,events=dict(evs),files=[dict(path=p,snapshot_bytes=s,final_bytes=Path(p).stat().st_size) for p,s in files])
    if n:result['IR']=.152*result['V']/n+.848*result['M']/n
    return result

def summarize_usage(results):
    groups=defaultdict(list)
    for r in results:
        cell=r['meta'].get('cell','')
        if not cell:continue
        key='_'.join(cell.split('_')[:2])
        for lib in r['libnames']:
            if lib in ('current','bpool_cs','bpool_all'):groups[(key,lib)].append(r)
    output=[]
    for (key,lib),rr in groups.items():
        d=ROOT/'library'/key/lib
        task=np.load(d/'task_id.npy',mmap_mode='r');L=len(task)
        nxt=np.load(d/'next.npy',mmap_mode='r');succ=np.load(d/'success.npy',mmap_mode='r')
        train=np.zeros(L,bool);test=np.zeros(L,bool);allr=np.zeros(L,bool);srows=np.zeros(L,bool);frows=np.zeros(L,bool)
        trap=np.zeros(L,bool);top1=np.zeros(L,bool);freq=np.zeros(L,np.int64);dec=0;test_sets=[];invalid=0
        for r in rr:
            a=np.load(OUT/'usage'/f'{r["root"]}__{r["arm"]}.npz');t=a['top'];i=a['info'];b=r['libnames'].index(lib)
            keep=(i[:,8]==b)&(i[:,4]==1);t=t[keep];i=i[keep];dec+=len(i)
            valid=(t>=0)&(t<L);invalid+=int(((t>=L)|(t< -1)).sum())
            t=np.where(valid,t,-1)
            def mark(dst,mask):
                q=t[mask].ravel();dst[q[q>=0]]=True
            mark(allr,np.ones(len(i),bool));mark(train,i[:,1]<25);mark(test,i[:,1]>=25)
            mark(srows,i[:,3]==1);mark(frows,i[:,3]==0)
            ids=i[:,6];good=(ids>=0)&(ids<L);top1[ids[good]]=True
            np.add.at(freq,t[valid],1)
            # Observable trap association: >=3 consecutive HITs selecting identical top1 in an episode.
            for j in range(2,len(i)):
                if i[j,2]==i[j-1,2]+1==i[j-2,2]+2 and np.array_equal(i[j,:2],i[j-2,:2]) and i[j,6]==i[j-1,6]==i[j-2,6] and 0<=i[j,6]<L:
                    trap[i[j,6]]=True
            test_sets.append(t[i[:,1]>=25])
        nq=members=missing=changed=0
        for t in test_sets:
            va=t>=0;covered=np.where(va,train[np.maximum(t,0)],True)
            nq+=len(t);members+=int(va.sum());missing+=int((~covered&va).sum());changed+=int((~covered).any(1).sum())
        # Next-two closure is a measured lower envelope for B2 continuation, not full trajectories.
        closed=allr.copy();front=np.flatnonzero(allr)
        for h in range(2):front=nxt[front];front=front[front>=0];closed[front]=True
        out=dict(cell=key,library=lib,L=L,arms=len(rr),hit_decisions=dec,reported_union=int(allr.sum()),top1_union=int(top1.sum()),
                 train_union=int(train.sum()),test_union=int(test.sum()),heldout_queries=nq,heldout_changed_reported_set=changed,
                 heldout_members=members,heldout_missing_members=missing,success_union=int(srows.sum()),failure_union=int(frows.sum()),
                 failure_only=int((frows&~srows).sum()),trap_top1=int(trap.sum()),trap_also_success_member=int((trap&srows).sum()),
                 union_next2=int(closed.sum()),failed_library_rows=int((~succ).sum()),failed_library_rows_selected=int((allr&~succ).sum()),invalid_members=invalid)
        np.savez_compressed(OUT/'usage'/f'pool_{key}_{lib}.npz',used=allr,train=train,test=test,success=srows,failure=frows,trap=trap,freq=freq,next2=closed)
        output.append(out)
    return output

if __name__=='__main__':
    (OUT/'usage').mkdir(exist_ok=True)
    jobs=[]
    for root in sorted(RUNS.iterdir()):
        if not (root.name.startswith(('r02_','r03_','r04_'))):continue
        mp=root/'arms.json';meta=json.loads(mp.read_text()) if mp.exists() else []
        byarm={a['arm']:a for a in meta}
        for arm in sorted((root/'runs').glob('*')):
            if not arm.is_dir():continue
            files=[(str(p),p.stat().st_size) for p in sorted(arm.glob('server_*/decisions*.jsonl'))]
            if not files:continue
            jp=arm/'client/journal.jsonl'
            jobs.append((root.name,arm.name,byarm.get(arm.name,{}),files,(str(jp),jp.stat().st_size) if jp.exists() else None))
    (OUT/'log_snapshot.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=jobs),indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool: results=list(pool.map(audit,jobs))
    (OUT/'log_audit.json').write_text(json.dumps(results,indent=2))
    usage=summarize_usage(results)
    (OUT/'usage_summary.json').write_text(json.dumps(usage,indent=2))
    print(json.dumps(dict(arms=len(results),files=sum(len(r['files']) for r in results),raw_decisions=sum(r['raw_decisions'] for r in results),accepted_decisions=sum(r['decisions'] for r in results),raw_miss=sum(r['raw_miss'] for r in results),input_npz=sum(len(r['input_npz']) for r in results))))
    for r in usage:print(json.dumps(r))
