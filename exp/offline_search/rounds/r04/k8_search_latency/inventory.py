import os;os.sched_setaffinity(0,{34})
from common import *
import collections, datetime, gc
cut=datetime.datetime.fromisoformat('2026-09-27T20:47:26.508504+00:00').timestamp()
roots=[p for p in RUNS.iterdir() if p.name in ('r02_g50','r02_g500','r03_mx','r03_full') or p.name.startswith('r04_')]
fits=[]
paths=sorted(p for r in RUNS.iterdir() for p in (r/'fits').glob('*.pkl'))
paths+=sorted(pathlib.Path('/tmp/k1_blind_fits').glob('*.pkl'))
paths+=sorted((STORE/'derived/r04/k3_cost/fits').glob('*.pkl'))
for p in paths:
    try:
        b=load(p);m=b['method'];bm=base(m)
        r={k:v for k,v in b.items() if k not in ('method','registered')}
        r.update(path=str(p),class_name=type(m).__name__,name=m.name,bytes=p.stat().st_size)
        if hasattr(bm,'tasks'):
            r.update(scale=500 if bm.lib=='big' else 50,entries=sum(len(t.rows) for t in bm.tasks.values()),per_task={str(k):len(t.rows) for k,t in bm.tasks.items()})
        fits.append(r);del b,m,bm;gc.collect()
    except Exception as e:fits.append(dict(path=str(p),error=repr(e)))
dump(OUT/'inventory.json',fits)
print('FITS',len(fits),collections.Counter(r.get('class_name','ERROR') for r in fits),flush=True)
groups=collections.defaultdict(lambda:collections.defaultdict(list));files=[];starts=[]
for root in sorted(roots):
    for p in sorted((root/'runs').glob('*/server*/decisions*.jsonl')):
        st=None;n=0;bad=0;size=p.stat().st_size
        with p.open() as f:
            for line in f:
                try:r=json.loads(line)
                except ValueError:bad+=1;continue
                if r.get('ev')=='startup':st=r;starts.append(dict(path=str(p),startup=r));continue
                if r.get('ev')!='dec' or st is None:continue
                n+=1;kw=st.get('kwargs',{});bk=kw.get('base_kwargs',kw)
                scale=500 if bk.get('lib')=='big' or r.get('lib') in ('bpool_cs','bpool_all') and st.get('lib_sizes',{}).get(r.get('lib'),0)>5000 else 50
                mode='SERIALIZED' if st.get('r4',False) and st['ts']<cut else 'CONCURRENT'
                fam=st.get('method_class',st.get('method_spec','native').split(':')[-1])
                if fam=='AWM':fam+=' CL2' if kw==dict(lib=bk.get('lib'),kref=5 if scale==50 else 8) else ' other'
                if 'Judge' in fam:fam+=' '+str(st.get('judge',{}).get('mode','pure'))
                key=(mode,st['model'],st['suite'],scale,fam,'vision' if r.get('vision',True) else 'blind')
                armkey=(root.name,p.parent.parent.name)+key
                for target in (key,armkey):
                    for field in ('q_us','search_us','native_us','blind_prepare_ms','blind_output_ms'):
                        val=r.get(field)
                        if isinstance(val,(int,float)) and np.isfinite(val):
                            if field=='native_us' and (not st.get('shadow_native',False) or not r.get('vision',True)):continue
                            if field.startswith('blind_') and r.get('vision',True):continue
                            groups[target][field].append(val/1000 if field.endswith('_us') else val)
                    if r.get('vision',True):
                        groups[target]['bookkeeping_ms'].append((r.get('search_us',0)-r.get('q_us',0)-r.get('native_us',0))/1000)
        files.append(dict(path=str(p),bytes_at_start=size,bytes_at_end=p.stat().st_size,decisions=n,bad_lines=bad,startup_ts=st.get('ts') if st else None))
    print('LOG ROOT',root.name,flush=True)
dump(OUT/'server_stats.json',[dict(key=k,metrics={f:stats(v) for f,v in g.items()}) for k,g in groups.items()])
dump(OUT/'log_inventory.json',dict(cutoff_utc=cut,files=files,startups=starts,load=os.getloadavg(),snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
