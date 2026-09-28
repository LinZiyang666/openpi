"""K8 fixed-task library scaling and shared-process thread contention."""
import argparse,os
pa=argparse.ArgumentParser();pa.add_argument('mode',choices=['scaling','concurrency','aux']);pa.add_argument('--run',type=int,default=1);pa.add_argument('--config',default='');args=pa.parse_args()
os.sched_setaffinity(0,{34})
from common import *
# Import replay helpers without executing benchmark main.
from benchmark import queries,reset_prof
import concurrent.futures,threading,collections,copy,contextlib

def timed(m,seq,n=1024):
    times=[]
    while len(times)<n:
        for ep,qs in seq:
            m.reset(ep)
            for q in qs:
                t=time.perf_counter_ns();m.query(q);times.append((time.perf_counter_ns()-t)/1e6)
                if len(times)>=n:return times

def scaling():
    configs=[c for c in json.loads((OUT/'configs.json').read_text()) if c['family']=='AWM']
    results=[]
    for cfg in configs:
        b=load(cfg['path']);m=b['method'];reset_prof(m,api.NULL_PROFILER);seq=[x for x in queries(cfg['cell']) if x[0].task_id==0]
        T=m.tasks[0];original=dict(vars(T));nn=len(T.rows)
        for factor in ([1] if cfg['scale']==50 else [1,2,4,10]):
            for k,v in original.items():
                if isinstance(v,np.ndarray) and v.ndim>=1 and v.shape[0]==nn:
                    setattr(T,k,np.concatenate([v]*factor,axis=0))
            timed(m,seq,256);before=cpu();ld=os.getloadavg();ts=timed(m,seq)
            r=dict(config=cfg['id'],factor=factor,nominal_episodes=cfg['scale']*factor,task=0,task_entries=nn*factor,ms=stats(ts),samples_ms=ts,load=ld,idle_percent=idle(before,cpu()))
            results.append(r);dump(OUT/f'scaling_r{args.run}.json',results);print('SCALE',r['config'],factor,r['ms'],flush=True)

def concurrency():
    cfg=next(c for c in json.loads((OUT/'configs.json').read_text()) if c['id']==args.config)
    m=load(cfg['path'])['method'];reset_prof(m,api.NULL_PROFILER);seq=queries(cfg['cell'])
    results=[]
    # Fixed 1,024 input calls at every thread count. Each segment owns a clone
    # warmed on its episode prefix, so a 32-thread sweep does not silently
    # change from mixed cache/inf histories to only the first task's stale rows.
    segments=[];left=1024
    while left:
        for ep,qs in seq:
            for start in range(0,len(qs),16):
                chunk=qs[start:start+min(16,left)]
                segments.append((ep,qs[:start],chunk));left-=len(chunk)
                if not left:break
            if not left:break
    for count in (1,4,8,16,24,32):
        os.sched_setaffinity(0,{34} if count==1 else {34,35,36,37,78,79,80,81})
        jobs=[[] for _ in range(count)]
        for index,(ep,prefix,qs) in enumerate(segments):
            mm=plugin.clone_method(m,strict=True)[0];mm.reset(ep)
            for q in prefix:mm.query(q)
            # Warm the exact timed chunk, then restore state at its start.
            state=plugin.clone_method(mm,strict=True)[0]
            for q in qs:mm.query(q)
            jobs[index%count].append((state,qs))
        barrier=threading.Barrier(count)
        def worker(i):
            barrier.wait();t=time.perf_counter();wall=[];dg=[]
            for mm,qs in jobs[i]:
                for q in qs:
                    t0=time.perf_counter_ns();r=mm.query(q);wall.append((time.perf_counter_ns()-t0)/1e6)
                    dg.append(hashlib.sha256(r.topk.tobytes()+r.action.tobytes()).hexdigest())
            return wall,time.perf_counter()-t,dg
        before=cpu();ld=os.getloadavg();t=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:rr=list(pool.map(worker,range(count)))
        wall=max(x[1] for x in rr);ts=sum((x[0] for x in rr),[])
        dg=sorted(sum((x[2] for x in rr),[]))
        if results:assert dg==results[0]['sorted_digests'],'thread count changed outputs'
        r=dict(config=cfg['id'],threads=count,affinity=sorted(os.sched_getaffinity(0)),samples_ms=ts,ms=stats(ts),throughput_qps=len(ts)/wall,active_wall_s=wall,total_with_warmup_s=time.perf_counter()-t,load=ld,idle_percent=idle(before,cpu()),sorted_digests=dg)
        results.append(r);dump(OUT/f"concurrency_{cfg['id']}_r{args.run}.json",results);print('THREADS',cfg['id'],count,r['ms'],r['throughput_qps'],flush=True)

def aux():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from openpi.cache.components.key_builder import _spatial_pool_tokens
    result=[]
    def measure(label,fn,n=1024,meta=None):
        for _ in range(64):fn()
        b=cpu();ld=os.getloadavg();ts=[]
        for i in range(n):
            t=time.perf_counter_ns();fn();ts.append((time.perf_counter_ns()-t)/1e6)
        r=dict(label=label,ms=stats(ts),samples_ms=ts,load=ld,idle_percent=idle(b,cpu()),meta=meta);result.append(r);dump(OUT/f'aux_r{args.run}.json',result)
    for model in ('pi05','groot'):
      for suite in ('spatial','l10'):
        qc=store.QueryCell(STORE,f'{model}_{suite}_cache');idx=int(np.flatnonzero(qc.tok_index>=0)[0]);ti=int(qc.tok_index[idx]);tok=[torch.from_numpy(np.array(qc.tok(v)[ti])).float() for v in ('v0','v1')]
        delta=[float(np.abs(_spatial_pool_tokens(t,16,4).numpy()-getattr(qc,'key_'+v)[idx]).max()) for v,t in zip(('v0','v1'),tok)]
        measure(f'{model}_{suite}_pool2_cpu',lambda:[_spatial_pool_tokens(t,16,4) for t in tok],meta=dict(source_row=idx,stored_key_max_abs=delta,caveat='CPU float32 pooling of stored fp16 tokens; live GPU pooling and D2H are excluded'))
    # Exact plugin writer, local scratch only; includes clean/JSON/open/write/close.
    log=json.loads((OUT/'log_inventory.json').read_text())
    for phase in ('vision','blind'):
        row=None
        for st in log['startups']:
            if st['startup'].get('method_class')!='BlindMixedJudge':continue
            with open(st['path']) as f:
                for line in f:
                    r=json.loads(line)
                    if r.get('ev')=='dec' and r.get('vision',True)==(phase=='vision'):row=r;break
            if row:break
        rt=SimpleNamespace(dec_path=pathlib.Path(f'/tmp/k8_latency/emit_{phase}_r{args.run}.jsonl'),r4=True,_wlock=threading.Lock())
        measure('plugin_emit_'+phase,lambda:plugin.PluginRuntime.emit(rt,dict(row)),meta={'source_row':row})
    for length in (16,64,128):
        bufs=[plugin._Buf((32768,),np.float32,cap=256) for _ in range(2)];key=np.ones(32768,np.float32)
        def history():
            for b in bufs:b.n=length;b.append(key)
        measure('history_two_pooled_key_copies_'+str(length),history)
    for c in json.loads((OUT/'configs.json').read_text()):
        if c['family'] not in ('AWM','WristAWM'):continue
        mm=base(load(c['path'])['method']);seq=queries(c['cell']);q=seq[0][1][1];T=mm.tasks[q.task_id]
        xv=np.concatenate([mm.B0T@q.key_v0-mm.muB0,mm.B1T@q.key_v1-mm.muB1]);x=np.concatenate([xv,np.asarray(q.rs[:8])])
        measure(c['id']+'_whiten_state',lambda:x@T.Wf-T.shift,meta={'dims':len(x)})
    print('AUX',len(result),flush=True)
from types import SimpleNamespace
if args.mode=='scaling':scaling()
elif args.mode=='concurrency':concurrency()
else:aux()
