"""One admitted CUDA process per config/repeat; actual final graph and result bridge."""
import argparse,collections,os,subprocess,threading,time
from common import *
from exp.offline_search.rounds.r05.q5_gpu.dev import gpu_retrieval as bridge
import torch

def smi():
    free,total=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
    own=0
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
        pid,used=line.split(',')
        if int(pid)==os.getpid():own=int(used)
    return dict(free_mib=free,total_mib=total,own_mib=own,time=time.time())
class Watch:
    def __init__(self):
        self.samples=[smi()]
        if self.samples[0]['free_mib']<14336:raise RuntimeError('GPU admission needs >=14 GiB free')
        self.stop=threading.Event()
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(5*1024/self.samples[0]['total_mib'])
        def poll():
            while not self.stop.wait(.5):
                row=smi();self.samples.append(row)
                if row['own_mib']>6144:
                    dump(B/'results/GPU_ABORT.json',self.samples);os._exit(86)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()
    def finish(self):
        self.stop.set();self.thread.join();self.samples.append(smi())
        return dict(samples=self.samples,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',type=int,required=True)
    a=p.parse_args();c=config(a.config);m=load(c);bridge.validate_method(m)
    from exp.offline_search.rounds.r04.k9_gpu_retrieval.gpu_awm import GPUAWM
    b=getattr(m,'base',m);lib=store.LibraryView(STORE,'pi05_l10',b.cand_name)
    module=GPUAWM(m,lib,precision='float64').eval()
    w=Watch()
    # Match the production model's constructor; the bridge must leave this alone.
    torch.set_float32_matmul_precision('high')
    module.cuda()
    graph=bridge.GraphConnection(module,torch.device('cuda:0'))
    gm,_=plugin.clone_method(m,strict=True)
    agg=collections.Counter();samples=[];sources=[];worst=[];by_reg=collections.defaultdict(list)
    for ep,qs in sequences('l10'):
        m.reset(ep);gm.reset(ep);sources.append(dict(uid=ep.uid,n=len(qs)))
        for q in qs:
            keys={k:torch.as_tensor(np.array(v),device='cuda') for k,v in [('vision_0',q.key_v0),('vision_1',q.key_v1),('robot_state',q.rs)]}
            t=time.perf_counter_ns();cp=m.query(q);cpu_ms=(time.perf_counter_ns()-t)/1e6
            out,timing=graph.run(keys,step=q.step,task=q.task_id,prev_hit=-1 if q.prev_hit is None else int(q.prev_hit),prev_action=q.prev_a_exec)
            gp=bridge.result(gm,q,out)
            delta=float(np.max(np.abs(cp.action-gp.action)));conf=abs(cp.confidence-gp.confidence)
            ag=dict(top1=bool(cp.topk[0]==gp.topk[0]),top16_set=np.array_equal(np.sort(cp.topk),np.sort(gp.topk)),
                    top16_order=np.array_equal(cp.topk,gp.topk),chunk=delta<=1e-4,confidence=conf<=1e-3)
            if module.judge:
                ag['guard_flags']=cp.extras['os_flags']==gp.extras['os_flags']
                ag['guard_reason']=cp.extras['os_reason']==gp.extras['os_reason']
            agg.update({k:int(v) for k,v in ag.items()});agg['n']+=1
            samples.append(dict(cpu_ms=cpu_ms,**timing,delta=delta,conf=conf));by_reg[int(out['regime'])].append(delta)
            if not all(ag.values()):worst.append(dict(uid=ep.uid,step=q.step,agreement=ag,delta=delta,confidence_abs=conf,cpu_topk=cp.topk,gpu_topk=gp.topk))
    # Graph buffers are independent and shared fitted tables unchanged under 8 concurrent connections.
    graphs=[graph]+[bridge.GraphConnection(module,torch.device('cuda:0')) for _ in range(7)]
    qs=next(sequences('l10',episodes_per_task=1,max_steps=8))[1]
    from concurrent.futures import ThreadPoolExecutor
    jobs=[]
    for i,g in enumerate(graphs):
        q=qs[min(i,len(qs)-1)]
        keys={k:torch.as_tensor(np.array(v),device='cuda') for k,v in [('vision_0',q.key_v0),('vision_1',q.key_v1),('robot_state',q.rs)]}
        jobs.append((g,keys,q.task_id))
    torch.cuda.synchronize()  # fixture publication, outside inference/timing
    def run(job):
        g,keys,task=job
        out,_=g.run(keys,step=0,task=task,prev_hit=-1,prev_action=None)
        return out['action'],out['topk']
    expected=[run(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=8) as pool:actual=list(pool.map(run,jobs))
    assert all(np.array_equal(e[0],v[0]) and np.array_equal(e[1],v[1]) for e,v in zip(expected,actual))
    assert len({g.inputs[0].data_ptr() for g in graphs})==8
    # New connection capture may overlap an existing connection's graph replay.
    started=threading.Event()
    def during_capture():
        started.set()
        for _ in range(32):
            got=run(jobs[0])
            assert all(np.array_equal(x,y) for x,y in zip(got,expected[0]))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(during_capture);started.wait()
        extra_graph=bridge.GraphConnection(module,torch.device('cuda:0'))
        future.result()
    record=dict(config=c,run=a.run,command=sys.argv,source_sha256={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (B/'dev').glob('*.py')},
                agreement=dict(agg),latency={k:stats([r[k] for r in samples]) for k in ('cpu_ms','event_ms','wall_ms')},
                chunk_by_regime={k:stats(v) for k,v in by_reg.items()},confidence_abs=stats([r['conf'] for r in samples]),
                sources=sources,mismatches=worst,samples=samples,resident_bytes=module.resident_bytes(),
                final_d2h_bytes=graph.copy_bytes,concurrency_connections=8,concurrency_exact=True,
                capture_during_replay=True,precision=module.precision,
                tf32_unchanged=bool(torch.backends.cuda.matmul.allow_tf32),gpu=w.finish())
    assert record['tf32_unchanged']
    dump(B/f'results/gpu_{a.config}_r{a.run}.json',record)
    print(json.dumps({k:v for k,v in record.items() if k in ('agreement','latency','final_d2h_bytes','resident_bytes')}),flush=True)
if __name__=='__main__':main()
