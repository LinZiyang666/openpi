"""K9 isolated I/O, real QueryViews, and GPU admission/watchdog utilities."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    assert os.environ.get(key) == '1', key
assert set(os.sched_getaffinity(0)) <= {26,27,28,29,70,71,72,73}
import hashlib, json, pathlib, pickle, subprocess, sys, threading, time
from types import SimpleNamespace
import numpy as np
ROOT = pathlib.Path('/home/weiland/projects/openpi')
OUT = ROOT/'exp/offline_search/rounds/r04/k9_gpu_retrieval'
STORE = pathlib.Path('/dev/shm/offline_search_store')
if not STORE.exists(): STORE = pathlib.Path('/home/weiland/trace_runs/offline_search_store')
from exp.offline_search.closed_loop import plugin
from exp.offline_search.harness import api, store

class Loader(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('osm_') and mod not in sys.modules:
            # Only filenames are enumerated; no test source is read.
            for folder in ('rounds/r02/g1_awm', 'rounds/r03/h3_judge', 'rounds/r02/g3_recovery'):
                for p in (ROOT/'exp/offline_search'/folder).glob('*.py'):
                    if 'osm_'+hashlib.md5(str(p).encode()).hexdigest()[:12] == mod:
                        plugin.load_method_class(str(p)+':'+name)
                        break
        return super().find_class(mod, name)

def load(path):
    with open(path, 'rb') as f: m = Loader(f).load()['method']
    m.prof = api.NULL_PROFILER
    if hasattr(m,'base'): m.base.prof = api.NULL_PROFILER
    return m

def configs():
    allcfg = json.loads((OUT.parent/'k8_search_latency/configs.json').read_text())
    return [c for c in allcfg if c['cell'].startswith('pi05_') and c['family'] in ('AWM','MixedJudge')]

def dump(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else str(x)))

def stats(a):
    x=np.asarray(a,float)
    return dict(n=len(x),min=float(x.min()),p50=float(np.median(x)),p90=float(np.percentile(x,90)),p99=float(np.percentile(x,99)),mean=float(x.mean()))

def sequences(suite, episodes_per_task=2, max_steps=64):
    result=[]
    for arm in ('cache','inf'):
        qc=store.QueryCell(STORE,f'pi05_{suite}_{arm}')
        for task in range(10):
            selected=[i for i,e in enumerate(qc.episodes) if e['task_id']==task][:episodes_per_task]
            for ix in selected:
                e=qc.episodes[ix];lo,hi=e['start'],min(e['end'],e['start']+max_steps)
                ep=api.EpisodeView(e['uid'],e['task'],task,e['init'],ix,plugin.ep_seed(0,e['uid']))
                s=SimpleNamespace(rt=SimpleNamespace(model='pi05',api=api,opts=SimpleNamespace(os_tokens='off')),
                                  hits=np.asarray(qc.exec_hit_flag[lo:hi]).tolist(),has_vision=[True]*(hi-lo),last_vision_step=0,_age_before=0)
                for name,field in [('b_v0','key_v0'),('b_v1','key_v1'),('b_rs','rs'),('b_raw','raw_state'),('b_aex','a_exec')]:
                    ar=np.array(getattr(qc,field)[lo:hi],copy=True);ar.flags.writeable=False
                    buf=plugin._Buf(ar.shape[1:],ar.dtype);buf.a=ar;buf.n=len(ar);setattr(s,name,buf)
                result.append((ep,[plugin.OnlineQueryView(s,j,task,ep) for j in range(hi-lo)]))
    return result

def smi():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total,utilization.gpu,clocks.sm,power.draw','--format=csv,noheader,nounits'],text=True).strip()
    v=[float(x.strip()) for x in raw.split(',')]
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),free_mib=v[0],total_mib=v[1],util_pct=v[2],sm_mhz=v[3],power_w=v[4])

class GPUWatch:
    """Admission at each job; 0.2 s free-memory watchdog; exit only our process."""
    def __init__(self,label):
        self.label=label; self.samples=[]; self.stop=threading.Event();self.check(admit=True)
        self.proc=subprocess.check_output(['nvidia-smi'],text=True)
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        # Leave allowance for CUDA context and non-PyTorch library allocations.
        torch.cuda.set_per_process_memory_fraction(7.0*1024/self.samples[0]['total_mib'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        def poll():
            while not self.stop.wait(.2):
                try: self.check()
                except BaseException as e:
                    dump(OUT/f'ABORT_{label}.json',dict(error=repr(e),samples=self.samples))
                    os._exit(86)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()
    def check(self,admit=False):
        r=smi();self.samples.append(r)
        if r['free_mib'] < (12288 if admit else 4096): raise RuntimeError('GPU memory admission/stop: '+str(r))
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                           '--format=csv,noheader,nounits'],text=True)
        r['own_mib']=0.
        for line in processes.splitlines():
            pid,used=line.split(',')
            if int(pid.strip())==os.getpid():r['own_mib']=float(used.strip())
        if r['own_mib']>8192:raise RuntimeError('K9 own GPU memory exceeds 8 GiB: '+str(r))
    def finish(self):
        import torch
        self.check();self.stop.set();self.thread.join()
        return dict(admission=self.samples[0],samples=self.samples,processes_at_start=self.proc,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)

if __name__=='__main__':
    out=[]
    for c in configs():
        m=load(c['path']);b=getattr(m,'base',m)
        r=dict(id=c['id'],path=c['path'],kwargs=c.get('kwargs'),base_name=b.name,attrs=list(vars(m)),
               task_shapes={k:list(v.shape) for k,v in vars(b.tasks[0]).items() if isinstance(v,np.ndarray)},
               action=list(b.act.shape),task_sizes=b.n_cand)
        out.append(r);print(c['id'],c['path'],b.name,b.act.shape,flush=True)
    dump(OUT/'inventory.json',out)
