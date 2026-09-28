"""Read-only fit loading and measurement utilities for K8."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): assert os.environ.get(k)=='1'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert set(os.sched_getaffinity(0)) <= {34,35,36,37,78,79,80,81}
import json, pathlib, pickle, hashlib, importlib.util, sys, time
import numpy as np
ROOT=pathlib.Path('/home/weiland/projects/openpi')
OUT=ROOT/'exp/offline_search/rounds/r04/k8_search_latency'
STORE=pathlib.Path('/home/weiland/trace_runs/offline_search_store')
RUNS=pathlib.Path('/home/weiland/trace_runs/os_closed_loop')
from exp.offline_search.closed_loop import plugin
from exp.offline_search.harness import api, store
class Loader(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('osm_') and mod not in sys.modules:
            for p in (ROOT/'exp/offline_search').rglob('*.py'):
                if 'osm_'+hashlib.md5(str(p).encode()).hexdigest()[:12]==mod:
                    plugin.load_method_class(str(p)+':'+name);break
        return super().find_class(mod,name)
def load(p):
    with open(p,'rb') as f:return Loader(f).load()
def stats(x):
    a=np.asarray(x,float)
    return dict(n=len(a),mean=float(a.mean()),p50=float(np.percentile(a,50)),p90=float(np.percentile(a,90)),p99=float(np.percentile(a,99))) if len(a) else dict(n=0)
def dump(p,x):
    pathlib.Path(p).write_text(json.dumps(x,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
def cpu():
    result={}
    for l in pathlib.Path('/proc/stat').read_text().splitlines():
        p=l.split()
        if p[0] in {'cpu'+str(c) for c in (34,35,36,37,78,79,80,81)}:
            a=list(map(int,p[1:]));result[p[0]]=[sum(a[:8]),a[3]+a[4]]
    return result
def idle(before,after):
    return {k:100*(after[k][1]-v[1])/max(after[k][0]-v[0],1) for k,v in before.items()}
def base(m):
    while hasattr(m,'base'):m=m.base
    return m
