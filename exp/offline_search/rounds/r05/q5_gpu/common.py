"""Q5 read-only fits and real recorded queries; no owner affinity code imported."""
import hashlib,json,os,pathlib,pickle,sys
from types import SimpleNamespace
import numpy as np
from exp.offline_search.closed_loop import plugin
from exp.offline_search.harness import api,store
ROOT=pathlib.Path('/home/weiland/projects/openpi')
B=ROOT/'exp/offline_search/rounds/r05/q5_gpu'
STORE=pathlib.Path('/home/weiland/trace_runs/offline_search_store')
class Loader(pickle.Unpickler):
    def find_class(self,mod,name):
        if mod.startswith('osm_') and mod not in sys.modules:
            for folder in ('rounds/r02/g1_awm','rounds/r03/h3_judge','rounds/r02/g3_recovery'):
                for p in (ROOT/'exp/offline_search'/folder).glob('*.py'):
                    if 'osm_'+hashlib.md5(str(p).encode()).hexdigest()[:12]==mod:
                        plugin.load_method_class(str(p)+':'+name)
        return super().find_class(mod,name)
def config(name):
    return next(c for c in json.loads((B.parents[1]/'r04/k8_search_latency/configs.json').read_text()) if c['id']==name)
def load(c):
    with open(c['path'],'rb') as f:blob=Loader(f).load()
    assert {k:blob[k] for k in ('spec','kwargs','cell')}=={k:c[k] for k in ('spec','kwargs','cell')}
    m=blob['method'];m.prof=api.NULL_PROFILER
    if hasattr(m,'base'):m.base.prof=api.NULL_PROFILER
    return m
def sequences(suite,episodes_per_task=2,max_steps=64):
    for arm in ('cache','inf'):
        qc=store.QueryCell(STORE,f'pi05_{suite}_{arm}')
        for task in range(10):
            for ix in [i for i,e in enumerate(qc.episodes) if e['task_id']==task][:episodes_per_task]:
                e=qc.episodes[ix];lo,hi=e['start'],min(e['end'],e['start']+max_steps)
                ep=api.EpisodeView(e['uid'],e['task'],task,e['init'],ix,plugin.ep_seed(0,e['uid']))
                s=SimpleNamespace(rt=SimpleNamespace(model='pi05',api=api,opts=SimpleNamespace(os_tokens='off')),
                                  hits=np.asarray(qc.exec_hit_flag[lo:hi]).tolist(),has_vision=[True]*(hi-lo),last_vision_step=0,_age_before=0)
                for name,field in [('b_v0','key_v0'),('b_v1','key_v1'),('b_rs','rs'),('b_raw','raw_state'),('b_aex','a_exec')]:
                    ar=np.array(getattr(qc,field)[lo:hi],copy=True);ar.flags.writeable=False
                    buf=plugin._Buf(ar.shape[1:],ar.dtype);buf.a=ar;buf.n=len(ar);setattr(s,name,buf)
                yield ep,[plugin.OnlineQueryView(s,j,task,ep) for j in range(hi-lo)]
def dump(path,v):
    pathlib.Path(path).write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
def stats(a):
    a=np.asarray(a,float)
    return dict(n=len(a),min=float(a.min()),p50=float(np.median(a)),p90=float(np.percentile(a,90)),max=float(a.max()),mean=float(a.mean()))
