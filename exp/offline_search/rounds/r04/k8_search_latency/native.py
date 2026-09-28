"""Actual native CP1 strategy/backend replay, no server and no GPU."""
import os,argparse
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--suite',required=True);p.add_argument('--run',type=int,default=1);a=p.parse_args()
os.sched_setaffinity(0,{34})
from common import *
import torch
import openpi.cache.config as cc
from openpi.cache.components.search_strategy import SearchContext
from openpi.cache.types import CheckpointID
torch.set_num_threads(1);torch.set_num_interop_threads(1)
name=f'{a.model}_{a.suite}';yaml=f'exp/trace_dual/config/tr_{a.model}_{"sp" if a.suite=="spatial" else "l10"}_cache.yaml'
cfg=cc.load_cache_config(yaml);assert cfg.backend.type=='in_memory';assert cfg.write_policy.type=='never'
shared=cc.build_shared_storage(cfg);comps=cc.build_per_connection_components(cfg,shared,quiet=True);strategy=comps['search_strategies'][CheckpointID.CP1]
qc=store.QueryCell(STORE,name+'_cache');lib=store.LibraryView(STORE,name,'current');ids={v:i for i,v in enumerate(lib.ids)}
ctxs=[];expected=[]
for task in range(10):
 e=next(e for e in qc.episodes if e['task_id']==task)
 for row in range(e['start'],min(e['end'],e['start']+64)):
  keys={k:torch.from_numpy(np.array(getattr(qc,f)[row])) for k,f in [('vision_0','key_v0'),('vision_1','key_v1'),('robot_state','rs')]}
  ctxs.append(SearchContext(keys,CheckpointID.CP1,current_step=row-e['start'],task_key=e['task']));expected.append(int(qc.rec_top1[row]))
for ctx in ctxs:strategy.search(ctx)
before=cpu();ld=os.getloadavg();ts=[];checks=[]
for i in range(1024):
 j=i%len(ctxs);t=time.perf_counter_ns();res=strategy.search(ctxs[j]);ts.append((time.perf_counter_ns()-t)/1e6)
 if i<len(ctxs):checks.append(dict(expected=expected[j],got=ids.get(res[0].id,-1),score=float(res[0].score)))
r=dict(model=a.model,suite=a.suite,method='deployed native cp1',yaml=yaml,preload=cfg.backend.in_memory.preload_path,entries=lib.L,scale=50,applies_to_method_scales=[50,500],ms=stats(ts),samples_ms=ts,recorded_top1_equal=sum(c['got']==c['expected'] for c in checks),recorded_n=len(checks),checks=checks,load=ld,idle_percent=idle(before,cpu()),affinity=sorted(os.sched_getaffinity(0)))
dump(OUT/f'native_{name}_r{a.run}.json',r);print(name,r['ms'],r['recorded_top1_equal'],r['recorded_n'],flush=True)
