"""Validate vectorized refit diagnostic against actual AWM Method.query, each cell/scale."""
import sys,pathlib,json,types,concurrent.futures,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import retrieval_diag as r
from exp.offline_search.harness import api,store
from exp.offline_search.rounds.r02.g1_awm import awm

def run(job):
 model,suite,lib=job;out=[]
 ctx=api.Context(root=r.ROOT,cell=f'{model}_{suite}_cache',seed=0,scratch=r.OUT)
 L=ctx.open_library(lib);sig=ctx.action_sigma;k=5 if lib=='current' else 8
 M=awm.AWM(lib='current' if lib=='current' else 'big',kref=k);M.fit(ctx.open_library(),ctx)
 for arm in ['cache','inf']:
  Q=store.QueryCell(r.ROOT,f'{model}_{suite}_{arm}')
  # Deliberately include every step-0 and subsequent/fresh/stale rows.
  rr=Q.tok_rows;rows=np.unique(np.r_[rr[Q.step[rr]==0],rr[::max(1,len(rr)//75)]])
  X,Xq,_=r.features(L,Q,rows,'full');ev=r.evaluate(L,Q,rows,X,Xq,sig,k)
  act=[];ids=[]
  for row in rows:
   e=Q.episodes[int(Q.ep[row])];start=e['start'];st=int(Q.step[row]);M.reset(None)
   q=types.SimpleNamespace(task_id=e['task_id'],step=st,prev_hit=None if st==0 else arm=='cache',key_v0=Q.key_v0[row],key_v1=Q.key_v1[row],rs=Q.rs[row],prev_a_exec=None if st==0 else Q.a_exec[row-1],hist_key_v0=Q.key_v0[start:row],hist_key_v1=Q.key_v1[start:row])
   res=M.query(q);act.append(res.action[:5,:7]);ids.append(res.topk)
  ids=np.array(ids);d=np.array(act)-ev['pred']
  out.append({'model':model,'suite':suite,'library':lib,'arm':arm,'n':len(rows),'top1_agreement':float(np.mean(ids[:,0]==ev['ids'][:,0])),'overlap16':float(np.mean([len(set(a)&set(b))/16 for a,b in zip(ids,ev['ids'])])),'max_action_abs_difference':float(np.max(np.abs(d))),'mean_action_RMS_sigma_difference':float(np.mean(np.sqrt(np.mean((d/sig)**2,axis=(1,2)))))})
 return out
if __name__=='__main__':
 jobs=[(m,s,l) for m in ['pi05','groot'] for s in ['spatial','l10'] for l in ['current','bpool_cs' if m=='pi05' else 'bpool_all']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:out=[a for x in ex.map(run,jobs) for a in x]
 (r.OUT/'retrieval_validation.json').write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2))
