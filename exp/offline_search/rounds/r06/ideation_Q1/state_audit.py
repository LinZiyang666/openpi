"""Online-observable quality versus held-out error and paired closed-loop outcomes."""
import json,collections
import numpy as np
from scipy.stats import spearmanr,rankdata
from score import O,joblist,macro
rng=np.random.default_rng(20260928);out=[];phase=[]
J=json.loads((O/'source_index.json').read_text());R=json.loads((O/'closed_loop_records.json').read_text())['A'];arms=J['arms']
def accepted(lab):
 acc={}
 for line in open(arms[lab]['journal']):
  r=json.loads(line)
  if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error'):acc[tuple(map(int,r['task_uid'].split(':')[-2:]))]=int(r['success'])
 return acc
for job in joblist():
 tag='_'.join([job['cell'],job['library'],job['variant']]);summary=json.loads((O/(tag+'.json')).read_text())
 for source in ['inf','cache']:
  a=np.load(O/(tag+'_'+source+'.npz'));valid=np.isfinite(a['q_online'])&np.isfinite(a['succ2']);tasks=np.unique(a['task']);rec={'tag':tag,'source':source,'n':int(valid.sum())}
  for metric in ['d_pair','d_self','q_online']:
   v=a[metric];sign=1 if metric=='q_online' else -1
   for target in ['err10','succ2']:
    bytask=[spearmanr(sign*v[valid&(a['task']==t)],-a[target][valid&(a['task']==t)]).statistic for t in tasks];rec[metric+'_vs_'+target]=float(np.mean(bytask));rec[metric+'_vs_'+target+'_ci']=np.quantile(np.mean(rng.choice(bytask,(2000,len(bytask)),replace=True),1),[.025,.975]).tolist()
  # Pair each recorded trajectory score to the same benchmark task/init A outcome: association, not on-policy state value.
  labels=list(dict.fromkeys(r['label'] for r in R if r['tag']==tag));states=[]
  if labels:
   lab=labels[0];aj=accepted(lab)
   for e in np.unique(a['ep']):
    z=(a['ep']==e)&np.isfinite(a['q_online']);ii=np.flatnonzero(z)
    if not len(ii):continue
    t=int(a['task'][ii[0]]);init=int(a['init'][ii[0]]);key=(t,init)
    if key in aj:states.append((t,float(a['q_online'][z].mean()),float(a['covered'][z].mean()),aj[key]))
   tt=np.array(states);rec['A_label']=lab;rec['episode_q_A_sr_rho']=float(spearmanr(tt[:,1],tt[:,3]).statistic)
   rec['episode_coverage_A_sr_rho']=float(spearmanr(tt[:,2],tt[:,3]).statistic)
   draws=[]
   for _ in range(2000):
    ix=np.concatenate([np.flatnonzero(tt[:,0]==t) for t in rng.choice(tasks,len(tasks),replace=True)]);rr=spearmanr(tt[ix,1],tt[ix,3]).statistic
    if np.isfinite(rr):draws.append(rr)
   rec['episode_q_A_sr_ci']=np.quantile(draws,[.025,.975]).tolist() if draws else None
  out.append(rec)
  # Macro-average every phase across task and episode. No semantic task phases are assumed.
  byphase=[]
  for b in range(5):
   means={}
   for metric in ['covered','d_pair','err10','succ2']:
    values=[]
    for t in tasks:
     z=(a['task']==t)&np.isfinite(a['d1'])&(np.minimum((a['progress']*5).astype(int),4)==b);values.append(macro(a[metric][z],a['ep'][z]))
    means[metric]=float(np.mean(values))
   byphase.append(means)
  phase.append({'tag':tag,'source':source,'bins':byphase})
 print(tag,flush=True)
(O/'state_audit.json').write_text(json.dumps(out,indent=2));(O/'phase_summary.json').write_text(json.dumps(phase,indent=2))
