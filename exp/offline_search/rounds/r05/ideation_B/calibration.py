"""Two distinct threshold audits: deployed-demo guard percentiles; borrowed rollout residual quantiles.
A percentile level or false-alarm budget remains a design choice, not a solved utility.
"""
import json,pathlib
import numpy as np
S=pathlib.Path('/home/weiland/trace_runs/offline_search_store');O=pathlib.Path(__file__).parent
A=pathlib.Path('exp/offline_search/rounds/r04/ideation_A')
def ar(p,k):return np.load(p/(k+'.npy'),mmap_mode='r')
def demo(cell,scale):
 model=cell.split('_')[0];name='current' if scale==50 else ('bpool_cs' if model=='pi05' else 'bpool_all');p=S/'library'/cell/name
 task=ar(p,'task_id');ep=ar(p,'episode');prev=ar(p,'prev');nxt=ar(p,'next');rs=ar(p,'rs')[:,:8];suc=ar(p,'success');valid=prev>=0
 motion=np.full(len(ep),np.inf);motion[valid]=np.linalg.norm(rs[valid]-rs[prev[valid]],axis=1)
 vcs=[]
 for cam in ('v0','v1'):
  key=ar(p,'key_'+cam);v=np.full(len(ep),np.nan)
  for t in np.unique(task):
   rr=np.flatnonzero(task==t);mean=np.asarray(key[rr],np.float64).mean(0);rr=rr[prev[rr]>=0]
   for lo in range(0,len(rr),128):
    ii=rr[lo:lo+128];x=np.asarray(key[ii],float)-mean;y=np.asarray(key[prev[ii]],float)-mean
    v[ii]=np.einsum('nd,nd->n',x,y)/np.maximum(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1),1e-12)
  vcs.append(v)
 v=np.minimum(*vcs);runs=[]
 for mp in [5,10,20,30]:
  for cp in [90,95,99]:
   mt=float(np.percentile(motion[valid],mp));ct=float(np.percentile(v[valid],cp));still=(motion<mt)&(v>=ct);dense=motion<mt
   for n in [1,2,3]:
    counts=[];dcounts=[]
    for e in np.unique(ep):
     rr=np.flatnonzero(ep==e);run=drun=0;alerts=dalerts=0
     for r in rr:
      run=run+1 if still[r] else 0;drun=drun+1 if dense[r] else 0;alerts+=run>=n;dalerts+=drun>=n
     counts.append([int(e),bool(suc[rr[0]]),alerts]);dcounts.append(dalerts)
    good=np.array([c[1] for c in counts]);ctt=np.array([c[2] for c in counts]);dd=np.array(dcounts)
    runs.append({'m_pct':mp,'c_pct':cp,'stuck_thr':n,'m_thr':mt,'c_thr':ct,'rows_flagged':int(ctt.sum()),'episode_flag_rate':float((ctt>0).mean()),'successful_episode_flag_rate':float((ctt[good]>0).mean()) if good.any() else None,'dense_rows_flagged':int(dd.sum()),'dense_successful_episode_flag_rate':float((dd[good]>0).mean()) if good.any() else None})
 # LOEO quantile stability (global, same stock specification); grouping is by episode.
 q=[]
 for e in np.unique(ep):
  tr=valid&(ep!=e);q.append([np.percentile(motion[tr],10),np.percentile(v[tr],95)])
 return {'cell':cell,'scale':scale,'episodes':len(np.unique(ep)),'successful_episodes':sum(bool(suc[np.flatnonzero(ep==e)[0]]) for e in np.unique(ep)),'rows':len(ep),'stock':next(r for r in runs if r['m_pct']==10 and r['c_pct']==95 and r['stuck_thr']==2),'LOEO_threshold_min':np.min(q,axis=0).tolist(),'LOEO_threshold_max':np.max(q,axis=0).tolist(),'grid':runs}
def residual(cell,scale,h):
 # Full-inference rollout calibration transferred to pure-cache distribution.
 inf=np.load(A/f'windows_{cell}_inf_{scale}_h{h}.npz');ca=np.load(A/f'windows_{cell}_cache_{scale}_h{h}.npz')
 qp=S/'queries'/f'{cell}_inf';ep=ar(qp,'ep')[inf['anchor']];eps=json.loads((qp/'episodes.json').read_text());t=np.array([e['task_id'] for e in eps])[ep]
 cp=S/'queries'/f'{cell}_cache';ec=ar(cp,'ep')[ca['anchor']];ceps=json.loads((cp/'episodes.json').read_text());tc=np.array([e['task_id'] for e in ceps])[ec]
 q95=float(np.quantile(inf['dev'],.95));ex=[]
 for task in range(10):
  tau=float(np.quantile(inf['dev'][t!=task],.95));vi=t==task;vc=tc==task
  ex.append({'task':task,'tau':tau,'inf_exceed':float(np.mean(inf['dev'][vi]>tau)),'cache_exceed':float(np.mean(ca['dev'][vc]>tau))})
 # Distribution-free episode-max calibration would be much less permissive: independent episodes, not rows.
 maxima=np.array([inf['dev'][ep==e].max() for e in np.unique(ep)])
 def err(b,k):return float(np.mean(b[k])) if k in b.files else None
 return {'cell':cell,'scale':scale,'h':h,'provenance':'borrowed big-library information: full-inference/cache logged query states, not deployed demonstrations','n_inf':len(ep),'n_cache':len(ec),'q95_inf':q95,'q95_cache':float(np.quantile(ca['dev'],.95)),'tau05_inf_exceed':float(np.mean(inf['dev']>.5)),'tau05_cache_exceed':float(np.mean(ca['dev']>.5)),'inf95_cache_exceed':float(np.mean(ca['dev']>q95)),'episode_max_q95':float(np.quantile(maxima,.95,method='higher')),'LOTO':ex,'cache_action_phase':err(ca,'err_phase_abs'),'cache_action_tail':err(ca,'err_anchor_chunk'),'cache_action_clock':err(ca,'err_kernel_clock'),'inf_action_phase':err(inf,'err_phase_abs'),'inf_action_tail':err(inf,'err_anchor_chunk')}
if __name__=='__main__':
 out={'demo':[],'residual':[]}
 for cell in ['pi05_spatial','pi05_l10','groot_spatial','groot_l10']:
  for scale in [50,500]:
   out['demo'].append(demo(cell,scale))
   for h in [1,2]:out['residual'].append(residual(cell,scale,h))
   (O/'calibration_results.json').write_text(json.dumps(out,indent=2));print(cell,scale,flush=True)
