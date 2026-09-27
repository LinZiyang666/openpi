"""Full-16 metric replay on a seeded sample of stale, non-step0 trace states.
No stateful rollout and no closed-loop success estimate. All outputs local.
"""
import json,time
import numpy as np
from diagnose import OUT,STORE,RESULTS,arr,mean
from metric_cv import stats

def main():
 out={};start=time.monotonic()
 for model in ['pi05','groot']:
  for suite in ['spatial','l10']:
   key=f'{model}_{suite}';cell=key+'_cache';p=STORE/'library'/key/'current'
   pc=STORE/'derived/r02/g1_awm/pca_current'/key
   x=np.c_[arr(pc/'v0','proj'),arr(pc/'v1','proj'),arr(p,'rs')[:,:8]].astype(float)
   a=np.array(arr(p,'action')[:,:5,:7],float)
   z=np.load(RESULTS/'AWM_joint_cur_fcur_kr5'/(cell+'.npz'))
   jj=json.loads((RESULTS/'AWM_joint_cur_fcur_kr5'/(cell+'.json')).read_text())
   sig=np.array(jj['sigma']);h=(a/sig).reshape(-1,35);e=arr(p,'episode');t=arr(p,'task_id')
   pool=np.flatnonzero(z['step']>0); ix=np.sort(np.random.default_rng(302).choice(pool,min(2000,len(pool)),replace=False))
   rows=z['row'][ix];qp=STORE/'queries'/cell
   parts=[]
   for cam in ['v0','v1']:
    basis=arr(pc/cam,'basis');mu=arr(pc/cam,'mean')
    keys=np.array(arr(qp,'key_'+cam)[rows]);parts.append(keys@basis-mu@basis)
   q=np.c_[parts[0],parts[1],arr(qp,'rs')[rows,:8]].astype(float)
   gt=np.array(arr(qp,'a_inf')[rows,:5,:7]);qt=z['task_id'][ix]
   fits={task:stats(x[t==task],h[t==task],e[t==task]) for task in np.unique(t)}
   preds={v:np.empty_like(gt) for v in ['baseline','pool05','ridge1']}
   for task,(mu,sd,sw) in fits.items():
    tr=t==task;qa=qt==task; pooled=np.mean([cv for tt,(_,_,cv) in fits.items() if tt!=task],axis=0)
    for v in preds:
     cov=.5*sw+.5*pooled if v=='pool05' else sw;lam=1. if v=='ridge1' else .1
     W=np.linalg.cholesky(np.linalg.inv(cov+lam*np.trace(cov)/len(cov)*np.eye(len(cov))))
     zz=(x[tr]-mu)/sd@W;qq=(q[qa]-mu)/sd@W
     d=np.sqrt(np.maximum((qq*qq).sum(1)[:,None]+(zz*zz).sum(1)[None,:]-2*qq@zz.T,0))
     inds=np.argsort(d,axis=1)[:,:16];ds=np.take_along_axis(d,inds,1)
     w=np.exp(-((ds-ds[:,:1])/np.maximum(ds[:,4]-ds[:,0],1e-6)[:,None])**2);w/=w.sum(1,keepdims=True)
     preds[v][qa]=np.einsum('nk,nktd->ntd',w,a[tr][inds])
   vv={};baseerr=np.sqrt(np.mean(((preds['baseline']-gt)/sig)**2,axis=(1,2)))
   for v,pr in preds.items():
    err=np.sqrt(np.mean(((pr-gt)/sig)**2,axis=(1,2)))
    vote=np.abs(pr[:,0,6]);gmis=np.mean((pr[:,:,6]>=0)!=(gt[:,:,6]>=0),axis=1)
    vv[v]={'err':mean(err),'median':float(np.median(err)),'delta':mean(err-baseerr),'grip_mis':mean(gmis),
       'split08':mean(vote<.8),'by_task':{str(i):{'n':int((qt==i).sum()),'err':mean(err[qt==i]),'split08':mean(vote[qt==i]<.8)} for i in np.unique(qt)}}
   out[cell]={'n':len(ix),'library_entries':len(x),'fit_episodes':len(np.unique(e)),
     'baseline_saved_action_rms_sigma':mean(np.sqrt(np.mean(((preds['baseline']-z['synth_seg'][ix])/sig)**2,axis=(1,2)))),
     'variants':vv}
   np.savez_compressed(OUT/(cell+'_metric_sample.npz'),row=rows,task=qt,ep=z['ep'][ix],gt=gt,**preds)
   print(cell,{v:round(m['err'],4) for v,m in vv.items()},'baseline_recon',out[cell]['baseline_saved_action_rms_sigma'],flush=True)
 out['wall_s']=time.monotonic()-start
 (OUT/'metric_trace_sample.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
