"""Metric-refitted LOEO sensitivity; frozen deployed PCA and normalization, so not full inductive LOEO."""
import json,time
import numpy as np
from score import *
results=[]
for job in joblist():
 if job['library']!='current':continue
 m=loadmethod(job);p=S/'library'/job['cell']/'current';D=dataset(p,True);X=features(m,p,job['cell']+'_current_refit_lib').astype(float);sig=D['act'][:,:5].reshape(-1,7).std(0);sig=np.where(sig>1e-12,sig,1);heads=(D['act'][:,:5]/m.sig).reshape(-1,35)
 frozen=np.load(O/(job['cell']+'_current_refit_loeo.npz'));errs=np.full(len(X),np.nan);dists=errs.copy();t0=time.time()
 for t,T in m.tasks.items():
  rows=T.rows
  for e in np.unique(D['ep'][rows]):
   tr=rows[D['ep'][rows]!=e];qr=rows[D['ep'][rows]==e];mu,sd,W=fit_metric(X[tr],heads[tr],D['ep'][tr]);z=(X[tr]-mu)/sd@W;qz=(X[qr]-mu)/sd@W;dd=cdist(qz,z)
   early=D['step'][qr]==0;train_early=tr[D['step'][tr]<=2];mu0,sd0,W0=fit_metric(X[train_early],heads[train_early],D['ep'][train_early]);dd[early]=cdist((X[qr[early]]-mu0)/sd0@W0,(X[tr]-mu0)/sd0@W0)
   ix=np.argsort(dd,axis=1,kind='stable')[:,:16];dk=np.take_along_axis(dd,ix,1);w=np.exp(-((dk-dk[:,:1])/np.maximum(dk[:,m.kref-1,None]-dk[:,:1],1e-6))**2);w/=w.sum(1)[:,None];pred=np.einsum('qk,qkhd->qhd',w,D['act'][tr[ix]]);errs[qr]=np.sqrt(np.mean(((pred-D['act'][qr])/sig)**2,axis=(1,2)));dists[qr]=dk[:,0]
  print(job['cell'],'task',t,'seconds',round(time.time()-t0,1),flush=True)
 task=[]
 for t,T in m.tasks.items():
  rr=T.rows;task.append({'task':int(t),'frozen_error':macro(frozen['err10'][rr],D['ep'][rr]),'refit_error':macro(errs[rr],D['ep'][rr]),'frozen_distance':macro(frozen['d1'][rr],D['ep'][rr]),'refit_distance':macro(dists[rr],D['ep'][rr])})
 results.append({'cell':job['cell'],'episodes':len(np.unique(D['ep'])),'tasks':task,'global':{k:float(np.mean([r[k] for r in task])) for k in ['frozen_error','refit_error','frozen_distance','refit_distance']},'seconds':time.time()-t0})
 (O/'metric_refit_audit.json').write_text(json.dumps(results,indent=2))
