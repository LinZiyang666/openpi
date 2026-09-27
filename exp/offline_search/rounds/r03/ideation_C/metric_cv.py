"""Current-only leave-one-episode-out metric audit. PCA/sigma fixed to current library.
No held-out episode is used in supervised covariance or candidate actions.
Not equivalent to the offline teacher scoreboard or to closed-loop SR.
"""
import json,time
import numpy as np
from diagnose import OUT,STORE,arr,mean

def stats(X,H,E,balanced=False):
    mu=X.mean(0);sd=X.std(0)+1e-6;Z=(X-mu)/sd
    dh=np.maximum((H*H).sum(1)[:,None]+(H*H).sum(1)[None,:]-2*H@H.T,0)
    dh[E[:,None]==E[None,:]]=np.inf
    if balanced:
        diffs=[]
        for ep in np.unique(E):
            cand=np.flatnonzero(E==ep);q=np.flatnonzero(E!=ep)
            n=cand[np.argmin(dh[np.ix_(q,cand)],axis=1)]
            diffs.extend(Z[q]-Z[n])
        diffs=np.array(diffs)
    else:
        ni=np.argpartition(dh,3,axis=1)[:,:3]
        diffs=(Z[:,None,:]-Z[ni]).reshape(-1,Z.shape[1])
    sw=diffs.T@diffs/len(diffs)
    return mu,sd,sw

def main():
 out={};start=time.monotonic()
 for model in ['pi05','groot']:
  for suite in ['spatial','l10']:
   key=f'{model}_{suite}';p=STORE/'library'/key/'current'
   pc=STORE/'derived/r02/g1_awm/pca_current'/key
   x=np.c_[arr(pc/'v0','proj'),arr(pc/'v1','proj'),arr(p,'rs')[:,:8]].astype(float)
   a=np.array(arr(p,'action')[:,:5,:7],float);sig=a.std((0,1));h=(a/sig).reshape(-1,35)
   e=arr(p,'episode');t=arr(p,'task_id')
   allsw={};summary={}
   for task in np.unique(t):
    ix=t==task;mu,sd,sw=stats(x[ix],h[ix],e[ix]);allsw[task]=sw
    ev=np.linalg.eigvalsh(sw);reg=.1*np.trace(sw)/len(sw)
    dh=(h[ix]*h[ix]).sum(1)[:,None]+(h[ix]*h[ix]).sum(1)[None,:]-2*h[ix]@h[ix].T
    et=e[ix];dh[et[:,None]==et[None,:]]=np.inf;nn=np.argpartition(dh,3,axis=1)[:,:3]
    pair_ep=et[nn]
    summary[str(task)]={'rows':int(ix.sum()),'episodes':len(np.unique(et)),
      'rank':int((ev>max(ev.max(),1)*1e-8).sum()),'condition_regularized':float((ev.max()+reg)/(max(ev.min(),0)+reg)),
      'trace_effective_rank':float(ev.sum()**2/(ev@ev)),
      '3pairs_same_episode_frac':mean(np.all(pair_ep==pair_ep[:,:1],axis=1))}
   variants=['baseline','ridge1','pool05','pool1','episode_pairs','pca16']
   errors={v:[] for v in variants};grips={v:[] for v in variants};tasks=[]
   for ep in np.unique(e):
    va=e==ep;task=t[va][0];tr=(t==task)&~va
    mu,sd,sw=stats(x[tr],h[tr],e[tr])
    pooled=np.mean([v for k,v in allsw.items() if k!=task],axis=0)
    for v in variants:
     xx=x;mm,ss,cov=mu,sd,sw;lam=.1
     if v=='ridge1':lam=1.
     if v=='pool05':cov=.5*sw+.5*pooled
     if v=='pool1':cov=pooled
     if v=='episode_pairs':mm,ss,cov=stats(xx[tr],h[tr],e[tr],True)
     if v=='pca16':
      xx=x[:,np.r_[0:16,64:80,128:136]];mm,ss,cov=stats(xx[tr],h[tr],e[tr])
     mat=cov+lam*np.trace(cov)/len(cov)*np.eye(len(cov))
     W=np.linalg.cholesky(np.linalg.inv(mat));z=(xx[tr]-mm)/ss@W;q=(xx[va]-mm)/ss@W
     d=np.sqrt(np.maximum((q*q).sum(1)[:,None]+(z*z).sum(1)[None,:]-2*q@z.T,0))
     idx=np.argsort(d,axis=1)[:,:16];ds=np.take_along_axis(d,idx,axis=1)
     w=np.exp(-((ds-ds[:,:1])/np.maximum(ds[:,4]-ds[:,0],1e-6)[:,None])**2);w/=w.sum(1,keepdims=True)
     pred=np.einsum('nk,nkd->nd',w,h[tr][idx]);err=np.sqrt(np.mean((pred-h[va])**2,axis=1))
     errors[v].extend(err);grips[v].extend(np.mean((pred.reshape(-1,5,7)[:,:,6]>=0)!=(a[va,:,6]>=0),axis=1))
    tasks.extend(t[va])
   tasks=np.array(tasks)
   out[key]={'rows':len(x),'episodes':len(np.unique(e)),'task_metric':summary,'cv':{}}
   for v in variants:
    er=np.array(errors[v]);out[key]['cv'][v]={'err':mean(er),'median':float(np.median(er)),
      'grip_mis':mean(grips[v]),'by_task':{str(i):mean(er[tasks==i]) for i in np.unique(tasks)}}
   print(key,{v:round(out[key]['cv'][v]['err'],4) for v in variants},flush=True)
 out['wall_s']=time.monotonic()-start
 (OUT/'metric_cv.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
