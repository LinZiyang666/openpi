"""Borrowed-big-library alpha fit: current-episode LOEO in a frozen BIG PCA/standardization.
Alpha=0 is already borrowed. Jackknife shrinkage is a moment heuristic because nearest-action
pair distributions depend on library size; its common-covariance premise is not guaranteed.
"""
import pathlib,json,numpy as np
S=pathlib.Path('/home/weiland/trace_runs/offline_search_store');O=pathlib.Path(__file__).parent
def a(p,k):return np.load(p/(k+'.npy'),mmap_mode='r')
def covariance(X,H,e):
 D=np.maximum(np.sum(H*H,1)[:,None]+np.sum(H*H,1)[None,:]-2*H@H.T,0);D[e[:,None]==e[None,:]]=np.inf
 ii=np.argsort(D,axis=1)[:,:3];dx=(X[:,None]-X[ii]).reshape(-1,X.shape[1]);return dx.T@dx/len(dx)
def project(X,p):
 B=np.asarray(a(p,'basis')[:,:64]);mu=np.asarray(a(p,'mean'));P=np.empty((len(X),64),np.float32)
 for s in range(0,len(X),128):P[s:s+128]=np.asarray(X[s:s+128])@B-mu@B
 return P
out=[]
for cell in ['pi05_spatial','pi05_l10','groot_spatial','groot_l10']:
 lib='bpool_cs' if cell.startswith('pi05') else 'bpool_all';cp=S/'library'/cell/'current';bp=S/'library'/cell/lib;pp=S/'derived/r01/f4_vision/pca'/cell/lib
 xc=np.c_[project(a(cp,'key_v0'),pp/'v0'),project(a(cp,'key_v1'),pp/'v1'),a(cp,'rs')[:,:8]].astype(float)
 xb=np.c_[a(pp/'v0','proj')[:,:64],a(pp/'v1','proj')[:,:64],a(bp,'rs')[:,:8]].astype(float)
 ac=np.array(a(cp,'action')[:,:5,:7],float);sig=ac.reshape(-1,7).std(0);hc=(ac/sig).reshape(len(ac),-1);hb=(a(bp,'action')[:,:5,:7]/sig).reshape(len(xb),-1)
 ec=a(cp,'episode');eb=a(bp,'episode');tc=a(cp,'task_id');tb=a(bp,'task_id');nxt=a(cp,'next');rs=xc[:,-8:];records=[];ebfit=[]
 for t in range(10):
  cr=np.flatnonzero(tc==t);br=np.flatnonzero(tb==t);mu=xb[br].mean(0);sd=xb[br].std(0)+1e-6;zc=(xc-mu)/sd;zb=(xb[br]-mu)/sd;sb=covariance(zb,hb[br],eb[br]);sc=covariance(zc[cr],hc[cr],ec[cr]);js=[]
  state_std=np.maximum(rs[cr].std(0),.05);delta=(rs[np.maximum(nxt,0)]-rs)/state_std;delta[nxt<0]=0
  for eid in np.unique(ec[cr]):
   tr=cr[ec[cr]!=eid];qr=cr[ec[cr]==eid];qr=qr[np.linspace(0,len(qr)-1,min(24,len(qr)),dtype=int)];sq=covariance(zc[tr],hc[tr],ec[tr]);js.append(sq)
   for alpha in [0,.25,.5,.75,1]:
    sw=(1-alpha)*sq+alpha*sb;W=np.linalg.cholesky(np.linalg.inv(sw+.1*np.trace(sw)/len(sw)*np.eye(len(sw))));zt=zc[tr]@W;zq=zc[qr]@W
    D=np.sqrt(np.maximum(np.sum(zq*zq,1)[:,None]+np.sum(zt*zt,1)[None,:]-2*zq@zt.T,0));ix=np.argsort(D,axis=1)[:,:16];d=np.take_along_axis(D,ix,1);w=np.exp(-((d-d[:,0,None])/np.maximum(d[:,4,None]-d[:,0,None],1e-6))**2);w/=w.sum(1)[:,None];idx=tr[ix]
    pred=np.einsum('qk,qkd->qd',w,hc[idx]);err=np.sqrt(np.mean((pred-hc[qr])**2,1));pd=np.einsum('qk,qkd->qd',w,delta[idx]);de=np.sqrt(np.mean((pd-delta[qr])**2,1));ok=nxt[qr]>=0
    records.append({'task':t,'episode':int(eid),'alpha':alpha,'action_RMS':float(err[a(cp,'step')[qr]>0].mean()),'state_RMS':float(de[ok].mean())})
  js=np.array(js);var=float((len(js)-1)/len(js)*np.sum((js-js.mean(0))**2));dist=float(np.sum((sc-sb)**2));ebfit.append({'task':t,'jackknife_variance':var,'target_squared_distance':dist,'moment_alpha':min(1.,var/max(dist,1e-12))})
 tabs={str(al):{k:float(np.mean([np.mean([r[k] for r in records if r['alpha']==al and r['task']==t]) for t in range(10)])) for k in ['action_RMS','state_RMS']} for al in [0,.25,.5,.75,1]}
 out.append({'cell':cell,'candidate_scale':50,'fit_scale':500,'disclosure':'borrowed big-library information, including alpha=0','table':tabs,'moment_fit':ebfit,'median_moment_alpha':float(np.median([r['moment_alpha'] for r in ebfit])),'episode_scores':records,'at_500':'No separate prior: fit and deploy the big library directly; alpha is redundant.'})
 (O/'prior_results.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(cell,tabs,flush=True)
