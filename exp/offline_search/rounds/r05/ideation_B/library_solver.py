"""Library-only hyperparameter audit, metric-refitted LOEO conditional on frozen deployed PCA.
Current: every episode; big: ten evenly spaced episodes/task; <=24 evenly spaced queries/episode.
No query-rollout labels. Fixed PCA is transductive; this is NOT full-pipeline inductive LOEO.
LOTO selects a global configuration on the other nine tasks, never retrieves cross-task.
"""
import pathlib,json,datetime,argparse
import numpy as np
from scipy.special import logsumexp
from covariance import oas
S=pathlib.Path('/home/weiland/trace_runs/offline_search_store')
O=pathlib.Path(__file__).parent
BASE=dict(pdim=64,nn=3,distinct=False,lam=.1,state=1.,k=16,kref=5)
CONFIGS={'base':{}}
for kr in (3,5,8,12): CONFIGS['kref'+str(kr)]={'kref':kr}
for k in (8,32):CONFIGS['kernel'+str(k)]={'k':k}
for p in (16,32):CONFIGS['pca'+str(p)]={'pdim':p}
for n in (1,8):CONFIGS['pair'+str(n)]={'nn':n}
CONFIGS['pair_distinct']={'distinct':True}
for r in (.01,1.):CONFIGS['ridge'+str(r)]={'lam':r}
CONFIGS['ridge_oas']={'lam':'oas'}
for w in (0.,3.):CONFIGS['state'+str(w)]={'state':w}

def load(p,n):return np.load(p/(n+'.npy'),mmap_mode='r')
def pairidx(D,ep,n,distinct):
 if not distinct:return np.argsort(D,axis=1,kind='stable')[:,:n]
 order=np.argsort(D,axis=1,kind='stable');out=np.empty((len(ep),n),int)
 for i,row in enumerate(order):
  seen={int(ep[i])};sel=[]
  for j in row:
   if int(ep[j]) not in seen:sel.append(j);seen.add(int(ep[j]))
   if len(sel)==n:break
  out[i]=sel
 return out

def evaluate(cell,scale,stock_sigma=False,tag=''):
 model=cell.split('_')[0];lib='current' if scale==50 else ('bpool_cs' if model=='pi05' else 'bpool_all');p=S/'library'/cell/lib
 arr={n:load(p,n) for n in ['task_id','episode','step','ep_len','progress','next','prev','action','rs','success']}
 rs=np.asarray(arr['rs'][:,:8],float);act=np.asarray(arr['action'][:,:,:7],float)
 sigma=np.asarray(load(S/'library'/cell/'current','action')[:,:5,:7],float).reshape(-1,7).std(0)
 # Original harness uses current sigma even on big. Also record deployed-only sigma to expose provenance.
 own_sigma=act[:,:5].reshape(-1,7).std(0)
 # Strict deployed-library information for this experiment, including big-library scale.
 # At 500 this differs from stock harness's current-library sigma; retain both in metadata.
 fit_sigma=sigma if stock_sigma else own_sigma
 H=(act[:,:5]/fit_sigma).reshape(len(act),-1);TAIL=(act[:,5:10]/fit_sigma).reshape(len(act),-1)
 pp=S/'derived'/('r02/g1_awm/pca_current' if scale==50 else 'r01/f4_vision/pca')/cell
 if scale==500:pp=pp/lib
 P0=np.asarray(load(pp/'v0','proj')[:,:64],float);P1=np.asarray(load(pp/'v1','proj')[:,:64],float)
 ep=np.asarray(arr['episode']);task=np.asarray(arr['task_id']);step=np.asarray(arr['step']);nxt=np.asarray(arr['next']);prv=np.asarray(arr['prev']);prog=np.asarray(arr['progress'])
 epstats=[];calib=[];fitdiagnostics=[]
 for t in np.unique(task):
  rows=np.flatnonzero(task==t);eids=np.unique(ep[rows]);chosen=eids if scale==50 else eids[np.linspace(0,len(eids)-1,min(10,len(eids)),dtype=int)]
  statesd=np.maximum(rs[rows].std(0),.05);delta=(rs[np.maximum(nxt,0)]-rs)/statesd
  delta[nxt<0]=0. # Absorbing terminal candidates, same as blind continuation.
  # Raw motion quantiles of deployed library: includes failures, same as existing fit.
  edges=rows[nxt[rows]>=0];motion=np.linalg.norm(rs[nxt[edges]]-rs[edges],axis=1)
  calib.append({'task':int(t),'episodes':len(eids),'rows':len(rows),'success_episode_fraction':float(np.mean([arr['success'][np.flatnonzero(ep==e)[0]] for e in eids])),'motion_q':dict(zip(['5','10','25','50'],np.quantile(motion,[.05,.1,.25,.5]).tolist())),'normalized_motion_q':dict(zip(['5','10','25','50'],np.quantile(np.sqrt(np.mean(delta[edges]**2,axis=1)),[.05,.1,.25,.5]).tolist())),'median_len':float(np.median([np.sum(ep[rows]==e) for e in eids]))})
  for eid in chosen:
   tr=rows[ep[rows]!=eid];qr=rows[ep[rows]==eid];qr=qr[np.unique(np.r_[0,np.linspace(0,len(qr)-1,min(24,len(qr)),dtype=int)])]
   ht=H[tr];hh=np.sum(ht*ht,axis=1);HD=np.maximum(hh[:,None]+hh[None,:]-2*ht@ht.T,0);HD[ep[tr,None]==ep[tr][None,:]]=np.inf
   pq={};metrics={};results={}
   # Step0 comparison and fresh continuity require no rollout policy calls: demo tail is a proxy.
   for name,over in CONFIGS.items():
    cfg=BASE|({'kref':8} if scale==500 else {})|over;d=cfg['pdim'];n=min(cfg['nn'],len(eids)-2) if cfg['distinct'] else cfg['nn']
    pk=(n,cfg['distinct'])
    if pk not in pq:pq[pk]=pairidx(HD,ep[tr],n,cfg['distinct'])
    ix=pq[pk];X=np.concatenate([P0[:,:d],P1[:,:d],rs],axis=1)
    mk=(d,n,cfg['distinct'],cfg['lam'])
    if mk not in metrics:
     mu=X[tr].mean(0);sd=X[tr].std(0)+1e-6;Z=(X[tr]-mu)/sd;diff=(Z[:,None]-Z[ix]).reshape(-1,X.shape[1]);sw=diff.T@diff/len(diff)
     lam=cfg['lam'];alpha=None
     if lam=='oas':_,alpha=oas(diff,assume_centered=True);lam=alpha/max(1-alpha,1e-9)
     C=sw+float(lam)*np.trace(sw)/len(sw)*np.eye(len(sw));W=np.linalg.cholesky(np.linalg.inv(C));metrics[mk]=(mu,sd,W,float(lam),C)
    mu,sd0,W,lam,C=metrics[mk];sd=sd0.copy();Xq=X[qr].copy();Xt=X[tr].copy()
    # Match AWM: state multiplier acts AFTER metric fitting.
    Zt=(Xt-mu)/sd;Zq=(Xq-mu)/sd;Zt[:,-8:]*=cfg['state'];Zq[:,-8:]*=cfg['state'];Zt=Zt@W;Zq=Zq@W
    D=np.sqrt(np.maximum(np.sum(Zq*Zq,1)[:,None]+np.sum(Zt*Zt,1)[None,:]-2*Zq@Zt.T,0))
    si=np.argsort(D,axis=1,kind='stable')[:,:cfg['k']];dist=np.take_along_axis(D,si,axis=1);bw=np.maximum(dist[:,min(cfg['kref'],si.shape[1])-1]-dist[:,0],1e-6)
    logw=-((dist-dist[:,0,None])/bw[:,None])**2;logw-=logsumexp(logw,axis=1)[:,None];w=np.exp(logw);idx=tr[si]
    pred=np.einsum('qk,qkd->qd',w,H[idx]);err=np.sqrt(np.mean((pred-H[qr])**2,axis=1))
    ok=nxt[qr]>=0 # Identical evaluation population across every candidate configuration.
    pd=np.einsum('qk,qkd->qd',w,delta[idx]);res=np.sqrt(np.mean((pd-delta[qr])**2,axis=1))
    # Gaussian mixture variance fitted from action-near cross-episode training displacement pairs.
    valid=(nxt[tr,None]>=0)&(nxt[tr[ix]]>=0);dd=(delta[tr,None,:]-delta[tr[ix]])[valid]
    var=np.maximum(np.mean(dd*dd,axis=0),1e-8)
    ll=-.5*(np.sum((delta[qr,None]-delta[idx])**2/var,axis=2)+np.log(2*np.pi*var).sum())
    nll=-logsumexp(logw+ll,axis=1)/8
    record={'task':int(t),'episode':int(eid),'config':name,'n':len(qr),'n_successor':int(ok.sum()),'action_RMS':float(err[step[qr]>0].mean()),'state_RMS':float(res[ok].mean()) if ok.any() else None,'state_NLL':float(nll[ok].mean()) if ok.any() else None,'phase_abs':float(np.mean(np.abs(np.einsum('qk,qk->q',w,prog[idx])-prog[qr]))),'step0_action_RMS':float(err[step[qr]==0].mean()),'ridge_equiv':lam}
    epstats.append(record)
    if name=='base':
     # Explicit early metric vs main, and continuity lambda on demo tails.
     et=tr[step[tr]<=2];xe=X[et];me=xe.mean(0);se=xe.std(0)+1e-6;hh0=H[et];hd0=np.sum(hh0**2,1)[:,None]+np.sum(hh0**2,1)[None,:]-2*hh0@hh0.T;hd0[ep[et,None]==ep[et][None,:]]=np.inf
     ii=np.argsort(hd0,axis=1)[:,:3];zz=(xe-me)/se;diff=(zz[:,None]-zz[ii]).reshape(-1,len(me));sw=diff.T@diff/len(diff);we=np.linalg.cholesky(np.linalg.inv(sw+.1*np.trace(sw)/len(sw)*np.eye(len(sw))))
     te=(Xt-me)/se@we;qe=(Xq[step[qr]==0]-me)/se@we;de=np.sqrt(np.maximum(np.sum(qe**2,1)[:,None]+np.sum(te**2,1)[None,:]-2*qe@te.T,0))
     ss=np.argsort(de,axis=1)[:,:16];ds=np.take_along_axis(de,ss,axis=1);ww=np.exp(-((ds-ds[:,0,None])/np.maximum(ds[:,cfg['kref']-1,None]-ds[:,0,None],1e-6))**2);ww/=ww.sum(1)[:,None]
     ee=np.sqrt(np.mean((np.einsum('qk,qkd->qd',ww,H[tr[ss]])-H[qr[step[qr]==0]])**2))
     early={'task':int(t),'episode':int(eid),'main_step0':record['step0_action_RMS'],'early_step0':float(ee),'n_early':len(et),'rank_early':int(np.linalg.matrix_rank(sw)),'dim':len(sw)}
     for lc in (0.,.5,1.):
      fq=prv[qr]>=0
      if fq.any():
       # sc from training demo tails to other-episode heads, as baseline scale.
       cc=np.sqrt(np.maximum(np.sum(TAIL[tr]**2,1)[:,None]+np.sum(H[tr]**2,1)[None,:]-2*TAIL[tr]@H[tr].T,0)/35);cc[ep[tr,None]==ep[tr][None,:]]=np.inf;sc=np.median(cc.min(1))
       qtail=TAIL[prv[qr[fq]]];ct=np.sqrt(np.maximum(np.sum(qtail**2,1)[:,None]+np.sum(H[tr]**2,1)[None,:]-2*qtail@H[tr].T,0)/35);df=D[fq]/np.maximum(np.median(D[fq],axis=1)[:,None],1e-12)+lc*ct/max(sc,1e-6)
       ss=np.argsort(df,axis=1)[:,:16];ds=np.take_along_axis(df,ss,axis=1);ww=np.exp(-((ds-ds[:,0,None])/np.maximum(ds[:,cfg['kref']-1,None]-ds[:,0,None],1e-6))**2);ww/=ww.sum(1)[:,None]
       early['fresh_demo_lc'+str(lc)]=float(np.sqrt(np.mean((np.einsum('qk,qkd->qd',ww,H[tr[ss]])-H[qr[fq]])**2,axis=1)).mean())
     fitdiagnostics.append(early)
   print(cell,scale,'task',t,'episode',eid,flush=True)
 # Episode macro means then task macro; only eligible metric rows.
 tables={}
 for name in CONFIGS:
  ss=[r for r in epstats if r['config']==name];table={}
  for key in ['action_RMS','state_RMS','state_NLL','phase_abs','step0_action_RMS','ridge_equiv']:
   tt={str(int(t)):float(np.mean([r[key] for r in ss if r['task']==t and r[key] is not None])) for t in np.unique(task)}
   table[key]={'mean':float(np.mean(list(tt.values()))),'tasks':tt}
  tables[name]=table
 loto={}
 for obj in ['action_RMS','state_RMS','state_NLL']:
  folds=[]
  for t in range(10):
   best=min(CONFIGS,key=lambda n:np.mean([v for k,v in tables[n][obj]['tasks'].items() if int(k)!=t]))
   folds.append({'held_task':t,'selected':best,'selected_loss':tables[best][obj]['tasks'][str(t)],'base_loss':tables['base'][obj]['tasks'][str(t)]})
  loto[obj]={'folds':folds,'selected_loss':float(np.mean([r['selected_loss'] for r in folds])),'base_loss':float(np.mean([r['base_loss'] for r in folds])),'all_tasks_selected':min(CONFIGS,key=lambda n:tables[n][obj]['mean'])}
 out={'cell':cell,'scale':scale,'library':str(p),'rows':len(ep),'episodes':len(np.unique(ep)),'heldout_episodes':len(fitdiagnostics),'metric_validation':'LOEO metric refit conditional on full deployed-library PCA and action scales; not strict full pipeline LOEO','sampling':'all 50 library episodes; 10 evenly spaced episodes/task at 500; <=24 rows/episode; episode then task macro average','action_sigma_current':sigma.tolist(),'action_sigma_own':own_sigma.tolist(),'configs':{n:BASE|({'kref':8} if scale==500 else {})|v for n,v in CONFIGS.items()},'table':tables,'LOTO':loto,'calibration':calib,'regime_audit':fitdiagnostics,'episode_scores':epstats}
 out['sigma_source']='current-library benchmark units (beyond deployed big-library rows)' if stock_sigma else 'deployed library only'
 out['algorithm_version']='r5b_v3_fixed_population_euclidean_distance'
 (O/f'{tag}{"stock_" if stock_sigma else ""}library_{cell}_{scale}.json').write_text(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--cells',nargs='*',default=['pi05_spatial','pi05_l10','groot_spatial','groot_l10']);a.add_argument('--scales',nargs='*',type=int,default=[50,500]);a.add_argument('--stock-sigma',action='store_true');a.add_argument('--restricted',action='store_true');a.add_argument('--tag',default='');args=a.parse_args()
 if args.restricted:CONFIGS={k:v for k,v in CONFIGS.items() if k in ('base','kref5','kref8','ridge1.0','state3.0')}
 for cell in args.cells:
  for scale in args.scales:evaluate(cell,scale,args.stock_sigma,args.tag)
