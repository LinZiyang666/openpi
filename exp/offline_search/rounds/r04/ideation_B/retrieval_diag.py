"""Refit AWM on same-library cheaper keys. Read-only cold store; tok query subset.
Camera deletion uses exact stored policy-tower keys. Pooling the stored 4x4 grid
commutes with pooling the 16x16 token grid. These pool variants DO NOT save tower work.
No teacher/query actions enter the fit; no borrowing. No arrays >50 MB are written.
"""
import os,sys,json,time,pathlib,concurrent.futures,types
import numpy as np
REPO=pathlib.Path(__file__).resolve().parents[5];sys.path.insert(0,str(REPO))
from exp.offline_search.harness import api,store
from exp.offline_search.rounds.r02.g1_awm import awm
OUT=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/home/weiland/trace_runs/offline_search_store')

def pooled(x,side):
 x=np.asarray(x,np.float32).reshape(-1,4,4,2048)
 if side==1:return x.mean((1,2))
 return x.reshape(-1,2,2,2,2,2048).mean((2,4)).reshape(len(x),-1)

class Reduced:
 def __init__(self,a,side):self.a=a;self.side=side;self.shape=(len(a),2048*side*side)
 def __getitem__(self,k):return pooled(self.a[k],self.side)

def features(L,Q,rows,variant):
 ps=[];pqs=[];fixed=0
 for cam in [0,1]:
  if variant=='cam0' and cam==1 or variant=='cam1' and cam==0:continue
  field=f'v{cam}'
  if variant in ('full','cam0','cam1'):
   if L.name=='current':mu,B,P=awm.pca_current(L,L.key,field)
   else:
    d=awm.PCA_BIG/L.key/L.name/field
    mu=np.load(d/'mean.npy');B=np.load(d/'basis.npy',mmap_mode='r')[:,:64];P=np.load(d/'proj.npy',mmap_mode='r')[:,:64]
   QP=awm.project(np.asarray(getattr(Q,'key_'+field)[rows]),mu,B)
  else:
   side={'pool1':1,'pool2':2}[variant]
   X=Reduced(getattr(L,'key_'+field),side)
   mu,B,P=awm.pca_fit(X)
   QP=awm.project(pooled(getattr(Q,'key_'+field)[rows],side),mu,B)
  ps.append(P);pqs.append(QP);fixed+=int(mu.nbytes+B.nbytes)
 return np.column_stack(ps+[np.asarray(L.rs[:,:8])]).astype(np.float64),np.column_stack(pqs+[np.asarray(Q.rs[rows,:8])]).astype(np.float64),fixed

def evaluate(L,Q,rows,X,Xq,sig,kref):
 H=np.asarray(L.action[:,:5,:7],np.float64)/sig; heads=H.reshape(len(H),-1)
 tail=np.asarray(L.action[:,5:10,:7],np.float64).reshape(len(H),-1)/np.tile(sig,5)
 ep=np.asarray(L.episode);step=np.asarray(L.step)
 tq=np.array([e['task_id'] for e in Q.episodes])[Q.ep[rows]]
 pred=np.empty((len(rows),5,7));ids=np.empty((len(rows),16),np.int64);weights=np.empty((len(rows),16));fixed=0
 for t in L.tasks():
  li=np.asarray(L.rows_of_task(t));qi=np.flatnonzero(tq==t)
  hf=heads[li];ef=ep[li]
  mn,sd,W=awm.fit_metric(X[li],hf,ef)
  early=step[li]<=2
  mn0,sd0,W0=awm.fit_metric(X[li][early],hf[early],ef[early])
  z=((X[li]-mn)/sd)@W;z0=((X[li]-mn0)/sd0)@W0
  s_c=float(np.median(awm._masked_min(tail[li],hf,ef,ef,rms_dim=35)))+1e-6
  fixed+=int(4*(3*W.size+4*W.shape[0]))
  for lo in range(0,len(qi),256):
   qr=qi[lo:lo+256]; rr=rows[qr]
   zq=((Xq[qr]-mn)/sd)@W;zq0=((Xq[qr]-mn0)/sd0)@W0
   d2=(zq*zq).sum(1)[:,None]+(z*z).sum(1)[None]-2*zq@z.T
   is0=Q.step[rr]==0
   d2[is0]=(zq0[is0]*zq0[is0]).sum(1)[:,None]+(z0*z0).sum(1)[None]-2*zq0[is0]@z0.T
   dt=np.sqrt(np.maximum(d2,0))
   if Q.arm=='inf':
    fresh=~is0
    prev=np.asarray(Q.a_exec[rr[fresh]-1,5:10,:7],np.float64).reshape(sum(fresh),35)/np.tile(sig,5)
    c=np.sqrt(np.maximum((prev*prev).sum(1)[:,None]+(hf*hf).sum(1)[None]-2*prev@hf.T,0)/35)
    dt[fresh]=dt[fresh]/(np.median(dt[fresh],axis=1)[:,None]+1e-12)+.5*c/s_c
   # stable row tie break
   ix=np.argsort(dt,axis=1,kind='stable')[:,:16];dk=np.take_along_axis(dt,ix,axis=1)
   w=awm._kernel_w(dk-dk[:,:1],kref); w/=w.sum(1)[:,None]
   ids[qr]=li[ix];weights[qr]=w
   pred[qr]=np.einsum('nk,nktd->ntd',w,np.asarray(L.action[li[ix],:5,:7],np.float64))
 vote=np.einsum('nk,nkt->nt',weights,np.where(L.action[ids,:5,6]>=0,1.,-1.))
 err=np.sqrt(np.mean(((pred-Q.a_inf[rows,:5,:7])/sig)**2,axis=(1,2)))
 grip=np.mean((pred[:,:,6]>=0)!=(Q.a_inf[rows,:5,6]>=0),axis=1)
 return {'ids':ids,'vote':vote,'err':err,'grip':grip,'pred':pred,'fixed_metric_bytes':fixed,'task':tq}

def event_masks(Q,rows):
 # Current policy's first executed head transition relative to previous ACTUALLY EXECUTED gripper,
 # including transitions within the first 5 teacher steps; metric-side labels, never online inputs.
 real=Q.a_inf[rows,:5,6]>=0
 prev=Q.a_exec[np.maximum(rows-1,0),4,6]>=0
 prev=np.where(Q.step[rows]>0,prev,real[:,0])
 before=np.column_stack([prev,real[:,:-1]])
 # Pi05 close=positive, GR00T close=negative (established in AWM3).
 closed=real if Q.model=='pi05' else ~real
 cb=before if Q.model=='pi05' else ~before
 return {'all':np.ones(len(rows),bool),'step0':Q.step[rows]==0,'grasp':np.any(~cb & closed,axis=1),'release':np.any(cb & ~closed,axis=1)}

def run(job):
 model,suite,libname=job;start=time.time();cell=f'{model}_{suite}_cache'
 ctx=api.Context(root=ROOT,cell=cell,seed=0,scratch=OUT)
 L=ctx.open_library(libname);sig=np.asarray(ctx.action_sigma,np.float64);kref=5 if libname=='current' else 8
 data=[]
 for arm in ['cache','inf']:
  Q=store.QueryCell(ROOT,f'{model}_{suite}_{arm}');rows=Q.tok_rows
  masks=event_masks(Q,rows);base=None
  for variant in ['full','cam0','cam1','pool2','pool1']:
   begin=time.time();X,Xq,fixed=features(L,Q,rows,variant);ev=evaluate(L,Q,rows,X,Xq,sig,kref)
   if base is None:base=ev
   overlap=np.array([len(set(a)&set(b))/16 for a,b in zip(ev['ids'],base['ids'])])
   rec={'model':model,'suite':suite,'library':libname,'episodes':len(np.unique(L.episode)),'entries':L.L,'arm':arm,'variant':variant,'query_rows':len(rows),'query_episodes':len(np.unique(Q.ep[rows])),'kref':kref,'fit_source':'same candidate library only','feature_dim':X.shape[1],'key_bytes':4*(X.shape[1]+8+1),'action_bytes':5*7*4,'fixed_bytes':fixed+ev['fixed_metric_bytes'],'compute_seconds':time.time()-begin,'metrics':{}}
   for name,mask in masks.items():
    n=int(mask.sum());rec['metrics'][name]={'n':n}
    if n:
     rec['metrics'][name].update(err=float(ev['err'][mask].mean()),err_median=float(np.median(ev['err'][mask])),delta_err_full=float((ev['err'][mask]-base['err'][mask]).mean()),overlap16=float(overlap[mask].mean()),top1_agree=float((ev['ids'][mask,0]==base['ids'][mask,0]).mean()),grip_mis=float(ev['grip'][mask].mean()),split08=float((np.abs(ev['vote'][mask,0])<.8).mean()),split05=float((np.abs(ev['vote'][mask,0])<.5).mean()),vote_sign_disagree=float(((ev['vote'][mask,0]>=0)!=(base['vote'][mask,0]>=0)).mean()))
   rec['per_task']={str(t):{'err':float(ev['err'][ev['task']==t].mean()),'split08':float((np.abs(ev['vote'][ev['task']==t,0])<.8).mean())} for t in L.tasks()}
   data.append(rec)
   print(model,suite,libname,arm,variant,round(rec['metrics']['all']['err'],4),flush=True)
 (OUT/f'retrieval_{model}_{suite}_{libname}.json').write_text(json.dumps(data,indent=2))
 return {'job':job,'seconds':time.time()-start}

if __name__=='__main__':
 jobs=[(m,s,l) for m in ['pi05','groot'] for s in ['spatial','l10'] for l in ['current','bpool_cs' if m=='pi05' else 'bpool_all']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
  for x in ex.map(run,jobs): print(x,flush=True)
