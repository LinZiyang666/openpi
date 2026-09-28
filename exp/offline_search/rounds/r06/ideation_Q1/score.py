"""CPU-only frozen-deployment LOEO and held-trajectory quality audit. No rollout simulation."""
import os,sys,json,pickle,pathlib,importlib,hashlib,types,argparse,time
import numpy as np
from scipy.spatial.distance import cdist
ROOT=pathlib.Path('/home/weiland/projects/openpi');sys.path.insert(0,str(ROOT))
from exp.offline_search.rounds.r02.g1_awm.awm import AWM,fit_metric,pca_fit,_Task
from exp.offline_search.harness import api
S=pathlib.Path('/home/weiland/trace_runs/offline_search_store');C=pathlib.Path('/home/weiland/trace_runs/os_closed_loop')
O=ROOT/'exp/offline_search/rounds/r06/ideation_Q1';TMP=pathlib.Path('/tmp/r6_Q1')
class U(pickle.Unpickler):
 def find_class(self,m,n):
  if m.startswith('osm_'):
   for p in ['exp.offline_search.rounds.r05.q4_growth.demo_method','exp.offline_search.rounds.r05.q4_growth.method','exp.offline_search.rounds.r02.g1_awm.awm','exp.offline_search.rounds.r04.k1_blind.blind_awm']:
    o=importlib.import_module(p)
    if hasattr(o,n):return getattr(o,n)
  return super().find_class(m,n)
def ar(p,k):return np.load(p/(k+'.npy'),mmap_mode='r')
def joblist():
 jobs=[]
 for cell in ['pi05_l10','pi05_spatial','groot_l10','groot_spatial']:
  m,s=cell.split('_');short=('p' if m=='pi05' else 'g')+'_'+('sp' if s=='spatial' else s)
  for lib in sorted((S/'library'/cell).iterdir()):
   if not (lib/'episode.npy').exists():continue
   name=lib.name
   if name in ('current','bpool_cs') or (m=='groot' and name=='bpool_all'):
    size=50 if name=='current' else 500;run='r02_g'+str(size);arm=f'oscl{size}_{short}_cl2';variant='refit'
   elif name.startswith('demo'):
    run='r05_demo_curve';arm=f'r5q4d_{short}_{name[4:]}_refit_tail1';variant='refit'
   elif name=='grow250':run='r05_growth';arm=f'r5q4_{short}_grow250_refit';variant='refit'
   else:run=arm=None;variant='refit'
   jobs.append(dict(cell=cell,library=name,variant=variant,fit=str(C/run/'fits'/(arm+'.pkl')) if run else None))
   if name in ('demo200','grow250'):
    arm=arm.replace('refit','frozen50' if name=='demo200' else 'frozen');jobs.append(dict(cell=cell,library=name,variant='frozen',fit=str(C/run/'fits'/(arm+'.pkl'))))
 return jobs

def loadmethod(job):
 if job['fit']:return U(open(job['fit'],'rb')).load()['method']
 # Nondeployed pi05 bpool_all: same AWM recipe, own-library units. No SR exists for this fit.
 p=S/'library'/job['cell']/job['library'];m=AWM(lib='current',kref=5);m.model=job['cell'].split('_')[0];m.prof=api.NULL_PROFILER
 m.act=np.array(ar(p,'action'));m.sig=np.asarray(m.act[:,:5,:7].reshape(-1,7).std(0));m.lib_ep=np.array(ar(p,'episode'));m.lib_step=np.array(ar(p,'step'));m.H=m.act.shape[1];m.tasks={};P=[]
 for i in range(2):
  mu,B,proj=pca_fit(ar(p,'key_v'+str(i)));setattr(m,'B'+str(i)+'T',B.T.copy());setattr(m,'muB'+str(i),mu@B);P.append(proj)
 X=np.concatenate(P+[ar(p,'rs')[:,:8]],axis=1).astype(float);H=(m.act[:,:5,:7]/m.sig).reshape(-1,35);task=ar(p,'task_id')
 for t in np.unique(task):
  T=_Task();r=np.flatnonzero(task==t);T.rows=r;mu,sd,W=fit_metric(X[r],H[r],m.lib_ep[r]);T.Wf=(W/sd[:,None]).astype('f');T.shift=((mu/sd)@W).astype('f');T.Z=(((X[r]-mu)/sd)@W).astype('f');T.z2=np.sum(T.Z.astype(float)**2,1).astype('f')
  early=m.lib_step[r]<=2;mu0,sd0,W0=fit_metric(X[r][early],H[r][early],m.lib_ep[r][early]);T.W0f=(W0/sd0[:,None]).astype('f');T.A0=(np.linalg.inv(W)@np.diag(sd/sd0)@W0).astype('f');b=((mu-mu0)/sd0)@W0;T.c0=((mu0/sd0)@W0+b).astype('f');Y=T.Z.astype(float)@T.A0;T.n20=np.sum(Y*Y,1).astype('f');T.Z0=None;T.As0=None;m.tasks[int(t)]=T
 return m

def features(m,p,tag):
 # Different fits can share PCA; hash protects projection cache identity.
 h=hashlib.sha256(m.B0T.tobytes()+m.B1T.tobytes()+m.muB0.tobytes()+m.muB1.tobytes()).hexdigest()[:16];f=TMP/(tag+'_'+h+'.npy')
 if f.exists():return np.load(f,mmap_mode='r')
 n=len(ar(p,'rs'));out=np.empty((n,136),'f');out[:,128:]=ar(p,'rs')[:,:8]
 for cam in range(2):
  key=ar(p,'key_v'+str(cam));B=getattr(m,'B'+str(cam)+'T');mb=getattr(m,'muB'+str(cam))
  for lo in range(0,n,256):out[lo:lo+256,cam*64:(cam+1)*64]=key[lo:lo+256]@B.T-mb
 np.save(f,out);return out

def dataset(p,library):
 eps=json.loads((p/'episodes.json').read_text());ep=np.array(ar(p,'episode' if library else 'ep'));step=np.array(ar(p,'step'));n=len(ep)
 task=np.empty(n,int);success=np.empty(n,bool);init=np.full(n,-1,int);progress=np.empty(n,float)
 for e in eps:
  r=np.arange(e['start'],e['end']);task[r]=e['task_id'];success[r]=e['success'];init[r]=e.get('init',e.get('orig_init_state_idx',-1)) if e.get('init',e.get('orig_init_state_idx',-1)) is not None else -1;progress[r]=np.arange(len(r))/max(len(r)-1,1)
 nxt=np.arange(n)+1;nxt[-1]=-1;nxt[:-1][ep[:-1]!=ep[1:]]=-1
 n2=np.full(n,-1,int);v=nxt>=0;n2[v]=nxt[nxt[v]]
 rs=np.asarray(ar(p,'rs')[:,:8],float);act=np.asarray(ar(p,'action' if library else 'a_inf')[:,:10,:7],float)
 exe=act if library else np.asarray(ar(p,'a_exec')[:,:10,:7],float)
 exec10=exe.copy();v=nxt>=0;exec10[v,5:10]=exe[nxt[v],:5]
 return dict(ep=ep,step=step,task=task,success=success,init=init,progress=progress,next=nxt,next2=n2,rs=rs,act=act,exec10=exec10,episodes=eps)

def dist(m,T,X,step):
 Z=X@T.Wf-T.shift;d2=T.z2[None,:]-2*(Z@T.Z.T)+np.sum(Z*Z,1)[:,None]
 early=step==0
 if early.any():
  Y=X[early]@T.W0f-T.c0
  cross=(Y@T.A0.T)@T.Z.T if T.Z0 is None else Y@T.Z0.T
  d2[early]=T.n20[None,:]-2*cross+np.sum(Y*Y,1)[:,None]
 return np.sqrt(np.maximum(d2,0))

def macro(v,ep):
 return float(np.nanmean([np.nanmean(v[ep==e]) for e in np.unique(ep) if np.isfinite(v[ep==e]).any()])) if np.isfinite(v).any() else None

def run(job):
 t0=time.time();tag='_'.join([job['cell'],job['library'],job['variant']]);print('START',tag,flush=True)
 p=S/'library'/job['cell']/job['library'];m=loadmethod(job);D=dataset(p,True);X=features(m,p,tag+'_lib');n=len(X)
 assert np.array_equal(m.act[:,:,:7],ar(p,'action')[:,:,:7]);sig=np.asarray(D['act'][:,:5].reshape(-1,7).std(0));sig=np.where(sig>1e-12,sig,1)
 scales={};cal={};res={};checks=[]
 for source in ['loeo','inf','cache']:
  qp=p if source=='loeo' else S/'queries'/(job['cell']+'_'+source);Q=D if source=='loeo' else dataset(qp,False);QX=X if source=='loeo' else features(m,qp,job['cell']+'_'+source)
  N=len(QX);out={k:np.full(N,np.nan,'f') for k in ['d1','d_pair','d_self','covered','err5','err10','exec10','succ1','succ2','edge1','edge2','phase','disp','q','q_online','pred_err','pred_succ','neff']};out.update({k:Q[k] for k in ['ep','step','task','success','init','progress']})
  for t,T in m.tasks.items():
   rr=T.rows;qr=np.flatnonzero(Q['task']==t)
   if source!='loeo' and job['library']=='grow250':qr=qr[Q['init'][qr]>=25]
   ss=D['rs'][rr].std(0);ss=np.where(ss>1e-12,ss,1)
   deltas={}
   for lag in [1,2]:
    nx=D['next' if lag==1 else 'next2'];dl=np.zeros_like(D['rs']);v=nx>=0;dl[v]=(D['rs'][nx[v]]-D['rs'][v])/ss;deltas[lag]=dl
   if source=='loeo':
    # Cross-episode pair scale, bounded deterministic sample, includes both endpoints.
    take=np.unique(np.linspace(0,len(rr)-1,min(512,len(rr)),dtype=int));dd=cdist(T.Z[take].astype(float),T.Z[take].astype(float));mask=D['ep'][rr[take],None]!=D['ep'][rr[take]][None,:];pair=float(np.median(dd[mask]));scales[t]={'pair':pair,'state_sd':ss.tolist()}
   else:pair=scales[t]['pair']
   for lo in range(0,len(qr),128):
    qq=qr[lo:lo+128];distance=dist(m,T,QX[qq],Q['step'][qq]);median=np.median(distance,axis=1)
    if source=='loeo':distance[Q['ep'][qq,None]==D['ep'][rr][None,:]]=np.inf
    k=min(m.k,int(np.min(np.isfinite(distance).sum(1))));ix=np.argpartition(distance,k-1,axis=1)[:,:k];dk=np.take_along_axis(distance,ix,1);order=np.argsort(dk,axis=1,kind='stable');ix=np.take_along_axis(ix,order,1);dk=np.take_along_axis(dk,order,1);rows=rr[ix]
    w=np.exp(-((dk-dk[:,:1])/np.maximum(dk[:,min(m.kref,k)-1,None]-dk[:,:1],1e-6))**2);w/=w.sum(1)[:,None];pred=np.einsum('qk,qkhd->qhd',w.astype('f'),D['act'][rows]);out['neff'][qq]=1/np.sum(w*w,1)
    out['d1'][qq]=dk[:,0];out['d_pair'][qq]=dk[:,0]/max(pair,1e-12)
    for key,h,target in [('err5',5,Q['act']),('err10',10,Q['act']),('exec10',10,Q['exec10'])]:out[key][qq]=np.sqrt(np.mean(((pred[:,:h]-target[qq,:h])/sig)**2,axis=(1,2)))
    out['exec10'][qq[Q['next'][qq]<0]]=np.nan
    for lag in [1,2]:
     nxt=Q['next' if lag==1 else 'next2'];valid_edge=D['next' if lag==1 else 'next2'][rows]>=0;ew=w*valid_edge;mass=ew.sum(1);out['edge'+str(lag)][qq]=mass;ew/=np.maximum(mass[:,None],1e-12);v=(nxt[qq]>=0)&(mass>0);actual=(Q['rs'][nxt[qq[v]]]-Q['rs'][qq[v]])/ss;pd=np.einsum('qk,qkd->qd',ew,deltas[lag][rows]);out['succ'+str(lag)][qq[v]]=np.sqrt(np.mean((actual-pd[v])**2,1))
    out['phase'][qq]=np.abs(np.sum(w*D['progress'][rows],1)-Q['progress'][qq]);out['disp'][qq]=np.sqrt(np.mean(((D['act'][rows[:,:5],:5]-pred[:,None,:5])/sig)**2,axis=(1,2,3)))
    if source!='loeo':
     out['d_self'][qq]=dk[:,0]/max(cal[t]['d50'],1e-12);out['covered'][qq]=dk[:,0]<=cal[t]['d95'];ae=res['loeo']['err10'][rows];se=res['loeo']['succ2'][rows];se=np.where(np.isfinite(se),se,cal[t]['succ2_mean']);out['pred_err'][qq]=np.sum(w*ae,1);out['pred_succ'][qq]=np.sum(w*se,1);out['q_online'][qq]=1/(1+out['d_pair'][qq]+out['pred_err'][qq]+out['pred_succ'][qq])
   if source=='loeo':
    cal[t]={k:float(np.nanmean(out[k][qr])) for k in ['err10','succ2']};cal[t]['succ2_mean']=cal[t]['succ2'];cal[t].update(d50=float(np.nanmedian(out['d1'][qr])),d95=float(np.nanquantile(out['d1'][qr],.95)))
    out['d_self'][qr]=out['d1'][qr]/max(cal[t]['d50'],1e-12);out['covered'][qr]=out['d1'][qr]<=cal[t]['d95']
  out['q']=1/(1+out['d_pair']+out['err10']+out['succ2'])
  res[source]=out;np.savez_compressed(O/(tag+'_'+source+'.npz'),**out)
  print(tag,source,'rows',int(np.isfinite(out['d1']).sum()),'seconds',round(time.time()-t0,1),flush=True)
  # Validate batched feature/distance/synthesis against deployed scalar AWM on 20 recorded states.
  if source=='inf' and job['fit']:
   eligible=np.flatnonzero(np.isfinite(out['d1']));sample=eligible[np.linspace(0,len(eligible)-1,min(20,len(eligible)),dtype=int)]
   for r in sample:
    q=types.SimpleNamespace(key_v0=ar(qp,'key_v0')[r],key_v1=ar(qp,'key_v1')[r],rs=ar(qp,'rs')[r],step=int(Q['step'][r]),prev_hit=True,task_id=int(Q['task'][r]))
    dd=m._dist(q)[-1];T=m.tasks[q.task_id];ix=np.argpartition(dd,15)[:16];ix=ix[np.lexsort((ix,dd[ix]))];dk=dd[ix].astype(float);w=np.exp(-((dk-dk[0])/max(dk[m.kref-1]-dk[0],1e-6))**2);pred=m._mix(T.rows[ix],w)[2][:10,:7];err=np.sqrt(np.mean(((pred-Q['act'][r])/sig)**2));checks.append({'row':int(r),'d_abs':float(abs(dd.min()-out['d1'][r])),'err10_abs':float(abs(err-out['err10'][r]))})
 summary={'job':job,'rows':n,'episodes':len(np.unique(D['ep'])),'successful_episodes':sum(bool(e['success']) for e in D['episodes']),'kref':m.kref,'action_sigma_own':sig.tolist(),'scales':scales,'calibration':cal,'parity':checks,'metrics':{},'seconds':time.time()-t0}
 for source,out in res.items():
  mt={};keys=[k for k in out if k not in ['ep','step','task','success','init','progress']]
  for t in m.tasks:
   sel=(out['task']==t)&np.isfinite(out['d1']);ep=out['ep'][sel];mt[str(t)]={k:macro(out[k][sel],ep) for k in keys};mt[str(t)]['n']=int(sel.sum());mt[str(t)]['episodes']=len(np.unique(ep));mt[str(t)]['success_rate']=macro(out['success'][sel],ep)
   for b in range(5):
    v=sel&(np.minimum((out['progress']*5).astype(int),4)==b);mt[str(t)]['phase'+str(b)]={k:macro(out[k][v],out['ep'][v]) for k in ['covered','err10','succ2','d_pair']}
   br=[];runs=[]
   for e in np.unique(ep):
    v=np.flatnonzero(sel&(out['ep']==e));bad=out['covered'][v]==0
    if bad.any():br.append(float(out['progress'][v[np.flatnonzero(bad)[0]]]))
    best=cur=0
    for z in bad:cur=cur+1 if z else 0;best=max(best,cur)
    runs.append(best/max(len(v),1))
   mt[str(t)]['episodes_any_break']=len(br)/max(len(np.unique(ep)),1);mt[str(t)]['first_break_progress']=float(np.mean(br)) if br else None;mt[str(t)]['longest_bad_fraction']=float(np.mean(runs))
  summary['metrics'][source]={'tasks':mt,'global':{k:float(np.mean([v[k] for v in mt.values() if v.get(k) is not None])) for k in keys+['episodes_any_break','longest_bad_fraction'] if any(v.get(k) is not None for v in mt.values())}}
 (O/(tag+'.json')).write_text(json.dumps(summary,indent=2,allow_nan=False));print('DONE',tag,round(time.time()-t0,1),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--job',type=int);a.add_argument('--list',action='store_true');args=a.parse_args();jobs=joblist()
 if args.list:print(json.dumps(jobs,indent=2))
 else:run(jobs[args.job])
