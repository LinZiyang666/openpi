"""Check a fixed 32-channel token witness on step 5/10 of recorded closed-loop token episodes."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import spearmanr,rankdata
from exp.offline_search.harness.store import LibraryView,QueryCell
from exp.offline_search.rounds.r04.ideation_C.library_diagnostic import ROOT,OUT

def auc(y,x):
 y=np.array(y,bool);return float((rankdata(x)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))
def project(a,R):
 a=np.asarray(a,np.float32);a/=np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12);return (a@R).astype(np.float16)
def work(job):
 m,s,n=job;q=QueryCell(ROOT,f'{m}_{s}_cache');lib=LibraryView(ROOT,f'{m}_{s}','current' if n==50 else ('bpool_cs' if m=='pi05' else 'bpool_all'))
 ri=np.load(q.dir/'tok/rows.npy');pos=np.flatnonzero(np.isin(q.step[ri],[5,10]));rows=ri[pos];method='AWM_joint_cur_fcur_kr5' if n==50 else 'AWM_joint_big_fbig'
 f=np.load(f'exp/offline_search/results/r02/{method}/{m}_{s}_cache.npz');inv=np.full(q.N,-1,int);inv[f['row']]=np.arange(len(f['row']));ix=inv[rows];kk=f['topk'][ix];unique,imap=np.unique(kk,return_inverse=True);imap=imap.reshape(kk.shape)
 R=np.random.default_rng(20260927).normal(0,1/np.sqrt(32),(2048,32)).astype(np.float32)
 qt=np.load(q.dir/'tok/v1.npy',mmap_mode='r');lt=np.load(lib.dir/'tok/v1.npy',mmap_mode='r');lri=np.load(lib.dir/'tok/rows.npy');linv=np.full(lib.L,-1,int);linv[lri]=np.arange(len(lri))
 Q=np.concatenate([project(qt[pos[a:a+8]],R) for a in range(0,len(pos),8)]);L=np.concatenate([project(lt[linv[unique[a:a+8]]],R) for a in range(0,len(unique),8)])
 dist=np.empty(kk.shape);exact=np.empty(len(rows))
 for i in range(len(rows)):
  d=.5*((L[imap[i]].astype(float)-Q[i].astype(float))**2).sum(-1);dist[i]=np.quantile(d,.95,axis=1)
  a=qt[pos[i]].astype(float);b=lt[linv[kk[i,0]]].astype(float);co=1-(a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-9);exact[i]=np.quantile(co,.95)
 ss=f['topk_scores'][ix];kr=5 if n==50 else 8;w=np.exp(-((ss-ss[:,:1])/np.maximum(ss[:,:1]-ss[:,kr-1:kr],1e-6))**2);w/=w.sum(1)[:,None]
 # Scale-free soft witness. No hard candidate exclusion or gripper rewrite.
 factor=np.exp(-dist/np.maximum(np.median(dist,axis=1,keepdims=True),1e-9));wa=w*factor;wa/=wa.sum(1)[:,None]
 sig=np.std(LibraryView(ROOT,f'{m}_{s}').action[:,:5,:7],axis=(0,1));heads=lib.action[kk,:5,:7]/sig
 base=np.einsum('nk,nktd->ntd',w,heads);alt=np.einsum('nk,nktd->ntd',wa,heads)
 metrics={}
 for step in [5,10]:
  mask=q.step[rows]==step;y=np.array([not q.episodes[e]['success'] for e in q.ep[rows[mask]]]);v=dist[mask,0]
  metrics[step]={'n':int(mask.sum()),'failures':int(y.sum()),'full_token_auc':auc(y,exact[mask]),'projected_auc':auc(y,v)}
 result={'model':m,'suite':s,'scale':n,'n':len(rows),'unique_library_rows':len(unique),'seed':20260927,'projection_dim':32,
 'projection_spearman':float(spearmanr(exact,dist[:,0]).statistic),'mean_relative_abs_dist_error':float(np.mean(np.abs(dist[:,0]-exact)/np.maximum(exact,1e-9))),
 'top_weight_changed':float((w.argmax(1)!=wa.argmax(1)).mean()),'served_head_change_rms':float(np.sqrt(((base-alt)**2).mean((1,2))).mean()),
 'metrics':metrics,'additional_entry_bytes':16384,'fixed_projection_bytes':R.nbytes}
 (OUT/f'p2_projection_{m}_{s}_{n}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(work,[(m,s,n) for m in ['pi05','groot'] for s in ['spatial','l10'] for n in [50,500]]))
