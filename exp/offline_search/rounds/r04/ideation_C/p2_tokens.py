"""Token mismatch/innovation on actual B0 closed-loop token subsample; AWM candidates replayed, no outcome fit."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import rankdata
from exp.offline_search.harness.store import LibraryView,QueryCell
from exp.offline_search.rounds.r04.ideation_C.library_diagnostic import ROOT,OUT

def auc(y,x):
 y=np.asarray(y,bool);x=np.asarray(x);good=np.isfinite(x);y=y[good];x=x[good]
 if not y.any() or y.all():return None
 return float((rankdata(x)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))
def compare(a,b):
 aa=np.sum(a*a,axis=-1);bb=np.sum(b*b,axis=-1)
 return 1-np.sum(a*b,axis=-1)/np.maximum(np.sqrt(aa*bb),1e-9)
def pooled(a):return a.reshape(-1,4,4,4,4,2048).mean(axis=(2,4)).reshape(-1,16,2048)
def work(job):
 m,s=job;q=QueryCell(ROOT,f'{m}_{s}_cache');ri=np.load(q.dir/'tok/rows.npy');N=len(ri)
 out={'model':m,'suite':s,'n_tokens':N,'episodes':len(np.unique(q.ep[ri])),'scales':{}}
 temporal=np.zeros((N,4));qt=[np.load(q.dir/f'tok/v{c}.npy',mmap_mode='r') for c in [0,1]]
 for c in [0,1]:
  for lo in range(1,N,16):
   inds=np.arange(lo,min(lo+16,N));a=qt[c][inds].astype(np.float32);b=qt[c][inds-1].astype(np.float32)
   dc=compare(a,b);dp=compare(pooled(a),pooled(b))
   temporal[inds,c*2]=np.quantile(dc,.95,axis=1);temporal[inds,c*2+1]=dp.mean(1)
 for n in [50,500]:
  lib=LibraryView(ROOT,f'{m}_{s}','current' if n==50 else ('bpool_cs' if m=='pi05' else 'bpool_all'))
  method='AWM_joint_cur_fcur_kr5' if n==50 else 'AWM_joint_big_fbig'
  results=np.load(f'exp/offline_search/results/r02/{method}/{m}_{s}_cache.npz');lookup=np.full(q.N,-1,int);lookup[results['row']]=np.arange(len(results['row']))
  assert (lookup[ri]>=0).all();ii=lookup[ri];top=results['topk'][ii,0]
  features={'awm_distance':-results['topk_scores'][ii,0],'awm_disp5':results['x_disp5'][ii]}
  tokrows=np.load(lib.dir/'tok/rows.npy');inv=np.full(lib.L,-1,int);inv[tokrows]=np.arange(len(tokrows));assert (inv[top]>=0).all()
  for c in [0,1]:
   lt=np.load(lib.dir/f'tok/v{c}.npy',mmap_mode='r');v=np.empty((N,3))
   for lo in range(0,N,16):
    a=qt[c][lo:lo+16].astype(np.float32);b=lt[inv[top[lo:lo+16]]].astype(np.float32);d=compare(a,b);p=compare(pooled(a),pooled(b))
    v[lo:lo+16,0]=d.mean(1);v[lo:lo+16,1]=np.quantile(d,.95,axis=1);v[lo:lo+16,2]=p.mean(1)
   for j,label in enumerate(['patch_mean','patch_p95','pooled_mean']):features[f'cam{c}_{label}']=v[:,j]
   features[f'cam{c}_motion_patch_p95']=temporal[:,2*c];features[f'cam{c}_motion_pool']=temporal[:,2*c+1]
  features['local_excess']=features['cam1_patch_p95']/(features['cam1_pooled_mean']+1e-5)
  ep=[]
  for e in np.unique(q.ep[ri]):
   jj=np.flatnonzero(q.ep[ri]==e);meta=q.episodes[e];picks=q.rec_top1[meta['start']:meta['end']];first=len(picks)
   for t in range(len(picks)-2):
    if picks[t]==picks[t+1]==picks[t+2]:first=t;break
   record={'task':meta['task_id'],'init':meta['init'],'success':bool(meta['success']),'first_spell':first,'n':len(jj),'features':{}}
   st=q.step[ri[jj]]
   for label,xx in features.items():
    vals=xx[jj];early=(st>=2)&(st<=min(20,first+1))
    record['features'][label]={'step5':float(vals[st==5][0]) if np.any(st==5) else None,'step10':float(vals[st==10][0]) if np.any(st==10) else None,'early_max':float(vals[early].max()) if early.any() else None}
   ep.append(record)
  metrics={}
  for label in features:
   metrics[label]={}
   for window in ['step5','step10','early_max']:
    rows=[e for e in ep if e['features'][label][window] is not None];y=np.array([not e['success'] for e in rows]);v=np.array([e['features'][label][window] for e in rows]);tasks=np.array([e['task'] for e in rows])
    # Within-task pair AUC removes between-task difficulty without fitting predictions.
    pair=[]
    for t in range(10):
     f=v[(tasks==t)&y];suc=v[(tasks==t)&~y]
     if len(f) and len(suc):pair.extend(((f[:,None]>suc)+.5*(f[:,None]==suc)).ravel().tolist())
    metrics[label][window]={'auc':auc(y,v),'within_task_auc':float(np.mean(pair)) if pair else None,'n':len(rows),'failure':int(y.sum()),'within_task_pairs':len(pair)}
  out['scales'][n]={'metrics':metrics,'per_episode':ep}
  np.savez_compressed(OUT/f'p2_token_rows_{m}_{s}_{n}.npz',row=ri,**features)
 (OUT/f'p2_tokens_{m}_{s}.json').write_text(json.dumps(out,indent=2));print(m,s,'done',flush=True)
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(work,[(m,s) for m in ['pi05','groot'] for s in ['spatial','l10']]))
