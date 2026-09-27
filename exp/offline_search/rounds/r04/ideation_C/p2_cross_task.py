"""Cross-task/suite candidate opportunity and aliasing; teacher errors are diagnostic, never SR estimates."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from exp.offline_search.harness.store import LibraryView,QueryCell
from exp.offline_search.rounds.r04.ideation_C.library_diagnostic import ROOT,OUT,load_library,dsq

def project(lib,pc):
 return np.c_[*[(np.asarray(getattr(lib,'key_'+f),np.float32)-np.load(pc/f/'mean.npy'))@np.load(pc/f/'basis.npy')[:,:64] for f in ['v0','v1']],lib.rs[:,:8]]

def work(job):
 m,n=job;lib,x=load_library(m,'l10',n);donor=LibraryView(ROOT,m+'_spatial','current' if n==50 else ('bpool_cs' if m=='pi05' else 'bpool_all'))
 pc=ROOT/'derived'/('r02/g1_awm/pca_current' if n==50 else 'r01/f4_vision/pca')/(m+'_l10')
 if n==500:pc=pc/lib.name
 xd=project(donor,pc)
 data=np.load(OUT/f'codes_{m}_l10_{n}.npz');sig=np.std(LibraryView(ROOT,m+'_l10').action[:,:5,:7],axis=(0,1))
 h=(lib.action[:,:5,:7]/sig).reshape(lib.L,-1).astype(float)
 dh=(donor.action[:,:5,:7]/sig).reshape(donor.L,-1).astype(float)
 families={0:[7],1:[7],7:[0,1],2:[8],8:[2],4:[6],6:[4]}
 out=[]
 for regime in ['cache','inf']:
  q=QueryCell(ROOT,m+'_l10_'+regime)
  # Every 5th decision across all 500 accepted recorded episodes, fixed observations.
  ri=np.flatnonzero(q.step%5==0);xx=np.empty((len(ri),136),np.float32)
  for j,f in enumerate(['v0','v1']):
   B=np.load(pc/f/'basis.npy')[:,:64];mu=np.load(pc/f/'mean.npy')
   for a in range(0,len(ri),256):xx[a:a+256,j*64:(j+1)*64]=(getattr(q,'key_'+f)[ri[a:a+256]]-mu)@B
  xx[:,128:]=q.rs[ri,:8];task=np.array([q.episodes[e]['task_id'] for e in q.ep[ri]])
  truth=(q.a_inf[ri,:5,:7]/sig).reshape(len(ri),35)
  success=np.array([q.episodes[e]['success'] for e in q.ep[ri]])
  for t in range(10):
   ii=np.flatnonzero(task==t);same=np.asarray(lib.rows_of_task(t));other=np.flatnonzero(lib.task_id!=t)
   mn=data[f't{t}_mean'];sd=data[f't{t}_std'];w=data[f't{t}_w']
   z=((x-mn)/sd)@w;zd=((xd-mn)/sd)@w;zq=((xx[ii]-mn)/sd)@w
   all_dist=dsq(zq,z);dd=dsq(zq,zd)
   ah=dsq(truth[ii],h)/35;adh=dsq(truth[ii],dh)/35
   same_best=same[all_dist[:,same].argmin(1)];or_same=np.sqrt(ah[:,same].min(1))
   base_error=np.sqrt(ah[np.arange(len(ii)),same_best])
   for kind,cands in [('other_task',other),('shared_object',np.flatnonzero(np.isin(lib.task_id,families.get(t,[])))),('spatial',np.arange(donor.L))]:
    if not len(cands):continue
    dist=dd if kind=='spatial' else all_dist;err=adh if kind=='spatial' else ah;hh=dh if kind=='spatial' else h
    sel=cands[dist[:,cands].argmin(1)];selected=dist[np.arange(len(ii)),sel]<all_dist[np.arange(len(ii)),same_best]
    e=np.sqrt(err[np.arange(len(ii)),sel]);oracle=np.sqrt(err[:,cands].min(1));g=hh[sel].reshape(-1,5,7)[:,:,6]>=0;tg=truth[ii].reshape(-1,5,7)[:,:,6]>=0
    out.append({'regime':regime,'task':t,'kind':kind,'n':len(ii),'selected_n':int(selected.sum()),
      'baseline_nn_error':float(base_error.mean()),'donor_nn_error':float(e.mean()),
      'selected_donor_nn_error':float(e[selected].mean()) if selected.any() else None,
      'selected_original_nn_error':float(base_error[selected].mean()) if selected.any() else None,
      'selected_grip_mismatch':float((g[selected]!=tg[selected]).mean()) if selected.any() else None,
      'own_oracle_error':float(or_same.mean()),'donor_oracle_error':float(oracle.mean()),
      'union_oracle_error':float(np.minimum(or_same,oracle).mean()),
      'donor_oracle_better_share':float((oracle<or_same).mean()),
      'selected_failed_share':float((~success[ii][selected]).mean()) if selected.any() else None})
 result={'model':m,'scale':n,'L_l10':lib.L,'L_spatial':donor.L,'rows':out}
 (OUT/f'p2_cross_task_{m}_{n}.json').write_text(json.dumps(result,indent=2))
 print(m,n,'done',flush=True)
 return result
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(work,[(m,n) for m in ['pi05','groot'] for n in [50,500]]))
