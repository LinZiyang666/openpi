"""Whole-chunk stochastic synthesis opportunity in the first actual AWM fixed-pick spell."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from exp.offline_search.harness.store import LibraryView
from exp.offline_search.rounds.r04.ideation_C.log_diagnostic import records,ROOT,OUT

def work(job):
 m,s,n=job;short='sp' if s=='spatial' else 'l10';arm=f'oscl{n}_{"p" if m=="pi05" else "g"}_{short}_cl2'
 lib=LibraryView(ROOT,f'{m}_{s}','current' if n==50 else ('bpool_cs' if m=='pi05' else 'bpool_all'))
 sig=np.std(LibraryView(ROOT,f'{m}_{s}').action[:,:5,:7],axis=(0,1));head=lib.action[:,:5,:7]/sig
 nx=np.asarray(lib.next);dv=np.zeros(lib.L);valid=nx>=0;dv[valid]=np.linalg.norm(lib.rs[nx[valid],:3]-lib.rs[valid,:3],axis=1)
 move_thr=float(np.median(dv[valid]));j,rec=records(f'r02_g{n}',arm);episodes={}
 for (u,step),d in sorted(rec.items()):episodes.setdefault(u,[]).append(d)
 out=[]
 for u,rr in episodes.items():
  # exactly one landmark per episode, the first spell; no long-failure weighting.
  for i in range(2,len(rr)):
   if rr[i]['top1']==rr[i-1]['top1']==rr[i-2]['top1']:break
  else:continue
  d=rr[i];kk=np.array(d['topk']);ss=np.array(d['scores']);kr=5 if n==50 else 8
  w=np.exp(-((ss-ss[0])/max(ss[0]-ss[kr-1],1e-6))**2);w/=w.sum();h=head[kk];mean=np.einsum('k,ktd->td',w,h)
  rms=np.sqrt(((h-mean)**2).mean((1,2)));diffgrip=((h[:,:,6]>=0)!=(mean[:,6]>=0)).any(1)
  norm=np.linalg.norm(h[:,:,:3].mean(1),axis=1);mean_norm=float(np.linalg.norm(mean[:,:3].mean(0)))
  continuous_rms=np.sqrt(((h[:,:,:6]-mean[:,:6])**2).mean((1,2)))
  substantial=(continuous_rms>.3)&(dv[kk]>move_thr)
  pg=float(w[diffgrip].sum());pm=float(w[substantial].sum())
  out.append({'uid':u,'task':rr[i]['task_id'],'init':rr[i]['init'],'success':bool(j[u]['success']),'spell_start':i-2,
     'conditional_grip_change_mass':pg,'conditional_active_alternative_mass':pm,'prob_change_10_draws_p01':float(1-(1-.1*pm)**10),
     'conditional_rms_mean':float(w@rms),'continuous_rms_mean':float(w@continuous_rms),
     'translation_retained':mean_norm/max(float(w@norm),1e-9),'effective_members':float(1/(w@w)),
     'mean_translation_norm':mean_norm,'member_translation_norm':float(w@norm),'top1_demo_motion':float(dv[d['top1']]),
     'no_terminal_mask_used':True})
 result={'model':m,'suite':s,'scale':n,'L':lib.L,'demo_motion_median':move_thr,'per_episode':out,'groups':{}}
 for ok in [False,True]:
  rr=[r for r in out if r['success']==ok];g={'n':len(rr)}
  for key in ['conditional_grip_change_mass','conditional_active_alternative_mass','prob_change_10_draws_p01','continuous_rms_mean','translation_retained','effective_members']:
   a=np.array([r[key] for r in rr]);g[key]={'mean':float(a.mean()),'median':float(np.median(a))}
  g['active_mass_ge_01']=sum(r['conditional_active_alternative_mass']>=.1 for r in rr)
  g['grip_mass_ge_01']=sum(r['conditional_grip_change_mass']>=.1 for r in rr)
  g['translation_retained_lt_half']=sum(r['translation_retained']<.5 for r in rr)
  result['groups'][str(ok)]=g
 (OUT/f'p2_kernel_{m}_{s}_{n}.json').write_text(json.dumps(result,indent=2));print(m,s,n,result['groups'],flush=True)
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(work,[(m,s,n) for m in ['pi05','groot'] for s in ['spatial','l10'] for n in [50,500]]))
