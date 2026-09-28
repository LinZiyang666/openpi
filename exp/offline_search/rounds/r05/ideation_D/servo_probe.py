"""Physics-based pose rebasing, diagnostic only. Top-10 pose, exact saved AWM action.

Uses no query labels for fitting. Forward map is fitted in the deployed library.
This is an action-direction plausibility test, NOT a closed-loop SR predictor.
"""
import json
import numpy as np
from diagnose_logs import HERE, STORE
from dynamics_probe import arr

def main():
    reports=[]
    for model in ['pi05','groot']:
      for suite in ['spatial','l10']:
       cur=STORE/'library'/f'{model}_{suite}'/'current'
       sigma=np.std(arr(cur,'action')[:,:5,:7],axis=(0,1)).astype(float)
       for size in [50,500]:
        name='current' if size==50 else ('bpool_cs' if model=='pi05' else 'bpool_all')
        lib=STORE/'library'/f'{model}_{suite}'/name
        state=arr(lib,'rs')[:,:3]; beta=np.load(HERE/f'dynamics_{model}_{suite}_{size}.npz')['beta']
        inv=np.linalg.pinv(beta[1:4])
        method='AWM_joint_cur_fcur_kr5' if size==50 else 'AWM_joint_big_fbig'
        kr=5 if size==50 else 8
        for regime in ['inf','cache']:
          q=STORE/'queries'/f'{model}_{suite}_{regime}'
          f=np.load(f'exp/offline_search/results/r02/{method}/{model}_{suite}_{regime}.npz')
          ix=f['row']; ss=f['topk_scores']; rows=f['topk']; mask=f['step']>0
          w=np.exp(-((ss-ss[:,:1])/np.maximum(ss[:,:1]-ss[:,kr-1:kr],1e-6))**2); w/=w.sum(1)[:,None]
          pose=(w[:,:,None]*state[rows]).sum(1)
          offset=pose-arr(q,'rs')[ix,:3]
          delta=.25*offset@inv
          mag=np.sqrt(np.mean((delta/sigma[:3])**2,axis=1))
          delta*=np.minimum(1,.15/np.maximum(mag,1e-12))[:,None]
          base=f['synth_seg'].astype(float); teacher=arr(q,'a_inf')[ix,:5,:7].astype(float)
          alt=base.copy();alt[:,:,:3]+=delta[:,None,:]
          rbase=np.mean((base-teacher)/sigma,axis=1)[:,:3]
          need=-rbase; dc=delta/sigma[:3]
          cos=(need*dc).sum(1)/np.maximum(np.linalg.norm(need,axis=1)*np.linalg.norm(dc,axis=1),1e-9)
          be=np.sqrt(np.mean(((base-teacher)/sigma)**2,axis=(1,2)))
          ae=np.sqrt(np.mean(((alt-teacher)/sigma)**2,axis=(1,2)))
          rep=dict(model=model,suite=suite,size=size,regime=regime,n=int(mask.sum()),
                   baseline_error=float(be[mask].mean()),servo_error=float(ae[mask].mean()),
                   residual_alignment_mean=float(cos[mask].mean()),alignment_positive=float((cos[mask]>0).mean()),
                   correction_capped=float((mag[mask]>.15).mean()),
                   improved_fraction=float((ae[mask]<be[mask]).mean()),
                   positive_alignment_by_task={str(t):float((cos[mask&(f['task_id']==t)]>0).mean()) for t in range(10)})
          reports.append(rep);print(json.dumps(rep),flush=True)
    (HERE/'servo_probe.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__': main()
