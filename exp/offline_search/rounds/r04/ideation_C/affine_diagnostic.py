"""Bounded local affine synthesis on saved AWM top-10 neighbours, fixed recorded observations.
This is a safety/mechanism diagnostic, never an estimator of closed-loop SR.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from exp.offline_search.harness.store import LibraryView,QueryCell
from exp.offline_search.rounds.r04.ideation_C.library_diagnostic import ROOT,OUT

def work(args):
    model,suite,scale=args
    lib=LibraryView(ROOT,f'{model}_{suite}','current' if scale==50 else ('bpool_cs' if model=='pi05' else 'bpool_all'))
    sig=np.std(LibraryView(ROOT,f'{model}_{suite}').action[:,:5,:7],axis=(0,1))
    name='AWM_joint_cur_fcur_kr5' if scale==50 else 'AWM_joint_big_fbig'
    kr=5 if scale==50 else 8
    rs=np.asarray(lib.rs[:,:3],float)
    sd=rs.std(0)+1e-6
    rr=rs/sd
    act=np.asarray(lib.action[:,:5,:7]/sig,dtype=float)
    result=[]
    for regime in ['inf','cache']:
        q=QueryCell(ROOT,f'{model}_{suite}_{regime}')
        f=np.load(Path('exp/offline_search/results/r02')/name/f'{model}_{suite}_{regime}.npz')
        ri=f['row']; top=f['topk']; ss=f['topk_scores']; target=np.asarray(q.a_inf[ri,:5,:7]/sig,float)
        truthstate=q.rs[ri,:3]/sd
        w=np.exp(-((ss-ss[:,:1])/np.maximum(ss[:,:1]-ss[:,kr-1:kr],1e-6))**2);w/=w.sum(1)[:,None]
        h=act[top]; states=rr[top]; mu=(w[:,:,None]*states).sum(1)
        xc=states-mu[:,None,:]; delta=truthstate-mu
        cov=np.einsum('nk,nka,nkb->nab',w,xc,xc)
        reg=.1*np.maximum(np.trace(cov,axis1=1,axis2=2)/3,1e-4)
        invdelta=np.linalg.solve(cov+reg[:,None,None]*np.eye(3),delta[:,:,None])[:,:,0]
        dw=w*np.einsum('nka,na->nk',xc,invdelta)
        raw=w+dw
        neg=np.maximum(-raw,0).sum(1)
        # Find largest interpolation coefficient with <= .10 total negative mass (convex in alpha).
        lo=np.zeros(len(w));hi=np.ones(len(w))
        for _ in range(24):
            mid=(lo+hi)/2
            good=np.maximum(-(w+mid[:,None]*dw),0).sum(1)<=.1
            lo=np.where(good,mid,lo);hi=np.where(good,hi,mid)
        wa=w+lo[:,None]*dw
        base=np.einsum('nk,nktd->ntd',w,h)
        alt=np.einsum('nk,nktd->ntd',wa,h)
        alt[:,:,6]=base[:,:,6]  # no gripper rewrite
        diff=alt-base
        magnitude=np.sqrt(np.mean(diff[:,:,:6]**2,axis=(1,2)))
        beta=np.minimum(1,.15/np.maximum(magnitude,1e-12))
        alt=base+beta[:,None,None]*diff
        # Necessary lower bound: any convex mixture stays inside each coordinate's min/max.
        outside=np.maximum(np.maximum(h.min(1)-target,target-h.max(1)),0)
        lb=np.sqrt(np.mean(outside**2,axis=(1,2)))
        eb=np.sqrt(np.mean((base-target)**2,axis=(1,2)))
        ea=np.sqrt(np.mean((alt-target)**2,axis=(1,2)))
        ec=np.sqrt(np.mean((alt[:,:,:6]-target[:,:,:6])**2,axis=(1,2)))
        bc=np.sqrt(np.mean((base[:,:,:6]-target[:,:,:6])**2,axis=(1,2)))
        gcap=(base[:,:,6]>=0)==(target[:,:,6]>=0)
        ca=np.mean(base[:,:,:3],axis=1); ta=np.mean(target[:,:,:3],axis=1); aa=np.mean(alt[:,:,:3],axis=1)
        cos=lambda a,b: np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-9)
        row_success=np.zeros(q.N,bool)
        for e in q.episodes: row_success[e['start']:e['end']]=e['success']
        for outcome in ['all','success','failure']:
            ok=(f['step']>0)
            if outcome!='all':ok &= row_success[ri]==(outcome=='success')
            result.append(dict(model=model,suite=suite,scale=scale,regime=regime,outcome=outcome,n=int(ok.sum()),
                base_err=float(eb[ok].mean()),affine_err=float(ea[ok].mean()),
                recon_rms=float(np.sqrt(np.mean((base-f['synth_seg']/sig)**2,axis=(1,2)))[ok].mean()),
                base_continuous_err=float(bc[ok].mean()),affine_continuous_err=float(ec[ok].mean()),
                useful_fraction=float((ea[ok]<eb[ok]).mean()),
                convex_lb_mean=float(lb[ok].mean()),convex_lb_gt02=float((lb[ok]>.2).mean()),
                outside_state_box=float(((truthstate<states.min(1))|(truthstate>states.max(1))).any(1)[ok].mean()),
                raw_negative_mass_mean=float(neg[ok].mean()),negative_mass_capped=float((neg[ok]>.1).mean()),
                mean_correction=float(np.sqrt(np.mean((alt-base)**2,axis=(1,2)))[ok].mean()),
                correction_cap_share=float((magnitude[ok]>.15).mean()),
                opposite_direction_before=float((cos(ca,ta)[ok]<0).mean()),
                opposite_direction_after=float((cos(aa,ta)[ok]<0).mean())))
    (OUT/f'affine_{model}_{suite}_{scale}.json').write_text(json.dumps(result,indent=2))
    return result[1::3]

if __name__=='__main__':
    jobs=[(m,s,n) for m in ['pi05','groot'] for s in ['spatial','l10'] for n in [50,500]]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for r in pool.map(work,jobs): print(json.dumps(r),flush=True)
