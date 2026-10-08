"""CPU light-key prototypes. Fits only train.npz of the requested cell.

Evaluation is controlled layer-1 retrieval: same early/stale regime, 16-NN
kernel and same library action payload for every method. No online correction.
"""
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric, _kernel_w
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,load_cell,dump

def grids(images):
    n=len(images)
    return images.astype(np.float32).reshape(n,14,4,14,4,3).mean((2,4)).reshape(n,-1)/255.

def pca_fit_small(x,k=64):
    mu=x.mean(0);_,_,vt=np.linalg.svd(x-mu,full_matrices=False)
    return mu,vt[:k].T

def task_metric(X,tr):
    out={}
    for t in range(10):
        r=np.flatnonzero(tr['task_id']==t)
        params=[]
        for early in (False,True):
            rr=r[tr['step'][r]<=2] if early else r
            mn,sd,w=fit_metric(X[rr].astype(np.float64),(tr['action'][rr,:5,:7]/tr['sig']).reshape(len(rr),-1).astype(np.float64),tr['episode'][rr])
            params.append({'wf':(w/sd[:,None]).astype(np.float32),'shift':((mn/sd)@w).astype(np.float32)})
        out[t]=params
    return out

def teacher_metric(base):
    out={}
    for t,T in base.tasks.items():
        # Early affine library codes use the same offset as runtime. Distances
        # invariant to common translation, so reconstruct the early shift.
        out[t]=[{'wf':T.Wf,'shift':T.shift}, {'wf':T.W0f,'shift':T.c0}]
    return out

def reference_codes(base,tr):
    z={}
    for t,T in base.tasks.items():
        z[t]=[T.Z,T.Z@T.A0]
    return z

def predict_ridge(model,x,task):
    out=np.empty((len(x),128),np.float32)
    for t,p in model.items():
        r=np.flatnonzero(task==t)
        out[r]=((x[r]-p['xm'])/p['xs'])@p['coef']+p['ym']
    return out

def fit_ridge(x,y,task,alpha):
    model={}
    for t in range(10):
        r=np.flatnonzero(task==t);a=x[r];b=y[r]
        xm=a.mean(0);xs=np.maximum(a.std(0),.03);a=((a-xm)/xs).astype(np.float64)
        ym=b.mean(0);yc=(b-ym).astype(np.float64)
        lam=alpha*len(r)
        if len(a)<a.shape[1]:
            coef=a.T@np.linalg.solve(a@a.T+lam*np.eye(len(a)),yc)
        else:coef=np.linalg.solve(a.T@a+lam*np.eye(a.shape[1]),a.T@yc)
        model[t]={'xm':xm,'xs':xs,'coef':coef.astype(np.float32),'ym':ym}
    return model

def evaluate(name,libx,qx,metric,tr,dev,base,teacher_top=None,libcodes=None,exclude_episode=False):
    n=len(qx);top=np.zeros((n,16),int);actions=np.zeros((n,10,7),np.float32);d1=np.zeros(n)
    for t in range(10):
        lr=np.flatnonzero(tr['task_id']==t)
        for early in (False,True):
            qr=np.flatnonzero((dev['task_id']==t)&((dev['step']==0)==early))
            if not len(qr):continue
            par=metric[t][int(early)];wf,shift=par['wf'],par['shift']
            z=libcodes[t][int(early)] if libcodes else libx[lr]@wf-shift
            q=qx[qr]@wf-shift
            ds=cdist(q,z)
            if exclude_episode:
                ds[dev['episode'][qr,None]==tr['episode'][lr][None,:]]=np.inf
            idx=np.argsort(ds,axis=1,kind='stable')[:,:16]
            if exclude_episode:
                assert np.isfinite(np.take_along_axis(ds,idx,1)).all()
            dk=np.take_along_axis(ds,idx,1); w=_kernel_w(dk-dk[:,:1],base.kref);w/=w.sum(1,keepdims=True)
            top[qr]=lr[idx];d1[qr]=dk[:,0]
            actions[qr]=(tr['action'][lr[idx],:10,:7]*w[:,:,None,None]).sum(1)
    e5=np.sqrt(np.mean(((actions[:,:5]-dev['label'][:,:5,:7])/tr['sig'])**2,axis=(1,2)))
    e10=np.sqrt(np.mean(((actions-dev['label'][:,:10,:7])/tr['sig'])**2,axis=(1,2)))
    # Phase proxy 1: deviation from teacher top-1 library progress.
    # Phase proxy 2: backwards jump (> .10 normalized donor progress) in a
    # consecutive recorded-look pair of the same accepted episode.
    prog=tr['progress'][top[:,0]]
    same=(dev['uid'][1:]==dev['uid'][:-1]);jumps=(prog[1:]-prog[:-1])<-.10
    def stats(mask):
        valid=mask&dev['label_valid']
        return {'n':int(mask.sum()),'labelled':int(valid.sum()),'err5':float(e5[valid].mean()) if valid.any() else None,
                'err10':float(e10[valid].mean()) if valid.any() else None,'err10_p90':float(np.quantile(e10[valid],.9)) if valid.any() else None,
                'top1_agreement':float((top[mask,0]==teacher_top[mask,0]).mean()) if teacher_top is not None else 1.,
                'top16_overlap':float(np.mean([len(set(a)&set(b))/16 for a,b in zip(top[mask],teacher_top[mask])])) if teacher_top is not None else 1.,
                'phase_teacher_mae':float(np.abs(prog[mask]-tr['progress'][teacher_top[mask,0]]).mean()) if teacher_top is not None else 0.,
                'phase_backwards_rate':float(jumps[same&mask[1:]&mask[:-1]].mean()) if (same&mask[1:]&mask[:-1]).any() else None}
    summary=stats(np.ones(n,bool));summary['by_task']={str(t):stats(dev['task_id']==t) for t in range(10)}
    summary['by_regime']={x:stats((dev['step']==0)==b) for x,b in [('early',True),('stale',False)]}
    summary['name']=name
    return summary,{'top':top,'action':actions,'err5':e5,'err10':e10,'progress':prog,'d1':d1}

def run(suite,size,devroot):
    out=HERE/f'pi05_{suite}_{size}';out.mkdir(exist_ok=True)
    tr=dict(np.load(DERIVED/f'pi05_{suite}_{size}/train.npz',allow_pickle=False))
    dp=DERIVED/f'dev_pi05_{suite}_{devroot}'
    raw=dict(np.load(dp/'dev.npz',allow_pickle=False));mask=np.load(dp/f'unseen_{size}.npy')
    dev={k:v[mask] for k,v in raw.items()};assert len(dev['rs'])>0
    m,b,_,rows,prov=load_cell(suite,size);assert np.array_equal(tr['rows'],rows)
    P=np.concatenate([tr['P0'],tr['P1']],axis=1)
    Q=np.concatenate([dev[f'key_v{c}']@getattr(b,f'B{c}T').T-getattr(b,f'muB{c}') for c in range(2)],1)
    lx=np.c_[P,tr['rs'][:,:8]];qx=np.c_[Q,dev['rs'][:,:8]]
    metric=teacher_metric(b);zc=reference_codes(b,tr)
    ref,rd=evaluate('deployed_key_geometry',lx,qx,metric,tr,dev,b,libcodes=zc)
    summaries=[ref];np.savez(out/'eval_reference.npz',**rd)
    trainG=[grids(tr[f'img{c}']) for c in range(2)];devG=[grids(dev[f'img{c}']) for c in range(2)]
    pc=[pca_fit_small(x) for x in trainG]
    lp=np.concatenate([(x-mu)@v for x,(mu,v) in zip(trainG,pc)],1)
    qp=np.concatenate([(x-mu)@v for x,(mu,v) in zip(devG,pc)],1)
    sx=np.c_[lp,tr['rs'][:,:8]];sq=np.c_[qp,dev['rs'][:,:8]]
    sm=task_metric(sx,tr)
    s,detail=evaluate('symmetric_rgb_spatial_awm',sx,sq,sm,tr,dev,b,rd['top']);summaries.append(s)
    np.savez(out/'eval_symmetric_rgb_spatial_awm.npz',**detail)
    with (out/'symmetric_rgb_spatial_awm.pkl').open('wb') as f:pickle.dump({'pca':pc,'metric':sm},f)
    print(s['name'],s['err5'],s['err10'],s['top1_agreement'],flush=True)
    x=np.c_[*trainG,tr['rs'][:,:8]];q=np.c_[*devG,dev['rs'][:,:8]]
    for alpha in (.01,.1,1.):
        start=time.perf_counter();model=fit_ridge(x,P,tr['task_id'],alpha)
        pred=predict_ridge(model,q,dev['task_id']);pq=np.c_[pred,dev['rs'][:,:8]]
        name=f'asymmetric_ridge_{alpha:g}'
        s,detail=evaluate(name,lx,pq,metric,tr,dev,b,rd['top'],libcodes=zc);s['fit_seconds']=time.perf_counter()-start
        summaries.append(s);np.savez(out/f'eval_{name}.npz',**detail)
        with (out/f'{name}.pkl').open('wb') as f:pickle.dump(model,f)
        print(name,s['err5'],s['err10'],s['top1_agreement'],flush=True)
    # State-only negative control, learned with exactly the same action-pair rule.
    sx=tr['rs'][:,:8];sq=dev['rs'][:,:8];sm=task_metric(sx,tr)
    s,detail=evaluate('state_only_awm',sx,sq,sm,tr,dev,b,rd['top']);summaries.append(s)
    np.savez(out/'eval_state_only_awm.npz',**detail)
    # Record real deployed served HIT chunks as a separate contextual reference.
    hit=dev['hit'].astype(bool)&dev['label_valid']
    served_err=np.sqrt(np.mean(((dev['action'][:,:10,:7]-dev['label'][:,:10,:7])/tr['sig'])**2,axis=(1,2)))
    dump(out/'results.json',{'suite':suite,'size':size,'devroot':devroot,'unseen_episodes':len(set(dev['uid'])),'looks':len(dev['rs']),'labelled':int(dev['label_valid'].sum()),'reference_scope':'AWM early at step0; stale otherwise; no continuity, insurance, guards or correction; exact teacher code geometry','served_hit_err10':float(served_err[hit].mean()) if hit.any() else None,'served_hit_n':int(hit.sum()),'methods':summaries,'provenance':prov})
    np.savez(out/'evaluation_inputs.npz',P=P,Q=Q,dev_mask=mask)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--suite',default='l10');ap.add_argument('--size',type=int,default=50);ap.add_argument('--devroot',default='r12_devlooks_50L');a=ap.parse_args();run(a.suite,a.size,a.devroot)
