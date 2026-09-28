"""Own-library forward dynamics and contact mismatch diagnostics, no policy fitting."""
from pathlib import Path
import json
import numpy as np
from diagnose_logs import HERE, STORE, FEATURES

def arr(path, name): return np.load(path/f'{name}.npy',mmap_mode='r')
def xmat(a):
    return np.c_[np.ones(len(a)),a[:,:5,:6].mean(1)]
def fit(x,y):
    penalty=np.eye(x.shape[1])*1e-4; penalty[0,0]=0
    return np.linalg.solve(x.T@x+penalty,x.T@y)
def pct(a): return np.percentile(a,[5,25,50,75,95]).tolist() if len(a) else []
def correlation(a,b): return float(np.corrcoef(a,b)[0,1])
def stats(a): return dict(n=int(len(a)),mean=float(np.mean(a)),pcts=pct(a))
def event_stats(d,mask):
    n=len(d['ep']); mask=np.asarray(mask,bool)
    first=np.r_[True,np.diff(d['ep'])!=0]
    flags=d['feats'][:,FEATURES.index('os_force_miss')]>0
    eps=np.unique(d['ep'][mask]); all_ep=d['ep'][first]; all_s=d['success'][first]
    esf=np.isin(all_ep,eps)
    trig=np.flatnonzero(mask)
    earliest=np.array([np.flatnonzero(mask & (d['ep']==ep))[0] for ep in eps],int)
    return dict(decisions=int(mask.sum()),episodes=len(eps),failed_episodes=int((esf&~all_s).sum()),
                successful_episodes=int((esf&all_s).sum()),
                hit_decisions=int((mask&d['hit']).sum()),blind_decisions=int((mask&~d['vision']).sum()),
                existing_guard_decisions=int((mask&flags).sum()),
                first_step_pcts=pct(d['step'][earliest]),
                once_per_episode_extra_ir=float(np.sum(np.where(d['hit'][earliest],
                    np.where(d['vision'][earliest],.848,1.),0.))/n))

def main():
    lib_reports=[]; log_reports=[]
    for model in ['pi05','groot']:
      for suite in ['spatial','l10']:
       for size,name in [(50,'current'),(500,'bpool_cs' if model=='pi05' else 'bpool_all')]:
        p=STORE/'library'/f'{model}_{suite}'/name
        s=arr(p,'rs')[:,:8].astype(float); a=arr(p,'action')[:,:5,:7].astype(float)
        nxt=arr(p,'next'); ep=arr(p,'episode'); st=arr(p,'step')
        r=np.flatnonzero(nxt>=0); r=r[(ep[r]==ep[nxt[r]])&(st[nxt[r]]==st[r]+1)]
        x=xmat(a[r]); y=s[nxt[r],:3]-s[r,:3]
        pred=np.zeros_like(y)
        for fold in range(5):
            test=ep[r]%5==fold; pred[test]=x[test]@fit(x[~test],y[~test])
        b=fit(x,y); scale=np.maximum(y.std(0),1e-4)
        resid=np.linalg.norm((y-pred)/scale,axis=1)
        q95=float(np.quantile(resid,.95))
        predsize=np.linalg.norm(pred/scale,axis=1)
        p50=float(np.median(predsize))
        aperture=(s[:,6]-s[:,7])/2
        lo,hi=np.percentile(aperture,[1,99]); width=(aperture-lo)/(hi-lo)
        closed=(a[:,:,6]*(1 if model=='pi05' else -1)>0).all(1)
        lib_reports.append(dict(model=model,suite=suite,size=size,rows=len(s),transitions=len(r),
           episodes=len(np.unique(ep)),fivefold_episode_cv_r2=1-float(((y-pred)**2).sum()/((y-y.mean(0))**2).sum()),
           axis_r2=(1-((y-pred)**2).sum(0)/((y-y.mean(0))**2).sum(0)).tolist(),
           residual=stats(resid),prediction_size=stats(predsize),beta=b.tolist(),delta_scale=scale.tolist(),
           aperture_lo=float(lo),aperture_hi=float(hi),closed_width=stats(width[closed]),
           cv_threshold95=q95,model_bytes=int(b.astype('f4').nbytes+scale.astype('f4').nbytes+16)))
        np.savez(HERE/f'dynamics_{model}_{suite}_{size}.npz',beta=b,scale=scale,lo=lo,hi=hi,q95=q95,p50=p50)
        if model!='pi05': continue
        ss='sp' if suite=='spatial' else suite
        for file in HERE.glob(f'r4k7_p_{ss}_{size}_*.npz'):
            d=dict(np.load(file)); n=len(d['ep'])
            rr=np.arange(n-1); rr=rr[(d['ep'][rr]==d['ep'][rr+1])&(d['step'][rr+1]==d['step'][rr]+1)]
            yy=d['state'][rr+1,:3]-d['state'][rr,:3]; pp=xmat(d['action'][rr])@b
            err=np.linalg.norm((yy-pp)/scale,axis=1); mag=np.linalg.norm(pp/scale,axis=1)
            ratio=np.sum((yy/scale)*(pp/scale),axis=1)/np.maximum(mag**2,1e-9)
            blocked=(mag>p50)&(ratio<.25)
            cur_width=((d['state'][:,6]-d['state'][:,7])/2-lo)/(hi-lo)
            cls=(d['action'][:,:,6]>0).all(1)
            streak=np.zeros(n,int)
            for j in range(1,n):
                if d['ep'][j]==d['ep'][j-1] and cls[j-1]: streak[j]=streak[j-1]+1
            last2=np.zeros(n,bool); last2[rr+1]=blocked
            bad2=last2&np.r_[False,last2[:-1]]&(d['step']>=2)
            alarm=np.zeros(n,bool); alarm[rr+1]=err>q95
            # At anchor-tail blind decision, rows and weights are the *previous*
            # anchor's full exact kernel. Expected aperture from next[row].
            exact=(d['rows'][:,0]>=0)&(~d['vision'])
            exwidth=np.full(n,np.nan); targetdelta=np.full((n,3),np.nan)
            for j in np.flatnonzero(exact):
                rows=d['rows'][j]; w=d['weights'][j]; ok=(rows>=0)&np.isfinite(w)
                rows=rows[ok]; w=w[ok]
                if 'tail' in file.stem:
                    nr=nxt[rows]
                    if np.any(nr<0): continue
                    exwidth[j]=w@width[nr]
                    targetdelta[j]=w@(s[nr,:3]-s[rows,:3])
                else:
                    exwidth[j]=w@width[rows]
            empty=(cur_width<.05)&(streak>=2)
            mismatch=empty&(exwidth>.25)
            record=dict(arm=file.stem,size=size,suite=suite,transitions=len(rr),
                observed_r2=1-float(((yy-pp)**2).sum()/((yy-yy.mean(0))**2).sum()),
                residual_success=stats(err[d['success'][rr]]),residual_fail=stats(err[~d['success'][rr]]),
                blocked=event_stats(d,np.isin(np.arange(n),rr[blocked]+1)),
                blocked2=event_stats(d,bad2),innovation95=event_stats(d,alarm),
                empty_grasp=event_stats(d,empty),contact_mismatch=event_stats(d,mismatch),
                success_width=stats(cur_width[(streak>=2)&d['success']]),
                fail_width=stats(cur_width[(streak>=2)&~d['success']]),
                exact_kernel_decisions=int(np.isfinite(exwidth).sum()))
            # Restrict to first 30 decisions to expose length confounding.
            record['blocked2_first30']=event_stats(d,bad2&(d['step']<30))
            record['empty_first30']=event_stats(d,empty&(d['step']<30))
            record['contact_mismatch_first30']=event_stats(d,mismatch&(d['step']<30))
            log_reports.append(record)
            np.savez_compressed(HERE/f'feedback_{file.stem}.npz',rr=rr,err=err,mag=mag,ratio=ratio,
                blocked2=bad2,innovation95=alarm,width=cur_width,close_streak=streak,expected_width=exwidth,
                contact_mismatch=mismatch,targetdelta=targetdelta)
            print(json.dumps({k:record[k] for k in ['arm','observed_r2','blocked2_first30','contact_mismatch_first30']}),flush=True)
    (HERE/'dynamics_library.json').write_text(json.dumps(lib_reports,indent=2))
    (HERE/'feedback_logs.json').write_text(json.dumps(log_reports,indent=2))

if __name__=='__main__': main()
