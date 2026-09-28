"""Library-only LOEO displacement monitor; observed-path tail and bridge audits.
No fit uses query or closed-loop evaluation data. No SR estimated from action error.
"""
from pathlib import Path
import json
from collections import Counter,defaultdict
import numpy as np
from analyze_logs import load
from analyze_chunks import arr,stat,rms
O=Path(__file__).parent;S=Path('/home/weiland/trace_runs/offline_search_store')

def features(rs,a):return np.column_stack((np.ones(len(rs)),rs[:,:8],a[:,:5,:7].mean(1)))
def fit_xy(X,Y):
    gram=X.T@X;lam=.01*np.trace(gram[1:,1:])/15
    reg=np.eye(16)*lam;reg[0,0]=0
    return np.linalg.solve(gram+reg,X.T@Y)

def fit(model,suite,ln):
    p=S/'library'/f'{model}_{suite}'/ln
    rs=np.asarray(arr(p,'rs')[:,:8],float);a=arr(p,'action');nx=arr(p,'next');ep=arr(p,'episode')
    r=np.flatnonzero(nx>=0);n=np.asarray(nx[r]);X=features(rs[r],a[r,:5,:7]);Y=rs[n,:3]-rs[r,:3]
    labels=ep[r];oof=np.empty_like(Y);err=np.empty(len(Y));scales=[]
    for e in np.unique(labels):
        test=labels==e;train=~test
        B=fit_xy(X[train],Y[train]);oof[test]=X[test]@B
        sc=np.maximum(Y[train].std(0),.005);err[test]=rms((Y[test]-oof[test])/sc,1)
    B=fit_xy(X,Y);yscale=np.maximum(Y.std(0),.005);q99=float(np.quantile(err,.99))
    M=dict(model=model,suite=suite,library=ln,rows=len(rs),edges=len(r),features=16,outputs=3,
           ridge_trace_fraction=.01,B=B.tolist(),yscale=yscale.tolist(),q99=q99,
           loeo_error=stat(err),loeo_xyz_rmse=float(rms(oof-Y,None)),
           zero_delta_xyz_rmse=float(rms(Y,None)),fit_bytes_float32=(B.size+yscale.size+1)*4)
    Q=S/'queries'/f'{model}_{suite}_inf';qa=arr(Q,'a_inf');qe=arr(Q,'ep');qr=arr(Q,'rs')
    rr=np.flatnonzero(qe[1:]==qe[:-1]);xn=features(qr[rr],qa[rr,:5,:7]);yn=qr[rr+1,:3]-qr[rr,:3]
    er=rms((yn-xn@B)/yscale,1);M['inf_evaluation']=dict(n=len(er),triggered=int((er>q99).sum()),trigger_share=float(np.mean(er>q99)),error=stat(er))
    return M

def audit(name,ln,suite,model):
    journal,rec,_,_,_=load('r04_k7',name);p=S/'library'/f'pi05_{suite}'/ln
    a=arr(p,'action');nx=arr(p,'next');rs=arr(p,'rs');ep=arr(p,'episode');tid=arr(p,'task_id');step=arr(p,'step')
    sig=a[:,:5,:7].std(axis=(0,1));B=np.array(model['B']);ys=np.array(model['yscale']);thr=model['q99']
    natural=rms((a[:,5,:6]-a[:,4,:6])/sig[:6],1);join95=float(np.quantile(natural,.95))
    groups=defaultdict(list);bridge=Counter();join=[];offphase=[];head_equiv=[];tail_equiv=[];orphan=0;replaced=0
    for uid,ds in rec.items():
        for i,d in enumerate(ds[:-1]):
            X=features(np.array([d['robot_state'][:8]]),np.array([d['served_head']]))
            y=np.array(ds[i+1]['robot_state'][:3])-np.array(d['robot_state'][:3])
            error=float(rms((y-X[0]@B)/ys,None));groups[d['src']].append(error)
            if i+2>=len(ds) or d['src']!='cache' or ds[i+1]['src']!='cache_blind':continue
            r=np.array(d['rows'],int);w=np.array(d['weights'],float);w/=w.sum()
            anchor=np.einsum('i,ijk->jk',w,a[r,:,:7]);
            head_equiv.append(float(np.max(np.abs(anchor[:5]-d['served_head']))))
            tail_equiv.append(float(np.max(np.abs(anchor[5:10]-ds[i+1]['served_head']))))
            bridge['anchors_with_observed_tail_and_followup']+=1
            head_event=bool(np.any((anchor[4:10,6]>=0)!=(anchor[4,6]>=0)))
            bridge['tail_checkpoint_grip_event']+=int(head_event)
            bridge['tail_checkpoint_residual']+=int(error>thr)
            bridge['tail_checkpoint_event_or_residual']+=int(head_event or error>thr)
            n1=nx[r];valid=n1>=0;n2=np.full(len(r),-1,int);n2[valid]=nx[n1[valid]];good=n2>=0
            bridge['all16_next2_valid']+=int(np.all(good))
            if not np.all(good):continue
            assert np.all(ep[n2]==ep[r]) and np.all(tid[n2]==tid[r]) and np.all(step[n2]==step[r]+2)
            br=np.einsum('i,ijk->jk',w,a[n2,:5,:7]);jm=float(rms((br[0,:6]-anchor[9,:6])/sig[:6],None));join.append(jm)
            bridge['join_below_025']+=int(jm<.25)
            # Event criterion is direct issued-plan sign stability, not a gripper dwell rule.
            gp=anchor[9,6]>=0;event=bool(np.any((br[:,6]>=0)!=gp))
            disagreement=float(np.max(np.minimum(w@(a[n2,:5,6]>=0),1-w@(a[n2,:5,6]>=0))))
            stable=(not event and disagreement<.2)
            bridge['grip_stable_and_agreement']+=int(stable)
            # Last executed segment is the actual cached tail, not the unused next library head.
            dt=ds[i+1];X=features(np.array([dt['robot_state'][:8]]),np.array([dt['served_head']]))
            y=np.array(ds[i+2]['robot_state'][:3])-np.array(dt['robot_state'][:3]);er=float(rms((y-X[0]@B)/ys,None))
            ok=jm<.25 and stable and er<=thr
            bridge['combined_probe_eligible']+=int(ok)
            bridge['eligible_next_was_miss']+=int(ok and not ds[i+2]['hit'])
            bridge['eligible_in_failed_episode']+=int(ok and not journal[uid].get('success',False))
            calibrated=jm<=join95 and stable and er<=thr and np.all(nx[n2]>=0)
            bridge['calibrated_join_and_nonterminal_eligible']+=int(calibrated)
            bridge['calibrated_eligible_next_miss']+=int(calibrated and not ds[i+2]['hit'])
            # A bridge's actions are unexecuted: disagreement with observed next command is descriptive only.
            if ok:offphase.append(float(rms((br[:,:6]-np.array(ds[i+2]['served_head'])[:,:6])/sig[:6],None)))
        # Correct the naive fixed-path accounting for inherited blind slots whose anchor was replaced.
        k=0
        while k<len(ds)-1:
            if not ds[k]['hit']:
                replaced+=1
                if k+2<len(ds) and not ds[k+2]['vision']:orphan+=1
                k+=2
            else:k+=1
    return dict(arm=name,library=ln,monitor_by_source={k:dict(n=len(v),error=stat(v),triggered=int(np.sum(np.array(v)>thr)),trigger_share=float(np.mean(np.array(v)>thr))) for k,v in groups.items()},
                bridge=dict(bridge),library_natural_join_p95=join95,bridge_join_cont6=stat(join),eligible_bridge_vs_observed_next_cont6=stat(offphase),
                reconstruction_max_head=max(head_equiv),reconstruction_max_tail=max(tail_equiv),
                fixed_path_policy_replacements=replaced,orphaned_original_blind_slots=orphan,
                orphan_vision_cost=.152*orphan,orphan_extra_miss_cost_upper=.848*orphan)

if __name__=='__main__':
    fits=[]
    for m in ('pi05','groot'):
        for s in ('spatial','l10'):
            for ln in ('current','bpool_cs' if m=='pi05' else 'bpool_all'):
                f=fit(m,s,ln);fits.append(f);print('FIT',m,s,ln,f['q99'],f['inf_evaluation']['trigger_share'],flush=True)
    (O/'dynamics_fits.json').write_text(json.dumps(fits,indent=2))
    audits=[]
    for s,scale,ln in (('l10',50,'current'),('l10',500,'bpool_cs'),('spatial',500,'bpool_cs')):
        name=f"r4k7_p_{'sp' if s=='spatial' else s}_{scale}_tail1ug"
        m=next(f for f in fits if (f['model'],f['suite'],f['library'])==('pi05',s,ln))
        a=audit(name,ln,s,m);audits.append(a);print('AUDIT',json.dumps(a),flush=True)
    (O/'cadence_audit.json').write_text(json.dumps(audits,indent=2))
