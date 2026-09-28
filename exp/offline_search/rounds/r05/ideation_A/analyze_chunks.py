"""Offline mechanism diagnostics. These are NOT counterfactual SR estimates.

All scales/quantiles in a cell come from that deployed library. Query trajectories
are evaluation-only, collected Sept 23/24, not the R4 horizon intervention.
"""
from pathlib import Path
import json
import numpy as np

O=Path(__file__).parent
S=Path('/home/weiland/trace_runs/offline_search_store')

def arr(p,n): return np.load(p/(n+'.npy'),mmap_mode='r')
def stat(x):
    x=np.asarray(x,float)
    return dict(n=len(x),mean=float(x.mean()),p50=float(np.median(x)),p90=float(np.quantile(x,.9)),p99=float(np.quantile(x,.99))) if len(x) else dict(n=0)
def rms(x,axis):return np.sqrt(np.mean(x*x,axis=axis))

def compare(a,rs,r,n,sig,scale,closed_positive=True):
    head=np.asarray(a[r,:5,:7],float); tail=np.asarray(a[r,5:10,:7],float); repl=np.asarray(a[n,:5,:7],float)
    overlap=rms((tail[:,:,:6]-repl[:,:,:6])/sig[:6],(1,2))
    natural=rms((tail[:,0,:6]-head[:,-1,:6])/sig[:6],1)
    boundary=rms((repl[:,0,:6]-head[:,-1,:6])/sig[:6],1)
    hs=head[:,-1,6]>=0;ts=tail[:,:,6]>=0;ns=repl[:,:,6]>=0
    planned=np.any(ts!=hs[:,None],axis=1)
    cancelled=planned & ~np.any(ns!=hs[:,None],axis=1)
    motion=rms((rs[n,:8]-rs[r,:8])/scale,1)
    # A physical-motion proxy, not object progress. No success-based fitting.
    return dict(edges=len(r),overlap_cont6=stat(overlap),natural_4to5=stat(natural),replanned_4to0=stat(boundary),
                boundary_gt_natural=float(np.mean(boundary>natural)),grip_element_disagree=float(np.mean(ts!=ns)),
                grip_any_disagree=int(np.any(ts!=ns,axis=1).sum()),planned_grip_switch=int(planned.sum()),
                planned_switch_not_in_next_head=int(cancelled.sum()),
                planned_close=int((planned & (hs!=closed_positive)).sum()),
                cancelled_close=int((cancelled & (hs!=closed_positive)).sum()),motion=stat(motion))

def run(model,suite,libname):
    L=S/'library'/f'{model}_{suite}'/libname; Q=S/'queries'/f'{model}_{suite}_inf'
    la,rs,nxt=arr(L,'action'),arr(L,'rs'),arr(L,'next')
    sig=np.asarray(la[:,:5,:7]).std(axis=(0,1)).astype(float)
    scale=np.maximum(np.asarray(rs[:,:8]).std(axis=0),.05)
    r=np.flatnonzero(nxt>=0);n=np.asarray(nxt[r]);eps=arr(L,'episode');step=arr(L,'step')
    assert np.all(eps[r]==eps[n]) and np.all(step[n]==step[r]+1)
    out=dict(model=model,suite=suite,lib=libname,library_rows=len(la),library_episodes=len(np.unique(eps)),
             sigma=sig.tolist(),state_scale=scale.tolist(),library=compare(la,rs,r,n,sig,scale,model=='pi05'))
    qa=arr(Q,'a_inf');qe=arr(Q,'ep');qs=arr(Q,'rs');episodes=json.loads((Q/'episodes.json').read_text())
    r=np.flatnonzero(qe[1:]==qe[:-1]);n=r+1
    out['queries']=compare(qa,qs,r,n,sig,scale,model=='pi05')
    out['query_success']=sum(e['success'] for e in episodes);out['query_episodes']=len(episodes)
    out['per_task']={}
    for t in range(10):
        ee=np.array([e['task_id']==t for e in episodes]);sel=ee[qe[r]]
        out['per_task'][str(t)]=compare(qa,qs,r[sel],n[sel],sig,scale,model=='pi05')
    libmove=rms((rs[np.asarray(nxt[np.flatnonzero(nxt>=0)]) ,:8]-rs[np.flatnonzero(nxt>=0),:8])/scale,1)
    threshold=float(np.quantile(libmove,.1));qmotion=rms((qs[n,:8]-qs[r,:8])/scale,1)
    success=np.array([e['success'] for e in episodes])[qe[r]]
    out['low_motion_threshold']=threshold
    out['query_by_outcome']={str(s):dict(low_motion_share=float(np.mean(qmotion[success==s]<threshold)),
                                        **compare(qa,qs,r[success==s],n[success==s],sig,scale,model=='pi05')) for s in (False,True)}
    # The original GR00T plan still has a second complete tail; this comparison
    # follows TWO replanning steps and is even further from a causal L15 rollout.
    if model=='groot':
        r2=np.flatnonzero(qe[2:]==qe[:-2]);out['second_tail_overlap_cont6']=stat(rms((qa[r2,10:15,:6]-qa[r2+2,:5,:6])/sig[:6],(1,2)))
        out['second_tail_grip_disagree']=float(np.mean((qa[r2,10:15,6]>=0)!=(qa[r2+2,:5,6]>=0)))
    # Endpoint coverage for a single fixed-phase bridge after ten controls.
    nn=np.full(len(la),-1,np.int32);ok=np.flatnonzero(nxt>=0);nn[ok]=nxt[nxt[ok]]
    out['bridge_rows_next2_available']=int(np.sum(nn>=0))
    (O/f'chunks_{model}_{suite}_{libname}.json').write_text(json.dumps(out,indent=2))
    print(model,suite,libname,'query',out['queries'],'library',out['library'],flush=True)
    return out

if __name__=='__main__':
    results=[run(m,s,l) for m in ('pi05','groot') for s in ('spatial','l10') for l in ('current','bpool_cs' if m=='pi05' else 'bpool_all')]
    (O/'chunks_summary.json').write_text(json.dumps(results,indent=2))
