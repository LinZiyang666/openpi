"""R4-A read-only-store diagnostics. This is a replay experiment, not a server implementation.

Run from repo root with taskset and single-thread BLAS (see RUN.md).
All predictions use only the deployed candidate library, anchor keys, and current/past rs[:8].
Query a_inf/episode length are touched only by the evaluator. No policy/GPU calls.
"""
from __future__ import annotations
import argparse, concurrent.futures, csv, json, pathlib, pickle, sys, time
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from exp.offline_search.harness import api, store
from exp.offline_search.closed_loop.plugin import load_method_class

OUT = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path('/home/weiland/trace_runs/offline_search_store')
RUNS = pathlib.Path('/home/weiland/trace_runs/os_closed_loop')
CELLS = ['pi05_spatial', 'pi05_l10', 'groot_spatial', 'groot_l10']

def library(key, scale):
    return store.LibraryView(ROOT, key, 'current' if scale == 50 else ('bpool_cs' if key.startswith('pi05') else 'bpool_all'))

def artifact(key, scale):
    m,s=key.split('_'); arm=f'oscl{scale}_{"p" if m=="pi05" else "g"}_{"sp" if s=="spatial" else s}_cl2'
    return RUNS/f'r02_g{scale}'/'fits'/f'{arm}.pkl'

def get_method(key, scale):
    load_method_class('exp/offline_search/rounds/r02/g1_awm/awm.py:AWM')
    blob=pickle.load(artifact(key,scale).open('rb'))
    M=blob['method']; M.prof=api.NULL_PROFILER
    return M

def mix(M, dist, T):
    ix=np.argsort(dist,axis=1,kind='stable')[:,:16]
    ds=np.take_along_axis(dist,ix,1).astype(np.float64)
    rel=ds-ds[:,:1]
    w=np.exp(-(rel/np.maximum(rel[:,M.kref-1:M.kref],1e-6))**2)
    w=(w/w.sum(1)[:,None]).astype(np.float32)
    rows=T.rows[ix]
    a=np.einsum('nk,nkhd->nhd',w,M.act[rows,:,:7],optimize=True)
    return rows,w,a,ds

def baseline_job(job):
    key,scale=job; M=get_method(key,scale)
    for arm in ['inf','cache']:
        cell=f'{key}_{arm}'; dest=OUT/f'anchors_{cell}_{scale}.npz'
        if dest.exists():
            print('cached',dest.name,flush=True); continue
        q=store.QueryCell(ROOT,cell); N=q.N
        rows=np.zeros((N,16),np.int32); weights=np.zeros((N,16),np.float32)
        action=np.zeros((N,M.H,7),np.float32); stale=np.zeros_like(action)
        scores=np.zeros((N,16),np.float32)
        # Batched projection/distance is mathematically the same AWM; scalar validation below.
        X=np.empty((N,136),np.float32)
        for lo in range(0,N,256):
            hi=min(N,lo+256)
            X[lo:hi,:64]=q.key_v0[lo:hi]@M.B0T.T-M.muB0
            X[lo:hi,64:128]=q.key_v1[lo:hi]@M.B1T.T-M.muB1
        X[:,128:]=q.rs[:,:8]
        tids=np.array([e['task_id'] for e in q.episodes])[q.ep]
        for tid,T in M.tasks.items():
            ids=np.flatnonzero(tids==tid)
            for lo in range(0,len(ids),256):
                ix=ids[lo:lo+256]; xx=X[ix]; z=xx@T.Wf-T.shift
                dd=T.z2[None]-2*z@T.Z.T+(z*z).sum(1)[:,None]
                st0=np.asarray(q.step[ix])==0
                if st0.any():
                    y=xx[st0]@T.W0f-T.c0
                    cross=(y@T.A0.T)@T.Z.T
                    dd[st0]=T.n20[None]-2*cross+(y*y).sum(1)[:,None]
                d=np.sqrt(np.maximum(dd,0))
                stale[ix]=mix(M,d,T)[2]
                dt=d.copy()
                fr=(~st0) if arm=='inf' else np.zeros(len(ix),bool)
                if fr.any():
                    tail=(q.a_exec[ix[fr]-1,5:10,:7]/M.sig).reshape(-1,35)
                    c=np.sqrt(np.maximum(T.h2[None]-2*tail@T.HD.T+(tail*tail).sum(1)[:,None],0)/35)
                    dt[fr]=d[fr]/(np.median(d[fr],axis=1)[:,None]+1e-12)+M.lam_c*c/T.s_c
                rr,ww,aa,ss=mix(M,dt,T)
                rows[ix]=rr;weights[ix]=ww;action[ix]=aa;scores[ix]=-ss
        # Non-counterfactual scalar API parity sample, deterministic across every cell/scale.
        A=api.QueryArrays(q); maxdiff=0.; agree=0; nkagree=0
        for i in np.linspace(0,N-1,40,dtype=int):
            e=q.episodes[int(q.ep[i])]; ev=api.EpisodeView(e['uid'],e['task'],e['task_id'],e['init'],int(q.ep[i]),0)
            view=api.QueryView(A,int(i),e['start'],int(q.step[i]),e['task_id'],ev)
            r=M.query(view);maxdiff=max(maxdiff,float(np.max(abs(r.action[:,:7]-action[i]))))
            agree+=int(r.topk[0]==rows[i,0]);nkagree+=int(np.array_equal(r.topk,rows[i]))
        assert agree==40 and maxdiff<.002,(cell,scale,agree,maxdiff)
        np.savez_compressed(dest,rows=rows,weights=weights,action=action,stale=stale,scores=scores)
        meta=dict(cell=cell,scale=scale,N=N,episodes=len(q.episodes),top1_agree=agree/40,top16_agree=nkagree/40,
                  scalar_max_abs_action_diff=maxdiff,artifact=str(artifact(key,scale)),artifact_bytes=artifact(key,scale).stat().st_size)
        (OUT/f'anchors_{cell}_{scale}.json').write_text(json.dumps(meta,indent=2))
        print(json.dumps(meta),flush=True)
    return key,scale

def topology(L):
    n=L.L; nxt=np.where(L.next>=0,L.next,np.arange(n)).astype(int)
    prv=np.where(L.prev>=0,L.prev,np.arange(n)).astype(int)
    adv=np.empty((9,n),int);adv[0]=np.arange(n)
    for h in range(1,9):adv[h]=nxt[adv[h-1]]
    g=L.action[:,:5,6]>=0
    event=(g!=g[:,0,None]).any(1)|(g[:,0]!=g[prv,4])|(g[:,4]!=g[nxt,0])
    near=event|event[prv]|event[nxt]
    # thresholds/scales fitted separately on the deployed library only.
    sig=np.empty((n,8),np.float32);motion10={}; motion50={}
    for t in L.tasks():
        r=L.rows_of_task(t);sig[r]=np.maximum(np.std(L.rs[r,:8],axis=0),.05)
        good=r[np.asarray(L.next[r])>=0]
        motion=np.sqrt(np.mean(((L.rs[nxt[good],:8]-L.rs[good,:8])/sig[good])**2,1))
        motion10[int(t)]=float(np.percentile(motion,10));motion50[int(t)]=float(np.median(motion))
    return nxt,prv,adv,event,near,sig,motion10,motion50

def summarize_values(meta,mask,err,gm,drift,base,extra=None):
    n=int(mask.sum())
    if not n:return None
    out={**meta,'n':n,'err_mean':float(err[mask].mean()),'err_median':float(np.median(err[mask])),
         'err_p90':float(np.percentile(err[mask],90)),'grip_mis':float(gm[mask].mean()),
         'awm_drift':float(drift[mask].mean()),'delta_awm':float((err[mask]-base[mask]).mean()),
         'bad_delta_02':float(((err-base)[mask]>.2).mean())}
    if extra:out.update(extra)
    return out

def analysis_job(job):
    key,scale=job;L=library(key,scale);nxt,prv,adv,event,near,ss,m10,m50=topology(L)
    sigma=store.action_sigma(str(ROOT),key); act=np.asarray(L.action[:,:,:7]); rs=np.asarray(L.rs[:,:8])
    records=[];triggers=[];calibrations=[]
    for arm in ['inf','cache']:
        cell=f'{key}_{arm}';q=store.QueryCell(ROOT,cell);B=np.load(OUT/f'anchors_{cell}_{scale}.npz')
        allrows=B['rows'];allw=B['weights'];allact=B['action'];N=q.N
        qrs=np.asarray(q.rs[:,:8]); tids=np.array([e['task_id'] for e in q.episodes])[q.ep]
        bins=np.minimum(2,3*np.asarray(q.step)//q.num_steps_row)
        motion=np.zeros(N);motion[1:]=np.sqrt(np.mean(((qrs[1:]-qrs[:-1])/ss[allrows[1:,0]])**2,1))
        isstill=(q.step>0)&(motion<np.array([m10[int(t)] for t in tids]))
        still2=isstill&np.r_[False,isstill[:-1]]&(q.step>=2)
        prev_phase={k:allrows.copy() for k in ['abs','delta','hybrid']}
        for h in range(1,5):
            ix=np.arange(N-h);ix=ix[q.ep[ix]==q.ep[ix+h]];target=ix+h
            rr=allrows[ix];ww=allw[ix]; clock=adv[h,rr]
            basea=allact[target,:5];gt=np.asarray(q.a_inf[target,:5,:7])
            baseerr=np.sqrt(np.mean(((basea-gt)/sigma)**2,(1,2)))
            nearflag=(ww*near[clock]).sum(1)>=.2
            # Each member has an independent, bounded phase hypothesis. No episode switching.
            cand=adv[np.arange(max(0,h-1),h+2)[:,None,None],rr[None]] # offset h-1,h,h+1
            expected=rs[cand];state_sig=ss[rr]
            absd=np.mean(((expected-qrs[target][None,:,None,:])/state_sig[None])**2,-1)
            reld=np.mean(((expected-rs[rr][None]-(qrs[target]-qrs[ix])[None,:,None,:])/state_sig[None])**2,-1)
            offsets=np.arange(max(0,h-1),h+2)
            penalty=.05*(offsets-h)**2
            def align(cost,name):
                # Monotone phase filter: hold/advance <=2, within anchor clock +/-1.
                previous=prev_phase[name][ix]
                allowed=(L.step[cand]>=L.step[previous][None])&(L.step[cand]<=L.step[previous][None]+2)
                cost=np.where(allowed,cost+penalty[:,None,None],np.inf)
                choice=np.argmin(cost,axis=0)
                cr=np.take_along_axis(cand,choice[None],0)[0]
                prev_phase[name][ix]=cr
                return cr,np.einsum('nk,nkhd->nhd',ww,act[cr,:5],optimize=True),choice
            ar,aa,ac=align(absd,'abs');dr,da,dc=align(reld,'delta');hr,ha,hc=align(.5*absd+.5*reld,'hybrid')
            serves={'awm':basea,'awm_allhit':B['stale'][target,:5],
                    'top1_clock':act[clock[:,0],:5],
                    'kernel_clock':np.einsum('nk,nkhd->nhd',ww,act[clock,:5],optimize=True),
                    'phase_abs':aa,'phase_delta':da,'phase_hybrid':ha,
                    'repeat_anchor_head':allact[ix,:5]}
            # Anchor-supported state likelihood; support never leaves the vision-selected episodes.
            selected_cost=np.take_along_axis(absd,ac[None],0)[0]
            for temp in [.25,1.0]:
                rw=ww*np.exp(-(selected_cost-selected_cost.min(1)[:,None])/temp)
                rw/=rw.sum(1)[:,None]
                serves[f'phase_reweight_{temp:g}']=np.einsum('nk,nkhd->nhd',rw,act[ar,:5],optimize=True)
            if (h+1)*5<=L.H:serves['anchor_chunk']=allact[ix,5*h:5*(h+1)]
            # The baseline is genuinely unavailable beyond its recorded horizon; do not extrapolate it.
            window_metrics={}
            for name,pred in serves.items():
                err=np.sqrt(np.mean(((pred-gt)/sigma)**2,(1,2)))
                gm=np.mean((pred[:,:,6]>=0)!=(gt[:,:,6]>=0),1)
                drift=np.sqrt(np.mean(((pred-basea)/sigma)**2,(1,2)))
                window_metrics[f'err_{name}']=err
                window_metrics[f'gm_{name}']=gm
                masks={'all':np.ones(len(ix),bool),'near_transition':nearflag,'far_transition':~nearflag,
                       **{b:bins[target]==j for j,b in enumerate(['early','mid','late'])}}
                masks.update({f'{b}_{n}':(bins[target]==j)&(nearflag if n=='near' else ~nearflag)
                              for j,b in enumerate(['early','mid','late']) for n in ['near','far']})
                for split,mask in masks.items():
                    r=summarize_values(dict(cell=cell,scale=scale,h=h,method=name,split=split),mask,err,gm,drift,baseerr)
                    if r:records.append(r)
            # Available before skipping vision, from propagated anchor members/current proprioception only.
            trans=(ww*(event[clock]|event[nxt[clock]])).sum(1)>=.2
            terminal=(ww*(L.step[clock]>=L.ep_len[clock]-2)).sum(1)>=.2
            expected_mean=np.einsum('nk,nkd->nd',ww,rs[clock],optimize=True)
            expected_delta=np.einsum('nk,nkd->nd',ww,rs[clock]-rs[rr],optimize=True)
            scale_state=np.einsum('nk,nkd->nd',ww,ss[rr],optimize=True)
            dev_abs=np.sqrt(np.mean(((qrs[target]-expected_mean)/scale_state)**2,1))
            dev_delta=np.sqrt(np.mean(((qrs[target]-qrs[ix]-expected_delta)/scale_state)**2,1))
            motionflag=still2[target]
            residual=dev_delta>.5
            union=trans|terminal|motionflag|residual
            for name,flag in [('grip_ahead',trans),('near_terminal',terminal),('still2',motionflag),
                              ('delta_dev05',residual),('absolute_dev05',dev_abs>.5),('union',union)]:
                for split,mask in [('all',np.ones(len(ix),bool)),('near_transition',nearflag),('far_transition',~nearflag),
                                   *[(b,bins[target]==j) for j,b in enumerate(['early','mid','late'])]]:
                    if mask.any():triggers.append(dict(cell=cell,scale=scale,h=h,trigger=name,split=split,n=int(mask.sum()),rate=float(flag[mask].mean())))
            for name,dd in [('delta_dev',dev_delta),('abs_dev',dev_abs)]:
                calibrations.append(dict(cell=cell,scale=scale,h=h,metric=name,q50=float(np.median(dd)),q90=float(np.percentile(dd,90)),q95=float(np.percentile(dd,95))))
            # Per-window compact trigger vectors are needed for causal schedule replay later.
            np.savez_compressed(OUT/f'windows_{cell}_{scale}_h{h}.npz',anchor=ix,target=target,
                grip=trans,terminal=terminal,still=motionflag,dev=dev_delta,absdev=dev_abs,
                phase_abs_adjust=(ww*(offsets[ac]!=h)).sum(1),phase_delta_adjust=(ww*(offsets[dc]!=h)).sum(1),
                near=nearflag,**window_metrics)
        print('analyzed',cell,scale,flush=True)
    for stem,data in [('metrics',records),('triggers',triggers),('residuals',calibrations)]:
        writecsv(OUT/f'{stem}_{key}_{scale}.csv',data)
    return key,scale

def writecsv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['anchors','analyze']);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    assert args.workers<=23
    jobs=[(k,s) for k in CELLS for s in [50,500]]
    with concurrent.futures.ProcessPoolExecutor(args.workers) as ex:
        for x in ex.map(baseline_job if args.mode=='anchors' else analysis_job,jobs):print('done',x,flush=True)

if __name__=='__main__':main()
