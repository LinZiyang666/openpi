"""Full held-out A-pool, fixed-observation demo size curves; no SR estimates."""
import argparse
import json
import pickle
import time
import numpy as np
from scipy.spatial.distance import cdist
from exp.offline_search.harness import api,store
from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r05.q4_growth.common import OUT,SHM,features,load_base,sha256
from exp.offline_search.rounds.r05.q4_growth.demo_method import DemoAWM,features,stable_linear
from exp.offline_search.rounds.r05.q4_growth.demo_prepare import RUN,SPEC,arm_name

RESULTS=OUT/'results'/'demo'
SIZES=(50,100,200,300,500)

def library_name(size):return 'current' if size==50 else 'bpool_cs' if size==500 else f'demo{size}'

def get_model(suite,size,variant):
    if (variant=='refit' and size in (100,200,300)) or (variant=='frozen50' and size==200):
        p=RUN/'fits'/f'{arm_name(suite,size,variant)}.pkl'
        load_method_class(SPEC)
        with p.open('rb') as f:M=pickle.load(f)['method']
        return M,dict(artifact=str(p),sha256=sha256(p))
    ctx=api.Context(root=SHM,cell=f'pi05_{suite}_cache',seed=0,scratch=RUN/'offline_scratch')
    M=DemoAWM(library=library_name(size),variant=variant,kref=5);M.prof=api.NULL_PROFILER
    started=time.time();M.fit(ctx.open_library('current'),ctx)
    return M,dict(fit_in_process=True,fit_seconds=time.time()-started,kwargs=dict(library=library_name(size),variant=variant,kref=5))

def evaluate(suite):
    key=f'pi05_{suite}';base=store.LibraryView(SHM,key);F=load_base(suite)
    libs={n:store.LibraryView(SHM,key,library_name(n)) for n in SIZES}
    common_x={n:features(F,L) for n,L in libs.items()}
    queries={}
    for stream in ('inf','cache'):
        Q=store.QueryCell(SHM,key+'_'+stream)
        assert len(Q.episodes)==500 and {(e['task_id'],e['init']) for e in Q.episodes}=={(t,i) for t in range(10) for i in range(50)}
        ids=np.concatenate([np.arange(e['start']+1,e['end'],5) for e in Q.episodes])
        task=np.array([Q.episodes[e]['task_id'] for e in Q.ep[ids]])
        has_next=np.array([r+1<Q.episodes[int(Q.ep[r])]['end'] and Q.step[r+1]==Q.step[r]+1 for r in ids])
        queries[stream]=(Q,ids,task,has_next,features(F,Q,ids))
    all_arrays={};fit_sources=[]
    for n in SIZES:
        for variant in ('refit','frozen50'):
            M,fit_info=get_model(suite,n,variant);M.prof=api.NULL_PROFILER;L=libs[n]
            fit_sources.append(dict(size=n,variant=variant,source=fit_info))
            for stream,(Q,qr,qt,has_next,qc) in queries.items():
                xx=qc if variant=='frozen50' or n==50 else features(M,Q,qr)
                gt=np.asarray(Q.a_inf[qr,:5,:7],np.float64)
                out={k:np.full(len(qr),np.nan) for k in ('d1','d16','common_d1','common_d16','action_rms','successor_rms','observed_edge_mass','tail_action_rms')}
                for t,T in M.tasks.items():
                    ix=np.flatnonzero(qt==t);rr=T.rows
                    z=T.Z.astype(np.float64);qz=stable_linear(xx[ix],T.Wf,T.shift).astype(np.float64)
                    BT=F.tasks[t];cz=stable_linear(common_x[n][rr],BT.Wf,BT.shift).astype(np.float64);cq=stable_linear(qc[ix],BT.Wf,BT.shift).astype(np.float64)
                    sd=np.maximum(np.asarray(base.rs[base.rows_of_task(t),:8],np.float64).std(0),.05)
                    edge=np.asarray(L.next[rr])>=0
                    delta=np.zeros((len(rr),8),np.float64);delta[edge]=(L.rs[L.next[rr[edge]],:8]-L.rs[rr[edge],:8])/sd
                    for lo in range(0,len(ix),128):
                        dst=ix[lo:lo+128];qq=qz[lo:lo+128]
                        d=np.sqrt(np.maximum((qq*qq).sum(1)[:,None]+(z*z).sum(1)[None]-2*qq@z.T,0))
                        order=np.argsort(d,axis=1,kind='stable')[:,:16]
                        ds=np.take_along_axis(d,order,axis=1);w=_kernel_w(ds-ds[:,:1],5);w/=w.sum(1)[:,None]
                        selected=rr[order];pred=np.einsum('qk,qktd->qtd',w,L.action[selected,:,:7].astype(np.float64))
                        out['d1'][dst]=ds[:,0];out['d16'][dst]=ds[:,15]
                        out['action_rms'][dst]=np.sqrt(np.mean(((pred[:,:5]-gt[dst])/F.sig)**2,axis=(1,2)))
                        mass=(w*edge[order]).sum(1);out['observed_edge_mass'][dst]=mass
                        wn=w*edge[order]/np.maximum(mass[:,None],1e-300);dp=np.einsum('qk,qkd->qd',wn,delta[order])
                        ok=has_next[dst]&(mass>1e-12)
                        true_delta=(Q.rs[qr[dst[ok]]+1,:8]-Q.rs[qr[dst[ok]],:8])/sd
                        out['successor_rms'][dst[ok]]=np.sqrt(np.mean((dp[ok]-true_delta)**2,axis=1))
                        nxt=has_next[dst]
                        tail_gt=Q.a_inf[qr[dst[nxt]]+1,:5,:7]
                        out['tail_action_rms'][dst[nxt]]=np.sqrt(np.mean(((pred[nxt,5:10]-tail_gt)/F.sig)**2,axis=(1,2)))
                        cqq=cq[lo:lo+128]
                        cd=cdist(cqq,cz,metric='euclidean')
                        near=np.partition(cd,15,axis=1)[:,:16]
                        out['common_d1'][dst]=near.min(1);out['common_d16'][dst]=near.max(1)
                all_arrays[stream,n,variant]=out
            print(json.dumps(dict(suite=suite,size=n,variant=variant,computed=True)),flush=True)
    records=[]
    for stream,(Q,qr,qt,has_next,qc) in queries.items():
        ok=has_next.copy()
        for n in SIZES:
            for variant in ('refit','frozen50'):ok &= np.isfinite(all_arrays[stream,n,variant]['successor_rms'])
        for n in SIZES:
            for variant in ('refit','frozen50'):
                a=all_arrays[stream,n,variant];a['successor_rms'][~ok]=np.nan
                r=dict(suite=suite,stream=stream,size=n,variant=variant,rows=libs[n].L,episodes=500,
                       queries=len(qr),successor_queries=int(ok.sum()),tail_queries=int(has_next.sum()),kref=5)
                r.update({k:float(np.nanmean(v)) for k,v in a.items()})
                for k in ('action_rms','successor_rms'):
                    em=[np.nanmean(a[k][Q.ep[qr]==eid]) for eid in np.unique(Q.ep[qr]) if np.isfinite(a[k][Q.ep[qr]==eid]).any()]
                    r[k+'_episode_mean']=float(np.mean(em))
                records.append(r)
                np.savez_compressed(RESULTS/f'curve_{suite}_{stream}_{n}_{variant}.npz',query_row=qr,episode=Q.ep[qr],task=qt,
                                    has_next=has_next,common_successor_mask=ok,**a)
                print(json.dumps(r),flush=True)
        # 50 is the same saved deployed model for both labels; no fake refit effect.
        for k in all_arrays[stream,50,'refit']:
            assert np.array_equal(all_arrays[stream,50,'refit'][k],all_arrays[stream,50,'frozen50'][k],equal_nan=True)
        # In a shared geometry, nested candidate density cannot get worse.
        for n,larger in zip((100,200,300),(200,300,500)):
            for k in ('common_d1','common_d16'):
                assert np.all(all_arrays[stream,larger,'frozen50'][k]<=all_arrays[stream,n,'frozen50'][k]+1e-5)
    report=dict(suite=suite,protocol='all 500 A-pool episodes per stream, all outcomes; steps1,6,11,...; main/stale geometry',
                nesting='100<200<300<500; deployed50 is a separate noncontained collection',
                kref_rule='fixed5 every size and variant; 500 refit calibration recomputed at5, unlike historical R2 kref8',
                loss_definition='same action/successor definitions as policy-growth offline study; common complete-case successor mask across all10 size/variant cells per stream; no missing edge imputation',
                tail_loss='extra fixed-observation one-decision-ahead kernel steps5:10 vs next recorded policy head; not closed-loop rollout',
                density='common_d1/d16 uses deployed50 geometry for all candidates; native d1/d16 not comparable across fitted metrics',
                fit_sources=fit_sources,records=records,no_SR_claim=True,nested_density_monotonic=True)
    (RESULTS/f'curve_{suite}.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True,choices=['l10','spatial']);a=p.parse_args();evaluate(a.suite)
