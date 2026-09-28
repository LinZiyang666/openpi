"""Held-out C-style main/stale geometry, action and observed-successor losses.

Both recorded streams, all task/init pairs 25..49, steps 1,6,11,... . These are
fixed-observation diagnostics, not simulator replay or estimates of success.
No missing candidate edge is imputed as an absorbing/terminal transition.
"""
import argparse
import json
import pickle
import numpy as np
from exp.offline_search.harness import store
from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r05.q4_growth.common import (
    OUT,RUN,SHM,SPEC,load_base,features,
)


def evaluate(suite):
    key=f'pi05_{suite}';short='sp' if suite=='spatial' else suite
    base=store.LibraryView(SHM,key);grown=store.LibraryView(SHM,key,'grow250');big=store.LibraryView(SHM,key,'bpool_cs')
    models={'static50':load_base(suite),'library500':load_base(suite,500)}
    load_method_class(SPEC)
    for variant in ('frozen','refit'):
        with (RUN/'fits'/f'r5q4_p_{short}_grow250_{variant}.pkl').open('rb') as f:
            models['grown_'+variant]=pickle.load(f)['method']
    libs={'static50':base,'grown_frozen':grown,'grown_refit':grown,'library500':big}
    # Common density frame: every library's rows go through the identical
    # deployed 50 fit (including old rows), as in C/library_study.py.
    common_x={name:features(models['static50'],lib) for name,lib in
              (('static50',base),('grown_frozen',grown),('library500',big))}
    common_x['grown_refit']=common_x['grown_frozen']
    records=[]
    for stream in ('inf','cache'):
        Q=store.QueryCell(SHM,key+'_'+stream)
        eps=[e for e in Q.episodes if 25<=e['init']<=49]
        assert len(eps)==250 and {(e['task_id'],e['init']) for e in eps}=={(t,i) for t in range(10) for i in range(25,50)}
        qr=np.concatenate([np.arange(e['start']+1,e['end'],5) for e in eps])
        qt=np.array([Q.episodes[e]['task_id'] for e in Q.ep[qr]])
        has_next=np.array([r+1<Q.episodes[int(Q.ep[r])]['end'] and Q.step[r+1]==Q.step[r]+1 for r in qr])
        qc=features(models['static50'],Q,qr)
        gt=np.asarray(Q.a_inf[qr,:5,:7],np.float64)
        all_arrays={}
        for name in ('static50','grown_frozen','grown_refit','library500'):
            M=models[name];L=libs[name];xx=qc if name in ('static50','grown_frozen') else features(M,Q,qr)
            out={k:np.full(len(qr),np.nan) for k in ('d1','d16','common_d1','common_d16','action_rms','successor_rms','observed_edge_mass','new_top1','new_top16')}
            for t,T in M.tasks.items():
                ix=np.flatnonzero(qt==t);rr=T.rows
                z=T.Z.astype(np.float64);qz=(xx[ix]@T.Wf-T.shift).astype(np.float64)
                BT=models['static50'].tasks[t]
                cz=(common_x[name][rr]@BT.Wf-BT.shift).astype(np.float64)
                cq=(qc[ix]@BT.Wf-BT.shift).astype(np.float64)
                state_sd=np.maximum(np.asarray(base.rs[base.rows_of_task(t),:8],np.float64).std(0),.05)
                edges=np.asarray(L.next[rr])>=0
                delta=np.zeros((len(rr),8),np.float64)
                delta[edges]=(L.rs[L.next[rr[edges]],:8]-L.rs[rr[edges],:8])/state_sd
                for lo in range(0,len(ix),128):
                    dst=ix[lo:lo+128];qq=qz[lo:lo+128]
                    dd=np.sqrt(np.maximum((qq*qq).sum(1)[:,None]+(z*z).sum(1)[None]-2*qq@z.T,0))
                    order=np.argsort(dd,axis=1,kind='stable')[:,:16]
                    ds=np.take_along_axis(dd,order,axis=1);w=_kernel_w(ds-ds[:,:1],M.kref);w/=w.sum(1)[:,None]
                    selected=rr[order]
                    pred=np.einsum('qk,qktd->qtd',w,L.action[selected,:5,:7].astype(np.float64))
                    out['d1'][dst]=ds[:,0];out['d16'][dst]=ds[:,15]
                    out['action_rms'][dst]=np.sqrt(np.mean(((pred-gt[dst])/models['static50'].sig)**2,axis=(1,2)))
                    mass=(w*edges[order]).sum(1);out['observed_edge_mass'][dst]=mass
                    wn=w*edges[order]/np.maximum(mass[:,None],1e-300)
                    dp=np.einsum('qk,qkd->qd',wn,delta[order])
                    ok=has_next[dst]&(mass>1e-12)
                    true_delta=(Q.rs[qr[dst[ok]]+1,:8]-Q.rs[qr[dst[ok]],:8])/state_sd
                    out['successor_rms'][dst[ok]]=np.sqrt(np.mean((dp[ok]-true_delta)**2,axis=1))
                    if name.startswith('grown'):
                        out['new_top1'][dst]=selected[:,0]>=base.L
                        out['new_top16'][dst]=(selected>=base.L).mean(1)
                    else:
                        out['new_top1'][dst]=out['new_top16'][dst]=0
                    cqq=cq[lo:lo+128]
                    cdd=np.sqrt(np.maximum((cqq*cqq).sum(1)[:,None]+(cz*cz).sum(1)[None]-2*cqq@cz.T,0))
                    cd=np.sort(cdd,axis=1)[:,[0,15]]
                    out['common_d1'][dst]=cd[:,0];out['common_d16'][dst]=cd[:,1]
            all_arrays[name]=out
        common_ok=has_next.copy()
        for out in all_arrays.values():common_ok &= np.isfinite(out['successor_rms'])
        for name,out in all_arrays.items():
            out['successor_rms'][~common_ok]=np.nan
            row=dict(suite=suite,stream=stream,variant=name,episodes=250,queries=len(qr),
                     successor_queries=int(common_ok.sum()),kref=models[name].kref,rows=libs[name].L)
            row.update({k:float(np.nanmean(v)) for k,v in out.items()})
            # Equal-episode averages expose decision-count weighting; no outcome filtering.
            for k in ('action_rms','successor_rms'):
                em=[np.nanmean(out[k][Q.ep[qr]==eid]) for eid in np.unique(Q.ep[qr]) if np.isfinite(out[k][Q.ep[qr]==eid]).any()]
                row[k+'_episode_mean']=float(np.mean(em))
            np.savez_compressed(OUT/'results'/f'offline_{suite}_{stream}_{name}.npz',query_row=qr,
                                episode=Q.ep[qr],task=qt,has_next=has_next,common_successor_mask=common_ok,**out)
            records.append(row)
            print(json.dumps(row),flush=True)
    report=dict(suite=suite,protocol='C-style main/stale metric only; steps 1,6,11,...; heldout init25..49; all outcomes',
                action='RMS of full top16 kernel head versus recorded a_inf; sigma from current library; first 5 steps x 7 dims',
                successor='RMS of kernel-weighted observed library rs[next]-rs versus observed query rs[next]-rs; per-task current std floored at .05; valid state8; renormalize only observed edges; identical complete-case queries across variants',
                density='d1/d16 are own-fit units, not comparable across refits; common_d1/common_d16 use the same deployed 50 PCA+metric for all rows',
                no_success_claim=True,records=records)
    (OUT/'results'/f'offline_{suite}.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True,choices=['l10','spatial']);a=p.parse_args();evaluate(a.suite)
