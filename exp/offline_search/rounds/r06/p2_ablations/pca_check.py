"""Recorded-query parity, learned-metric refit, held-out descriptive metrics, CPU projection timing."""
import argparse
import copy
import contextlib
import json
import os
import pickle
import resource
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.harness import api, dims, store
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r04.k1_blind.checks import view, blind_view
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, RUN, STORE, artifact, sources
from exp.offline_search.rounds.r06.p2_ablations.token_pca import TokenPCAAWM, pool_tokens


class CacheView:
    def __init__(self,q):
        self.q=q
        self.prev_hit=True if q.step else None
    def __getattr__(self,k): return getattr(self.q,k)


def load(path):
    with open(path,'rb') as f: return pickle.load(f)['method']


def payload_equal(a,b):
    for k in ('topk','scores','action'):
        assert np.array_equal(getattr(a,k),getattr(b,k)),k
    assert a.confidence==b.confidence and a.library==b.library
    assert {k:v for k,v in a.extras.items() if k!='still'}=={k:v for k,v in b.extras.items() if k!='still'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--arm',required=True); a=p.parse_args()
    row,=[r for r in json.loads((HERE/'arms_pca.json').read_text()) if r['name']==a.arm]
    m=load(artifact(row).replace('<RUN>',str(RUN)))
    scale=50 if m.lib=='current' else 500
    src=sources()[row['model'],row['suite'],scale,'A'][1]
    A=load(artifact(src))
    control,_=clone_method(m); control.__class__=BlindAWM
    grid4,_=clone_method(A); grid4.__class__=TokenPCAAWM; grid4.pooling_grid=4; grid4._previous_tokens=None
    for key in ('act','sig','lib_ep','lib_step','blind_rs','blind_next','blind_event','blind_terminal'):
        assert np.array_equal(getattr(m,key),getattr(A,key)),key
    for key in ('features','k','kref','early','codes','lam','nn','state_scale','feat0','lam_c','norm_cap','hyst',
                'serving','budget','gates','lib','fit_data'):
        assert getattr(m,key)==getattr(A,key),key
    lib=store.LibraryView(STORE,f"{row['model']}_{row['suite']}",m.cand_name)
    root=Path(m.pca_cache)/'grid16'/f"{row['model']}_{row['suite']}"/m.cand_name
    P=[np.load(root/v/'proj.npy') for v in ('v0','v1')]
    rs=np.asarray(dims.valid_state(lib.rs,m.model),np.float64)
    heads=(np.asarray(lib.action[:,:5,:7],np.float64)/m.sig.astype(np.float64)).reshape(lib.L,35)
    # Use the identical unrounded action_sigma from Context, as stock fit does.
    sigma=np.asarray(store.action_sigma(STORE,f"{row['model']}_{row['suite']}"),np.float64)
    heads=(np.asarray(lib.action[:,:5,:7],np.float64)/sigma).reshape(lib.L,35)
    metric_checks=0
    for t,T in m.tasks.items():
        rr=T.rows
        X=np.concatenate([P[0][rr],P[1][rr],rs[rr]],axis=1).astype(np.float64)
        assert np.array_equal(X[:,-8:],rs[rr]) and np.array_equal(T.RS,A.tasks[t].RS)
        for early in (False,True):
            selected=np.asarray(lib.step)[rr]<=2 if early else np.ones(len(rr),bool)
            xx=X[selected]; ss=rs[rr][selected]
            assert np.array_equal(xx.mean(0)[-8:],ss.mean(0))
            assert np.array_equal(xx.std(0)[-8:]+1e-6,ss.std(0)+1e-6)
            mean,std,W=awm.fit_metric(xx,heads[rr][selected],np.asarray(lib.episode)[rr][selected],
                                       nn=m.nn,lam=m.lam,rank=m.codes)
            assert np.array_equal((W/std[:,None]).astype(np.float32),T.W0f if early else T.Wf)
            metric_checks+=1
    reports=[]; total=0; tails=0; timing_query=None
    for stream in ('cache','inf'):
        qc=store.QueryCell(STORE,f"{row['model']}_{row['suite']}_{stream}")
        arrays=api.QueryArrays(qc)
        # Every token-subsample episode start plus 300 evenly spaced token rows.
        token_rows=np.asarray(qc.tok_rows)
        ids=sorted(set(token_rows[np.linspace(0,len(token_rows)-1,300,dtype=int)].tolist()+
                       [e['start'] for e in qc.episodes if qc.tok_index[e['start']]>=0]))
        overlaps=[]; newerr=[]; olderr=[]; newmae=[]; oldmae=[]
        for i in ids:
            q=CacheView(view(qc,arrays,int(i)))
            for x in (m,A,control,grid4): x.reset(q.episode)
            tq=m._token_query(q)
            for k in ('rs','hist_rs','raw_state','hist_a_exec','hist_hit'):
                assert np.array_equal(getattr(tq,k),getattr(q,k))
            got=m.query(q); ref=control.query(tq); old=A.query(q); same=grid4.query(q)
            payload_equal(got,ref); payload_equal(old,same)
            assert got.extras['regime']==(0. if q.step==0 else 2.)
            overlaps.append(len(set(got.topk)&set(old.topk))/16.)
            target=np.asarray(qc.a_inf[i,:5,:7],np.float64)
            for result,errs,maes in ((got,newerr,newmae),(old,olderr,oldmae)):
                diff=(result.action[:5,:7].astype(np.float64)-target)/sigma
                errs.append(float(np.sqrt(np.mean(diff**2))))
                maes.append(float(np.mean(np.abs(diff))))
            if i+1<qc.N and qc.ep[i+1]==qc.ep[i]:
                qn=view(qc,arrays,int(i+1)); bq=blind_view(qn,prev_hit=True)
                # This check follows an anchor at q.step, so only its last-step continuity matters.
                bq.blind_age=0
                br=m.blind_step(bq); cr=control.blind_step(bq)
                assert type(br) is type(cr)
                assert np.array_equal(br.action,cr.action)
                assert np.array_equal(br.action[:5,:7],got.action[5:10,:7])
                tails+=1
            total+=1; timing_query=q
        reports.append(dict(stream=stream,queries=len(ids),mean_top16_overlap=float(np.mean(overlaps)),
                            mean_sigma_RMS_action_error_direct=float(np.mean(newerr)),
                            mean_sigma_RMS_action_error_A=float(np.mean(olderr)),
                            mean_sigma_MAE_direct=float(np.mean(newmae)),mean_sigma_MAE_A=float(np.mean(oldmae)),
                            query_rows=ids))
    timings={}
    q=timing_query; k0=pool_tokens(q.tok_v0,16); k1=pool_tokens(q.tok_v1,16)
    for threads in (int(os.environ['OPENBLAS_NUM_THREADS']),):
        with contextlib.nullcontext():
            measurements=[]; baseline=[]; complete=[]
            for _ in range(3): m.B0T@k0; m.B1T@k1
            for _ in range(30):
                t=time.perf_counter(); m.B0T@k0-m.muB0; m.B1T@k1-m.muB1
                measurements.append((time.perf_counter()-t)*1e3)
                t=time.perf_counter(); A.B0T@q.key_v0-A.muB0; A.B1T@q.key_v1-A.muB1
                baseline.append((time.perf_counter()-t)*1e3)
                m.reset(q.episode)
                t=time.perf_counter(); m.query(q); complete.append((time.perf_counter()-t)*1e3)
            timings[str(threads)]=dict(repetitions=30,two_camera_projection_median_ms=float(np.median(measurements)),
                                       two_camera_projection_p95_ms=float(np.percentile(measurements,95)),
                                       A_two_camera_projection_median_ms=float(np.median(baseline)),
                                       full_anchor_query_median_ms=float(np.median(complete)))
    out=dict(PASS=True,arm=a.arm,recorded_queries=total,identical_stock_visual_input_code_path=True,
             grid4_control_payloads_identical_to_A=total,state_block_exact=True,learned_metric_refit_checks=metric_checks,
             blind_tail_checks=tails,descriptive=reports,cpu_timings=timings,
             projection_matrix_bytes_per_camera=int(m.B0T.nbytes),pca=m.token_pca_info,
             process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             GPU='not measured: task requires CUDA_VISIBLE_DEVICES empty; installed GPU path consumes pooled keys')
    (HERE/'results/pca'/f'{a.arm}_check.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('pca','descriptive')}),flush=True)


if __name__=='__main__': main()
