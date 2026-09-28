"""Frozen-fit growth proxy, full-top16 pruning and scene-prior diagnostics.

Growth source = actual full-inference store trajectories, NOT reconstructed mixed
MISS keys. All source decisions cost IR=1; successes published after completion.
Training init 0..24; evaluation init 25..49. Main/stale metric density only.
No SR estimate is produced. Existing R4-A exact-ish full16 replays are reused
with their recorded scalar parity tolerances, never labelled closed-loop logs.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json, pickle, argparse, hashlib
import numpy as np
from exp.offline_search.harness import api, store
from exp.offline_search.closed_loop.plugin import load_method_class

OUT=Path(__file__).resolve().parent
ROOT=Path('/home/weiland/trace_runs/offline_search_store')
RUNS=Path('/home/weiland/trace_runs/os_closed_loop')
OLD=OUT.parents[1]/'r04'/'ideation_A'
CELLS=['pi05_spatial','pi05_l10','groot_spatial','groot_l10']

def lib(key,scale):
    return store.LibraryView(ROOT,key,'current' if scale==50 else ('bpool_cs' if key.startswith('pi05') else 'bpool_all'))

def method(key,scale):
    m,s=key.split('_');arm=f'oscl{scale}_{"p" if m=="pi05" else "g"}_{"sp" if s=="spatial" else s}_cl2'
    p=RUNS/f'r02_g{scale}'/'fits'/f'{arm}.pkl'
    load_method_class('exp/offline_search/rounds/r02/g1_awm/awm.py:AWM')
    M=pickle.load(p.open('rb'))['method'];M.prof=api.NULL_PROFILER
    return M,p

def feats(M,obj,ids=None):
    if ids is None:ids=np.arange(len(obj.rs))
    x=np.empty((len(ids),136),np.float32)
    for st in range(0,len(ids),128):
        jj=ids[st:st+128]
        x[st:st+len(jj),:64]=np.asarray(obj.key_v0[jj],np.float32)@M.B0T.T-M.muB0
        x[st:st+len(jj),64:128]=np.asarray(obj.key_v1[jj],np.float32)@M.B1T.T-M.muB1
        x[st:st+len(jj),128:]=obj.rs[jj,:8]
    return x

def d2(q,z):
    return np.maximum((q*q).sum(1)[:,None]+(z*z).sum(1)[None]-2*q@z.T,0)

def density(q,z,base_n):
    vals=[];new1=[];new16=[]
    for st in range(0,len(q),128):
        dd=d2(q[st:st+128],z);ii=np.argsort(dd,axis=1,kind='stable')[:,:16]
        vals.append(np.sqrt(np.take_along_axis(dd,ii[:,[0,15]],axis=1)))
        new1.append(ii[:,0]>=base_n);new16.append((ii>=base_n).mean(1))
    return np.concatenate(vals),np.concatenate(new1),np.concatenate(new16)

def coverage(top,w,mask):
    a=mask[top]
    return dict(n=len(top),all16=float(a.all(1).mean()),top1=float(a[:,0].mean()),weight_retained=float((w*a).sum(1).mean()))

def study(job):
    key,scale=job;M,artifact=method(key,scale);L=lib(key,scale);Q=store.QueryCell(ROOT,key+'_inf')
    source_eps=sorted([e for e in Q.episodes if e['init']<25],key=lambda e:(e['init'],e['task_id']))
    source_ids=np.concatenate([np.arange(e['start'],e['end']) for e in source_eps])
    source_x=feats(M,Q,source_ids);srcmap=np.full(Q.N,-1,np.int32);srcmap[source_ids]=np.arange(len(source_ids))
    query={}
    for arm in ('inf','cache'):
        qc=store.QueryCell(ROOT,key+'_'+arm)
        ids=np.concatenate([np.arange(e['start']+1,e['end'],5) for e in qc.episodes if e['init']>=25])
        query[arm]=(qc,ids,feats(M,qc,ids),np.array([e['task_id'] for e in qc.episodes])[qc.ep[ids]])
    big=lib(key,500);bx=feats(M,big) if scale==50 else None
    result=dict(cell=key,scale=scale,L=L.L,artifact=str(artifact),artifact_bytes=artifact.stat().st_size,
                source='queries/*_inf, train init 0..24; full policy, successful episode commit',
                fit='own deployed library, frozen; big-candidate density reference is diagnostic only',growth=[],pruning=[],scene=[])
    zs={};acts={};states={};radii={};mt={};basez={};testz={};bigz={};base_stats={}
    for t,T in M.tasks.items():
        rows=T.rows;z=T.Z.astype(np.float64);basez[t]=z;zs[t]=list(z)
        acts[t]=list(np.asarray(L.action[rows,:5,:7]/M.sig,np.float64).reshape(-1,35));states[t]=list(np.asarray(L.rs[rows,:8],np.float64))
        dd=d2(z,z);dd[L.episode[rows,None]==L.episode[rows][None,:]]=np.inf
        radii[t]=float(np.quantile(np.sqrt(dd.min(1)),.1))
        good=rows[L.next[rows]>=0];mt[t]=float(np.quantile(np.linalg.norm(L.rs[L.next[good],:8]-L.rs[good,:8],axis=1),.1))
        for arm,(qc,ids,xx,tids) in query.items():testz[arm,t]=(xx[tids==t]@T.Wf-T.shift).astype(np.float64)
        if scale==50:
            br=big.rows_of_task(t);bigz[t]=(bx[br]@T.Wf-T.shift).astype(np.float64)
        else:bigz[t]=z
    totals=dict(policy_calls=0,successful_policy_rows=0,admitted=0,dedup=0,failed_episode_rows=0)
    for n,e in enumerate([None]+source_eps):
        if e is not None:
            ids=np.arange(e['start'],e['end']);totals['policy_calls']+=len(ids)
            if not e['success']:totals['failed_episode_rows']+=len(ids)
            else:
                totals['successful_policy_rows']+=len(ids);t=e['task_id'];T=M.tasks[t]
                zz=(source_x[srcmap[ids]]@T.Wf-T.shift).astype(np.float64)
                aa=np.asarray(Q.a_exec[ids,:5,:7]/M.sig,np.float64).reshape(-1,35);ss=np.asarray(Q.rs[ids,:8],np.float64)
                # Dedup within the just-completed episode too; nothing published before completion.
                for z,a,s in zip(zz,aa,ss):
                    known=np.asarray(zs[t]);dd=np.sum((known-z)**2,axis=1);j=int(np.argmin(dd))
                    dup=(dd[j] <= radii[t]**2 and np.sqrt(np.mean((acts[t][j]-a)**2))<=.1 and np.linalg.norm(states[t][j]-s)<=mt[t])
                    if dup:totals['dedup']+=1
                    else:zs[t].append(z);acts[t].append(a);states[t].append(s);totals['admitted']+=1
        if n not in (0,50,100,250):continue
        rr=dict(episodes=n,**totals,by_stream={})
        for arm in query:
            res=[];br=[];new1=[];new16=[]
            for t in M.tasks:
                a,b,c=density(testz[arm,t],np.asarray(zs[t]),len(basez[t]));res.append(a);new1.append(b);new16.append(c)
                if n==0:br.append(density(testz[arm,t],bigz[t],len(bigz[t]))[0])
            val=np.concatenate(res)
            if n==0:base_stats[arm]=(val,np.concatenate(br))
            base,reference=base_stats[arm];den=base.mean(0)-reference.mean(0)
            rr['by_stream'][arm]=dict(heldout_queries=len(val),mean_d1=float(val[:,0].mean()),mean_d16=float(val[:,1].mean()),
                relative_d1=float(val[:,0].mean()/base[:,0].mean()),relative_d16=float(val[:,1].mean()/base[:,1].mean()),
                big_reference_d1=float(reference[:,0].mean()),big_reference_d16=float(reference[:,1].mean()),
                gap_closed_d1=float((base[:,0].mean()-val[:,0].mean())/den[0]) if scale==50 and abs(den[0])>1e-9 else None,
                gap_closed_d16=float((base[:,1].mean()-val[:,1].mean())/den[1]) if scale==50 and abs(den[1])>1e-9 else None,
                new_top1_share=float(np.concatenate(new1).mean()),new_member_share=float(np.concatenate(new16).mean()))
        result['growth'].append(rr)
    # Full16 support diagnostic, using independently evaluated old replay arrays.
    freq=np.zeros(L.L,np.int64);all_used=np.zeros(L.L,bool);test=[]
    for arm in ('inf','cache'):
        qc=store.QueryCell(ROOT,key+'_'+arm);a=np.load(OLD/f'anchors_{key}_{arm}_{scale}.npz')
        top=a['rows'];weights=a['weights'];init=np.array([e['init'] for e in qc.episodes])[qc.ep]
        np.add.at(freq,top[init<25].ravel(),1);all_used[top.ravel()]=True
        test.append((arm,top[init>=25],weights[init>=25]))
        # Scene prior uses only actual step0 vision, first library rows, task-local early metric.
        eps=[e for e in qc.episodes if e['init']>=25];x0=feats(M,qc,np.array([e['start'] for e in eps]))
        for frac in (.25,.5,.75,1.):
            acc=defaultdict_list();retained=[]
            for e,x in zip(eps,x0):
                t=e['task_id'];T=M.tasks[t];starts=np.array([u['start'] for u in L.episodes if u['task_id']==t])
                pos=np.searchsorted(T.rows,starts);y=x@T.W0f-T.c0
                dd=T.n20[pos]-2*T.Z[pos]@(T.A0@y)+float(y@y)
                order=starts[np.argsort(dd,kind='stable')];count=max(1,int(np.ceil(frac*len(order))))
                mask=np.isin(L.episode,L.episode[order[:count]])
                ix=np.arange(e['start']+1,e['end']);cc=coverage(top[ix],weights[ix],mask)
                for k,v in cc.items():acc[k].append(v)
                retained.append(float(mask[L.task_id==t].mean()))
            result['scene'].append(dict(stream=arm,fraction=frac,episodes=len(eps),mean_task_rows_retained=float(np.mean(retained)),
               all16_episode_mean=float(np.mean(acc['all16'])),top1_episode_mean=float(np.mean(acc['top1'])),weight_retained_episode_mean=float(np.mean(acc['weight_retained']))))
    for frac in (.25,.5,.75,.9,1.):
        mask=np.zeros(L.L,bool)
        for t in L.tasks():
            rr=L.rows_of_task(t);order=rr[np.lexsort((rr,-freq[rr]))];mask[order[:max(16,int(np.ceil(len(rr)*frac)))]]=True
        for arm,top,weights in test:
            result['pruning'].append(dict(fraction=frac,stream=arm,retained=int(mask.sum()),**coverage(top,weights,mask)))
    train=freq>0
    result['full16_union']=int(all_used.sum());result['full16_train_union']=int(train.sum())
    result['full16_heldout_union_coverage']={arm:coverage(top,w,train) for arm,top,w in test}
    result['dedup_thresholds']={str(t):dict(code10=radii[t],state10=mt[t],head_rms=.1) for t in M.tasks}
    result['source_big_same_file_overlap']=len({e['file'] for e in Q.episodes}&{e['file'] for e in big.episodes})
    (OUT/f'library_{key}_{scale}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(cell=key,scale=scale,growth=result['growth'],full16_union=result['full16_union'],full16_train_union=result['full16_train_union'])),flush=True)
    return key,scale

def defaultdict_list():
    from collections import defaultdict
    return defaultdict(list)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(study,[(c,s) for c in CELLS for s in (50,500)]))
