"""Identity-first round6 log diagnosis and fixed-path offline diagnosis."""
import json
from collections import Counter
import numpy as np
from .data import *
from exp.offline_search.rounds.r09.explore_astra.round6.tools.learn import load_control, control_prediction, library
from exp.offline_search.rounds.r09.explore_astra.round6.tools.numeric import residual, dense_predict


OWNER = json.loads((REPO/'exp/libero_groot/config/rit/cost_groot_libero_measured.json').read_text())
LOOK = OWNER['stage1_ms'] / sum(OWNER[k] for k in ('stage1_ms','stage2_ms','stage3_full_loop_ms'))


def summarize(rows):
    n=len(rows); look=sum(bool(r['vision']) for r in rows); call=sum(not r['hit'] for r in rows)
    def avg(key):
        a=[r.get('extras',{}).get(key) for r in rows if r['vision']]
        a=[x for x in a if x is not None and np.isfinite(x)]
        return float(np.mean(a)) if a else None
    return dict(n=n,look=look,calls=call,ir=(LOOK*look+(1-LOOK)*call)/n,
        dnn=avg('dnn'),lag=avg('lag'),noprog=avg('noprog_n'),
        reasons=dict(Counter(str(r.get('extras',{}).get('os_reason')) for r in rows if not r['hit'])))


def live(cell):
    lib=library(cell)
    arms={}; outcomes={}; eps={}
    for variant in ('control','taskfree'):
        arm=f'r9a6_{cell}_{variant}'; root=RUN.parent/'r09_astra_r6/runs'/arm
        accepted={}
        for r in admitted_jsonl(root/'client/journal.jsonl'):
            if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error'):
                key=tuple(map(int,r['task_uid'].rsplit(':',2)[-2:])); accepted[key]=r
        assert len(accepted)==100
        groups={key:[] for key in accepted}
        for path in sorted(root.glob('server_*/decisions_*.jsonl')):
            for r in admitted_jsonl(path):
                key=(r['task_id'],r['init'])
                if r.get('ev')=='dec' and r.get('attempt')==accepted[key]['attempt']: groups[key].append(r)
        for key,rr in groups.items():
            rr.sort(key=lambda r:r['step'])
            assert [r['step'] for r in rr]==list(range(len(rr))), (arm,key)
            served=np.array([r['served_head'] for r in rr]); closed=(served[:,:,6]<0).mean(1)>.5
            phase=phases(closed,np.zeros(len(rr)),np.arange(len(rr)))
            for j,r in enumerate(rr):
                r['_phase']=str(phase[j]); r['_closed']=bool(closed[j]); r['_event']=bool(j and closed[j]!=closed[j-1])
                r['_next_call']=any(not a['hit'] for a in rr[j+1:j+5])
                if r['src']=='cache' and r['vision']:
                    rows=np.array(r['rows']); w=np.array(r['weights']); w/=w.sum()
                    b=np.einsum('k,kha->ha',w,lib[rows,:5,:7]); a=np.array(r['served_head'])
                    r['_corr_rms']=float(np.sqrt(np.mean((a[:,:6]-b[:,:6])**2)))
                    r['_grip_difference']=float(np.max(np.abs(a[:,6]-b[:,6])))
        rr=[r for group in groups.values() for r in group]
        def strata(key): return {v:summarize([r for r in rr if r[key]==v]) for v in sorted(set(r[key] for r in rr))}
        cc=[r for r in rr if '_corr_rms' in r]
        if variant=='control': edges=np.quantile([r['extras']['dnn'] for r in rr if r['vision']],[.5,.9])
        arms[variant]=dict(total=summarize(rr),success=sum(r['success'] for r in accepted.values()),
            by_task={t:dict(summarize([r for r in rr if r['task_id']==t]),success=sum(v['success'] for (tt,i),v in accepted.items() if tt==t)) for t in range(10)},
            by_phase=strata('_phase'),by_grip_event=strata('_event'),distance_edges=edges,
            distance_bins={str(k):summarize([r for r in rr if r['vision'] and np.searchsorted(edges,r['extras']['dnn'])==k]) for k in range(3)},
            correction_rms=float(np.mean([r['_corr_rms'] for r in cc])),grip_reconstruction_max=max(r['_grip_difference'] for r in cc),
            correction_next_call={str(v):dict(n=sum(r['_next_call']==v for r in cc),mean_rms=float(np.mean([r['_corr_rms'] for r in cc if r['_next_call']==v]))) for v in (False,True)})
        eps[variant]={key:dict(summarize(rr),success=accepted[key]['success']) for key,rr in groups.items()}
        outcomes[variant]=accepted
    paired={}
    for label in ('both_win','both_lose','lost','gained'):
        pairs=[key for key in eps['control'] if ('both_win' if eps['control'][key]['success'] and eps['taskfree'][key]['success'] else 'lost' if eps['control'][key]['success'] else 'gained' if eps['taskfree'][key]['success'] else 'both_lose')==label]
        paired[label]=dict(pairs=pairs,n=len(pairs),
            control_calls=sum(eps['control'][p]['calls'] for p in pairs),taskfree_calls=sum(eps['taskfree'][p]['calls'] for p in pairs),
            control_decisions=sum(eps['control'][p]['n'] for p in pairs),taskfree_decisions=sum(eps['taskfree'][p]['n'] for p in pairs))
    delta=np.array([np.mean([int(eps['taskfree'][t,i]['success'])-int(eps['control'][t,i]['success']) for t in range(10)]) for i in range(20,30)])
    boot=delta[np.random.default_rng(260702).integers(0,10,(5000,10))].mean(1)
    return dict(owner_look_price=LOOK,arms=arms,paired=paired,sr_ci95=np.quantile(boot,[.025,.975]))


def offline(cell):
    tr=annotated(cell,'A','train'); d=annotated(cell,'A','eval'); m=load_control(cell)
    with np.load(HERE.parent/'round6/artifacts'/f'{cell}.npz') as z:
        head={k[5:]:np.array(z[k]) for k in z.files if k.startswith('head_')}; table=z['table']
    pred=d['base'][:,:,:6]+.5*residual(head,table,d['x'],d['rows'],d['weights'])
    ctl=control_prediction(m,d)
    e=((pred-d['teacher'][:,:,:6])**2).mean((1,2)); c=((ctl-d['teacher'][:,:,:6])**2).mean((1,2))
    diff=((pred-ctl)**2).mean((1,2)); shared=dense_predict(head,d['x']); local=np.einsum('nk,nko->no',d['weights'],table[d['rows']])
    def score(take):
        return dict(n=int(take.sum()),r6_mse=float(e[take].mean()),control_mse=float(c[take].mean()),delta=float((e-c)[take].mean()),
            correction_disagreement_rms=float(np.sqrt(diff[take].mean())),shared_rms=float(np.sqrt((shared[take]**2).mean())),local_rms=float(np.sqrt((local[take]**2).mean())))
    edges=np.nanquantile(tr['distance'],[.5,.9]); bins=np.searchsorted(edges,d['distance'])
    train=combine([dataset(cell,v,'train') for v in VARIANTS]); nrows=len(library(cell))
    mass=np.bincount(train['rows'].ravel(),weights=train['weights'].ravel(),minlength=nrows)
    count=np.zeros(nrows)
    for ep in np.unique(train['episode']): count[np.unique(train['rows'][train['episode']==ep])]+=1
    support=(d['weights']*count[d['rows']]).sum(1)
    return dict(total=score(np.ones(len(e),bool)),by_task={str(t):score(d['task']==t) for t in range(10)},
        by_phase={p:score(d['phase']==p) for p in np.unique(d['phase'])},
        by_grip_event={str(v):score(d['grip_event']==v) for v in (False,True)},distance_edges=edges,
        by_distance={str(k):score(bins==k) for k in range(3)},
        by_chunk_half={str(k):dict(r6_mse=float(((pred[:,k:k+5]-d['teacher'][:,k:k+5,:6])**2).mean()),control_mse=float(((ctl[:,k:k+5]-d['teacher'][:,k:k+5,:6])**2).mean())) for k in (0,5)},
        coverage=dict(rows=nrows,zero_mass=int((mass==0).sum()),mass_quantiles=np.quantile(mass,[0,.1,.5,.9,1]),
            episode_support_quantiles=np.quantile(count,[0,.1,.5,.9,1]),eval_weight_unsupported=float((d['weights']*(mass[d['rows']]==0)).sum()/len(e)),
            eval_weighted_episode_support=np.quantile(support,[0,.1,.5,.9,1])),
        action=dict(library_shape=list(library(cell).shape),padding_max=float(np.max(np.abs(library(cell)[:,:,7:]))),
            gripper_teacher_disagreement=float(np.mean((d['base'][:,:,6]<0)!=(d['teacher'][:,:,6]<0)))))


def main():
    out={}
    for cell in CELLS:
        out[cell]=dict(live=live(cell),offline=offline(cell))
        dump(HERE/'results'/f'diagnosis_{cell}.json',out[cell]); print(cell,json.dumps(out[cell]['live']['paired']),flush=True)
    dump(HERE/'results/diagnosis.json',out)

if __name__=='__main__': main()
