"""Train a task-blind late-transition detector; evaluate disjoint initial states.

Fit on A/CU inits 0..19. Hyperparameters/threshold selected using four init-group
OOF folds entirely inside 0..19. Freeze before evaluating 20..29. IP paths are
excluded from fitting and reserved for randomized local-effect estimation.
"""
import json
import numpy as np
from .metrics import average_precision_score,roc_auc_score
from ...round2.tools.learn import fit
from ...round2.inference import predict
from ..inference import Monitor,gate_schedule,RecoveryGate
from .data import HERE,DERIVED,CELLS,load,library,subset,epmeans,interval,dump,sha,balanced_weights


def featurize(d,cell):
    lib=library(cell)
    phase=np.load(lib/'step.npy',mmap_mode='r')/np.maximum(np.load(lib/'ep_len.npy',mmap_mode='r')-1,1)
    x=[];last=None
    for j,ep in enumerate(d['episode']):
        if ep!=last:monitor=Monitor();last=ep
        x.append(monitor.observe(d['visual'][j],d['state'][j],d['base'][j],d['rows'][j],d['weights'][j],
            phase[d['rows'][j]],int(d['seq'][j]),d['d1_rel'][j]))
    return np.stack(x)


def population(cell,variant):
    d=load(cell,variant)
    if d is not None:d['x']=featurize(d,cell)
    return d


def merge(ds):
    keys=set.intersection(*(set(d) for d in ds))
    out={k:np.concatenate([d[k] for d in ds]) for k in keys}
    out['episode']=np.concatenate([d['episode']+300*j for j,d in enumerate(ds)])
    return out


def target(d):
    # Privileged/current stall plus future no-new-goal100 label. Late censored
    # failures are excluded from FIT and discrimination metrics.
    return d['late_trap100'].astype(float)


def selected_entries(d,score,threshold,burst=3):
    first=np.zeros(len(score),bool);call=np.zeros(len(score),bool);start=np.zeros(len(score),bool)
    for ep in np.unique(d['episode']):
        ii=np.flatnonzero(d['episode']==ep)
        gate=RecoveryGate(threshold,burst=burst)
        for j in ii:
            call[j],start[j]=gate.observe(float(score[j]),int(d['seq'][j]))
        idx=ii[start[ii]]
        if len(idx):first[idx[0]]=True
    return first,call,start


def assess(d,score,threshold):
    eligible=d['valid100']&(d['seq']>=12)
    y=target(d)[eligible];s=score[eligible]
    w=balanced_weights(d['episode'][eligible])
    first,call,start=selected_entries(d,score,threshold)
    validfirst=first&d['valid100']
    ep=np.unique(d['episode']);fep=d['episode'][first]
    failed=np.array([not bool(d['success'][np.flatnonzero(d['episode']==e)[0]]) for e in ep])
    fire=np.isin(ep,fep)
    cost=sum(float(d['cost'][np.flatnonzero(d['episode']==e)[0]]) for e in ep)
    n=sum(int(d['decisions'][np.flatnonzero(d['episode']==e)[0]]) for e in ep)
    motion=(((d['base'][:,:,:6]-d['teacher'][:,:,:6])**2).mean((1,2)))
    grip=((d['base'][:,:,6]>=0)!=(d['teacher'][:,:,6]>=0)).mean(1)
    out=dict(n_episodes=len(ep),anchors=len(d['seq']),valid_anchors=int(eligible.sum()),
        target_prevalence=float(np.average(y,weights=w)),
        auroc=float(roc_auc_score(y,s,sample_weight=w)) if len(np.unique(y))==2 else None,
        average_precision=float(average_precision_score(y,s,sample_weight=w)),
        fired_episodes=int(fire.sum()),failed_episodes=int(failed.sum()),
        failure_coverage=float(fire[failed].mean()) if failed.any() else None,
        success_fire_rate=float(fire[~failed].mean()) if (~failed).any() else None,
        first_alert_failure_precision=float(failed[fire].mean()) if fire.any() else None,
        first_alert_late_stall_precision=float(d['late_stall'][first].mean()) if first.any() else None,
        first_alert_trap_precision=float(target(d)[validfirst].mean()) if validfirst.any() else None,
        first_alert_valid100=int(validfirst.sum()),
        first_alert_control_median=float(np.median(d['seq'][first]*5)) if first.any() else None,
        first_alert_remaining_median=float(np.median(d['remaining_controls'][first])) if first.any() else None,
        first_alert_gripper_error=float(grip[first].mean()) if first.any() else None,
        first_alert_motion_error=float(motion[first].mean()) if first.any() else None,
        factual_sr=float((~failed).mean()),factual_ir=float(cost/n),
        replay_calls=int(call.sum()),replay_bursts=int(start.sum()),
        # Counterfactual paths/lengths unknown. Only replacing cache looks adds
        # the non-vision policy cost. For IP/CU this isn't incremental IR.
        replay_policy_anchor_fraction=float(call.mean()),
        warning='Fixed-path exposure accounting, not candidate closed-loop IR or SR')
    return out


def train(cell):
    ds=[population(cell,v) for v in ['A','CU']]
    train=merge([subset(d,d['init']<20) for d in ds])
    valid=train['valid100']
    tr=subset(train,valid)
    y=target(tr)
    candidates=[]
    for nrf in [0,128]:
        oof=np.zeros(len(tr['seq']))
        for fold in range(4):
            te=tr['init']%4==fold
            head=fit(tr['x'][~te],y[~te,None],tr['episode'][~te],random_features=nrf,alpha=100.)
            oof[te]=predict(head,tr['x'][te])[:,0]
        eligible=tr['seq']>=12
        ap=float(average_precision_score(y[eligible],oof[eligible],sample_weight=balanced_weights(tr['episode'][eligible])))
        candidates.append(dict(random_features=nrf,ap=ap,scores=oof))
    chosen=max(candidates,key=lambda r:r['ap'])
    # Training OOF threshold: maximize F0.5 on current/future target. No eval
    # outcomes or thresholds participate. Fixed grid chosen before eval.
    eligible=tr['seq']>=12
    w=balanced_weights(tr['episode'][eligible]);yy=y[eligible];ss=chosen['scores'][eligible]
    thresholds=[]
    for quantile in [.8,.9,.95]:
        threshold=float(np.quantile(ss,quantile));yes=ss>=threshold
        tp=float((w*yes*yy).sum());fp=float((w*yes*(1-yy)).sum());fn=float((w*~yes*yy).sum())
        f=float(1.25*tp/max(1.25*tp+.25*fn+fp,1e-12))
        thresholds.append(dict(quantile=quantile,threshold=threshold,f05=f))
    selected=max(thresholds,key=lambda r:r['f05'])
    model=fit(tr['x'],y[:,None],tr['episode'],random_features=chosen['random_features'],alpha=100.)
    # OOF/final score scales are not assumed identical: use the selected OOF
    # quantile on FINAL TRAIN scores, still never evaluating on training starts.
    score=predict(model,tr['x'])[:,0]
    threshold=float(np.quantile(score[eligible],selected['quantile']))
    path=HERE/'artifacts'/f'{cell}_monitor.npz';np.savez_compressed(path,**model)
    meta=dict(cell=cell,training_inits=list(range(20)),eval_inits=list(range(20,30)),
        source_variants=['A','CU'],target='late_stall and no new goal/success within next 100 controls',
        censoring='fit/discrimination exclude unsuccessful horizons truncated before 100 controls',
        input_dim=tr['x'].shape[1],train_anchors=len(tr['seq']),random_features=chosen['random_features'],alpha=100,
        cv_candidates=[{k:v for k,v in c.items() if k!='scores'} for c in candidates],
        cv_thresholds=thresholds,selected_quantile=selected['quantile'],threshold=threshold,
        head_sha256=sha(path),warmup_decisions=12,consecutive=2,burst_calls=3,cooldown_anchors=6,max_calls=12,
        task_agnostic=True,predicate_inputs=False)
    dump(path.with_suffix('.json'),meta)
    print('FROZEN',cell,meta['random_features'],threshold,flush=True)
    return model,meta


def evaluate(cell,model,meta):
    out=dict(cell=cell,head_sha256=meta['head_sha256'],threshold=meta['threshold'],populations={})
    for variant in ['A','CU','IP']:
        d=population(cell,variant)
        if d is None:continue
        d=subset(d,d['init']>=20)  # evaluation BEFORE any outcome aggregation
        s=predict(model,d['x'])[:,0]
        result=assess(d,s,meta['threshold'])
        first,call,start=selected_entries(d,s,meta['threshold'])
        _,one,_=selected_entries(d,s,meta['threshold'],burst=1)
        a=.152 if cell.startswith('pi05') else .148
        count=sum(int(d['decisions'][np.flatnonzero(d['episode']==e)[0]]) for e in np.unique(d['episode']))
        result['cache_path_IR_burst3']=result['factual_ir']+(1-a)*call.sum()/count if variant=='A' else None
        result['cache_path_IR_burst1']=result['factual_ir']+(1-a)*one.sum()/count if variant=='A' else None
        result['replay_calls_burst1']=int(one.sum())
        out['populations'][variant]=result
        np.savez_compressed(DERIVED/f'{cell}_{variant}_eval_scores.npz',task=d['task'],init=d['init'],
            decision_id=d['decision_id'],score=s,first=first,call=call,start=start)
    dump(HERE/'results'/f'screen_{cell}.json',out)
    print('EVAL',cell,json.dumps(out['populations']['A']),flush=True)
    return out


def main():
    # Freeze every trained model before inspecting any new evaluation metric.
    fitted={c:train(c) for c in CELLS}
    dump(HERE/'results/screen.json',[evaluate(c,*fitted[c]) for c in CELLS])


if __name__=='__main__':main()
