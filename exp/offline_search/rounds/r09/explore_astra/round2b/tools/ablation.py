"""Check whether the transition monitor is merely an elapsed-time trigger.

New ablations fit only 0..19; thresholds match FULL detector training-path call
count, not evaluation outcomes. No task index enters any predictor or budget.
"""
import json
import numpy as np
from ...round2.tools.learn import fit
from ...round2.inference import predict
from .data import HERE,CELLS,subset,dump
from .screen import population,merge,target,selected_entries,assess


def one(cell):
    train=merge([subset(d,d['init']<20) for d in [population(cell,'A'),population(cell,'CU')]])
    with np.load(HERE/'artifacts'/f'{cell}_monitor.npz',allow_pickle=False) as z:full={k:z[k] for k in z.files}
    meta=json.loads((HERE/'artifacts'/f'{cell}_monitor.json').read_text())
    _,calls,_=selected_entries(train,predict(full,train['x'])[:,0],meta['threshold'])
    target_calls=int(calls.sum());valid=train['valid100']
    frozen={}
    for name,cols in [('clock',[0]),('no_clock',list(range(1,train['x'].shape[1])))]:
        m=fit(train['x'][valid][:,cols],target(train)[valid,None],train['episode'][valid],random_features=128,alpha=100.)
        score=predict(m,train['x'][:,cols])[:,0]
        thresholds=np.quantile(score[train['seq']>=12],np.linspace(.5,.995,100))
        counts=[int(selected_entries(train,score,float(t))[1].sum()) for t in thresholds]
        j=int(np.argmin(np.abs(np.array(counts)-target_calls)))
        frozen[name]=(m,cols,float(thresholds[j]),counts[j])
    eval=population(cell,'A');eval=subset(eval,eval['init']>=20)
    out={}
    for name,(m,cols,t,ncalls) in frozen.items():
        result=assess(eval,predict(m,eval['x'][:,cols])[:,0],t)
        result.update(train_threshold=t,train_calls=ncalls,target_train_calls=target_calls)
        out[name]=result
    dump(HERE/'results'/f'ablation_{cell}.json',dict(cell=cell,ablations=out))
    print(cell,{k:{v:r[v] for v in ['failure_coverage','success_fire_rate','first_alert_failure_precision','first_alert_control_median','replay_calls']} for k,r in out.items()},flush=True)
    return dict(cell=cell,ablations=out)


if __name__=='__main__':dump(HERE/'results/ablation.json',[one(c) for c in CELLS])
