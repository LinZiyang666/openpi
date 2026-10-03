"""Frozen recipe head audit on R8 cache/random-call paths, with train-only cuts.

This is fixed-observation shadow imitation error, not a closed-loop ablation.
The four diagnostic zero-correction gates are specified before evaluating them.
"""
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from .safe import HERE, owned, dump
from .analyze import compact, recipe, phases, is_closed, clean_json
from exp.offline_search.rounds.r09.recipe.recipe import _predict

VARIANTS=('A','CU','IP')
GATES=('crossing','open','large_correction','far_retrieval')


def fit_cuts(train):
    if not len(train) or not train.init.between(0,19).all():
        raise ValueError('threshold fitting is restricted to inits 0..19')
    return dict(correction=float(train.correction.quantile(.9)),distance=float(train.distance.quantile(.9)))


def load(model, variant, base):
    a=compact(model,variant,['seq','vision','served_chunk','state_norm',
        'shadow_look_cache_chunk','policy_shadow_chunk','shadow_look_keys_pca_third',
        'shadow_look_keys_pca_wrist','d1','shadow_look_rows','shadow_look_weights'])
    phase=np.full(len(a['init']),'',dtype='<U10')
    eid=a['task'].astype(int)*30+a['init']
    for ep in np.unique(eid):
        ii=np.flatnonzero(eid==ep); ii=ii[np.argsort(a['seq'][ii])]
        phase[ii]=phases(is_closed(a['served_chunk'][ii,:5,6],model).mean(1)>.5)
    b=a['shadow_look_cache_chunk']; y=a['policy_shadow_chunk']
    good=(a['vision'] & np.isfinite(b).all((1,2)) & np.isfinite(y).all((1,2))
        & np.isfinite(a['state_norm']).all(1) & np.isfinite(a['shadow_look_keys_pca_third']).all(1)
        & np.isfinite(a['shadow_look_keys_pca_wrist']).all(1))
    a={k:v[good] for k,v in a.items()}; phase=phase[good]; b=b[good]; y=y[good]
    x=np.c_[a['shadow_look_keys_pca_third'],a['shadow_look_keys_pca_wrist'],a['state_norm'][:,:8],
        (b/base.sig_head).reshape(len(b),-1),np.minimum(a['seq'],120)/120,
        np.eye(10,dtype=np.float32)[a['task']]].astype(np.float32)
    c=np.empty((len(b),10,6),np.float32)
    for t in np.unique(a['task']):
        sel=a['task']==t
        c[sel]=.5*_predict(base.heads[str(int(t))],x[sel]).reshape(-1,10,6)*base.sig_head[:6]
    corrected=b[:,:,:6]+c
    rows=a['shadow_look_rows']; w=a['shadow_look_weights'].astype(float); w/=w.sum(1,keepdims=True)
    reconstructed=np.einsum('nk,nkha->nha',w,base.act[rows,:10,:7])
    assert np.max(np.abs(reconstructed-b))<1e-4
    err0=((b[:,:,:6]-y[:,:,:6])**2).mean((1,2))
    err1=((corrected-y[:,:,:6])**2).mean((1,2))
    df=pd.DataFrame(dict(model=model,variant=variant,task=a['task'],init=a['init'],seq=a['seq'],phase=phase,
        distance=a['d1'],correction=np.sqrt((c**2).mean((1,2))),cache_mse=err0,recipe_mse=err1,
        crossing=np.any(np.diff(is_closed(b[:,:,6],model),axis=1),axis=1),
        open=~(is_closed(b[:,:5,6],model).mean(1)>.5),
        grip_disagree=(is_closed(b[:,:,6],model)!=is_closed(y[:,:,6],model)).mean(1),
        early_delta=((corrected[:,:5]-y[:,:5,:6])**2).mean((1,2))-((b[:,:5,:6]-y[:,:5,:6])**2).mean((1,2)),
        tail_delta=((corrected[:,5:]-y[:,5:,:6])**2).mean((1,2))-((b[:,5:,:6]-y[:,5:,:6])**2).mean((1,2))))
    return df


def metric(g):
    if len(g)==0:return dict(n=0)
    e=g.groupby(['variant','task','init'])[['cache_mse','recipe_mse','correction','grip_disagree','early_delta','tail_delta']].mean()
    return dict(n=len(g),episodes=len(e),cache=float(e.cache_mse.mean()),recipe=float(e.recipe_mse.mean()),
        relative=float(e.recipe_mse.mean()/e.cache_mse.mean()-1),better_fraction=float((e.recipe_mse<e.cache_mse).mean()),
        correction=float(e.correction.mean()),grip_disagree=float(e.grip_disagree.mean()),
        early_delta=float(e.early_delta.mean()),tail_delta=float(e.tail_delta.mean()))


def gate_metrics(df,gate):
    # Equal weight per episode; inactive episodes remain in the denominator.
    e=df.assign(delta=np.where(df[gate],df.cache_mse-df.recipe_mse,0)).groupby(['variant','task','init']).agg(
        delta=('delta','mean'),recipe=('recipe_mse','mean'))
    init=e.groupby('init').delta.mean().to_numpy()
    boot=init[np.random.default_rng(260802).integers(0,len(init),(5000,len(init)))].mean(1)
    return dict(anchors=int(df[gate].sum()),fraction=float(df[gate].mean()),delta=float(e.delta.mean()),
        relative=float(e.delta.mean()/e.recipe.mean()),ci95=np.quantile(boot,[.025,.975]),
        episodes_improved=int((e.delta<0).sum()),episodes_worsened=int((e.delta>0).sum()))


def main():
    protocol=HERE/'SCREEN_PROTOCOL.json'
    if not protocol.exists():
        dump(protocol,dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
            objective='diagnostic removal of correction; no new task-indexed logic; no escalation',
            variants=VARIANTS,fit_inits=list(range(20)),eval_inits=list(range(20,30)),
            gates=dict(crossing='proposal gripper sign crossing anywhere in committed ten controls',
                open='majority of first five proposed gripper commands open',
                large_correction='correction RMS above pooled 0..19 90th percentile',
                far_retrieval='within-model shadow d1 above pooled 0..19 90th percentile'),
            arm_rule='Only consider freezing if train improves and eval improvement repeats on A/CU/IP; offline gain alone does not demonstrate success.'))
    out={}; frames=[]
    for model in ('groot','pi05'):
        base=recipe(model).inner.base
        d=pd.concat([load(model,v,base) for v in VARIANTS],ignore_index=True)
        train=d[d.init<20]
        cuts=fit_cuts(train)
        d['large_correction']=d.correction>cuts['correction'];d['far_retrieval']=d.distance>cuts['distance']
        out[model]=dict(fit_cuts=cuts,splits={})
        for split,keep in [('fit',d.init<20),('eval',d.init>=20)]:
            g=d[keep]
            out[model]['splits'][split]=dict(total=metric(g),
                by_variant={k:metric(h) for k,h in g.groupby('variant')},
                by_phase={k:metric(h) for k,h in g.groupby('phase')},
                by_task={str(k):metric(h) for k,h in g.groupby('task')},
                by_crossing={str(k):metric(h) for k,h in g.groupby('crossing')},
                gates={k:gate_metrics(g,k) for k in GATES},
                gates_by_variant={v:{k:gate_metrics(h,k) for k in GATES} for v,h in g.groupby('variant')})
        frames.append(d)
        print(model,json.dumps(clean_json(out[model]['splits']['eval'])),flush=True)
    pd.concat(frames,ignore_index=True).to_parquet(owned(HERE/'results/corrector.parquet'),index=False)
    dump(HERE/'results/corrector_summary.json',clean_json(out))


if __name__=='__main__':main()
