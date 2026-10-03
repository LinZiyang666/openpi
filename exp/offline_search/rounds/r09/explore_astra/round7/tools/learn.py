"""Train-only init-group CV; freeze two families before evaluation."""
import argparse,json,time
from datetime import datetime,timezone
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .data import *
from .numeric import features,confidence,predict
from exp.offline_search.rounds.r09.explore_fable.round2.tools.corrector import fit_head
from exp.offline_search.rounds.r09.explore_astra.round6.tools.numeric import dense_predict
from exp.offline_search.rounds.r09.explore_astra.round6.tools.learn import library,design,load_control,control_prediction,metric

CONFIGS={
    'r6':dict(n_rff=768,alpha=100.,enriched=False),
    'wide1536':dict(n_rff=1536,alpha=100.,enriched=False),
    'wide3072':dict(n_rff=3072,alpha=100.,enriched=False),
    'grip1536':dict(n_rff=1536,alpha=100.,enriched=True),
    'grip3072':dict(n_rff=3072,alpha=100.,enriched=True),
    'grip3072_a30':dict(n_rff=3072,alpha=30.,enriched=True),
}
GAINS=(0.,.1,.25)


def train_assert(d):
    if not np.all((d['init']>=0)&(d['init']<20)): raise ValueError('fit outside 0..19')
    if d['x'].shape[1]!=207: raise ValueError('non-observation inputs')


def fit(d,lib,config):
    train_assert(d)
    x=features(d['x'],config['enriched']); y=(d['teacher'][:,:,:6]-d['base'][:,:,:6]).reshape(len(x),60)
    wt=balanced(d['episode'])
    head=fit_head(x,y,wt,seed=260602,n_rff=config['n_rff'],alpha=config['alpha'])
    w=design(d,len(lib)); wtw=w.T.multiply(wt)
    table=splu((wtw@w).tocsc()+sparse.eye(len(lib),format='csc')).solve(wtw@(y-dense_predict(head,x))).astype(np.float32)
    # Recover variance from confidence with scale=1, then freeze a cell-level scale.
    cf=confidence(lib,d['rows'],d['weights'],1.)
    scale=float(np.median(1/cf-1))
    meta=dict(config,enriched=config['enriched'],dispersion_scale=scale,confidence_gain=0.)
    return head,table,meta


def prediction(model,d,lib,gain=None):
    h,t,m=model
    m=dict(m)
    if gain is not None: m['confidence_gain']=gain
    return d['base'][:,:,:6]+predict(h,t,d['x'],d['rows'],d['weights'],lib,m)


def select():
    if (HERE/'SELECTION.json').exists(): raise ValueError('selection already frozen')
    result={}
    for cell in CELLS:
        d=combine([dataset(cell,v,'train') for v in VARIANTS]); lib=library(cell)
        sums={f'{name}:{gain}':[] for name in CONFIGS for gain in GAINS}
        for fold in range(3):
            tr=subset(d,d['init']%3!=fold); va=subset(d,d['init']%3==fold)
            _,base=episode_errors(va['base'][:,:,:6],va)
            for name,cfg in CONFIGS.items():
                start=time.monotonic(); model=fit(tr,lib,cfg)
                for gain in GAINS:
                    _,e=episode_errors(prediction(model,va,lib,gain),va)
                    sums[f'{name}:{gain}'].append((float(e.sum()),float(base.sum())))
                print(cell,fold,name,round(time.monotonic()-start,2),flush=True)
        scores={k:sum(x for x,y in v)/sum(y for x,y in v) for k,v in sums.items()}
        wide=min((k for k in scores if k.endswith(':0.0') and not k.startswith('r6:')),key=scores.get)
        conf=min((k for k in scores if not k.endswith(':0.0') and not k.startswith('r6:')),key=scores.get)
        result[cell]=dict(scores=scores,winners=dict(capacity=wide,confidence=conf),n_train=len(d['init']),episodes=len(np.unique(d['episode'])))
        dump(HERE/'results'/f'cv_{cell}.json',result[cell]);print('WINNERS',cell,wide,conf,flush=True)
    dump(HERE/'SELECTION.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),fit_inits=list(range(20)),
        eval_accessed_for_selection=False,selection='3-fold init modulo 3; per-cell episode-balanced MSE; capacity gain=0; confidence gain>0',
        configs=CONFIGS,gains=GAINS,cells=result))


def evaluate():
    s=json.loads((HERE/'SELECTION.json').read_text()); result={}
    for cell in CELLS:
        d=combine([dataset(cell,v,'train') for v in VARIANTS]); lib=library(cell); ctl=load_control(cell)
        result[cell]={}
        for variant,key in s['cells'][cell]['winners'].items():
            name,gain=key.split(':'); head,table,meta=fit(d,lib,CONFIGS[name]); meta.update(confidence_gain=float(gain),cell=cell,
                variant=variant,fit_inits=list(range(20)),task_id_input=False,task_dispatch=False,input_dimension=len(head['mean']),
                selection_sha256=sha(HERE/'SELECTION.json'),fit_variants=list(VARIANTS),anchors=len(d['init']),row_lambda=1.)
            path=owned(HERE/'artifacts'/f'{cell}_{variant}.npz')
            np.savez_compressed(path,table=table,**{'head_'+k:v for k,v in head.items()},meta_json=json.dumps(meta))
            paths={}
            for v in VARIANTS:
                te=dataset(cell,v,'eval'); control=control_prediction(ctl,te)
                paths[v]=dict(control=metric(control,te),candidate=metric(prediction((head,table,meta),te,lib),te,control))
            result[cell][variant]=dict(meta=meta,paths=paths,sha256=sha(path))
            print(cell,variant,{v:round(p['candidate']['relative_to_control'],4) for v,p in paths.items()},flush=True)
        dump(HERE/'results'/f'eval_{cell}.json',result[cell])
    dump(HERE/'results/evaluation.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','evaluate']);a=p.parse_args()
    select() if a.action=='select' else evaluate()
