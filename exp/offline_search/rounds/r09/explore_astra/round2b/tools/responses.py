"""Fixed-path cost accounting and TRAIN-only matching of random burst control."""
import json
import numpy as np
from ...round2.inference import predict
from exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.methods import uniform
from ..inference import RecoveryGate,LatchedGate,RandomBurstGate
from .data import HERE,CELLS,library,subset,dump,sha
from .screen import population,selected_entries,merge


def expected_random_calls(steps,p,burst=3,cap=12,cooldown=6,warmup=12):
    """Exact finite-state expectation; no Monte Carlo or evaluation fitting."""
    states={(0,0,0):1.}
    for step in steps:
        new={}
        for (left,cool,used),mass in states.items():
            options=[(left,cool,used,1.)]
            if cool>0:options=[(left,cool-1,used,1.)]
            elif left==0 and step>=warmup and used<cap:
                options=[(min(burst,cap-used),cool,used,p),(0,cool,used,1-p)]
            for rem,cd,u,prob in options:
                if rem>0:
                    rem-=1;u+=1
                    if rem==0:cd=cooldown
                key=(rem,cd,u);new[key]=new.get(key,0.)+mass*prob
        states=new
    return sum(u*mass for (_,_,u),mass in states.items())


def replay(d,s,threshold,mode,p=0):
    out=np.zeros(len(s),bool);starts=np.zeros(len(s),bool)
    for ep in np.unique(d['episode']):
        gate=LatchedGate(threshold) if mode=='latch' else RandomBurstGate(p) if mode=='random3' else RecoveryGate(threshold,burst=1 if mode=='burst1' else 3)
        for j in np.flatnonzero(d['episode']==ep):
            coin=uniform('R9R2b-transition-control',20261002,int(d['task'][j]),int(d['init'][j]),int(d['seq'][j]),'burst-start')
            out[j],starts[j]=gate.observe(coin if mode=='random3' else float(s[j]),int(d['seq'][j]))
    return out,starts


def one(cell):
    with np.load(HERE/'artifacts'/f'{cell}_monitor.npz',allow_pickle=False) as z:model={k:z[k] for k in z.files}
    meta=json.loads((HERE/'artifacts'/f'{cell}_monitor.json').read_text());threshold=meta['threshold']
    train=merge([subset(d,d['init']<20) for d in [population(cell,'A'),population(cell,'CU')]])
    score=predict(model,train['x'])[:,0]
    target=int(selected_entries(train,score,threshold)[1].sum())
    # A/CU can have extra looks. Match on their exact PRE-EXISTING train grids.
    schedules={}
    for ep in np.unique(train['episode']):
        key=tuple(train['seq'][train['episode']==ep].tolist());schedules[key]=schedules.get(key,0)+1
    def expectation(p):return sum(n*expected_random_calls(steps,p) for steps,n in schedules.items())
    lo,hi=0.,1.
    for _ in range(24):
        mid=(lo+hi)/2
        if expectation(mid)<target:lo=mid
        else:hi=mid
    p=(lo+hi)/2
    lib=library(cell);phase=np.load(lib/'step.npy')/np.maximum(np.load(lib/'ep_len.npy')-1,1)
    np.save(HERE/'artifacts'/f'{cell}_phase.npy',phase,allow_pickle=False)
    freeze=dict(cell=cell,fit_inits=list(range(20)),eval_inits=list(range(20,30)),threshold=threshold,
        random_probability=p,random_seed=20261002,target_train_calls=target,
        expected_random_train_calls=expectation(p),phase_sha256=sha(HERE/'artifacts'/f'{cell}_phase.npy'),
        responses=['burst1','burst3','latch','random3'],max_calls=12)
    dump(HERE/'artifacts'/f'{cell}_response.json',freeze)
    result={}
    for variant in ['A','CU','IP']:
        d=population(cell,variant)
        if d is None:continue
        d=subset(d,d['init']>=20);s=predict(model,d['x'])[:,0]
        eps=np.unique(d['episode']);first=[np.flatnonzero(d['episode']==e)[0] for e in eps]
        decisions=float(d['decisions'][first].sum());cost=float(d['cost'][first].sum())
        a=.152 if cell.startswith('pi05') else .148
        stats={}
        for mode in freeze['responses']:
            call,start=replay(d,s,threshold,mode,p)
            stats[mode]=dict(calls=int(call.sum()),starts=int(start.sum()),
                fired_episodes=int(len(np.unique(d['episode'][start]))),
                cache_path_IR=float((cost+(1-a)*call.sum())/decisions) if variant=='A' else None,
                expected_random_calls=float(sum(expected_random_calls(d['seq'][d['episode']==ep],p) for ep in eps)) if mode=='random3' else None)
        result[variant]=stats
    out=dict(cell=cell,freeze=freeze,fixed_path_evaluation=result,
        warning='Only cache-path replacement IR is meaningful; recovery may change path/length. No SR prediction.')
    dump(HERE/'results'/f'responses_{cell}.json',out)
    print(cell,json.dumps(result['A']),flush=True)
    return out


if __name__=='__main__':dump(HERE/'results/responses.json',[one(c) for c in CELLS])
