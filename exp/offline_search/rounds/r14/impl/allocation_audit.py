"""Independently reproduce frozen task priorities, R11 caps and B-path bias."""
import json
from pathlib import Path
import numpy as np
from exp.offline_search.closed_loop.devset import library_episodes,PARENT
from exp.offline_search.rounds.r11.knob.calibration import tables
from exp.offline_search.rounds.r11.knob.recipe import uniform
from exp.offline_search.rounds.r14.knob.controller import caps
from .common import *

COSTS={'pi05':(.152,.848),'groot':(.148,.852)}

def priorities(p):
    values=[];counts=[]
    for task in range(10):
        ev=[]
        for j in np.flatnonzero(p['task']==task):
            eligible=p['alive'][:,j]&~p['guard'][:,j]&np.isfinite(p['risk'][:,j])
            if eligible.any():ev.append(float(np.log1p(p['risk'][eligible,j].mean())))
        values.append(float(np.mean(ev)) if ev else np.nan);counts.append(len(ev))
    r,n=np.asarray(values),np.asarray(counts)
    mu=float(np.nanmean(r));spread=max(float(np.nanstd(r)),.1)
    z=np.where(np.isfinite(r),np.clip((r-mu)/spread,-2,2)*n/(n+10),0.)
    return dict(raw=values,n=counts,mean=mu,spread=spread,shrink=(n/(n+10)).tolist(),z=z.tolist())

def prepare(p):
    H,E=p['alive'].shape;coins=np.empty((H+1,E))
    for j in range(E):
        for i in range(-1,H):
            step=-1 if i<0 else int(p['step'][i,j])
            coins[i+1,j]=uniform('R11-gap-v1',0,int(p['task'][j]),int(p['init'][j]),step,'cap')
    p['coins']=coins
    return p

def replay(p,model,K):
    if 'coins' not in p:prepare(p)
    settings=np.broadcast_to(np.asarray(K),(10,))[p['task']]
    floor=np.floor(settings);frac=settings-floor;cap=floor+(p['coins'][0]<frac)
    run=np.zeros(len(settings),int);calls=np.zeros_like(p['alive'])
    for i in range(len(calls)):
        alive=p['alive'][i];call=alive&(p['guard'][i]|(run>=cap));calls[i]=call
        run=np.where(alive,np.where(call,0,run+1),run)
        cap=np.where(call,floor+(p['coins'][i+1]<frac),cap)
    cv,cm=COSTS[model]
    return float((cv*p['alive'].sum(0)+cm*calls.sum(0)).sum()/p['N'].sum())

def solve(paths,model,K,z):
    refs=[replay(p,model,K) for p in paths];choices=[];lo,hi=-8.,8.
    for _ in range(32):
        bias=(lo+hi)/2;settings=caps(K,z,bias,.35)
        irs=[replay(p,model,settings) for p in paths];errors=np.asarray(irs)-refs
        choices.append((float(np.max(np.abs(errors))),abs(bias),bias,settings,irs))
        if (errors.max()+errors.min())/2>0:lo=bias
        else:hi=bias
    error,_,bias,settings,irs=min(choices,key=lambda r:r[:3])
    return dict(bias=bias,K=settings.tolist(),reference_irs=refs,irs=irs,max_match_error=error)

def libpack(cell):
    model,suite,n=cell.split('_');_,_,eps,sched=tables(model,suite,int(n))
    _,metadata,_=library_episodes(model,suite,PARENT[model])
    H=max(len(e.steps) for e in eps);E=len(eps)
    p={k:np.zeros((H,E),bool if k in ('alive','guard') else float) for k in ('alive','guard','step','risk')}
    for j,e in enumerate(eps):
        n=len(e.steps);p['alive'][:n,j]=True;p['guard'][:n,j]=e.guard
        p['step'][:n,j]=e.steps;p['risk'][:n,j]=e.risk**2
    p.update(N=np.array([e.n_slots for e in eps]),task=np.array([e.task for e in eps]),
             init=np.array([metadata[e.ep]['orig_init_state_idx'] for e in eps]))
    return p,sched

def devpack(meta):
    path=ASTRA/'data'/f'{meta["root"]}__{meta["arm"]}.npz'
    with np.load(path,allow_pickle=False) as z:d={k:np.array(z[k]) for k in z.files}
    E=len(d['keys']);H=int(d['epV'].max())
    p={k:np.zeros((H,E),bool if k in ('alive','guard') else float) for k in ('alive','guard','step','risk')}
    p['risk'][:]=np.nan
    for j in range(E):
        ix=np.flatnonzero((d['ep']==j)&d['vision'].astype(bool));n=len(ix);p['alive'][:n,j]=True
        for k in ('guard','step','risk'):p[k][:n,j]=d[k][ix]
    p.update(N=d['epN'],task=d['keys'][:,0].astype(int),init=d['keys'][:,1].astype(int))
    assert p['alive'].sum()==meta['V'] and p['N'].sum()==meta['N']
    assert meta['rejected']==meta['duplicates']==0
    return p,path

def main():
    frozen=json.loads((ASTRA/'allocation_fits.json').read_text());records=[];inputs={}
    before=json.loads((HERE/'protected_before.json').read_text())
    cells=json.loads((ASTRA/'design.json').read_text())['cells']
    for cell in cells:
        model,suite,n=cell.split('_');library,sched=libpack(cell);priority=priorities(library)
        paths=[];sources=[]
        for file in sorted((ASTRA/'data').glob('*.json')):
            meta=json.loads(file.read_text())
            if meta['cell']!=cell or meta['method'] not in ('off','periodic'):continue
            p,data=devpack(meta)
            for f in (file,data):
                assert sha(f)==before[str(f)]
                inputs[str(f)]=sha(f)
            paths.append(prepare(p));sources.append(meta['root']+':'+meta['method'])
        if not paths:paths=[prepare(library)];sources=['library']
        for fit in (r for r in frozen if r['cell']==cell):
            K=round(sched.solve('periodic',fit['target'])[0],4)
            assert K==fit['reference_K'] and priority==fit['priority']
            assert sources==fit['calibration_sources']
            computed=solve(paths,model,K,priority['z'])
            for k,v in computed.items():assert v==fit[k],(cell,k,v,fit[k])
            records.append(dict(cell=cell,target=fit['target'],priority_exact=True,reference_K=K,
                                bias_exact=True,caps_exact=True,paths=sources,**computed))
        print('ALLOCATION_EXACT',cell,flush=True)
    write_json(HERE/'allocation_audit.json',dict(PASS=True,records=records,inputs_sha256=inputs,
        source_tables='R11 SHA-verified whole-episode-held-out tables; RMS squared once',
        no_R13_fit_inputs=True,no_A_data=True,no_pilot_refit=True))

if __name__=='__main__':main()
