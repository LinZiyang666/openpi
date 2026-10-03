"""Offline falsification of cheap transition responses on the same states.

No teacher is used for choosing actions. Existing shared head trained 0..19.
Evaluate only 20..29. Oracle best-neighbour error is a labelled lower bound,
explicitly unavailable online. No action metric is interpreted as SR.
"""
import json
import numpy as np
from ...round2.tools.data import dataset,features
from ...round2.inference import predict,correct
from .data import HERE,CELLS,library,subset,epmeans,interval,dump
from .screen import population,selected_entries


def err(a,b):return ((a[:,:,:6]-b[:,:,:6])**2).mean((1,2)),((a[:,:,6]>=0)!=(b[:,:,6]>=0)).mean(1)


def one(cell):
    d=population(cell,'A');d=subset(d,d['init']>=20)
    with np.load(HERE/'artifacts'/f'{cell}_monitor.npz',allow_pickle=False) as z:head={k:z[k] for k in z.files}
    meta=json.loads((HERE/'artifacts'/f'{cell}_monitor.json').read_text())
    score=predict(head,d['x'])[:,0];first,call,start=selected_entries(d,score,meta['threshold'])
    lib=library(cell);act=np.load(lib/'action.npy',mmap_mode='r')
    nxt=np.load(lib/'next.npy',mmap_mode='r');ep=np.load(lib/'episode.npy',mmap_mode='r');step=np.load(lib/'step.npy',mmap_mode='r')
    valid=np.flatnonzero(nxt>=0)
    if not np.all((ep[valid]==ep[nxt[valid]])&(step[nxt[valid]]==step[valid]+1)):raise ValueError('nonconsecutive library edge')
    r=d['rows'];w=d['weights'];chunks=act[r,:10,:7]
    candidates={}
    for shift in [1,2,4]:
        nr=r.copy()
        for _ in range(shift):nr=np.where(nxt[nr]>=0,nxt[nr],nr)
        candidates[f'advance{shift}']=np.einsum('nk,nkha->nha',w,act[nr,:10,:7])
    # Existing R2 motion/gripper head has disjoint-fit provenance.
    old=dataset(cell,'A');old=subset(old,old['init']>=20)
    if not np.array_equal(old['episode'],d['episode']) or not np.array_equal(old['seq'],d['seq']):raise ValueError('head row join')
    hp=HERE.parent/'round2/artifacts'/f'{cell}_temporal.npz'
    with np.load(hp,allow_pickle=False) as z:shared={k:z[k] for k in z.files}
    teacher_hat=predict(shared,features(old,True)).reshape(-1,10,7)
    candidates['shared_motion']=correct(shared,features(old,True),d['base'],.5,False)
    # Mode-specific reweighting: favor neighbours agreeing with confident
    # predicted gripper controls, retaining .1 of disagreeing kernel mass.
    confident=np.abs(teacher_hat[:,:,6])>=.5
    disagree=((chunks[:,:,:,6]>=0)!=(teacher_hat[:,None,:,6]>=0))&confident[:,None,:]
    penalty=disagree.sum(2)/np.maximum(confident.sum(1)[:,None],1)
    ww=w*np.exp(-np.log(10)*penalty);ww/=ww.sum(1,keepdims=True)
    mode=np.einsum('nk,nkha->nha',ww,chunks)
    candidates['gripper_mode']=mode
    # Entire coherent neighbour chunk nearest predicted corrected action.
    pred=candidates['shared_motion']
    distance=((chunks[:,:,:,:6]-pred[:,None,:,:6])**2).mean((2,3))
    pick=np.argmin(distance+np.maximum(distance.mean(1),1e-4)[:,None]*penalty,axis=1)
    candidates['coherent_chunk']=chunks[np.arange(len(r)),pick]
    base_m,base_g=err(d['base'],d['teacher']);out={}
    for name,a in candidates.items():
        m,g=err(a,d['teacher']);result={}
        for label,take in [('all',np.ones(len(m),bool)),('first_alert',first),('burst_anchors',call),('privileged_late_stall',d['late_stall'])]:
            if not take.any():continue
            sub=subset(d,take);eps,dm=epmeans((m-base_m)[take],sub);_,dg=epmeans((g-base_g)[take],sub)
            _,bm=epmeans(base_m[take],sub);_,bg=epmeans(base_g[take],sub)
            result[label]=dict(episodes=len(eps),anchors=int(take.sum()),base_motion=float(bm.mean()),
                relative_motion=float(dm.mean()/bm.mean()),gripper_change_pp=float(dg.mean()*100),
                motion_delta=interval(dm,eps),gripper_delta=interval(dg,eps))
        out[name]=result
    result=dict(cell=cell,eval_inits=list(range(20,30)),candidates=out)
    dump(HERE/'results'/f'reanchor_{cell}.json',result)
    print(cell,{k:{p:round(v['relative_motion'],3) for p,v in x.items()} for k,x in out.items()},flush=True)
    return result


if __name__=='__main__':dump(HERE/'results/reanchor.json',[one(c) for c in CELLS])
