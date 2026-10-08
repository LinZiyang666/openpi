"""Pooled proposals on recorded B cache states. No counterfactual SR claims."""
from common import *
from partition import *
import argparse

def one(meta):
    cell=meta['cell'];model,suite,n=cell.split('_');C=load(model,suite,int(n))
    with np.load(HERE/'data'/f'{meta["root"]}__{meta["arm"]}.npz') as zz:d={k:zz[k] for k in zz.files}
    ix=np.flatnonzero(d['vision'].astype(bool)&np.isfinite(d['Xq']).all(1))
    L=len(C['ep']);tr=np.arange(L);tq=anchors(C,tr);sig=sigma(C,tr)
    scales,_=inner_scales(C,tr)
    # Negative episode IDs cannot collide with library donors.
    ep=-(d['ep'][ix].astype(int)+1);step=d['step'][ix].astype(int);task=d['task'][ix].astype(int)
    C={**C,'X':np.concatenate([C['X'],d['Xq'][ix]]),'P':np.concatenate([C['P'],d['Xq'][ix,:128]]),
       'ep':np.r_[C['ep'],ep],'step':np.r_[C['step'],step],'task':np.r_[C['task'],task],
       'act':np.concatenate([C['act'],d['policy_chunk'][ix]])}
    qv=np.arange(L,L+len(ix));gp=Geometry(C,tr,tr,sig)
    gt={t:Geometry(C,tr[C['task'][tr]==t],tr,sig) for t in range(10)}
    cache={};outputs=[]
    for scope,metric in [('task','task'),('pool','task'),('task','pool'),('pool','pool')]:
        result={}
        for split,qq in [('train',tq),('val',qv)]:
            pieces=[]
            for t in range(10):
                q=qq[C['task'][qq]==t];don=tr if scope=='pool' else tr[C['task'][tr]==t]
                if not len(q):continue
                g=gp if metric=='pool' else gt[t];g.prepare(C['X'][don]);pieces.append((q,retrieve(C,q,don,g)))
            q=np.concatenate([r[0] for r in pieces]);r={k:np.concatenate([r[1][k] for r in pieces]) for k in pieces[0][1]}
            order=np.argsort(q);result[split]=(q[order],{k:v[order] for k,v in r.items()})
        cache[(scope,metric)]=result
    ep_len={e:int(C['step'][:L][C['ep'][:L]==e].max()+1) for e in np.unique(C['ep'][:L])}
    for name,(scope,metric,hkind) in VARIANTS.items():
        tq2,rt=cache[(scope,metric)]['train'];vq,rv=cache[(scope,metric)]['val']
        ft=features(C,tq2,rt['chunk'],sig);fv=features(C,vq,rv['chunk'],sig)
        y=((C['act'][tq2,:,:6]-rt['chunk'][:,:,:6])/sig[:6]).reshape(-1,60)
        pred=np.zeros((len(vq),60),np.float32)
        for t in (range(10) if hkind=='task' else [None]):
            kt=np.ones(len(tq2),bool) if t is None else C['task'][tq2]==t
            kv=np.ones(len(vq),bool) if t is None else C['task'][vq]==t
            if not kv.any():continue
            X,XV=ft[kt],fv[kv]
            if t is None:X=X[:,:-10];XV=XV[:,:-10]
            w=train.anchor_weights(tq2[kt],C['ep'][tq2][kt]);w*=(len(tr) if t is None else np.sum(C['task'][tr]==t))/w.sum()
            h=train.fit_head(X,y[kt],w);pred[kv]=predict(h,XV)
        strength=.5*np.clip((2-rv['d']/scales[(scope,metric)])/1.25,0,1);strength[step==0]=0
        out=rv['chunk'].copy();out[:,:,:6]+=strength[:,None,None]*pred.reshape(-1,10,6)*sig[:6]
        raw=error(C,vq,rv['chunk'],sig);corr=error(C,vq,out,sig)
        grip=np.mean((C['act'][vq,:,6]>=0)!=(out[:,:,6]>=0),axis=1)
        for e in np.unique(ep):
            k=ep==e;valid=k&np.isfinite(corr);top=rv['top'][k]
            denom=np.array([max(ep_len[int(C['ep'][r])]-1,1) for r in top])
            prog=C['step'][top]/denom;st=step[k];span=0;g=[]
            for j in range(len(top)):
                if j:span=span+int(st[j]-st[j-1]) if (prog[j]-prog[j-1])*denom[j]<=.5 else 0
                g.append(span>=2)
            recorded=d['guard'][ix[k]].astype(bool)
            outputs.append(dict(cell=cell,root=meta['root'],variant=name,ep=int(e),task=int(task[k][0]),
                looks=int(k.sum()),shadow_looks=int(valid.sum()),mse_raw=float(raw[valid].mean()) if valid.any() else None,
                mse_corrected=float(corr[valid].mean()) if valid.any() else None,
                frozen_mse=float(d['risk'][ix[valid]].mean()) if valid.any() else None,
                grip_error=float(grip[valid].mean()) if valid.any() else None,
                top1_same_task=float(rv['same'][k].mean()),donor_task_mass=float(rv['purity'][k].mean()),
                proposed_guard_rate=float(np.mean(g)),recorded_guard_rate=float(recorded.mean()),
                guard_disagreement=float(np.mean(np.asarray(g)!=recorded))))
    csvout(HERE/f'dev_partition_{cell}.csv',outputs)
    print(cell,'B looks',len(ix),'shadow',int(np.isfinite(d['policy_chunk'][ix]).all(axis=(1,2)).sum()),flush=True)

def main():
    fence();metas=[]
    for mp in sorted((HERE/'data').glob('r12_devlooks*json')):
        m=read(mp)
        if m['cell'] in CELLS and m['method']=='off':
            assert m['r14_projected'];metas.append(m)
    with ThreadPoolExecutor(4) as ex:list(ex.map(one,metas))
    provenance('dev_partition')
if __name__=='__main__':main()
