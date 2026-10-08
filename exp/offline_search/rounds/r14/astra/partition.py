"""Economical B-library cross-validation of scope, metric and head partitions.

All training-fold rows fit metrics and remain donors. Deterministic evenly spaced
anchor samples fit/evaluate heads: a CPU diagnostic, not the production fitter.
PCA is the verified selected-library PCA, shared and transductive across folds.
"""
from common import *
import numpy as np
from exp.offline_search.rounds.r10 import train
from exp.offline_search.rounds.r10.astra.offline import load, features, sigma, median
from exp.offline_search.rounds.r10.recipe.recipe import predict
from exp.offline_search.rounds.r02.g1_awm import awm
from concurrent.futures import ThreadPoolExecutor
import time

ANCHORS = 16
FOLDS = 5
VARIANTS = {'base':('task','task','task'), 'scope':('pool','task','task'),
            'metric':('task','pool','task'), 'head':('task','task','pool'),
            'scope_metric':('pool','pool','task'), 'scope_only':('task','pool','pool'),
            'all_pool':('pool','pool','pool')}

def metric_fit(X,H,ep,block=256):
    """AWM full-rank equation with bounded-memory action-neighbor discovery."""
    X,H=np.asarray(X,np.float64),np.asarray(H,np.float64)
    mean,std=X.mean(0),X.std(0)+1e-6
    Xn=(X-mean)/std; h2=(H*H).sum(1); gram=np.zeros((X.shape[1],X.shape[1])); count=0
    for lo in range(0,len(X),block):
        hi=min(lo+block,len(X))
        D=h2[lo:hi,None]-2*H[lo:hi]@H.T+h2[None,:]
        D[ep[lo:hi,None]==ep[None,:]]=np.inf
        k=min(3,len(X)-1); ix=np.argpartition(D,k,axis=1)[:,:k]
        ok=np.isfinite(np.take_along_axis(D,ix,1))
        dif=(Xn[lo:hi,None]-Xn[ix])[ok]
        gram+=dif.T@dif;count+=len(dif)
    sw=gram/max(count,1)
    W=np.linalg.cholesky(np.linalg.inv(sw+.1*np.trace(sw)/len(sw)*np.eye(len(sw))))
    return mean,std,W

class Geometry:
    def __init__(self,C,fit_rows,donors,sig):
        X=C['X'][fit_rows].astype(float)
        H=(C['act'][fit_rows,:5].astype(float)/sig.astype(float)).reshape(len(X),35)
        self.main=metric_fit(X,H,C['ep'][fit_rows])
        early=C['step'][fit_rows]<=2
        self.early=metric_fit(X[early],H[early],C['ep'][fit_rows][early])
        self.prepare(C['X'][donors])
    def prepare(self,X):
        self.codes=[]
        for m,s,w in (self.main,self.early):
            z=(((X.astype(float)-m)/s)@w).astype(np.float32)
            self.codes.append((z,np.sum(z.astype(float)**2,axis=1).astype(np.float32),
                              (w/s[:,None]).astype(np.float32),((m/s)@w).astype(np.float32)))
    def distance(self,X,step):
        out=np.empty((len(X),len(self.codes[0][0])),np.float32)
        for early in (False,True):
            ix=np.flatnonzero((step==0)==early)
            if not len(ix):continue
            z,z2,w,b=self.codes[int(early)];q=X[ix]@w-b
            out[ix]=np.sqrt(np.maximum(z2[None]-2*q@z.T+(q*q).sum(1)[:,None],0))
        return out

def anchors(C,rows):
    result=[]
    for e in np.unique(C['ep'][rows]):
        r=rows[C['ep'][rows]==e]
        result.extend(r[np.unique(np.linspace(0,len(r)-1,min(ANCHORS,len(r))).astype(int))])
    return np.array(result,int)

def retrieve(C,query,donors,geom):
    chunks=[];near=[];same=[];purity=[];tops=[]
    for lo in range(0,len(query),64):
        qr=query[lo:lo+64];D=geom.distance(C['X'][qr],C['step'][qr])
        D[C['ep'][qr,None]==C['ep'][donors][None]]=np.inf
        ix=np.argsort(D,axis=1,kind='stable')[:,:16]
        d=np.take_along_axis(D,ix,1).astype(float);assert np.isfinite(d).all()
        w=awm._kernel_w(d-d[:,:1],C['kref']);w=(w/w.sum(1,keepdims=True)).astype(np.float32)
        r=donors[ix];chunks.append(np.einsum('qk,qktc->qtc',w,C['act'][r],optimize=False))
        near.extend(d[:,0]); tops.extend(r[:,0]);same.extend(C['task'][r[:,0]]==C['task'][qr])
        purity.extend((w*(C['task'][r]==C['task'][qr,None])).sum(1))
    return dict(chunk=np.concatenate(chunks),d=np.array(near),same=np.array(same),
                purity=np.array(purity),top=np.array(tops))

def error(C,q,chunk,sig):return np.mean(((C['act'][q,:,:6]-chunk[:,:,:6])/sig[:6])**2,axis=(1,2))

def inner_scales(C,tr):
    # One inner episode holdout per outer fold, rather than a leaky train-LOEO
    # scale; every task retains at least three donor episodes in the small cell.
    hold=[]
    for t in range(10):
        es=np.unique(C['ep'][tr][C['task'][tr]==t]);hold.extend(es[::4])
    ir=tr[~np.isin(C['ep'][tr],hold)];iq=anchors(C,tr[np.isin(C['ep'][tr],hold)])
    sig=sigma(C,ir);gp=Geometry(C,ir,ir,sig)
    gt={t:Geometry(C,ir[C['task'][ir]==t],ir,sig) for t in range(10)}
    scales={}
    for scope,metric in [('task','task'),('pool','task'),('task','pool'),('pool','pool')]:
        ds=[];eps=[]
        for t in range(10):
            qr=iq[(C['task'][iq]==t)&(C['step'][iq]>0)]
            donors=ir if scope=='pool' else ir[C['task'][ir]==t]
            g=gp if metric=='pool' else gt[t];g.prepare(C['X'][donors])
            r=retrieve(C,qr,donors,g);ds.extend(r['d']);eps.extend(C['ep'][qr])
        scales[(scope,metric)]=median(np.array(ds),np.array(eps))
    return scales,dict(metric_episodes=np.unique(C['ep'][ir]),heldout_episodes=np.array(hold))

def run_cell(cell):
    tic=time.monotonic();model,suite,n=cell.split('_');C=load(model,suite,int(n))
    records=[];audits=[]
    for fold in range(FOLDS):
        tr=np.flatnonzero(C['fold']!=fold);va=np.flatnonzero(C['fold']==fold)
        tq,vq=anchors(C,tr),anchors(C,va);sig=sigma(C,tr)
        assert not set(C['ep'][tr])&set(C['ep'][va])
        scales,inner_audit=inner_scales(C,tr)
        # Each query metric scores every candidate in that query's coordinate system.
        gp=Geometry(C,tr,tr,sig)
        gt={t:Geometry(C,tr[C['task'][tr]==t],tr,sig) for t in range(10)}
        fold_caches={}
        for scope,metric in [('task','task'),('pool','task'),('task','pool'),('pool','pool')]:
            joined={}
            for split,qs in [('train',tq),('val',vq)]:
                pieces=[]
                for t in range(10):
                    q=qs[C['task'][qs]==t];don=tr if scope=='pool' else tr[C['task'][tr]==t]
                    g=gp if metric=='pool' else gt[t]
                    g.prepare(C['X'][don]);r=retrieve(C,q,don,g)
                    pieces.append((q,r))
                qs=np.concatenate([q for q,r in pieces]);
                joined[split]=(qs,{k:np.concatenate([r[k] for q,r in pieces]) for k in pieces[0][1]})
            fold_caches[(scope,metric)]=joined
        for name,(scope,metric,headkind) in VARIANTS.items():
            tq2,rt=fold_caches[(scope,metric)]['train'];vq2,rv=fold_caches[(scope,metric)]['val']
            scale=scales[(scope,metric)]
            ft=features(C,tq2,rt['chunk'],sig);fv=features(C,vq2,rv['chunk'],sig)
            yt=((C['act'][tq2,:,:6]-rt['chunk'][:,:,:6])/sig[:6]).reshape(len(tq2),60)
            pred=np.zeros((len(vq2),60),np.float32)
            for t in (range(10) if headkind=='task' else [None]):
                kt=np.ones(len(tq2),bool) if t is None else C['task'][tq2]==t
                kv=np.ones(len(vq2),bool) if t is None else C['task'][vq2]==t
                X,Y=ft[kt],yt[kt];XV=fv[kv]
                if t is None:X=X[:,:-10];XV=XV[:,:-10]
                weight=train.anchor_weights(tq2[kt],C['ep'][tq2][kt])
                # Preserve full represented-row mass per task despite quadrature.
                fullmass=len(tr) if t is None else np.sum(C['task'][tr]==t)
                weight*=fullmass/weight.sum()
                h=train.fit_head(X,Y,weight);pred[kv]=predict(h,XV)
            strength=.5*np.clip((2-rv['d']/scale)/1.25,0,1)
            strength[C['step'][vq2]==0]=0
            out=rv['chunk'].copy();out[:,:,:6]+=strength[:,None,None]*pred.reshape(-1,10,6)*sig[:6]
            m0=error(C,vq2,rv['chunk'],sig);m1=error(C,vq2,out,sig)
            blend=rv['chunk'].copy();blend[:,:,:6]+=.5*pred.reshape(-1,10,6)*sig[:6]
            mb=error(C,vq2,blend,sig)
            grip=np.mean((C['act'][vq2,:,6]>=0)!=(out[:,:,6]>=0),axis=1)
            for e in np.unique(C['ep'][vq2]):
                k=C['ep'][vq2]==e
                records.append(dict(cell=cell,variant=name,fold=fold,ep=int(e),task=int(C['task'][vq2[k][0]]),
                    anchors=int(k.sum()),mse_raw=float(m0[k].mean()),mse_corrected=float(m1[k].mean()),
                    mse_blend05=float(mb[k].mean()),
                    grip_error=float(grip[k].mean()),top1_same_task=float(rv['same'][k].mean()),
                    donor_task_mass=float(rv['purity'][k].mean()),gate_mean=float(strength[k].mean()),scale=scale))
        audits.append(dict(fold=fold,train_episodes=np.unique(C['ep'][tr]),heldout_episodes=np.unique(C['ep'][va]),
                           metric_fit_rows=len(tr),head_train_anchors=len(tq),validation_anchors=len(vq),inner=inner_audit))
        print(cell,'fold',fold,'seconds',round(time.monotonic()-tic,1),flush=True)
    csvout(HERE/f'partition_{cell}.csv',records)
    dump(HERE/f'partition_{cell}_audit.json',dict(source_fit_sha=C['source_fit_sha256'],folds=audits,
        anchors_per_episode=ANCHORS,elapsed_s=time.monotonic()-tic))
    return records

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--cell',action='append');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    fence();cells=a.cell or CELLS
    with ThreadPoolExecutor(a.workers) as ex:list(ex.map(run_cell,cells))
    provenance('partition')

if __name__=='__main__':main()
