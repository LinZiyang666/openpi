"""Third design: local spatial image residual over a free-state teacher fit."""
import argparse
import pickle
import time
import numpy as np
from scipy.spatial.distance import cdist
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,load_cell,dump
from exp.offline_search.rounds.r15.qk.astra.prototype import grids,evaluate,teacher_metric,reference_codes

def fit(x,rs,y,task,length,alpha):
    models={}
    for t in range(10):
        rr=np.flatnonzero(task==t);a=x[rr];r=rs[rr];target=y[rr]
        xm=a.mean(0);xs=np.maximum(a.std(0),.03);a=(a-xm)/xs
        rm=r.mean(0);rsd=np.maximum(r.std(0),.03);r=np.c_[np.ones(len(rr)),(r-rm)/rsd]
        rb=np.linalg.solve(r.T@r+.1*np.eye(r.shape[1]),r.T@target)
        residual=target-r@rb
        d=cdist(a,a,'sqeuclidean');ell=float(np.median(d[d>0]))*length
        k=np.exp(-d/ell);coef=np.linalg.solve(k+alpha*np.eye(len(k)),residual)
        models[t]={k:v.astype(np.float32) if isinstance(v,np.ndarray) else v for k,v in dict(xm=xm,xs=xs,rm=rm,rs=rsd,rb=rb,centers=a,center2=(a*a).sum(1),ell=ell,coef=coef).items()}
    return models

def predict(models,x,rs,task):
    out=np.empty((len(x),128),np.float32)
    for t,p in models.items():
        rr=np.flatnonzero(task==t);a=(x[rr]-p['xm'])/p['xs'];r=np.c_[np.ones(len(rr)),(rs[rr]-p['rm'])/p['rs']]
        k=np.exp(-cdist(a,p['centers'],'sqeuclidean')/p['ell'])
        out[rr]=r@p['rb']+k@p['coef']
    return out

def run(suite,size,root):
    out=HERE/f'pi05_{suite}_{size}';tr=dict(np.load(DERIVED/f'pi05_{suite}_{size}/train.npz'))
    dp=DERIVED/f'dev_pi05_{suite}_{root}';mask=np.load(dp/f'unseen_{size}.npy');dev={k:v[mask] for k,v in dict(np.load(dp/'dev.npz')).items()}
    _,base,_,_,_=load_cell(suite,size);P=np.c_[tr['P0'],tr['P1']];lx=np.c_[P,tr['rs'][:,:8]]
    x=np.c_[grids(tr['img0']),grids(tr['img1'])];q=np.c_[grids(dev['img0']),grids(dev['img1'])]
    ref=np.load(out/'eval_reference.npz')['top'];results=[]
    for length,alpha in ((.25,.01),(1.,.01),(.25,.001),(1.,.001)):
        start=time.perf_counter();m=fit(x,tr['rs'][:,:8],P,tr['task_id'],length,alpha)
        pred=predict(m,q,dev['rs'][:,:8],dev['task_id']);name=f'local_residual_krr_l{length:g}_a{alpha:g}'
        s,r=evaluate(name,lx,np.c_[pred,dev['rs'][:,:8]],teacher_metric(base),tr,dev,base,ref,libcodes=reference_codes(base,tr))
        s['fit_predict_seconds']=time.perf_counter()-start;results.append(s)
        np.savez(out/f'eval_{name}.npz',**r)
        with (out/f'{name}.pkl').open('wb') as f:pickle.dump(m,f)
        print(size,name,s['err5'],s['err10'],s['top1_agreement'],flush=True)
    dump(out/'kernel_results.json',{'methods':results,'scope':'library-trained state affine teacher fit plus spatial image RBF residual; hyperparameter selection only on unseen B-dev'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',default='l10');p.add_argument('--size',type=int,default=50);p.add_argument('--devroot',default='r12_devlooks_50L');a=p.parse_args();run(a.suite,a.size,a.devroot)
