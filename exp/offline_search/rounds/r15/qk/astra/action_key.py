"""Direct action query-key: predict actions but serve only library synthesis.

Library keys are exact sigma-scaled 10-control chunks plus state. Query predicts
that action key with a per-task ridge. State disambiguates repeated actions.
"""
import argparse
import pickle
import numpy as np
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,load_cell,dump
from exp.offline_search.rounds.r15.qk.astra.prototype import grids,fit_ridge,evaluate

def run(size):
    out=HERE/f'pi05_l10_{size}';tr=dict(np.load(DERIVED/f'pi05_l10_{size}/train.npz'));dp=DERIVED/'dev_pi05_l10_r12_devlooks_50L';mask=np.load(dp/f'unseen_{size}.npy');dev={k:v[mask] for k,v in dict(np.load(dp/'dev.npz')).items()}
    _,base,_,_,_=load_cell('l10',size);ref=np.load(out/'eval_reference.npz')['top']
    x=np.c_[grids(tr['img0']),grids(tr['img1']),tr['rs'][:,:8]];q=np.c_[grids(dev['img0']),grids(dev['img1']),dev['rs'][:,:8]]
    h=(tr['action'][:,:10,:7]/tr['sig']).reshape(len(x),70)
    results=[]
    for alpha in (.1,1.):
        model=fit_ridge(x,h,tr['task_id'],alpha);pred=np.zeros((len(q),70),np.float32)
        for t,p in model.items():
            rr=np.flatnonzero(dev['task_id']==t);pred[rr]=((q[rr]-p['xm'])/p['xs'])@p['coef']+p['ym']
        for stateweight in (.25,1.):
            lr=tr['rs'][:,:8].copy();qr=dev['rs'][:,:8].copy()
            norms={}
            for t in range(10):
                a=tr['task_id']==t;b=dev['task_id']==t;mu=lr[a].mean(0);sd=np.maximum(lr[a].std(0),.01);norms[t]={'mean':mu,'std':sd}
                lr[a]=(lr[a]-mu)/sd;qr[b]=(qr[b]-mu)/sd
            lx=np.c_[h/np.sqrt(70),stateweight*lr/np.sqrt(8)].astype(np.float32);qx=np.c_[pred/np.sqrt(70),stateweight*qr/np.sqrt(8)].astype(np.float32)
            met={t:[{'wf':np.eye(78,dtype=np.float32),'shift':np.zeros(78,np.float32)}]*2 for t in range(10)}
            name=f'action_qk_a{alpha:g}_s{stateweight:g}'
            s,r=evaluate(name,lx,qx,met,tr,dev,base,ref);s['predicted_action_err10']=float(np.sqrt(np.mean((pred-dev['label'][:,:10,:7].reshape(len(q),70)/np.tile(tr['sig'],10))**2,1)).mean());results.append(s)
            np.savez(out/f'eval_{name}.npz',**r)
            with (out/f'{name}.pkl').open('wb') as f:pickle.dump({'ridge':model,'state_norm':norms,'stateweight':stateweight},f)
            print(size,name,s['err5'],s['err10'],s['top1_agreement'],'direct regression',s['predicted_action_err10'],flush=True)
    dump(out/'action_key_results.json',{'methods':results})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=50);a=p.parse_args();run(a.size)
