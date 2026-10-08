"""Alternative wrist representations from B-library keys/tokens/images."""
from .common import *
install()
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .retrieval import Metric,retrieve
from exp.offline_search.rounds.r02.g1_awm.awm import pca_fit

def run(cell,size):
    start=time.monotonic();C,lib=load(cell,size)
    d=HERE/'representations'/f'{cell}_{size}';d.mkdir(parents=True,exist_ok=True)
    key=np.asarray(lib.parent.key_v1[lib.rows],np.float32).reshape(-1,4,4,2048)
    raw={'mean':key.mean((1,2)),
         'pool2':key.reshape(len(key),2,2,2,2,2048).mean((2,4)).reshape(len(key),-1),
         'pca128':key.reshape(len(key),-1)}
    tok=lib.parent.tok('v1'); ti=lib.parent.tok_index[lib.rows]; assert (ti>=0).all()
    raw['max']=np.empty((len(key),2048),np.float32)
    # Preserve an 8x8 grid while reducing each channel by a fixed sparse sketch.
    raw['fine8']=np.empty((len(key),64*32),np.float32)
    signs=np.random.default_rng(1203).choice([-1.,1.],2048).astype(np.float32)
    for lo in range(0,len(key),24):
        hi=min(lo+24,len(key));T=np.asarray(tok[ti[lo:hi]],np.float32)
        raw['max'][lo:hi]=T.max(1)
        S=(T*signs).reshape(-1,256,64,32).sum(2)/8
        raw['fine8'][lo:hi]=S.reshape(-1,8,2,8,2,32).mean((2,4)).reshape(hi-lo,-1)
    img=lib.parent.tok('img1')
    if img is not None:
        raw['rgb']=np.empty((len(key),16*16*3),np.float32)
        for lo in range(0,len(key),64):
            hi=min(lo+64,len(key));I=np.asarray(img[ti[lo:hi]],np.float32)/255
            raw['rgb'][lo:hi]=I.reshape(-1,16,14,16,14,3).mean((2,4)).reshape(hi-lo,-1)
    reps={};bases={}
    for name,X in raw.items():
        # Same fixed-library unsupervised PCA convention as the R10 reference.
        mu,B,P=pca_fit(X,keep=128 if name=='pca128' else 64)
        reps[name]=np.c_[P,C['rs']].astype(np.float32)
        np.savez_compressed(d/f'{name}_pca.npz',mean=mu,basis=B,proj=P)
        print(f'rep {cell}_{size} {name} secs={time.monotonic()-start:.1f}',flush=True)
    reps['mean_plus64']=np.c_[reps['mean'][:,:64],C['P'][:,64:],C['rs']]
    if 'rgb' in reps:reps['rgb_plus64']=np.c_[reps['rgb'][:,:64],C['P'][:,64:],C['rs']]
    summary=[]
    for name,X in reps.items():
        vals=[];eps=[]
        for f in range(5):
            tr=np.flatnonzero(C['fold']!=f);sig=np.maximum(C['act'][tr,:5].std((0,1)),1e-6);C['sigma']=sig
            for t in range(10):
                cr=tr[C['task'][tr]==t];qr=np.flatnonzero((C['fold']==f)&(C['task']==t)&(C['step']>=2)&(C['step']%2==0))
                met=Metric(X[cr],C['act'][cr],C['ep'][cr],sig)
                r=retrieve(met,X[qr],qr,cr,C)
                vals.append(np.mean(((r['action'][:,:,:6]-C['act'][qr,:,:6])/sig[:6])**2,(1,2)));eps.append(C['ep'][qr])
        summary.append(dict(variant=name,motion_mse=mean(np.concatenate(vals),np.concatenate(eps))))
    result=dict(timestamp=now(),cell=cell,size=size,seconds=time.monotonic()-start,variants=summary,
                pca_convention='fixed selected B library, heldout episode actions excluded from metric/sigma/donors')
    dump(d/'summary.json',result);dump(d/'reads.json',sorted(READS));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sizes',type=int,nargs='+',default=[50]);args=ap.parse_args()
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(run,c,s) for s in args.sizes for c in ('pi05_l10','pi05_spatial','groot_l10','groot_spatial')]
        for f in fs:print(json.dumps(f.result()),flush=True)
