"""Build final selected-library safety tables for sol; no deployable server fit.

PCA files are B-subset hash-verified. Full recipe and half heads/scales must still
be built by sol according to SPEC; this artifact supplies the pre-look atlas.
"""
from .common import *
install()
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .retrieval import Metric,retrieve,lag
from .atlas_reference import AtlasConfig,label_safe

def run(cell,size):
    C,lib=load(cell,size);cfg=AtlasConfig();P,rs=C['P'],C['rs'];past=lag(C,2)
    sigma=np.maximum(C['act'][:,:5].std((0,1)),1e-6);C['sigma']=sigma
    F=np.c_[P,rs];H=np.c_[P[past,:64],P[:,64:],rs,rs-rs[past]]
    safe=np.zeros(len(P),bool);agreement=np.zeros(len(P),bool)
    lens=np.asarray(lib.ep_len);metrics={}
    for t in range(10):
        cr=np.flatnonzero(C['task']==t);qr=cr[C['step'][cr]>=4]
        full=Metric(F[cr],C['act'][cr],C['ep'][cr],sigma);half=Metric(H[cr],C['act'][cr],C['ep'][cr],sigma)
        af=retrieve(full,F[qr],qr,cr,C);ah=retrieve(half,H[qr],qr,cr,C)
        ap=retrieve(full,F[past[qr]],past[qr],cr,C)
        pf=C['progress'][ap['rows'][:,0]]
        forward_full=(C['progress'][af['rows'][:,0]]-pf)*np.maximum(lens[af['rows'][:,0]]-1,1)>.5
        forward_half=(C['progress'][ah['rows'][:,0]]-pf)*np.maximum(lens[ah['rows'][:,0]]-1,1)>.5
        agreement[qr]=label_safe(af['action'],ah['action'],sigma,cfg.label_rms)
        safe[qr]=agreement[qr]&(forward_full==forward_half)
        for name,m in [('full',full),('half',half)]:
            for field in ('W','shift','Z','z2'):metrics[f'{name}_{t}_{field}']=getattr(m,field)
            metrics[f'{name}_{t}_rows']=cr
    nxt=np.full(len(P),-1,int);valid=np.flatnonzero(C['prev']>=0);nxt[C['prev'][valid]]=valid
    next2=np.where(nxt>=0,nxt[np.maximum(nxt,0)],-1)
    d=HERE/'final_atlas'/f'{cell}_{size}';d.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(d/'table.npz',row_subset=C['row'],safe=safe,action_agreement=agreement,next2=next2,
                        rs=rs,state_scale=np.maximum(rs.std(0),1e-6),sigma=sigma,**metrics)
    info=dict(timestamp=now(),cell=cell,size=size,rows=len(P),safe_rows=int(safe.sum()),config=vars(cfg),
              scope='safety table and main retrieval metric reference only; not a plugin fit',
              progress_label=True,fit_pool='B selected library only',pca=C['provenance'],table_sha256=sha(d/'table.npz'))
    dump(d/'fit.json',info);dump(d/'reads.json',sorted(READS));return info

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(run,c,s) for s in (50,200,500) for c in ('pi05_l10','pi05_spatial','groot_l10','groot_spatial')]
        for f in fs:print(json.dumps(f.result()),flush=True)
