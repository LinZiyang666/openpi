"""DIRECTIVE_2 cache research. Library fits; B-recording fixed-state shadows.

The gate never receives the current missing camera. Acquisition rejection
completes that camera before any guard or action is committed.
"""
from .dev_assess import *
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from concurrent.futures import ProcessPoolExecutor

D2=HERE/'directive2'

def query_many(m, cr, X, C):
    d=m.dist(X);idx=np.argsort(d,axis=1,kind='stable')[:,:16]
    ds=np.take_along_axis(d,idx,1).astype(np.float64)
    w=_kernel_w(ds-ds[:,:1],C['kref']);w=(w/w.sum(1,keepdims=True)).astype(np.float32)
    rows=cr[idx];raw=np.einsum('qk,qktc->qtc',w,C['act'][rows],optimize=False)
    return rows,w,ds[:,0],raw

def correct_many(head,scale,X,raw,dist,step,task,sigma):
    f=feat(X,raw,np.full(len(X),step),np.full(len(X),task),sigma)
    res=predict(head,f).reshape(-1,10,6)*sigma[:6]
    out=raw.copy();out[:,:,:6]+=.5*(res*np.clip((2-dist/scale)/1.25,0,1).astype(np.float32)[:,None,None])
    return out

def fit_wrist(cell,size):
    dest=D2/'fits'/f'{cell}_{size}';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'wrist.pkl'
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    C,lib=load(cell,size);C['sigma']=np.maximum(C['act'][:,:5].std((0,1)),1e-6)
    past=lag(C,2);X=np.c_[C['P'][:,64:],C['rs'],C['rs']-C['rs'][past]].astype(np.float32)
    mets={};heads={};ds=[];es=[]
    for task in range(10):
        cr=np.flatnonzero(C['task']==task);qr=cr[C['step'][cr]>=2]
        m=Metric(X[cr],C['act'][cr],C['ep'][cr],C['sigma']);mets[task]=(m,cr)
        r=retrieve(m,X[qr],qr,cr,C)
        y=((C['act'][qr,:,:6]-r['action'][:,:,:6])/C['sigma'][:6]).reshape(len(qr),60)
        heads[task]=fit_head(feat(X[qr],r['action'],C['step'][qr],C['task'][qr],C['sigma']),y,anchor_weights(qr,C['ep'][qr]))
        ds.append(r['distance']);es.append(C['ep'][qr])
    # Diagnostic wrist head is a disagreement sensor, never the served branch.
    out=dict(metrics=mets,heads=heads,scale=wm(np.concatenate(ds),np.concatenate(es)),sigma=C['sigma'])
    with path.open('wb') as f:pickle.dump(out,f)
    dump(dest/'fit.json',dict(timestamp=now(),pool='selected B library only',sha256=sha(path),
        wrist_scale='training LOEO; sensor only',rows=len(C['ep']),cell=cell,size=size))
    return out

def candidate(*,saved_p0,current_wrist,current_rs,saved_rs,step,task,C,table,fit,base,wrist,past,swap_full=False):
    """Pure legal-input proposal. No current v0 / full answer is accepted."""
    x=half_features(last_full_v0=saved_p0,current_wrist=current_wrist,current_rs=current_rs,last_full_rs=saved_rs)
    rows,w,d,raw,corr,_=half_query(table,fit,C,task,x,step)
    xw=x[64:][None];wm_,cr=wrist['metrics'][task]
    wr,ww,wd,wa=query_many(wm_,cr,xw,C)
    wc=correct_many(wrist['heads'][task],wrist['scale'],xw,wa,wd,step,task,C['sigma'])
    # Transport only library camera innovations relative to two-slot predecessors.
    innovations=C['P'][rows,:64]-C['P'][past[rows],:64]
    hypotheses=np.r_[np.einsum('k,kd->d',w,innovations)[None],innovations]
    X=np.c_[saved_p0[None]+hypotheses,np.broadcast_to(current_wrist,(17,64)),np.broadcast_to(current_rs,(17,8))].astype(np.float32)
    fm=metric_from(table,'full',task);fcr=table[f'full_{task}_rows']
    fr,fw,fd,fa=query_many(fm,fcr,X,C)
    headX=np.c_[X[:,64:128],X[:,:64],X[:,128:]] if swap_full else X
    fc=correct_many(base.heads[str(task)],base.distance_scale,headX,fa,fd,step,task,base.sig_head)
    rms=lambda a:np.sqrt(np.mean(((a[:,:,:6]-corr[None,:,:6])/C['sigma'][:6])**2,(1,2)))
    return dict(rows=rows,weights=w,distance=d,raw=raw,corrected=corr,
        witness_rows=np.r_[wr[:,0],fr[:,0]],witness_rms=np.r_[rms(wc),rms(fc)],
        witness_grip=np.r_[np.any((wc[:,:,6]>=0)!=(corr[None,:,6]>=0),1),np.any((fc[:,:,6]>=0)!=(corr[None,:,6]>=0),1)])

def build(root,cell,size):
    start=time.monotonic();src=DEST/root/f'{cell}_{size}';dest=D2/root/f'{cell}_{size}';dest.mkdir(parents=True,exist_ok=True)
    A=dict(np.load(src/'looks.npz',allow_pickle=False));meta=json.loads((src/'extract.json').read_text())
    C,lib=load(cell,size);table=dict(np.load(HERE/'final_atlas'/f'{cell}_{size}'/'table.npz',allow_pickle=False));C['sigma']=table['sigma']
    with (DEST/'fits'/f'{cell}_{size}'/'half.pkl').open('rb') as f:fit=pickle.load(f)
    with Path(meta['fit_path']).open('rb') as f:base=pickle.load(f)['method'].inner.base
    wrist=fit_wrist(cell,size);past=lag(C,2);n=len(A['step']);lookup={(u,int(s)):i for i,(u,s) in enumerate(zip(A['uid'],A['step']))}
    out=dict(reference=np.full(n,-1,int),rows=A['rows'].copy(),weights=A['weights'].copy(),
        distance=A['distance'].copy(),raw=A['raw'].copy(),corrected=A['full'].copy(),
        witness_rows=np.zeros((n,18),int),witness_rms=np.full((n,18),np.inf),witness_grip=np.ones((n,18),bool),
        support=np.zeros(n),safe_mass=np.zeros(n),state_error=np.full(n,np.inf))
    for i in range(n):
        j=lookup.get((A['uid'][i],int(A['step'][i])-2),-1)
        if j<0 or A['step'][i]<4:continue
        out['reference'][i]=j
        p=plan(AtlasConfig(),step=int(A['step'][i]),last_full_step=int(A['step'][j]),current_rs=A['rs'][i],last_full_rs=A['rs'][j],
            rows=A['rows'][j],weights=A['weights'][j],next2=table['next2'],library_rs=table['rs'],safe=table['safe'],state_scale=table['state_scale'])
        for k in ('support','safe_mass','state_error'):out[k][i]=p[k]
        c=candidate(saved_p0=A['P'][j,:64],current_wrist=A['P'][i,64:],current_rs=A['rs'][i],saved_rs=A['rs'][j],
            step=int(A['step'][i]),task=int(A['task'][i]),C=C,table=table,fit=fit,base=base,wrist=wrist,past=past)
        for k,v in c.items():out[k][i]=v
        if i%500==0:print('consensus',root,cell,size,i,round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(dest/'candidates.npz',**out)
    dump(dest/'candidates.json',dict(timestamp=now(),cell=cell,size=size,root=root,seconds=time.monotonic()-start,
        eligible_pairs=int((out['reference']>=0).sum()),source_looks_sha256=sha(src/'looks.npz'),
        candidate_sha256=sha(dest/'candidates.npz'),fit_pool='selected B library only',
        note='Design revised after frozen atlas dev inspection; no dev-fit parameters in these proposals. GR00T uses recorded joint-path wrist until mismatch study.'))
    dump(dest/'candidate_reads.json',sorted(READS))
    return dict(root=root,cell=cell,size=size,seconds=time.monotonic()-start)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--roots',nargs='+',default=['r12_devlooks_50L','r12_devlooks_400']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    jobs=[]
    for root in args.roots:
        for p in sorted((DEST/root).glob('*/extract.json')):
            m=json.loads(p.read_text());jobs.append((root,m['cell'],m['size']))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in [pool.submit(build,*j) for j in jobs]:print(f.result(),flush=True)
