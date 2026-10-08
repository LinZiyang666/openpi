"""Library-fitted external-camera alternative and causal pre-encoding choice.

The two alternatives are evaluated offline, but camera selection observes only
the saved full donor safety masses and current state, never both current views.
"""
from .d2_screen import *
from .atlas_reference import label_safe

def swapped(C):
    return dict(C,P=np.c_[C['P'][:,64:],C['P'][:,:64]].astype(np.float32))

def fit_alternate(cell,size):
    dest=D2/'fits'/f'{cell}_{size}';dest.mkdir(parents=True,exist_ok=True);path=dest/'alternate.pkl'
    if path.exists():
        with path.open('rb') as f:return pickle.load(f)
    start=time.monotonic();C,lib=load(cell,size);C=swapped(C);past=lag(C,2)
    old=dict(np.load(HERE/'final_atlas'/f'{cell}_{size}'/'table.npz',allow_pickle=False));C['sigma']=old['sigma']
    table={k:v.copy() for k,v in old.items()};H=np.c_[C['P'][past,:64],C['P'][:,64:],C['rs'],C['rs']-C['rs'][past]].astype(np.float32)
    F=np.c_[C['P'],C['rs']];Xw=H[:,64:];heads={};wheads={};wmets={};safe=np.zeros(len(H),bool);wd=[];we=[]
    ds=[];es=[]
    for task in range(10):
        cr=np.flatnonzero(C['task']==task);qr=cr[C['step'][cr]>=2];sr=cr[C['step'][cr]>=4]
        # Same full metric, with only its input coordinate order permuted.
        table[f'full_{task}_W']=old[f'full_{task}_W'][np.r_[64:128,0:64,128:136]]
        fm=metric_from(table,'full',task);m=Metric(H[cr],C['act'][cr],C['ep'][cr],C['sigma'])
        for k in ('W','shift','Z','z2'):table[f'half_{task}_{k}']=getattr(m,k)
        r=retrieve(m,H[qr],qr,cr,C);ds.append(r['distance']);es.append(C['ep'][qr])
        y=((C['act'][qr,:,:6]-r['action'][:,:,:6])/C['sigma'][:6]).reshape(len(qr),60)
        heads[task]=fit_head(feat(H[qr],r['action'],C['step'][qr],C['task'][qr],C['sigma']),y,anchor_weights(qr,C['ep'][qr]))
        ah=retrieve(m,H[sr],sr,cr,C);af=retrieve(fm,F[sr],sr,cr,C);ap=retrieve(fm,F[past[sr]],past[sr],cr,C)
        pf=C['progress'][ap['rows'][:,0]]
        fwd=lambda x:(C['progress'][x['rows'][:,0]]-pf)*np.maximum(lib.ep_len[x['rows'][:,0]]-1,1)>.5
        safe[sr]=label_safe(af['action'],ah['action'],C['sigma'],.15)&(fwd(af)==fwd(ah))
        wmet=Metric(Xw[cr],C['act'][cr],C['ep'][cr],C['sigma']);wmets[task]=(wmet,cr)
        rw=retrieve(wmet,Xw[qr],qr,cr,C);wd.append(rw['distance']);we.append(C['ep'][qr])
        yw=((C['act'][qr,:,:6]-rw['action'][:,:,:6])/C['sigma'][:6]).reshape(len(qr),60)
        wheads[task]=fit_head(feat(Xw[qr],rw['action'],C['step'][qr],C['task'][qr],C['sigma']),yw,anchor_weights(qr,C['ep'][qr]))
    table['safe']=safe;loeo_scale=wm(np.concatenate(ds),np.concatenate(es));ds=[];es=[]
    for fold in range(5):
        tr=np.flatnonzero(C['fold']!=fold);cf=dict(C,sigma=np.maximum(C['act'][tr,:5].std((0,1)),1e-6))
        for task in range(10):
            cr=tr[C['task'][tr]==task];qr=np.flatnonzero((C['fold']==fold)&(C['task']==task))
            m=Metric(H[cr],C['act'][cr],C['ep'][cr],cf['sigma']);r=retrieve(m,H[qr],qr,cr,cf);keep=C['step'][qr]>0
            ds.append(r['distance'][keep]);es.append(C['ep'][qr][keep])
    fit=dict(heads=heads,scale=wm(np.concatenate(ds),np.concatenate(es)),loeo_scale=loeo_scale,sigma=C['sigma'])
    wrist=dict(metrics=wmets,heads=wheads,scale=wm(np.concatenate(wd),np.concatenate(we)),sigma=C['sigma'])
    out=dict(table=table,fit=fit,wrist=wrist)
    with path.open('wb') as f:pickle.dump(out,f)
    dump(dest/'alternate_fit.json',dict(timestamp=now(),cell=cell,size=size,sha256=sha(path),seconds=time.monotonic()-start,pool='selected B library only',safe_rows=int(safe.sum())))
    print('alternate fit',cell,size,round(time.monotonic()-start,1),flush=True);return out

def build_alt(root,cell,size):
    info=fit_alternate(cell,size);table,fit,wrist=(info[k] for k in ('table','fit','wrist'))
    src=DEST/root/f'{cell}_{size}';dest=D2/root/f'{cell}_{size}';dest.mkdir(parents=True,exist_ok=True)
    A=dict(np.load(src/'looks.npz',allow_pickle=False));meta=json.loads((src/'extract.json').read_text())
    C,lib=load(cell,size);C=swapped(C);C['sigma']=table['sigma'];past=lag(C,2)
    with Path(meta['fit_path']).open('rb') as f:base=pickle.load(f)['method'].inner.base
    Q=dict(np.load(dest/'candidates.npz',allow_pickle=False));out={k:v.copy() for k,v in Q.items()}
    for i in np.flatnonzero(Q['reference']>=0):
        j=Q['reference'][i]
        out['safe_mass'][i]=float(np.sum(A['weights'][j]*(table['next2'][A['rows'][j]]>=0)*table['safe'][np.maximum(table['next2'][A['rows'][j]],0)]))
        c=candidate(saved_p0=A['P'][j,64:],current_wrist=A['P'][i,:64],current_rs=A['rs'][i],saved_rs=A['rs'][j],
            step=int(A['step'][i]),task=int(A['task'][i]),C=C,table=table,fit=fit,base=base,wrist=wrist,past=past,swap_full=True)
        for k,v in c.items():out[k][i]=v
    np.savez_compressed(dest/'alternate_candidates.npz',**out)
    # This comparison depends only on the previous real full donor mixture.
    external=out['safe_mass']>Q['safe_mass']
    mixed={k:np.where(external.reshape((-1,)+(1,)*(v.ndim-1)),out[k],v) for k,v in Q.items()}
    np.savez_compressed(dest/'chosen_candidates.npz',**mixed,external=external)
    dec=json.loads((src/'decisions.json').read_text());results=[]
    for mode,q in [('external',out),('prelook_choice',mixed)]:
        for cfg in [dict(name='frozen_atlas',witnesses=0),dict(name='consensus2_r0.1',witnesses=2,rms=.1,progress=.5,advance=.5),dict(name='consensus2_r0.05',witnesses=2,rms=.05,progress=.5,advance=.5)]:
            r=replay_gate(A,q,C,lib,dec,cfg,meta.get('periodic_setting'));stats=short(A,q,r,C,meta,dec)
            stats['external_halves']=int((r['half']&(external if mode=='prelook_choice' else True)).sum())
            results.append(dict(mode=mode,config=cfg,result=stats))
            np.savez_compressed(dest/f"{mode}_{cfg['name']}.npz",**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
    dump(dest/'camera_choice.json',dict(timestamp=now(),root=root,cell=cell,size=size,results=results,note=__doc__))
    dump(dest/'camera_choice_reads.json',sorted(READS));print('camera choice',root,cell,size,[(x['mode'],x['config']['name'],x['result']['H'],x['result']['missed'],x['result']['gripper_disagree_half']) for x in results],flush=True)

if __name__=='__main__':
    jobs=[]
    for root in ('r12_devlooks_50L','r12_devlooks_400'):
        for p in sorted((D2/root).glob('*/candidates.json')):
            a=json.loads(p.read_text());jobs.append((root,a['cell'],a['size']))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in [pool.submit(build_alt,*j) for j in jobs]:f.result()
