"""Observable half-view agreement gates; unobserved full look is evaluation only."""
from .common import *
install()
import numpy as np
import argparse

ap=argparse.ArgumentParser();ap.add_argument('--dynamic',action='store_true');args=ap.parse_args()
category='correction_dynamic' if args.dynamic else 'correction'
outname='gates_dynamic' if args.dynamic else 'gates'

out=[]
for path in sorted((HERE/category).glob('*/summary.json')):
    info=json.loads(path.read_text());cell,size=info['cell'],info['size'];C,lib=load(cell,size)
    p=path.parent
    def read(name):
        with np.load(p/f'{name}.npz') as z:return {k:np.array(z[k]) for k in z.files}
    meta=read('meta');legacy=read('full');full=read('full_delta') if args.dynamic else legacy;half=read('memory2');wrist=read('wrist')
    with np.load(HERE/'retrieval'/f'{cell}_{size}'/'meta.npz') as z:
        assert np.array_equal(z['row'],meta['row']);sig=np.array(z['sigma'])
    score=np.sqrt(np.mean(((half['action'][:,:,:6]-wrist['action'][:,:,:6])/sig[:,None,:6])**2,(1,2)))
    half['full_delta_mse']=np.mean(((half['action'][:,:,:6]-full['action'][:,:,:6])/sig[:,None,:6])**2,(1,2))
    half['grip_flip_full']=np.mean((half['action'][:,:,6]>=0)!=(full['action'][:,:,6]>=0),1)
    grip_equal=((half['action'][:,:,6]>=0)==(wrist['action'][:,:,6]>=0)).all(1)
    local=np.searchsorted(C['row'],meta['row'])
    pos={int(r):i for i,r in enumerate(local)}
    prevlocal=C['prev'][np.maximum(C['prev'][local],0)]
    prevpos=np.array([pos.get(int(r),-1) for r in prevlocal])
    valid=prevpos>=0
    prevtop=full['rows'][np.maximum(prevpos,0),0]
    ep_len=np.asarray(lib.ep_len)
    previous=C['progress'][prevtop]
    advance={}
    for name,r in [('full',full),('half',half),('wrist',wrist)]:
        top=r['rows'][:,0]
        advance[name]=(C['progress'][top]-previous)*np.maximum(ep_len[top]-1,1)>.5
    # This uses only previous full progress, not current full data.
    progress_ok=advance['half']&advance['wrist']&valid
    candidates=[]
    for threshold in (.05,.075,.1,.125,.15,.2,.3):
        for guard in (False,True):
            accept=(score<=threshold)&grip_equal
            if guard:accept &= progress_ok
            selected=np.zeros(len(accept),bool)
            for ep in np.unique(meta['ep']):
                rr=np.flatnonzero(meta['ep']==ep);rr=rr[np.argsort(meta['step'][rr])]
                last_full=0
                for i in rr:
                    if meta['step'][i]-last_full==2 and accept[i]:selected[i]=True
                    else:last_full=int(meta['step'][i])
            # Fractions include non-step-zero actual look rows; step0 has been
            # omitted throughout and will slightly dilute deployment exposure.
            delta=half['motion_mse']-full['motion_mse']
            loss=mean(np.where(selected,delta,0),meta['ep'])
            aeps=np.unique(meta['ep']); per_ep=np.array([np.mean(np.where(selected,delta,0)[meta['ep']==e]) for e in aeps])
            rng=np.random.default_rng(1203)
            ci=np.quantile(per_ep[rng.integers(len(per_ep),size=(1000,len(per_ep)))].mean(1),[.025,.975]).tolist()
            candidates.append(dict(threshold=threshold,progress_veto=guard,
                gate_pass=mean(accept,meta['ep']),half_share=mean(selected,meta['ep']),
                normalized_mse_delta=loss,mse_delta_ci=ci,relative_mse_change=loss/mean(full['motion_mse'],meta['ep']),
                hybrid_mse_vs_legacy=mean(np.where(selected,half['motion_mse'],full['motion_mse'])-legacy['motion_mse'],meta['ep']),
                full_discrepancy_mse=mean(np.where(selected,half['full_delta_mse'],0),meta['ep']),
                full_grip_flip_rate=mean(np.where(selected,half['grip_flip_full'],0),meta['ep']),
                missed_full_stall_rate=mean((selected&valid&~advance['full']&advance['half']).astype(float),meta['ep']),
                selected_full_delta_rms=float(np.sqrt(mean(half['full_delta_mse'][selected],meta['ep'][selected]))) if selected.any() else None,
                selected_top1_agreement=mean((half['rows'][selected,0]==full['rows'][selected,0]).astype(float),meta['ep'][selected]) if selected.any() else None))
    dest=HERE/outname/f'{cell}_{size}';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'rows.npz',**meta,score=score,grip_equal=grip_equal,progress_ok=progress_ok,
                        full_advances=advance['full'],half_advances=advance['half'],valid_progress=valid)
    out.append(dict(cell=cell,size=size,candidates=candidates))
    print(cell,size,[(a['threshold'],a['progress_veto'],round(a['half_share'],3),round(100*a['relative_mse_change'],2),round(a['missed_full_stall_rate'],4)) for a in candidates if a['threshold'] in (.1,.15)])
dump(HERE/f'{outname}.json',dict(timestamp=now(),cells=out,
     caveat='Exploratory B-only threshold screen; action errors do not certify SR. Causal greedy schedule: only accept when last real full was two slots ago; a rejected half completes to full and refreshes memory. Progress check omits earliest candidate because step0 early retrieval was not screened.'))
dump(HERE/f'{outname}_reads.json',sorted(READS))
