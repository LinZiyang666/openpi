"""Measured GR00T batch-shape numerical perturbation sensitivity.

The radius is post-freeze B-dev calibrated on reconstructed 56px images.
This cannot certify exact original-pixel one-camera retrieval or policy parity.
"""
from .d2_screen import *

def run(root,size):
    cell='groot_l10';src=DEST/root/f'{cell}_{size}';dest=D2/root/f'{cell}_{size}'
    A=dict(np.load(src/'looks.npz',allow_pickle=False));Q=dict(np.load(dest/'candidates.npz',allow_pickle=False))
    meta=json.loads((src/'extract.json').read_text());decisions=json.loads((src/'decisions.json').read_text())
    C,lib=load(cell,size);table=dict(np.load(HERE/'final_atlas'/f'{cell}_{size}'/'table.npz',allow_pickle=False));C['sigma']=table['sigma']
    with (DEST/'fits'/f'{cell}_{size}'/'half.pkl').open('rb') as f:fit=pickle.load(f)
    with Path(meta['fit_path']).open('rb') as f:base=pickle.load(f)['method'].inner.base
    wrist=fit_wrist(cell,size);past=lag(C,2)
    delta=np.load(D2/'groot_mismatch'/'deltas.npz')[f'P{size}']
    # Fixed numerical stress rule: eight largest measured projected norms,
    # both signs. No action/guard labels select these directions.
    ix=np.argsort(np.sum(delta**2,1),kind='stable')[-8:];pert=np.r_[delta[ix],-delta[ix]]
    configs_=[dict(name='consensus2_r0.1',witnesses=2,rms=.1,progress=.5,advance=.5),
              dict(name='consensus2_r0.05',witnesses=2,rms=.05,progress=.5,advance=.5)]
    masks=[gate_mask(A,Q,C,lib,cfg) for cfg in configs_];eligible=np.flatnonzero(masks[0]);oks=[x.copy() for x in masks]
    maxrms=np.zeros(len(A['step']));top_changed=np.zeros(len(A['step']),bool);grip_changed=top_changed.copy()
    start=time.monotonic()
    for i in eligible:
        j=Q['reference'][i];h=Q['rows'][i,0]
        for d in pert:
            c=candidate(saved_p0=A['P'][j,:64],current_wrist=A['P'][i,64:]+d,current_rs=A['rs'][i],saved_rs=A['rs'][j],
                step=int(A['step'][i]),task=int(A['task'][i]),C=C,table=table,fit=fit,base=base,wrist=wrist,past=past)
            err=float(np.sqrt(np.mean(((c['corrected'][:,:6]-Q['corrected'][i,:,:6])/C['sigma'][:6])**2)))
            maxrms[i]=max(maxrms[i],err);tc=c['rows'][0]!=h;gc=np.any((c['corrected'][:,6]>=0)!=(Q['corrected'][i,:,6]>=0))
            top_changed[i]|=tc;grip_changed[i]|=gc
            for k,cfg in enumerate(configs_):
                if not oks[k][i]:continue
                wh=c['witness_rows'][:2];hh=c['rows'][0]
                pd=np.abs(C['progress'][hh]-C['progress'][wh])*np.maximum(np.maximum(lib.ep_len[hh],lib.ep_len[wh])-1,1)
                adv=(C['progress'][hh]-C['progress'][A['rows'][j,0]])*max(lib.ep_len[hh]-1,1)
                oks[k][i]&=bool(not tc and not gc and err<=.025 and c['witness_rms'][:2].max()<=cfg['rms']
                    and not c['witness_grip'][:2].any() and pd.max()<=.5 and adv>.5)
    results=[]
    for cfg,mask in zip(configs_,oks):
        qr=dict(Q,support=Q['support'].copy());qr['support'][~mask]=0
        R=replay_gate(A,qr,C,lib,decisions,cfg,meta.get('periodic_setting'))
        results.append(dict(config=cfg,result=short(A,Q,R,C,meta,decisions),gate_original=int(gate_mask(A,Q,C,lib,cfg).sum()),gate_robust=int(mask.sum())))
        np.savez_compressed(dest/f"robust_{cfg['name']}.npz",**{k:v for k,v in R.items() if isinstance(v,np.ndarray)})
    np.savez_compressed(dest/'numeric_stress.npz',maxrms=maxrms,top_changed=top_changed,grip_changed=grip_changed,perturbations=pert,ok_main=oks[0],ok_strict=oks[1])
    info=dict(timestamp=now(),root=root,size=size,cell=cell,seconds=time.monotonic()-start,eligible=len(eligible),
        top_changed=int(top_changed.sum()),grip_changed=int(grip_changed.sum()),action_change=quant(maxrms[eligible]),
        results=results,note=__doc__,calibration='post-freeze B-dev image numerical perturbations; gate stability tolerance .025 chosen for sensitivity, not SR calibration')
    dump(dest/'numeric_stress.json',info);print(root,size,{k:info[k] for k in ('eligible','top_changed','grip_changed','seconds')},results,flush=True)

if __name__=='__main__':
    for root,size in [('r12_devlooks_50L',50),('r12_devlooks_400',400),('r12_devlooks_knob',50)]:
        if (D2/root/f'groot_l10_{size}'/'candidates.npz').exists():run(root,size)
