"""Aggregate executed D2 evidence and explicit forecast sensitivity assumptions."""
from .d2_screen import *

def episodes(A,meta,dec):
    out={}
    for uid in np.unique(A['uid']):
        ix=np.flatnonzero(A['uid']==uid);t=int(A['task'][ix[0]]);init=int(A['init'][ix[0]])
        seq=dec[str(uid)];out[(t,init)]=dict(uid=str(uid),task=t,init=init,N=len(seq),V=len(ix),M=int((~A['hit'][ix]).sum()),success=int(meta['accepted'][str(uid)]['success']))
    return out

def ratio_ci(A,dec,value):
    num=[];den=[];tasks=[]
    for uid in np.unique(A['uid']):
        ix=A['uid']==uid;num.append(float(value[ix].sum()));den.append(len(dec[str(uid)]));tasks.append(int(A['task'][ix][0]))
    num=np.array(num);den=np.array(den);tasks=np.array(tasks);rng=np.random.default_rng(121203)
    draws=np.concatenate([rng.choice(np.flatnonzero(tasks==t),(2000,int((tasks==t).sum())),replace=True) for t in range(10)],1)
    s=num[draws].sum(1)/den[draws].sum(1)
    return dict(point=float(num.sum()/den.sum()),ci95=np.quantile(s,[.025,.975]).tolist(),replicates=len(s))

def main():
    out=[];epdata={}
    for p in sorted(D2.glob('r12_devlooks_*/*/screen.json')):
        if p.parent.parent.name not in VALID_ROOTS:continue
        screen=json.loads(p.read_text());root,cell,size=(screen[k] for k in ('root','cell','size'));src=DEST/root/f'{cell}_{size}'
        A=dict(np.load(src/'looks.npz',allow_pickle=False));Q=dict(np.load(p.parent/'candidates.npz',allow_pickle=False))
        meta=json.loads((src/'extract.json').read_text());dec=json.loads((src/'decisions.json').read_text())
        C,lib=load(cell,size);C['sigma']=np.load(HERE/'final_atlas'/f'{cell}_{size}'/'table.npz')['sigma']
        price=.076 if cell.startswith('pi05') else .074;policy=1-2*price
        r=dict(root=root,cell=cell,size=size,N=meta['N'],V=meta['V'],M=meta['M'],episodes=meta['episodes'],
            baseline_sr=meta['successes']/meta['episodes'],baseline_ir=meta['owner_ir'],periodic_setting=meta.get('periodic_setting'),
            allhalf_ceiling=price*(meta['V']-meta['M'])/meta['N'],alternating_ceiling=price*min(meta['V']/2,meta['V']-meta['M'])/meta['N'],methods={})
        epdata[(root,cell,size)]=episodes(A,meta,dec)
        for name in ('frozen_atlas','consensus2_r0.1','consensus2_r0.05','consensus6_r0.15','consensus18_r0.1'):
            s=next(x for x in screen['results'] if x['config']['name']==name);R=dict(np.load(p.parent/f'{name}.npz',allow_pickle=False))
            m=dict(s['all']);m['calibration']=s['calibration'];m['evaluation']=s['evaluation']
            m['saving_ci']=ratio_ci(A,dec,price*(R['half']&A['hit']))
            m['extra_calls_only_penalized_saving']=m['saving_fixed_calls']-policy*m['extra']/meta['N']
            target=A['shadow'];has=np.isfinite(target).all((1,2))
            if has.any():
                err=lambda a:np.mean(((a[:,:,:6]-target[:,:,:6])/C['sigma'][:6])**2,(1,2))
                m['shadow_mse_delta_ci']=bootstrap(A,err(R['corrected'])-err(A['full']),has)
            r['methods'][name]=m
        numeric=p.with_name('numeric_stress.json')
        if numeric.exists():r['numeric_stress']=json.loads(numeric.read_text())
        camera=p.with_name('camera_choice.json')
        if camera.exists():r['camera_choice']=json.loads(camera.read_text())
        out.append(r)
    paired=[]
    for cell in ('pi05_l10','groot_l10'):
        a=epdata[('r12_devlooks_50L',cell,50)];b=epdata[('r12_devlooks_knob',cell,50)];keys=sorted(a.keys()&b.keys())
        price=.152 if cell.startswith('pi05') else .148
        vals=np.array([[a[k]['success'],b[k]['success'],a[k]['N'],b[k]['N'],
            price*a[k]['V']+(1-price)*a[k]['M'],price*b[k]['V']+(1-price)*b[k]['M']] for k in keys]);tasks=np.array([k[0] for k in keys])
        def stats(v):
            sr0,sr1=v[:,:2].mean(0);ir0=v[:,4].sum()/v[:,2].sum();ir1=v[:,5].sum()/v[:,3].sum()
            return np.array([sr0,sr1,ir0,ir1,100*(sr1-sr0),(100*(sr1-sr0)/(ir1-ir0))])
        point=stats(vals);rng=np.random.default_rng(121204);samples=[]
        for _ in range(2000):
            ix=np.concatenate([rng.choice(np.flatnonzero(tasks==t),int((tasks==t).sum()),replace=True) for t in range(10)])
            samples.append(stats(vals[ix]))
        ci=np.quantile(samples,[.025,.975],axis=0)
        paired.append(dict(cell=cell,paired=len(keys),pairs=[list(k) for k in keys],tasks_covered=int(len(np.unique(tasks))),pairs_by_task={str(t):int((tasks==t).sum()) for t in range(10)},off_sr=float(point[0]),knob_sr=float(point[1]),off_ir=float(point[2]),knob_ir=float(point[3]),
            sr_delta_pp=float(point[4]),sr_delta_ci95_pp=ci[:,4].tolist(),chord_pp_per_ir=float(point[5]),chord_ci95=ci[:,5].tolist(),
            note='Only the actual intersecting B inits, pair-weighted across represented tasks; whole-episode bootstrap within available task groups. This small, incomplete-task overlap cannot identify a reliable ten-task or local look-half buyback slope.'))
    forecasts=[]
    for r in out:
        for name in ('consensus2_r0.1','consensus2_r0.05'):
            m=r['methods'][name];S=m['saving_fixed_calls'];isg=r['cell'].startswith('groot')
            # Subjective scenarios are inputs, never fitted from action error or success labels.
            loss=(.25 if isg else .10) if name.endswith('r0.1') else .02
            sr_range=[-1.,.5] if name.endswith('r0.1') else [-.5,.25]
            forecasts.append(dict(root=r['root'],cell=r['cell'],size=r['size'],method=name,
                raw_ir_saving=S,raw_ir_ci95=m['saving_ci']['ci95'],sr_delta_central_scenario_pp=-loss,
                sr_delta_subjective_range_pp=sr_range,matched_ir_saving_at_10pp_per_ir=S-loss/10,
                matched_ir_saving_at_25pp_per_ir=S-loss/25,break_even_loss_at_25pp_per_ir=25*S,
                lossless_matched_saving=S,observed_half_sr=None,
                note='Post-freeze dev-calibrated structural/threshold choice. Raw IR transports fixed full-controller B states; SR is an explicit subjective scenario, not an action-MSE conversion.'))
    info=dict(timestamp=now(),cells=out,paired_knob=paired,forecasts=forecasts,
        ideal_no_call_ceilings={m:dict(allhalf=c/2,alternating=c/4) for m,c in [('pi05',.076),('groot',.074)]},
        models=dict(pi05=dict(vision=.152,half=.076,completion=.076,policy=.848),groot=dict(vision=.148,half=.074,completion=.074,policy=.852)))
    dump(D2/'evidence.json',info)
    print('PAIRED',paired,flush=True)
    for r in out:
        print(r['root'],r['cell'],r['size'],[(name,m['H'],m['saving_fixed_calls'],m['missed'],m['extra'],m['gripper_disagree_half']) for name,m in r['methods'].items() if name in ('consensus2_r0.1','consensus2_r0.05')],flush=True)

if __name__=='__main__':main()
