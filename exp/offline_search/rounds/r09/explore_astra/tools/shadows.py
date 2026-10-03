"""Same-observation candidate benchmark and calibration against completed experiments.

Action errors are NORMALIZED MODEL action coordinates: motion dims 0..5 only;
gripper sign mismatch is separate. Shadows are teacher draws, never value labels.
All candidate metrics are episode-balanced, preventing long failures dominating.
"""
import json
import numpy as np
from scipy.stats import spearmanr
from .common import HERE, DERIVED, STORE, load_compact, dump, paired_bootstrap


def episode_mean(value, task, init, mask=None):
    if mask is None:
        mask = np.ones(len(task), bool)
    totals = np.zeros((10,30))
    counts = np.zeros((10,30))
    valid = mask & np.isfinite(value)
    np.add.at(totals, (task[valid],init[valid]), value[valid])
    np.add.at(counts, (task[valid],init[valid]), 1)
    return np.divide(totals,counts,out=np.full_like(totals,np.nan),where=counts>0)


def errors(action, teacher, steps=10):
    a,b=action[:,:steps,:7],teacher[:,:steps,:7]
    motion=((a[:,:,:6]-b[:,:,:6])**2).mean((1,2))
    grip=((a[:,:,6]>=0)!=(b[:,:,6]>=0)).mean(1)
    invalid=~np.isfinite(a).all((1,2))|~np.isfinite(b).all((1,2))
    motion[invalid]=np.nan
    grip[invalid]=np.nan
    return motion,grip


def normalize(w, fallback):
    mass=w.sum(1,keepdims=True)
    return np.divide(w,mass,out=fallback.copy(),where=mass>1e-12)


def candidates(chunks, weights, success, episode):
    """All choices use only cached neighbors; never the policy shadow target."""
    w=weights/weights.sum(1,keepdims=True)
    mean=np.einsum('nk,nkha->nha',w,chunks)
    yield 'mean',mean
    yield 'top1',chunks[:,0]
    for k in [4,8]:
        wk=w.copy();wk[:,k:]=0
        yield f'top{k}',np.einsum('nk,nkha->nha',normalize(wk,w),chunks)
    ws=normalize(w*success,w)
    yield 'successful_only',np.einsum('nk,nkha->nha',ws,chunks)
    # Downweight multiple near-identical rows from a single demonstration.
    same=episode[:,:,None]==episode[:,None,:]
    epmass=np.einsum('nij,nj->ni',same,w)
    we=normalize(w/np.maximum(epmass,1e-9),w)
    yield 'episode_balanced',np.einsum('nk,nkha->nha',we,chunks)
    d=((chunks[:,:,:,:6]-mean[:,None,:,:6])**2).mean((2,3))
    medoid=chunks[np.arange(len(chunks)),d.argmin(1)]
    yield 'motion_medoid',medoid
    for power in [.5,2.]:
        wp=normalize(w**power,w)
        yield f'weight_power_{power:g}',np.einsum('nk,nkha->nha',wp,chunks)
    capped=mean.copy()
    expected=np.einsum('nk,nkh->nh',w,np.linalg.norm(chunks[:,:,:,:6],axis=-1))
    factor=np.minimum(1.15,expected/np.maximum(np.linalg.norm(mean[:,:,:6],axis=-1),1e-9))
    capped[:,:,:6]*=factor[:,:,None]
    yield 'norm_cap_1.15',capped
    voted=mean.copy()
    voted[:,:,6]=np.where(np.einsum('nk,nkh->nh',w,np.where(chunks[:,:,:,6]>=0,1.,-1.))>=0,1.,-1.)
    yield 'gripper_vote',voted


def benchmark(meta):
    z=load_compact(meta['arm'])
    mask=z['vision'] & np.isfinite(z['shadow_look_weights']).all(1)
    ix=np.flatnonzero(mask)
    task,init=z['task'][ix],z['init'][ix]
    rows=z['shadow_look_rows'][ix]
    weights=z['shadow_look_weights'][ix]
    name='current' if meta['library_size']==50 else ('bpool_cs' if meta['model']=='pi05' else 'bpool_all')
    lib=STORE/'library'/f"{meta['model']}_{meta['suite']}"/name
    actions=np.load(lib/'action.npy',mmap_mode='r')
    success=np.load(lib/'success.npy',mmap_mode='r')[rows]
    episode=np.load(lib/'episode.npy',mmap_mode='r')[rows]
    chunks=actions[rows,:10,:7]
    teacher=z['policy_shadow_chunk'][ix]
    base_m,base_g=errors(z['cache_chunk'][ix],teacher)
    baseline=episode_mean(base_m,task,init)
    outputs={}
    generated=list(candidates(chunks,weights,success,episode))
    if meta['model']=='pi05':
        generated += [(cam,z[f'camera_shadow_{cam}_cache_chunk'][ix]) for cam in ['wrist','third']]
    for method,pred in generated:
        m,g=errors(pred,teacher)
        em,eg=episode_mean(m,task,init),episode_mean(g,task,init)
        outputs[method]=dict(motion_mse=float(em.mean()),gripper_disagreement=float(eg.mean()),
                             relative_motion=float(em.mean()/baseline.mean()-1),
                             paired_motion_change=paired_bootstrap(em-baseline,draws=3000),
                             task_motion_mse=em.mean(1).tolist())
    mean=generated[0][1]
    parity=float(np.max(np.abs(mean-z['cache_chunk'][ix])))
    # A few machine-precision differences are expected in contraction order.
    out=dict(arm=meta['arm'],n=len(ix),baseline_motion_mse=float(baseline.mean()),
             baseline_gripper_disagreement=float(episode_mean(base_g,task,init).mean()),
             mean_reconstruction_max_abs=parity,candidates=outputs)
    print(meta['arm'],json.dumps({k:round(v['relative_motion'],4) for k,v in outputs.items()}),flush=True)
    return out


def calibration(metas):
    out=[]
    for meta in metas:
        z=load_compact(meta['arm'])
        if 'policy_shadow_chunk' not in z:
            continue
        m,g=errors(z['served_chunk'],z['policy_shadow_chunk'],5)
        em=episode_mean(m,z['task'],z['init'])
        eg=episode_mean(g,z['task'],z['init'])
        first=episode_mean(m,z['task'],z['init'],z['seq']==0)
        rec={k:meta[k] for k in ['arm','model','suite','library_size','variant','sr','ir']}
        rec.update(motion_mse=float(em.mean()),gripper_disagreement=float(eg.mean()),
                   initial_motion_mse=float(first.mean()),task_motion_mse=em.mean(1).tolist(),
                   task_sr=z['success'].mean(1).tolist())
        draws=z.get('policy_draws_chunks')
        if draws is not None:
            sampled=np.isfinite(draws).all((1,2,3))
            # SAME sampled observations: compare live/shadow and two independent draws.
            live_noise=errors(z['served_chunk'][sampled],z['policy_shadow_chunk'][sampled],5)[0]
            draw_noise=errors(draws[sampled,0],draws[sampled,1],5)[0]
            rec.update(noise_sample_count=int(sampled.sum()),same_sample_served_mse=float(live_noise.mean()),
                       same_sample_draw_mse=float(draw_noise.mean()))
        out.append(rec)
    contrasts=[]
    for r in out:
        if r['variant'] in ('A','P10'):
            continue
        a=next((a for a in out if a['variant']=='A' and all(a[k]==r[k] for k in ['model','suite','library_size'])),None)
        if a:
            contrasts.append(dict(arm=r['arm'],variant=r['variant'],delta_sr=r['sr']-a['sr'],
                                   delta_motion=r['motion_mse']-a['motion_mse'],
                                   delta_initial=r['initial_motion_mse']-a['initial_motion_mse']))
    corr={}
    for subset in ['all','cache_only']:
        cc=[c for c in contrasts if subset=='all' or c['variant'] in ['SF1','FL','SW','W10','W5','A5','SHIFT']]
        rho,p=spearmanr([c['delta_motion'] for c in cc],[c['delta_sr'] for c in cc])
        corr[subset]=dict(n=len(cc),spearman=float(rho),p_descriptive=float(p),
                         sign_correct=sum(c['delta_motion']*c['delta_sr']<0 for c in cc),
                         warning='Correlated contrasts; p is descriptive, not an independent-arm test')
    return dict(arms=out,contrasts=contrasts,calibration=corr)


def main():
    metas=sorted([json.loads(p.read_text()) for p in (DERIVED/'compact').glob('*.json')],key=lambda m:m['arm'])
    dump(HERE/'results/synthesis.json',[benchmark(m) for m in metas if m['variant']=='A'])
    dump(HERE/'results/shadow_calibration.json',calibration(metas))


if __name__=='__main__':
    main()
