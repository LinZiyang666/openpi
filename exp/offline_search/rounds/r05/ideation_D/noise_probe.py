"""Separate observed policy handoff jumps from fixed-observation sampling variation."""
import json
import numpy as np
from diagnose_logs import HERE, STORE
from dynamics_probe import arr, pct

def main():
    out=dict(floor=[],logs=[])
    for model in ['pi05','groot']:
      for suite in ['spatial','l10']:
        f=np.load(STORE/'floor'/f'{model}_{suite}'/'resample.npz')
        a=f['a_fresh'][:,:,:5,:7]; b=f['a_recorded'][:,:5,:7]; sig=f['sigma']
        err=np.sqrt(np.mean(((a-b[:,None])/sig)**2,(2,3)))
        flip=((a[:,:,:,6]>0)!=(b[:,None,:,6]>0)).any(2)
        ownflip=((a[:,:,:,6]>0).any(1)!=(a[:,:,:,6]>0).all(1)).any(1)
        replay=np.sqrt(np.mean(((f['a_replay_recorded_noise'][:,:5,:7]-b)/sig)**2,(1,2)))
        out['floor'].append(dict(model=model,suite=suite,states=len(a),fresh_comparisons=int(flip.size),
            fresh_error=float(err.mean()),fresh_any_grip_flip=int(flip.sum()),fresh_any_grip_flip_share=float(flip.mean()),
            states_with_fresh_draw_grip_disagreement=int(ownflip.sum()),
            replay_error=float(replay.mean()),replay_any_grip_flip=int((((f['a_replay_recorded_noise'][:,:5,6]>0)!=(b[:,:,6]>0)).any(1)).sum()),
            by_task={str(t):dict(states=int((f['task_id']==t).sum()),flip=int(flip[f['task_id']==t].sum())) for t in range(10)}))
    for path in sorted(HERE.glob('r4k7_*.npz')):
        d=dict(np.load(path)); suite='spatial' if '_sp_' in path.stem else 'l10'
        sig=np.std(arr(STORE/'library'/f'pi05_{suite}'/'current','action')[:,:5,:7],axis=(0,1))
        new=np.r_[True,np.diff(d['ep'])!=0]; same=~new
        pm=np.r_[False,~d['hit'][:-1]]; cm=~d['hit']; run=np.zeros(len(cm),int)
        for i in range(len(cm)):
            if cm[i]: run[i]=1 if new[i] else run[i-1]+1
        pairs=same&pm&cm
        jerk=np.full(len(cm),np.nan); jerk[1:]=np.sqrt(np.mean(((d['action'][1:,0,:6]-d['action'][:-1,4,:6])/sig[:6])**2,1))
        flip=np.r_[False,(d['action'][1:,0,6]>0)!=(d['action'][:-1,4,6]>0)]
        rec=dict(arm=path.stem,decisions=len(cm),misses=int(cm.sum()),consecutive_miss_pairs=int(pairs.sum()),
                 consecutive_miss_episodes=len(np.unique(d['ep'][pairs])),
                 paired_noise_eligible_misses=int((cm&(run%2==0)).sum()),
                 pairs_in_failed_episodes=int((pairs&~d['success']).sum()),
                 episode_max_miss_streak_pcts=pct([run[d['ep']==e].max() for e in np.unique(d['ep'])]),
                 categories={})
        for key,mask in [('MM',pairs),('HM',same&~pm&cm),('MH',same&pm&~cm),('HH',same&~pm&~cm)]:
            rec['categories'][key]=dict(n=int(mask.sum()),mean_continuous_boundary_jump=float(jerk[mask].mean()),
                         jump_pcts=pct(jerk[mask]),grip_flips=int((flip&mask).sum()))
        out['logs'].append(rec)
    (HERE/'noise_probe.json').write_text(json.dumps(out,indent=2))
    for r in out['floor']: print(json.dumps(r))
    for r in out['logs']: print(json.dumps({k:v for k,v in r.items() if k!='categories'}))

if __name__=='__main__': main()
