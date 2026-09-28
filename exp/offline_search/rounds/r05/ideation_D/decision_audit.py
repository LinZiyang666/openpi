"""Task/initialization controls, contact action followups, and deployment bytes."""
import json
import numpy as np
from diagnose_logs import HERE, ROOT, STORE
from dynamics_probe import arr

def main():
    output=dict(contact=[],bytes=[],noise_boundaries=[])
    for size in [50,500]:
        arm=f'r4k7_p_l10_{size}_tail1ug'; d=dict(np.load(HERE/f'{arm}.npz'))
        c=np.load(HERE/f'contact_{arm}.npz'); mask=c['mask']
        prm=np.load(HERE/f'dynamics_pi05_l10_{size}.npz'); beta=prm['beta']
        sig=np.std(arr(STORE/'library/pi05_l10/current','action')[:,:5,:7],axis=(0,1))
        contact=[]
        for ep in np.unique(d['ep']):
            ix=np.flatnonzero(d['ep']==ep); alarms=ix[mask[ix]]
            if not len(alarms):continue
            j=int(alarms[0]); fut=ix[(ix>j)&(~d['hit'][ix])]
            pred=np.r_[1.,d['action'][j,:,:6].mean(0)]@beta
            rec=dict(task=int(d['task'][j]),init=int(d['init'][j]),success=bool(d['success'][j]),
                     step=int(d['step'][j]),cached_reopen=bool((d['action'][j,:,6]<0).any()),
                     cached_predicted_z=float(pred[2]),cached_translation_norm=float(np.linalg.norm(pred)))
            if j+1 in ix:
                rec['cached_observed_z']=float(d['state'][j+1,2]-d['state'][j,2])
            if len(fut):
                k=int(fut[0]); rec.update(next_miss_gap=k-j,next_miss_reopen=bool((d['action'][k,:,6]<0).any()),
                    action_difference=float(np.sqrt(np.mean(((d['action'][j]-d['action'][k])/sig)**2))))
            contact.append(rec)
        rep=dict(size=size,odd_even=[],outcomes=[])
        for parity in [0,1]:
            first=np.r_[True,np.diff(d['ep'])!=0]; eps=first&(d['init']%2==parity)
            selected=[r for r in contact if r['init']%2==parity]
            rep['odd_even'].append(dict(init_parity=parity,n=int(eps.sum()),fails=int((eps&~d['success']).sum()),
                     alarms=len(selected),alarm_failures=sum(not r['success'] for r in selected)))
        for succ in [True,False]:
            x=[r for r in contact if r['success']==succ]; y=[r for r in x if r.get('next_miss_gap',999)<=3]
            rep['outcomes'].append(dict(success=succ,n=len(x),cached_reopen=sum(r['cached_reopen'] for r in x),
                    cached_up=sum(r['cached_predicted_z']>.02 for r in x),nextmiss_le3=len(y),
                    cached_observed_up=sum(r.get('cached_observed_z',0)>.02 for r in x),
                    nextmiss_le3_reopen=sum(r['next_miss_reopen'] for r in y),
                    nextmiss_le3_actiondiff_mean=float(np.mean([r['action_difference'] for r in y])) if y else None))
        output['contact'].append(rep)
        (HERE/f'contact_followups_{size}.json').write_text(json.dumps(contact,indent=2))
    for model,letter in [('pi05','p'),('groot','g')]:
      for suite,ss,deploy in [('spatial','sp',431 if model=='pi05' else 429),('l10','l10',1103 if model=='pi05' else 1068)]:
       for size in [50,500]:
        lib=STORE/'library'/f'{model}_{suite}'/('current' if size==50 else ('bpool_cs' if model=='pi05' else 'bpool_all'))
        rows=len(arr(lib,'next'))
        path=ROOT/f'r02_g{size}'/'fits'/f'oscl{size}_{letter}_{ss}_cl2.pkl'
        k7=ROOT/'r04_k7/fits'/f'r4k7_p_{ss}_{size}_tail1ug.pkl'
        # Reuse retained next/state/actions, or append a compact float aperture,
        # 5-bit command mask in uint8 and int32 next row (9 B/entry).
        output['bytes'].append(dict(model=model,suite=suite,size=size,rows=rows,deployed_mb=deploy,
            awm_bytes=path.stat().st_size if path.exists() else None,
            k7_bytes=k7.stat().st_size if model=='pi05' and k7.exists() else None,
            contact_addition_bytes=9*rows+8,live_contact_bytes=16*(4+4)+16,
            compact_contact_mb=(9*rows+8)/1e6,
            miss_noise_live_bytes=10*32*4 if model=='pi05' else 16*32*4))
    for model in ['pi05','groot']:
      for suite in ['spatial','l10']:
        f=np.load(STORE/'floor'/f'{model}_{suite}'/'resample.npz')
        a=f['a_fresh'][:,:,0,:6]; b=f['a_recorded'][:,0,:6]; sigma=f['sigma'][:6]
        output['noise_boundaries'].append(dict(model=model,suite=suite,
              same_observation_first_action_continuous_rms_mean=float(np.sqrt(np.mean(((a-b[:,None])/sigma)**2,2)).mean())))
    (HERE/'decision_audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
