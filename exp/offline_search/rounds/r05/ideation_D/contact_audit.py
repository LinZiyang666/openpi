"""Audit command-matched aperture evidence on exact 16-member tail anchors."""
import json
import numpy as np
from diagnose_logs import HERE, STORE, FEATURES
from dynamics_probe import arr, event_stats, pct

def main():
    reports=[]
    for size in [50,500]:
      for suite,ss in [('l10','l10'),('spatial','sp')]:
        path=HERE/f'r4k7_p_{ss}_{size}_tail1ug.npz'
        if not path.exists(): continue
        d=dict(np.load(path)); fb=np.load(HERE/f'feedback_{path.stem}.npz')
        lib=STORE/'library'/f'pi05_{suite}'/('current' if size==50 else 'bpool_cs')
        s=arr(lib,'rs')[:,:8]; a=arr(lib,'action')[:,:5,:7]; nxt=arr(lib,'next')
        prm=np.load(HERE/f'dynamics_pi05_{suite}_{size}.npz')
        width=((s[:,6]-s[:,7])/2-prm['lo'])/(prm['hi']-prm['lo'])
        n=len(d['ep']); pred=np.full(n,np.nan); mass=np.full(n,np.nan); contactmass=np.full(n,np.nan)
        for j in np.flatnonzero(~d['vision']):
            rows=d['rows'][j]; w=d['weights'][j]; valid=(rows>=0)&np.isfinite(w)
            rows=rows[valid]; w=w[valid]; nr=nxt[rows]
            valid=nr>=0
            if not len(rows) or j==0 or d['ep'][j]!=d['ep'][j-1]: continue
            valid &= ((a[rows,:,6]>0)==(d['action'][j-1,:,6]>0)).all(1)
            mass[j]=w[valid].sum()
            if mass[j]<.5: continue
            ww=w[valid]/mass[j]
            pred[j]=ww@width[nr[valid]]
            contactmass[j]=ww@(width[nr[valid]]>.25)
        eligible=(fb['close_streak']>=2)&(fb['width']<.05)
        mask=eligible&(pred>.25)&(contactmass>=.75)
        rep=dict(arm=path.stem,unconditional=event_stats(d,fb['contact_mismatch']),
                 command_matched_mean=event_stats(d,eligible&(pred>.25)),
                 consensus75=event_stats(d,mask),
                 consensus75_first30=event_stats(d,mask&(d['step']<30)),tasks={},sensitivity=[])
        first=np.r_[True,np.diff(d['ep'])!=0]
        epd=[]
        for ep in np.unique(d['ep']):
            ix=np.flatnonzero(d['ep']==ep); idx=ix[mask[ix]]
            base=dict(task=int(d['task'][ix[0]]),init=int(d['init'][ix[0]]),success=bool(d['success'][ix[0]]),n=len(ix),alarm=len(idx)>0)
            if len(idx):
                j=idx[0]; later=ix[(ix>j)&(~d['hit'][ix])]; earlier=ix[(ix<j)&(~d['hit'][ix])]
                base.update(alarm_step=int(d['step'][j]),width=float(fb['width'][j]),expected=float(pred[j]),
                            command_mass=float(mass[j]),contact_mass=float(contactmass[j]),
                            next_miss_gap=int(later[0]-j) if len(later) else None,
                            previous_misses=int(len(earlier)),remaining=len(ix)-int(d['step'][j]),
                            next6_misses=int((~d['hit'][j:min(j+6,ix[-1]+1)]).sum()))
            epd.append(base)
        for task in range(10):
            eps=[e for e in epd if e['task']==task]; alarm=[e for e in eps if e['alarm']]
            rep['tasks'][task]=dict(n=len(eps),fail=sum(not e['success'] for e in eps),alarm=len(alarm),
                alarm_fail=sum(not e['success'] for e in alarm))
        rep['within_task_expected_alarm_fail']=sum(t['alarm']*t['fail']/t['n'] for t in rep['tasks'].values())
        ee=[e for e in epd if e['alarm']]
        for outcome in [True,False]:
            se=[e for e in ee if e['success']==outcome]
            rep['alarm_success' if outcome else 'alarm_failure']=dict(n=len(se),
                step_pcts=pct([e['alarm_step'] for e in se]),
                next_miss_gap_pcts=pct([e['next_miss_gap'] for e in se if e['next_miss_gap'] is not None]),
                no_later_miss=sum(e['next_miss_gap'] is None for e in se),
                previous_misses_pcts=pct([e['previous_misses'] for e in se]),
                next6_misses=sum(e['next6_misses'] for e in se),
                remaining_decisions=sum(e['remaining'] for e in se))
        for floor in [.025,.05,.10]:
          for expectation in [.20,.25,.35]:
            mm=(fb['close_streak']>=2)&(fb['width']<floor)&(pred>expectation)&(contactmass>=.75)
            rep['sensitivity'].append(dict(floor=floor,expectation=expectation,**event_stats(d,mm)))
        reports.append(rep)
        (HERE/f'contact_episodes_{path.stem}.json').write_text(json.dumps(epd,indent=2))
        np.savez_compressed(HERE/f'contact_{path.stem}.npz',mask=mask,pred=pred,mass=mass,contactmass=contactmass)
        print(json.dumps(rep),flush=True)
    (HERE/'contact_audit.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__': main()
