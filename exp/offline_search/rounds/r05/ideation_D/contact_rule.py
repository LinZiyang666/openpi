"""Online-legal pure helper for proposal D1; no server or Method modifications.

Call only on an eligible one-decision-old cache anchor. The wrapper owns episode
reset, the once-per-episode budget, vision fallback, and the subsequent MISS flag.
"""
import json
import time
import numpy as np
from diagnose_logs import HERE, STORE
from dynamics_probe import arr

def fit_contact(state, action, next_row):
    aperture=(state[:,6].astype(float)-state[:,7])/2
    lo,hi=np.percentile(aperture,[1,99]).astype(np.float32)
    width=((aperture-lo)/(hi-lo)).astype(np.float32)
    pattern=np.sum((action[:,:5,6]>=0).astype(np.uint8)*(1<<np.arange(5,dtype=np.uint8)),axis=1).astype(np.uint8)
    return dict(width=width,pattern=pattern,next=np.array(next_row,np.int32),lo=lo,hi=hi)

def contact_alarm(table, model, state, last_two_heads, rows, weights):
    if len(last_two_heads)<2:return False
    close_sign=1 if model=='pi05' else -1
    if not np.all(last_two_heads[-2:,:5,6]*close_sign>0):return False
    width=((float(state[6])-float(state[7]))/2-table['lo'])/(table['hi']-table['lo'])
    if width>=.05:return False
    rows=np.asarray(rows); weights=np.asarray(weights)
    if not len(rows) or np.any(rows<0) or not np.isfinite(weights).all():return False
    pattern=int(np.sum((last_two_heads[-1,:5,6]>=0)*(1<<np.arange(5))))
    nr=table['next'][rows]; valid=(nr>=0)&(table['pattern'][rows]==pattern)
    mass=weights[valid].sum()
    if mass<.5:return False
    ww=weights[valid]/mass; expected=table['width'][nr[valid]]
    return bool(ww@expected>.25 and ww@(expected>.25)>=.75)

def main():
    result=[]
    for size in [50,500]:
        lib=STORE/'library/pi05_l10'/('current' if size==50 else 'bpool_cs')
        table=fit_contact(arr(lib,'rs'),arr(lib,'action'),arr(lib,'next'))
        arm=f'r4k7_p_l10_{size}_tail1ug';d=dict(np.load(HERE/f'{arm}.npz'))
        expected=np.load(HERE/f'contact_{arm}.npz')['mask']
        got=np.zeros(len(d['ep']),bool); timings=[]
        for j in np.flatnonzero(~d['vision']):
            if d['step'][j]<2:continue
            start=time.perf_counter_ns()
            got[j]=contact_alarm(table,'pi05',d['state'][j],d['action'][j-2:j],d['rows'][j],d['weights'][j])
            timings.append((time.perf_counter_ns()-start)/1e3)
        assert np.array_equal(got,expected),np.flatnonzero(got!=expected)
        result.append(dict(size=size,n_timed=len(timings),alarm_decisions=int(got.sum()),
            disagreement=int((got!=expected).sum()),median_us=float(np.median(timings)),
            p95_us=float(np.quantile(timings,.95)),table_bytes=int(table['width'].nbytes+table['pattern'].nbytes+table['next'].nbytes+8)))
    (HERE/'contact_rule_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
