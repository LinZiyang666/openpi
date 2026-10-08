"""Best single raw-distance threshold across B calibration behavior paths."""
from common import *
fence()
import numpy as np
from replay_study import trace_pack, take, DOSES

def value(p,model,t):
    cv,cm=COSTS[model]
    calls=p['alive']&(p['guard']|(p['distance']>t))
    return float((cv*p['alive'].sum()+cm*calls.sum())/p['N'].sum())

def main():
    bycell={}
    for mp in sorted((HERE/'data').glob('*.json')):
        m=read(mp)
        if m['cell'] not in CELLS: continue
        p=trace_pack(m)
        if p is not None: bycell.setdefault(m['cell'],[]).append((m,p))
    fits=[];checks=[]
    for cell,entries in bycell.items():
        model=cell.split('_')[0]
        cals=[take(p,p['cal']) for _,p in entries]
        for target in DOSES:
            left=0.;right=max(float(p['distance'][p['alive']].max()) for p in cals)+1
            candidates=[]
            for _ in range(45):
                t=(left+right)/2
                vals=[value(p,model,t) for p in cals]
                candidates.append((max(abs(x-target) for x in vals),t,vals))
                if (min(vals)+max(vals))/2>target: left=t
                else: right=t
            err,t,vals=min(candidates)
            fits.append(dict(cell=cell,target=target,threshold=t,calibration_irs=vals,max_calibration_error=err))
            for m,p in entries:
                v=take(p,~p['cal']);ir=value(v,model,t)
                checks.append(dict(cell=cell,target=target,source=m['root']+':'+m['method'],ir=ir,error=ir-target))
    dump(HERE/'static_b_fits.json',fits);csvout(HERE/'static_b_replay.csv',checks)
    summary=dict(checks=len(checks),passes=sum(abs(r['error'])<=.02 for r in checks),
                 mae=float(np.mean([abs(r['error']) for r in checks])),max_error=max(abs(r['error']) for r in checks))
    dump(HERE/'static_b_summary.json',summary);provenance('static');print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
