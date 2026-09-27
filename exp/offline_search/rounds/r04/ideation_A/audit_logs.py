"""Additional command-phase anatomy and top-10 kernel episode diversity; no external writes."""
import concurrent.futures,sys,pathlib,json
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[5]))
from exp.offline_search.rounds.r04.ideation_A.measure_logs import load
from exp.offline_search.rounds.r04.ideation_A.measure_blind import *

def job(js):
    key,scale=js;L=library(key,scale);m,s=key.split('_');arm=f'oscl{scale}_{"p" if m=="pi05" else "g"}_{"sp" if s=="spatial" else s}_cl2'
    E=load(f'r02_g{scale}',arm);out=[];eff=[]
    for R,success in E:
        rr=np.array([r['topk'] for r in R]);sc=np.array([r['scores'] for r in R]);rel=sc[:,:1]-sc
        kr=5 if scale==50 else 8;w=np.exp(-(rel/np.maximum(rel[:,kr-1:kr],1e-6))**2);w/=w.sum(1)[:,None]
        eps=L.episode[rr]
        for ids,ww in zip(eps,w):
            mass=np.bincount(np.unique(ids,return_inverse=True)[1],weights=ww);eff.append(1/(mass*mass).sum())
        head=np.einsum('nk,nkhd->nhd',w,L.action[rr,:5,:7],optimize=True)
        closed=head[:,:,6]>=0 if m=='pi05' else head[:,:,6]<0
        first=None
        for i in range(len(R)-2):
            if rr[i,0]==rr[i+1,0]==rr[i+2,0]:first=i;break
        if success or first is None:continue
        flat=closed.ravel();trans=np.r_[False,flat[1:]!=flat[:-1]]
        closes=np.flatnonzero(trans&flat);releases=np.flatnonzero(trans&~flat)
        prior_close=bool((closes<5*first).any());prior_release=bool((releases<5*first).any())
        phase='pre_close' if not prior_close else ('post_release' if prior_release else 'after_close_before_release')
        out.append(dict(key=key,scale=scale,arm=arm,task=R[0]['task_id'],spell_start=first,
                        elapsed=first/len(R),command_phase=phase,
                        within2_close=int(np.any(abs(closes/5-first)<=2)),within2_release=int(np.any(abs(releases/5-first)<=2)),
                        first_close=float(closes[0]/5) if len(closes) else -1,
                        first_release=float(releases[0]/5) if len(releases) else -1))
    writecsv(OUT/f'command_phase_{key}_{scale}.csv',out)
    return dict(key=key,scale=scale,kernel_effective_episodes_mean=float(np.mean(eff)),first_spells=len(out),
                **{p:sum(r['command_phase']==p for r in out)/len(out) for p in ['pre_close','after_close_before_release','post_release']},
                within2_close=float(np.mean([r['within2_close'] for r in out])),within2_release=float(np.mean([r['within2_release'] for r in out])))

if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(8) as ex:rows=list(ex.map(job,[(k,s) for k in CELLS for s in [50,500]]))
    (OUT/'command_phase_summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
