"""Causal schedule replay on immutable offline observation paths; no SR inference.

Recorded inf/cache provenance affects anchor AWM branch, not a forced-MISS schedule here.
All hypothetical decisions after the chosen anchor are HITs; this is an explicit frozen-path diagnostic.
"""
import concurrent.futures,collections,json,pathlib,sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[5]))
from exp.offline_search.rounds.r04.ideation_A.measure_blind import *

def job(js):
    key,scale=js; records=[]
    for arm in ['inf','cache']:
        cell=f'{key}_{arm}';q=store.QueryCell(ROOT,cell);N=q.N
        B=np.load(OUT/f'anchors_{cell}_{scale}.npz');a=B['action'][:,:5];sigma=store.action_sigma(str(ROOT),key)
        baseerr=np.sqrt(np.mean(((a-q.a_inf[:,:5,:7])/sigma)**2,(1,2)))
        basegm=np.mean((a[:,:,6]>=0)!=(q.a_inf[:,:5,6]>=0),1)
        windows={}
        for h in range(1,5):
            w=np.load(OUT/f'windows_{cell}_{scale}_h{h}.npz')
            windows[h]={k:np.asarray(v) for k,v in w.items()}
            index=np.full(N,-1,int);index[w['anchor']]=np.arange(len(w['anchor']));windows[h]['index']=index
        for cap in [1,2,3,4]:
            for triggers in ['budget','phase','state','phase_state','phase_state025','phase_state1']:
                threshold=.25 if triggers.endswith('025') else (1. if triggers.endswith('1') else .5)
                for method in ['phase_abs','phase_delta','kernel_clock','anchor_chunk']:
                    if method=='anchor_chunk' and (cap+1)*5>q.H:continue
                    look=np.zeros(N,bool);err=baseerr.copy();gm=basegm.copy();reasons=collections.Counter()
                    for e in q.episodes:
                        anchor=e['start'];look[anchor]=True;reasons['start']+=1
                        for i in range(anchor+1,e['end']):
                            h=i-anchor;reason=None
                            if h>cap:reason='budget'
                            else:
                                w=windows[h];j=w['index'][anchor];assert j>=0
                                if triggers.startswith('phase'):
                                    if w['grip'][j]:reason='grip_ahead'
                                    elif w['terminal'][j]:reason='near_terminal'
                                if reason is None and 'state' in triggers:
                                    if w['still'][j]:reason='still2'
                                    elif w['dev'][j]>threshold:reason='delta_dev'
                            if reason:
                                look[i]=True;anchor=i;reasons[reason]+=1
                            else:
                                err[i]=w[f'err_{method}'][j];gm[i]=w[f'gm_{method}'][j]
                    v=float(look.mean())
                    records.append(dict(cell=cell,scale=scale,cap=cap,triggers=triggers,method=method,N=N,
                                        vision_share=v,ir=.152*v,err_mean=float(err.mean()),delta_awm=float((err-baseerr).mean()),
                                        blind_err=float(err[~look].mean()),blind_delta=float((err-baseerr)[~look].mean()),grip_mis=float(gm.mean()),
                                        **{f'why_{k}':reasons[k]/N for k in ['start','budget','grip_ahead','near_terminal','still2','delta_dev']}))
        print('scheduled',cell,scale,flush=True)
    writecsv(OUT/f'offline_schedules_{key}_{scale}.csv',records)

if __name__=='__main__':
    with concurrent.futures.ProcessPoolExecutor(8) as ex:list(ex.map(job,[(k,s) for k in CELLS for s in [50,500]]))
