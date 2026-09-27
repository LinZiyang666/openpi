"""Accepted-attempt, deduplicated R2 AWM and R3 mixed-log diagnostics (no simulator)."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from exp.offline_search.harness.store import LibraryView
from exp.offline_search.rounds.r04.ideation_C.library_diagnostic import ROOT,OUT,phase_labels
RUN=Path('/home/weiland/trace_runs/os_closed_loop')

def journal(run,arm):
    ans={}
    for line in (RUN/run/'runs'/arm/'client/journal.jsonl').open():
        r=json.loads(line)
        if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error'):
            ans[r['task_uid']]=r
    return ans

def records(run,arm):
    j=journal(run,arm); rec={}
    for f in sorted((RUN/run/'runs'/arm).glob('server_*/decisions_*.jsonl')):
        for line in f.open():
            if '"ev": "dec"' not in line: continue
            d=json.loads(line); u=d.get('uid')
            if u not in j or d.get('attempt')!=j[u].get('attempt'): continue
            rec[u,int(d['step'])]=d
    return j,rec

def work(job):
    run,arm,m,s,n=job
    lib=LibraryView(ROOT,f'{m}_{s}','current' if n==50 else ('bpool_cs' if m=='pi05' else 'bpool_all'))
    phase=phase_labels(lib); j,rec=records(run,arm)
    sig=np.std(LibraryView(ROOT,f'{m}_{s}').action[:,:5,:7],axis=(0,1))
    hd=(lib.action[:,:5,:7]/sig).reshape(lib.L,35)
    nx=np.asarray(lib.next); ni=np.where(nx>=0,nx,np.arange(lib.L))
    mixed=run=='r03_mx'; kr=5 if n==50 else 8
    ep_out=[]; episodes={}; scalar=[]
    for (u,step),d in sorted(rec.items()): episodes.setdefault(u,[]).append(d)
    for u,rr in episodes.items():
        success=bool(j[u]['success']); tt=[]
        previous=None; state=0
        for d in rr:
            step=int(d['step']); kk=np.asarray(d['topk'],int); sc=np.asarray(d['scores'])
            if mixed: w=np.exp(sc-sc[0])
            else: w=np.exp(-((sc-sc[0])/max(sc[0]-sc[kr-1],1e-6))**2)
            w/=w.sum()
            ph=phase[kk]; pw=np.bincount(ph,weights=w)
            hh=hd[kk]; h=w@hh
            fn=w@hd[ni[kk]]
            curdis=np.sqrt(np.sum(w*np.mean((hh-h)**2,axis=1)))
            futdis=np.sqrt(np.sum(w*np.mean((hd[ni[kk]]-fn)**2,axis=1)))
            # Only top-10 is logged; quantities are truncated-kernel diagnostics, not exact served heads.
            e=d.get('extras',{})
            hit=bool(d.get('hit',True))
            pos='step0' if step==0 else ('after_miss' if not previous.get('hit',True) else ('second_after_miss' if state==1 else 'other'))
            phasejump=np.nan; back=np.nan; switch=np.nan
            if previous:
                a=int(previous['top1']); b=int(d['top1'])
                phasejump=float(phase[a]!=phase[b]); back=float(lib.progress[b]-lib.progress[a]<-.1)
                switch=float(lib.episode[a]!=lib.episode[b])
            metric=dict(uid=u,task=int(d['task_id']),init=int(d['init']),step=step,success=int(success),
                        hit=int(hit),pos=pos,phase_impurity=float(1-(pw*pw).sum()),
                        phase_majority_mass=float(pw.max()),current_disp=float(curdis),future_disp=float(futdis),
                        phasejump=phasejump,back=back,ep_switch=switch,
                        noprog=float(e.get('noprog_n',np.nan)),reason=int(e.get('os_reason',0)),
                        conf=float(d['conf']),top1=int(d['top1']))
            scalar.append(metric); tt.append(metric)
            state=1 if previous and not previous.get('hit',True) else 0
            previous=d
        def avg(key,prefix=None):
            vals=[r[key] for r in tt if prefix is None or r['step']<prefix]
            return float(np.mean(vals)) if vals else None
        ep_out.append(dict(uid=u,task=tt[0]['task'],init=tt[0]['init'],success=int(success),decisions=len(tt),
            phase_impurity=avg('phase_impurity'),phase_impurity_first10=avg('phase_impurity',10),
            current_disp=avg('current_disp'),future_disp=avg('future_disp')))
    a={k:np.asarray([r[k] for r in scalar]) for k in scalar[0] if k not in ('uid',)}
    stats=[]
    for outcome in [0,1]:
        for window in ['all','first10','first20']:
            ok=a['success']==outcome
            if window!='all': ok &= a['step']<int(window[5:])
            if not ok.sum(): continue
            stat=dict(outcome=outcome,window=window,decisions=int(ok.sum()))
            for k in ['phase_impurity','phase_majority_mass','current_disp','future_disp','phasejump','back','ep_switch']:
                stat[k]=float(np.nanmean(a[k][ok]))
            stat['phase_ambiguous_share']=float((a['phase_majority_mass'][ok]<.8).mean())
            stats.append(stat)
    positions=[]
    if mixed:
        for pos in ['after_miss','second_after_miss','other']:
            for outcome in [0,1]:
                ok=(a['pos']==pos)&(a['success']==outcome)
                if not ok.sum(): continue
                positions.append(dict(pos=pos,outcome=outcome,n=int(ok.sum()),hit=float(a['hit'][ok].mean()),
                    phasejump=float(np.nanmean(a['phasejump'][ok])),back=float(np.nanmean(a['back'][ok])),
                    ep_switch=float(np.nanmean(a['ep_switch'][ok])),phase_impurity=float(a['phase_impurity'][ok].mean())))
    result=dict(run=run,arm=arm,model=m,suite=s,scale=n,episodes=len(j),
        sr=float(np.mean([x['success'] for x in ep_out])),n_dec=len(scalar),
        ir_pi05=float(.152+.848*(1-a['hit'].mean())) if m=='pi05' else None,
        stats=stats,positions=positions,per_episode=ep_out)
    (OUT/f'logs_{arm}.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/f'logs_{arm}.npz',**a)
    return {k:v for k,v in result.items() if k not in ('stats','positions','per_episode')}

if __name__=='__main__':
    jobs=[]
    for n in [50,500]:
        for m,l in [('pi05','p'),('groot','g')]:
            for s,short in [('spatial','sp'),('l10','l10')]:
                jobs.append((f'r02_g{n}',f'oscl{n}_{l}_{short}_cl2',m,s,n))
    for suffix,n in [('g',50),('g500',500),('awm_h70',50),('awm500_h70',500),('perk5',50)]:
        jobs.append(('r03_mx',f'r3mx_p_l10_{suffix}','pi05','l10',n))
    with ProcessPoolExecutor(max_workers=8) as pool:
        for result in pool.map(work,jobs): print(json.dumps(result),flush=True)
