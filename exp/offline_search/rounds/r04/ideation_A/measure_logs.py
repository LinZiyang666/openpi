"""Read-only accepted-attempt log reconstruction and frozen-path skip-vision schedules.
Top ten of sixteen members are all the closed-loop logs retain; weights are approximate.
No robot states/images were logged in these runs: proprioception firing is NOT inferred from picks.
"""
import concurrent.futures, json, pathlib, sys, collections
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[5]))
from exp.offline_search.rounds.r04.ideation_A.measure_blind import OUT,ROOT,RUNS,library,topology,writecsv,CELLS

def load(run,arm):
    base=RUNS/run/'runs'/arm; accepted={}
    for line in (base/'client/journal.jsonl').open():
        r=json.loads(line)
        if r.get('accepted') and r.get('status') in ['done','failed'] and not r.get('error'):
            accepted[r['task_uid']]=r
    rec={}
    for p in sorted(base.glob('server_*/decisions_*.jsonl')):
        for line in p.open():
            if '"ev": "dec"' not in line:continue
            r=json.loads(line);uid=r['uid']
            if uid in accepted and r.get('attempt')==accepted[uid].get('attempt'):
                rec[(uid,r['step'])]=r
    eps=collections.defaultdict(list)
    for (uid,step),r in sorted(rec.items()):eps[uid].append(r)
    assert len(eps)==500,(run,arm,len(eps))
    return [(rows,bool(accepted[uid]['success'])) for uid,rows in eps.items()]

def runjob(job):
    key,scale,run,arm,kref=job;L=library(key,scale);nxt,prv,adv,event,near,ss,m10,m50=topology(L)
    eps=load(run,arm); summary=[];timing=[];sched=[];dec=[];fail=[]
    for records,success in eps:
        N=len(records);assert [r['step'] for r in records]==list(range(N)),arm
        rr=np.array([r['topk'] for r in records],int);scores=np.array([r['scores'] for r in records]);top=rr[:,0]
        rel=scores[:,:1]-scores
        # G3 wrappers log log-kernel weights; bare AWM logs negative distances.
        ww=np.exp(-rel) if records[0]['method'].startswith('MXJ') else np.exp(-(rel/np.maximum(rel[:,kref-1:kref],1e-6))**2)
        ww/=ww.sum(1)[:,None]
        hit=np.array([r.get('hit',True) for r in records]);steps=np.arange(N)
        prog=np.sum(ww*L.progress[rr],1);nearflag=(ww*near[rr]).sum(1)>=.2
        trans=(ww*(event[rr]|event[nxt[rr]])).sum(1)>=.2
        term=(ww*(L.step[rr]>=L.ep_len[rr]-2)).sum(1)>=.2
        vote=(ww*np.where(L.action[rr,0,6]>=0,1.,-1.)).sum(1)
        # AWM still is two-camera centred cosine, NOT the no-motion proprioceptive trigger.
        visual=np.array([r.get('extras',{}).get('still',np.nan) for r in records])>1.98
        flags={'grip_ahead':trans,'near_terminal':term,'union_phase':trans|term,'visual_static_proxy':visual}
        repeat=np.r_[False,top[1:]==top[:-1]]
        first=None
        for s in range(N-2):
            if top[s]==top[s+1]==top[s+2] and hit[s:s+3].all():first=s;break
        for i,r in enumerate(records):
            dec.append(dict(success=success,phase=min(2,int(3*i/N)),libphase=min(2,int(3*prog[i])),
                            near=bool(nearflag[i]),grip=bool(trans[i]),terminal=bool(term[i]),visual=bool(visual[i]),
                            hit=bool(hit[i]),repeat=bool(repeat[i]),
                            sameep=bool(i>0 and L.episode[top[i]]==L.episode[top[i-1]]),
                            successor=bool(i>0 and L.next[top[i-1]]==top[i])))
        if not success and first is not None:
            s=first;kind='T' if L.next[top[s]]<0 else ('G' if abs(vote[s])<.5 else 'other')
            fail.append(dict(start=s,frac=s/N,prog=float(prog[s]),near=bool(nearflag[s]),kind=kind,
                             task=int(records[0]['task_id']),N=N))
            for name,flag in flags.items():
                prev=np.flatnonzero(flag[:s+1]);future=np.flatnonzero(flag[s:])
                timing.append(dict(arm=arm,key=key,scale=scale,kind=kind,trigger=name,start=s,frac=s/N,lib_progress=prog[s],
                                  at_start=int(flag[s]),within_prior2=int(flag[max(0,s-2):s+1].any()),
                                  within_prior4=int(flag[max(0,s-4):s+1].any()),ever_before=int(len(prev)>0),
                                  nearest_prior_lead=int(s-prev[-1]) if len(prev) else -1,
                                  first_after_delay=int(future[0]) if len(future) else -1))
        # Frozen observation/action/MISS paths. These are IR estimates, not closed-loop SR forecasts.
        for cap in [1,2,3,4]:
            for trigger_set in ['budget','phase','phase_vote']:
                looks=np.zeros(N,bool);predphase=np.zeros(N,bool);blind_age=0;anchor=0;why=collections.Counter()
                for i in range(N):
                    reason=None
                    forecast_rows=adv[min(i-anchor,8),rr[anchor]];forecast_w=ww[anchor]
                    predphase[i]=((forecast_w*(event[forecast_rows]|event[nxt[forecast_rows]])).sum()>=.2 or
                                  (forecast_w*(L.step[forecast_rows]>=L.ep_len[forecast_rows]-2)).sum()>=.2)
                    if i==0:reason='start'
                    elif not hit[i]:reason='recorded_miss'
                    elif not hit[i-1]:reason='after_miss'
                    elif blind_age>=cap:reason='budget'
                    else:
                        h=i-anchor;rows=adv[h,rr[anchor]];w=ww[anchor]
                        if trigger_set!='budget':
                            if (w*(event[rows]|event[nxt[rows]])).sum()>=.2:reason='grip_ahead'
                            elif (w*(L.step[rows]>=L.ep_len[rows]-2)).sum()>=.2:reason='near_terminal'
                            elif trigger_set=='phase_vote' and abs((w*np.where(L.action[rows,0,6]>=0,1.,-1.)).sum())<.8:reason='split_vote'
                    if reason:
                        looks[i]=True;anchor=i;blind_age=0;why[reason]+=1
                    else:blind_age+=1
                sched.append(dict(arm=arm,key=key,scale=scale,cap=cap,triggers=trigger_set,N=N,success=success,
                                  nlook=int(looks.sum()),nmiss=int((~hit).sum()),nblind=int((~looks).sum()),
                                  first_failed_spell=first if not success and first is not None else -1,
                                  predicted_phase_at_spell=int(predphase[first]) if not success and first is not None else -1,
                                  predicted_phase_prior2=int(predphase[max(0,first-2):first+1].any()) if not success and first is not None else -1,
                                  predicted_phase_prior4=int(predphase[max(0,first-4):first+1].any()) if not success and first is not None else -1,
                                  **{f'why_{s}':why[s] for s in ['start','recorded_miss','after_miss','budget','grip_ahead','near_terminal','split_vote']}))
    row=dict(arm=arm,key=key,scale=scale,episodes=len(eps),sr=float(np.mean([s for r,s in eps])),N=len(dec),
             failures=sum(not s for r,s in eps),failed_with_spell=len(fail),
             sameep=float(np.mean([d['sameep'] for d in dec])),successor=float(np.mean([d['successor'] for d in dec])),
             repeat=float(np.mean([d['repeat'] for d in dec])),motion10=m10,motion50=m50,
             first_spell_mean=float(np.mean([f['start'] for f in fail])) if fail else None,
             first_spell_elapsed_mean=float(np.mean([f['frac'] for f in fail])) if fail else None,
             first_spell_library_progress_mean=float(np.mean([f['prog'] for f in fail])) if fail else None,
             first_spell_near_transition=float(np.mean([f['near'] for f in fail])) if fail else None,
             first_spell_classes=dict(collections.Counter(f['kind'] for f in fail)))
    for label,sel in [('all',dec),('success',[d for d in dec if d['success']]),('failure',[d for d in dec if not d['success']]),
                      *[(p,[d for d in dec if d['phase']==i]) for i,p in enumerate(['early','mid','late'])]]:
        for f in ['grip','terminal','visual','near']:
            row[f'{label}_{f}']=float(np.mean([d[f] for d in sel])) if sel else None
    (OUT/f'log_summary_{arm}.json').write_text(json.dumps(row,indent=2));writecsv(OUT/f'log_timing_{arm}.csv',timing)
    writecsv(OUT/f'log_schedules_{arm}.csv',sched)
    print(arm,row['sr'],row['N'],flush=True)
    return row

if __name__=='__main__':
    jobs=[]
    for key in CELLS:
        m,s=key.split('_');p='p' if m=='pi05' else 'g';s='sp' if s=='spatial' else s
        for scale in [50,500]:jobs.append((key,scale,f'r02_g{scale}',f'oscl{scale}_{p}_{s}_cl2',5 if scale==50 else 8))
    for key,scale,arm in [('pi05_spatial',50,'r3mx_p_sp_g'),('pi05_l10',50,'r3mx_p_l10_g'),
                          ('pi05_l10',500,'r3mx_p_l10_g500'),('pi05_l10',50,'r3mx_p_l10_perk5'),
                          ('pi05_spatial',50,'r3mx_p_sp_awm_h70'),('pi05_spatial',500,'r3mx_p_sp_awm500_h70'),
                          ('pi05_l10',50,'r3mx_p_l10_perk3'),('pi05_l10',500,'r3mx_p_l10_awm500_h70')]:
        # MixedJudge log scores are already transformed; kref is unused for those.
        jobs.append((key,scale,'r03_mx',arm,5 if scale==50 else 8))
    with concurrent.futures.ProcessPoolExecutor(8) as ex:
        rows=list(ex.map(runjob,jobs))
    (OUT/'log_summaries.json').write_text(json.dumps(rows,indent=2))
