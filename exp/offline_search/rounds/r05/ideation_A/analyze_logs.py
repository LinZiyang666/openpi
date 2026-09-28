"""R5-A: accepted-attempt horizon and tail diagnostics; no external mutations.
L10 logs hold only first five actions. Never infer their missing control boundaries.
"""
from pathlib import Path
import json, math, csv
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
OUT=Path('exp/offline_search/rounds/r05/ideation_A')
ROOT=Path('/home/weiland/trace_runs/os_closed_loop')
STORE=Path(json.loads((OUT/'inventory.json').read_text())['store'])

def rows(p):
    if not p.exists():return
    for line in p.open():
        try:yield json.loads(line)
        except ValueError:pass

def stats(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()),p50=float(np.median(a)),p90=float(np.quantile(a,.9))) if len(a) else dict(n=0)

def rms(x,axis):return np.sqrt(np.mean(np.square(x),axis=axis))

def longest(mask):
    best=cur=0
    for v in mask:
        cur=cur+1 if v else 0;best=max(best,cur)
    return best

def load(run,name):
    rd=ROOT/run/'runs'/name
    journal={r['task_uid']:r for r in rows(rd/'client/journal.jsonl') if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error')}
    rec=defaultdict(dict);dups=conflicts=0;files=[];fields=Counter()
    for p in sorted(rd.glob('server_*/decisions_*.jsonl')):
        files.append(dict(path=str(p),bytes=p.stat().st_size))
        for d in rows(p):
            if d.get('ev')!='dec' or d.get('uid') not in journal:continue
            j=journal[d['uid']]
            if d.get('attempt') is not None and j.get('attempt') is not None and d['attempt']!=j['attempt']:continue
            uid,step=d['uid'],int(d['step'])
            if step in rec[uid]:
                dups+=1
                if any(rec[uid][step].get(k)!=d.get(k) for k in ('hit','top1','served_head','a_exec')):conflicts+=1
            rec[uid][step]=d;fields.update(d.keys())
    assert conflicts==0,(run,name,conflicts)
    for uid in journal:
        assert uid in rec,(name,uid)
        assert sorted(rec[uid])==list(range(len(rec[uid]))),(name,uid,'gap')
    return journal,{u:[m[k] for k in sorted(m)] for u,m in rec.items()},files,dict(fields),dups

def process(job):
    run,name=job;journal,rec,files,fields,dups=load(run,name)
    meta=next(x for x in json.loads((ROOT/run/'arms.json').read_text()) if x['arm']==name)
    suite=meta.get('suite_short') or ('l10' if meta['suite']=='libero_10' else 'spatial')
    L=meta.get('replan_steps',meta.get('client_overrides',{}).get('replan_steps',5))
    sigma=np.load(STORE/'library'/f'pi05_{suite}'/'current/action.npy',mmap_mode='r')[:,:5,:7].std(axis=(0,1)).astype(float)
    eps=[];source=Counter();reason=Counter();all_jitter=defaultdict(list);trans=Counter();policytail=Counter();lr=Counter()
    N=V=M=0
    for uid,ds in sorted(rec.items()):
        j=journal[uid];task,init=map(int,uid.split(':')[-2:]);n=len(ds)
        hh=np.array([d.get('hit',True) for d in ds],bool);vv=np.array([d.get('vision',True) for d in ds],bool)
        N+=n;V+=int(vv.sum());M+=int((~hh).sum());source.update(d.get('src','cache') for d in ds)
        reason.update(str(d.get('extras',{}).get('os_reason',0)) for d in ds if not d.get('hit',True));lr.update(str(d.get('look_reason')) for d in ds)
        acts=np.asarray([d.get('served_head',d.get('a_exec',np.full((5,7),np.nan))) for d in ds],float)
        assert acts.shape==(n,5,7),(name,acts.shape)
        step_rms=rms(np.diff(acts[:,:,:6],axis=1)/sigma[:6],axis=2)
        head_rms=rms(np.diff(acts[:,:,:6],axis=0)/sigma[:6],axis=(1,2))
        repeat=head_rms<=.1
        grip=acts[:,:,6]>=0
        finite=np.isfinite(acts[:,:,6])
        within=(grip[:,1:]!=grip[:,:-1]) & finite[:,1:] & finite[:,:-1]
        boundary=(grip[1:,0]!=grip[:-1,-1]) & finite[1:,0] & finite[:-1,-1]
        observed_flips=int(within.sum()+(boundary.sum() if L==5 else 0))
        if L==5:
            b=rms((acts[1:,0,:6]-acts[:-1,4,:6])/sigma[:6],axis=1)
            all_jitter['boundary_cont6'].extend(b.tolist())
        all_jitter['within_head_cont6'].extend(step_rms.ravel().tolist())
        for k in range(1,n):
            a,b=ds[k-1],ds[k]
            typ=a.get('src','cache')+'>'+b.get('src','cache');trans[typ]+=1
            if L==5:
                jump=rms((acts[k,0,:6]-acts[k-1,4,:6])/sigma[:6],axis=0)
                all_jitter[typ].append(float(jump))
            if not a.get('hit',True):
                ex=a.get('extras',{});span=ex.get('noprog_span',ex.get('noprog_n',0))
                policytail['post_miss_total']+=1
                policytail['post_miss_next_miss']+=int(not b.get('hit',True))
                policytail['span_positive']+=int(span>0)
                policytail['span_zero_next_miss']+=int(span==0 and not b.get('hit',True))
                policytail['span_zero']+=int(span==0)
        # Fixed-path replacement: retain original branch decisions outside replaced tail slots.
        # Candidate A keeps K10 progress veto. B waives it only for the policy tail.
        projections={}
        for mode in ('k10_span_veto','policy_commit','dual_commit'):
            vnew=vv.copy();mnew=~hh.copy();replaced=[];k=0
            while k<n-1:
                a=ds[k];miss=not a.get('hit',True);isvision=a.get('vision',True)
                span=a.get('extras',{}).get('noprog_span',a.get('extras',{}).get('noprog_n',0))
                eligible=(miss and (mode!='k10_span_veto' or span==0)) or (mode=='dual_commit' and isvision)
                if eligible:
                    vnew[k+1]=False;mnew[k+1]=False;replaced.append(k+1);k+=2
                else:k+=1
            projections[mode]={'v':int(vnew.sum()),'m':int(mnew.sum()),'replaced':len(replaced),'removed_m':int((~hh)[replaced].sum())}
        srstate=np.asarray([d.get('robot_state',[])[:8] for d in ds],float)
        ep=dict(run=run,arm=name,task=task,init=init,success=int(j.get('success',False)),requests=n,nominal_controls=n*L,
                head_repeat_pairs=int(repeat.sum()),longest_head_repeat_requests=longest(repeat)+1,
                within_gripper_flips=int(within.sum()),observed_gripper_flips=observed_flips,
                boundary_gripper_flips=int(boundary.sum()) if L==5 else None,
                within_gripper_opportunities=int((finite[:,1:]&finite[:,:-1]).sum()),
                first_observed_close=next((int(i*L+t) for i in range(n) for t in range(5) if finite[i,t] and grip[i,t]),None),
                any_observed_release=bool(np.any((grip[:,:-1]&~grip[:,1:])&finite[:,:-1]&finite[:,1:]) or (L==5 and np.any(grip[:-1,4]&~grip[1:,0]))),
                projections=projections,vision=int(vv.sum()),misses=int((~hh).sum()))
        if srstate.shape==(n,8):
            motion=np.linalg.norm(np.diff(srstate,axis=0),axis=1)
            ep['motion_mean']=float(motion.mean()) if len(motion) else None
            ep['low_motion_pairs_001']=int((motion<.01).sum())
        eps.append(ep)
    per_task=[]
    for t in range(10):
        e=[e for e in eps if e['task']==t];per_task.append(dict(task=t,n=len(e),success=sum(e0['success'] for e0 in e),sr=np.mean([e0['success'] for e0 in e])))
    projections={}
    for mode in ('k10_span_veto','policy_commit','dual_commit'):
        totals={k:sum(e['projections'][mode][k] for e in eps) for k in ('v','m','replaced','removed_m')}
        totals['ir']=(.152*totals['v']+.848*totals['m'])/N*5/L;projections[mode]=totals
    out=dict(run=run,arm=name,meta=meta,L=L,episodes=len(eps),success=sum(e['success'] for e in eps),sr=np.mean([e['success'] for e in eps]),
             N=N,V=V,M=M,v=V/N,m=M/N,ir_owner_fullmiss=(.152*V+.848*M)/N*5/L,
             sources=dict(source),reasons=dict(reason),look_reasons=dict(lr),transitions=dict(trans),policytail=dict(policytail),
             projections=projections,jitter={k:stats(v) for k,v in all_jitter.items()},per_task=per_task,
             by_outcome={str(s):{'n':sum(e['success']==s for e in eps),**{k:stats([e[k] for e in eps if e['success']==s and e.get(k) is not None]) for k in ('requests','nominal_controls','longest_head_repeat_requests','observed_gripper_flips','first_observed_close','within_gripper_flips')},'never_observed_close':sum(e['success']==s and e['first_observed_close'] is None for e in eps),'no_observed_release':sum(e['success']==s and not e['any_observed_release'] for e in eps)} for s in (0,1)},
             fields=fields,files=files,duplicates=dups,full_policy_chunks_available=False)
    (OUT/f'log_{name}.json').write_text(json.dumps(out,indent=2))
    (OUT/f'episodes_{name}.json').write_text(json.dumps(eps,indent=1))
    print(name,'SR',out['sr'],'NVM',N,V,M,'IR',out['ir_owner_fullmiss'],flush=True)
    return out

def pair(a,b):
    ea=json.loads((OUT/f"episodes_{a['arm']}.json").read_text());eb=json.loads((OUT/f"episodes_{b['arm']}.json").read_text())
    A={(x['task'],x['init']):x for x in ea};B={(x['task'],x['init']):x for x in eb};keys=sorted(A.keys()&B.keys())
    def calc(kk):
        c=Counter((A[k]['success'],B[k]['success']) for k in kk);sf,fs=c[1,0],c[0,1];nn=sf+fs
        p=min(1.,2*sum(math.comb(nn,k) for k in range(min(sf,fs)+1))/2**nn) if nn else 1.
        return dict(n=len(kk),SF=sf,FS=fs,delta=(fs-sf)/len(kk),mcnemar_p=p,
                    controls_on_both_success=stats([B[k]['nominal_controls']-A[k]['nominal_controls'] for k in kk if A[k]['success'] and B[k]['success']]))
    return dict(reference=a['arm'],candidate=b['arm'],**calc(keys),per_task={str(t):calc([k for k in keys if k[0]==t]) for t in range(10)})

if __name__=='__main__':
    inv=json.loads((OUT/'inventory.json').read_text())
    jobs=[(a['run'],a['name']) for a in inv['arms'] if a['episodes']==500 and (a['run']=='r04_k7' or (a['run']=='r04_cost' and '_inf_' in a['name']))]
    jobs += [('r03_mx',x) for x in ('r3mx_p_l10_g','r3mx_p_l10_g500','r3mx_p_sp_g')]
    with ProcessPoolExecutor(max_workers=4) as ex:summary=list(ex.map(process,jobs))
    (OUT/'log_summary.json').write_text(json.dumps(summary,indent=2))
    pairs=[]
    for suite in ('l10','sp'):
        b=next(x for x in summary if x['arm']==f'r4f_p_{suite}_inf_k10_L10')
        for name in (f'r4f_p_{suite}_inf_s1001',f'r4b2_p_{suite}_inf_k2_s1101'):
            pairs.append(pair(next(x for x in summary if x['arm']==name),b))
    (OUT/'horizon_pairs.json').write_text(json.dumps(pairs,indent=2))
    for p in pairs:print(p['reference'],p['candidate'],p['SF'],p['FS'],p['mcnemar_p'])
