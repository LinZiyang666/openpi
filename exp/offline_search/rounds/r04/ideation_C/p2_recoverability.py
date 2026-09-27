"""Observational recovery, accepted attempts, episode-balanced landmarks; no causal rescue claim."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from exp.offline_search.harness.store import LibraryView
from exp.offline_search.rounds.r04.ideation_C.log_diagnostic import records,journal,ROOT,OUT

def group(rows,field):
 out={}
 for r in rows:out.setdefault(str(r[field]),[]).append(r)
 ans={}
 for key,rr in out.items():
  b=[r for r in rr if not r['base_success']]
  ans[key]={'n':len(rr),'success':sum(r['success'] for r in rr),'base_failed':len(b),'paired_rescued':sum(r['success'] for r in b),
    'return6':sum(r.get('return6',0) for r in rr),'misses_remaining_mean':float(np.mean([r['remaining_miss'] for r in rr]))}
 return ans

def work(job):
 arm,suite,scale=job
 j,rec=records('r03_mx',arm);bj=journal(f'r02_g{scale}',f'oscl{scale}_p_{"sp" if suite=="spatial" else "l10"}_cl2')
 base={tuple(map(int,u.split(':')[-2:])):bool(r['success']) for u,r in bj.items()}
 lib=LibraryView(ROOT,'pi05_'+suite,'current' if scale==50 else 'bpool_cs')
 sig=np.std(LibraryView(ROOT,'pi05_'+suite).action[:,:5,:7],axis=(0,1))
 episodes={}
 for (u,s),d in sorted(rec.items()):episodes.setdefault(u,[]).append(d)
 first=[];landmarks={k:[] for k in [1,2,3,5,10,20]};events=[];eps=[];cost_rules={str(k):[] for k in [5,10,15,20]}
 for u,rr in episodes.items():
  n=len(rr);ok=bool(j[u]['success']);b=base[(rr[0]['task_id'],rr[0]['init'])]
  hit=np.array([r.get('hit',True) for r in rr]);miss=~hit
  prog=np.array([r['extras'].get('top1_prog',float(lib.progress[r['top1']])) for r in rr])
  flags=np.array([r['extras'].get('os_flags',0) for r in rr],int)
  stall=np.array([r['extras'].get('stuck_n',0)>=2 or r['extras'].get('noprog_n',0)>=2 for r in rr])
  firststall=int(np.flatnonzero(stall)[0]) if stall.any() else n+1000
  # GRIP phase is executed action transitions, descriptive only, no gating proposal.
  phase=0;prevsign=0;run=0;rulehit={};seen_land=set()
  for i,r in enumerate(rr):
   ex=r['extras'];sg=int(ex.get('gexec',0)); phase+=int(prevsign!=0 and sg!=prevsign and sg!=0);prevsign=sg
   if hit[i]:run=0;continue
   run+=1
   if run==1:start=i;startprog=prog[i]
   return6=bool(ok and n-i<=6)
   for z in range(i+1,min(i+7,n-2)):
    if hit[z:z+3].all() and prog[z+2]>prog[i]+.03:return6=True
   since='before' if i<firststall else ('0-2' if i-firststall<=2 else ('3-9' if i-firststall<10 else '10+'))
   f={'uid':u,'task':int(r['task_id']),'init':int(r['init']),'step':i,'success':ok,'base_success':b,
      'reason':r['judge'],'conf':float(r['conf']),'progress':float(prog[i]),'gexec':sg,'phase':min(phase,3),
      'time':'0-19' if i<20 else ('20-39' if i<40 else ('40-59' if i<60 else '60+')),
      'progress_bin':'0-.25' if prog[i]<.25 else ('.25-.5' if prog[i]<.5 else ('.5-.75' if prog[i]<.75 else '.75+')),
      'conf_bin':'high(>-.3)' if r['conf']>-.3 else ('mid(-.6,-.3]' if r['conf']>-.6 else 'low(<=-.6)'),
      'since_stall':since,'run':run,'return6':return6,'remaining_miss':int(miss[i:].sum()),
      'stuck':ex.get('stuck_n',0),'noprog':ex.get('noprog_n',0),'motion':ex.get('motion',0),'flags':int(flags[i]),
      'max_prog_gain':float(np.max(prog[start:i+1])-startprog),'run_age':i-start}
   if not miss[:i].any():first.append(f)
   if run in landmarks and run not in seen_land:landmarks[run].append(f);seen_land.add(run)
   if run==1 and 'a_exec' in r:
    kk=np.array(r['topk']);w=np.exp(np.array(r['scores'])-r['scores'][0]);w/=w.sum()
    ah=lib.action[kk,:5,:7]/sig;pred=np.einsum('n,ntd->td',w,ah);teacher=np.array(r['a_exec'])/sig
    f['head_diff']=float(np.sqrt(np.mean((pred-teacher)**2)))
    f['teacher_grip_diff']=float(((pred[:,6]>=0)!=(teacher[:,6]>=0)).mean())
    f['teacher_available_member_error']=float(np.sqrt(((ah-teacher)**2).mean((1,2))).min())
    f['direction_opposed']=bool(np.dot(pred[:,:3].mean(0),teacher[:,:3].mean(0))<0)
   if run==1:events.append(f)
   for k in [5,10,15,20]:
    # Online rule: k consecutive MISS, no progress beyond .03 since run began, current stuck >=2.
    if k not in rulehit and run>=k and f['max_prog_gain']<.03 and f['stuck']>=2:rulehit[k]=i
  for k,i in rulehit.items():cost_rules[str(k)].append({'success':ok,'remaining':int(miss[i:].sum()),'uid':u,'step':i})
  eps.append({'uid':u,'task':rr[0]['task_id'],'init':rr[0]['init'],'success':ok,'base_success':b,'n':n,'miss':int(miss.sum())})
 out={'arm':arm,'suite':suite,'scale':scale,'episodes':len(eps),'sr':sum(e['success'] for e in eps)/len(eps),
      'n_dec':sum(e['n'] for e in eps),'n_miss':sum(e['miss'] for e in eps),'paired_rescues':sum(e['success'] and not e['base_success'] for e in eps),
      'paired_losses':sum(not e['success'] and e['base_success'] for e in eps),
      'first':{f:group(first,f) for f in ['time','progress_bin','conf_bin','gexec','phase','since_stall','reason']},
      'landmarks':{k:group(v,'since_stall') for k,v in landmarks.items()},
      'landmark_all':{k:{'n':len(v),'success':sum(x['success'] for x in v),'return6':sum(x['return6'] for x in v),'remaining_miss':sum(x['remaining_miss'] for x in v)} for k,v in landmarks.items()},
      'miss_start':{f:group(events,f) for f in ['time','progress_bin','conf_bin','gexec','phase','since_stall','reason']},
      'cost_rules':{k:{'affected':len(v),'affected_success':sum(x['success'] for x in v),'remaining_miss':sum(x['remaining'] for x in v),'episodes':v} for k,v in cost_rules.items()},
      'events':events,'per_episode':eps}
 out['ir']=.152+.848*out['n_miss']/out['n_dec']
 (OUT/f'p2_recovery_{arm}.json').write_text(json.dumps(out,indent=2))
 print(arm,'SR',out['sr'],'IR',round(out['ir'],4),'landmarks',out['landmark_all'],'cost', {k:{a:b for a,b in v.items() if a!='episodes'} for k,v in out['cost_rules'].items()},flush=True)
 return {k:out[k] for k in ['arm','suite','scale','episodes','sr','n_dec','n_miss','ir','paired_rescues','paired_losses','first','landmark_all','miss_start']}

if __name__=='__main__':
 jobs=[]
 for s,ss in [('l10','l10'),('spatial','sp')]:
  for a in ['g','awm_h70','awm_h50','ev_h70','perk3','awm500_h70']+(['g500','perk5'] if s=='l10' else []):
   jobs.append((f'r3mx_p_{ss}_{a}',s,500 if '500' in a else 50))
 with ProcessPoolExecutor(max_workers=4) as pool:out=list(pool.map(work,jobs))
 (OUT/'p2_recoverability_summary.json').write_text(json.dumps(out,indent=2))
