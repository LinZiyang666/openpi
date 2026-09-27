"""Replay guard/cap budgets on ACCEPTED observed trajectories, never predict SR.
Counterfactual after a changed decision; physical trajectory and extras stay frozen.
"""
import os,json,pathlib,collections,concurrent.futures,numpy as np
OUT=pathlib.Path(__file__).resolve().parent
RUN=pathlib.Path('/home/weiland/trace_runs/os_closed_loop/r03_mx/runs')
S1,S2,S3=.152,.410,.438

def episodes(arm):
 d=RUN/arm;accept={}
 for line in (d/'client/journal.jsonl').open():
  x=json.loads(line)
  if x.get('status') in ('done','failed') and x.get('accepted',True) and not x.get('error'):accept[x['task_uid']]=(int(x.get('attempt',1)),bool(x['success']))
 rows={}
 for f in sorted(d.glob('server_*/decisions_*.jsonl')):
  for line in f.open():
   x=json.loads(line)
   if x.get('ev')!='dec' or x['uid'] not in accept:continue
   if int(x.get('attempt',1))!=accept[x['uid']][0]:continue
   rows[(x['uid'],x['step'])]=x
 out=collections.defaultdict(list)
 for (uid,step),r in sorted(rows.items()):out[uid].append(r)
 return [(seq,accept[uid][1]) for uid,seq in out.items()]

def run(arm):
 eps=episodes(arm);N=sum(len(e) for e,s in eps)
 stats={'arm':arm,'episodes':len(eps),'decisions':N,'successes':sum(s for e,s in eps),'observed_miss':sum(not r['hit'] for e,s in eps for r in e),'variants':{}}
 missrows=[r for e,s in eps for r in e if not r['hit']]
 stats['reason_counts']=dict(collections.Counter(r['judge'] for r in missrows))
 stats['flag_counts']={str(i):sum(bool(int(r['extras'].get('os_flags',0))&(1<<(i-1))) for r in missrows) for i in range(1,5)}
 stats['mean_s1_ms']=float(np.mean([r['s1_ms'] for e,s in eps for r in e]))
 stats['mean_s23_ms']=float(np.mean([r['s23_ms'] for r in missrows]))
 configs=[('observed',0,0,False),('guard3',3,0,False),('guard4',4,0,False),('guard5',5,0,False),('guard4_cap8',4,8,False),('guard3_cap8',3,8,False),('guard4_cap12',4,12,False),('guard3_reset',3,0,True),('guard4_reset',4,0,True)]
 if not all('os_flags' in r['extras'] for e,s in eps for r in e):configs=configs[:1]
 configs += [(f'periodic{k}',0,k,False) for k in [6,8,12,16,24,32]]
 for name,nprog,cap,reset in configs:
  nh=nm=firstS=firstF=0;missS=missF=0;runs=[];touched=0;counts=[]
  for seq,success in eps:
   hr=mr=0;lastmiss=False;since=0;em=0
   for row in seq:
    step=row['step'];ex=row['extras']
    if name=='observed':m=not row['hit']
    elif name.startswith('periodic'):m=step%cap==cap-1
    else:
     # All non-progress guards remain exactly as observed, including overlaps.
     flags=int(ex['os_flags'])&7
     if reset and lastmiss:since=0
     # noprog_n is number of consecutive NONADVANCING TRANSITIONS, not rows.
     since=min(since+1,int(ex.get('noprog_n',0))) if step else 0
     npv=since if reset else ex.get('noprog_n',0)
     m=bool(flags or npv>=nprog-1 or cap and hr>=cap)
    nm+=m;nh+=not m;em+=m
    if m:mr+=1;hr=0
    else:
     if mr:runs.append(mr)
     mr=0;hr+=1
    lastmiss=m
   if mr:runs.append(mr)
   touched+=em>0
   counts.append(em)
   if success:missS+=em
   else:missF+=em
  m=nm/N
  stats['variants'][name]={'misses':nm,'miss_share':m,'IR_K10':S1+m*(S2+S3),'IR_K2':S1+m*(S2+S3*.2),'IR_K1':S1+m*(S2+S3*.1),'episodes_touched':touched,'mean_miss_run':float(np.mean(runs)) if runs else 0,'misses_success':missS,'misses_failure':missF}
 return stats
if __name__=='__main__':
 arms=['r3mx_p_l10_g','r3mx_p_l10_g500','r3mx_p_sp_g','r3mx_p_l10_perk5','r3mx_p_sp_awm_h70','r3mx_p_l10_awm_h70']
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:out=list(ex.map(run,arms))
 (OUT/'schedule_replay.json').write_text(json.dumps(out,indent=2))
 for a in out:
  print(a['arm'],a['episodes'],a['decisions'],a['observed_miss'])
  for n in ['observed','guard4','guard5','guard3_reset','guard4_cap8','periodic8','periodic12']:
   if n not in a['variants']:continue
   x=a['variants'][n];print(n,round(x['miss_share'],6),round(x['IR_K10'],6),round(x['IR_K2'],6))
