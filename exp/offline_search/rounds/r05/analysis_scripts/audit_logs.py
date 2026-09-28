from pathlib import Path
from multiprocessing import Pool
import json,collections,hashlib
import numpy as np
OUT=Path('/tmp/r5_analysis');A=json.loads((OUT/'analysis.json').read_text())['arms'];CL=Path('/home/weiland/trace_runs/os_closed_loop')
def one(item):
 lab,r=item;acc={}
 for l in Path(r['journal']).open():
  d=json.loads(l)
  if d.get('accepted') and d.get('status') in ('done','failed') and not d.get('error'):acc[d['task_uid']]=d
 dec={};duplicates=0;conflict=set();sources=[];start=[];all_gpu=[];ev=collections.Counter()
 for fp in sorted((CL/r['run']/'runs'/r['arm']).glob('server_*/decisions_*.jsonl')):
  sources.append({'path':str(fp),'bytes':fp.stat().st_size})
  for line in fp.open():
   x=json.loads(line);ev[str(x.get('ev'))]+=1
   if x.get('ev')=='startup':start.append({k:x.get(k) for k in ['stage1_mode','gpu_retrieval','method','bytes_per_entry']});continue
   if x.get('ev')!='dec':continue
   if x.get('gpu_retrieval'):all_gpu.append((x.get('ok'),x['step'],x['gpu_retrieval']))
   uid=x.get('uid');a=acc.get(uid)
   if a is None or a.get('attempt')!=x.get('attempt'):continue
   key=(uid,x['step'])
   data={k:x.get(k) for k in ['step','hit','vision','src','ok','judge','look_reason','extras','gpu_retrieval','top1','conf']}
   data['action_hash']=hashlib.sha256(json.dumps(x.get('served_head',x.get('a_exec'))).encode()).hexdigest()
   if key in dec:
    duplicates+=1
    if any(dec[key][k]!=data[k] for k in ['hit','vision','src','top1','conf','action_hash']):conflict.add(uid)
   dec[key]=data
 groups=collections.defaultdict(list)
 for (uid,step),d in dec.items():groups[uid].append(d)
 src=collections.Counter();look=collections.Counter();judge=collections.Counter();tail_eligible=0;policy_tails=0;terminal_misses=0;blind_runs=collections.Counter();d1=[];ntot=V=M=bad=0;tails_from_miss=0;incomplete=[];eps=[];gpus=[];maxd1=0
 for uid,ds in groups.items():
  ds.sort(key=lambda d:d['step']);n=len(ds)
  if [d['step'] for d in ds]!=list(range(n)):incomplete.append(uid)
  v=m=pt=0;br=0;ng=0
  for i,d in enumerate(ds):
   vision=d.get('vision');vision=True if vision is None else vision;hit=d.get('hit');hit=True if hit is None else hit
   source=d.get('src') or ('cache' if hit else 'policy')
   src[source]+=1;look[str(d.get('look_reason'))]+=1;judge[str(d.get('judge'))]+=1;v+=vision;m+=not hit;pt+=source=='policy_tail';bad+=d.get('ok') is False
   if not vision:br+=1
   elif br:blind_runs[br]+=1;br=0
   if not hit:
    if i+1<n:tail_eligible+=1;tails_from_miss+=ds[i+1].get('src')=='policy_tail'
    else:terminal_misses+=1
   if d.get('look_reason')==9 or (d.get('extras') or {}).get('grasp_check')==1:
    ng+=1;d1.append({'uid':uid,'step':d['step'],'success':bool(acc[uid]['success']),'hit':hit,'vision':vision,'judge':d.get('judge'),'base_reason':(d.get('extras')or{}).get('grasp_base_reason')})
   if d.get('gpu_retrieval'):gpus.append((d.get('ok'),d['step'],d['gpu_retrieval']))
  if br:blind_runs[br]+=1
  maxd1=max(maxd1,ng);ntot+=n;V+=v;M+=m;policy_tails+=pt
  eps.append({'uid':uid,'success':bool(acc[uid]['success']),'N':n,'V':v,'M':m,'policy_tails':pt,'grasp':ng})
 def gpu_summary(rr):
  if not rr:return None
  out={'n':len(rr),'failed':sum(not ok for ok,s,g in rr)}
  for k in ['top1_agree','top16_set_agree','top16_order_agree','chunk_agree','confidence_agree']:out[k]={'count':sum(bool(g[k]) for ok,s,g in rr),'fraction':sum(bool(g[k]) for ok,s,g in rr)/len(rr)}
  for k in ['event_ms','gpu_path_event_ms','wall_ms','cpu_query_ms','cpu_path_ms','chunk_max_abs','confidence_abs']:
   ar=[g[k] for ok,s,g in rr if g.get(k) is not None]
   if ar:out[k]={'p50':float(np.median(ar)),'p90':float(np.percentile(ar,90)),'max':max(ar)}
  out['chunk_mismatch_step0']=sum(not g['chunk_agree'] and s==0 for ok,s,g in rr);out['chunk_mismatch_later']=sum(not g['chunk_agree'] and s>0 for ok,s,g in rr)
  out['later_chunk_max']=max(g['chunk_max_abs'] for ok,s,g in rr if s>0)
  return out
 out={'N':ntot,'V':int(V),'M':int(M),'v':V/ntot,'m':M/ntot,'episodes':len(groups),'duplicates':duplicates,'conflicting_uids':sorted(conflict),'incomplete':incomplete,'bad_decisions':bad,'src':dict(src),'look':dict(look),'judge':dict(judge),'blind_runs':dict(blind_runs),'policy_tails':policy_tails,'miss_with_followup':tail_eligible,'tail_after_miss':tails_from_miss,'terminal_misses':terminal_misses,'grasp':d1,'max_grasp_per_ep':maxd1,'gpu_accepted':gpu_summary(gpus),'gpu_all':gpu_summary(all_gpu),'sources':sources,'startup':start,'episodes_data':eps,'events':dict(ev)}
 assert len(groups)==r['n'],(lab,len(groups))
 if 'cost_ledger' in r['cost_source']:assert (ntot,V,M)==(r['N'],r['V'],r['M']),(lab,ntot,V,M,r['N'],r['V'],r['M'])
 return lab,out
if __name__=='__main__':
 prior=json.loads((OUT/'decisions.json').read_text()) if (OUT/'decisions.json').exists() else {}
 items=[(l,r) for l,r in A.items() if r['status']=='OK' and (r['run'].startswith('r05') or r['cost_source'].startswith('legacy') or l.startswith('k7') or l.startswith('wrist'))]
 items=[(l,r) for l,r in items if l not in prior]
 print('Auditing',len(items),'arms with 7 workers + parent',flush=True)
 with Pool(max(1,min(7,len(items)))) as pool:
  out=prior
  for l,r in pool.imap_unordered(one,items):out[l]=r;print(l,r['N'],'duplicates',r['duplicates'],'conflicts',len(r['conflicting_uids']),flush=True)
 (OUT/'decisions.json').write_text(json.dumps(out,indent=1))
 print('Saved',len(out),'arms',sum(x['N'] for x in out.values()),'decisions')
