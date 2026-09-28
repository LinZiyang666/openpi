"""Check completeness, repeat parity, affinity and timing sample counts."""
import os;os.sched_setaffinity(0,{35})
from common import *
configs=json.loads((OUT/'configs.json').read_text());issues=[];bench=[]
for c in configs:
 rr=[]
 for rep in ((2,3) if c["id"]=="groot_l10_500_BlindAWM" else (1,2)):
  p=OUT/f"bench_{c['id']}_r{rep}.json"
  if not p.exists():issues.append('missing '+str(p));continue
  r=json.loads(p.read_text());rr.append(r)
  if r['n']<1000 or r['vision_ms']['n']!=len(r['samples_ms']):issues.append('count '+c['id'])
  if r['affinity']!=[34]:issues.append('affinity '+c['id'])
  budget=c.get('kwargs',{}).get('base_kwargs',c.get('kwargs',{})).get('budget',2)
  if 'blind' in r and r['blind']['success_ms']['n']<1000 and budget>0:issues.append('blind count '+c['id'])
  if not all(np.isfinite(r['samples_ms'])):issues.append('nonfinite '+c['id'])
 if len(rr)==2:
  if rr[0]['digests']!=rr[1]['digests']:issues.append('repeat output parity '+c['id'])
  if rr[0].get('blind',{}).get('digests')!=rr[1].get('blind',{}).get('digests'):issues.append('blind repeat parity '+c['id'])
  bench.append(dict(config=c['id'],selected_runs=[r['run'] for r in rr],p50_r1=rr[0]['vision_ms']['p50'],p50_r2=rr[1]['vision_ms']['p50'],p50_variation_pct=100*(max(r['vision_ms']['p50'] for r in rr)/min(r['vision_ms']['p50'] for r in rr)-1)))
for name,expected in [('scaling',20),('aux',21)]:
 for rep in (1,2):
  p=OUT/f'{name}_r{rep}.json'
  if not p.exists():issues.append('missing '+str(p));continue
  rr=json.loads(p.read_text())
  if len(rr)!=expected:issues.append(f'{name} length {len(rr)}')
  for r in rr:
   if r['ms']['n']<1000:issues.append(f'{name} small n')
for model in ('pi05','groot'):
 for suite in ('spatial','l10'):
  for rep in (1,2):
   p=OUT/f'native_{model}_{suite}_r{rep}.json'
   if not p.exists():issues.append('missing '+str(p));continue
   r=json.loads(p.read_text())
   if r['recorded_top1_equal']!=r['recorded_n']:issues.append('native parity '+str(p))
   if r['ms']['n']<1000:issues.append('native small n '+str(p))
cs=('pi05_l10_50_AWM','pi05_l10_500_AWM','pi05_l10_50_MixedJudge','pi05_l10_500_MixedJudge','pi05_l10_500_BlindMixedJudge','groot_l10_50_AWM','groot_l10_500_AWM')
for c in cs:
 rr=[]
 for rep in (1,2):
  p=OUT/f'concurrency_{c}_r{rep}.json'
  if not p.exists():issues.append('missing '+str(p));continue
  r=json.loads(p.read_text());rr.append(r)
  if [x['threads'] for x in r]!=[1,4,8,16,24,32]:issues.append('thread sweep '+c)
  if any(x['ms']['n']<1000 for x in r):issues.append('thread count '+c)
  if any(x['sorted_digests']!=r[0]['sorted_digests'] for x in r):issues.append('thread parity '+c)
 if len(rr)==2 and rr[0][0]['sorted_digests']!=rr[1][0]['sorted_digests']:issues.append('thread process repeat parity '+c)
ver=json.loads((OUT/'replay_verification.json').read_text())
if any(r['topk_equal']!=r['decisions'] or r['action_equal']!=r['decisions'] or r['failures'] for r in ver):issues.append('logged replay parity')
r=dict(issues=issues,paired_benchmark_configs=len(bench),benchmark_repeats=bench,logged_replay_decisions=sum(r['decisions'] for r in ver),logged_replay_blind=sum(r['blind'] for r in ver),snapshot=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
dump(OUT/'AUDIT.json',r);print(json.dumps(r,indent=2));assert not issues
