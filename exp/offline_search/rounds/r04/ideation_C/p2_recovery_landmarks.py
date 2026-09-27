"""Episode-balanced conditional outcomes, cost ceilings, and policy/candidate action decomposition."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent

def stat(rr):
 n=len(rr);s=sum(r['success'] for r in rr);b=[r for r in rr if not r['base_success']];p=s/max(n,1);z=1.96;den=1+z*z/max(n,1)
 half=z*np.sqrt(p*(1-p)/max(n,1)+z*z/(4*max(n,1)**2))/den;cen=(p+z*z/(2*max(n,1)))/den
 return {'episodes':n,'success':s,'success_fraction':p,'wilson95':[cen-half,cen+half],'base_failed':len(b),'paired_rescued':sum(r['success'] for r in b),'return6':sum(r['return6'] for r in rr)}
out={}
for file in sorted(OUT.glob('p2_recovery_r3mx_*.json')):
 d=json.loads(file.read_text());a={}
 for field in ['since_stall','time','progress_bin','gexec','phase','conf_bin','reason']:
  seen=set();bins={}
  for r in d['events']:
   key=(r['uid'],str(r[field]))
   if key in seen:continue
   seen.add(key);bins.setdefault(str(r[field]),[]).append(r)
  a[field]={k:stat(rr) for k,rr in bins.items()}
 a['rules']={k:{'episodes':v['affected'],'success':v['affected_success'],'max_historical_suffix_misses':v['remaining_miss'],
      'fixed_path_ir_ceiling':.848*v['remaining_miss']/d['n_dec'],'observed_successes_at_risk_pp':100*v['affected_success']/500} for k,v in d['cost_rules'].items()}
 a['policy_comparison']={}
 for success in [False,True]:
  rr=[r for r in d['events'] if 'head_diff' in r and r['success']==success]
  a['policy_comparison'][str(success)]={'events':len(rr),**{f:float(np.mean([r[f] for r in rr])) if rr else None for f in ['head_diff','teacher_available_member_error','teacher_grip_diff','direction_opposed']}}
 out[d['arm']]=a
(OUT/'p2_recovery_landmarks.json').write_text(json.dumps(out,indent=2))
for arm in ['r3mx_p_l10_g','r3mx_p_l10_g500','r3mx_p_l10_awm_h70','r3mx_p_l10_awm500_h70']:
 print(arm, {k:out[arm][k] for k in ['since_stall','conf_bin']})
