"""Compact, auditable tables for REPORT_2; no original first-pass artifacts are written."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
out={'libraries':json.loads((OUT/'evidence_summary.json').read_text())['cells'],'recovery':{},'cross_task':[],'tokens':[],'kernel':[]}
for arm in ['g','g500','awm_h70','awm500_h70']:
 d=json.loads((OUT/f'p2_recovery_r3mx_p_l10_{arm}.json').read_text())
 out['recovery'][arm]={k:d[k] for k in ['sr','n_dec','n_miss','ir','paired_rescues','paired_losses','first','landmark_all']}
for m in ['pi05','groot']:
 for n in [50,500]:
  d=json.loads((OUT/f'p2_cross_task_{m}_{n}.json').read_text())
  for reg in ['cache','inf']:
   for kind in ['other_task','shared_object','spatial']:
    rr=[r for r in d['rows'] if r['regime']==reg and r['kind']==kind];N=sum(r['n'] for r in rr);S=sum(r['selected_n'] for r in rr)
    avg=lambda k:sum(r[k]*r['n'] for r in rr)/N
    sel=lambda k:sum(r[k]*r['selected_n'] for r in rr if r['selected_n'])/S if S else None
    out['cross_task'].append({'model':m,'scale':n,'regime':reg,'kind':kind,'n':N,'selected':S,'share':S/N,
      'chosen_original_error':sel('selected_original_nn_error'),'chosen_donor_error':sel('selected_donor_nn_error'),
      'chosen_grip_mismatch':sel('selected_grip_mismatch'),'oracle_own':avg('own_oracle_error'),'oracle_union':avg('union_oracle_error')})
 for s in ['spatial','l10']:
  d=json.loads((OUT/f'p2_tokens_{m}_{s}.json').read_text())
  for n in ['50','500']:
   out['tokens'].append({'model':m,'suite':s,'scale':int(n),'n_tokens':d['n_tokens'],'episodes':d['episodes'],
     'metrics':d['scales'][n]['metrics']})
   a=json.loads((OUT/f'p2_kernel_{m}_{s}_{n}.json').read_text());out['kernel'].append({k:a[k] for k in ['model','suite','scale','L','demo_motion_median','groups']})
for k in out['libraries']:
 k['value_table_bytes']=k['fit_bytes']+48*5*4
out['coverage']={'mixed_arms':len(list(OUT.glob('p2_recovery_r3mx_*.json'))),'mixed_episodes':sum(x['episodes'] for x in json.loads((OUT/'p2_recoverability_summary.json').read_text())),
 'pure_cache_awm_episodes':4000,'token_closed_loop_episodes':200,'token_decisions':sum(x['n_tokens'] for x in out['tokens'] if x['scale']==50)}
(OUT/'p2_summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out['coverage']))
for t in out['tokens']:
 if t['suite']=='l10':
  print(t['model'],t['scale'],{k:t['metrics'][k]['step5'] for k in ['awm_distance','cam1_patch_p95','cam1_pooled_mean']})
