"""Build reviewable tables from diagnostics and explicit owner cost arithmetic."""
import pathlib,json,numpy as np
O=pathlib.Path(__file__).resolve().parent
rows=[r for p in O.glob('retrieval_*.json') if p.name!='retrieval_validation.json' for r in json.loads(p.read_text())]
rows.sort(key=lambda r:(r['model'],r['suite'],r['library'],r['arm'],r['variant']))
md=['# Refit retrieval diagnostics','', 'Tok query subset; entire candidate library fit and searched; no borrowing. Error is RMS sigma on [:5,:7]. Full baseline validates against AWM. No SR prediction.','', '| Cell | Library | Key | N | overlap16 | top1 | error | Δ error | split .8 | grasp N / split .8 | release N / split .8 |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 a=r['metrics']['all'];g=r['metrics']['grasp'];z=r['metrics']['release']
 md.append(f"| {r['model']}-{r['suite']}-{r['arm']} | {r['episodes']}/{r['entries']} | {r['variant']} | {a['n']} | {a['overlap16']:.4f} | {a['top1_agree']:.4f} | {a['err']:.5f} | {a['delta_err_full']:+.5f} | {a['split08']:.4f} | {g['n']} / {g.get('split08',float('nan')):.4f} | {z['n']} / {z.get('split08',float('nan')):.4f} |")
md+=['','## Compact representation bytes (decimal MB; full valid action =280 B/entry pi05, 448 B/entry GR00T)','', '| Cell | Library | Key | Key B/entry | Fixed MB | Code+action+fixed MB | Deployed MB |','|---|---|---|---:|---:|---:|---:|']
for r in rows:
 if r['arm']!='cache':continue
 size=r['fixed_bytes']+r['entries']*(r['key_bytes']+(280 if r['model']=='pi05' else 448))
 dep={('pi05','spatial'):431,('pi05','l10'):1103,('groot','spatial'):429,('groot','l10'):1068}[r['model'],r['suite']]
 md.append(f"| {r['model']}-{r['suite']} | {r['episodes']} | {r['variant']} | {r['key_bytes']} | {r['fixed_bytes']/1e6:.4f} | {size/1e6:.4f} | {dep} |")
(O/'RETRIEVAL_TABLES.md').write_text('\n'.join(md)+'\n')

S1,S2,S3=.152,.410,.438
T=10.26+27.69+29.57
# s=vision ratio on HIT, D=duplicated cheap pass normalized to stage1 on MISS.
def ir(m,n=10,s=1,D=0,allvision=None):
 if allvision is not None:return S1*allvision+m*(S2+S3*n/10)
 return (1-m)*S1*s + m*(S1*(1+D)+S2+S3*n/10)
rp=json.loads((O/'schedule_replay.json').read_text());by={x['arm']:x for x in rp}
stack=[]
for arm in ['r3mx_p_l10_g500','r3mx_p_l10_g','r3mx_p_sp_g','r3mx_p_l10_perk5']:
 for sched in ['observed','guard4','guard5','guard4_cap8','guard3_reset']:
  if sched not in by[arm]['variants']:continue
  m=by[arm]['variants'][sched]['miss_share']
  stack.append({'arm':arm,'schedule':sched,'m':m,'full_K10':ir(m),'full_K2':ir(m,2),'full_K1':ir(m,1),'reusable_r05_K2':ir(m,2,.5),'duplicate_r05_K2':ir(m,2,.5,.5),'reusable_r025_K2':ir(m,2,.25),'dummy_r_two_thirds_all_K2':ir(m,2,allvision=2/3)})
period=[{'k':k,'m_asymptotic':1/k,'K10':ir(1/k),'K2':ir(1/k,2),'K1':ir(1/k,1)} for k in [3,4,5,6,8,12,16,24,32,48]]
bases=[{'K':n,'L':L,'IR':5/L*(S1+S2+S3*n/10),'cost_per_call_ms':10.26+27.69+29.57*n/10} for L in [5,8,10] for n in [1,2,4,10]]
out={'definition':'owner-rounded shares .152/.410/.438; latency ms from 10.26/27.69/29.57; cheaper-key ratios illustrative, not GPU measured','stacks':stack,'periodic':period,'baselines':bases,'miss_costs':[{'K':n,'s23_ms':27.69+29.57*n/10,'full_ms':10.26+27.69+29.57*n/10,'full_IR':S1+S2+S3*n/10} for n in [1,2,4,5,10]]}
(O/'cost_tables.json').write_text(json.dumps(out,indent=2))
md=['# Stacked IR arithmetic','',out['definition'],'','HIT r=.5/.25 = hypothetical measured-relative vision budget. Reusable means continue to full tower on MISS; duplicate means recompute full vision after the cheap branch. Frozen trajectory m: not a new rollout.','', '| Arm / frozen schedule | m | full K10 | full K2 | full K1 | reuse r=.5 K2 | duplicate r=.5 K2 | reuse r=.25 K2 | dummy r=2/3 all K2 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in stack:md.append('| '+x['arm']+'/'+x['schedule']+' | '+' | '.join(f'{x[k]:.5f}' for k in ['m','full_K10','full_K2','full_K1','reusable_r05_K2','duplicate_r05_K2','reusable_r025_K2','dummy_r_two_thirds_all_K2'])+' |')
md+=['','| Period k | K10 | K2 | K1 |','|---:|---:|---:|---:|']
for x in period:md.append('| '+str(x['k'])+' | '+' | '.join(f'{x[k]:.5f}' for k in ['K10','K2','K1'])+' |')
md+=['','| Pure policy denoise K | Execute L | IR per 5 controls | Call ms |','|---:|---:|---:|---:|']
for x in bases:md.append(f"| {x['K']} | {x['L']} | {x['IR']:.5f} | {x['cost_per_call_ms']:.3f} |")
(O/'COST_TABLES.md').write_text('\n'.join(md)+'\n')
# Actual fit artifact sizes, read-only filesystem stat.
fits=[]
base=pathlib.Path('/home/weiland/trace_runs/os_closed_loop')
for group in ['r02_g50','r02_g500','r03_mx']:
 for p in (base/group/'fits').glob('*.pkl'):
  if any(a in p.name for a in ['cl2','_g.pkl','_g500.pkl','awm500_h70','awm_h70']):fits.append({'path':str(p),'bytes':p.stat().st_size,'MB_decimal':p.stat().st_size/1e6})
(O/'fit_bytes.json').write_text(json.dumps(fits,indent=2))
# Existing reduced-step LIBERO measurements; macro SR computed from task cell n/sr.
step=[]
for model in ['pi05','groot']:
 for suite in ['spatial','10']:
  p=pathlib.Path('exp/step_diag/data/analysis')/f'warm_variants_{model}_libero_{suite}.json'
  for panel,d in json.loads(p.read_text()).items():
   if not isinstance(d,dict) or 'tasks' not in d:continue
   arms={a for t in d['tasks'].values() for a in t.get('cells',{})}
   for arm in sorted(arms):
    if not (arm.startswith('plain') or arm=='full' or arm.startswith('warm_t')):continue
    cs=[t['cells'][arm] for t in d['tasks'].values() if arm in t.get('cells',{})]
    step.append({'model':model,'suite':suite,'panel':panel,'arm':arm,'episodes':sum(x['n'] for x in cs),'sr':sum(x['n']*x['sr'] for x in cs)/sum(x['n'] for x in cs),'complete':all(x['complete'] for x in cs),'equal_nfe':all(x['equal_nfe'] for x in cs),'source':str(p)})
(O/'step_evidence.json').write_text(json.dumps(step,indent=2))
print('retrieval rows',len(rows),'cost rows',len(stack),'formal step rows',len(step))
