"""Small reproducible index for REPORT's numeric tables; reads only finalized artifacts."""
import json,pathlib,numpy as np
O=pathlib.Path(__file__).parent;R=pathlib.Path('/home/weiland/trace_runs/os_closed_loop')
from select_configs import pick
# Importing select_configs deterministically re-emits this directory's proposed configurations.
summary={'library':[],'bytes':[],'guard_sensitivity':[]}
for cell in ['pi05_spatial','pi05_l10','groot_spatial','groot_l10']:
 for scale in [50,500]:
  d=json.loads((O/f'verified_library_{cell}_{scale}.json').read_text());c,_=pick(d,list(range(10)));cvs=[]
  for t in range(10):
   n,_=pick(d,[j for j in range(10) if j!=t]);cvs.append({k:d['table'][n][k]['tasks'][str(t)]-d['table']['base'][k]['tasks'][str(t)] for k in ['action_RMS','state_RMS']})
  if any(np.mean([r[k] for r in cvs])>0 for k in ['action_RMS','state_RMS']):c='base'
  summary['library'].append({'cell':cell,'scale':scale,'deployed_unit_choice':c,'held_episodes':d['heldout_episodes'],'query_rows':sum(r['n'] for r in d['episode_scores'] if r['config']=='base'),'NLL':d['LOTO']['state_NLL'],'OAS_mean_ridge':d['table']['ridge_oas']['ridge_equiv']['mean']})
  m,s=cell.split('_');letter='p' if m=='pi05' else 'g';short='sp' if s=='spatial' else s;p=R/f'r02_g{scale}/fits'/f'oscl{scale}_{letter}_{short}_cl2.pkl';summary['bytes'].append({'cell':cell,'scale':scale,'path':str(p),'bytes':p.stat().st_size})
for name in ['r4k7_p_l10_50_tail1ug','r4k7_p_l10_500_tail1ug','r4k7_p_sp_500_tail1ug']:
 p=R/'r04_k7/fits'/f'{name}.pkl';summary['bytes'].append({'arm':name,'path':str(p),'bytes':p.stat().st_size})
for row in json.loads((O/'calibration_results.json').read_text())['demo']:
 rec={'cell':row['cell'],'scale':row['scale']}
 for name,mp,cp,n in [('strict',5,99,2),('stock',10,95,2),('loose',30,90,2),('run1',10,95,1),('run3',10,95,3)]:
  rr=next(r for r in row['grid'] if (r['m_pct'],r['c_pct'],r['stuck_thr'])==(mp,cp,n));rec[name]=rr['successful_episode_flag_rate']
 summary['guard_sensitivity'].append(rec)
summary['arithmetic']={'g50_m':6664/32967,'g500_m':2983/29340,'tail50_v':18566/31186,'tail500_v':16547/28888,'direct_call_savings_g50':.848*500/32967,'direct_call_savings_g500':.848*500/29340,'direct_call_savings_tail50':.848*500/31186,'direct_call_savings_tail500':.848*500/28888}
(O/'measurement_summary.json').write_text(json.dumps(summary,indent=2))
print('Own-unit choices',[(r['cell'],r['scale'],r['deployed_unit_choice']) for r in summary['library']]);print('Guard sensitivity',summary['guard_sensitivity'])
