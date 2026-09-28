"""Freeze one existing, blind-compatible OAT knob per cell using two held-episode losses.
The one-SE evidence rule is a declared design constraint, not a statistically guaranteed SR rule.
"""
import json,pathlib,numpy as np,hashlib
O=pathlib.Path(__file__).parent;R=pathlib.Path('/home/weiland/trace_runs/os_closed_loop')
NAMES=['base','kref5','kref8','ridge1.0','state3.0']
def pick(d,ts):
 b=d['table']['base'];accepted=[];scores={}
 for n in NAMES[1:]:
  vals={}
  for k in ['action_RMS','state_RMS']:
   dd=np.array([d['table'][n][k]['tasks'][str(t)]-b[k]['tasks'][str(t)] for t in ts]);vals[k]={'mean_delta':float(dd.mean()),'SE_across_tasks':float(dd.std(ddof=1)/np.sqrt(len(dd)))}
  scores[n]=vals
  if all(v['mean_delta'] < -v['SE_across_tasks'] for v in vals.values()):accepted.append(n)
 best=min(accepted,key=lambda n:np.mean([d['table'][n]['state_RMS']['tasks'][str(t)] for t in ts])) if accepted else 'base'
 return best,scores
choices=[];arms=[]
for cell in ['pi05_spatial','pi05_l10','groot_spatial','groot_l10']:
 for scale in [50,500]:
  p=O/f'{"stock_" if scale==500 else "verified_"}library_{cell}_{scale}.json';d=json.loads(p.read_text());best,scores=pick(d,list(range(10)));cv=[]
  for t in range(10):
   n,_=pick(d,[k for k in range(10) if k!=t]);cv.append({'task':t,'choice':n,**{k:d['table'][n][k]['tasks'][str(t)]-d['table']['base'][k]['tasks'][str(t)] for k in ['action_RMS','state_RMS']}})
  if any(np.mean([r[k] for r in cv])>0 for k in ['action_RMS','state_RMS']):best='base'
  kw={'lib':'current' if scale==50 else 'big','kref':5 if scale==50 else 8}
  if best.startswith('kref'):kw['kref']=int(best[4:])
  elif best.startswith('ridge'):kw['ridge_main']=float(best[5:])
  elif best.startswith('state'):kw['state_scale']=float(best[5:])
  rec={'cell':cell,'scale':scale,'selected':best,'kwargs':kw,'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scores':scores,'LOTO':cv,'LOTO_mean_delta':{k:float(np.mean([r[k] for r in cv])) for k in ['action_RMS','state_RMS']},'provenance':'deployed demos only at 50; at 500 inherits stock fixed current-library action units, conservatively labeled borrowed big-library information (normalization only)'}
  choices.append(rec);model,suite=cell.split('_');letter='p' if model=='pi05' else 'g';ss='sp' if suite=='spatial' else 'l10'
  for treatment in ['hand','solve']:
   params={'lib':'current' if scale==50 else 'big','kref':5 if scale==50 else 8} if treatment=='hand' else dict(kw)
   use3='ridge_main' in kw # Paired class twin prevents tie-order changes from confounding ridge.
   if use3:params.setdefault('ridge_main',.1)
   if model=='pi05':
    base='exp.offline_search.rounds.r04.k1_blind.blind_awm:'+('BlindAWM3' if use3 else 'BlindAWM');params.update(serving='anchor_tail',budget=1,gates='budget_only')
    method='exp.offline_search.rounds.r04.k7_guard.judge:VisionConfirmedBlindMixedJudge';kwargs={'base':base,'base_kwargs':params,'events':'none','progress_guard':'noprog_span','stuck_guard':'vision_confirmed','noprog_n':3};pargs=['--os-no-shadow-native','--os-blind','--os-judge','guard_only']
   else:
    method='exp.offline_search.rounds.r03.h1_trap.awm3:AWM3' if use3 else 'exp.offline_search.rounds.r02.g1_awm.awm:AWM';kwargs=params;pargs=['--os-no-shadow-native']
   arms.append({'arm':f'r5b_{letter}_{ss}_{scale}_{treatment}','model':model,'suite':'libero_spatial' if suite=='spatial' else 'libero_10','cell':cell+'_cache','mode':'plugin','method':method,'kwargs':kwargs,'plugin_args':pargs,'full_model':model=='pi05','selected_rule':best,'run_needed':best!='base','note':'Proposal only; coordinator must prefit into its own output path, emit configs and run. No server launched.'})
(O/'solver_choices.json').write_text(json.dumps(choices,indent=2));(O/'arms_proposed.json').write_text(json.dumps(arms,indent=2))
print([(r['cell'],r['scale'],r['selected'],r['LOTO_mean_delta']) for r in choices])
