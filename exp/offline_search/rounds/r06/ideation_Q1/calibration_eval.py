import json,collections,ast,warnings
import numpy as np
from scipy.stats import spearmanr,rankdata
from score import O,macro
# Reuse just the pure correlation functions, without rerunning source ingestion.
a=ast.parse((O/'analyze.py').read_text());ns=globals();ns['BOOT']=2000;ns['SEED']=20260928
for n in a.body:
 if isinstance(n,ast.FunctionDef) and n.name in ('rho','corr'):exec(compile(ast.Module(body=[n],type_ignores=[]),'<correlation-function>','exec'),ns)
R=json.loads((O/'closed_loop_records.json').read_text())['A'];src=json.loads((O/'source_index.json').read_text());sup=json.loads((O/'supplement.json').read_text());factors={(r['tag'],r['metric']):r['factor'] for r in sup['small_calibration']};out=[]
for lab in list(dict.fromkeys(r['label'] for r in R)):
 rr=[r for r in R if r['label']==lab];tag=rr[0]['tag'];a=np.load(O/(tag+'_inf.npz'));pred=1/(1+a['d_pair']+factors[tag,'err10']*a['pred_err']+factors[tag,'succ2']*a['pred_succ']);pairs={}
 for line in open(src['arms'][lab]['journal']):
  e=json.loads(line)
  if e.get('accepted') and e.get('status') in ('done','failed') and not e.get('error'):
   key=tuple(map(int,e['task_uid'].split(':')[-2:]));pairs[key]=int(e['success'])
 for r in rr:
  t=r['task'];z=(a['task']==t)&(a['init']>=5);r=r.copy();r['y']=float(np.mean([v for (task,i),v in pairs.items() if task==t and i>=5]));r['calibrated_q']=macro(pred[z],a['ep'][z]);r['uncalibrated_q']=macro(a['q_online'][z],a['ep'][z]);out.append(r)
res={k:{mode:corr(out,k,mode) for mode in ['global','task','within_config','within_task']} for k in ['calibrated_q','uncalibrated_q']}
(O/'calibration_eval.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
