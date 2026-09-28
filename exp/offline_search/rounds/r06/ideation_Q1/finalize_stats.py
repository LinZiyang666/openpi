"""Recompute vectorized rank intervals from the audited task table without raw-data ingestion."""
import ast,json,warnings,collections
import numpy as np
from scipy.stats import spearmanr,rankdata
from score import O
SEED=20260928;BOOT=2000
for n in ast.parse((O/'analyze.py').read_text()).body:
 if isinstance(n,ast.FunctionDef) and n.name in ('rho','corr'):exec(compile(ast.Module(body=[n],type_ignores=[]),'<correlation-function>','exec'))
R=json.loads((O/'closed_loop_records.json').read_text());out={}
focus=['loeo_err5','loeo_exec10','loeo_succ1','inf_err5','inf_succ1','cache_err5','cache_exec10','cache_succ1','cache_d_self','cache_covered','cache_q_online','loeo_d_self','loeo_covered','loeo_longest_bad_fraction','inf_longest_bad_fraction','cache_longest_bad_fraction','loeo_d_pair','loeo_err10','loeo_succ2','loeo_q','inf_d_pair','inf_d_self','inf_covered','inf_err10','inf_exec10','inf_succ2','inf_q','inf_q_online','cache_d_pair','cache_err10','cache_succ2','cache_q','library_success','log_episodes']
for panel,rows in R.items():
 out[panel]={k:{m:corr(rows,k,m) for m in ['global','task']} for k in focus}
 for k in ['loeo_q','inf_q','inf_d_pair','inf_err10','inf_succ2','inf_q_online','log_episodes']:out[panel][k].update({m:corr(rows,k,m) for m in ['within_config','within_task']})
 print(panel,flush=True)
(O/'rank_correlations.json').write_text(json.dumps(out,indent=2,allow_nan=False))
bycell={}
for cell in sorted({r['cell'] for r in R['A']}):
 rows=[r for r in R['A'] if r['cell']==cell];bycell[cell]={k:corr(rows,k,'global') for k in ['loeo_q','inf_q','inf_d_pair','inf_err10','inf_succ2','log_episodes']}
(O/'rank_within_cell.json').write_text(json.dumps(bycell,indent=2,allow_nan=False))
print('DONE',flush=True)
