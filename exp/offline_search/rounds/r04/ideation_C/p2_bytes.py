"""Exact existing pickle bytes; analytic token and value-table additions (decimal bytes)."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent;out={}
for n in ['r3mx_p_sp_g','r3mx_p_l10_g','r3mx_p_l10_g500','r3mx_p_sp_awm500_h70','r3mx_p_l10_awm_h70','r3mx_p_l10_awm500_h70']:
 f=Path('/home/weiland/trace_runs/os_closed_loop/r03_mx/fits')/(n+'.pkl')
 out[n]={'base_bytes':f.stat().st_size,'plus_960_byte_table':f.stat().st_size+960}
d=json.loads((OUT/'evidence_summary.json').read_text())
out['token_bytes']=[{'model':c['model'],'suite':c['suite'],'scale':c['scale'],'full_wrist_token_bytes':c['L']*1048576,'32proj_pkl_bytes':c['fit_bytes']+c['L']*16384+262144} for c in d['cells']]
(OUT/'p2_bytes.json').write_text(json.dumps(out,indent=2))
