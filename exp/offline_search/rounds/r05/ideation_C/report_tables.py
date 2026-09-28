"""Generate compact, auditable report tables and cost arithmetic."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
OUT=Path(__file__).resolve().parent
memory=json.loads((OUT/'memory.json').read_text());usage=json.loads((OUT/'usage_summary.json').read_text())
focused=json.loads((OUT/'focused_usage_yield.json').read_text());audit=json.loads((OUT/'log_audit.json').read_text())
rows=[]
def table(head,body):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in body])
rows.append(table(['Cell','Scale','Rows','AWM pickle MB','AWM arrays MB','Arrays with valid full actions MB','Deployed MB','Uniform / task-specific slots'],[
 [r['cell'],r['scale'],r['rows'],f"{r['pickle_bytes']/1e6:.6f}",f"{r['array_bytes']/1e6:.6f}",f"{r['packed_existing_arrays_bytes']/1e6:.6f}",f"{r['deployed_bytes']/1e6:.6f}",f"{r['uniform_capacity_rows']}/{r['ragged_capacity_rows']}"] for r in memory]))
rows.append(table(['Cell','Library','Reported union / L','+ next-two closure','Trap top1 / also successful member','Failed-library rows selected / all','Held-out changed sets'],[
 [r['cell'],r['library'],f"{r['reported_union']}/{r['L']}",r['union_next2'],f"{r['trap_top1']}/{r['trap_also_success_member']}",f"{r['failed_library_rows_selected']}/{r['failed_library_rows']}",f"{r['heldout_changed_reported_set']}/{r['heldout_queries']}"] for r in usage]))
growth=[];prune=[];scene=[];cost=[]
for p in sorted(OUT.glob('library_*.json')):
    r=json.loads(p.read_text());scale=r['scale'];g=r['growth'][-1];cc=g['by_stream']['cache']
    growth.append([r['cell'],scale,g['policy_calls'],g['successful_policy_rows'],g['admitted'],g['dedup'],
      '/'.join(f"{x['by_stream']['cache']['gap_closed_d1']:.4f}" for x in r['growth'][1:]) if scale==50 else f"{cc['relative_d1']:.6f}",
      f"{cc['gap_closed_d16'] if scale==50 else cc['relative_d16']:.6f}",f"{cc['new_top1_share']:.6f}"])
    a=next(x for x in r['pruning'] if x['stream']=='cache' and x['fraction']==.9)
    b=next(x for x in r['pruning'] if x['stream']=='cache' and x['fraction']==.5)
    prune.append([r['cell'],scale,r['full16_union'],r['full16_train_union'],f"{b['all16']:.6f}",f"{a['all16']:.6f}",f"{a['weight_retained']:.6f}"])
    s=next(x for x in r['scene'] if x['stream']=='cache' and x['fraction']==.5)
    scene.append([r['cell'],scale,f"{s['mean_task_rows_retained']:.6f}",f"{s['all16_episode_mean']:.6f}",f"{s['top1_episode_mean']:.6f}",f"{s['weight_retained_episode_mean']:.6f}"])
rows.append(table(['Cell','Start','Policy calls at 250 ep','Successful source rows','Admitted','Dedup','50: d1 gap closure at 50/100/250 ep; 500: d1 ratio','d16 closure or ratio at250','New top1 share'],growth))
rows.append(table(['Cell','Scale','Full16 union','Train union','50% rows: full16 retained','90% rows: full16 retained','90% rows: weight retained'],prune))
rows.append(table(['Cell','Scale','Half-episode scene prior: row share','All16 recall','Top1 recall','Weight recall'],scene))
for r in focused:
    if not r['M']:continue
    cost.append([r['arm'],r['N'],r['V'],r['M'],r['counts'].get('success_M',0),f"{r['IR']:.9f}",f"{r['successful_miss_per_deployment_episode']:.3f}",f"{r['row_gap_fill_episodes_at_stationary_success_yield']:.2f}"])
rows.append(table(['Arm','N','V','M','Successful M','IR','Successful M / deployment ep','Row-gap episodes, stationary yield'],cost))
(OUT/'TABLES.md').write_text('\n\n'.join(rows)+'\n')
fields=Counter();miss=Counter()
for r in audit:fields.update(r['fields']);miss.update(r['miss_fields'])
npzs=[]
for r in audit:
 for p in r['input_npz']:
  a=np.load(p);npzs.append(dict(path=p,N=len(a['hit']),M=int((a['hit']==0).sum())))
tail50=next(r for r in focused if r['arm']=='r4k7_p_l10_50_tail1ug')
tail500=next(r for r in focused if r['arm']=='r4k7_p_l10_500_tail1ug')
ps500=next(r for r in focused if r['arm']=='r4k7_p_sp_500_tail1ug')
extra=14849
summary=dict(raw_N=sum(r['raw_decisions'] for r in audit),accepted_N=sum(r['decisions'] for r in audit),
 raw_M=sum(r['raw_miss'] for r in audit),fields=fields,miss_fields=miss,npz=npzs,
 removed_conflicting_episodes=sum(len(r['conflicting_uids']) for r in audit),
 growth_forecast_scenarios=dict(l10_50_IR_if_m_drops_1_to_2pp=[tail50['IR']-.848*.02,tail50['IR']-.848*.01],
 l10_500_IR_if_m_drops_0_to_halfpp=[tail500['IR']-.848*.005,tail500['IR']],
 spatial_500_IR_if_m_drops_0_to_quarterpp=[ps500['IR']-.848*.0025,ps500['IR']],
 acquisition250_then_500_tail50_IR=(extra+tail50['IR']*tail50['N'])/(extra+tail50['N'])),
 graph_search_increment_per_vision=[.36/67.52,.63/67.52],CPU_search_increment_per_vision=[1.2/67.52,2.2/67.52])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
