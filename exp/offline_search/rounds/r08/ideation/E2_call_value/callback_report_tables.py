"""Read only E2 discovery artifacts and emit compact report tables."""
import json
from pathlib import Path

import pandas as pd

from callback_calls import OUT, save


p=OUT/'summary'
arms=pd.read_csv(p/'arms.csv')
effects=pd.read_csv(p/'centered_first_entry.csv')
support=pd.read_csv(p/'support_by_stage.csv')
costs=pd.read_csv(p/'stage_costs.csv')
raw=pd.read_csv(p/'raw_tool_effects.csv')
interaction=pd.read_csv(p/'interactions.csv')
pooled=pd.read_csv(p/'pooled_effects.csv')
totals=['episodes','successes','decisions','fresh','supported','p_zero','p_one','nominal_one_nonstall','forced_stall',
        'supported_p_over_95','supported_p_under_05','dev_entries','dev_entries_supported','controls','V','M','work']
numbers=dict(total=arms[totals].sum().to_dict(),groups=arms.groupby(['variant','size'])[totals].sum().reset_index().to_dict('records'))
numbers['checks']=arms[['formula_mismatches','hash_coin_max_error','coin_treatment_mismatches','dispatch_mismatches',
                       'source_mismatches','physical_available','stage_pre_present','prefix_checks','prefix_failures']].sum().to_dict()
numbers['reader_issues']={}
numbers['triggers']=[]
cross=[]
firsts=[]
for row in arms.to_dict('records'):
    name=row['arm'];base=OUT/'calls'/name
    summary=json.loads((base/'summary.json').read_text())
    numbers['reader_issues'][name]=summary['reader_issues']
    for r in summary['triggers']:
        numbers['triggers'].append(dict(arm=name,variant=row['variant'],**r))
    df=pd.read_parquet(base/'decisions.parquet')
    a=df[df.fresh].copy()
    counts=a.groupby(['r7_stage','truth_pre']).size().reset_index(name='anchors')
    cross.extend(counts.assign(arm=name,variant=row['variant'],size=row['size']).to_dict('records'))
    s=a[a.eligible.eq(True)&a.p.between(0,1,inclusive='neither')]
    first=s[s.truth_pre.eq('approach')].sort_values('decision_seq').drop_duplicates('episode_key')
    firsts.append(dict(arm=name,reached=len(first),zero_predicate_20=int(first.predicate_delta_20.eq(0).sum()),
                       zero_carry_20=int(first.carry_gain_20.eq(0).sum())))
numbers['first_approach']=firsts
pd.DataFrame(cross).to_csv(p/'stage_overlap.csv',index=False)
save(p/'report_numbers.json',numbers)

def show(label,data):
    print('\n'+label+'\n'+data.to_string(index=False))

show('GROUPS',pd.DataFrame(numbers['groups']))
show('POOLED SUCCESS',pooled.query('endpoint == "final_success" and stage in ["event","interior","grasp_window","carry","place"]'))
show('POOLED INTERACTIONS',interaction.query('arm == "equal_cell_pool"'))
show('POINTWISE INTERACTION SIGNALS',interaction.query('lo > 0 or hi < 0'))
show('CT-CU',pd.read_csv(p/'paired_CT_CU.csv'))
t=pd.DataFrame(numbers['triggers'])
show('TRIGGERS',t.groupby(['variant','trigger'])[['available','onset_episode_denominator','hit','actionable_hit',
     'no_onset_episode_denominator','false_alert','onset_time_unavailable']].sum().reset_index())
show('COST BY STAGE',costs[costs.stage.ne('__arm__')].groupby(['variant','size','stage'])[['N','V','M','owner_work','active_controls']].sum().reset_index())
show('R7 SUPPORT',support.query('family == "r7_stage"').groupby(['variant','size','stage'])[['anchors','supported','p0','p1']].sum().reset_index())
show('FIRST APPROACH',pd.DataFrame(firsts))
if (p/'look_summary.csv').exists():
    look=pd.read_csv(p/'look_summary.csv')
    show('LOOK',look[look.stage.eq('__all__')][['arm','age','decisions','episodes','tail_motion_rms','fresh_motion_rms',
         'motion_improvement','motion_improvement_lo','motion_improvement_hi','motion_mse_improvement',
         'tail_look_motion_rms','tail_look_gripper_flip','gripper_mse_share']])
print('\nAUDIT',json.dumps(json.loads((p/'audit.json').read_text()) | {'extras':'see report_numbers.json'},default=str))
