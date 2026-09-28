"""Small derived tables and a final read-only completion/footprint snapshot."""
from pathlib import Path
import json,datetime
O=Path(__file__).parent;R=Path('/home/weiland/trace_runs/os_closed_loop')
ss=json.loads((O/'log_summary.json').read_text());chunks=json.loads((O/'chunks_summary.json').read_text())
out={'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'horizon':[],'mixed':[],'footprints':[],'current_completion':[]}
for x in ss:
    if x['run']=='r04_cost':
        es=json.loads((O/f"episodes_{x['arm']}.json").read_text());f=sum(e['within_gripper_flips'] for e in es);n=sum(e['within_gripper_opportunities'] for e in es)
        out['horizon'].append(dict(arm=x['arm'],success=x['success'],tasks=[t['success'] for t in x['per_task']],
                                    within_head_gripper_flips=f,opportunities=n,within_head_grip_flip_rate=f/n,
                                    failed_repeated_head_at_least5=sum(e['success']==0 and e['longest_head_repeat_requests']>=5 for e in es)))
    if 'tail1ug' in x['arm']:
        q=x['M']/x['V'];out['mixed'].append(dict(arm=x['arm'],N=x['N'],V=x['V'],M=x['M'],q=q,IR=x['ir_owner_fullmiss'],
                                              same_anchor_miss_rate_full10_IR=.076+.424*q,
                                              blocked_policy_tail_fraction=x['policytail']['span_positive']/x['policytail']['post_miss_total']))
for p in sorted((R/'r04_k7/fits').glob('*tail1ug.pkl')):out['footprints'].append(dict(path=str(p),bytes=p.stat().st_size))
for root in sorted(R.glob('r04_*')):
    ap=root/'arms.json'
    if not ap.exists():continue
    for a in json.loads(ap.read_text()):
        name=a.get('arm',a.get('name'));jp=root/'runs'/name/'client/journal.jsonl'
        if not jp.exists():continue
        accepted={}
        for l in jp.open():
            try:d=json.loads(l)
            except ValueError:continue
            if d.get('accepted') and d.get('status') in ('done','failed') and not d.get('error'):accepted[d['task_uid']]=d
        out['current_completion'].append(dict(run=root.name,arm=name,N=len(accepted),success=sum(d.get('success',False) for d in accepted.values())))
out['task_names']=json.loads(Path('/home/weiland/trace_runs/offline_search_store/library/pi05_l10/current/manifest.json').read_text())['tasks']
out['cadence_cost_audit']=[]
for c in json.loads((O/'cadence_audit.json').read_text()):
    x=next(x for x in ss if x['arm']==c['arm']);n=x['N'];base=x['projections']['policy_commit']['ir']
    out['cadence_cost_audit'].append(dict(arm=x['arm'],
        fixed_path_restored_vision_IR=base+c['orphan_vision_cost']/n,
        fixed_path_all_restored_looks_miss_IR=base+(c['orphan_vision_cost']+c['orphan_extra_miss_cost_upper'])/n,
        inspection_vision_only_increment=.152*c['bridge']['tail_checkpoint_event_or_residual']/n,
        calibrated_bridge_eligible_fraction=c['bridge']['calibrated_join_and_nonterminal_eligible']/c['bridge']['anchors_with_observed_tail_and_followup']))
out['gripper_cancellation_by_task']={str(t):{x['model']+'_'+x['suite']:{k:x['per_task'][str(t)][k] for k in ('planned_grip_switch','planned_switch_not_in_next_head')} for x in chunks if x['lib']=='current'} for t in range(10)}
(O/'review_tables.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ('current_completion','gripper_cancellation_by_task')},indent=2))
print('Completed new tail/mixed arms:',[x for x in out['current_completion'] if x['N']==500 and any(s in x['arm'] for s in ('k10','tail','_g_'))])
