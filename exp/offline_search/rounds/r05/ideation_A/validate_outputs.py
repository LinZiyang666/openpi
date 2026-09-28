"""Consistency checks for this analysis, not tests of proposed controllers."""
from pathlib import Path
import json,math,ast
O=Path(__file__).parent
def read(n):return json.loads((O/n).read_text())
ss=read('log_summary.json');pairs=read('horizon_pairs.json');chunks=read('chunks_summary.json');fits=read('dynamics_fits.json');audit=read('cadence_audit.json');arms=read('ARM_SPECS.json')['arms']
assert len(ss)==17 and len(pairs)==4 and len(chunks)==len(fits)==8 and len(audit)==3
for s in ss:
    assert s['episodes']==500 and sum(t['n'] for t in s['per_task'])==500
    assert sum(t['success'] for t in s['per_task'])==s['success']
    assert 0<=s['M']<=s['V']<=s['N']
    assert math.isclose(s['ir_owner_fullmiss'],(.152*s['V']+.848*s['M'])/s['N']*5/s['L'],abs_tol=1e-12)
    assert s['duplicates']==0
    ep=read('episodes_'+s['arm']+'.json')
    assert len({(e['task'],e['init']) for e in ep})==500
for p in pairs:
    assert p['n']==500
    assert p['SF']==sum(x['SF'] for x in p['per_task'].values())
    assert p['FS']==sum(x['FS'] for x in p['per_task'].values())
    assert math.isclose(p['delta'],(p['FS']-p['SF'])/500,abs_tol=1e-12)
assert next(p for p in pairs if p['reference']=='r4f_p_l10_inf_s1001')['FS']==52
assert next(p for p in pairs if p['reference']=='r4f_p_l10_inf_s1001')['SF']==24
for c in chunks:
    assert c['query_episodes']==500 and len(c['sigma'])==7 and len(c['state_scale'])==8
    assert c['queries']['edges']==sum(t['edges'] for t in c['per_task'].values())
    assert c['queries']['planned_switch_not_in_next_head']<=c['queries']['planned_grip_switch']
for f in fits:
    assert len(f['B'])==16 and all(len(row)==3 for row in f['B'])
    assert f['fit_bytes_float32']==208 and f['q99']>0 and f['edges']<f['rows']
    assert all(math.isfinite(v) for row in f['B'] for v in row)
for c in audit:
    assert max(c['reconstruction_max_head'],c['reconstruction_max_tail'])<3e-7
    b=c['bridge'];assert b['calibrated_eligible_next_miss']<=b['calibrated_join_and_nonterminal_eligible']<=b['all16_next2_valid']
    assert b['tail_checkpoint_event_or_residual']>=b['tail_checkpoint_grip_event']
assert len(arms)==len({a['arm'] for a in arms})==58
for a in arms:
    assert a['full_model'] and a['replan_steps'] in (5,10,15)
    if 'horizon_controller:' in a['method']:assert a['requires_implementation']
    bk=a['kwargs'].get('base_kwargs',{})
    if bk:assert bk['fit_data']=='same' and bk['lib']==('current' if a['library_episodes']==50 else 'big')
for p in O.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
too_large=[p.name for p in O.iterdir() if p.is_file() and p.stat().st_size>50_000_000]
assert not too_large,too_large
result=dict(status='PASS',log_arms=len(ss),paired_comparisons=len(pairs),library_diagnostic_cells=len(chunks),
            independent_library_only_fits=len(fits),closed_loop_tail_audits=len(audit),review_recipes=len(arms),
            production_controllers_implemented=False,closed_loop_runs_started=0,large_arrays_written=0)
(O/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
