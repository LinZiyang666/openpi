"""Publish evidence only after the final candidate and live post-apply checks pass."""
import datetime,hashlib,json
from pathlib import Path
from exp.offline_search.rounds.r17.autoscale_1017.regression_report import cases
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[4]
comparison=json.loads((HERE/'BEFORE_AFTER.json').read_text());assert comparison['status']=='PASS'
live=cases(HERE/'POST_APPLY.xml');assert live and all(v['status']=='passed' for v in live.values())
longseg=cases(HERE/'LONGSEG_UNIT.xml');assert longseg and all(v['status']=='passed' for v in longseg.values())
activation=json.loads((HERE/'ACTIVATION.json').read_text());assert activation['state']=='ACTIVE'
for name,pin in activation['candidate_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin
assert hashlib.sha256((BASE/'fairness.py').read_bytes()).hexdigest()==activation['new_modules']['fairness.py']
assert hashlib.sha256((ROOT/'src/openpi/conductor/worker.py').read_bytes()).hexdigest()==activation['generation_fix']['after_sha256']
names=('fa_co_f_f_oc1','fa_ls_fw_r_p_m','fa_ls_gr_r_p_m','fa_pi_r_f_l','fa_gr_r_u_l','fa_ls_dp_r_b_l','fa_ls_dp_r_u_m')
rows=[json.loads((BASE/'new_chain_dry_run'/(n+'.json')).read_text()) for n in names]
assert all(r['status']=='PASS' and r['remote_calls']==r['launches']==0 for r in rows)
v=json.loads((HERE/'VERIFICATION_BEFORE.json').read_text())
v['historical_default_on_verification']='followup_1018/VERIFICATION_BEFORE.json'
v.update(status='PASS',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    baseline_method=comparison['baseline'],baseline_dependency_snapshot='followup_1018/BEFORE_SOURCES.json',
    before_after_report='followup_1018/BEFORE_AFTER.json',before_after_status='PASS',
    before_after_table=[{k:r[k] for k in ('suite','before','after','regressions','missing')} for r in comparison['suites']],
    new_cpu_tests=dict(passed=len(live),failed=0,post_apply_imports=True,report='followup_1018/POST_APPLY.xml'),
    desired_maxima_policy='physical-v2: promote old small fixed counts; retain larger maxima; smaller technical caps require an explicit reason',
    source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in v['source_sha256'] if (ROOT/n).is_file()})
v['source_sha256'][str((BASE/'fairness.py').relative_to(ROOT))]=hashlib.sha256((BASE/'fairness.py').read_bytes()).hexdigest()
v['replacement_chains'].pop('remote_calls_this_followup',None)
v['replacement_chains'].update(remote_calls_in_prepare_dry_runs=0,prepare_dry_runs=rows,autoscale_tests_passed=len(live),prepare_tests_passed=9,sol_unit_tests_passed=len(longseg))
v['followup_1018']=dict(activated=activation['activated_utc'],runtime_modules_changed=10,new_module='fairness.py',
    fairness_tests=sum('test_fairness' in k for k in live),all_new_tests=len(live),
    incident_report='followup_1018/incident/INCIDENT.md',incident_record='followup_1018/incident/INCIDENT.json',
    remote_calls=2,remote_scope='coordinator-authorized read-only incident ls/stat/sha256 on timan107 only',
    formal_preparations=0,launches=0,incident_files_deleted=0,formal_files_deleted=0,queue_audit='followup_1018/QUEUE_AUDIT.json',
    smaller_technical_caps_in_actual_queue=[],fixed_continuation=dict(name='fa_co_f_f_oc1',execution_at_dry_run=next(r.get('desired_maxima') for r in rows if r['name']=='fa_co_f_f_oc1'),reason='frozen source pins'),
    comparison='followup_1018/BEFORE_AFTER.md',post_apply='followup_1018/POST_APPLY.xml',
    fairness_contract='equal dominant-resource admission budgets; capped by lane demand, unused budgets borrowed, returns only at episode boundaries; per-card integer admission still applies')
hang='test_unchanged_single_L_reuses_existing_sealed_artifacts'
v['reported_hang_case']={side:next(c for k,c in cases(HERE/path).items() if hang in k)
    for side,path in [('before','ops_before.xml'),('after','combined_regression/ops_after.xml')]}
v['generation_identity_fix']=dict(source='src/openpi/conductor/worker.py',reproduction='followup_1018/GENERATION_BEFORE.xml',passing='followup_1018/GENERATION_AFTER.xml',driver_validator_unchanged=True,slot_retirement_preserved=True,R15_same_pid_failure=False,R16_native_libero_probe_has_pid=False,reset_command_module='exp.offline_search.rounds.r17.autoscale_1017.reset_zero',formal_reset_executed=False)
v['restriction_events'][0].update(details='followup_1018/incident/INCIDENT.json',remote_leftovers=1,bytes=409600,node='timan107')
(BASE/'VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\n')
p=BASE/'DEFAULT_ON.md';s=p.read_text();s+='\n本次后续验证：直接导入最终共享树的CPU测试 '+str(len(live))+' 项通过，其中公平共享 '+str(v['followup_1018']['fairness_tests'])+' 项；sol单元测试 '+str(len(longseg))+' 项通过。最终源码的8组前后比较无新增回归、无超时。源码、命令、逐例状态见 followup_1018/BEFORE_AFTER.md 和 VERIFICATION.json。\n';p.write_text(s)
p=BASE/'DESIGN.md';s=p.read_text();s+='\n本次后续验证已完成：'+str(len(live))+'项post-apply CPU测试通过（共享公平性'+str(v['followup_1018']['fairness_tests'])+'项），8组既有套件前后比较无新增回归、无超时；详细表见followup_1018/BEFORE_AFTER.md。没有正式prepare或launch；此次仅进行获准的两次事故只读远端取证。\n';p.write_text(s)
print('PASS',len(live),'post-apply tests;',len(longseg),'longseg unit tests; no regression')
