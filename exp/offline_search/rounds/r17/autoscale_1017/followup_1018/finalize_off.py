"""Finish evidence only; never enables autoscaling or prepares/launches a run."""
import datetime, hashlib, json, os, subprocess, time
from pathlib import Path
from exp.offline_search.rounds.r17.autoscale_1017.regression_report import cases
ROOT=Path.cwd();BASE=ROOT/'exp/offline_search/rounds/r17/autoscale_1017';HERE=BASE/'followup_1018';OUT=HERE/'off_final'
while 'FINAL CPU verification complete' not in (HERE/'finish_off.log').read_text():
    if not Path('/proc/1207666').exists():raise SystemExit('verification stopped; inspect finish_off.log; switch remains OFF')
    time.sleep(5)
assert (BASE/'ENABLED').read_text().strip()=='off'
comparison=json.loads((OUT/'BEFORE_AFTER.json').read_text());assert comparison['status']=='PASS'
own=cases(OUT/'own_all.xml');assert own and all(v['status']=='passed' for v in own.values())
units=cases(OUT/'longseg_unit.xml');assert units and all(v['status']=='passed' for v in units.values())
# The preceding process loaded an obsolete --output-dir CLI literal before it
# was corrected on disk. Retain that log and run the actual --out interface.
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='.:src')
for name in ('AUTOSCALE_BASELINE_ROOT','AUTOSCALE_SOURCE_VARIANT','AUTOSCALE_GENERATION_VARIANT'):env.pop(name,None)
# Reuse the same command/socket guard, including in the standalone checker.
wrapper="from types import SimpleNamespace as S; from exp.offline_search.rounds.r17.autoscale_1017.cpu_only import pytest_sessionstart; pytest_sessionstart(S(config=S(option=S()))); import runpy; runpy.run_module('exp.offline_search.rounds.r17.longseg_1017.check',run_name='__main__')"
cmd=[str(ROOT/'.venv/bin/python'),'-c',wrapper,'--local-only','--out',str(OUT/'longseg_check/CPU_PROOF.json')]
started=time.time();print('corrected local-only sol checker started',flush=True)
with (OUT/'longseg_check_corrected.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
(OUT/'LONGSEG_CORRECTED_COMMAND.json').write_text(json.dumps(dict(command=cmd,code=result.returncode,seconds=time.time()-started),indent=2)+'\n')
if result.returncode:raise SystemExit('sol checker failed; global OFF; inspect corrected log')
proof=json.loads((OUT/'longseg_check/CPU_PROOF.json').read_text())
extra=['exp/r17_integration/ops/build_client_bundle.py','exp/r17_integration/qualification/deploy_cpu.py',
    'exp/offline_search/rounds/r16/conductor_runner.py','exp/offline_search/rounds/r16/test_conductor.py',
    'exp/offline_search/rounds/r17/filter_failed/stage.py','exp/offline_search/rounds/r17/filter_failed/resume.py',
    'exp/offline_search/rounds/r17/filter_failed/launch.py','exp/offline_search/rounds/r17/filter_failed_knob/launch.py',
    'exp/offline_search/rounds/r17/filter_failed_knob/formal_a_queue.py','exp/offline_search/rounds/r17/filter_failed_knob/vla_workers.py',
    'exp/offline_search/rounds/r17/longseg_1017/check.py','exp/r17_integration/ops/RUN_CONFIGS.md']
files=set(json.loads((BASE/'PATCH_BASE.json').read_text()))|set(extra)
files|={str(p.relative_to(ROOT)) for p in BASE.glob('*.py')}
files|={str(p.relative_to(ROOT)) for p in (BASE/'tests').glob('*.py')}
files.add(str((BASE/'ENABLED').relative_to(ROOT)))
hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(files)}
cache=[]
for suite in comparison['suites']:
    for name,row in suite['cases'].items():
        if 'test_unchanged_single_L_reuses_existing_sealed_artifacts' in name:cache.append(dict(test=name,**row))
assert cache and all(v['before']['status']==v['after']['status']=='passed' for v in cache)
# Verify scientific dependencies were the same on both sides of the ops run.
a=json.loads((HERE/'off_regression/ops_before.sources.json').read_text())['hashes']
b=json.loads((HERE/'off_regression/ops_after.sources.json').read_text())['hashes']
deps=['exp/r17_integration/common/serving.py','exp/r17_integration/ops/cache_study.py',
      'exp/r17_integration/ops/cache_knob.py','exp/r17_integration/ops/library_campaign.py']
assert all(a[name]==b[name] for name in deps)
assert (BASE/'ENABLED').read_text().strip()=='off'
value=json.loads((BASE/'VERIFICATION.json').read_text())
value.update(status='CPU_PASS_WAITING_OWNER_LIVE_TRIALS',global_switch='off',new_campaign_default='native_fixed',
    updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),dynamic_activated=False,live_acceptance='NOT_RUN',
    own_tests=f'{len(own)} passed, 0 failed; explicit CPU candidate/OFF coverage',
    new_cpu_tests=dict(passed=len(own),failed=0,report='followup_1018/off_final/own_all.xml'),
    longseg_unit_tests=len(units),longseg_check_code=0,longseg_check_status=proof['status'],
    longseg_check=dict(items=len(proof['rows']),statuses={s:sum(r['status']==s for r in proof['rows']) for s in sorted({r['status'] for r in proof['rows']})},
                      report='followup_1018/off_final/longseg_check/CPU_PROOF.json',remote_calls=0),
    source_sha256=hashes,cache_lane_hang_check=cache,scientific_dependency_hashes_equal=True,
    cpu_guard_exceptions=dict(deselected_real_rsync_daemon=1,blocked_production_bind_cases=3,
        note='H100 detached listener and two FastWAM MPS fixtures are blocked on both sides; not counted as passed'),
    trials_document='TRIAL.md',reset_document='followup_1018/URGENT_FIX.md')
(BASE/'VERIFICATION.json').write_text(json.dumps(value,indent=2)+'\n')
print('FINAL VERIFIED: global OFF; own',len(own),'sol',len(proof['rows']),'new regressions 0',flush=True)
