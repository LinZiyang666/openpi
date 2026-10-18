"""Serial final verification only. NEVER activates the global switch."""
import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
from exp.offline_search.rounds.r17.autoscale_1017.regression_run import SUITES,EXCLUDE
from exp.offline_search.rounds.r17.autoscale_1017.regression_report import report,cases
ROOT=Path.cwd();BASE=ROOT/'exp/offline_search/rounds/r17/autoscale_1017';HERE=BASE/'followup_1018';OUT=HERE/'off_final'
OUT.mkdir(exist_ok=True)
assert (BASE/'ENABLED').read_text().strip()=='off'
# Never overlap the large ops fixture reads with real queue-plan reads.
while not (HERE/'off_regression/ops_after.xml').exists():time.sleep(5)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',AUTOSCALE_TEST_TIMEOUT='180',PYTHONPATH='.:src')
for n in ('AUTOSCALE_SOURCE_VARIANT','AUTOSCALE_BASELINE_ROOT','AUTOSCALE_GENERATION_VARIANT'):env.pop(n,None)
python=str(ROOT/'.venv/bin/python');records=[]
def tests(name,paths):
    stem=OUT/name
    cmd=[python,'-m','pytest','-p','exp.offline_search.rounds.r17.autoscale_1017.cpu_only',*paths,
        '-q','--tb=short','--basetemp='+str(stem)+'_tmp','--junitxml='+str(stem)+'.xml',
        '--deselect='+EXCLUDE,'-o','cache_dir='+str(stem)+'_cache']
    print(name,'started',flush=True);start=time.time()
    with stem.with_suffix('.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(name=name,command=cmd,code=r.returncode,seconds=time.time()-start))
    (OUT/'COMMANDS.json').write_text(json.dumps(records,indent=2)+'\n')
    print(name,'exit',r.returncode,flush=True)
    return stem.with_suffix('.xml')
manifest={'ops':[str(HERE/'off_regression/ops_before.xml'),str(HERE/'off_regression/ops_after.xml')]}
for suite,paths in SUITES.items():
    if suite=='ops':continue
    after=tests(suite+'_after',paths)
    manifest[suite]=[str(HERE/'off_regression'/(suite+'_before.xml')),str(after)]
value=report(manifest)
value['baseline']='19 original SHA-verified prepatch backups, full current dependency tree; current OFF prepares use native payload bytes. timan108 remains disabled. All tests use the same production-port/remote/GPU guard.'
value['sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')}
(OUT/'BEFORE_AFTER.json').write_text(json.dumps(value,indent=2)+'\n')
lines=['全局 OFF 与故障修复：逐例前后对照','', '| 套件 | 前：通过/失败 | 后：通过/失败 | 新回归 | 超时 |','|---|---:|---:|---:|---:|']
for v in value['suites']:
 a,b=v['before'],v['after'];lines.append(f"| {v['suite']} | {a['passed']}/{a['failed']} | {b['passed']}/{b['failed']} | {len(v['regressions'])} | {b['timeouts']} |")
lines+=['','control.py 属于 h100 套件；owner 的 SO_REUSEADDR 三行修复保留。禁止真实绑定23100–23199，相关用例在两侧都被阻止；真实rsync daemon用例两侧排除。不能将这些禁止执行的测试写成通过。',
'','本表逐例比较，不只比较失败总数。原已知失败与新端口防护阻止项均保留。动态模式仍须 owner 两条实跑验收，CPU结果不授权打开开关。']
(OUT/'BEFORE_AFTER.md').write_text('\n'.join(lines)+'\n')
if value['status']!='PASS':raise SystemExit('regressions: inspect off_final/BEFORE_AFTER.json; global remains OFF')
own=tests('own_all',[str(BASE/'tests')]);own_cases=cases(own)
if not own_cases or any(v['status']!='passed' for v in own_cases.values()):raise SystemExit('own tests failed; global remains OFF')
longseg=tests('longseg_unit',['exp/offline_search/rounds/r17/longseg_1017'])
if any(v['status']!='passed' for v in cases(longseg).values()):raise SystemExit('longseg unit failure')
# Existing sol checker: local-only prevents an H100 fallback. No launcher invoked.
cmd=[python,'-m','exp.offline_search.rounds.r17.longseg_1017.check','--local-only','--out',str(OUT/'longseg_check/CPU_PROOF.json')]
with (OUT/'longseg_check.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
records.append(dict(name='longseg_check',command=cmd,code=r.returncode));(OUT/'COMMANDS.json').write_text(json.dumps(records,indent=2)+'\n')
assert (BASE/'ENABLED').read_text().strip()=='off'
current=json.loads((BASE/'VERIFICATION.json').read_text())
current.update(status='CPU_PASS_WAITING_OWNER_LIVE_TRIALS' if r.returncode==0 else 'CHECK_REQUIRED',
    updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_campaign_default='native_fixed',global_switch='off',
    before_after_report=str((OUT/'BEFORE_AFTER.json').relative_to(BASE)),before_after_table=value['suites'],
    new_cpu_tests=dict(passed=len(own_cases),failed=0,post_apply_imports=True,report=str(own.relative_to(BASE))),
    longseg_unit_tests=len(cases(longseg)),longseg_check_code=r.returncode,
    source_sha256=value['sources'],live_acceptance='NOT_RUN',dynamic_activated=False)
(BASE/'VERIFICATION.json').write_text(json.dumps(current,indent=2)+'\n')
print('FINAL CPU verification complete; global OFF; no launches',flush=True)
