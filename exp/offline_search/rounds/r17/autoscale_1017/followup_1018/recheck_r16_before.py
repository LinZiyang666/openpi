"""Recheck native/dynamic lifecycle assertions with the verified baseline."""
import json,os,subprocess,time
from pathlib import Path
root=Path.cwd();base=root/'exp/offline_search/rounds/r17/autoscale_1017';out=base/'followup_1018/off_regression'
env=dict(os.environ,PYTHONPATH=str(base/'baseline_hook')+':'+str(root/'src')+':'+str(root),AUTOSCALE_BASELINE_ROOT=str(root),
         CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',AUTOSCALE_TEST_TIMEOUT='180')
stem=out/'r16_before_modes'
cmd=[str(root/'.venv/bin/python'),'-m','pytest','-p','exp.offline_search.rounds.r17.autoscale_1017.cpu_only','exp/offline_search/rounds/r16','-q','--tb=short','--basetemp='+str(stem)+'_tmp','--junitxml='+str(stem)+'.xml','-o','cache_dir='+str(stem)+'_cache']
start=time.time()
with stem.with_suffix('.log').open('w') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
# Preserve the first reports. The final suite table compares the same test matrix.
from exp.offline_search.rounds.r17.autoscale_1017.regression_report import cases
observed=cases(stem.with_suffix('.xml'))
assert len(observed)>=220
for suffix in ('.xml','.log','.sources.json'):
 old=out/('r16_before'+suffix);old.rename(out/('r16_before_before_mode_fixture'+suffix))
 (out/('r16_before_modes'+suffix)).rename(old)
(out/'R16_MODE_RECHECK.json').write_text(json.dumps(dict(command=cmd,code=result.returncode,seconds=time.time()-start,
    reason='Exercise both explicit ON and OFF after incident rollback; no real process/remote/port operations',
    old_reports='r16_before_before_mode_fixture.*'),indent=2)+'\n')
print('baseline mode matrix complete',result.returncode,flush=True)
