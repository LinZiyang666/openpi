"""Final immutable candidate ops run, serially after the first regression batch."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path.cwd();HERE=ROOT/'exp/offline_search/rounds/r17/autoscale_1017/followup_1018'
while True:
    try:records=json.loads((HERE/'REGRESSION_COMMANDS.json').read_text())
    except (OSError,ValueError):records=[]
    if len(records)==15:break
    time.sleep(5)
stem=HERE/'ops_after_final'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',AUTOSCALE_TEST_TIMEOUT='180',
    AUTOSCALE_SOURCE_VARIANT=str(HERE/'candidate'),PYTHONPATH=str(HERE/'hook')+':'+str(ROOT/'src')+':'+str(ROOT))
cmd=[str(ROOT/'.venv/bin/python'),'-m','pytest','-p','exp.offline_search.rounds.r17.autoscale_1017.cpu_only',
    'exp/r17_integration/tests/ops','-vv','--tb=short','--basetemp='+str(stem)+'_tmp','--junitxml='+str(stem)+'.xml',
    '-o','cache_dir='+str(stem)+'_cache']
manifest=json.loads((HERE/'BEFORE_SOURCES.json').read_text())
pins={n:hashlib.sha256((HERE/'candidate'/Path(n).name).read_bytes()).hexdigest() for n in manifest}
pins['fairness.py']=hashlib.sha256((HERE.parent/'fairness.py').read_bytes()).hexdigest()
started=time.time();print('final ops started',flush=True)
with stem.with_suffix('.log').open('w') as out:result=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT)
(HERE/'OPS_FINAL_COMMAND.json').write_text(json.dumps(dict(command=cmd,source_sha256=pins,code=result.returncode,seconds=time.time()-started),indent=2)+'\n')
print('final ops exit',result.returncode,flush=True)
