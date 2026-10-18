"""Serial fresh before/candidate regression, after the already running ops baseline."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from exp.offline_search.rounds.r17.autoscale_1017.regression_run import SUITES,EXCLUDE
ROOT=Path.cwd();HERE=ROOT/'exp/offline_search/rounds/r17/autoscale_1017/followup_1018'
# The other process emits its XML only when its entire serial suite completes.
while not (HERE/'ops_before.xml').exists():time.sleep(5)
records=[]
jobs=[('ops','candidate')]+[(s,v) for s in SUITES if s!='ops' for v in ('before','candidate')]
for suite,variant in jobs:
    stem=HERE/(suite+'_'+('after' if variant=='candidate' else 'before'))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',AUTOSCALE_TEST_TIMEOUT='180' if suite=='ops' else '60',
        AUTOSCALE_SOURCE_VARIANT=str(HERE/variant),PYTHONPATH=str(HERE/'hook')+':'+str(ROOT/'src')+':'+str(ROOT))
    cmd=[str(ROOT/'.venv/bin/python'),'-m','pytest','-p','exp.offline_search.rounds.r17.autoscale_1017.cpu_only',
        *SUITES[suite],'-vv','--tb=short','--basetemp='+str(stem)+'_tmp','--junitxml='+str(stem)+'.xml',
        '--deselect='+EXCLUDE,'-o','cache_dir='+str(stem)+'_cache']
    manifest=json.loads((HERE/'BEFORE_SOURCES.json').read_text())
    pins={n:hashlib.sha256((HERE/variant/Path(n).name).read_bytes()).hexdigest() for n in manifest}
    if variant=='candidate':pins['fairness.py']=hashlib.sha256((HERE.parent/'fairness.py').read_bytes()).hexdigest()
    started=time.time();print(suite,variant,'start',flush=True)
    with stem.with_suffix('.log').open('w') as out:result=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT)
    records.append(dict(suite=suite,variant=variant,command=cmd,source_sha256=pins,code=result.returncode,seconds=time.time()-started))
    (HERE/'REGRESSION_COMMANDS.json').write_text(json.dumps(records,indent=2)+'\n')
    print(suite,variant,'exit',result.returncode,flush=True)
