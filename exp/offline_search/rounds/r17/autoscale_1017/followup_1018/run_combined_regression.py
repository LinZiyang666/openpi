"""Final 8-suite after run: fairness/maxima plus distinct generation identity."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from exp.offline_search.rounds.r17.autoscale_1017.regression_run import SUITES,EXCLUDE
ROOT=Path.cwd();HERE=ROOT/'exp/offline_search/rounds/r17/autoscale_1017/followup_1018'
OUT=HERE/'combined_regression';OUT.mkdir(exist_ok=True)
records=[]
for suite,paths in SUITES.items():
    stem=OUT/(suite+'_after')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        AUTOSCALE_TEST_TIMEOUT='180' if suite=='ops' else '60',
        AUTOSCALE_SOURCE_VARIANT=str(HERE/'candidate'),AUTOSCALE_GENERATION_VARIANT='generation_candidate',
        PYTHONPATH=str(HERE/'generation_hook')+':'+str(ROOT/'src')+':'+str(ROOT))
    cmd=[str(ROOT/'.venv/bin/python'),'-m','pytest','-p','exp.offline_search.rounds.r17.autoscale_1017.cpu_only',
        *paths,'-vv','--tb=short','--basetemp='+str(stem)+'_tmp','--junitxml='+str(stem)+'.xml',
        '--deselect='+EXCLUDE,'-o','cache_dir='+str(stem)+'_cache']
    pins={n:hashlib.sha256((HERE/'candidate'/Path(n).name).read_bytes()).hexdigest()
          for n in json.loads((HERE/'BEFORE_SOURCES.json').read_text())}
    pins['fairness.py']=hashlib.sha256((HERE.parent/'fairness.py').read_bytes()).hexdigest()
    pins['src/openpi/conductor/worker.py']=hashlib.sha256((HERE/'generation_candidate/worker.py').read_bytes()).hexdigest()
    started=time.time();print(suite,'started',flush=True)
    with stem.with_suffix('.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(suite=suite,command=cmd,source_sha256=pins,code=r.returncode,seconds=time.time()-started))
    (OUT/'COMMANDS.json').write_text(json.dumps(records,indent=2)+'\n')
    print(suite,'exit',r.returncode,flush=True)
