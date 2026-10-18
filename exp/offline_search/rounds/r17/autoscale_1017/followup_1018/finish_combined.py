raise SystemExit("OWNER HOLD: no automatic activation; two live validation lanes must pass")
"""Publish only after the complete combined regression and candidate tests pass."""
import datetime,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[4]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',AUTOSCALE_TEST_TIMEOUT='180',PYTHONPATH='.:src')
env.pop('AUTOSCALE_SOURCE_VARIANT',None);env.pop('AUTOSCALE_BASELINE_ROOT',None)
python=str(ROOT/'.venv/bin/python');records=[]
def run(name,cmd,extra=None):
    print(name,'started',flush=True);started=time.time()
    with (HERE/(name+'.log')).open('w') as out:r=subprocess.run(cmd,cwd=ROOT,env=env|dict(extra or {}),stdout=out,stderr=subprocess.STDOUT)
    records.append(dict(name=name,command=cmd,code=r.returncode,seconds=time.time()-started))
    (HERE/'FINISH_COMBINED_COMMANDS.json').write_text(json.dumps(records,indent=2)+'\n')
    print(name,'exit',r.returncode,flush=True)
    if r.returncode:raise SystemExit(r.returncode)
def tests(name,path,xml,extra=None):
    run(name,[python,'-m','pytest','-p','exp.offline_search.rounds.r17.autoscale_1017.cpu_only',str(path),
        '-q','--tb=short','--basetemp='+str(HERE/(name+'_tmp')),'--junitxml='+str(HERE/xml),
        '-o','cache_dir='+str(HERE/(name+'_cache'))],extra)
while True:
    try:done=len(json.loads((HERE/'combined_regression/COMMANDS.json').read_text()))==8
    except (OSError,ValueError):done=False
    if done:break
    time.sleep(5)
run('comparison_combined',[python,str(HERE/'report_combined.py')])
assert json.loads((HERE/'BEFORE_AFTER.json').read_text())['status']=='PASS'
tests('combined_candidate',BASE/'tests','COMBINED_CANDIDATE.xml',dict(
    AUTOSCALE_SOURCE_VARIANT=str(HERE/'candidate'),AUTOSCALE_GENERATION_VARIANT='generation_candidate',
    PYTHONPATH=str(HERE/'generation_hook')+':'+str(ROOT/'src')+':'+str(ROOT)))
hold=HERE/'PUBLICATION_HOLD'
hold.rename(HERE/('PUBLICATION_HOLD.verified_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
run('activation_combined',[python,str(HERE/'activate.py')])
tests('postapply_combined',BASE/'tests','POST_APPLY.xml')
tests('longseg_unit_combined','exp/offline_search/rounds/r17/longseg_1017','LONGSEG_UNIT.xml')
run('verification_combined',[python,str(HERE/'update_verification.py')])
print('FINISHED',flush=True)
