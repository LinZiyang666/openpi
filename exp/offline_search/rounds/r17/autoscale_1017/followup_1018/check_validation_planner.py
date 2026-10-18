"""Bounded CPU check of the actual source planner, no archive copying or staging."""
from pathlib import Path
import json,os,subprocess
from unittest.mock import patch
from exp.offline_search.rounds.r17.autoscale_1017 import trial_lane as t
source=Path('/data/mw_finetune/eval/formal_fastwam_mw_mt50_500_20261009_223917_0b845f00')
root=t.BASE/'followup_1018/validation_plan/fa_asv_cpu_plan'
copy=t.shutil.copy2
# Runtime tar bytes are not needed by plan(); use the existing frozen runtime.
def small_copy(a,b):
    if Path(a).name in ('runtime.tar','workers.tar'):return Path(b).write_bytes(b'not used by planner')
    return copy(a,b)
if not root.exists():
    with patch.object(t.shutil,'copy2',small_copy):t.build(source,root,'r17')
c=t.read(root/'campaign.json');runtime=c['runtime']
if not Path(runtime).exists():
    import tarfile
    mirror=root.parent/'frozen_runtime'
    if not mirror.exists():
        mirror.mkdir()
        with tarfile.open(source/'runtime.tar') as archive:
            for member in archive.getmembers():
                if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:raise ValueError('unsafe source archive')
                target=mirror/member.name;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,target.open('wb') as out:t.shutil.copyfileobj(src,out)
    runtime=str(mirror)
c['autoscale_validation']=True;t.write(root/'campaign.json',c)
(Path(runtime)/'exp/r17_integration/ops/formal_driver.py').write_bytes(t.planner_overlay(source))
code='import json;from pathlib import Path;from dataclasses import asdict;from exp.r17_integration.ops.formal_driver import plan;c=json.loads(Path('+repr(str(root/'campaign.json'))+').read_text());print(json.dumps([asdict(t) for t in plan(c)[2]]))'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=runtime+':'+runtime+'/src:'+str(t.REPO))
r=subprocess.run([str(t.REPO/'.venv/bin/python'),'-c',code],cwd=runtime,env=env,text=True,capture_output=True,timeout=60)
(root.parent/'planner.stderr').write_text(r.stderr)
if r.returncode:raise RuntimeError(r.stderr[-4000:])
actual=json.loads(r.stdout);expected=t.read(root/'EPISODE_PLAN.json')['tasks']
result=dict(status='PASS' if actual==expected else 'FAIL',actual=len(actual),expected=len(expected),full_task_dicts_identical=actual==expected,source=str(source),remote_calls=0,launches=0)
t.write(root.parent/'PLANNER.json',result);print(json.dumps(result))
if actual!=expected:
 t.write(root.parent/'actual_tasks.json',actual)
 raise AssertionError('frozen validation plan changed')
