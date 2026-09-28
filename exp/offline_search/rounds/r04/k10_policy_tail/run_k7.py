import json
from pathlib import Path
import subprocess
import sys
BASE=Path(__file__).resolve().parent
PREFIX=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',sys.executable]
out=BASE/'results/installed';out.mkdir(parents=True,exist_ok=True)
commands=[]
for r in json.loads((BASE.parent/'k7_guard/arms_k7.json').read_text()):
    name=r['name'];suite=r['suite']
    commands.append((name,PREFIX+[str(BASE/'launch_test.py'),'installed','normal','--blind','--cell',f'pi05_{suite}_cache','--yaml',f"exp/trace_dual/config/tr_pi05_{'sp' if suite=='spatial' else suite}_cache.yaml",'--root','/home/weiland/trace_runs/offline_search_store','--method',r['method'],'--kwargs',json.dumps(r['kwargs']),'--fit-artifact',f'/tmp/k7_guard_fits/{name}.pkl','--judge','guard_only','--out',f'/tmp/k10_installed/k7/{name}']))
commands.append(('edges',PREFIX+[str(BASE/'k7_unit.py'),'edges']))
for mode in ('parity','rates'):
    for key in ('pi05_l10','pi05_spatial'):
        for scale in (50,500):commands.append((f'{mode}_{key}_{scale}',PREFIX+[str(BASE/'k7_unit.py'),mode,key,str(scale)]))
report=[]
for name,cmd in commands:
    with (out/f'k7_{name}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    report.append(dict(name=name,returncode=r.returncode,command=cmd))
    (out/'k7_commands.json').write_text(json.dumps(report,indent=2))
    assert r.returncode==0,name
    print(name,'PASS',flush=True)
