import os;os.sched_setaffinity(0,{34})
from common import *
import subprocess
prefix=['taskset','-c','34-37,78-81','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',str(ROOT/'.venv/bin/python')]
commands=[]
for run in (1,2):
 for c in json.loads((OUT/'configs.json').read_text()):
    cmd=prefix+[str(OUT/'benchmark.py'),'--config',c['id'],'--run',str(run),'--profile']
    commands.append(cmd);dump(OUT/'benchmark_commands.json',commands)
    print('START',c['id'],run,flush=True)
    r=subprocess.run(cmd,cwd=ROOT)
    if r.returncode:print('ERROR',r.returncode,c['id'],flush=True)
