"""Supplement for K7 artifacts created during the main experiment."""
import os;os.sched_setaffinity(0,{34})
from common import *
import subprocess
cs=json.loads((OUT/'configs.json').read_text());late=json.loads((OUT/'late_fit_inventory.json').read_text());ids={c['id'] for c in cs}
for r in late:
 r.update(id='K7_'+pathlib.Path(r['path']).stem,family='K7 '+pathlib.Path(r['path']).stem.split('_')[-1],extract_base=False,supplemental=True)
 if r['id'] not in ids:cs.append(r)
dump(OUT/'configs.json',cs)
prefix=['taskset','-c','34-37,78-81','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',str(ROOT/'.venv/bin/python')]
commands=[]
for rep in (1,2):
 for r in late:
  cmd=prefix+[str(OUT/'benchmark.py'),'--config',r['id'],'--run',str(rep),'--profile'];commands.append(cmd);dump(OUT/'late_commands.json',commands)
  print('START',r['id'],rep,flush=True);p=subprocess.run(cmd,cwd=ROOT);assert p.returncode==0
