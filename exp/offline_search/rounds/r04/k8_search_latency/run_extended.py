import os;os.sched_setaffinity(0,{34})
from common import *
import subprocess
prefix=['taskset','-c','34-37,78-81','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',str(ROOT/'.venv/bin/python')]
commands=[]
def run(args):
 cmd=prefix+args;commands.append(cmd);dump(OUT/'extended_commands.json',commands);print('START',args,flush=True)
 p=subprocess.run(cmd,cwd=ROOT)
 if p.returncode:print('ERROR',p.returncode,args,flush=True)
for rep in (1,2):
 run([str(OUT/'extended.py'),'scaling','--run',str(rep)])
 run([str(OUT/'extended.py'),'aux','--run',str(rep)])
 for model in ('pi05','groot'):
  for suite in ('spatial','l10'):run([str(OUT/'native.py'),'--model',model,'--suite',suite,'--run',str(rep)])
 for config in ('pi05_l10_50_AWM','pi05_l10_500_AWM','pi05_l10_50_MixedJudge','pi05_l10_500_MixedJudge','pi05_l10_500_BlindMixedJudge','groot_l10_50_AWM','groot_l10_500_AWM'):
  run([str(OUT/'extended.py'),'concurrency','--config',config,'--run',str(rep)])
