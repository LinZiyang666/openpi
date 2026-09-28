"""CPU-only serial scheduler. Admission failures are retained and retried safely."""
import os,json,pathlib,subprocess,time,sys
assert set(os.sched_getaffinity(0)) <= {26,27,28,29,70,71,72,73}
HERE=pathlib.Path(__file__).resolve().parent
prefix=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src','TMPDIR=/tmp/k9_scratch','JAX_PLATFORMS=cpu',
        'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','TORCHINDUCTOR_CACHE_DIR=/tmp/k9_scratch/inductor',
        'TRITON_CACHE_DIR=/tmp/k9_scratch/triton','.venv/bin/python']
jobs=[]
for rep in (1,2):
 for suite in ('spatial','l10'):
  for scale in (50,500):
   for family in ('AWM','MixedJudge'):
    cell=f'pi05_{suite}_{scale}_{family}'
    jobs.append(dict(kind='stage',cell=cell,rep=rep,precision='float32',minimum=15872))
    for precision in ('float32','float64'):
     jobs.append(dict(kind='bench',cell=cell,rep=rep,precision=precision,minimum=15360))
ledger=[];ledger_file=HERE/'final_commands.json'
if ledger_file.exists():ledger=json.loads(ledger_file.read_text())
final_code_time=float((HERE/'FINAL_CODE_TIME.txt').read_text())
def job_id(job):return tuple(job[k] for k in ('kind','cell','rep','precision'))
def done(job):return any(job_id(r['job'])==job_id(job) and r['exit']==0 and r['started']>=final_code_time for r in ledger)
jobs=[j for j in jobs if not done(j)]
while jobs:
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
 free=float(raw.split(',')[0]);eligible=next((j for j in jobs if free>=j['minimum']),None)
 if eligible is None:
  time.sleep(5);continue
 job=eligible;attempt=1+sum(job_id(r['job'])==job_id(job) for r in ledger);run=f'final{job["rep"]}_a{attempt}'
 script='bench_stage1.py' if job['kind']=='stage' else 'benchmark.py'
 cmd=prefix+[str(HERE/script),'--config',job['cell'],'--run',run]
 if job['kind']=='bench':cmd+=['--precision',job['precision']]
 stem=f"{job['kind']}_{job['cell']}_{job['precision']}_{run}"
 log=HERE/(stem+'.log')
 (HERE/(stem+'_admission.txt')).write_text(subprocess.check_output(['nvidia-smi'],text=True))
 print('START',stem,raw,flush=True)
 began=time.time()
 with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 rec=dict(job=job,command=cmd,run=run,log=str(log),exit=r.returncode,started=began,ended=time.time(),admission=raw)
 ledger.append(rec);ledger_file.write_text(json.dumps(ledger,indent=2))
 if r.returncode==0:
  jobs.remove(job);print('DONE',stem,'remaining',len(jobs),flush=True)
 else:
  txt=log.read_text()
  if 'GPU memory admission/stop:' in txt and 'own GPU memory' not in txt:
   print('DEFER memory',stem,flush=True);time.sleep(5)
  else:
   print('STOP error',stem,txt[-3000:],flush=True);sys.exit(r.returncode or 1)
print('ALL COMPLETE',flush=True)
