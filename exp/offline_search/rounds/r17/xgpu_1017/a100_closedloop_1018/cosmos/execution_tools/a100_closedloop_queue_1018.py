"""Run the fixed four A100 cold starts serially; stop on infrastructure failure."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/exouser/xgpu_1017/cl18')


def analyse(model):
 from paired_sr import paired_interval
 report=[]
 for arm in ('pure','cache'):
  base=ROOT/model/arm
  reference=json.loads((base/'REFERENCE.json').read_text())['outcomes']
  runs=[]
  for cold in (1,2):
   r=ROOT/'runs'/f'{model}_{arm}_cold{cold}'
   receipt=json.loads((r/'RUN.json').read_text())
   if receipt['status']!='COMPLETE' or not receipt['mps_cleanup']['daemon_closed']:
    raise ValueError('four complete, closed formal cold starts required')
   runs.append(json.loads((r/'OUTCOMES.json').read_text()))
  uids=json.loads((base/'UIDS.json').read_text())
  if any(set(r)!=set(uids) for r in [reference,*runs]):raise ValueError('missing accepted original UID')
  v=[[int(r[u]['success']) for u in uids] for r in [reference,*runs]]
  report.append(dict(model=model,arm=arm,episodes=len(uids),
     same_card=paired_interval(v[1],v[2]),
     a100_cold1_vs_formal=paired_interval(v[1],v[0]),
     a100_cold2_vs_formal=paired_interval(v[2],v[0]),
     per_episode=[dict(uid=u,formal=v[0][i],cold1=v[1][i],cold2=v[2][i],
         same_card_difference=v[1][i]-v[2][i],cold1_vs_formal=v[1][i]-v[0][i],
         cold2_vs_formal=v[2][i]-v[0][i]) for i,u in enumerate(uids)]))
 target=ROOT/'analysis';target.mkdir(exist_ok=True)
 (target/(model+'_PAIRED.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps([{k:v for k,v in r.items() if k!='per_episode'} for r in report]),flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['pi05','groot','dp','fastwam','cosmos']);p.add_argument('--analyse-only',action='store_true');p.add_argument('--resume-startup',action='store_true');p.add_argument('--wait-lock',action='store_true')
 a=p.parse_args()
 if a.analyse_only:return analyse(a.model)
 # --wait-lock queues this model directly behind the current GPU holder (no idle gap).
 lock=(ROOT/'exclusive_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX if a.wait_lock else fcntl.LOCK_EX|fcntl.LOCK_NB)
 names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
 if 'A100' not in names or 'H100' in names:raise ValueError('dedicated A100 only')
 root=ROOT/'queue';root.mkdir(exist_ok=True)
 receipt=root/(a.model+'.json')
 if receipt.exists():
  old=json.loads(receipt.read_text())
  if not a.resume_startup or old['status']!='FAILED' or old['completed']:
   raise ValueError('queue already submitted; never rerun accepted episodes')
  current=old['current'];failed=ROOT/'runs'/f'{a.model}_{current["arm"]}_cold{current["cold"]}'
  journal=failed/'driver/journal.jsonl'
  if journal.exists() and any(json.loads(line).get('accepted') for line in journal.read_text().splitlines()):
   raise ValueError('accepted episodes exist; a native missing-UID continuation is required')
  pidfile=failed/'OWNED_PIDS.json'
  if pidfile.exists():
   # This branch is only for failures whose cleanup returned every owned PID.
   run=json.loads((failed/'RUN.json').read_text())
   if run.get('mps_cleanup') and not run['mps_cleanup'].get('daemon_closed'):raise ValueError('private MPS not closed')
   if any(p.get('returncode') is None for p in run.get('cleanup',[])):raise ValueError('owned child still live')
  stamp=str(int(time.time()))
  if failed.exists():failed.rename(failed.with_name(failed.name+'_startup_failed_'+stamp))
  receipt.rename(receipt.with_name(a.model+'_startup_failed_'+stamp+'.json'))
 state=dict(model=a.model,status='RUNNING',pid=os.getpid(),started=time.time(),completed=[])
 def save():receipt.write_text(json.dumps(state,indent=2)+'\n')
 save()
 try:
  for arm,cold in [('pure',1),('pure',2),('cache',1),('cache',2)]:
   cmd=[sys.executable,str(Path(__file__).with_name('a100_closedloop_run_1018.py')),'run',
        '--model',a.model,'--arm',arm,'--cold',str(cold)]
   state['current']=dict(arm=arm,cold=cold,argv=cmd);save()
   subprocess.run(cmd,check=True)
   r=ROOT/'runs'/f'{a.model}_{arm}_cold{cold}'/'RUN.json';run=json.loads(r.read_text())
   if run['status']!='COMPLETE' or not run['mps_cleanup']['daemon_closed']:raise RuntimeError('run did not close cleanly')
   state['completed'].append(dict(arm=arm,cold=cold));save()
  # Model islands may lack scipy (Fast-WAM); pairing runs in the openpi island (CPU only).
  subprocess.run(['/home/exouser/xgpu_1017/env/bin/python',__file__,'--model',a.model,'--analyse-only'],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  if a.model=='dp':
   environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
   subprocess.run([sys.executable,str(Path(__file__).with_name('dp_distance_replay_1018.py')),
      str(ROOT/'runs/dp_cache_cold1/servers'),str(ROOT/'runs/dp_cache_cold2/servers'),
      '--output',str(ROOT/'analysis/dp_DISTANCE.json')],env=environment,check=True)
  state['status']='COMPLETE'
 except BaseException as e:state.update(status='FAILED',error=repr(e));raise
 finally:state['ended']=time.time();save()


if __name__=='__main__':main()
