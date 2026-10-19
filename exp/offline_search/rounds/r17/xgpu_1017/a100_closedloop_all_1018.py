"""Serial A100 study coordinator. Waits for the existing pi05 queue, then stages
and runs the remaining fixed plans. All model/worker processes run on A100.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from a100_closedloop_prepare_1018 import HERE,OUT,REMOTE,write


def control(code):
 r=subprocess.run(['tether','exec','--timeout','45s','a100','--','python3','-c',code],
       text=True,capture_output=True,check=True)
 return json.loads(r.stdout)


def state(model):
 return control(f'from pathlib import Path;import json;p=Path({str(REMOTE/"queue"/(model+".json"))!r});print(p.read_text() if p.exists() else "null")')


def wait(model):
 while True:
  s=state(model)
  if s and s['status']=='COMPLETE':return s
  if s and s['status']=='FAILED':raise RuntimeError('A100 '+model+' queue failed; inspect its new study log, no next model staged')
  if s is None:raise RuntimeError('A100 '+model+' queue not submitted')
  time.sleep(30)


def cleanup(model):
 # Only this study's known model files, after all four fixed arms closed.
 target=('/home/exouser/xgpu_1017/rootfs/data/ckpt/pi05_metaworld_rlinf' if model=='pi05'
         else str(REMOTE/'models'/model))
 code=('from pathlib import Path;import json,shutil,subprocess,time; '
    f'r=Path({str(REMOTE)!r});model={model!r};p=Path({target!r}); '
    'q=json.loads((r/"queue"/(model+".json")).read_text());assert q["status"]=="COMPLETE"; '
    'receipts=[json.loads((r/"runs"/(model+"_"+a+"_cold"+str(c))/"RUN.json").read_text()) for a in ("pure","cache") for c in (1,2)]; '
    'assert all(x["status"]=="COMPLETE" and x["mps_cleanup"]["daemon_closed"] for x in receipts); '
    'used=float(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True));assert used<128; '
    'assert not p.is_symlink() and p.resolve().is_relative_to(Path("/home/exouser/xgpu_1017")); '
    'size=sum(x.stat().st_size for x in p.rglob("*") if x.is_file() and not x.is_symlink()) if p.exists() else 0; '
    'shutil.rmtree(p) if p.exists() else None; '
    'record=dict(model=model,path=str(p),bytes=size,time=time.time(),reason="completed four formal cold starts; serial capacity handoff; retain all results and traces"); '
    'f=(r/"CLEANUP.jsonl").open("a");f.write(json.dumps(record)+"\\n");f.close();print(json.dumps(record))')
 return control(code)


def collect(model):
 # Derived paired statistics and per-episode binary outcomes only; no per_step.
 code=('from pathlib import Path;import subprocess,json,socket; '
    f'r=Path({str(REMOTE)!r});d=r/"transfer"/({("collect_"+model+"_")!r}+str(__import__("time").time_ns()));d.mkdir(); '
    'port=next(q for q in (29771,29773,29775,29776) if not __import__("subprocess").run(["bash","-c","ss -ltn | grep -q :%d\\ "%q]).returncode==0); '
    's=socket.socket();s.bind(("0.0.0.0",port));s.close(); '
    'c=d/"rsyncd.conf";c.write_text("use chroot = false\\nread only = true\\nlist = false\\nhosts allow = 192.80.109.240\\npid file = "+str(d/"pid")+"\\nlog file = "+str(d/"log")+"\\n[analysis]\\npath = "+str(r/"analysis")+"\\n"); '
    'log=(d/"stdio").open("ab");p=subprocess.Popen(["rsync","--daemon","--no-detach","--config="+str(c),"--port="+str(port)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True); '
    'print(json.dumps(dict(pid=p.pid,directory=str(d),port=port)))')
 result=control(code);destination=OUT/'collected';destination.mkdir(exist_ok=True)
 try:
  time.sleep(.3)
  subprocess.run(['rsync','-a','--bwlimit=16384','--contimeout=10','--timeout=120',
     f'rsync://149.165.152.232:{result["port"]}/analysis/{model}_PAIRED.json',str(destination)+'/'],check=True)
  if model=='dp':
   subprocess.run(['rsync','-a','--bwlimit=16384','--contimeout=10','--timeout=120',
      f'rsync://149.165.152.232:{result["port"]}/analysis/dp_DISTANCE.json',str(destination)+'/'],check=True)
 finally:
  subprocess.run(['tether','exec','--timeout','30s','a100','--','kill','-TERM',str(result['pid'])],check=True)
  write(destination/(model+'_TRANSFER.json'),dict(**result,closed=time.time(),per_step=False))
 return destination/(model+'_PAIRED.json')


def launch(model):
 c=json.loads((OUT/model/'pure/campaign_template.json').read_text())
 argv=[c['server_python'],'-u',str(REMOTE/model/'execution_tools/a100_closedloop_queue_1018.py'),'--model',model]
 code=('from pathlib import Path;import subprocess,json; '
       f'r=Path({str(REMOTE)!r});log=(r/{(model+"_queue.log")!r}).open("xb"); '
       f'p=subprocess.Popen({argv!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); '
       'print(json.dumps(dict(pid=p.pid)))')
 return control(code)


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--collect',choices=['pi05','groot','dp','fastwam','cosmos']);a=p.parse_args()
 if a.collect:return collect(a.collect)
 if not a.run:
  print('wait pi05; collect; cleanup only its study weights; groot -> dp -> fastwam -> cosmos, each stage + four cold starts + collect + cleanup')
  return
 import fcntl
 with (OUT/'all_models.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  receipt=OUT/'ALL_MODELS.json'
  completed=[]
  if receipt.exists():
   previous=json.loads(receipt.read_text())
   if not a.resume or previous['status']!='FAILED':raise ValueError('serial coordinator already submitted; only a failed coordinator may resume')
   completed=previous['completed']
   receipt.rename(receipt.with_name('ALL_MODELS_FAILED_'+str(int(time.time()))+'.json'))
  s=dict(status='RUNNING',pid=os.getpid(),started=time.time(),current='pi05',completed=completed)
  def save():write(receipt,s)
  save()
  try:
   if 'pi05' not in s['completed']:
    wait('pi05');collect('pi05');s['completed'].append('pi05');save();cleanup('pi05')
   for model in ('groot','dp','fastwam','cosmos'):
    if model in s['completed']:continue
    s['current']=model;save()
    log=OUT/(model+'_stage.log')
    remote_state=state(model)
    if remote_state is None:
     if log.exists():log.rename(log.with_name(log.stem+'_failed_'+str(int(time.time()))+'.log'))
     with log.open('xb') as f:
      subprocess.run([sys.executable,str(HERE/'a100_closedloop_stage_1018.py'),model],stdout=f,stderr=subprocess.STDOUT,check=True)
     s['launch']=launch(model);save()
    wait(model);collect(model);s['completed'].append(model);save();cleanup(model)
   s['status']='COMPLETE'
  except BaseException as e:s.update(status='FAILED',error=repr(e));raise
  finally:s['ended']=time.time();save()


if __name__=='__main__':main()
