"""Temporary public read-only rsync, capped single stream, closed in finally."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from a100_closedloop_prepare_1018 import HERE,OUT,REMOTE,sha,write
from concurrent_prepare import include_list


def remote(argv,seconds='120s'):
 return subprocess.run(['tether','exec','--timeout',seconds,'a100','--',*map(str,argv)],check=True,text=True)


def reverse_stage(model,tools_only=False):
 """A100 has public ingress; local NAT only forwards the forbidden 231xx range."""
 root=OUT/model;tools=OUT/'tools';tools.mkdir(exist_ok=True)
 resource=None
 if model!='pi05':
  from a100_closedloop_resources_1018 import prepare
  resource=prepare(model)
  # A future model may not overwrite the execution tools of a live queue.
  code='from pathlib import Path;import json; r=Path("/home/exouser/xgpu_1017/cl18/queue"); active=[p.name for p in r.glob("*.json") if json.loads(p.read_text()).get("status")=="RUNNING"]; assert not active, active'
  remote(['python3','-c',code],'30s')
 # The pre-existing A100 openpi island omitted the formal socket server wheel.
 # Copy the exact installed serving version into a separate user overlay.
 extra=OUT/'serving_extra';extra.mkdir(exist_ok=True)
 for name in ('websockets','websockets-16.0.dist-info'):
  source=Path('/home/weiland/projects/openpi/.venv/lib/python3.11/site-packages')/name
  if source.is_dir():shutil.copytree(source,extra/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ('a100_closedloop_run_1018.py','a100_closedloop_queue_1018.py','dp_distance_capture_1018.py','dp_distance_replay_1018.py','client_trace.py','pilot_readmap.py','paired_sr.py',
              'replay.py','artifacts.py','capture.py','inventory.py','concurrent_server.py'):
  shutil.copyfile(HERE/name,tools/name)
 snapshot=root/'execution_tools';snapshot.mkdir(exist_ok=True)
 for source in tools.glob('*.py'):shutil.copyfile(source,snapshot/source.name)
 write(root/'EXECUTION_TOOLS.json',{p.name:sha(p) for p in snapshot.glob('*.py')})
 for arm in ('pure','cache'):
  d=root/arm
  if not (d/'DEPENDENCIES.json').exists():write(d/'DEPENDENCIES.json',{})
  if not (d/'SOURCE_ATTESTATIONS.json').exists():shutil.copyfile('/data/mw_finetune/eval/tpi7_0536/SERVER_ATTESTATIONS.json',d/'SOURCE_ATTESTATIONS.json')
  write(d/'READMAP.json',{old:str(REMOTE/'deps'/old.lstrip('/')) for old in json.loads((d/'DEPENDENCIES.json').read_text())})
 if model=='pi05':
  write(root/'MODEL_PATHS.json',{'/data/ckpt':'/home/exouser/xgpu_1017/rootfs/data/ckpt'})
  write(root/'WORKER.json',dict(python=str(REMOTE/'simenv/bin/python'),extra_paths=[],env=dict(
       MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',
       OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')))
 transfer=OUT/'transfer'/('reverse_'+str(int(time.time())));transfer.mkdir(parents=True)
 dependencies={}
 for arm in ('pure','cache'):dependencies.update(json.loads((root/arm/'DEPENDENCIES.json').read_text()))
 files=transfer/'deps.files';files.write_text(''.join(str(Path(p).relative_to('/data/mw_finetune'))+'\n' for p in dependencies))
 modules=dict(payload=str(REMOTE),sim=str(REMOTE/'simenv/lib/python3.11/site-packages'),deps=str(REMOTE/'deps/data/mw_finetune'))
 if resource:modules['bulk']=resource['destination']
 runtime_sources={}
 for i,arm in enumerate(('pure','cache')):
  p=Path(json.loads((root/arm/'PREREGISTRATION.json').read_text())['runtime_source'])
  modules['runtime'+str(i)]=str(REMOTE/'runtimes'/p.name);runtime_sources['runtime'+str(i)]=p
 subprocess.run(['ss','-ltn'],check=True);remote(['ss','-ltn'],'30s')
 port=29770;daemon_root=REMOTE/'transfer'/transfer.name
 config='use chroot = false\nread only = false\nlist = false\nhosts allow = 192.80.109.240\n'+f'pid file = {daemon_root}/daemon.pid\nlog file = {daemon_root}/daemon.log\n'
 for name,dest in modules.items():config+=f'[{name}]\npath = {dest}\nread only = false\nlist = false\n'
 code=('from pathlib import Path; import subprocess,json; '
       f'd=Path({str(daemon_root)!r});d.mkdir(parents=True,exist_ok=False); '
       f'modules={modules!r}; '
       '[Path(p).mkdir(parents=True,exist_ok=True) for p in modules.values()]; '
       f'c=d/"rsyncd.conf";c.write_text({config!r}); '
       'f=(d/"stdio.log").open("ab"); '
       f'p=subprocess.Popen(["rsync","--daemon","--no-detach","--config="+str(c),"--port={port}"],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True); '
       'print(json.dumps({"pid":p.pid}))')
 result=subprocess.run(['tether','exec','--timeout','30s','a100','--','python3','-c',code],check=True,text=True,capture_output=True)
 pid=json.loads(result.stdout)['pid'];journal=dict(model=model,port=port,pid=pid,commands=[],started=time.time(),io_before=Path('/proc/pressure/io').read_text())
 try:
  time.sleep(.3)
  def copy(source,module,options=()):
   cmd=['rsync','-a','--bwlimit=16384','--contimeout=10','--timeout=120','--exclude=__pycache__','--exclude=*.pyc',*options,str(source)+'/',f'rsync://149.165.152.232:{port}/{module}/']
   journal['commands'].append(cmd);write(transfer/'TRANSFER.json',journal)
   subprocess.run(cmd,check=True)
  copy(OUT,'payload',['--include=/tools/***','--include=/serving_extra/***','--include=/'+model+'/',
       '--include=/'+model+'/pure/***','--include=/'+model+'/cache/***',
       '--include=/'+model+'/execution_tools/***','--include=/'+model+'/EXECUTION_TOOLS.json',
       '--include=/'+model+'/libero_config/***',
       '--include=/'+model+'/MODEL_PATHS.json','--include=/'+model+'/WORKER.json',
       '--include=/'+model+'/RESOURCES.json','--exclude=*'])
  if tools_only:
   journal['status']='COMPLETE_TOOLS_ONLY'
   return
  for name,source in runtime_sources.items():copy(source,name)
  if resource:
   remote(['python3','-c','import shutil; assert shutil.disk_usage("/home/exouser/xgpu_1017/cl18").free > '+str(resource['bytes']+2*2**30)+', "insufficient disk for one model plus 2GiB headroom"'],'30s')
   copy('/','bulk',['--files-from='+resource['bulk_list']])
   model_env=REMOTE/'models'/model/'env'
   remote([resource['interpreter'],'-m','venv','--without-pip',model_env])
   site=model_env/('lib/python'+resource['python_version']+'/site-packages')
   code=f'from pathlib import Path; p=Path({str(site/"formal_source.pth")!r});p.write_text({resource["site"]!r}+"\\n")'
   remote(['python3','-c',code],'30s')
   if resource['libero_asset_link']:
    remote(['/home/exouser/xgpu_1017/python/bin/python3.11','-m','venv','--without-pip',REMOTE/'liberoenv'])
    code=('from pathlib import Path; '
      f'p=Path({str(REMOTE/"liberoenv/lib/python3.11/site-packages/formal_source.pth")!r});'
      'p.write_text("/home/exouser/xgpu_1017/env/lib/python3.11/site-packages\\n"); '
      f'link=Path({resource["libero_asset_link"]!r}); '
      f'assert not link.exists() and not link.is_symlink();link.symlink_to({resource["libero_assets"]!r},target_is_directory=True)')
    remote(['python3','-c',code],'30s')
   if model=='cosmos':
    vendor=resource['model_paths']['/data/cosmos/cosmos-policy']
    code=f'from pathlib import Path;p=Path({vendor!r})/".venv";assert not p.exists();p.symlink_to({str(model_env)!r},target_is_directory=True)'
    remote(['python3','-c',code],'30s')
  else:
   remote(['/home/exouser/xgpu_1017/python/bin/python3.11','-m','venv','--without-pip',REMOTE/'simenv'])
   copy('/home/weiland/projects/openpi_ext/envs/metaworld_sim/lib/python3.11/site-packages','sim')
  if dependencies:copy('/data/mw_finetune','deps',['--files-from='+str(files)])
  journal['status']='COMPLETE'
 finally:
  remote(['kill','-TERM',str(pid)],'30s')
  journal.update(closed=time.time(),io_after=Path('/proc/pressure/io').read_text());write(transfer/'TRANSFER.json',journal)
  write(transfer/'DAEMON_CLOSED.json',dict(pid=pid,port=port,signal='SIGTERM by PID'))


def stage(model):
 root=OUT/model
 if model!='pi05':raise ValueError('stage is currently implemented and audited for pi05; do not run another environment yet')
 # These are copied source bytes only; no symlink aliases into a formal tree.
 tools=OUT/'tools';tools.mkdir(exist_ok=True)
 for name in ('a100_closedloop_run_1018.py','client_trace.py','pilot_readmap.py',
              'paired_sr.py','replay.py','artifacts.py','capture.py','inventory.py','concurrent_server.py'):
  shutil.copyfile(HERE/name,tools/name)
 pure=root/'pure'
 if not (pure/'DEPENDENCIES.json').exists():write(pure/'DEPENDENCIES.json',{})
 if not (pure/'SOURCE_ATTESTATIONS.json').exists():
  shutil.copyfile('/data/mw_finetune/eval/tpi7_0536/SERVER_ATTESTATIONS.json',pure/'SOURCE_ATTESTATIONS.json')
 dependencies={}
 for arm in ('pure','cache'):
  dep=json.loads((root/arm/'DEPENDENCIES.json').read_text());dependencies.update(dep)
  write(root/arm/'READMAP.json',{old:str(REMOTE/'deps'/old.lstrip('/')) for old in dep})
 write(root/'MODEL_PATHS.json',{'/data/ckpt':'/home/exouser/xgpu_1017/rootfs/data/ckpt'})
 write(root/'WORKER.json',dict(python=str(REMOTE/'simenv/bin/python'),extra_paths=[],env=dict(
      MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',
      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')))
 transfer=OUT/'transfer';transfer.mkdir(exist_ok=True)
 nonce=str(int(time.time()));directory=transfer/nonce;directory.mkdir()
 include=directory/'deps.include';include.write_text(include_list([Path(p) for p in dependencies],Path('/data/mw_finetune')))
 files=directory/'deps.files';files.write_text(''.join(str(Path(p).relative_to('/data/mw_finetune'))+'\n' for p in dependencies))
 # Check listeners on both hosts before any daemon, model or worker starts.
 subprocess.run(['ss','-ltn'],check=True);remote(['ss','-ltn'],'30s')
 port=None
 for candidate in range(29770,29790):
  try:
   with socket.socket() as s:s.bind(('0.0.0.0',candidate))
   port=candidate;break
  except OSError:continue
 if port is None:raise RuntimeError('no study transfer port available')
 modules=[('payload',OUT)]
 for i,arm in enumerate(('pure','cache')):
  runtime=Path(json.loads((root/arm/'PREREGISTRATION.json').read_text())['runtime_source'])
  modules.append(('runtime'+str(i),runtime))
 modules += [('sim','/home/weiland/projects/openpi_ext/envs/metaworld_sim/lib/python3.11/site-packages'),
             ('deps','/data/mw_finetune')]
 config=directory/'rsyncd.conf'
 lines=[f'pid file = {directory}/daemon.pid',f'log file = {directory}/daemon.log',
        f'lock file = {directory}/daemon.lock','use chroot = false','read only = true',
        'list = false',f'uid = {os.getuid()}',f'gid = {os.getgid()}']
 for name,path in modules:
  lines += [f'[{name}]',f'path = {path}','read only = true','list = false']
  if name=='deps':lines += [f'include from = {include}','exclude = *']
 config.write_text('\n'.join(lines)+'\n')
 journal=dict(model=model,port=port,io_before=Path('/proc/pressure/io').read_text(),started=time.time(),commands=[])
 daemon=subprocess.Popen(['rsync','--daemon','--no-detach','--config='+str(config),'--port='+str(port)])
 try:
  time.sleep(.3)
  if daemon.poll() is not None:raise RuntimeError('temporary daemon failed')
  remote(['mkdir','-p',REMOTE,REMOTE/'runtimes',REMOTE/'deps/data/mw_finetune'])
  def copy(module,destination,options=()):
   cmd=['rsync','-a','--bwlimit=16384','--exclude=__pycache__','--exclude=*.pyc',*options,
        f'rsync://ziyanglin.com:{port}/{module}/',str(destination)+'/']
   journal['commands'].append(cmd);write(directory/'TRANSFER.json',journal)
   remote(cmd,'20m')
  # Explicit payload allowlist avoids failed staging evidence and other models.
  copy('payload',REMOTE,['--include=/tools/***','--include=/'+model+'/',
       '--include=/'+model+'/pure/***','--include=/'+model+'/cache/***',
       '--include=/'+model+'/MODEL_PATHS.json','--include=/'+model+'/WORKER.json','--exclude=*'])
  for i,arm in enumerate(('pure','cache')):
   runtime=Path(json.loads((root/arm/'PREREGISTRATION.json').read_text())['runtime_source'])
   copy('runtime'+str(i),REMOTE/'runtimes'/runtime.name)
  remote(['/home/exouser/xgpu_1017/python/bin/python3.11','-m','venv','--without-pip',REMOTE/'simenv'])
  copy('sim',REMOTE/'simenv/lib/python3.11/site-packages')
  if dependencies:
   # rsync --files-from needs its list on the receiving A100. Transfer the
   # small list through the same public daemon, never tether file transport.
   copy('payload',REMOTE,['--include=/transfer/','--include=/transfer/'+nonce+'/',
        '--include=/transfer/'+nonce+'/deps.files','--exclude=*'])
   copy('deps',REMOTE/'deps/data/mw_finetune',['--files-from='+str(REMOTE/'transfer'/nonce/'deps.files')])
  journal['status']='COMPLETE'
 finally:
  if daemon.poll() is None:
   daemon.terminate()
   try:daemon.wait(timeout=10)
   except subprocess.TimeoutExpired:daemon.kill();daemon.wait(timeout=10)
  journal.update(daemon_pid=daemon.pid,daemon_returncode=daemon.returncode,closed=time.time(),
                 io_after=Path('/proc/pressure/io').read_text())
  write(directory/'TRANSFER.json',journal)
  write(directory/'DAEMON_CLOSED.json',dict(pid=daemon.pid,returncode=daemon.returncode,port=port))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--tools-only',action='store_true');a=p.parse_args();reverse_stage(a.model,a.tools_only)
