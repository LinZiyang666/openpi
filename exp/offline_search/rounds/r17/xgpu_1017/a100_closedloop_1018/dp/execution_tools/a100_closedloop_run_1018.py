"""A100-only formal server/planner/worker path with a fixed original-UID subset.

No remote hosts, formal launchers, fleet autoscaler, chains or shared MPS.
Each arm/cold start owns fresh paths and Popen children; signals use PID only.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/exouser/xgpu_1017/cl18')


def read(p):return json.loads(Path(p).read_text())


def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_suffix(p.suffix+'.part');temp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');temp.replace(p)


def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def paths(c,mode=None):
 r=Path(c['runtime'])
 # LIBERO formal workers run their campaign's own sealed workers.tar payload.
 if mode=='worker' and c.get('xgpu_worker_payload'):r=Path(c['xgpu_worker_payload'])
 sys.path[:0]=[str(r),str(r/'src'),str(r/'packages/openpi-client/src'),str(HERE)]
 if c['model'] in ('pi05','dp'):sys.path.insert(0,str(ROOT/'serving_extra'))  # openpi island lacks websockets


def filtered_plan(c):
 from exp.r17_integration.ops import formal_driver
 cache_study=None
 if c.get('cache_study'):
  from exp.r17_integration.ops import cache_study
 wanted={t['task_uid']:t for t in read(c['xgpu_subset'])['tasks']}
 expected=read(c['xgpu_subset'])['expected_episodes']
 if len(wanted)!=expected:raise ValueError('preregistered original UIDs required')
 if c.get('xgpu_group'):
  original=cache_study.selected_groups
  def groups(cc):
   seal,gs=original(cc);selected=[g for g in gs if g['bundle_id']==c['xgpu_group']['bundle_id']]
   if len(selected)!=1:raise ValueError('original group missing')
   return seal,selected
  cache_study.selected_groups=groups
 native=cache_study.plan if c.get('cache_study') else formal_driver.plan
 def subset(cc):
  benchmark,model,tasks,endpoints=native(cc)
  ignored=('server_host','server_port','task_uid','yaml_id')
  def key(t):return json.dumps({k:v for k,v in t.items() if k not in ignored},sort_keys=True)
  expected_by_science={key(t):t for t in wanted.values()}
  selected=[]
  for t in tasks:
   old=expected_by_science.get(key(asdict(t)))
   if old is not None:selected.append(replace(t,task_uid=old['task_uid'],yaml_id=old['yaml_id']))
  if len(selected)!=len(wanted):raise ValueError('native formal planner lost original UIDs')
  for t in selected:
   actual={k:v for k,v in asdict(t).items() if k not in ('server_host','server_port')}
   expected={k:v for k,v in wanted[t.task_uid].items() if k not in ('server_host','server_port')}
   if actual!=expected:raise ValueError('scientific task fields changed '+t.task_uid)
  return benchmark,model,selected,endpoints
 formal_driver.plan=subset
 if c.get('cache_study'):cache_study.plan=subset
 return subset


def child(a):
 c=read(a.campaign);paths(c,a.mode)
 if a.mode=='driver':
  planner=filtered_plan(c)
  bm,spec,tasks,eps=planner(c)
  write(Path(a.directory).parent/'EPISODE_PLAN.json',dict(tasks=[asdict(t) for t in tasks],
        expected_episodes=len(tasks),benchmark_protocol=bm.identity,model=spec))
  import importlib
  driver=importlib.import_module(c.get('driver_module','exp.r17_integration.ops.formal_driver'))
  driver.run(c,a.directory)
 if a.mode=='worker':
  from client_trace import install
  install(Path(a.directory)/'trace')
  if c.get('cache_study'):
   from exp.r17_integration.ops.cache_study_worker import main
  else:from exp.r17_integration.conductor.worker_entry import main
  return main(a.native)
 if a.mode=='resident':
  from pilot_readmap import install
  aliases=read(ROOT/c['model']/c['xgpu_arm']/'READMAP.json')
  with install(aliases):
   if c.get('xgpu_distance_capture'):
    from dp_distance_capture_1018 import install as capture_distance
    capture_distance(Path(a.directory)/'dp_distance_heads',campaign=c.get('cache_plan_name',c['run_name']))
   sys.argv=['formal_server','--campaign',a.campaign,'--port',str(a.port),'--directory',a.directory]
   if c.get('xgpu_cuda_graph'):sys.argv+=['--cuda-graph']
   boot=Path(c['assets'])/'ffk_bootstrap.py'
   if boot.exists():exec(compile(boot.read_text(),str(boot),'exec'),{'__name__':'__main__','__file__':str(boot)})
   else:runpy.run_module('exp.r17_integration.ops.formal_server',run_name='__main__')


class Children:
 def __init__(self,out,env,cwd):self.out,self.env,self.cwd,self.rows=out,env,cwd,[]
 def spawn(self,label,cmd,env=None):
  with (self.out/(label+'.log')).open('xb') as log:
   p=subprocess.Popen(cmd,cwd=self.cwd,env=env or self.env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  self.rows.append((label,p));write(self.out/'OWNED_PIDS.json',[dict(label=n,pid=x.pid,returncode=x.poll()) for n,x in self.rows])
  return p
 def port(self,port,deadline):
  while time.monotonic()<deadline:
   for label,p in self.rows:
    if p.poll() is not None:raise RuntimeError(f'{label} exited {p.returncode}; see {label}.log')
   try:
    with socket.create_connection(('127.0.0.1',port),timeout=.2):return
   except OSError:time.sleep(.2)
  raise TimeoutError('startup timeout')
 def stop(self):
  # Every target is an unreaped child of this supervisor; no name matching or
  # process-group signaling. Native resident/mps exec keeps the same PID.
  for label,p in reversed(self.rows):
   if p.poll() is None:
    p.terminate()
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
  return [dict(label=n,pid=p.pid,returncode=p.returncode) for n,p in self.rows]


def run_one(model,arm,cold):
 base=ROOT/model/arm;c=read(base/'campaign_template.json');pr=read(base/'PREREGISTRATION.json')
 if sha(base/'PLAN.json')!=pr['plan_sha256']:raise ValueError('preregistration changed')
 out=ROOT/'runs'/f'{model}_{arm}_cold{cold}'
 out.mkdir(parents=True,exist_ok=False)
 c['server_run']=str(out/'servers');c['xgpu_subset']=str(base/'PLAN.json')
 c['ports']=[c['public_port'],*c['backend_ports'],c['driver_port']]
 c['driver_runtime']=c['runtime'];c['coordinator_runtime']=c['runtime']
 paths(c)
 from replay import verify_runtime
 if pr.get('runtime_kind')=='legacy_archive':
  seal=read(base/'LEGACY_RUNTIME_FILES.json')
  if seal['archive_sha256']!=pr['runtime_identity']:raise ValueError('legacy runtime archive identity changed')
  for name,pin in seal['files'].items():
   if sha(Path(c['runtime'])/name)!=pin:raise ValueError('legacy runtime bytes changed '+name)
 else:verify_runtime(c['runtime'],pr['runtime_identity'])
 from exp.r17_integration.ops.formal_eval import server_env
 from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
 identity=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.free,uuid','--format=csv,noheader,nounits'],text=True).strip()
 name,used,free,uuid=identity.split(',')
 if 'A100' not in name or float(used)>128:raise RuntimeError('idle dedicated A100 required')
 gpu=dict(name=name.strip(),used_mib=float(used),free_mib=float(free),uuid=uuid.strip(),index=0)
 for port in c['ports']:
  if 23100<=port<=23199:raise ValueError('formal-lane port forbidden')
  with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
 # This descriptor changes physical paths only. Formal YAMLs and recipe IDs
 # remain byte-identical; READMAP aliases only pinned, verified cache files.
 private=out/'assets';private.mkdir()
 for p in (base/'assets').rglob('*'):
  if p.is_file():
   q=private/p.relative_to(base/'assets');q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
 desc=read(private/'load.json')
 def relocate(v):
  if isinstance(v,dict):return {k:relocate(x) for k,x in v.items()}
  if isinstance(v,list):return [relocate(x) for x in v]
  if isinstance(v,str):
   for old,new in sorted(read(ROOT/model/'MODEL_PATHS.json').items(),key=lambda p:-len(p[0])):
    if v==old or v.startswith(old+'/'):return new+v[len(old):]
  return v
 write(private/'load.json',relocate(desc));c['assets']=c['local_assets']=str(private)
 att=read(base/'SOURCE_ATTESTATIONS.json')
 c['xgpu_cuda_graph']=bool(att[0]['metadata'].get('cuda_graph'))
 if model=='dp' and c['xgpu_cuda_graph']:
  c['graph_probe_observations']=relocate('/data/mw_finetune/qualification/observations/dp_libero_v2')
 expected=att[0]['metadata']['r17_model_spec']
 c['vendor']=relocate(c.get('vendor',''))
 payload=base/'worker_payload'
 if payload.is_dir():
  seal=read(base/'WORKER_PAYLOAD.json')
  for name,pin in seal['files'].items():
   if sha(payload/name)!=pin:raise ValueError('formal worker payload changed '+name)
  c['xgpu_worker_payload']=str(payload)
 write(out/'campaign.json',c)
 env=dict(os.environ,**server_env(c,c['runtime']))
 env['PYTHONPATH']=':'.join(relocate(p) for p in env['PYTHONPATH'].split(':'))
 env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            PYTHONPATH=env['PYTHONPATH']+':'+str(HERE)+(':'+str(ROOT/'serving_extra') if model in ('pi05','dp') else '')+':/home/exouser/xgpu_1017/tools',
            OPENPI_DATA_HOME='/home/exouser/xgpu_1017/tokenizer',HF_HOME=relocate(desc.get('hf_home',str(ROOT/'model_cache'))))
 env.update(HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_MODULES_CACHE=str(ROOT/'model_cache/hf_modules'),
       TORCH_HOME=str(ROOT/'model_cache/torch'),XDG_CACHE_HOME=str(ROOT/'model_cache/xdg'),
       MPLCONFIGDIR=str(ROOT/'model_cache/matplotlib'),NUMBA_CACHE_DIR=str(ROOT/'model_cache/numba'),
       TORCHINDUCTOR_CACHE_DIR=str(ROOT/'model_cache/inductor'),TRITON_CACHE_DIR=str(ROOT/'model_cache/triton'),
       NO_ALBUMENTATIONS_UPDATE='1')
 owner=Children(out,env,c['runtime']);mps=None;report=dict(status='STARTING',model=model,arm=arm,cold=cold,
       episodes=pr['episodes'],preregistration=pr,gpu=gpu,started=time.time())
 write(out/'RUN.json',report)
 def interrupted(*unused):raise KeyboardInterrupt('owned A100 run interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  for old,pin in read(base/'DEPENDENCIES.json').items():
   p=read(base/'READMAP.json')[old]
   if Path(p).stat().st_size!=pin['bytes'] or sha(p)!=pin['sha256']:raise ValueError('cache dependency differs '+old)
  renv=env
  if c.get('mps'):
   from exp.r17_integration.models.fastwam.ops.mps import Session,client_env,paths as mps_paths
   from concurrent_server import reserve_mps_directory
   mps_dir=reserve_mps_directory(ROOT,gpu['uuid'],mps_paths)
   mps=Session(Path(mps_dir['campaign_directory']),'cuda:0');state=mps.start();write(out/'MPS_READY.json',state)
   renv=dict(env,**client_env(state))
  attestations=[]
  for i,port in enumerate(c['backend_ports']):
   cmd=[c['server_python'],'-u',str(HERE/'a100_closedloop_run_1018.py'),'resident',
        '--campaign',str(out/'campaign.json'),'--directory',str(out/'servers'/f'replica_{i}'),'--port',str(port)]
   if c.get('mps'):cmd=[c['server_python'],'-m','exp.r17_integration.models.fastwam.ops.mps','exec','--ready',str(out/'MPS_READY.json'),'--',*cmd]
   owner.spawn(f'replica_{i}',cmd,renv);owner.port(port,time.monotonic()+1200)
   ctl=WebsocketClientPolicy('127.0.0.1',port)
   try:meta=ctl.get_server_metadata()
   finally:ctl.close()
   if meta.get('r17_model_spec')!=expected:
    write(out/'SPEC_MISMATCH.json',dict(expected=expected,actual=meta.get('r17_model_spec')))
    raise ValueError('resident differs from actual source formal ModelSpec')
   if c['xgpu_cuda_graph'] and not meta.get('cuda_graph_self_check',{}).get('passed'):raise ValueError('formal graph self check missing')
   attestations.append(meta);write(out/'SERVER_ATTESTATIONS.json',attestations)
  owner.spawn('proxy',[c['server_python'],'-u','-c','from openpi.serving.replica_proxy import run_proxy; run_proxy('+str(c['public_port'])+','+repr(c['backend_ports'])+')'])
  owner.port(c['public_port'],time.monotonic()+60)
  ctl=WebsocketClientPolicy('127.0.0.1',c['public_port'])
  try:
   g=c.get('xgpu_group');file=g['file'] if g else 'full.yaml';bid=g['bundle_id'] if g else 'formal_full_v1'
   ack=ctl.load_cache_config(str(private/file),yaml_id=bid,bundle_id=bid)
   if ack.get('__ack__')=='error' or not ack.get('bundle_identity'):raise ValueError('bundle ACK failed '+repr(ack))
   if g and ack['bundle_identity']!=g['bundle_identity']:raise ValueError('recipe bundle changed')
  finally:ctl.close()
  write(out/'BUNDLE_ACK.json',ack)
  if g:write(out/'STUDY_BUNDLE_ACKS.json',dict(seal_digest=read(private/'CACHE_STUDY.json')['digest'],rows=[dict(bundle_id=bid,bundle_identity=ack['bundle_identity'],ack=ack)]))
  driver=owner.spawn('driver',[c['server_python'],'-u',str(HERE/'a100_closedloop_run_1018.py'),'driver',
       '--campaign',str(out/'campaign.json'),'--directory',str(out/'driver')])
  owner.port(c['driver_port'],time.monotonic()+120)
  worker_config=read(ROOT/model/'WORKER.json');wenv=dict(env,**worker_config['env'])
  # Use worker code from this arm's immutable formal runtime.
  wroot=c.get('xgpu_worker_payload',c['runtime'])
  wenv['PYTHONPATH']=':'.join([wroot+'/packages/openpi-client/src',wroot+'/src',wroot,str(HERE),*worker_config.get('extra_paths',[])])
  workers=[]
  for i in range(8):
   wd=out/'workers'/str(i);wd.mkdir(parents=True)
   argv=['--worker-id',out.name+'_'+str(i),'--server-key',c['endpoint'],
        '--driver-host','127.0.0.1','--driver-port',str(c['driver_port']),
        '--benchmark',c['domain'],'--benchmark-spec',str(private/'benchmark.json'),'--renderer','egl','--egl-device','0']
   workers.append(owner.spawn('worker_'+str(i),[worker_config['python'],'-u',str(HERE/'a100_closedloop_run_1018.py'),
       'worker','--campaign',str(out/'campaign.json'),'--directory',str(wd),'--',*argv],wenv))
  report['status']='RUNNING';write(out/'RUN.json',report)
  deadline=time.monotonic()+14460
  while driver.poll() is None:
   if time.monotonic()>deadline:raise TimeoutError('fixed 4-hour campaign watchdog')
   if any(p.poll() not in (None,0) for p in workers):raise RuntimeError('native worker failed')
   if any(p.poll() is not None for n,p in owner.rows if n.startswith('replica_') or n=='proxy'):raise RuntimeError('resident/proxy exited')
   time.sleep(2)
  for p in workers:p.wait(timeout=90)
  if driver.returncode or any(p.returncode for p in workers):raise RuntimeError('native driver/worker failed completion')
  outcomes={}
  with (out/'driver/journal.jsonl').open() as f:
   for line in f:
    r=json.loads(line)
    if r.get('accepted') is True:
     if r.get('error') or r['task_uid'] in outcomes:raise ValueError('accepted error or duplicate')
     outcomes[r['task_uid']]=r
  if set(outcomes)!=set(read(base/'UIDS.json')):raise ValueError('incomplete original-UID population')
  write(out/'OUTCOMES.json',outcomes);report['status']='COMPLETE'
 except BaseException as e:
  report.update(status='FAILED',error=repr(e));raise
 finally:
  report['cleanup']=owner.stop()
  report['mps_cleanup']=mps.close() if mps else dict(enabled=False,daemon_closed=True,scope='no private MPS daemon was created')
  report['ended']=time.time();write(out/'RUN.json',report)


def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','driver','worker','resident'])
 p.add_argument('--model');p.add_argument('--arm',choices=['pure','cache']);p.add_argument('--cold',type=int,choices=[1,2])
 p.add_argument('--campaign');p.add_argument('--directory');p.add_argument('--port',type=int)
 # Native worker argv follows an explicit -- and must not consume run options.
 argv=sys.argv[1:];native=[]
 if '--' in argv:i=argv.index('--');native=argv[i+1:];argv=argv[:i]
 a=p.parse_args(argv);a.native=native
 if a.mode=='run':
  if Path('/home/exouser/xgpu_1017/cl18')!=ROOT or not ROOT.is_dir():raise RuntimeError('A100 stage missing')
  run_one(a.model,a.arm,a.cold)
 else:child(a)


if __name__=='__main__':main()
