"""Negative device-selection and child-lifecycle checks; no disallowed render imports."""
import json, os, pathlib, subprocess, time
ROOT=pathlib.Path('/scratch/zixuans8/mw_r17/timan1_provision')
CFG=json.loads((ROOT/'NODE_CONFIG.json').read_text()); env=dict(os.environ,**CFG['env']); cases=[]
for key,values in [('MUJOCO_EGL_DEVICE_ID',['0','1','2','4','']),('CUDA_VISIBLE_DEVICES',['0','1','2','3','0,3','GPU-wrong'])]:
    for value in values:
        changed=dict(env);changed[key]=value
        r=subprocess.run([CFG['python'],'-c','print("BODY_EXECUTED")'],env=changed,capture_output=True,text=True)
        cases.append({'env_key':key,'value':value,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
for i in (0,1,2):
    r=subprocess.run([str(ROOT/'python-gpu3'),'-c','print("BODY_EXECUTED")','--egl-device',str(i)],env=env,capture_output=True,text=True)
    cases.append({'cli_device':i,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
for devices in ([0],[0,3],None):
    path=ROOT/'negative_campaign.json';path.write_text(json.dumps({'worker_node':'timan1','worker_egl_device':3,'worker_egl_devices':devices}))
    r=subprocess.run([CFG['python'],'-m','exp.r17_integration.ops.formal_workers','--campaign',str(path)],env=env,capture_output=True,text=True)
    cases.append({'campaign_devices':devices,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
# Direct native imports must refuse outside the private device namespace.
r=subprocess.run([CFG['python'],'-c','import mujoco; print("BODY_EXECUTED")'],env=env,capture_output=True,text=True)
cases.append({'native_outside_namespace':True,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
# Correct device UUID passes both the namespace and UUID selection checks.
code='from mujoco.egl import egl_ext; import os,json; egl_ext.eglQueryDevicesEXT(); print(json.dumps({"egl_uuid":os.environ["R17_GPU3_EGL_UUID_HEX"],"minor":os.minor(os.stat("/dev/nvidia0").st_rdev)}))'
r=subprocess.run([str(ROOT/'python-gpu3'),'-c',code],env=env,capture_output=True,text=True)
positive={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
# Stop only a newly created, exact-PID CPU fixture and verify namespace cleanup.
pid_file=ROOT/'lifecycle_cpu_pid.json'
if pid_file.exists(): raise RuntimeError('lifecycle fixture already exists')
code='import os,time,json;from pathlib import Path;Path('+repr(str(pid_file))+').write_text(json.dumps({"pid":os.getpid()}));time.sleep(30)'
p=subprocess.Popen([str(ROOT/'python-gpu3'),'-c',code],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
until=time.monotonic()+5
while not pid_file.exists() and p.poll() is None and time.monotonic()<until: time.sleep(.05)
if not pid_file.exists(): raise RuntimeError('lifecycle fixture did not start')
actual=json.loads(pid_file.read_text())['pid']; p.terminate(); exit_code=p.wait(timeout=5)
until=time.monotonic()+5
while pathlib.Path('/proc',str(actual)).exists() and time.monotonic()<until:
    status=pathlib.Path('/proc',str(actual),'stat')
    if status.exists() and status.read_text().split()[2]=='Z': break
    time.sleep(.05)
status=pathlib.Path('/proc',str(actual),'stat')
closed=not status.exists() or status.read_text().split()[2]=='Z'
if not closed:
    os.kill(actual,15);raise RuntimeError('own lifecycle fixture survived wrapper termination')
result={'status':'PASS' if all(c['returncode']==78 and not c['stdout'] for c in cases) and positive['returncode']==0 and closed else 'FAIL','negative_cases':cases,'positive_uuid':positive,'lifecycle':{'wrapper_pid':p.pid,'actual_pid':actual,'wrapper_exit_code':exit_code,'actual_child_closed':closed}}
(ROOT/'GUARD_PROOF_FINAL.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':result['status'],'negative_cases':len(cases),'positive_uuid':positive,'lifecycle':result['lifecycle']}))
