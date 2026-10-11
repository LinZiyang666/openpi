import os,time,json,pathlib,platform,resource
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('MUJOCO_GL')=='osmesa'
import numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from OpenGL.GL import glGetString,GL_RENDERER
b=benchmark.get_benchmark_dict()['libero_spatial']();rows=[]
for tid in [0,2]:
 task=b.get_task(tid)
 for size in [128,256]:
  t=time.perf_counter();e=OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=size,camera_widths=size,hard_reset=False);construct=time.perf_counter()-t
  t=time.perf_counter();e.reset();reset=time.perf_counter()-t
  renderer=str(glGetString(GL_RENDERER));assert 'llvmpipe' in renderer.lower() or 'softpipe' in renderer.lower(),renderer
  phases={}
  for cameras in [True,False]:
   for key,v in e.env._observables.items():
    if key.endswith('_image'):v.set_enabled(cameras)
   vals=[];cpu=[]
   for i in range(24):
    t=time.perf_counter();c=time.process_time();obs,*_=e.step(np.array([0,0,0,0,0,0,-1]))
    if i>=4:vals.append(time.perf_counter()-t);cpu.append(time.process_time()-c)
   phases['step_two_rgb' if cameras else 'physics']={'mean_s':float(np.mean(vals)),'mean_cpu_s':float(np.mean(cpu)),'p95_s':float(np.percentile(vals,95))}
  row=dict(task=task.name,size=size,construct_s=construct,reset_s=reset,phases=phases,renderer=renderer,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
  rows.append(row);print(json.dumps(row),flush=True);e.close()
pathlib.Path(__file__).with_name('libero_cpu_results.json').write_text(json.dumps(dict(host=platform.node(),affinity=list(os.sched_getaffinity(0)),rows=rows),indent=2))
