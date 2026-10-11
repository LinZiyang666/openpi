import os,time,json,resource,platform,argparse,pathlib
p=argparse.ArgumentParser();p.add_argument('--split',default='fixed');p.add_argument('--history',type=int,default=1);p.add_argument('--render-every',type=int,default=1);p.add_argument('--task',default='OpenDrawer');p.add_argument('--size',type=int,default=128);p.add_argument('--steps',type=int,default=40);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=7);args=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('MUJOCO_GL')=='osmesa'
start=time.perf_counter()
import numpy as np
import robocasa,robosuite,mujoco
from robocasa.utils.env_utils import create_env
from robocasa.utils.dataset_registry_utils import get_task_horizon
row={'task':args.task,'size':args.size,'steps':args.steps,'seed':args.seed,'host':platform.node(),'cpu_affinity':list(os.sched_getaffinity(0)),'mujoco':mujoco.__version__,'robocasa_file':robocasa.__file__,'robosuite_file':robosuite.__file__,'renderer':'OSMesa','probe_version':5,'split':args.split,'skip_method':'sensor_wrapper_preserve_clock','render_every':args.render_every,'history':args.history,'env_vars':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','MUJOCO_GL','PYOPENGL_PLATFORM','LP_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},'import_s':time.perf_counter()-start}
try:
 t=time.perf_counter(); c=time.process_time()
 split_kwargs=dict(split=None,obj_instance_split='target',layout_ids=[1],style_ids=[1]) if args.split=='fixed' else dict(split=args.split)
 env=create_env(env_name=args.task,camera_widths=args.size,camera_heights=args.size,seed=args.seed,**split_kwargs)
 row['construct_s']=time.perf_counter()-t;row['construct_cpu_s']=time.process_time()-c
 t=time.perf_counter();c=time.process_time();obs=env.reset();row['reset_s']=time.perf_counter()-t;row['reset_cpu_s']=time.process_time()-c
 from OpenGL.GL import glGetString,GL_RENDERER,GL_VENDOR
 try:
  row['gl_renderer']=str(glGetString(GL_RENDERER));row['gl_vendor']=str(glGetString(GL_VENDOR))
 except Exception as e:
  row['gl_metadata_warning']=repr(e)
 row['images']={k:list(v.shape) for k,v in obs.items() if k.endswith('_image')};row['action_dim']=env.action_dim;row['horizon']=get_task_horizon(args.task)
 action=np.zeros(env.action_dim);times=[];cpus=[];image_keys=[k for k in env._observables if k.endswith('_image')];row['image_observables']=image_keys
 render_flag=[True]
 for key in image_keys:
  original=env._observables[key]._sensor
  def sensor(cache, original=original, key=key):
   return original(cache) if render_flag[0] else cache.get(key,np.zeros((args.size,args.size,3),dtype=np.uint8))
  sensor.__modality__=original.__modality__
  env._observables[key]._sensor=sensor
 warmup=2*args.render_every
 for i in range(args.steps+warmup):
  
  render_flag[0]=(i%args.render_every)>=args.render_every-args.history
  t=time.perf_counter();c=time.process_time();obs,reward,done,info=env.step(action)
  if i>=warmup:times.append(time.perf_counter()-t);cpus.append(time.process_time()-c)
 row['step_s']=times;row['step_cpu_s']=cpus;row['step_mean_s']=float(np.mean(times));row['step_p50_s']=float(np.median(times));row['step_p95_s']=float(np.percentile(times,95));row['step_mean_cpu_s']=float(np.mean(cpus));row['max_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 row['episode_at_horizon_s']=row['reset_s']+row['step_mean_s']*row['horizon'];env.close()
except Exception as e:
 import traceback
 row['error']=repr(e);row['traceback']=traceback.format_exc();print(row['traceback'],flush=True)
pathlib.Path(args.out).write_text(json.dumps(row,indent=2));print(json.dumps({k:v for k,v in row.items() if k not in ['step_s','step_cpu_s','traceback']},indent=2),flush=True)
