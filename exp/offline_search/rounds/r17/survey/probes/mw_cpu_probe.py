"""CPU-only physics / RGB / depth microprobe; no learned policy or network service."""
import os,time,json,pathlib,resource,platform,importlib.metadata
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ['MUJOCO_GL']=='osmesa'
import numpy as np,metaworld,mujoco
from OpenGL.GL import glGetString,GL_RENDERER
rows=[]
for task in ['assembly-v3','button-press-wall-v3','box-close-v3']:
 t=time.perf_counter();bench=metaworld.MT1(task,seed=7);bench_s=time.perf_counter()-t
 for size in [128,256,480,512]:
  env=bench.train_classes[task](render_mode='rgb_array',camera_name='corner2',width=size,height=size)
  env.set_task(bench.train_tasks[0]);env.model.cam_pos[2]=[.75,.075,.7]
  resets=[]
  for i in range(3):
   t=time.perf_counter();env.reset();resets.append(time.perf_counter()-t)
  env.render();renderer=str(glGetString(GL_RENDERER));assert 'llvmpipe' in renderer.lower() or 'softpipe' in renderer.lower(),renderer
  phases={}
  for name in ['physics','rgb','rgb_two_views','depth']:
   if name=='depth': depth_renderer=mujoco.Renderer(env.model,height=size,width=size);depth_renderer.enable_depth_rendering()
   values=[];cpus=[]
   for i in range(45):
    t=time.perf_counter();c=time.process_time()
    if name=='physics':env.step(np.array([0.,0.,0.,0.]))
    elif name=='rgb':env.render()
    elif name=='rgb_two_views':
     env.mujoco_renderer.camera_id=2;env.render();env.mujoco_renderer.camera_id=0;env.render();env.mujoco_renderer.camera_id=2
    else:depth_renderer.update_scene(env.data,camera='corner2');depth_renderer.render()
    if i>=5:values.append(time.perf_counter()-t);cpus.append(time.process_time()-c)
   phases[name]={'mean_s':float(np.mean(values)),'p50_s':float(np.median(values)),'p95_s':float(np.percentile(values,95)),'mean_cpu_s':float(np.mean(cpus))}
   if name=='depth':depth_renderer.close()
  row=dict(task=task,size=size,bench_construct_s=bench_s,reset_s=resets,renderer=renderer,phases=phases,max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
  rows.append(row);env.close();print(json.dumps(row),flush=True)
result=dict(host=platform.node(),affinity=list(os.sched_getaffinity(0)),versions={n:importlib.metadata.version(n) for n in ['metaworld','mujoco','numpy']},rows=rows)
pathlib.Path(__file__).with_name('mw_cpu_results.json').write_text(json.dumps(result,indent=2))
