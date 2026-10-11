import argparse, json, os, pathlib, platform, statistics, time, importlib.metadata
import numpy as np
import metaworld
from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR
out=pathlib.Path(__file__).with_name('metaworld_cpu.json')
rows=[]; renderer=None
for name in ['reach-v3','reach-wall-v3','pick-place-v3','drawer-close-v3','assembly-v3']:
    t=time.perf_counter(); suite=metaworld.MT1(name,seed=7); build=time.perf_counter()-t
    for size in [224,480]:
        t=time.perf_counter(); env=suite.train_classes[name](render_mode='rgb_array',width=size,height=size,camera_name='corner2'); ctor=time.perf_counter()-t
        for idx in [0,1]:
            t=time.perf_counter();env.set_task(suite.train_tasks[idx]);env.reset();reset=time.perf_counter()-t
            t=time.perf_counter();env.render(); first=time.perf_counter()-t
            renderer=glGetString(GL_RENDERER).decode(); vendor=glGetString(GL_VENDOR).decode()
            assert 'llvmpipe' in renderer.lower(), renderer
            step_t=render_t=0.0; nrender=0
            w=time.perf_counter(); c=time.process_time()
            for i in range(500):
                t=time.perf_counter();env.step(np.zeros(4,dtype=np.float64));step_t+=time.perf_counter()-t
                if i%5==0:
                    t=time.perf_counter();env.render();render_t+=time.perf_counter()-t;nrender+=1
            row=dict(task=name,init_idx=idx,resolution=size,steps=500,renders=nrender,wall_s=time.perf_counter()-w,cpu_s=time.process_time()-c,physics_s=step_t,render_s=render_t,reset_s=reset,first_render_s=first,env_ctor_s=ctor,mt1_build_s=build)
            rows.append(row);print(json.dumps(row),flush=True)
        env.close()
report=dict(protocol='Timing only: zero actions, fixed 500 steps, one RGB view every 5 steps, no policy/no success score; first render excluded from warm episode',renderer=renderer,vendor=vendor,cpu_affinity=sorted(os.sched_getaffinity(0)),platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['metaworld','mujoco','gymnasium','numpy','PyOpenGL']},env={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','MUJOCO_GL','PYOPENGL_PLATFORM','LIBGL_ALWAYS_SOFTWARE','MESA_LOADER_DRIVER_OVERRIDE','__EGL_VENDOR_LIBRARY_FILENAMES','LP_NUM_THREADS']},rows=rows)
out.write_text(json.dumps(report,indent=2))
