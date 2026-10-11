"""Four independent native LIBERO workers; fixed A states, no policy/model calls."""
import argparse, hashlib, json, os, pathlib, resource, subprocess, sys, time
import xml.etree.ElementTree as ET
CASES=[('libero_spatial',0,0),('libero_object',0,0),('libero_goal',0,0),('libero_10',0,0)]

def write(path,value):
    part=path.with_suffix(path.suffix+'.part'); part.write_text(json.dumps(value,sort_keys=True)+'\n'); os.replace(part,path)

def selected_egl():
    import ctypes as c
    egl=c.CDLL('libEGL.so.1'); egl.eglGetCurrentDisplay.restype=c.c_void_p
    egl.eglGetProcAddress.argtypes=[c.c_char_p]; egl.eglGetProcAddress.restype=c.c_void_p
    display=egl.eglGetCurrentDisplay(); dev=c.c_void_p()
    query_display=c.CFUNCTYPE(c.c_uint,c.c_void_p,c.c_int,c.POINTER(c.c_void_p))(egl.eglGetProcAddress(b'eglQueryDisplayAttribEXT'))
    query_device=c.CFUNCTYPE(c.c_uint,c.c_void_p,c.c_int,c.POINTER(c.c_ssize_t))(egl.eglGetProcAddress(b'eglQueryDeviceAttribEXT'))
    out={'gpu3_egl_uuid_hex':os.environ.get('R17_GPU3_EGL_UUID_HEX'),'gpu3_egl_native_index':os.environ.get('R17_GPU3_EGL_NATIVE_INDEX'),'display_device_query_ok':bool(query_display(display,0x322C,c.byref(dev)))}
    query_string=c.CFUNCTYPE(c.c_char_p,c.c_void_p,c.c_int)(egl.eglGetProcAddress(b'eglQueryDeviceStringEXT'))
    for key,attribute in [('drm_device',0x3233),('drm_render_node',0x3377)]:
        value=query_string(dev,attribute); out[key]=value.decode() if value else None
    out['nvidia_device_nodes']={str(i):{'major':os.major(os.stat('/dev/nvidia'+str(i)).st_rdev),'minor':os.minor(os.stat('/dev/nvidia'+str(i)).st_rdev)} for i in range(4)}
    ordinal=c.c_ssize_t(-1); out['cuda_device_query_ok']=bool(query_device(dev,0x323A,c.byref(ordinal))); out['physical_cuda_ordinal']=ordinal.value
    from OpenGL.GL import glGetString, glGetIntegerv, GL_RENDERER, GL_VENDOR, GL_VERSION
    out['gl_nvx_memory_kib']={key:int(glGetIntegerv(attribute)) for key,attribute in [('dedicated',0x9047),('total_available',0x9048),('current_available',0x9049),('evicted',0x904B)]}
    out.update(renderer=glGetString(GL_RENDERER).decode(),vendor=glGetString(GL_VENDOR).decode(),version=glGetString(GL_VERSION).decode())
    return out

def child(args):
    import numpy as np
    from libero.libero import benchmark
    from examples.libero import main as native
    from openpi_client import image_tools
    out=pathlib.Path(args.out); out.mkdir(parents=True,exist_ok=True)
    ready=out/'ready.json'; write(ready,{'pid':os.getpid(),'worker':args.child})
    while not (out.parent/'GO').exists(): time.sleep(.05)
    episodes=[]; arrays={}; init=time.monotonic()
    for case_index,(suite_name,task_id,idx) in enumerate(CASES):
        suite=benchmark.get_benchmark_dict()[suite_name](); task=suite.get_task(task_id)
        state=suite.get_task_init_states(task_id)[idx]; state=state.detach().cpu().numpy() if hasattr(state,'detach') else np.asarray(state)
        started=time.monotonic(); env,prompt=native._get_libero_env(task,256,7)
        try:
            env.reset(); obs=env.set_init_state(state)
            for unused in range(10): obs,_,_,_=env.step(native.LIBERO_DUMMY_ACTION)
            gl=selected_egl()
            if case_index==0:
                write(out/'CONTEXT_LIVE.json',{'pid':os.getpid(),'egl':gl})
                time.sleep(8)
            for step in range(61):
                if step in (0,30,60):
                    prefix=f'case{case_index}_step{step}_'
                    for key in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','robot0_joint_pos','robot0_joint_vel'):
                        if key in obs: arrays[prefix+key]=np.asarray(obs[key]).copy()
                    arrays[prefix+'sim_state']=np.asarray(env.get_sim_state()).copy()
                    for key in ('agentview_image','robot0_eye_in_hand_image'):
                        for size in (96,224,256):
                            arrays[prefix+key+f'_wire{size}']=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[key][::-1,::-1]),size,size))
                if step==0: sim_start=time.monotonic()
                if step==60: break
                obs,_,_,_=env.step(native.LIBERO_DUMMY_ACTION)
            sim_seconds=time.monotonic()-sim_start
            fds=[]; fd_devices=[]
            for p in pathlib.Path('/proc/self/fd').iterdir():
                try:
                    target=os.readlink(p)
                    if 'nvidia' in target or '/dev/dri/' in target:
                        fds.append(target)
                        if target.startswith('/dev/'):
                            rdev=p.stat().st_rdev; fd_devices.append({'name':target,'major':os.major(rdev),'minor':os.minor(rdev)})
                except OSError: pass
            episode={'case':case_index,'suite':suite_name,'task_id':task_id,'original_init_idx':idx,'task_name':task.name,'prompt':prompt,'state_sha256':hashlib.sha256(state.tobytes()).hexdigest(),'steps':60,'sim_seconds':sim_seconds,'sim_steps_per_second':60/sim_seconds,'setup_settle_seconds':sim_start-started,'egl':gl,'gpu_fds':sorted(set(fds)),'gpu_fd_devices':fd_devices}
            episodes.append(episode); write(out/'progress.json',{'pid':os.getpid(),'episodes':episodes})
        finally: env.close()
    np.savez_compressed(out/'observations.npz',**arrays)
    hashes={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest(),'bytes':v.nbytes} for k,v in arrays.items()}
    usage=resource.getrusage(resource.RUSAGE_SELF)
    receipt={'status':'PASS','pid':os.getpid(),'worker':args.child,'episodes':episodes,'observations':hashes,'maxrss_mib':usage.ru_maxrss/1024,'cpu_seconds':usage.ru_utime+usage.ru_stime,'elapsed_seconds':time.monotonic()-init,'env':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID','MUJOCO_GL','PYOPENGL_PLATFORM')}}
    write(out/'RESULT.json',receipt)

def meminfo():
    rows={}
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemAvailable','AnonPages','Slab','SUnreclaim','VmallocUsed'): rows[key]=int(value.split()[0])/1024
    return rows

def parent(args):
    out=pathlib.Path(args.out); out.mkdir(parents=True,exist_ok=False)
    children=[]; logs=[]; samples=[]; baseline=meminfo(); started=time.monotonic()
    try:
        for i in range(args.workers):
            work=out/f'worker{i}'; work.mkdir(); scratch=work/'scratch'; scratch.mkdir()
            env=dict(os.environ,TMPDIR=str(scratch),XDG_CACHE_HOME=str(scratch/'cache'),HF_HOME=str(scratch/'hf_cache'))
            log=(work/'worker.log').open('wb'); logs.append(log)
            p=subprocess.Popen([sys.executable,str(pathlib.Path(__file__).resolve()),'--child',str(i),'--out',str(work),'--gpu',str(args.gpu)],env=env,cwd=scratch,stdout=log,stderr=subprocess.STDOUT)
            children.append(p)
        while not all((out/f'worker{i}'/'ready.json').exists() for i in range(args.workers)):
            if any(p.poll() is not None for p in children): raise RuntimeError('worker failed during import')
            if time.monotonic()-started>180: raise TimeoutError('worker import timeout')
            time.sleep(.2)
        worker_pids={p.pid:json.loads((out/f'worker{i}'/'ready.json').read_text())['pid'] for i,p in enumerate(children)}
        (out/'GO').write_text('synchronized start\n')
        last_gpu=0; gpus={}; aggregate_start=time.monotonic()
        while any(p.poll() is None for p in children):
            now=time.monotonic(); workers={}
            if now-last_gpu>=2:
                q=subprocess.run(['nvidia-smi','-i',('GPU-ba276422-a0b1-e371-8327-f744be942f15' if args.gpu==3 and os.uname().nodename.split('.')[0]=='timan1' else str(args.gpu)),'-q','-x'],capture_output=True,text=True,timeout=10,check=True)
                card=ET.fromstring(q.stdout).find('gpu'); gpus={int(p.findtext('pid')):int(p.findtext('used_memory').split()[0]) for p in card.findall('processes/process_info') if p.findtext('used_memory')!='N/A'}
                gpu_used=int(card.findtext('fb_memory_usage/used').split()[0]); last_gpu=now
            for p in children:
                try:
                    data={}
                    for line in pathlib.Path(f'/proc/{worker_pids[p.pid]}/smaps_rollup').read_text().splitlines():
                        if line.startswith(('Rss:','Pss:','Private_Clean:','Private_Dirty:')):
                            k,v=line.split(':',1); data[k+'_mib']=int(v.split()[0])/1024
                    data['vram_mib']=gpus.get(worker_pids[p.pid],0); workers[str(worker_pids[p.pid])]=data
                except (FileNotFoundError,ProcessLookupError): pass
            sample={'seconds':now-aggregate_start,'workers':workers,'gpu_used_mib':gpu_used,'meminfo_mib':meminfo()}; samples.append(sample)
            write(out/'MONITOR_PROGRESS.json',sample)
            if now-started>600: raise TimeoutError('bounded render probe timeout')
            time.sleep(.5)
        codes=[p.wait(timeout=5) for p in children]
        results=[json.loads((out/f'worker{i}'/'RESULT.json').read_text()) if (out/f'worker{i}'/'RESULT.json').exists() else None for i in range(args.workers)]
        totals=[]
        for p,r in zip(children,results):
            pid=worker_pids[p.pid]; mine=[s['workers'][str(pid)] for s in samples if str(pid) in s['workers']]
            totals.append({'pid':pid,'peak_rss_mib':max([m['Rss_mib'] for m in mine]+[r['maxrss_mib'] if r else 0]),'peak_pss_mib':max([m['Pss_mib'] for m in mine]+[0]),'peak_vram_mib':max([m['vram_mib'] for m in mine]+[0]),'sim_steps':sum(e['steps'] for e in r['episodes']) if r else 0,'sim_seconds':sum(e['sim_seconds'] for e in r['episodes']) if r else None})
        final={'status':'PASS' if all(c==0 for c in codes) else 'FAIL','node':os.uname().nodename,'gpu':args.gpu,'workers':args.workers,'exit_codes':codes,'resources':totals,'baseline_meminfo_mib':baseline,'final_meminfo_mib':meminfo(),'peak_gpu_used_mib':max(s['gpu_used_mib'] for s in samples),'probe_wall_seconds':time.monotonic()-aggregate_start,'elapsed_seconds':time.monotonic()-started,'samples':samples}
        write(out/'BENCHMARK.json',final); print(json.dumps({k:v for k,v in final.items() if k!='samples'}),flush=True)
        if final['status']!='PASS': raise SystemExit(2)
    finally:
        for p in children:
            if p.poll() is None: p.terminate()
        for p in children:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill(); p.wait(timeout=5)
        for log in logs: log.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--child',type=int)
    args=ap.parse_args()
    if os.uname().nodename.split('.')[0]=='timan1' and (args.gpu!=3 or os.environ.get('MUJOCO_EGL_DEVICE_ID')!='3'): raise RuntimeError('timan1 probe refuses any GPU except physical 3')
    child(args) if args.child is not None else parent(args)
