"""Explicit synthetic-policy local EGL smoke; never connects to live lanes.

Use a CPU-prepared local worker directory. GPU work is only native EGL rendering.
The original benchmark is shortened explicitly; this is not an SR qualification.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def noop(port):
    import numpy as np
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    class Policy:
        def infer(self,obs):
            return {'actions':np.tile(np.asarray([0,0,0,0,0,0,-1],np.float32),(5,1)),
                    '__hit_meta__':{'source':'policy','cursor':0,'cache_hit':False}}
        def reset(self):pass
    WebsocketPolicyServer(Policy(),host='127.0.0.1',port=port,metadata={'synthetic_noop':True},
        concurrent=True,connection_policy_factory=lambda *args,**kw:Policy(),allow_dynamic_bundles=False).serve_forever()


def driver(campaign,directory):
    from exp.r17_integration.conductor.manifest import load_benchmark_spec,episode_tasks,EndpointSpec
    from exp.r17_integration.conductor.strategy import MetaWorldStrategy
    from exp.r17_integration.ops.formal_driver import probe_validator
    from exp.r17_integration.qualification.driver import ProbeRecordingDriver
    from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
    from exp.r17_integration.ops.formal_process import write_json
    from openpi.conductor.task import ServerEndpoint
    spec=load_benchmark_spec(Path(campaign['local_assets'])/'benchmark.json')
    endpoint=ServerEndpoint('ziyanglin.com',campaign['public_port'])
    tasks=episode_tasks(campaign=campaign['run_name'],arm='synthetic_noop',bundle_id='default',endpoint=endpoint,
        model_identity='synthetic_noop',norm_identity='synthetic_none',layout_identity='synthetic_H5_D7',
        spec=spec,num_steps=1,partition='smoke',names=[spec.tasks[i][0] for i in (0,10,20,30)],indices=[0,1,48])
    compatible=EndpointSpec(endpoint,'synthetic_noop','synthetic_none','synthetic_H5_D7',spec.identity,allowed_k=(1,),capacity=4)
    strategy=MetaWorldStrategy(tasks,endpoints=[compatible])
    directory=Path(directory);directory.mkdir()
    class Driver(ProbeRecordingDriver):
        def handle_result(self,result):
            record=dict(result,worker_id=getattr(self._shutdown_local,'worker_id',None),time=time.time())
            record.pop('per_step_rows',None)
            with (directory/'attempts.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            return super().handle_result(result)
    d=Driver(strategy,dispatch_path=directory/'dispatch.jsonl',rejection_path=directory/'rejected_probes.json',
        expected_workers=4,shutdown_seconds=30,yaml_weights={t.yaml_id:1 for t in tasks},servers=[endpoint],
        journal_path=str(directory/'journal.jsonl'),bind_host='127.0.0.1',bind_port=campaign['driver_port'],
        ctl_factory=lambda e:WebsocketClientPolicy('127.0.0.1',e.port),worker_probe_validator=probe_validator(spec),
        episode_timeout_s=120,scheduler_kwargs=dict(eval_concurrency=4,max_episode_retries=1))
    started=time.monotonic()
    try:
        d.run(stop=lambda:time.monotonic()-started>480)
        endings=[json.loads(line) for line in (directory/'journal.jsonl').read_text().splitlines()]
        clean=len(endings)==len(tasks) and all(not row.get('error') for row in endings)
        report=dict(status='PASS' if d.scheduler.all_done() and not d.probe_rejections and clean else 'FAIL',
            all_done=d.scheduler.all_done(),expected_episodes=len(tasks),shutdown=d.worker_shutdown,
            rejected_probes=d.probe_rejections,census=d.worker_census,elapsed_seconds=time.monotonic()-started)
        write_json(directory/'DRIVER_RECEIPT.json',report)
    finally:d.close()


def render(directory,node,admission_parent=None):
    """Native original pools + seed7 + 10 settle controls + original paired views."""
    from exp.r17_integration.ops.formal_process import write_json,ticks
    from exp.r17_integration.ops.worker_gpu import acquire
    from exp.r17_integration.ops.worker_memory import admission
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    c=dict(domain='libero',node_workers=1,worker_node=node,worker_egl_device=0,
           worker_vram_mib=1024,worker_memory_mib=4096,worker_run=str(directory))
    if node=='local':
        c.update(worker_egl_devices=[0],worker_server_vram_headroom_mib=16384,worker_memory_headroom_mib=32768)
    write_json(directory/'manifest.json',c)
    # The reference node's root filesystem can be full. Keep tiny reference
    # outputs on its home mount, but inspect and lock the native node fleet
    # admission namespace throughout this bounded diagnostic rendering.
    shared_lock=None
    if admission_parent:
        import fcntl
        from exp.r17_integration.ops import worker_gpu
        original=worker_gpu.reservations
        def reservations(run,cards,**kwargs):
            pending,evidence=original(run,cards,**kwargs)
            other,details=original(Path(admission_parent)/directory.name,cards,**kwargs)
            return {d:pending[d]+other[d] for d in cards},evidence+details
        worker_gpu.reservations=reservations
        shared_lock=(Path(admission_parent)/'.egl_admission.lock').open('a')
        fcntl.flock(shared_lock,fcntl.LOCK_EX)
    lease=acquire(c,directory)
    write_json(directory/'receipt.json',dict(status='RUNNING',workers=1,pids=[],worker_memory_mib=4096,
        fleet_owner=dict(pid=os.getpid(),start_ticks=ticks(os.getpid()))))
    started=time.monotonic()
    try:
        memory=admission(c,directory)
        if not memory['admitted']:raise RuntimeError(memory['reason'])
        assignment=lease.result['assignments'][0]
        os.environ['MUJOCO_EGL_DEVICE_ID']=str(assignment['device'])
        import numpy as np
        from libero.libero import benchmark
        from examples.libero import main as native
        # Avoid requiring a newer ops module in the read-only remote release.
        frames={};cells=[];gl_identity=None
        for suite,task,index in [('libero_spatial',0,0),('libero_object',0,48),('libero_goal',0,1)]:
            b=benchmark.get_benchmark_dict()[suite]();task_obj=b.get_task(task)
            env,prompt=native._get_libero_env(task_obj,256,7)
            try:
                state=b.get_task_init_states(task)[index]
                env.reset();obs=env.set_init_state(state)
                env.sim._render_context_offscreen.gl_ctx.make_current()
                from OpenGL import GL
                gl_identity={key:GL.glGetString(token).decode() for key,token in
                             [('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER),('version',GL.GL_VERSION)]}
                for _ in range(10):obs,_,_,_=env.step([0,0,0,0,0,0,-1])
                key=f'{suite}_{task}_{index}'
                frames[key+'_agent']=obs['agentview_image'][::-1,::-1].copy()
                frames[key+'_wrist']=obs['robot0_eye_in_hand_image'][::-1,::-1].copy()
                frames[key+'_state']=env.sim.get_state().flatten().copy()
                raw=state.detach().cpu().numpy() if hasattr(state,'detach') else np.asarray(state)
                cells.append(dict(key=key,suite=suite,task=task,init=index,prompt=prompt,
                    init_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
            finally:env.close()
        np.savez_compressed(directory/'frames.npz',**frames)
        write_json(directory/'RENDER.json',dict(status='PASS',node=node,seed=7,settle_steps=10,
            wire_size=256,cells=cells,gpu=assignment,gl=gl_identity,memory=memory,
            frames_sha256=hashlib.file_digest((directory/'frames.npz').open('rb'),'sha256').hexdigest(),
            elapsed_seconds=time.monotonic()-started))
    finally:
        lease.close();write_json(directory/'receipt.json',dict(status='CLOSED',pids=[],fleet_owner=None))
        if shared_lock:shared_lock.close()


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def smoke(prepared):
    from exp.r17_integration.ops.formal_process import write_json,ticks,stop_group,alive
    from exp.r17_integration.ops import formal_eval as f
    prepared=Path(prepared);c=read(prepared/'campaign.json');deployment=read(prepared/'WORKER_DEPLOYMENT.json')['local']
    spec=f.benchmark();spec=replace(spec,max_steps_by_suite=(20,20,20,20),protocol_id='q_local_noop_smoke_v1')
    write_json(prepared/'assets/benchmark.json',spec.manifest())
    c.update(public_port=free_port(),driver_port=free_port(),seconds=520)
    c['endpoint']='ziyanglin.com:'+str(c['public_port'])
    w=read(prepared/'local.json');w.update(c)
    from exp.r17_integration.ops.local_workers import route
    route(w,c);write_json(prepared/'local.json',w);write_json(prepared/'campaign.json',c)
    f.transfer(['tether','push',str(prepared/'assets/benchmark.json'),'local:'+deployment['stage']+'/benchmark.json'])
    env=dict(os.environ,**deployment['env'],CUDA_VISIBLE_DEVICES='')
    owned=[];handles=[];started=time.monotonic()
    def spawn(label,argv,environment):
        h=(prepared/(label+'.log')).open('wb');handles.append(h)
        child=subprocess.Popen(argv,env=environment,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
        entry=dict(label=label,pid=child.pid,pgid=child.pid,start_ticks=ticks(child.pid));owned.append((child,entry))
        write_json(prepared/'SMOKE_OWNERS.json',[e for _,e in owned]);return child
    helper=str(Path(__file__).resolve())
    status='FAIL';error=None
    try:
        server=spawn('noop',[sys.executable,helper,'--role','noop','--port',str(c['public_port'])],dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        until=time.monotonic()+30
        while True:
            try:
                with socket.create_connection(('127.0.0.1',c['public_port']),timeout=1):break
            except OSError:
                if server.poll() is not None or time.monotonic()>until:raise RuntimeError('no-op endpoint did not start')
                time.sleep(.1)
        drive=spawn('driver',[sys.executable,helper,'--role','driver','--prepared',str(prepared)],dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        time.sleep(1)
        fleet=spawn('fleet',[deployment['python'],'-u','-m','exp.r17_integration.ops.formal_workers','--campaign',str(prepared/'local.json')],env)
        while (fleet.poll() is None or drive.poll() is None) and time.monotonic()-started<550:time.sleep(.2)
        if drive.poll()!=0 or fleet.poll()!=0:raise RuntimeError(f'driver/fleet rc: {drive.poll()}/{fleet.poll()}')
        if read(prepared/'smoke_driver/DRIVER_RECEIPT.json')['status']!='PASS':
            raise RuntimeError('synthetic driver has missing/error episodes')
        status='PASS'
    except Exception as exc:error=repr(exc)
    finally:
        cleanup=[]
        for child,entry in reversed(owned):
            cleanup.append(stop_group(entry,grace=8));child.wait(timeout=15)
        for h in handles:h.close()
        report=dict(status=status,error=error,elapsed_seconds=time.monotonic()-started,
            owned=[dict(e,returncode=p.returncode) for p,e in owned],cleanup=cleanup,
            all_owned_closed=all(not alive(e) for _,e in owned))
        write_json(prepared/'SMOKE_RECEIPT.json',report);print(json.dumps(report))
    return 0 if status=='PASS' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--role',choices=('noop','driver','render','smoke'),required=True)
    p.add_argument('--port',type=int);p.add_argument('--prepared',type=Path);p.add_argument('--directory',type=Path);p.add_argument('--node',choices=('local','timan107'))
    p.add_argument('--admission-parent',type=Path)
    a=p.parse_args()
    if a.role=='noop':noop(a.port)
    elif a.role=='driver':driver(read(a.prepared/'campaign.json'),a.prepared/'smoke_driver')
    elif a.role=='render':render(a.directory,a.node,a.admission_parent)
    else:raise SystemExit(smoke(a.prepared))


if __name__=='__main__':main()
