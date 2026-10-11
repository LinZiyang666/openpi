"""Python 3.8 compatible Q context shim over the unchanged LIBERO kernel."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import tarfile
import time


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.part');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(str(temp),str(path))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--endpoint',required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--model',choices=('dp','fastwam'),required=True)
    p.add_argument('--protocol',required=True)
    p.add_argument('--benchmark-spec',required=True)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--transfer-stage',action='store_true')
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('MUJOCO_GL')=='osmesa'
    run=Path(a.run);run.mkdir(parents=True,exist_ok=True)
    report=dict(status='PARTIAL',cpu_renderer=True,renderer='osmesa',actual_simulator=not a.plan_only,
        episode_kernel='examples.libero.main._run_episode',supervisor_pid=os.getpid(),
        supervisor_start_ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],journal=[],quality_claim=False)
    save(run/'receipt.json',report)
    def interrupted(*args): raise RuntimeError('Q LIBERO independent 500-second deadline / owned interrupt')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(500)
    env=None;client=None
    started=time.monotonic()
    try:
        import numpy as np
        from examples.libero import main as original
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        np.random.seed(7)
        suite=original.benchmark.get_benchmark_dict()['libero_spatial']()
        task=suite.get_task(0)
        benchmark=json.loads(Path(a.benchmark_spec).read_text())
        assert benchmark['protocol_identity']==a.protocol
        assert (benchmark['replan_steps'],benchmark['wire_size'],benchmark['settle_steps'],benchmark['seed'],benchmark['seed_mode'])==(5,256,10,7,'paired_semantic')
        assert benchmark['max_steps_by_suite']==[220,280,300,520]
        task_name='libero_spatial/'+task.name
        assert benchmark['tasks'][0]==[task_name,task.language,'libero_spatial']
        states=suite.get_task_init_states(0)
        bddl=Path(original.get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        plans=[]
        expected='pick up the black bowl between the plate and the ramekin and place it on the plate'
        assert task.language==expected,'LIBERO spatial task0 vocabulary changed; reseal Q synthetic donors'
        for arm,idx in (('full',0),('retrieval',1)):
            initial=np.asarray(states[idx])
            data_hash=hashlib.sha256(bddl.read_bytes()+initial.tobytes()).hexdigest()
            uid=hashlib.sha256((run.name+':'+arm+':'+data_hash).encode()).hexdigest()[:24]
            plans.append(dict(campaign='Q-'+run.name,arm=arm,bundle_generation=arm+'_gen1',
                benchmark_protocol=a.protocol,task_name=task_name,orig_init_idx=idx,original_init_idx=idx,
                task_data_hash=data_hash,task_uid=uid,attempt=1,task_id=0,split='mechanics',
                kernel='examples.libero.main._run_episode',seed=7,replan_steps=5,max_policy_controls=220))
        save(run/'episode_manifest.json',dict(status='PASS',plan_only=a.plan_only,tasks=plans,
            suite='libero_spatial',original_init_indices=[0,1],two_real_cameras=True,quality_claim=False))
        if a.plan_only:
            report.update(status='PASS',plan_only=True);return 0
        lock=open('/tmp/rc365_env_build.lock','a')
        def acquire():
            until=time.monotonic()+30
            while True:
                try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return
                except BlockingIOError:
                    if time.monotonic()>until: raise RuntimeError('LIBERO context lock busy for 30 seconds')
                    time.sleep(.1)
        acquire()
        try: env,description=original._get_libero_env(task,original.LIBERO_ENV_RESOLUTION,7)
        finally: fcntl.flock(lock,fcntl.LOCK_UN)
        host,port=a.endpoint.rsplit(':',1)
        client=WebsocketClientPolicy(host,int(port))
        args=original.Args(resize_size=256,replan_steps=5,task_suite_name='libero_spatial',
            num_steps_wait=10,seed=7,save_trajectory=True,cuda_visible_devices='')
        class WrappedEnv:
            def __init__(self,inner): self.inner=inner;self.current=None
            def __getattr__(self,key): return getattr(self.inner,key)
            def reset(self,*args,**kwargs):
                acquire()
                try: return self.inner.reset(*args,**kwargs)
                finally: fcntl.flock(lock,fcntl.LOCK_UN)
            def set_init_state(self,*args,**kwargs):
                acquire()
                try: return self.inner.set_init_state(*args,**kwargs)
                finally: fcntl.flock(lock,fcntl.LOCK_UN)
            def step(self,action):
                result=self.inner.step(action)
                if self.current is not None:
                    shim.count+=1
                    row=dict(self.current,event='control',decision_seq=shim.seq-1,control_idx=shim.control_idx)
                    path=run/('control_%d_%03d.npz'%(self.current['orig_init_idx'],shim.control_idx))
                    np.savez_compressed(str(path),requested_action=np.asarray(action,dtype=np.float64),
                        observed_actuator_ctrl=np.asarray(self.inner.env.sim.data.ctrl).copy(),
                        sim_state=np.asarray(self.inner.get_sim_state()).copy())
                    row.update(block=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    with (run/'physical.jsonl').open('a') as f: f.write(json.dumps(row)+'\n');f.flush()
                    shim.control_idx+=1
                return result
        wrapped=WrappedEnv(env)
        class Shim:
            def infer(self,obs):
                req=dict(campaign=wrapped.current['campaign'],benchmark_protocol=a.protocol,task_name=task_name,
                    original_init_idx=wrapped.current['orig_init_idx'],decision_seq=self.seq,task_id=0,seed=7,attempt=1,
                    arm=wrapped.current['arm'],bundle_generation=wrapped.current['bundle_generation'])
                # Arm-independent seeds, actual native initial-state identity retained in episode metadata.
                req['explicit_seed']=int.from_bytes(hashlib.sha256(('libero:7:%s:%d:%d'%(task_name,req['original_init_idx'],self.seq)).encode()).digest()[:4],'little')
                element=dict(obs,__request__=req)
                if self.seq:
                    element['__execution__']=dict(decision_seq=self.seq-1,executed_count=self.count,terminal=False)
                got=client.infer(element)
                actions=np.asarray(got['actions'])
                assert actions.ndim==2 and actions.shape[1]==7 and np.isfinite(actions).all()
                save(run/('decision_%d_%03d.json'%(req['original_init_idx'],self.seq)),dict(request=req,hit=got['__hit_meta__']))
                self.seq+=1;self.count=0
                return got
        shim=Shim()
        for plan in plans:
            client.select_bundle(plan['bundle_generation'])
            client.episode_start(plan['campaign'],task_name,plan['orig_init_idx'],extra_metadata=dict(plan,
                ack_protocol='r17.execution.v1',partial_tail_protocol='r17.partial_tail.v1'))
            shim.seq=shim.count=shim.control_idx=0
            # Initial settle controls are deliberately outside the policy-control ledger.
            class Delayed:
                def infer(self,obs): wrapped.current=plan;return shim.infer(obs)
            wrapped.current=None
            timing={}
            outcome=original._run_episode(wrapped,Delayed(),states[plan['orig_init_idx']],description,args,220,client_timing=timing)
            assert timing['termination_reason']!='exception',timing
            success=bool(outcome[0])
            terminal=client._send_ctrl(dict(__ctrl__='episode_end',__success__=success,
                __execution__=dict(decision_seq=shim.seq-1,executed_count=shim.count,terminal=True)),expected_ack='episode_end')
            trajectory=outcome[3]
            for i,row in enumerate(trajectory):
                np.savez_compressed(str(run/('trajectory_%d_%03d.npz'%(plan['orig_init_idx'],i))),
                    **{k:v for k,v in row.items() if v is not None})
            report['journal'].append(dict(plan,status='done' if success else 'failed',accepted=True,error=None,
                success=success,n_decisions=shim.seq,applied_control_calls=shim.control_idx,
                final_decision_control_count=shim.count,final_execution_receipt=terminal.get('execution'),termination=timing['termination_reason']))
            save(run/'receipt.json',report)
        report.update(status='PASS')
    except BaseException as exc:
        report.update(status='FAIL',error=type(exc).__name__+': '+str(exc));raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if client is not None: client.close()
        if env is not None:
            acquire()
            try: env.close()
            finally: fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        report['elapsed_seconds']=time.monotonic()-started
        save(run/'receipt.json',report)
        if not a.plan_only:
            archive=run.parent/'libero.tar'
            hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
            manifest=dict(schema='r17.snapshot.v1',files=hashes,partials=[])
            def canonical(value): return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
            manifest['digest']=hashlib.sha256(canonical(manifest)).hexdigest()
            with tarfile.open(str(archive),'w') as t:
                for path in sorted(run.rglob('*')):
                    if path.is_file(): t.add(str(path),arcname=str(path.relative_to(run)),recursive=False)
                data=canonical(manifest);info=tarfile.TarInfo('TRANSFER_SHA256.json');info.size=len(data);t.addfile(info,io.BytesIO(data))
            from .archive_stage import stage_archive
            sha=hashlib.sha256(archive.read_bytes()).hexdigest()
            transfer=(stage_archive(archive,run.parent.name,'libero',sha) if a.transfer_stage else
                dict(archive=str(archive),archive_sha256=sha))
            print('Q_LIBERO_ARCHIVE '+json.dumps(transfer),flush=True)
        print(report['status'],flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
