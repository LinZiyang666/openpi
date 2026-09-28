"""Local synthetic tests only: loopback ephemeral ports, <=2 processes, no LIBERO."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from . import telemetry
from .stream_protocol import attempt_key, request
from .stream_receiver import Receiver, Store, digest
from .stream_sink import StreamSink
from .stream_collect import certify_tree, cleanup_command, collect, replay_spills, verify_archive, verify_arm
from .test_matrix import PREFIX

TOKEN = "a"*64
HERE = Path(__file__).resolve().parent


def setup(path, arms=("test",)):
    path.mkdir(parents=True, exist_ok=False)
    (path/"arms.json").write_text(json.dumps([dict(arm=x) for x in arms]))
    return path


def receiver(run, **kw):
    server = Receiver(("127.0.0.1", 0), Store(run), TOKEN, **kw)
    port = server.server_address[1]
    assert not 23100 <= port <= 23199
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, "127.0.0.1:"+str(port)


def stop(server, thread):
    server.shutdown(); server.server_close(); thread.join(3)


def line(uid, event, **kw):
    return json.dumps(dict(schema="r6p3.client.v2", task_uid=uid, attempt=1, ev=event, **kw), separators=(",", ":"))+"\n"


def journal(run, arm, uids):
    path = run/"runs"/arm/"client/journal.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(task_uid=u, attempt=1, accepted=True, status="done", success=True, error=None))+"\n" for u in uids))


def tree(root):
    return {str(f.relative_to(root)): (f.stat().st_size, digest(f)) for f in root.rglob("*") if f.is_file()}


def adapter_episode(module, directory):
    class Env:
        def __init__(self):
            self.env = self
            self.timestep, self.cur_time = 0, 0.
            self.state = np.zeros(12); self.robots = []
            self.sim = NS(data=NS(ctrl=np.zeros(7), qpos=self.state, qvel=np.zeros(12)), step=lambda: None)
        def get_sim_state(self): return self.state.copy()
        def set_init_state(self, state):
            self.state[:] = state
            return {"robot0_gripper_qpos": self.state[6:8].copy()}
        def step(self, action):
            self.sim.data.ctrl[:] = np.clip(action,-1,1)*2
            for _ in range(3): self.sim.step()
            self.state[:7] += action; self.timestep += 1; self.cur_time += .05
            return {"robot0_gripper_qpos":self.state[6:8].copy()},0.,self.timestep==7,{}
    class Inner:
        n=0
        def episode_start(self, **kw): pass
        def infer(self, obs):
            n=self.n; self.n+=1
            return dict(actions=np.full((10,7),.125,np.float32), __p3__=dict(p3_version=2,
                p3_anchor=n==0,p3_step=n,parent_anchor=0,source="cache" if n==0 else "cache_blind",commit_controls=10))
    random.seed(7); np.random.seed(7)
    c=module.Client(Inner(),directory)
    c.episode_start(task="test",extra_metadata=dict(task_uid="test:eval:0:0",attempt=1))
    env=Env(); c.env=env; c.max_env_timestep=10
    tap=module.EnvTap(env,c); tap.set_init_state(np.zeros(12))
    obs={"observation/image":np.zeros((2,2,3),np.uint8),"observation/wrist_image":np.zeros((2,2,3),np.uint8),"observation/state":np.zeros(8),"prompt":"test"}
    for n in (5,2):
        r=c.infer(obs)
        for action in r['actions'][:n]: tap.step(action.tolist())
    c.emit(dict(ev="rollout_end",success=True,controls=7,final_chunk_completed=2))
    c.file.close()
    assert np.array_equal(env.state[:7],np.full(7,.875))
    return c


def test_parity(out):
    run=setup(out/"parity")
    server,thread,address=receiver(run,drop_once=True)
    original=HERE/'client_bundle/payload/exp/offline_search/rounds/r06/p3_profiling/telemetry.py'
    spec=importlib.util.spec_from_file_location(telemetry.__package__+'.telemetry_reference',original)
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    try:
        with patch.dict(os.environ,{},clear=True), patch.object(time,'perf_counter',return_value=100.), patch.object(time,'time_ns',return_value=123), patch.object(time,'monotonic_ns',return_value=456):
            adapter_episode(old,out/'reference')
            adapter_episode(telemetry,out/'file_mode')
            with patch.dict(os.environ,dict(P3_STREAM=address,P3_STREAM_TOKEN=TOKEN,P3_STREAM_RUN=run.name,P3_STREAM_ARM='test')):
                client=adapter_episode(telemetry,out/'remote')
        expected=tree(out/'reference')
        assert tree(out/'file_mode')==expected
        assert tree(run/'runs/test/client_telemetry')==expected
        assert not (out/'remote').exists() and not client.stream.spilled
        assert client.stream.stats['reconnects']>=2
        journal(run,'test',['test:eval:0:0'])
        verified=verify_arm(run,'test')
        # Duplicate data after finalization is idempotent; conflicts are refused.
        key=attempt_key('test:eval:0:0',1)
        data=(run/'runs/test/client_telemetry'/key/'controls.jsonl').read_bytes()[:1024]
        h=dict(op='data',run=run.name,arm='test',uid='test:eval:0:0',attempt=1,key=key,file='controls.jsonl',offset=0,token=TOKEN)
        request(address,h,data); request(address,h,data)
        rejected=0
        for header,body in [({**h,'file':'../escape'},data),({**h,'arm':'../test'},data),({**h,'key':'../x'},data),
                            ({**h,'run':'wrong'},data),({**h,'offset':999999999},data),({**h,'token':'bad'},data),(h,b'X'+data[1:])]:
            try: request(address,header,body)
            except RuntimeError: rejected+=1
            else: raise AssertionError('invalid frame accepted')
        assert tree(run/'runs/test/client_telemetry')==expected
        # Missing receipt/corrupt bytes both prevent collection certification.
        receipt=run/'runs/test/p3_stream'/key/'complete.json'
        saved=receipt.read_bytes(); receipt.unlink()
        try: verify_arm(run,'test')
        except FileNotFoundError: pass
        else: raise AssertionError('missing completion accepted')
        receipt.write_bytes(saved)
        certify_tree(run,'test')
        assert tree(run/'runs/test/client_telemetry')==expected
        trace=run/'runs/test/client_telemetry'/key/'controls.jsonl'
        original_bytes=trace.read_bytes();trace.write_bytes(original_bytes+b'corrupt')
        try: verify_arm(run,'test')
        except ValueError: pass
        else: raise AssertionError('corrupt committed file accepted')
        trace.write_bytes(original_bytes)
        return dict(files=len(expected),bytes=sum(v[0] for v in expected.values()),old_file_new_file_stream_equal=True,
                    reconnects=client.stream.stats['reconnects'],duplicate_frames=2,invalid_frames_rejected=rejected,verified=verified)
    finally: stop(server,thread)


def child_receiver(run, port, ready):
    token=run/'token'; token.write_text(TOKEN)
    cmd=PREFIX+['-m','exp.offline_search.rounds.r06.p3_profiling.stream_receiver','--run-root',str(run),'--bind','127.0.0.1','--port',str(port),'--token-file',str(token),'--ready-file',str(ready)]
    log=ready.with_suffix('.log').open('w')
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    for _ in range(500):
        if ready.exists():
            address=json.loads(ready.read_text())['port']
            assert not 23100<=address<=23199
            return p,log,address
        if p.poll() is not None: raise RuntimeError('receiver exited: '+ready.with_suffix('.log').read_text())
        time.sleep(.01)
    p.kill();p.wait();log.close();raise RuntimeError('receiver ready timeout')


def test_restart(out):
    run=setup(out/'restart')
    p,log,port=child_receiver(run,0,out/'ready1.json')
    uid='test:eval:0:1';expected=[]
    sink=StreamSink('127.0.0.1:'+str(port),TOKEN,run.name,'test',uid,1,out/'restart_remote',fail_seconds=5,timeout=.2)
    def write(s): expected.append(s);sink.write(s)
    try:
        write(line(uid,'attempt_start'))
        for i in range(50): write(line(uid,'control',i=i,data='1234567890'*1000))
        limit=time.monotonic()+5
        while sink.stats['frames_acked']<30 and time.monotonic()<limit: time.sleep(.01)
        assert sink.stats['frames_acked']>=30
        # Kill only this explicitly spawned test receiver, not any shared process.
        p.kill();p.wait();log.close()
        for i in range(50,100): write(line(uid,'control',i=i,data='9876543210'*1000))
        time.sleep(.25)
        p,log,_=child_receiver(run,port,out/'ready2.json')
        snapshot=io.BytesIO()
        np.savez_compressed(snapshot,data=np.random.default_rng(0).integers(0,256,600000,dtype=np.uint8))
        sink.write_file('step_000000.npz',snapshot.getvalue())
        write(line(uid,'rollout_end',success=True));sink.close()
        assert not sink.spilled
        got=run/'runs/test/client_telemetry'/attempt_key(uid,1)/'controls.jsonl'
        assert got.read_bytes()==''.join(expected).encode()
        assert (got.parent/'step_000000.npz').read_bytes()==snapshot.getvalue()
        journal(run,'test',[uid]);v=verify_arm(run,'test')
        return dict(killed_mid_episode=True,restarted_same_ephemeral_port=True,bytes=got.stat().st_size,
                    reconnects=sink.stats['reconnects'],snapshot_bytes=len(snapshot.getvalue()),verified=v)
    finally:
        sink.close()
        if p.poll() is None: p.terminate();p.wait(timeout=5)
        log.close()


def test_spill(out):
    run=setup(out/'spill')
    server,thread,address=receiver(run,delay=.04)
    uid='test:eval:0:2'; expected=[]
    remote=out/'spill_remote'
    sink=StreamSink(address,TOKEN,run.name,'test',uid,1,remote,queue_bytes=64*1024,timeout=.2,close_seconds=2)
    try:
        for i in range(100):
            row=line(uid,'attempt_start' if i==0 else 'control',i=i,data='x'*8192)
            expected.append(row);sink.write(row)
        row=line(uid,'rollout_end',success=True);expected.append(row);sink.write(row);sink.close()
        assert sink.spilled and sink.stats['spill_reason']=='bounded_queue_full'
    finally: stop(server,thread)
    archive=out/'spills.tar'
    with tarfile.open(archive,'w') as tf: tf.add(remote,arcname='p3_telemetry')
    journal(run,'test',[uid])
    report=collect(run,'test',archive,no_remote=True)
    assert report['spills_replayed']==1
    assert replay_spills(run,'test',run/'runs/test/client_spills')==1
    got=run/'runs/test/client_telemetry'/attempt_key(uid,1)/'controls.jsonl'
    assert got.read_bytes()==''.join(expected).encode()
    # Cleanup scripts are executed only after substituting exclusively owned
    # temporary paths, never the delivered remote paths and never tether.
    proof=verify_archive(archive,run/'runs/test/client_spills')
    cmd=cleanup_command(run,'test',proof,digest(archive))
    assert 'rm -r -- "$root"' in cmd and 'rm -- "$archive"' in cmd and 'rm -f' not in cmd
    from .stream_collect import remote_paths
    parent,tarpath=remote_paths(run,'test')
    cleanup_root=out/'cleanup_copy';cleanup_root.mkdir()
    from .collect_client import unpack
    unpack(archive,cleanup_root/'p3_telemetry')
    cleanup_tar=out/'cleanup_copy.tar';cleanup_tar.write_bytes(archive.read_bytes())
    local=cmd.replace(parent,str(cleanup_root)).replace(tarpath,str(cleanup_tar))
    victim=next((cleanup_root/'p3_telemetry').rglob('stream.p3spill'))
    before=victim.read_bytes();victim.write_bytes(before+b'changed')
    assert subprocess.run(['bash','-c',local],capture_output=True).returncode!=0
    assert victim.exists() and cleanup_tar.exists()
    victim.write_bytes(before)
    subprocess.run(['bash','-c',local],check=True)
    assert not (cleanup_root/'p3_telemetry').exists() and not cleanup_tar.exists()
    return dict(bytes=got.stat().st_size,slow_receiver=True,bounded_queue_spill=True,queue_high_water=sink.stats['queue_high_water'],
                idempotent_replay=True,mutated_remote_tree_blocks_cleanup=True,verified_cleanup_local_fixture=True)


def test_timeout_spill(out):
    run=setup(out/'timeout')
    server,thread,address=receiver(run)
    stop(server,thread)
    uid='test:eval:0:3';root=out/'timeout_remote'
    sink=StreamSink(address,TOKEN,run.name,'test',uid,1,root,fail_seconds=.15,timeout=.1,close_seconds=2)
    expected=line(uid,'attempt_start')+line(uid,'rollout_end',success=True)
    sink.write(expected);sink.close()
    assert sink.spilled and sink.stats['spill_reason']!='bounded_queue_full'
    assert replay_spills(run,'test',root)==1
    path=run/'runs/test/client_telemetry'/attempt_key(uid,1)/'controls.jsonl'
    assert path.read_bytes()==expected.encode()
    journal(run,'test',[uid]);verify_arm(run,'test')
    return dict(prolonged_failure_spill=True,reason=sink.stats['spill_reason'],bytes=path.stat().st_size)


def test_close_spill(out):
    run=setup(out/'close_deadline')
    server,thread,address=receiver(run,delay=.5)
    uid='test:eval:0:4';root=out/'close_remote'
    sink=StreamSink(address,TOKEN,run.name,'test',uid,1,root,timeout=2,close_seconds=.05)
    expected=line(uid,'attempt_start')+line(uid,'rollout_end',success=True)
    try:
        sink.write(expected);sink.close()
        assert sink.spilled and sink.stats['spill_reason']=='episode_close_deadline' and not sink.thread.is_alive()
        assert replay_spills(run,'test',root)==1
        journal(run,'test',[uid]);verify_arm(run,'test')
    finally:stop(server,thread)
    return dict(close_deadline_spill=True,sender_stopped=True,replay_with_live_receiver=True)


def test_multiarm(out):
    run=setup(out/'multiarm',('left','right'))
    server,thread,address=receiver(run)
    def worker(arm):
        uid=arm+':eval:0:0'
        sink=StreamSink(address,TOKEN,run.name,arm,uid,1,out/(arm+'_remote'))
        sink.write(line(uid,'attempt_start'))
        for i in range(48):sink.write(line(uid,'control',i=i))
        sink.write(line(uid,'rollout_end',success=True));sink.close()
        assert not sink.spilled
        journal(run,arm,[uid]);return verify_arm(run,arm)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:checks=list(pool.map(worker,('left','right')))
    finally:stop(server,thread)
    return dict(arms=2,attempts=sum(r['accepted_attempts'] for r in checks),lines=100)


def benchmark(out, paced):
    run=setup(out/('paced' if paced else 'burst'))
    server,thread,address=receiver(run)
    workers=20
    target=35_000_000 if paced else 2_000_000
    rate=3_500_000. if paced else None
    barrier=threading.Barrier(workers)
    def worker(i):
        uid='test:eval:0:'+str(i)
        sink=StreamSink(address,TOKEN,run.name,'test',uid,1,out/('bench_remote_'+str(paced)),close_seconds=30)
        times=[];nbytes=0;expected=hashlib.sha256()
        barrier.wait();start=time.monotonic()
        def write(s):
            nonlocal nbytes
            b=s.encode();expected.update(b);nbytes+=len(b)
            t=time.perf_counter_ns();sink.write(s);times.append(time.perf_counter_ns()-t)
        write(line(uid,'attempt_start'))
        payload='0123456789abcdef'*2048
        for j in range((target-1000)//(len(payload)+150)):
            write(line(uid,'control',control=j,data=payload))
            if rate:
                remaining=start+nbytes/(rate/workers)-time.monotonic()
                if remaining>0: time.sleep(remaining)
        write(line(uid,'rollout_end',success=True));sink.close()
        elapsed=time.monotonic()-start
        got=run/'runs/test/client_telemetry'/attempt_key(uid,1)/'controls.jsonl'
        assert not sink.spilled and digest(got)==expected.hexdigest()
        return dict(bytes=nbytes,seconds=elapsed,enqueue_ns=times,stats=sink.stats)
    started=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool: results=list(pool.map(worker,range(workers)))
        elapsed=time.monotonic()-started
        samples=sorted(t for r in results for t in r.pop('enqueue_ns'))
        journal(run,'test',['test:eval:0:'+str(i) for i in range(workers)])
        v=verify_arm(run,'test')
        return dict(workers=workers,bytes=sum(r['bytes'] for r in results),seconds=elapsed,
                    MB_per_second=sum(r['bytes'] for r in results)/elapsed/1e6,
                    target_MB_per_second=rate/1e6 if rate else None,
                    per_worker_episode_MB=target/1e6,enqueue_samples=len(samples),
                    enqueue_us=dict(p50=samples[len(samples)//2]/1000,p95=samples[int(len(samples)*.95)]/1000,
                                    p99=samples[int(len(samples)*.99)]/1000,max=max(samples)/1000),
                    queue_high_water_max=max(r['stats']['queue_high_water'] for r in results),
                    spill_workers=0,verification=v)
    finally: stop(server,thread)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--paced',action='store_true')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if a.paced:
        report=dict(paced=benchmark(a.out,True))
    else:
        report=dict(parity=test_parity(a.out),restart=test_restart(a.out),spill=test_spill(a.out),timeout_spill=test_timeout_spill(a.out),
                    close_spill=test_close_spill(a.out),multiarm=test_multiarm(a.out),burst=benchmark(a.out,False))
    report.update(PASS=True,evidence='synthetic CPU loopback only; no simulator, protected ports, remote host or tether')
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
