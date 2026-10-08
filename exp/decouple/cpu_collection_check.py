"""Recorded B observations through deployed worker serialization, relay and collector.

Transport doubles exchange the actual serialized frames in memory. No listener,
model, GPU, or remote operation is created. Captured frames are retained for replay.
"""
import asyncio
import io
import json
from pathlib import Path
import time

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi.conductor.task import EpisodeTask
from examples.libero.episode_runner import _episode_extra_metadata

from exp.decouple.collect_gpu import collect
from exp.decouple.latency_proxy import relay
from exp.decouple.package_config import write_json, sha
from exp.decouple.recorded import load, observation


def capture(config, model, path):
    case = config['cases'][model]
    name = case['row']['arm']
    def task(uid, attempt):
        return EpisodeTask(uid,name,'eval','libero_spatial',0,0,0,'fixture',23100,name,
                           attempt=attempt,extra=dict(num_trials_per_task=50,init_pool='B'))
    uids = [name+':warmup:0:0', name+':eval:0:0',name+':eval:1:0']
    data = load(model,limit=1)
    class Recorder:
        def __init__(self):
            self.frames=[]
        def send(self, frame):
            self.frames.append(frame)
        def recv(self):
            request=msgpack_numpy.unpackb(self.frames[-1])
            return msgpack_numpy.Packer().pack({'__ack__':'ok'} if '__ctrl__' in request else {'actions':np.zeros((10,7),np.float32)})
    def connection(episodes):
        client=WebsocketClientPolicy.__new__(WebsocketClientPolicy)
        client._packer=msgpack_numpy.Packer()
        client._ws=Recorder()
        for uid,attempt in episodes:
            t=task(uid,attempt)
            client.episode_start(t.experiment,extra_metadata=_episode_extra_metadata(t))
            client.infer(observation(data,0))
            client.episode_end(True)
        return client._ws.frames
    captured=dict(schema='decouple.worker_frames.v1',model=model,recorded_inputs=str(data.image_source),
                  connections=[connection([(uids[0],1),(uids[1],1),(uids[1],2)]),connection([(uids[2],1)])],
                  measured_uids=uids[1:],warmup_uid=uids[0])
    Path(path).write_bytes(msgpack_numpy.Packer().pack(captured))
    return captured


async def replay(captured, log):
    class Socket:
        def __init__(self, inbound):
            self.inbound=list(inbound)
            self.sent=[]
        async def send(self, frame):
            await asyncio.sleep(0)
            self.sent.append(frame)
        async def recv(self):
            await asyncio.sleep(0)
            return self.inbound.pop(0)
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self.inbound:
                raise StopAsyncIteration
            return await self.recv()
    pack=msgpack_numpy.Packer().pack
    async def one(frames):
        replies=[pack({'concurrent':True})]+[pack({'__ack__':'ok'} if '__ctrl__' in msgpack_numpy.unpackb(frame)
                                               else {'actions':np.zeros((10,7),np.float32)}) for frame in frames]
        front,back=Socket(frames),Socket(replies)
        await relay(front,back,log)
        assert front.sent==replies and back.sent==frames
    await asyncio.gather(*(one(frames) for frames in captured['connections']))


def check(config, root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    completed={}
    for model in ('groot','pi05'):
        out=root/model
        out.mkdir(exist_ok=True)
        capture(config,model,out/'worker_frames.msgpack')
        # Replay the retained recording, not the in-memory construction.
        captured=msgpack_numpy.unpackb((out/'worker_frames.msgpack').read_bytes())
        before=time.monotonic()
        log=io.StringIO()
        asyncio.run(replay(captured,log))
        after=time.monotonic()
        (out/'latencies.jsonl').write_text(log.getvalue())
        rows=[json.loads(line) for line in log.getvalue().splitlines()]
        assert all(row['uid'] is not None and row['attempt'] is not None for row in rows)
        a,b=captured['measured_uids']
        journal=[dict(task_uid=captured['warmup_uid'],attempt=1,status='done',success=True,accepted=True),
                 dict(task_uid=a,attempt=1,status='done',success=True,accepted=False),
                 dict(task_uid=a,attempt=2,status='done',success=True,accepted=True),
                 dict(task_uid=b,attempt=1,status='done',success=False,accepted=True)]
        (out/'journal.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in journal))
        write_json(out/'measurement.json',dict(start=before,end=after,manifest_sha256='cpu_fixture',measured_uids=[a,b]))
        write_json(out/'DONE.json',dict(accepted=2,manifest_sha256='cpu_fixture'))
        write_json(out/'startup.json',dict(model_processes=1,gpu_name='CPU stub',gpu_memory_peak_mb=100))
        (out/'resource_samples.jsonl').write_text(json.dumps(dict(t=(before+after)/2,gpu_memory_mb=100,
            gpu_util_pct=50,cpu_proc_pct=200,system_cpu_pct=10))+'\n')
        arm=dict(name='cpu_fixture_'+model,logical_arm='current_1',model=model,model_processes=1,fronts=None,
                 manifest_sha256='cpu_fixture',expected_episodes=2,gpu_memory_budget_mb=46000)
        write_json(out/'arm.json',arm)
        result=collect(out)
        assert result['episodes_completed']==2 and result['latency_count']==2 and result['episode_failures']==1
        completed[model+'/initial/current_1']=result
        write_json(out/'metrics.json',result)
    write_json(root/'state.json',dict(package_id='CPU recorded worker replay',status='CPU stubs passed',expected_episodes=2,
                                    completed=completed,selected={},gates={}))
    from exp.decouple.summarize_gpu import summarize
    summarize(root/'state.json',root)
    report=dict(passed=True,models=['groot','pi05'],recorded_worker_frames=True,interleaved_connections=2,
                warmup_excluded=True,rejected_attempt_excluded=True,collector='same as run_arm and offline re-analysis',
                summary=str(root/'SUMMARY.md'),gpu_used=False,listeners_created=False,remote_operations=0)
    write_json(root/'validation.json',report)
    return report
