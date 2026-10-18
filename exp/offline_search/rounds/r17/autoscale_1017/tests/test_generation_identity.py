"""A real R17 driver sees two generations spawned by the resident fleet."""
import copy
import os
from types import SimpleNamespace as NS

import pytest

from exp.offline_search.rounds.r17.autoscale_1017.node import LocalGovernor, atomic, identity
from exp.offline_search.rounds.r17.autoscale_1017.policy import GIB


def test_real_driver_fleet_recycle_has_distinct_attested_worker_identity(tmp_path, monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017 import fleet
    from exp.r17_integration.ops import formal_workers, worker_disk, worker_lifetime
    from exp.r17_integration.tests.conductor.test_dispatch import setup
    from exp.r17_integration.conductor.driver import MetaWorldConductorDriver
    from exp.r17_integration.benchmarks.metaworld import MetaWorldBenchmarkAdapter
    from exp.r17_integration.conductor.manifest import validate_worker_probe
    from openpi.conductor.worker import WorkerLoop
    from openpi.conductor import protocol
    from openpi.conductor.task import EpisodeResult

    endpoints, tasks, strategy, _, _ = setup(tmp_path, names=['reach-v3'])
    driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp_path/'dispatch-real.jsonl',
        yaml_weights={tasks[0].yaml_id: 2}, servers=[endpoints[0].endpoint],
        journal_path=str(tmp_path/'accepted.jsonl'), ctl_factory=lambda e: NS(close=lambda: None),
        worker_probe_validator=validate_worker_probe)
    for stage in driver.scheduler.pending_setups():
        driver.scheduler.mark_setup_running(stage.stage_id)
        driver.scheduler.mark_setup_done(stage.stage_id)
    clock=[0.]
    monkeypatch.setattr(fleet.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(fleet.time, 'time', lambda: 1700000000+clock[0])
    monkeypatch.setattr(fleet.time, 'sleep', lambda n: clock.__setitem__(0, clock[0]+n))
    monkeypatch.setattr(worker_disk, 'admission', lambda *a: dict(admitted=True))
    monkeypatch.setattr(worker_lifetime, 'planned_recycle', lambda entry, code: code==75)
    monkeypatch.setenv('OPENPI_WORKER_OBSERVER','exp.offline_search.rounds.r17.autoscale_1017.worker:Observer')
    monkeypatch.setenv('OPENPI_AUTOSCALE_MAX_EPISODES','32')
    monkeypatch.delenv('R17_WORKER_RECYCLE_RECEIPT',raising=False)
    class Slots:
        def __init__(self,*a):pass
        def view(self):return None
        def acquire(self,i):return True
        def release(self,i):pass
    monkeypatch.setattr(fleet,'Slots',Slots)
    c=dict(run_name='fa_ls_fw_r_p_m',domain='metaworld',worker_node='timan107',
        node_workers=1,autoscale={},seconds=20,worker_max_episodes=32,
        worker_egl_device=0,endpoint='local:1',driver_host='localhost',driver_port=1,
        worker_benchmark='fixture.json')
    governor=LocalGovernor(c,tmp_path/'worker-run',ledger=tmp_path/'ledger',
        clock=lambda:clock[0],reader=lambda:(80*GIB,{}),process_reader=lambda *a:(3*GIB,4*GIB))
    atomic(governor.history_path,[dict(peak=4*GIB,complete=True)])
    atomic(governor.control_path,dict(campaign=c['run_name'],time=1700000000,targets={'timan107':1}))
    frames=[]; replies=[]; loops=[]; entries=[]; received=[]; logs={}
    peer=('127.0.0.1',1234)
    def send(sock,kind,payload):
        if kind==protocol.MSG_PULL:
            frames.append(copy.deepcopy(payload))
            driver._record_pull(payload,peer)
            replies.append(driver.handle_pull(payload['server_host']))
        elif kind==protocol.MSG_REPORT_RESULT:
            received.append(payload['task_uid']);driver.handle_result(payload)
    monkeypatch.setattr(protocol,'send_message',send)
    monkeypatch.setattr(protocol,'recv_message',lambda sock:replies.pop(0))
    class Runner:
        def run(self,task,report):return EpisodeResult(task.task_uid,True,1,attempt=task.attempt)
        def close(self):pass
    class Child:
        pid=os.getpid()
        def __init__(self):self.born=clock[0]
        def poll(self):return 75 if clock[0]>=self.born+2 else None
        def wait(self,timeout):return self.poll()
    def launch(index,generation):
        slot,argv=formal_workers.worker_command(c,index)
        probe=MetaWorldBenchmarkAdapter().worker_probe(
            dict(renderer='egl',GL_VENDOR='NVIDIA fixture',pid=638357+116*generation,
                 gpu_uuid='GPU-pinned'),dict(task_data_identity='fixture'))
        loop=WorkerLoop(slot,c['endpoint'],Runner(),connect=lambda:None,probe=probe)
        loops.append(loop)
        task=loop._pull(None)  # BEFORE FIX: second generation fails here, renderer.pid differs.
        loop._run_one(None,task)
        child=Child();logs[index]=NS(close=lambda:None)
        entry=dict(pid=child.pid,start_ticks=identity(child.pid),worker_id=slot,index=index,generation=generation,argv=argv)
        entries.append(entry)
        return child,entry
    report=dict(worker_failures=[],worker_recycles=[])
    try:
        fleet.run_fleet(c,governor.run,report,[],logs,launch,lambda:False,lambda:len(loops)>=2,governor)
        assert report['status']=='PASS' and len(report['worker_recycles'])==2
        assert entries[0]['worker_id']==entries[1]['worker_id']  # stable supervisor slot
        assert frames[0]['worker_id']!=frames[1]['worker_id']  # independent driver identities
        assert all(p['server_host']==c['endpoint'] for p in frames)
        assert len(driver.worker_census)==2 and not driver.rejected_worker_probes
        assert set(received)=={t.task_uid for t in tasks} and len(received)==len(tasks)
        # Reconnect keeps one generation identity and immutable attestation.
        driver._record_pull(frames[1],('127.0.0.1',1235))
        assert loops[1]._worker_id==frames[1]['worker_id']
        for key,value in [('model_identity','wrong'),('runtime_identity','wrong'),('renderer',{'gpu_uuid':'wrong'})]:
            changed=copy.deepcopy(frames[1]);changed['probe'][key]=value
            with pytest.raises(ValueError):driver._record_pull(changed,peer)
        with pytest.raises(ValueError,match='endpoint changed'):
            driver._record_pull(dict(frames[1],server_host='wrong:2'),peer)
        changed=copy.deepcopy(frames[1]);changed['probe']['renderer']['pid']+=1
        with pytest.raises(ValueError,match='probe changed'):driver._record_pull(changed,peer)
        unattested={k:v for k,v in frames[1].items() if k!='probe'}
        unattested['worker_id']='new-generation-without-probe'
        with pytest.raises(ValueError,match='lacks probe'):driver._record_pull(unattested,peer)
        changed=copy.deepcopy(frames[1]);changed['worker_id']='new-generation-with-wrong-gpu'
        changed['probe']['renderer']['GL_VENDOR']='Mesa'
        with pytest.raises(ValueError,match='renderer unqualified'):driver._record_pull(changed,peer)
    finally:driver.close()


@pytest.mark.parametrize('pipeline',['R15','R16'])
def test_real_legacy_agent_respawn_keeps_slot_and_uses_new_census_identity(tmp_path,monkeypatch,pipeline):
    from exp.offline_search.rounds.r17.autoscale_1017 import fleet,legacy
    from openpi.conductor.agent import WorkerAgent,WorkerSpec
    from openpi.conductor.worker import WorkerLoop
    from openpi.conductor.driver import ConductorDriver
    from openpi.conductor import protocol
    from exp.offline_search.rounds.r16.test_conductor import fixture_group,driver
    group=fixture_group(tmp_path);d=driver(group)
    group['pool_digests']={'libero_10':{'task':'sha-pinned'}}
    probe=dict(hostname='timan107',num_steps_wait=10,replan_steps=5,resize_size=224,
        max_steps={'libero_10':520,'libero_spatial':220},per_task_digests=group['pool_digests'])
    record=d._record_pull if pipeline=='R16' else lambda p,peer:ConductorDriver._record_pull(d,p,peer)
    monkeypatch.setenv('OPENPI_WORKER_GOVERNOR','exp.offline_search.rounds.r17.autoscale_1017.legacy')
    monkeypatch.setenv('OPENPI_WORKER_MEMORY_GUARD','1')
    monkeypatch.setenv('OPENPI_AUTOSCALE_LEDGER',str(tmp_path/'ledger'))
    monkeypatch.setenv('OPENPI_AUTOSCALE_RUN',str(tmp_path/'agent'))
    monkeypatch.setattr(legacy.AgentGovernor,'supervise',lambda self:None)
    class Slots:
        def __init__(self,*a):self.assignments={}
        def view(self):return None
        def acquire(self,i):self.assignments[i]=dict(device=0,uuid='GPU-pinned');return True
        def release(self,i):self.assignments.pop(i,None)
    monkeypatch.setattr(fleet,'Slots',Slots)
    frames=[];loops=[]
    def send(sock,kind,payload):frames.append(copy.deepcopy(payload));record(payload,('127.0.0.1',1234))
    monkeypatch.setattr(protocol,'send_message',send)
    monkeypatch.setattr(protocol,'recv_message',lambda sock:(protocol.MSG_ASSIGN,{'none':True}))
    class Child:
        pid=os.getpid();code=None
        def poll(self):return self.code
        def wait(self,timeout):return self.code
    def spawn(spec,*a):
        with monkeypatch.context() as context:
            for k,v in spec.env.items():context.setenv(k,str(v))
            # Native LIBERO has no PID field. Include one to ensure a future
            # process-specific attestation does not collide across generations.
            p=dict(probe,renderer={'pid':638357+len(loops),'gpu_uuid':'GPU-pinned'})
            loop=WorkerLoop(spec.worker_id,spec.server_key,object(),connect=lambda:None,probe=p)
            loops.append(loop);loop._pull(None)
        return Child()
    a=WorkerAgent([WorkerSpec('slot0','a:1','0')],'localhost',1,spawn_fn=spawn)
    g=a._governor.local;clock=[0.];g.clock=lambda:clock[0]
    g.reader=lambda:(80*GIB,{})
    atomic(g.history_path,[dict(peak=4*GIB,complete=True)])
    atomic(g.control_path,dict(campaign=g.campaign['run_name'],time=__import__('time').time(),targets={g.node:1}))
    a.start();assert len(loops)==1
    a._handles['slot0'].code=75;clock[0]=2;a.supervise_once()
    clock[0]=4;a.supervise_once()  # refresh admission after the old child is reaped
    assert len(loops)==2 and frames[0]['worker_id']!=frames[1]['worker_id']
    assert set(a._handles)=={'slot0'} and set(g.entries)=={'slot0'}
    assert len(d.worker_census)==2 and not any(v.get('probe_conflict') for v in d.worker_census.values())
    record(frames[1],('127.0.0.1',1235))  # reconnect, same generation
    atomic(g.snapshot_path,dict(retired=['slot0']))
    with monkeypatch.context() as context:
        context.setenv('OPENPI_AUTOSCALE_NODE',str(g.snapshot_path))
        assert loops[1]._observer.payload()['retire'] is True
    if pipeline=='R16':
        bad=copy.deepcopy(frames[1]);bad['probe']['per_task_digests']={}
        with pytest.raises(ValueError,match='attestation mismatch'):record(bad,('127.0.0.1',1236))
