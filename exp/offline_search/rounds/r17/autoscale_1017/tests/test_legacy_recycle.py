"""Reproduce the 4 zero-step errors through the actual legacy runner/worker."""
import copy
import os
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from exp.offline_search.rounds.r17.autoscale_1017.node import atomic, read
from exp.offline_search.rounds.r17.autoscale_1017.policy import GIB

@pytest.mark.parametrize('pipeline',['R15','R16'])
def test_legacy_agent_real_runner_completes_four_before_recycle(tmp_path,monkeypatch,pipeline):
    from openpi.conductor.agent import WorkerAgent,WorkerSpec
    from openpi.conductor.worker import WorkerLoop
    from openpi.conductor import protocol
    from exp.offline_search.rounds.r17.autoscale_1017 import fleet,legacy
    from tests.conductor.test_episode_runner import _make_runner,_ep
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
    reports=[];generations=[];clients=[]
    def send(sock,kind,payload):
        if kind==protocol.MSG_REPORT_RESULT:reports.append(copy.deepcopy(payload))
    monkeypatch.setattr(protocol,'send_message',send)
    def spawn(spec,*a):
        with monkeypatch.context() as m:
            for k,v in spec.env.items():m.setenv(k,str(v))
            made={};runner=_make_runner(made);clients.append(made)
            loop=WorkerLoop(spec.worker_id,spec.server_key,runner,connect=lambda:None)
            assert loop._observer.completed==0
            assert loop._observer.stop_requested() is False
            start=len(reports)
            for i in range(4):
                if i<3:loop._run_one(None,_ep(idx=i+len(generations)*4))
                else:
                    with pytest.raises(SystemExit) as recycle:loop._run_one(None,_ep(idx=i+len(generations)*4))
                    assert recycle.value.code==75
            receipt=read(Path(spec.env['R17_WORKER_RECYCLE_RECEIPT']))
            assert receipt['completed_episodes']==receipt['maximum_episodes']==4
            generations.append((loop,receipt))
            # RED: old Observer lambda calls episode_setup(ServerEndpoint), four
            # AttributeErrors, zero simulation steps, then the same N4 marker.
            assert all(p['error'] is None and p['n_steps']==2 for p in reports[start:])
            runner.close()
        return NS(pid=os.getpid(),poll=lambda:75,wait=lambda timeout:75)
    agent=WorkerAgent([WorkerSpec('w00','127.0.0.1:8001','0')],'localhost',1,spawn_fn=spawn)
    g=agent._governor.local;clock=[0.];g.clock=lambda:clock[0];g.reader=lambda:(80*GIB,{})
    atomic(g.history_path,[dict(peak=3*GIB,complete=True)])
    atomic(g.control_path,dict(campaign=g.campaign['run_name'],time=__import__('time').time(),targets={g.node:1}))
    agent.start()
    for tick in (2,4,6):
        clock[0]=tick;agent.supervise_once()
        if len(generations)>=2:break
    assert len(generations)==2
    assert len(reports)==8 and len({p['task_uid'] for p in reports})==8
    assert len(clients)==2 and all(len(c)==1 for c in clients)
