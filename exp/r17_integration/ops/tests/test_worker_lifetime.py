"""CPU wire regression: recycle only after a fenced result, never mid-episode."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

from exp.r17_integration.ops import formal_eval, formal_workers
from exp.r17_integration.ops.formal_process import ticks
from exp.r17_integration.ops.worker_lifetime import limited_loop, planned_recycle, RECYCLE_EXIT
from openpi.conductor import protocol
from openpi.conductor.task import EpisodeTask, EpisodeResult
from openpi.conductor.worker import WorkerLoop


def test_recycle_sends_both_attempt_fenced_results_and_closes_runner_first(tmp_path):
    left,right=socket.socketpair()
    class Runner:
        closed=False
        def run(self,task,report):
            return EpisodeResult(task.task_uid,True,5)
        def close(self):self.closed=True
    runner=Runner()
    marker=tmp_path/'RECYCLE.json'
    results=[]
    errors=[]
    def driver():
        try:
            for i in range(2):
                kind,payload=protocol.recv_message(right)
                assert kind==protocol.MSG_PULL
                task=EpisodeTask(f'cell{i}','yaml','eval','libero_spatial',0,i,i,'host',1,'formal_full_v1',attempt=3)
                protocol.send_message(right,protocol.MSG_ASSIGN,{'task':protocol.task_to_wire(task)})
                kind,payload=protocol.recv_message(right)
                assert kind==protocol.MSG_REPORT_RESULT
                results.append(payload)
            # No third assignment was requested; two complete results preceded EOF.
            assert right.recv(1)==b''
        except BaseException as exc:errors.append(exc)
        finally:right.close()
    thread=threading.Thread(target=driver)
    thread.start()
    loop=limited_loop(WorkerLoop,2,marker)('owned-worker','host:1',runner,connect=lambda:left)
    with pytest.raises(SystemExit) as exc:loop.run_forever()
    thread.join(timeout=5)
    assert exc.value.code==RECYCLE_EXIT and not thread.is_alive() and not errors
    assert runner.closed
    assert [(r['task_uid'],r['attempt']) for r in results]==[('cell0',3),('cell1',3)]
    entry=dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),recycle_receipt=str(marker))
    assert planned_recycle(entry,RECYCLE_EXIT)
    assert not planned_recycle(dict(entry,start_ticks='reused'),RECYCLE_EXIT)
    assert not planned_recycle(entry,-9)
    assert not planned_recycle(dict(entry,recycle_receipt=str(tmp_path/'absent')),RECYCLE_EXIT)


def test_stock_shutdown_does_not_request_replacement(tmp_path):
    class Base:
        def run_forever(self,stop=None):pass
    loop=limited_loop(Base,8,tmp_path/'marker')()
    loop.run_forever()
    assert not (tmp_path/'marker').exists()


def test_future_libero_defaults_bound_worker_lifetime_and_keep_inference_flags_identical():
    args=formal_eval.parse_args(['--prepare-only'])
    assert args.worker_max_episodes==4
    assert args.worker_memory_mib==6144
    assert formal_eval.parse_args(['--model','dp_mw','--prepare-only']).worker_max_episodes==0
    with pytest.raises(SystemExit):formal_eval.parse_args(['--prepare-only','--worker-max-episodes','-1'])
    campaign=dict(run_name='same',worker_node='timan107',endpoint='host:23180',driver_host='driver',
        driver_port=23189,domain='libero',worker_benchmark='/tmp/sealed.json',worker_egl_device=0)
    worker_id,old=formal_workers.worker_command(campaign,0)
    new_id,new=formal_workers.worker_command(dict(campaign,worker_max_episodes=8),0)
    assert new_id==worker_id
    assert new[:3]==old[:3] and new[4:-2]==old[4:]
    assert new[3]=='exp.r17_integration.ops.worker_lifetime'
    assert new[-2:]==['--max-episodes','8']


def test_planned_recycle_and_sigkill_use_independent_budgets_and_unique_generations(tmp_path):
    from exp.r17_integration.ops.formal_process import write_json, alive
    fake=tmp_path/'worker.py'
    fake.write_text('''import json,os,pathlib,signal
from exp.r17_integration.ops.formal_process import ticks,write_json
generation=int(pathlib.Path.cwd().name.split('_')[-1])
if generation==0:
    write_json(os.environ['R17_WORKER_RECYCLE_RECEIPT'],dict(schema='r17.worker_recycle.v1',
        pid=os.getpid(),start_ticks=ticks(os.getpid()),completed_episodes=8,maximum_episodes=8,reason='episode_boundary'))
    raise SystemExit(75)
if generation==1:os.kill(os.getpid(),signal.SIGKILL)
''')
    campaign=dict(run_name='owned',domain='libero',worker_run=str(tmp_path/'runs/owned'),node_workers=1,
                  seconds=30,worker_memory_mib=6144,worker_restart_limit=1,worker_max_episodes=8)
    write_json(tmp_path/'campaign.json',campaign)
    harness=f'''import sys
from exp.r17_integration.ops import formal_workers as f,worker_memory as m
m.admission=lambda *a,**kw: dict(admitted=True)
f.worker_command=lambda *a,**kw: ('owned',[sys.executable,{str(fake)!r}])
sys.argv=['cpu','--campaign',{str(tmp_path/'campaign.json')!r}]
f.main()
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(formal_eval.ROOT))
    result=subprocess.run([sys.executable,'-c',harness],env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    receipt=json.loads((tmp_path/'runs/owned/receipt.json').read_text())
    assert receipt['status']=='PASS'
    assert [p['generation'] for p in receipt['pids']]==[0,1,2]
    assert [p['exit_code'] for p in receipt['pids']]==[75,-9,0]
    assert len(receipt['worker_recycles'])==len(receipt['worker_failures'])==1
    assert all(not alive(p) for p in receipt['pids'])
