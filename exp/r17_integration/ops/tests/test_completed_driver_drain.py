"""CPU-only completed driver recovery; no subprocesses, sockets, or signals."""
import json

import pytest

from exp.r17_integration.ops import formal_eval as launch
from exp.r17_integration.ops.formal_process import write_json


@pytest.fixture
def completed(tmp_path):
    write_json(tmp_path/'campaign.json',dict(run_name='complete',smoke=False))
    write_json(tmp_path/'EPISODE_PLAN.json',dict(tasks=[dict(task_uid='a',attempt=1,
        experiment='libero_10',extra={'task_name':'libero_10/a'})],benchmark_protocol='sealed'))
    driver=tmp_path/'driver'
    driver.mkdir()
    (driver/'journal.jsonl').write_text(json.dumps(dict(task_uid='a',attempt=1,accepted=True,
        status='done',success=True,error=None))+'\n')
    (driver/'attempts.jsonl').write_text(json.dumps(dict(task_uid='a',attempt=1,accepted=True,
        success=True,error=None,n_steps=20,worker_id='worker0'))+'\n')
    (driver/'per_step.jsonl').write_text(json.dumps(dict(task_uid='a',attempt=1,accepted=True))+'\n')
    write_json(driver/'DRIVER_RECEIPT.json',dict(status='PARTIAL',all_done=True,rejected_worker_probes=0,
        worker_shutdown={'status':'PARTIAL','workers':['worker0']},elapsed_seconds=2))
    return tmp_path


def test_complete_exit2_with_zero_errors_and_missing_shutdown_ack(completed):
    raw=(completed/'driver/DRIVER_RECEIPT.json').read_bytes()
    proof=launch.driver_completion(completed,2)
    assert proof['verified_completion'] and proof['recovered_worker_shutdown']
    assert not proof['recovered_retried_errors']
    assert proof['error_attempts']==0 and proof['resolved_retry_errors']==[]
    assert launch.aggregate(completed,driver_returncode=2)['status']=='PASS'
    assert (completed/'driver/DRIVER_RECEIPT.json').read_bytes()==raw


@pytest.mark.parametrize('change',['no_shutdown_evidence','not_done','rejected_probe','missing_final',
                                  'duplicate_final','wrong_success','unknown_uid','later_error','missing_retry_error'])
def test_exit2_needs_complete_attempt_history(completed,change):
    path=completed/'driver/DRIVER_RECEIPT.json'
    receipt=json.loads(path.read_text())
    attempts_path=completed/'driver/attempts.jsonl'
    attempts=[json.loads(line) for line in attempts_path.read_text().splitlines()]
    if change=='no_shutdown_evidence':receipt['worker_shutdown']['status']='PASS'
    elif change=='not_done':receipt['all_done']=False
    elif change=='rejected_probe':receipt['rejected_worker_probes']=1
    elif change=='missing_final':attempts=[]
    elif change=='duplicate_final':attempts+=attempts
    elif change=='wrong_success':attempts[0]['success']=False
    elif change=='unknown_uid':attempts.append(dict(attempts[0],task_uid='unknown',error='error'))
    elif change=='later_error':attempts.append(dict(attempts[0],attempt=2,error='late error'))
    elif change=='missing_retry_error':
        attempts[0]['attempt']=2
        journal=completed/'driver/journal.jsonl'
        row=json.loads(journal.read_text());row['attempt']=2
        journal.write_text(json.dumps(row)+'\n')
    write_json(path,receipt)
    attempts_path.write_text(''.join(json.dumps(row)+'\n' for row in attempts))
    with pytest.raises(RuntimeError,match='driver incomplete'):
        launch.driver_completion(completed,2)


def test_drain_targets_only_owned_complete_workers(completed):
    calls=[]
    owned=dict(node='timan107',label='workers',request={'receipt':'/exact-owned-run/workers.json'})
    def managed(target,mode):
        calls.append((target,mode))
        return dict(status='CLOSED',alive=False,exit_code=2,child_alive=False,
                    supervisor_alive=False,owned_worker_survivors=[])
    proof=launch.driver_completion(completed,2)
    assert launch.drain_completed_workers(owned,proof,managed)['alive'] is False
    assert calls==[(owned,'stop')]
    for bad in (dict(proof,verified_completion=False),):
        with pytest.raises(ValueError):launch.drain_completed_workers(owned,bad,managed)
    with pytest.raises(ValueError):launch.drain_completed_workers(dict(owned,label='replica_0'),proof,managed)
    assert calls==[(owned,'stop')]
    with pytest.raises(RuntimeError):
        launch.drain_completed_workers(owned,proof,lambda *_:dict(status='CLOSED',alive=True,exit_code=2))


def test_complete_driver_defers_killed_snapshot_but_never_live_fleet(completed):
    proof=launch.driver_completion(completed,2)
    owned=dict(node='timan107',label='workers')
    closed=dict(status='CLOSED',alive=False,child_alive=False,supervisor_alive=False,
                exit_code=-9,owned_worker_survivors=[])
    result=launch.drain_completed_workers(owned,proof,lambda *_:dict(closed))
    assert result['snapshot_incomplete'] and result['exit_code']==-9
    # Rc0 drivers may also finish before the native snapshot stop grace expires.
    assert launch.drain_completed_workers(owned,dict(proof,recovered_worker_shutdown=False),
        lambda *_:dict(closed))['snapshot_incomplete']
    for patch in ({'child_alive':True},{'supervisor_alive':True},{'owned_worker_survivors':[{'pid':42}]},
                  {'exit_code':None},{'status':'RUNNING'}):
        with pytest.raises(RuntimeError):
            launch.drain_completed_workers(owned,proof,lambda *_:dict(closed,**patch))
