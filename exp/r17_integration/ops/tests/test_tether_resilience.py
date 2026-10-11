"""Exact rc70 repros, bounded bulk transfers, and resumed-control lineage (CPU)."""
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from exp.r17_integration.ops import evidence_transfer as bulk, tether_retry as tr
from exp.r17_integration.ops import formal_eval, formal_runtime, formal_queue, formal_prestage
from exp.r17_integration.ops import formal_recovery, transport_operations, sync_back
from exp.r17_integration.ops.deploy import tether_transfer
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.immutable_release import sha
from exp.r17_integration.tests.ops.test_tether_retry import Clock


@pytest.mark.parametrize('message',[
    'error: exec: timed out after 10s waiting for chunk\n',
    'error: pull (tier B): write local: nats: timeout\n',
    'error: pull (tier B): write local: read pipe: i/o timeout\n',
])
def test_real_baseline_rc70_timeouts_recover_and_are_disclosed(tmp_path, message):
    clock, calls = Clock(), []
    def operation():
        calls.append(True)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(70,['tether','exec'],stderr=message)
        return 'RUNNING'
    with tr.retry_scope(tmp_path):
        assert tr.retry(operation,label='controls',node='h100',clock=clock.now,sleeper=clock.sleep,
                        replay_safe=True) == 'RUNNING'
    receipt = json.loads((tmp_path/'TETHER_RETRY_RECEIPT.json').read_text())
    assert calls == [True,True] and clock.delays == [2]
    assert receipt['blips'] == 1 and receipt['events'][-1]['event'] == 'recovered'


@pytest.mark.parametrize('message',[
    'error: exec: timed out after 10s waiting for chunk',
    'error: exec: remote command exited 1: nats: timeout',
    'error: exec: remote execution deadline exceeded: waiting for chunk',
])
def test_ambiguous_spawn_and_real_remote_death_are_not_replayed(message):
    error = subprocess.CalledProcessError(70,['tether','exec'],stderr=message)
    assert not tr.transient(error)
    if 'remote ' in message:
        assert not tr.transient(error,replay_safe=True)


def local_remote(node, argv, **kwargs):
    assert node in ('timan107','h100') and kwargs['replay_safe'] is True
    stream = io.StringIO()
    with redirect_stdout(stream):
        exec(argv[-1], {})
    return stream.getvalue()


def test_many_chunks_retry_only_failed_chunk_throttle_and_keep_original(tmp_path):
    source = tmp_path/'original.tar'
    source.write_bytes(b'Q'* (2*bulk.CHUNK_BYTES+19))
    destination = tmp_path/'pulled.tar'
    calls, heartbeats, clock = [], [], Clock()
    def runner(argv, **kwargs):
        path = Path(argv[2].split(':',1)[1]); calls.append((str(path),path.stat().st_size))
        if len(calls) == 2:
            return SimpleNamespace(returncode=70,stdout='',stderr='error: pull (tier B): write local: nats: timeout')
        shutil.copyfile(path,argv[3])
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    def transfer(argv):
        return tether_transfer(argv,lane='tether',runner=runner,retry_seconds=300,
                              sleeper=clock.sleep,clock=clock.now)
    with tr.retry_scope(tmp_path/'receipt'):
        receipt = bulk.pull('timan107',source,destination,remote=local_remote,transfer=transfer,
            expected_sha256=sha(source),sleeper=clock.sleep,clock=clock.now,
            heartbeat=lambda:heartbeats.append(True),bulk_lock=str(tmp_path/'test_bulk.lock'))
    assert sha(destination) == sha(source) and source.exists()
    assert receipt['status'] == 'PASS' and len(receipt['chunks']) == 3
    assert len(calls) == 4 and calls[1] == calls[2] and calls[0] != calls[1]
    assert max(length for _,length in calls) <= bulk.CHUNK_BYTES
    assert clock.now() >= source.stat().st_size/bulk.BYTES_PER_SECOND
    assert len(heartbeats) >= 3 and not Path('/tmp/r17_evidence_'+receipt['token']).exists()
    assert json.loads(destination.with_name(destination.name+'.TRANSFER.json').read_text())['status'] == 'PASS'


def test_wrong_attested_archive_sha_fails_before_any_pull(tmp_path):
    source = tmp_path/'original'; source.write_bytes(b'original')
    def transfer(argv):
        pytest.fail('SHA mismatch must not issue a pull')
    with pytest.raises(ValueError,match='remote evidence archive SHA mismatch'):
        bulk.pull('h100',source,tmp_path/'result',remote=local_remote,transfer=transfer,expected_sha256='0'*64)
    assert source.read_bytes() == b'original'


def test_corrupt_chunk_is_never_published_or_classified_as_transport(tmp_path):
    source = tmp_path/'original';source.write_bytes(b'original')
    def transfer(argv):
        Path(argv[3]).write_bytes(b'corrupt!')
    with pytest.raises(ValueError,match='chunk SHA mismatch') as failed:
        bulk.pull('h100',source,tmp_path/'result',remote=local_remote,transfer=transfer,
                  bulk_lock=str(tmp_path/'test_bulk.lock'))
    assert not tr.transient(failed.value,replay_safe=True)
    assert not (tmp_path/'result').exists() and source.read_bytes()==b'original'
    report=json.loads((tmp_path/'result.TRANSFER.json').read_text())
    # Only this fixture's exact owned temporary chunk is cleaned.
    stage=Path('/tmp/r17_evidence_'+report['token'])
    for child in stage.iterdir():child.unlink()
    stage.rmdir()


def lineage_fixture(root):
    original,resumed = root/'original',root/'resumed'
    for path in (original,resumed):
        (path/'assets').mkdir(parents=True)
        for name in ('runtime.tar','workers.tar'): (path/name).write_bytes(name.encode())
        for name in ('model_spec.json','model_manifest.json','benchmark.json','full.yaml'):
            write_json(path/'assets'/name,{'sealed':name})
    old=dict(model='fastwam',domain='libero',checkpoint='/immutable',server_node='h100',
        official_50=True,smoke=False,runtime_identity='frozen',run_name='original')
    write_json(original/'campaign.json',old)
    pins={'campaign.json':sha(original/'campaign.json')}
    new=dict(old,run_name='resumed',resume_from=str(original),resume_source_pins=pins)
    write_json(resumed/'campaign.json',new)
    tasks=[dict(task_uid='a',server_host='old',server_port=1,orig_init_state_idx=0),
           dict(task_uid='b',server_host='old',server_port=1,orig_init_state_idx=1)]
    write_json(original/'EPISODE_PLAN.json',{'tasks':tasks})
    write_json(resumed/'FULL_EPISODE_PLAN.json',{'tasks':tasks})
    selection=dict(source=str(original),source_pins=pins,retained_episodes=1,dispatched_episodes=1,
        expected_episodes=2,retained_task_uids=['a'],missing_task_uids=['b'])
    write_json(resumed/'RESUME.json',selection)
    return original,resumed,dict(old,paired_control=str(resumed)),selection


def test_resumed_control_may_prepare_from_original_runtime_but_cannot_dispatch_pending(tmp_path):
    original,resumed,arm,_=lineage_fixture(tmp_path)
    proof=formal_runtime.paired_lineage(arm,original)
    assert proof['control']==str(resumed) and proof['source_runtime']==str(original)
    assert proof['completion_required_at_launch'] is True
    with pytest.raises(FileNotFoundError):
        formal_runtime.paired_lineage(arm,original,require_complete=True)
    assert not (resumed/'RECEIPT.json').exists()


@pytest.mark.parametrize('change',['checkpoint','runtime','worker','cell','overlap','sourcepin'])
def test_unrelated_or_changed_resumed_control_is_refused(tmp_path,change):
    original,resumed,arm,selection=lineage_fixture(tmp_path)
    if change=='checkpoint':
        new=json.loads((resumed/'campaign.json').read_text());new['checkpoint']='other';write_json(resumed/'campaign.json',new)
    elif change in ('runtime','worker'):
        (resumed/('runtime.tar' if change=='runtime' else 'workers.tar')).write_bytes(b'different')
    elif change=='cell':
        write_json(resumed/'FULL_EPISODE_PLAN.json',{'tasks':[{'task_uid':'a'},{'task_uid':'c'}]})
    elif change=='overlap':
        selection['missing_task_uids']=['a'];write_json(resumed/'RESUME.json',selection)
    else:
        (original/'campaign.json').write_text('changed')
    with pytest.raises((ValueError,json.JSONDecodeError)):
        formal_runtime.paired_lineage(arm,original)


def test_queue_rebind_cli_requires_cpu_only_existing_queue():
    for args in (['--jobs','unused','--prepare-only','--rebind-control','unused'],
                 ['--prepared-queue','unused','--run','--rebind-control','unused']):
        with pytest.raises(SystemExit):formal_queue.main(args)


def test_lost_spawn_reply_recovers_only_its_owned_receipt_without_second_spawn(monkeypatch,tmp_path):
    request=dict(receipt=str(tmp_path/'owned.process.json'),argv=['owned'],env={},cwd='/owned')
    argv=['python3','-m','exp.r17_integration.ops.formal_process','spawn','--request',json.dumps(request)]
    calls,clock=[],Clock()
    def execute(command,seconds):
        calls.append(command)
        if len(calls)==1:
            raise subprocess.CalledProcessError(70,command,stderr='error: exec: timed out after 10s waiting for chunk')
        assert '-c' in command and 'owned.process.json' in command[-1]
        if len(calls)==2:return 'null'
        return json.dumps(dict(request=request,child={'pid':123},status='RUNNING'))
    monkeypatch.setattr(formal_eval,'execute',execute)
    monkeypatch.setattr(tr.time,'monotonic',clock.now)
    monkeypatch.setattr(tr.time,'sleep',clock.sleep)
    with tr.retry_scope(tmp_path):
        value=json.loads(formal_eval.remote('timan107',argv))
    assert value['child']['pid']==123 and len(calls)==3
    assert sum('spawn' in command for command in calls)==1
    assert clock.delays==[2,4]


def test_cleanup_lost_reply_can_be_replayed_without_deleting_other_stage(tmp_path,monkeypatch):
    from exp.r17_integration.qualification import archive_stage
    monkeypatch.setattr(archive_stage,'STAGE_ROOT',tmp_path/'stage')
    source=tmp_path/'source.tar';source.write_bytes(b'closed archive')
    staged=archive_stage.stage_archive(source,'owned','workers',sha(source))
    request=dict(archive=staged['archive'],key='owned',sha256=sha(source))
    first=transport_operations.cleanup_archive(request)
    second=transport_operations.cleanup_archive(request)
    assert first==second and first['stage_removed'] is True
    assert source.exists()
    with pytest.raises(ValueError,match='exact Q run stage|ownership mismatch'):
        transport_operations.cleanup_archive(dict(request,key='another_campaign'))


def recovery_fixture(root):
    root.mkdir(); (root/'assets').mkdir(); (root/'driver').mkdir()
    model={'default_sampler':{'num_steps':2}}
    write_json(root/'campaign.json',dict(run_name='recovery',server_node='local',replicas=1,smoke=False))
    write_json(root/'assets/model_spec.json',model)
    write_json(root/'assets/model_manifest.json',dict(spec=model,content_digests={'weights':'frozen'}))
    write_json(root/'SERVER_ATTESTATIONS.json',[dict(metadata=dict(r17_model_spec=model,
        cache_enabled=False,debug_capture=False))])
    tasks=[dict(task_uid=uid,experiment='suite',extra=dict(task_name=uid)) for uid in ('a','b')]
    write_json(root/'EPISODE_PLAN.json',dict(tasks=tasks,model=model,benchmark_protocol='pinned'))
    for name,rows in [
        ('journal.jsonl',[dict(task_uid=uid,accepted=True,attempt=1,status='done',error=None,success=True) for uid in ('a','b')]),
        ('attempts.jsonl',[dict(task_uid=uid,accepted=True,attempt=1,error=None,worker_id='worker') for uid in ('a','b')]),
        ('per_step.jsonl',[dict(task_uid=uid,accepted=True,attempt=1,capture_enabled=False,source='policy',
                               actual_nfe=2,stage_calls=[1,1,2]) for uid in ('a','b')])]:
        (root/'driver'/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write_json(root/'driver/DRIVER_RECEIPT.json',dict(status='PASS',elapsed_seconds=2))
    worker=root/'remote_workers';worker.mkdir()
    manifest={'campaign':'recovery'}
    write_json(root/'timan107.json',manifest);write_json(worker/'manifest.json',manifest)
    write_json(worker/'receipt.json',dict(status='PASS',reason='normal_exit',worker_failures=[]))
    transfer=sync_back.snapshot_run(worker,root/'timan107.tar')
    transfer['archive']='/tmp/r17_Q_qualification/recovery_timan107/workers_'+transfer['archive_sha256']+'_'+'a'*32+'.tar'
    write_json(root/'attestation.json',transfer)
    write_json(root/'WORKER_DEPLOYMENT.json',dict(timan107=dict(worker_run='/scratch/owned/recovery_timan107')))
    closed=dict(node='timan107',label='workers',status='CLOSED',exit_code=0,closure_confirmed=True)
    report=dict(status='FAIL',all_owned_closed=True,error='error: pull (tier B): write local: nats: timeout',
                owned_local=[],owned_remote=[dict(node='timan107',label='workers')],cleanup=[closed],
                driver_completion=formal_eval.driver_completion(root,0))
    write_json(root/'RECEIPT.json',report)
    return report


def test_final_pull_recovery_preserves_failure_and_seals_verified_summary(tmp_path):
    root=tmp_path/'run';report=recovery_fixture(root)
    original=(root/'RECEIPT.json').read_bytes()
    result=formal_recovery.recover_pull(root,'timan107',root/'attestation.json')
    assert result['status']=='PASS'
    assert (root/'RECEIPT_BEFORE_RECOVERY.json').read_bytes()==original
    recovered=json.loads((root/'RECEIPT.json').read_text())
    assert recovered['status']=='PASS' and recovered['recovery']['original_error']==report['error']
    summary=json.loads((root/'SUMMARY.json').read_text())
    assert summary['status']=='PASS' and summary['completed_episodes']==2 and summary['terminal_errors']==0


@pytest.mark.parametrize('change',['missing','unaccepted','unclosed','real_failure','wrong_sha'])
def test_evidence_recovery_refuses_incomplete_or_unbound_campaign(tmp_path,change):
    root=tmp_path/'run';report=recovery_fixture(root)
    if change in ('missing','unaccepted'):
        rows=[json.loads(row) for row in (root/'driver/journal.jsonl').read_text().splitlines()]
        if change=='missing':rows.pop()
        else:rows[-1]['accepted']=False
        (root/'driver/journal.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif change=='unclosed':report['all_owned_closed']=False
    elif change=='real_failure':report['error']='remote worker fleet failed: timan107'
    else:(root/'timan107.tar').write_bytes(b'tampered')
    write_json(root/'RECEIPT.json',report)
    original=(root/'RECEIPT.json').read_bytes()
    with pytest.raises((ValueError,RuntimeError)):
        formal_recovery.recover_pull(root,'timan107',root/'attestation.json')
    assert (root/'RECEIPT.json').read_bytes()==original
    assert not (root/'SUMMARY.json').exists()
