"""Actual CPU worker/socket shutdown, supervisor exit and fail-closed receipts."""
import json
import socket
import sys
import threading
import time

import pytest


@pytest.mark.parametrize('changed',[False,True])
def test_dp_seal_requires_serving_island_dependency_versions(changed):
    from .run_slot import validate_dp_dependencies,Partial
    expected=dict(torch='2.7.1',numpy='1.26.4')
    actual=dict(expected)
    if changed:
        actual['torch']='2.10.0'
        with pytest.raises(Partial,match='serving interpreter dependency versions'):
            validate_dp_dependencies(dict(dependency_versions=expected),actual)
    else:
        validate_dp_dependencies(dict(dependency_versions=expected),actual)


@pytest.mark.parametrize('worker_count',[2,1])
def test_driver_drains_delayed_stock_worker_final_pull(tmp_path,monkeypatch,worker_count):
    from .driver import ProbeRecordingDriver
    from exp.r17_integration.tests.conductor import test_p2_gpu_fix2 as fixture
    from openpi.conductor.worker import WorkerLoop
    from openpi.conductor.task import EpisodeResult
    monkeypatch.setattr(fixture,'BenchmarkConductorDriver',lambda *args,**kwargs:
        ProbeRecordingDriver(*args,rejection_path=tmp_path/'rejected.json',shutdown_seconds=1,**kwargs))
    driver,probe=fixture.make_driver(tmp_path,'metaworld')
    stop=threading.Event();errors=[]
    class Runner:
        def run(self,task,report):
            time.sleep(.1)
            return EpisodeResult(task.task_uid,True,1)
        def close(self):pass
    class DelayedWorker(WorkerLoop):
        def _pull(self,sock):
            if self._probe_sent:time.sleep(.35)
            return super()._pull(sock)
    def checked(fn):
        try:fn()
        except BaseException as exc:errors.append(exc)
    started=time.monotonic()
    server=threading.Thread(target=lambda:checked(lambda:driver.run(stop=lambda:stop.is_set() or time.monotonic()-started>4)))
    workers=[];server.start()
    try:
        until=time.monotonic()+2
        while driver.port is None and time.monotonic()<until:time.sleep(.01)
        assert driver.port is not None
        for i,worker_class in enumerate((WorkerLoop,DelayedWorker)[:worker_count]):
            worker=worker_class('worker'+str(i),'local:1',Runner(),probe=probe,max_backoff_s=.05,
                connect=lambda:socket.create_connection(('127.0.0.1',driver.port),timeout=1))
            t=threading.Thread(target=lambda w=worker:checked(lambda:w.run_forever(stop=stop.is_set)))
            workers.append(t);t.start()
        server.join(timeout=3)
        for t in workers:t.join(timeout=1)
        assert not server.is_alive() and all(not t.is_alive() for t in workers)
        assert not errors,errors
        assert driver.scheduler.all_done()
        assert driver.worker_shutdown['status']==('PASS' if worker_count==2 else 'PARTIAL')
        assert driver.worker_shutdown['workers']==['worker'+str(i) for i in range(worker_count)]
        journal=[json.loads(s) for s in (tmp_path/'journal.jsonl').read_text().splitlines()]
        assert len(journal)==2 and all(r['accepted'] for r in journal)
        assert time.monotonic()-started<3
    finally:
        stop.set();server.join(timeout=2)
        for t in workers:t.join(timeout=2)
        driver.close()


@pytest.mark.parametrize('hang',[False,True])
def test_remote_supervisor_normal_exit_and_true_timeout(tmp_path,monkeypatch,hang,capsys):
    from . import remote_workers
    from exp.r17_integration.ops.sync_back import install_snapshot
    def command(args,run,index):
        return 'cpu_'+str(index),[sys.executable,'-c',
            'import time; time.sleep('+str(10 if hang else .05)+')']
    monkeypatch.setattr(remote_workers,'worker_command',command)
    run=tmp_path/'remote';started=time.monotonic()
    rc=remote_workers.main(['--endpoint','unused:1','--driver-host','unused','--driver-port','1',
        '--run',str(run),'--seconds','3'])
    receipt=json.loads((run/'receipt.json').read_text())
    assert time.monotonic()-started<5
    if hang:
        assert rc==2 and receipt['status']=='PARTIAL' and receipt['completion_reason']=='worker_deadline'
    else:
        assert rc==0 and receipt['status']=='PASS' and receipt['completion_reason']=='normal_exit'
        assert [p['exit_code'] for p in receipt['pids']]==[0,0]
        from exp.r17_integration.ops.build_client_bundle import sha256
        install_snapshot(run.with_suffix('.tar'),tmp_path/'closed',expected_sha256=sha256(run.with_suffix('.tar')))
        assert json.loads((tmp_path/'closed/receipt.json').read_text())['status']=='PASS'
    from .archive_stage import cleanup_stage
    output=capsys.readouterr().out
    transfer=json.loads(next(line.split('Q_REMOTE_ARCHIVE ',1)[1] for line in output.splitlines() if line.startswith('Q_REMOTE_ARCHIVE ')))
    install_snapshot(transfer['archive'],tmp_path/'verified_stage',expected_sha256=transfer['archive_sha256'])
    cleanup_stage(transfer['archive'],run.name,'workers',transfer['archive_sha256'])


@pytest.mark.parametrize('damage',[None,'worker_timeout','worker_failed','driver_drain','journal'])
def test_clean_worker_exit_and_closed_journal_gate(tmp_path,damage):
    from .run_slot import validate_closed_worker_run,Partial
    (tmp_path/'workers').mkdir()
    remote=dict(status='PASS',completion_reason='normal_exit',pids=[dict(exit_code=0),dict(exit_code=0)])
    driver=dict(status='PASS',worker_shutdown=dict(status='PASS',workers=['one','two']))
    journal=[dict(task_uid=str(i),accepted=True,error=None) for i in range(2)]
    if damage=='worker_timeout':remote.update(status='PARTIAL',completion_reason='worker_deadline')
    if damage=='worker_failed':remote['pids'][1]['exit_code']=-15
    if damage=='driver_drain':driver['worker_shutdown']['status']='PARTIAL'
    if damage=='journal':journal.pop()
    (tmp_path/'workers/receipt.json').write_text(json.dumps(remote))
    (tmp_path/'DRIVER_RECEIPT.json').write_text(json.dumps(driver))
    (tmp_path/'episode_manifest.json').write_text(json.dumps(dict(tasks=[dict(task_uid=str(i)) for i in range(2)])))
    (tmp_path/'journal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in journal))
    transfer=dict(archive_sha256='cpu-sha',manifest=dict(partials=[]))
    if damage:
        with pytest.raises(Partial):validate_closed_worker_run(tmp_path,transfer)
    else:
        result=validate_closed_worker_run(tmp_path,transfer)
        assert result['status']=='PASS' and result['accepted_episodes']==2 and result['normal_workers']==2


def test_native_source_drift_blocks_even_after_runtime_snapshot_refresh(tmp_path,monkeypatch):
    import hashlib
    from . import identity_audit
    from .run_slot import preflight,Partial
    q=tmp_path/'exp/r17_integration/qualification';(q/'specs').mkdir(parents=True)
    monkeypatch.setattr(identity_audit,'__file__',str(q/'identity_audit.py'))
    source=tmp_path/'model.py';source.write_text('old native code\n')
    original=hashlib.sha256(source.read_bytes()).hexdigest()
    assets=tmp_path/'assets';(assets/'synthetic_teacher_store').mkdir(parents=True)
    checkpoint=tmp_path/'checkpoint-1';checkpoint.mkdir()
    job=dict(asset_id='cpu',host='local',model='pi05',phase='native',domain='metaworld',
        runtime=str(tmp_path),assets=str(assets),checkpoint=str(checkpoint),python=sys.executable)
    (q/'specs/asset_cpu.json').write_text(json.dumps(job))
    (assets/'CPU_RECEIPT.json').write_text(json.dumps(dict(status='PASS')))
    (assets/'model_manifest.json').write_text(json.dumps(dict(source_digests={'model.py':original})))
    (assets/'model_spec.json').write_text(json.dumps(dict(checkpoint_identity='cpu',norm_identity='cpu',code_digest='cpu')))
    for name in ['load.json','synthetic_teacher_store/READY.json']:(assets/name).write_text('{}')
    assert identity_audit.audit('local',assets)['status']=='PASS'
    source.write_text('owner changed native code\n')
    (q/'runtime_code_snapshot.json').write_text(json.dumps(dict(files={'model.py':hashlib.sha256(source.read_bytes()).hexdigest()})))
    with pytest.raises(Partial,match='native asset source identity changed'):
        preflight(job)
