"""Allowed-root archive transfer, SHA validation and verified-only cleanup."""
import json
from pathlib import Path
import shutil

import pytest


def snapshot(tmp_path, monkeypatch, *, run_key='cpu_transfer', kind='workers'):
    from . import archive_stage as stage
    from exp.r17_integration.ops.sync_back import snapshot_run
    monkeypatch.setattr(stage,'STAGE_ROOT',tmp_path/'allowed_tmp')
    monkeypatch.setattr(stage,'RUN_ROOT',tmp_path/'durable_runs')
    source=tmp_path/'scratch_qualification'/run_key;source.mkdir(parents=True)
    (source/'manifest.json').write_text(json.dumps(dict(cpu_only=True,gpu_executed=False)))
    (source/'receipt.json').write_text(json.dumps(dict(status='PASS',completion_reason='normal_exit',
        pids=[dict(exit_code=0),dict(exit_code=0)])))
    archive=source.with_suffix('.tar');transfer=snapshot_run(source,archive)
    transfer.update(stage.stage_archive(archive,run_key,kind,transfer['archive_sha256']))
    return stage,archive,transfer


def test_stage_p6_verified_pull_then_cleanup_preserves_source(tmp_path,monkeypatch):
    from exp.r17_integration.ops.sync_back import install_snapshot
    stage,archive,t=snapshot(tmp_path,monkeypatch)
    staged=Path(t['archive']);assert staged.is_file() and archive.is_file()
    assert stage.stage_identity(staged,t['run_key'],t['kind'])==t['archive_sha256']
    downloaded=tmp_path/'pulled.tar';shutil.copyfile(staged,downloaded)
    install_snapshot(downloaded,tmp_path/'verified',expected_sha256=t['archive_sha256'])
    receipt=stage.cleanup_stage(staged,t['run_key'],t['kind'],t['archive_sha256'])
    assert receipt['stage_removed'] and not staged.exists() and archive.is_file()
    assert json.loads((tmp_path/'verified/receipt.json').read_text())['status']=='PASS'


@pytest.mark.parametrize('damage',['wrong_sha','wrong_run','wrong_kind','outside_root'])
def test_stage_cleanup_rejects_wrong_attestation_and_retains_archive(tmp_path,monkeypatch,damage):
    stage,archive,t=snapshot(tmp_path,monkeypatch);path=t['archive'];key=t['run_key'];kind=t['kind'];sha=t['archive_sha256']
    if damage=='wrong_sha':sha='0'*64
    if damage=='wrong_run':key='someone_else'
    if damage=='wrong_kind':kind='libero'
    if damage=='outside_root':path=str(archive)
    with pytest.raises(ValueError):stage.cleanup_stage(path,key,kind,sha)
    assert Path(t['archive']).is_file() and archive.is_file()


@pytest.mark.parametrize('corrupt',[False,True])
def test_h100_allowed_push_stage_sha_verified_durable_copy(tmp_path,monkeypatch,corrupt):
    stage,archive,t=snapshot(tmp_path,monkeypatch)
    path=Path(t['archive']);destination=stage.RUN_ROOT/t['run_key']/'workers.tar'
    if corrupt:
        path.write_bytes(b'corrupt')
        with pytest.raises(ValueError,match='SHA/size'):stage.install_relay_archive(path,destination,t['run_key'],'workers',t['archive_sha256'])
        assert path.is_file() and not destination.exists()
    else:
        receipt=stage.install_relay_archive(path,destination,t['run_key'],'workers',t['archive_sha256'])
        assert receipt['incoming_stage_removed'] and not path.exists()
        assert stage.file_sha256(destination)==t['archive_sha256'] and archive.is_file()


@pytest.mark.parametrize('failure',[None,'corrupt','timeout'])
@pytest.mark.parametrize('host',['local','h100'])
def test_worker_pull_allowed_stage_cleanup_only_after_p6_and_closed_journal(tmp_path,monkeypatch,failure,host):
    from . import run_slot
    stage,archive,t=snapshot(tmp_path,monkeypatch)
    run=stage.RUN_ROOT/t['run_key'];run.mkdir(parents=True)
    (run/'DRIVER_RECEIPT.json').write_text(json.dumps(dict(status='PASS',worker_shutdown=dict(status='PASS'))))
    (run/'episode_manifest.json').write_text(json.dumps(dict(tasks=[dict(task_uid=str(i)) for i in range(2)])))
    (run/'journal.jsonl').write_text(''.join(json.dumps(dict(task_uid=str(i),accepted=True,error=None))+'\n' for i in range(2)))
    cleaned=[]
    def cleanup(job,actual,owned,node,transfer):
        assert (actual/'WORKER_COMPLETION.json').is_file()
        cleaned.append(stage.cleanup_stage(transfer['archive'],actual.name,'workers',transfer['archive_sha256']))
    monkeypatch.setattr(run_slot,'cleanup_transfer_stage',cleanup)
    class Owned:
        report={}
        def spawn(self,name,argv):
            assert argv[0:2]==['tether','pull'] and argv[2]=='timan107:'+t['archive']
            if failure=='corrupt':Path(argv[3]).write_bytes(b'corrupt')
            elif failure!='timeout':shutil.copyfile(t['archive'],argv[3])
            return object()
        def wait(self,child,seconds):
            assert seconds==(125 if host=='h100' else 85)
            return 75 if failure=='timeout' else 0
    owned=Owned()
    if failure:
        with pytest.raises((ValueError,run_slot.Partial)):run_slot.pull_closed_workers(dict(host=host),run,owned,'timan107',t)
        assert not cleaned and Path(t['archive']).is_file()
    else:
        run_slot.pull_closed_workers(dict(host=host),run,owned,'timan107',t)
        assert len(cleaned)==1 and owned.report['worker_transfer']['verified_pull']
        assert not Path(t['archive']).exists() and archive.is_file()


@pytest.mark.parametrize('corrupt',[False,True])
def test_relay_push_uses_allowed_stage_and_verifies_before_push(tmp_path,monkeypatch,corrupt):
    from .relay import Broker
    stage,archive,t=snapshot(tmp_path,monkeypatch)
    broker=Broker.__new__(Broker);broker.run=stage.RUN_ROOT/t['run_key'];broker.run.mkdir(parents=True)
    commands=[]
    def command(argv,*args):
        commands.append(argv)
        if argv[1]=='pull':
            Path(argv[3]).write_bytes(b'corrupt' if corrupt else Path(t['archive']).read_bytes())
        if argv[1]=='push':
            assert argv[3]=='h100:'+t['archive'] and argv[3]!='h100:'+str(broker.run/'workers.tar')
        result=dict(rc=0,stdout='',stderr='')
        if '--install' in argv:result['stdout']=json.dumps(dict(status='PASS',archive_sha256=t['archive_sha256']))
        return result
    broker.command=command
    args=['pull','timan108:'+t['archive'],str(broker.run/'workers.tar'),'--timeout','35s']
    if corrupt:
        with pytest.raises(ValueError,match='pulled archive SHA'):broker.execute(args)
        assert not any(argv[1]=='push' for argv in commands)
    else:
        assert broker.execute(args)['rc']==0
        assert any('--install' in argv for argv in commands)
        assert (broker.run/'RELAY_TRANSFER_workers.json').is_file()


@pytest.mark.parametrize('source',['/scratch/zixuans8/mw_r17/qualification/runs/old.tar','/data/mw_finetune/qualification/runs/old.tar'])
def test_relay_refuses_unstaged_scratch_or_data_source(tmp_path,source):
    from .relay import Broker
    broker=Broker.__new__(Broker);broker.run=Path('/data/mw_finetune/qualification/runs/cpu_transfer')
    broker.command=lambda *args:pytest.fail('no transfer should start')
    with pytest.raises(ValueError,match='exact Q run stage'):
        broker.execute(['pull','timan108:'+source,str(broker.run/'workers.tar')])
