import json
import os
from pathlib import Path
import tarfile
import time

import pytest

from exp.r17_integration.ops import incremental_sync as sync
from exp.r17_integration.ops.formal_process import ticks,write_json
from exp.r17_integration.ops.immutable_release import sha
from exp.r17_integration.ops.sync_back import snapshot_run,install_snapshot


def fleet(tmp_path):
    root=tmp_path/'runs';root.mkdir();run=root/'campaign_timan107';run.mkdir()
    write_json(run/'manifest.json',{'domain':'libero','model':'fastwam','scientific':'fixed'})
    owner={'pid':os.getpid(),'start_ticks':ticks(os.getpid())}
    entries=[dict(pid=987654321,start_ticks='not-live',exit_code=75,index=0,generation=0),
             dict(owner,exit_code=None,index=1,generation=0)]
    write_json(run/'receipt.json',{'status':'RUNNING','fleet_owner':owner,'pids':entries})
    for i in [0,1]:
        p=run/'scratch'/('worker_'+str(i))/'generation_0';p.mkdir(parents=True)
        (p/'cache.bin').write_bytes(('closed' if i==0 else 'active').encode()*1024)
        write_json(p/'RECYCLE.json',{'keep':'identity marker'})
        (p/'steps.jsonl').write_text('keep scientific rows\n')
        log=run/('worker_'+str(i)+'_generation_0.log');log.write_bytes(b'log evidence')
        for f in [p/'cache.bin',log]:os.utime(f,(time.time()-60,time.time()-60))
    (run/'journal.jsonl').write_text('never prune accepted episodes\n')
    request={'run':str(run),'node':'timan107','manifest_sha256':sha(run/'manifest.json'),'max_bytes':1024**2}
    return root,run,request


def call(root,request,**fields):return sync.remote_operation(dict(request,**fields),_run_parent=root)


def test_verified_chunk_then_prune_then_closed_snapshot_restore(tmp_path):
    root,run,request=fleet(tmp_path)
    chunk=call(root,request,action='stage')
    assert {x['name'] for x in chunk['files'] if x['prunable']}=={'worker_0_generation_0.log','scratch/worker_0/generation_0/cache.bin'}
    assert {'manifest.json','receipt.json'}<={x['name'] for x in chunk['files'] if not x['prunable']}
    campaign=tmp_path/'local';campaign.mkdir()
    base=campaign/'INCREMENTAL_SYNC'/'timan107'
    receipt=sync.verify_chunk(chunk,chunk['archive'],base)
    result=call(root,request,action='ack',chunk=chunk,verified_archive_sha256=chunk['archive_sha256'],local_receipt_sha256=sha(receipt))
    assert result['pruned'] and result['bytes_freed']>0
    assert not (run/'worker_0_generation_0.log').exists()
    assert (run/'worker_1_generation_0.log').exists() and (run/'journal.jsonl').exists()
    assert (run/'scratch/worker_0/generation_0/RECYCLE.json').exists()
    assert (run/'scratch/worker_0/generation_0/steps.jsonl').exists()
    archive=tmp_path/'closed.tar'
    snap=snapshot_run(run,archive)
    assert not snap['manifest']['partials']
    install_snapshot(archive,campaign/'workers'/'timan107',expected_sha256=snap['archive_sha256'])
    assert sync.restore(campaign,'timan107')['files']==2
    assert (campaign/'workers/timan107/worker_0_generation_0.log').read_bytes()==b'log evidence'
    assert sync.restore(campaign,'timan107')['status']=='PASS'  # idempotent


def test_transport_hiccup_reuses_one_chunk_and_keeps_originals(tmp_path):
    root,run,request=fleet(tmp_path)
    first=call(root,request,action='stage')
    assert first==call(root,request,action='stage')
    assert len(list(Path(first['archive']).parent.glob('*.tar.gz')))==1
    assert (run/'worker_0_generation_0.log').exists()
    with pytest.raises(ValueError,match='local ACK'):
        call(root,request,action='ack',chunk=first,verified_archive_sha256='0'*64,local_receipt_sha256='1'*64)
    assert (run/'worker_0_generation_0.log').exists()


def test_corrupt_archive_cannot_publish_verification_receipt(tmp_path):
    root,run,request=fleet(tmp_path)
    chunk=call(root,request,action='stage')
    Path(chunk['archive']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='archive SHA'):
        sync.verify_chunk(chunk,chunk['archive'],tmp_path/'local')
    assert not list((tmp_path/'local').glob('records/*.json'))
    assert (run/'worker_0_generation_0.log').exists()


def test_fleet_close_before_ack_preserves_finalizer_inputs(tmp_path):
    root,run,request=fleet(tmp_path)
    chunk=call(root,request,action='stage')
    receipt=sync.verify_chunk(chunk,chunk['archive'],tmp_path/'local')
    report=json.loads((run/'receipt.json').read_text());report['status']='PASS';write_json(run/'receipt.json',report)
    result=call(root,request,action='ack',chunk=chunk,verified_archive_sha256=chunk['archive_sha256'],local_receipt_sha256=sha(receipt))
    assert not result['pruned'] and (run/'worker_0_generation_0.log').exists()
    assert not (run/'INCREMENTAL_OFFLOAD.json').exists()


def test_unowned_root_manifest_changes_symlinks_and_live_native_refuse(tmp_path):
    root,run,request=fleet(tmp_path)
    with pytest.raises(ValueError,match='unowned'):sync.remote_operation(dict(request,action='stage'))
    with pytest.raises(ValueError,match='manifest'):call(root,dict(request,manifest_sha256='0'*64),action='stage')
    p=run/'scratch/worker_0/generation_0'
    write_json(p/'NATIVE_PROCESS.json',{'native':{'pid':os.getpid(),'start_ticks':ticks(os.getpid())}})
    (p/'foreign.bin').symlink_to(tmp_path/'never-touch')
    assert call(root,request,action='stage')['status']=='SKIP'


def test_changed_source_after_copy_is_never_deleted(tmp_path):
    root,run,request=fleet(tmp_path)
    chunk=call(root,request,action='stage')
    receipt=sync.verify_chunk(chunk,chunk['archive'],tmp_path/'local')
    (run/'worker_0_generation_0.log').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed source'):
        call(root,request,action='ack',chunk=chunk,verified_archive_sha256=chunk['archive_sha256'],local_receipt_sha256=sha(receipt))
    assert (run/'worker_0_generation_0.log').read_bytes()==b'changed'


def test_setup_writes_private_authenticated_config_without_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(sync,'ROOT',tmp_path)
    sync.setup(tmp_path/'sync',23197)
    assert (tmp_path/'sync/rsync.password').stat().st_mode & 0o077==0
    config=(tmp_path/'sync/rsyncd.conf').read_text()
    assert 'auth users = q_incremental' in config and 'use chroot = no' in config
    assert json.loads((tmp_path/'sync/CONFIG.json').read_text())['launched'] is False


def test_snapshot_lock_is_not_exported(tmp_path):
    root,run,request=fleet(tmp_path)
    info=snapshot_run(run,tmp_path/'run.tar')
    assert '.incremental_sync.lock' not in info['manifest']['files']


def test_bulk_push_uses_rsync_bandwidth_limit_and_leaves_sources_until_ack(tmp_path,monkeypatch):
    import subprocess,uuid
    from unittest import mock
    root,run,request=fleet(tmp_path)
    chunk=call(root,request,action='stage')
    password=Path('/tmp/r17_q_incremental')/(uuid.uuid4().hex+'.password')
    password.parent.mkdir(exist_ok=True);password.write_text('private-test-secret\n');password.chmod(0o600)
    with mock.patch.object(subprocess,'run',return_value=mock.Mock(returncode=0)) as transfer:
        result=call(root,request,action='push',chunk=chunk,password_file=str(password),
            url='rsync://q_incremental@ziyanglin.com:23197/r17q/run/timan107/'+chunk['token']+'.tar.gz',bwlimit_kibps=4096)
    argv=transfer.call_args.args[0]
    assert argv[0]=='rsync' and '--bwlimit=4096' in argv and 'tether' not in argv
    assert all('private-test-secret' not in x for x in argv)
    assert result['status']=='PUSHED' and not password.exists()
    assert (run/'worker_0_generation_0.log').exists()


def test_tampered_preserved_chunk_blocks_sealing(tmp_path):
    root,run,request=fleet(tmp_path)
    chunk=call(root,request,action='stage')
    campaign=tmp_path/'local';campaign.mkdir()
    base=campaign/'INCREMENTAL_SYNC/timan107'
    receipt=sync.verify_chunk(chunk,chunk['archive'],base)
    call(root,request,action='ack',chunk=chunk,verified_archive_sha256=chunk['archive_sha256'],local_receipt_sha256=sha(receipt))
    archive=tmp_path/'closed.tar';snap=snapshot_run(run,archive)
    install_snapshot(archive,campaign/'workers/timan107',expected_sha256=snap['archive_sha256'])
    (base/'versions'/chunk['token']/'receipt.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='preserved chunk file changed'):sync.restore(campaign,'timan107')


def test_active_scan_deduplicates_prepared_aliases_and_skips_stale_claims(tmp_path,monkeypatch):
    run=tmp_path/'live';run.mkdir();write_json(run/'WORKER_DEPLOYMENT.json',{})
    write_json(run/'LAUNCH.json',{'pid':os.getpid()})
    alias=tmp_path/'old_alias';alias.symlink_to(run,target_is_directory=True)
    stale=tmp_path/'stale';stale.mkdir();write_json(stale/'WORKER_DEPLOYMENT.json',{})
    write_json(stale/'LAUNCH.json',{'pid':987654321})
    original=Path.read_bytes
    def argv(path):
        if str(path)=='/proc/'+str(os.getpid())+'/cmdline':
            return ('python\0-m\0exp.r17_integration.ops.formal_eval\0--prepared-run\0'+str(alias)+'\0').encode()
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',argv)
    assert sync.active_campaigns(tmp_path)==[run]
    write_json(run/'GPU_RELEASED.json',{})
    assert sync.active_campaigns(tmp_path)==[]


def test_verified_chunk_cleanup_after_finalizer_removes_the_run(tmp_path):
    root,run,request=fleet(tmp_path)
    chunk=call(root,request,action='stage')
    receipt=sync.verify_chunk(chunk,chunk['archive'],tmp_path/'local')
    run.rename(root/'preserved_after_cleanup')
    result=call(root,request,action='ack',chunk=chunk,verified_archive_sha256=chunk['archive_sha256'],local_receipt_sha256=sha(receipt))
    assert result['run_already_cleaned'] and not Path(chunk['archive']).exists()
    assert not run.exists()
