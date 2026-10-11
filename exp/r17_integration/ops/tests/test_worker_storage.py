"""CPU regressions for root-filesystem exhaustion and sealed-copy cleanup."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from exp.r17_integration.ops import formal_eval as launch, formal_workers as fleet
from exp.r17_integration.ops import worker_disk as disk, worker_storage as storage
from exp.r17_integration.ops.formal_process import ticks, write_json
from exp.r17_integration.ops.sync_back import snapshot_run, install_snapshot


@pytest.fixture
def closed(tmp_path,monkeypatch):
    root = tmp_path/'runs';root.mkdir()
    monkeypatch.setattr(storage,'REMOTE_RUNS',str(root))
    monkeypatch.setattr(storage,'LOCAL_RUNS',str(root))
    run = root/'closed_timan107';run.mkdir()
    owner = dict(pid=999999999,start_ticks='gone')
    write_json(run/'manifest.json',dict(run_name='closed',worker_node='timan107'))
    write_json(run/'receipt.json',dict(status='PASS',fleet_owner=owner,pids=[]))
    (run/'nested').mkdir();(run/'nested/data.log').write_bytes(b'complete evidence')
    archive = Path(str(run)+'.tar')
    value = snapshot_run(run,archive,compressed=True)
    write_json(Path(str(run)+'.transfer.json'),value)
    directory = tmp_path/'local';directory.mkdir()
    (directory/'timan107.tar').write_bytes(archive.read_bytes())
    install_snapshot(directory/'timan107.tar',directory/'workers/timan107',expected_sha256=value['archive_sha256'])
    proof = dict(node='timan107',run=str(run),run_name='closed',owners=[owner],
        source_archive=str(archive),sealed_status='PASS',sealed_at='sealed',evidence_verified=True,
        archive_sha256=value['archive_sha256'],archive_size_bytes=value['archive_size_bytes'],
        manifest_digest=value['manifest']['digest'],campaign_manifest_sha256=value['manifest']['files']['manifest.json'])
    return root,run,archive,directory,value,proof


def test_exact_closed_campaign_removed_with_plain_rm_and_local_evidence_retained(closed,monkeypatch):
    root,run,archive,directory,value,proof = closed
    other = root/'other_running';other.mkdir();(other/'keep').write_bytes(b'owned elsewhere')
    commands = [];native = subprocess.run
    def execute(argv,**kw):
        commands.append(argv)
        return native(argv,**kw)
    monkeypatch.setattr(storage.subprocess,'run',execute)
    result = storage.remove_closed_run(proof)
    assert result['status'] == 'REMOVED' and result['bytes_removed'] > 0
    assert not run.exists() and not archive.exists() and not Path(str(run)+'.transfer.json').exists()
    assert (other/'keep').read_bytes() == b'owned elsewhere'
    assert storage.file_sha(directory/'timan107.tar') == value['archive_sha256']
    assert (directory/'workers/timan107/nested/data.log').read_bytes() == b'complete evidence'
    assert all(a[0] in ('rm','rmdir') and a[1] == '--' for a in commands)
    assert storage.remove_closed_run(proof)['status'] == 'ALREADY_ABSENT'


@pytest.mark.parametrize('bad',['pending','failed','unsealed','sha','size','outside','source_outside',
                               'symlink','changed_file','added_partial','live','live_worker','busy','sidecar'])
def test_cleanup_refuses_without_deleting_any_evidence(closed,bad):
    root,run,archive,directory,value,proof = closed
    held = None
    if bad == 'pending':proof['evidence_verified'] = False
    elif bad == 'failed':proof['sealed_status'] = 'FAIL'
    elif bad == 'unsealed':proof['sealed_at'] = None
    elif bad == 'sha':proof['archive_sha256'] = '0'*64
    elif bad == 'size':proof['archive_size_bytes'] += 1
    elif bad == 'outside':proof['run'] = str(root.parent/'closed_timan107')
    elif bad == 'source_outside':proof['source_archive'] = str(root.parent/'foreign.tar')
    elif bad == 'symlink':(run/'foreign').symlink_to(directory,target_is_directory=True)
    elif bad == 'changed_file':(run/'nested/data.log').write_bytes(b'changed')
    elif bad == 'added_partial':(run/'incomplete.part').write_bytes(b'torn')
    elif bad == 'live':proof['owners'] = [dict(pid=os.getpid(),start_ticks=ticks(os.getpid()))]
    elif bad == 'live_worker':
        write_json(run/'receipt.json',dict(status='PASS',fleet_owner=proof['owners'][0],
            pids=[dict(pid=os.getpid(),start_ticks=ticks(os.getpid()))]))
        new = snapshot_run(run,archive,compressed=True)
        proof.update(archive_sha256=new['archive_sha256'],archive_size_bytes=new['archive_size_bytes'],
                     manifest_digest=new['manifest']['digest'])
    elif bad == 'busy':
        held = (run/'.eval.lock').open('rb');fcntl.flock(held,fcntl.LOCK_EX)
    else:write_json(Path(str(run)+'.transfer.json'),dict(archive_sha256='wrong'))
    try:
        with pytest.raises(ValueError):storage.remove_closed_run(proof)
    finally:
        if held:held.close()
    assert run.exists() and archive.exists() and (run/'manifest.json').exists()
    assert (directory/'timan107.tar').exists()


@pytest.mark.parametrize('node',['timan107','timan108','local','timan1'])
def test_same_cleanup_rules_for_every_worker_node(closed,node):
    root,run,archive,directory,value,proof = closed
    if node != 'timan107':
        run.rename(root/('closed_'+node));archive.rename(root/('closed_'+node+'.tar'))
        Path(str(run)+'.transfer.json').rename(root/('closed_'+node+'.transfer.json'))
        proof.update(node=node,run=str(root/('closed_'+node)),source_archive=str(root/('closed_'+node+'.tar')))
    assert storage.remove_closed_run(proof)['status'] == 'REMOVED'


def test_foreign_file_owner_is_refused_before_any_removal(closed,monkeypatch):
    root,run,archive,directory,value,proof = closed
    native = Path.stat
    def stat(path,**kwargs):
        result = native(path,**kwargs)
        if path == archive:
            fields = list(result);fields[4] = os.getuid()+1
            return os.stat_result(fields)
        return result
    monkeypatch.setattr(Path,'stat',stat)
    with pytest.raises(ValueError,match='foreign owner'):storage.remove_closed_run(proof)
    assert run.exists() and archive.exists() and (run/'nested/data.log').exists()


def test_sealed_archive_owner_must_equal_closed_process_owner(closed):
    root,run,archive,directory,value,proof = closed
    proof['owners'] = [dict(pid=999999998,start_ticks='different')]
    with pytest.raises(ValueError,match='fleet ownership'):storage.remove_closed_run(proof)
    assert run.exists() and archive.exists()


def report_for(value,proof):
    closure = dict(node='timan107',label='workers',status='CLOSED',exit_code=0,alive=False,
        child_alive=False,supervisor_alive=False,owned_worker_survivors=[],closure_confirmed=True,child=proof['owners'][0])
    return dict(status='PASS',sealed_at='sealed',evidence_verified=True,evidence_pending=False,
        all_owned_closed=True,owned_local=[],owned_remote=[dict(node='timan107',label='workers')],
        cleanup=[closure],transfers={'timan107':dict(remote=value)})


@pytest.mark.parametrize('scientific_witness',[False,True])
def test_automatic_cleanup_only_after_durable_seal_and_archive_verification(closed,monkeypatch,scientific_witness):
    root,run,archive,directory,value,proof = closed
    report = report_for(value,proof)
    deployment = dict(timan107=dict(worker_run=str(run),python=sys.executable))
    write_json(directory/'RECEIPT.json',dict(report,status='PENDING_EVIDENCE'))
    calls = []
    def remote(node,argv,**kw):
        calls.append(argv)
        assert json.loads((directory/'RECEIPT.json').read_text())['status'] == ('PENDING_EVIDENCE' if scientific_witness else 'PASS')
        result = subprocess.run(argv,capture_output=True,text=True,check=True)
        return result.stdout
    monkeypatch.setattr(launch,'remote',remote)
    with pytest.raises(ValueError,match='durable sealed PASS'):
        storage.cleanup(dict(run_name='closed'),directory,deployment,report)
    assert not calls and run.exists()
    if scientific_witness:
        witness = directory/'SEALED_EVIDENCE_RECEIPT.json'
        write_json(witness,report)
        report = dict(report,status='PENDING_EVIDENCE',evidence_pending=True,worker_storage_cleanup_pending=True,
            scientific_evidence_seal=dict(path=str(witness),sha256=storage.file_sha(witness)))
    write_json(directory/'RECEIPT.json',report)
    storage.cleanup(dict(run_name='closed'),directory,deployment,report)
    assert not run.exists() and not archive.exists() and len(calls) == 1
    recorded = json.loads((directory/'RECEIPT.json').read_text())['worker_storage_cleanup']['timan107']
    assert recorded['status'] == 'REMOVED' and recorded['archive_size_bytes'] == value['archive_size_bytes']
    assert recorded['local_archive_retained'] == str(directory/'timan107.tar')
    assert len(calls[0][-1].encode()) < 18000


@pytest.mark.parametrize('change',['sha','size','missing_size','manifest','local_install'])
def test_local_evidence_disagreement_retains_remote_copy(closed,monkeypatch,change):
    root,run,archive,directory,value,proof = closed
    report = report_for(value,proof)
    if change == 'sha':(directory/'timan107.tar').write_bytes(b'torn')
    elif change == 'size':value['archive_size_bytes'] += 1
    elif change == 'missing_size':value.pop('archive_size_bytes')
    elif change == 'manifest':value['manifest']['digest'] = 'wrong'
    else:(directory/'workers/timan107/nested/data.log').write_bytes(b'torn')
    write_json(directory/'RECEIPT.json',report)
    monkeypatch.setattr(launch,'remote',lambda *a,**kw:pytest.fail('unverified evidence cannot authorize remote deletion'))
    storage.cleanup(dict(run_name='closed'),directory,
        dict(timan107=dict(worker_run=str(run),python=sys.executable)),report)
    assert report['worker_storage_cleanup']['timan107']['status'] == 'RETAINED'
    assert run.exists() and archive.exists()


def usage(free,inodes=100):
    return SimpleNamespace(f_bavail=free,f_frsize=1,f_blocks=100*1024**3,f_files=1000,f_favail=inodes)


@pytest.mark.parametrize('free,admitted',[(8*1024**3-1,False),(8*1024**3,True),(20*1024**3,True)])
def test_disk_guard_counts_unprivileged_blocks_on_nearest_existing_filesystem(tmp_path,monkeypatch,free,admitted):
    observed = []
    monkeypatch.setattr(disk.os,'statvfs',lambda p:(observed.append(p) or usage(free)))
    path = tmp_path/'not_created/run'
    result = disk.admission(path,disk.minimum_bytes({}))
    assert observed == [tmp_path] and not path.exists()
    assert result['admitted'] is admitted and result['free_bytes'] == free
    assert result['minimum_free_bytes'] == 8*1024**3
    if not admitted:assert 'available_bytes=' in result['reason'] and 'minimum_free_bytes=' in result['reason']


def test_disk_guard_inode_exhaustion_and_configurable_threshold(tmp_path,monkeypatch):
    monkeypatch.setattr(disk.os,'statvfs',lambda p:usage(20*1024**3,0))
    assert not disk.admission(tmp_path,disk.minimum_bytes({}))['admitted']
    assert disk.minimum_bytes(dict(worker_min_free_gib=12)) == 12*1024**3
    for bad in (0,-1,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):disk.minimum_bytes(dict(worker_min_free_gib=bad))


@pytest.mark.parametrize('node',['local','timan107','timan1'])
def test_launcher_disk_probe_records_and_refuses_before_staging(tmp_path,monkeypatch,node):
    paths = [str(tmp_path/'runs/new'),str(tmp_path/'stage/new')]
    replies = [dict(path=p,admitted=False,free_bytes=123,reason='worker disk admission refused') for p in paths]
    monkeypatch.setattr(launch,'remote',lambda *a,**kw:json.dumps(replies))
    with pytest.raises(RuntimeError,match=node+': worker disk admission refused'):
        disk.probe({},tmp_path,node,paths[0],stage=paths[1])
    report = json.loads((tmp_path/(node+'_DISK_ADMISSION.json')).read_text())
    assert report['node'] == node and not report['admitted'] and report['filesystems'][0]['free_bytes'] == 123
    assert not Path(paths[0]).exists() and not Path(paths[1]).exists()


def test_fleet_disk_refusal_happens_before_run_directory_or_gpu_admission(tmp_path,monkeypatch,capsys):
    config = dict(worker_run=str(tmp_path/'new'))
    write_json(tmp_path/'campaign.json',config)
    monkeypatch.setattr(sys,'argv',['fleet','--campaign',str(tmp_path/'campaign.json')])
    monkeypatch.setattr(fleet,'disk_admission',lambda *a:dict(admitted=False,reason='worker disk admission refused: free_bytes=1'))
    with mock.patch('exp.r17_integration.ops.worker_gpu.acquire',side_effect=AssertionError('GPU touched')):
        with pytest.raises(SystemExit) as done:fleet.main()
    assert done.value.code == 2 and not (tmp_path/'new').exists()
    assert 'worker disk admission refused' in capsys.readouterr().err


def test_recycle_pauses_for_disk_then_reuses_same_slot(tmp_path,monkeypatch):
    config = dict(run_name='cpu',worker_node='cpu',worker_run=str(tmp_path/'run'),domain='libero',
        node_workers=1,seconds=30,endpoint='host:1',driver_host='host',driver_port=2,
        worker_benchmark='/sealed',worker_egl_device=0,worker_max_episodes=4,cache_study='sealed')
    write_json(tmp_path/'campaign.json',config)
    monkeypatch.setattr(sys,'argv',['fleet','--campaign',str(tmp_path/'campaign.json')])
    old = mock.Mock(pid=987654);old.poll.return_value=75;old.wait.return_value=75
    new = mock.Mock(pid=987653);new.poll.return_value=0;new.wait.return_value=0
    lease = mock.Mock();lease.result=dict(assignments=[dict(index=0,device=0,uuid='GPU-0',worker_vram_mib=576)])
    lease.reuse.return_value=dict(admitted=True,reused=True)
    ok = dict(admitted=True,free_bytes=20*1024**3,minimum_free_bytes=8*1024**3)
    low = dict(admitted=False,free_bytes=1,minimum_free_bytes=8*1024**3,reason='worker disk admission refused')
    spawned = []
    pauses = []
    def sleep(seconds):
        if not pauses:assert len(spawned) == 1
        pauses.append(seconds)
    monkeypatch.setattr(fleet,'disk_admission',mock.Mock(side_effect=[ok,ok,low,ok,ok]))
    monkeypatch.setattr(fleet.subprocess,'Popen',lambda *a,**kw:(spawned.append(a) or [old,new][len(spawned)-1]))
    with mock.patch('exp.r17_integration.ops.worker_gpu.acquire',return_value=lease), \
         mock.patch('exp.r17_integration.ops.worker_memory.admission',side_effect=[dict(admitted=True),AssertionError('fresh RAM')]), \
         mock.patch('exp.r17_integration.ops.worker_lifetime.planned_recycle',return_value=True), \
         mock.patch.object(fleet.time,'sleep',side_effect=sleep), \
         mock.patch('exp.r17_integration.ops.sync_back.snapshot_run',return_value=dict(archive_sha256='cpu')), \
         mock.patch('exp.r17_integration.qualification.archive_stage.stage_archive',return_value={}):
        with pytest.raises(SystemExit) as done:fleet.main()
    assert done.value.code == 0 and len(spawned) == 2
    receipt = json.loads((tmp_path/'run/receipt.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['disk_admission']['admitted']
    assert receipt['last_replacement']['disk_admission']['admitted'] and not receipt.get('dispatch_pause')
    assert receipt['last_replacement']['admission']['requested_extra_ram_bytes'] == 0
    assert pauses
    assert [e['event'] for e in receipt['worker_disk_events']] == ['replacement_paused','replacement_resumed']
    assert receipt['worker_disk_events'][0]['disk_admission']['free_bytes'] == 1


def test_prepare_guard_precedes_any_remote_mkdir_or_transfer(tmp_path,monkeypatch):
    from exp.r17_integration.ops import build_client_bundle
    monkeypatch.setattr(build_client_bundle,'worker_payload_files',lambda root:[])
    monkeypatch.setattr(build_client_bundle,'build_client_bundle',lambda *a:dict(archive_sha256='cpu'))
    commands = []
    def remote(node,argv,**kw):
        commands.append(argv)
        assert argv[:2] == ['python3','-c']
        return json.dumps([dict(path=p,admitted=False,free_bytes=1,reason='worker disk admission refused')
            for p in ('/scratch/zixuans8/mw_r17/formal/runs/new_timan107','/tmp/r17_formal/new/timan107')])
    monkeypatch.setattr(launch,'remote',remote)
    monkeypatch.setattr(launch,'transfer',lambda *a,**kw:pytest.fail('must not stage while disk is low'))
    with pytest.raises(RuntimeError,match='worker disk admission refused'):
        launch.prepare_workers(dict(run_name='new',worker_nodes={'timan107':1}),tmp_path)
    assert len(commands) == 1


def test_prepared_guard_precedes_refresh_and_one_shot_claim(tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_prestage as stage,cache_control
    run = tmp_path/'prepared';run.mkdir()
    monkeypatch.setattr(launch,'FORMAL_ROOT',tmp_path)
    monkeypatch.setattr(stage,'validate_prepared',lambda p:({'cache_study':'study','cache_control':'control'},{}))
    monkeypatch.setattr(disk,'prepared',mock.Mock(side_effect=RuntimeError('disk low: 1 byte')))
    monkeypatch.setattr(cache_control,'refresh_prepared_workers',lambda *a:pytest.fail('disk guard must precede refresh'))
    with pytest.raises(RuntimeError,match='disk low'):stage.launch_prepared(run)
    assert not (run/'LAUNCH.json').exists()


def test_local_probe_reuses_cpu_process_path_without_tether(tmp_path,monkeypatch):
    native = launch.execute
    commands = []
    def execute(argv,**kw):
        commands.append(argv)
        assert argv[0] != 'tether'
        return native(argv,**kw)
    monkeypatch.setattr(launch,'execute',execute)
    result = disk.probe({},tmp_path,'local',str(tmp_path/'run'),stage=str(tmp_path/'stage'),python=sys.executable)
    assert result['admitted'] and len(commands) == 1
    assert result['filesystems'][0]['filesystem_device'] == os.stat(tmp_path).st_dev


def test_seal_barrier_waits_for_cleanup_receipt_to_stop_changing(tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_finalize as seal
    final = dict(status='PASS',all_owned_closed=True,worker_storage_cleanup_pending=False)
    write_json(tmp_path/'RECEIPT.json',final)
    reads = [dict(final,worker_storage_cleanup_pending=True),final]
    monkeypatch.setattr(seal,'read',lambda p:reads.pop(0))
    sleeps = [];monkeypatch.setattr(seal.time,'sleep',lambda n:sleeps.append(n))
    result = seal.wait_seal(tmp_path)
    assert result['status'] == 'PASS' and sleeps == [.2] and not reads


def test_cleanup_pending_receipt_is_not_a_pin_source(tmp_path):
    from exp.r17_integration.ops import formal_resume,formal_runtime,cache_control,cache_resume,library_repair
    report = dict(status='PASS',all_owned_closed=True,worker_storage_cleanup_pending=True)
    write_json(tmp_path/'RECEIPT.json',report)
    write_json(tmp_path/'campaign.json',dict(cache_study='sealed'))
    write_json(tmp_path/'SUMMARY.json',{})
    with pytest.raises(ValueError,match='sealing is pending'):formal_resume.inspect_source(tmp_path)
    with pytest.raises(ValueError,match='sealing may still be pending'):
        formal_runtime.paired_lineage(dict(paired_control=str(tmp_path)),tmp_path,require_complete=True)
    with pytest.raises(ValueError,match='evidence sealed'):cache_control.control_proof(tmp_path,tmp_path,{})
    with pytest.raises(ValueError,match='evidence-sealed'):cache_resume.select(tmp_path)
    with pytest.raises(ValueError,match='closed collection receipt'):
        library_repair.closed_receipt({},tmp_path,passing=True)


def test_low_disk_recycle_timeout_is_bounded_and_concrete(tmp_path,monkeypatch):
    config = dict(run_name='cpu',worker_node='cpu',worker_run=str(tmp_path/'run'),domain='libero',
        node_workers=1,seconds=5000,endpoint='host:1',driver_host='host',driver_port=2,
        worker_benchmark='/sealed',worker_egl_device=0,worker_max_episodes=4,cache_study='sealed')
    write_json(tmp_path/'campaign.json',config)
    monkeypatch.setattr(sys,'argv',['fleet','--campaign',str(tmp_path/'campaign.json')])
    old = mock.Mock(pid=987654);old.poll.return_value=75;old.wait.return_value=75
    lease = mock.Mock();lease.result=dict(assignments=[dict(index=0,device=0,uuid='GPU-0',worker_vram_mib=576)])
    lease.reuse.return_value=dict(admitted=True,reused=True)
    clock = [0]
    monkeypatch.setattr(fleet.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(fleet.time,'sleep',lambda n:clock.__setitem__(0,clock[0]+601))
    ok = dict(admitted=True,free_bytes=20*1024**3)
    low = dict(admitted=False,free_bytes=1,reason='worker disk admission refused: available_bytes=1; minimum_free_bytes=8589934592')
    probes = [ok,ok]
    monkeypatch.setattr(fleet,'disk_admission',lambda *a:probes.pop(0) if probes else low)
    with mock.patch.object(fleet.subprocess,'Popen',return_value=old) as popen, \
         mock.patch('exp.r17_integration.ops.worker_gpu.acquire',return_value=lease), \
         mock.patch('exp.r17_integration.ops.worker_memory.admission',return_value=dict(admitted=True)), \
         mock.patch('exp.r17_integration.ops.worker_lifetime.planned_recycle',return_value=True), \
         mock.patch('exp.r17_integration.ops.sync_back.snapshot_run',return_value=dict(archive_sha256='cpu')), \
         mock.patch('exp.r17_integration.qualification.archive_stage.stage_archive',return_value={}):
        with pytest.raises(SystemExit) as done:fleet.main()
    assert done.value.code == 2 and popen.call_count == 1
    report = json.loads((tmp_path/'run/receipt.json').read_text())
    assert 'available_bytes=1' in report['reason'] and report['last_replacement']['remaining_seconds'] == 0
    assert report['last_replacement']['admission']['requested_extra_ram_bytes'] == 0
    assert report['disk_admission']['free_bytes'] == 1 and report['worker_disk_events']


def test_guard_configuration_is_exposed_by_both_launchers():
    from exp.r17_integration.ops.library_campaign import parse_args as library_args
    args = launch.parse_args(['--model','dp_libero','--prepare-only','--worker-min-free-gib','12'])
    assert args.worker_min_free_gib == 12
    assert library_args(['--model','dp_libero','--suite','libero_10','--prepare-only']).worker_min_free_gib == 8
    with pytest.raises(SystemExit):
        launch.parse_args(['--model','dp_libero','--prepare-only','--worker-min-free-gib','0'])


@pytest.mark.parametrize('cache_study',[False,True])
def test_long_cache_snapshot_bound_preserves_qualification_default(cache_study,monkeypatch):
    from exp.r17_integration.qualification import archive_stage
    previous = archive_stage.MAX_BYTES
    observed = []
    def stage(*args):
        observed.append(archive_stage.MAX_BYTES)
        return dict(archive_sha256='sealed')
    monkeypatch.setattr(archive_stage,'stage_archive',stage)
    result = fleet.stage_snapshot('archive','run',dict(archive_sha256='sealed'),cache_study=cache_study)
    assert observed == [8*1024**3 if cache_study else previous]
    assert archive_stage.MAX_BYTES == previous
    assert result.get('formal_max_bytes') == (8*1024**3 if cache_study else None)
    monkeypatch.setattr(archive_stage,'stage_archive',mock.Mock(side_effect=ValueError('oversize')))
    with pytest.raises(ValueError,match='oversize'):
        fleet.stage_snapshot('archive','run',dict(archive_sha256='sealed'),cache_study=cache_study)
    assert archive_stage.MAX_BYTES == previous
