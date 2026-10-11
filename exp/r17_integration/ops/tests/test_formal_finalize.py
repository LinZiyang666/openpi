"""Real CPU archive/background handoff plus strict shutdown/queue regressions."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
from unittest import mock

import pytest

from exp.r17_integration.ops import formal_eval as launch, formal_finalize as seal
from exp.r17_integration.ops import formal_recovery, formal_queue, formal_resume, queue_retry
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.immutable_release import archive_manifest, identity, sha
from exp.r17_integration.ops.sync_back import snapshot_run, install_snapshot


@pytest.fixture
def complete(tmp_path):
    directory=tmp_path/'campaign';directory.mkdir();(directory/'assets').mkdir();(directory/'driver').mkdir()
    model={'default_sampler':{'num_steps':2}}
    campaign=dict(run_name='closed',model='dp',domain='libero',server_node='local',replicas=2,
                  workers=2,smoke=False,local_assets=str(directory/'assets'))
    write_json(directory/'campaign.json',campaign)
    tasks=[dict(task_uid=uid,attempt=1,experiment='suite',extra={'task_name':uid}) for uid in ('a','b')]
    write_json(directory/'EPISODE_PLAN.json',dict(tasks=tasks,model=model,benchmark_protocol='sealed'))
    write_json(directory/'assets/model_spec.json',model)
    write_json(directory/'assets/model_manifest.json',dict(spec=model,content_digests={'weights':'frozen'}))
    write_json(directory/'SERVER_ATTESTATIONS.json',[dict(metadata=dict(r17_model_spec=model,
        cache_enabled=False,debug_capture=False))]*2)
    for name,rows in [('journal',[dict(task_uid=t['task_uid'],attempt=1,accepted=True,status='done',
        success=True,error=None) for t in tasks]),('attempts',[dict(task_uid=t['task_uid'],attempt=1,
        accepted=True,success=True,error=None,worker_id='w'+t['task_uid']) for t in tasks]),
        ('per_step',[dict(task_uid=t['task_uid'],attempt=1,accepted=True,source='policy',capture_enabled=False,
        actual_nfe=2,stage_calls=[1,1,2]) for t in tasks])]:
        (directory/'driver'/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write_json(directory/'driver/DRIVER_RECEIPT.json',dict(status='PARTIAL',all_done=True,error=None,
        rejected_worker_probes=0,worker_shutdown=dict(status='PARTIAL',missing_worker_ids=['wa','wb']),elapsed_seconds=1))
    worker=tmp_path/'worker';worker.mkdir()
    write_json(worker/'manifest.json',dict(run='closed'))
    write_json(directory/'timan107_launch.json',dict(run='closed'))
    write_json(worker/'receipt.json',dict(status='PARTIAL',reason='interrupted',worker_failures=[]))
    value=snapshot_run(worker,directory/'timan107.tar')
    row=dict(node='timan107',label='workers',status='CLOSED',exit_code=2,alive=False,
             child_alive=False,supervisor_alive=False,owned_worker_survivors=[],closure_confirmed=True)
    report=dict(status='PARTIAL',all_owned_closed=True,owned_local=[],
        owned_remote=[dict(node='timan107',label='workers')],cleanup=[row],
        driver_completion=launch.driver_completion(directory,2),transfers={'timan107':dict(remote=value,verified={})})
    write_json(directory/'RECEIPT.json',report)
    deployment={'timan107':dict(worker_run='/scratch/closed_timan107',python=sys.executable,payload=str(launch.ROOT))}
    write_json(directory/'WORKER_DEPLOYMENT.json',deployment)
    return directory,campaign,deployment,report


@pytest.mark.parametrize('model',['dp','fastwam','pi05','groot','cosmos'])
def test_shutdown_gap_pass_after_closed_archive_and_full_execution(complete,model):
    directory,campaign,deployment,report=complete
    campaign['model']=model
    raw=(directory/'driver/DRIVER_RECEIPT.json').read_bytes()
    result=seal.finalize(campaign,directory,deployment,report)
    assert result['status']=='PASS' and result['evidence_verified'] and not result['evidence_pending']
    assert result['worker_shutdown_gap']['worker_shutdown']['missing_worker_ids']==['wa','wb']
    assert result['worker_shutdown_gap']['fleet_closure']['timan107']['owned_worker_survivors']==[]
    summary=seal.read(directory/'SUMMARY.json')
    assert summary['status']=='PASS' and summary['completed_episodes']==2 and summary['terminal_errors']==0
    assert summary['worker_shutdown_gap']==result['worker_shutdown_gap']
    assert (directory/'driver/DRIVER_RECEIPT.json').read_bytes()==raw


@pytest.mark.parametrize('change',['survivor','child_alive','supervisor_alive','abnormal_exit','cleanup',
    'owner_missing','journal_missing','journal_duplicate','unresolved_error','driver_error','worker_error','wrong_manifest'])
def test_fail_closed_before_scientific_pass(complete,change):
    directory,campaign,deployment,report=complete
    row=report['cleanup'][0]
    if change=='survivor':row['owned_worker_survivors']=[{'pid':42}]
    elif change in ('child_alive','supervisor_alive'):row[change]=True
    elif change=='abnormal_exit':row['exit_code']=-9
    elif change=='cleanup':row['closure_confirmed']=False
    elif change=='owner_missing':report['cleanup']=[]
    elif change.startswith('journal') or change=='unresolved_error':
        p=directory/'driver/journal.jsonl';rows=[json.loads(line) for line in p.read_text().splitlines()]
        if change=='journal_missing':rows.pop()
        elif change=='journal_duplicate':rows.append(rows[0])
        else:rows[0]['error']='unresolved'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif change=='driver_error':
        p=directory/'driver/DRIVER_RECEIPT.json';r=seal.read(p);r['error']='fatal';write_json(p,r)
    elif change=='worker_error':
        report['transfers']['timan107']['remote']['manifest']['partials']=['broken.part']
    else:write_json(directory/'timan107_launch.json',dict(run='other'))
    with pytest.raises((ValueError,RuntimeError)):
        seal.finalize(campaign,directory,deployment,report)
    assert seal.read(directory/'RECEIPT.json')['status']!='PASS'
    assert not (directory/'SUMMARY.json').exists()


def test_recovery_preserves_exact_original_and_records_missing_ack(complete):
    directory,campaign,_,_=complete
    original=(directory/'RECEIPT.json').read_bytes()
    result=formal_recovery.recover_shutdown(directory)
    assert result['status']=='PASS'
    assert (directory/'RECEIPT_BEFORE_SHUTDOWN_RECOVERY.json').read_bytes()==original
    assert seal.read(directory/'SHUTDOWN_RECOVERY.json')['original_receipt_sha256']==sha(
        directory/'RECEIPT_BEFORE_SHUTDOWN_RECOVERY.json')


@pytest.mark.parametrize('model',['dp','fastwam','pi05','groot','cosmos'])
def test_recover_c20_closed_snapshot_failure_keeps_original_and_audits(complete,model):
    directory,campaign,_,report=complete
    campaign['model']=model;write_json(directory/'campaign.json',campaign)
    report.update(status='FAIL',error="RuntimeError('remote worker close/snapshot incomplete')")
    report['cleanup'][0]['exit_code']=-9
    write_json(directory/'RECEIPT.json',report)
    original=(directory/'RECEIPT.json').read_bytes()
    result=formal_recovery.recover_shutdown(directory)
    assert result['status']=='PASS'
    assert (directory/'RECEIPT_BEFORE_SHUTDOWN_RECOVERY.json').read_bytes()==original
    sealed=seal.read(directory/'RECEIPT.json')
    assert sealed['worker_close_gap']['timan107']['exit_code']==-9
    assert sealed['shutdown_recovery']['original_error']==report['error']
    assert sealed['recovered_close_error']==report['error']


@pytest.mark.parametrize('change',['live','missing_cell','unresolved_fleet','wrong_error','other_failure'])
def test_c20_recovery_never_waives_real_failures(complete,change):
    directory,_,_,report=complete
    report.update(status='FAIL',error="RuntimeError('remote worker close/snapshot incomplete')")
    report['cleanup'][0]['exit_code']=-9
    if change=='live':report['cleanup'][0]['child_alive']=True
    elif change=='missing_cell':(directory/'driver/journal.jsonl').write_text('')
    elif change=='wrong_error':report['error']="RuntimeError('real worker death')"
    elif change=='other_failure':report['worker_fleet_failure']={'reason':'unresolved'}
    else:
        worker=directory.parent/'worker'
        write_json(worker/'receipt.json',dict(status='PARTIAL',reason='worker_restart_budget_exhausted',worker_failures=[]))
        report['transfers']['timan107']['remote']=snapshot_run(worker,directory/'timan107.tar')
    write_json(directory/'RECEIPT.json',report)
    with pytest.raises((ValueError,RuntimeError)):
        formal_recovery.recover_shutdown(directory)
    assert seal.read(directory/'RECEIPT.json')['status']!='PASS'


def test_rebuild_missing_native_snapshot_after_gone_fleet(complete,monkeypatch):
    directory,campaign,deployment,report=complete
    report['cleanup'][0]['exit_code']=-9
    seal.record_close_gap(report,'timan107',report['cleanup'][0])
    old=report.pop('transfers')['timan107']
    calls=[]
    monkeypatch.setattr(launch,'remote',lambda *a,**kw:'null')
    def rebuild(*args):
        calls.append(args)
        return dict(old['remote'],rebuilt_closed_snapshot=True,formal_max_bytes=seal.MAX_BYTES)
    monkeypatch.setattr(seal,'worker_snapshot',rebuild)
    def pull(node,value,destination,**kwargs):
        verified=install_snapshot(directory/'timan107.tar',destination,expected_sha256=value['archive_sha256'])
        return dict(old,remote=value,verified=verified)
    monkeypatch.setattr(seal,'pull_snapshot',pull)
    result=seal.finalize(campaign,directory,deployment,report)
    assert result['status']=='PASS' and len(calls)==1
    assert result['transfers']['timan107']['remote']['rebuilt_closed_snapshot']


def test_rc0_completed_driver_can_recover_only_snapshot_close(complete):
    directory,campaign,deployment,report=complete
    raw=seal.read(directory/'driver/DRIVER_RECEIPT.json')
    raw.update(status='PASS',worker_shutdown={'status':'PASS'})
    write_json(directory/'driver/DRIVER_RECEIPT.json',raw)
    report['driver_completion']=launch.driver_completion(directory,0)
    report.update(status='FAIL',error="RuntimeError('remote worker close/snapshot incomplete')")
    report['cleanup'][0]['exit_code']=-9
    assert seal.completed_close_error(directory,report)
    assert not report.get('error') and not report['driver_completion']['recovered_worker_shutdown']
    assert seal.finalize(campaign,directory,deployment,report)['status']=='PASS'
    assert seal.read(directory/'SUMMARY.json')['worker_shutdown_gap']['worker_close_gap']['timan107']['exit_code']==-9


def test_worker_source_proves_pid_ticks_manifest_and_receipt(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    owner=dict(pid=99999999,start_ticks='old')
    write_json(source/'manifest.json',dict(campaign='owned'))
    write_json(source/'receipt.json',dict(status='PARTIAL',reason='interrupted',worker_failures=[],
        fleet_owner=owner,pids=[dict(pid=99999998,start_ticks='worker',index=0,generation=0)]))
    manifest=sha(source/'manifest.json')
    seal.closed_worker_source(source,[owner],manifest,True)
    with pytest.raises(ValueError,match='manifest'):
        seal.closed_worker_source(source,[owner],'wrong',True)
    with pytest.raises(ValueError,match='unresolved'):
        seal.closed_worker_source(source,[owner],manifest,False)
    monkeypatch.setattr(seal,'alive',lambda p:p['pid']==99999998)
    with pytest.raises(ValueError,match='surviving'):
        seal.closed_worker_source(source,[owner],manifest,True)


def test_actual_by_value_worker_snapshot_preserves_native_files(tmp_path,monkeypatch):
    import contextlib
    import io
    from exp.r17_integration.qualification import archive_stage
    directory=tmp_path/'campaign';directory.mkdir()
    run=tmp_path/'native_source';run.mkdir()
    owner=dict(pid=99999999,start_ticks='closed')
    manifest=dict(campaign='frozen',worker_node='timan108')
    write_json(run/'manifest.json',manifest);write_json(directory/'timan108_launch.json',manifest)
    write_json(run/'receipt.json',dict(status='PARTIAL',reason='interrupted',worker_failures=[],
                                     fleet_owner=owner,pids=[]))
    (run/'worker.log').write_bytes(b'unchanged original evidence\n'*10000)
    original={p.name:sha(p) for p in run.iterdir()}
    monkeypatch.setattr(archive_stage,'STAGE_ROOT',tmp_path/'allowed_tmp_stage')
    def remote(node,argv,**kwargs):
        assert node=='timan108' and kwargs['replay_safe']
        output=io.StringIO();old_bound=archive_stage.MAX_BYTES
        try:
            with contextlib.redirect_stdout(output):exec(argv[-1],{})
        finally:archive_stage.MAX_BYTES=old_bound
        return output.getvalue()
    monkeypatch.setattr(launch,'remote',remote)
    item=dict(stage=str(tmp_path/'private_stage'),worker_run=str(run),payload=str(launch.ROOT),python=sys.executable)
    value=seal.worker_snapshot({},directory,'timan108',item,dict(child=owner),True)
    verified=install_snapshot(value['archive'],tmp_path/'verified',expected_sha256=value['archive_sha256'])
    assert verified['digest']==value['manifest']['digest'] and not verified['partials']
    assert value['compressed'] and value['rebuilt_closed_snapshot']
    assert value['formal_max_bytes']==seal.MAX_BYTES
    assert all(sha(run/name)==pin for name,pin in original.items())


def test_pending_source_cannot_resume_or_pair(complete):
    directory,campaign,_,report=complete
    report['status']='PENDING_EVIDENCE';write_json(directory/'RECEIPT.json',report)
    with pytest.raises(ValueError,match='sealing is pending'):
        formal_resume.inspect_source(directory)
    # The existing paired-control gate requires receipt.status == PASS.
    from exp.r17_integration.ops.formal_runtime import paired_lineage
    with pytest.raises((ValueError,KeyError)):
        paired_lineage(dict(campaign,paired_control=str(directory)),directory,require_complete=True)


def test_compressed_snapshot_keeps_every_file_and_sha(tmp_path):
    source=tmp_path/'servers';source.mkdir();write_json(source/'manifest.json',{'frozen':'campaign'})
    (source/'serving.jsonl').write_bytes(b'large resident decision evidence\n'*100000)
    archive=tmp_path/'servers.tar'
    proof=snapshot_run(source,archive,compressed=True)
    assert archive.stat().st_size<(source/'serving.jsonl').stat().st_size//20
    verified=install_snapshot(archive,tmp_path/'installed',expected_sha256=proof['archive_sha256'])
    assert verified['digest']==proof['manifest']['digest'] and not verified['partials']
    assert sha(source/'serving.jsonl')==sha(tmp_path/'installed/serving.jsonl')


def test_formal_size_override_is_bounded_owned_and_does_not_change_default(tmp_path,monkeypatch):
    from exp.r17_integration.qualification import archive_stage
    from exp.r17_integration.ops import transport_operations
    monkeypatch.setattr(archive_stage,'STAGE_ROOT',tmp_path/'stage')
    source=tmp_path/'source.tar';source.write_bytes(b'closed archive'*1000)
    value=archive_stage.stage_archive(source,'owned_servers','workers',sha(source))
    monkeypatch.setattr(archive_stage,'MAX_BYTES',1024)
    request=dict(archive=value['archive'],key='owned_servers',sha256=sha(source))
    with pytest.raises(ValueError,match='SHA/size'):
        transport_operations.cleanup_archive(request)
    assert Path(value['archive']).exists()
    with pytest.raises(ValueError,match='size bound'):
        transport_operations.cleanup_archive(dict(request,max_bytes=2**50))
    result=transport_operations.cleanup_archive(dict(request,max_bytes=seal.MAX_BYTES))
    assert result['status']=='PASS' and source.exists()
    assert archive_stage.MAX_BYTES==1024


def test_native_audit_keeps_cache_schema_and_requires_pure_batch_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(seal,'verify',lambda *a:{'present':True})
    calls=[]
    def execute(argv,**kw):
        calls.append((argv,kw))
        return json.dumps({'status':'PASS','decisions':2})
    monkeypatch.setattr(launch,'execute',execute)
    campaign=dict(runtime='/frozen/native',runtime_identity='pin',cache_study=True,
                  fastwam_batching={'max_batch':16},fastwam_batch_treatment_identity='qualified')
    assert seal.native_batch_audit(campaign,tmp_path)['status']=='PASS'
    argv,kw=calls[0]
    assert 'audit_batches' in argv[-1] and kw['cwd']=='/frozen/native'
    assert kw['env']['CUDA_VISIBLE_DEVICES']==''
    campaign['cache_study']=False
    with pytest.raises(ValueError,match='batching audit failed'):
        seal.native_batch_audit(campaign,tmp_path)
    assert 'audit_decisions' in calls[1][0][-1]


def test_handoff_uses_cpu_only_detached_child_and_no_inherited_gpu_lock(complete,monkeypatch):
    directory,campaign,deployment,report=complete
    monkeypatch.setattr(seal,'pin_release',lambda *a:dict(root=str(launch.ROOT),identity='evidence-only'))
    class Child:pid=99999999
    calls=[]
    monkeypatch.setattr(seal.subprocess,'Popen',lambda argv,**kw:calls.append((argv,kw)) or Child())
    monkeypatch.setattr(seal,'await_ready',lambda *a:None)
    result=seal.handoff(campaign,directory,deployment,report)
    argv,kw=calls[0]
    assert result['status']=='PENDING_EVIDENCE' and result['gpu_released']
    assert kw['close_fds'] and kw['start_new_session']
    assert kw['env']['CUDA_VISIBLE_DEVICES']=='' and kw['env']['JAX_PLATFORMS']=='cpu'
    assert 'formal_server' not in str(argv) and 'nvidia' not in str(argv)
    assert seal.read(directory/'GPU_RELEASED.json')['all_owned_closed']
    assert (directory/'EVIDENCE_FINALIZER_START.json').exists()


@pytest.mark.parametrize('legacy_overlay',[False,True])
def test_real_background_cpu_seal_and_native_identity_unchanged(complete,monkeypatch,tmp_path,legacy_overlay):
    directory,campaign,deployment,report=complete
    if legacy_overlay:
        monkeypatch.setattr(seal,'OVERLAY',tuple(n for n in seal.OVERLAY if n not in ('worker_storage','worker_disk')))
    # A small actual native release, containing the CPU helpers and a sentinel
    # model file; the background child executes real journal/archive checks.
    base_names=['exp/__init__.py','exp/r17_integration/ops/immutable_release.py',
        'exp/r17_integration/ops/build_client_bundle.py']
    with tarfile.open(directory/'runtime.tar','w') as tar:
        for name in base_names:
            data=(launch.ROOT/name).read_bytes();m=tarfile.TarInfo(name);m.size=len(data);tar.addfile(m,io.BytesIO(data))
        data=b'frozen model bytes';m=tarfile.TarInfo('exp/r17_integration/models/native_sentinel.py')
        m.size=len(data);tar.addfile(m,io.BytesIO(data))
    campaign['runtime_identity']=identity(archive_manifest(directory/'runtime.tar'))
    write_json(directory/'campaign.json',campaign)
    # Immutable installer roots stay under the CPU test temp directory.
    real_install=seal.install
    def install(archive,root,a_sha,m_identity):
        root=tmp_path/Path(root).name
        return real_install(archive,root,a_sha,m_identity)
    original_pin=seal.pin_release
    def pin(c,d):
        with mock.patch.object(seal,'install',install):
            value=original_pin(c,d)
        value['root']=str(tmp_path/Path(value['root']).name)
        seal.durable_json(d/'EVIDENCE_ORCHESTRATION.json',value)
        return value
    monkeypatch.setattr(seal,'pin_release',pin)
    native_before=sha(directory/'runtime.tar')
    child=[];popen=seal.subprocess.Popen
    def spawn(*a,**kw):
        proc=popen(*a,**kw);child.append(proc);return proc
    monkeypatch.setattr(seal.subprocess,'Popen',spawn)
    result=seal.handoff(campaign,directory,deployment,report)
    assert result['status']=='PENDING_EVIDENCE'
    assert seal.wait_seal(directory,seconds=30)['status']=='PASS'
    assert child[0].wait(timeout=10)==0
    assert sha(directory/'runtime.tar')==native_before
    pin=seal.read(directory/'EVIDENCE_ORCHESTRATION.json')
    assert (Path(pin['root'])/'exp/r17_integration/models/native_sentinel.py').read_bytes()==b'frozen model bytes'
    assert pin['native_runtime_identity']==campaign['runtime_identity']
    assert seal.read(directory/'EVIDENCE_FINALIZATION.json')['status']=='PASS'
    if legacy_overlay:
        assert 'legacy CPU overlay' in seal.read(directory/'RECEIPT.json')['worker_storage_cleanup']['timan107']['reason']


def test_cleanup_follows_immutable_scientific_seal_before_public_pass(complete,monkeypatch):
    from exp.r17_integration.ops import worker_storage
    directory,campaign,deployment,report=complete
    observations=[]
    def cleanup(c,d,deployment,report):
        public=seal.read(d/'RECEIPT.json')
        scientific=seal.read(d/'SEALED_EVIDENCE_RECEIPT.json')
        assert public['status']=='PENDING_EVIDENCE' and scientific['status']=='PASS'
        assert scientific['evidence_verified'] and scientific['all_owned_closed']
        assert public['scientific_evidence_seal']['sha256']==sha(d/'SEALED_EVIDENCE_RECEIPT.json')
        observations.append(public['status'])
        report['worker_storage_cleanup']={'cpu':dict(status='REMOVED')}
    monkeypatch.setattr(worker_storage,'cleanup',cleanup)
    result=seal.finalize(campaign,directory,deployment,report)
    assert observations==['PENDING_EVIDENCE'] and result['status']=='PASS'
    assert result['worker_storage_cleanup']['cpu']['status']=='REMOVED'
    assert seal.read(directory/'RECEIPT.json')['worker_storage_cleanup_pending'] is False


def test_queue_can_switch_pending_then_waits_all_seals(complete,monkeypatch):
    directory,_,_,report=complete
    report.update(status='PENDING_EVIDENCE',gpu_released=True);write_json(directory/'RECEIPT.json',report)
    write_json(directory/'GPU_RELEASED.json',{'status':'GPU_RELEASED'})
    class Runner:
        stopped=False
        def invoke(self,*args):pass
    runner=Runner()
    result=queue_retry.run_job(dict(name='campaign',run=str(directory)),runner)
    assert result['status']=='PENDING_EVIDENCE'
    calls=[]
    monkeypatch.setattr(seal,'wait_seal',lambda path,**kw:calls.append(path) or {'status':'PASS'})
    formal_queue.check_seals([result],runner)
    assert calls==[]
    formal_queue.check_seals([result],runner,wait=True)
    assert calls==[directory] and result['status']=='PASS'


def test_queue_stops_on_late_evidence_failure_and_unverified_pending(complete):
    directory,_,_,report=complete
    report.update(status='FAIL',finalization_error='archive SHA mismatch')
    write_json(directory/'RECEIPT.json',report)
    with pytest.raises(RuntimeError,match='archive SHA mismatch'):
        formal_queue.check_seals([dict(run=str(directory))],type('Runner',(),{'stopped':False})())
    report.update(status='PENDING_EVIDENCE',gpu_released=False);write_json(directory/'RECEIPT.json',report)
    class Runner:
        stopped=False
        def invoke(self,*args):pass
    with pytest.raises(RuntimeError):
        queue_retry.run_job(dict(name='campaign',run=str(directory)),Runner(),limit=0)


def test_next_job_dispatches_during_blocked_pull_and_queue_waits_to_seal(complete,monkeypatch,tmp_path):
    directory,campaign,deployment,report=complete
    value=report['transfers']['timan107']['remote']
    archive=directory/'timan107.tar';preserved=tmp_path/'remote_archive.tar';archive.rename(preserved)
    value['archive']='/tmp/r17_Q_qualification/closed_timan107/workers_'+value['archive_sha256']+'_'+'a'*32+'.tar'
    report['transfers']={}
    report['gpu_released']=True
    write_json(directory/'GPU_RELEASED.json',dict(status='GPU_RELEASED'))
    started,release,finished=threading.Event(),threading.Event(),threading.Event()
    def transfer(argv,**kw):
        started.set()
        assert release.wait(10), 'dispatch blocked on the previous evidence pull'
        Path(argv[3]).write_bytes(preserved.read_bytes())
    monkeypatch.setattr(launch,'remote',lambda *a,**kw:json.dumps(value))
    monkeypatch.setattr(launch,'transfer',transfer)
    from exp.r17_integration.ops import transport_operations
    monkeypatch.setattr(transport_operations,'cleanup',lambda *a,**kw:dict(status='PASS',stage_removed=True))
    errors=[]
    def finalizer():
        try:seal.finalize(campaign,directory,deployment,report)
        except BaseException as exc:errors.append(exc)
        finally:finished.set()
    thread=threading.Thread(target=finalizer);thread.start()
    next_run=tmp_path/'next';next_run.mkdir()
    class Runner:
        stopped=False
        def invoke(self,job,*args):
            if job['run']==str(next_run):
                assert not finished.is_set()
                assert seal.read(directory/'RECEIPT.json')['status']=='PENDING_EVIDENCE'
                write_json(next_run/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
                release.set()
    try:
        assert started.wait(10)
        results=formal_queue.run_prepared_jobs([dict(name='closed',run=str(directory)),
            dict(name='next',run=str(next_run))],Runner(),startup_retries=0)
        assert all(row['status']=='PASS' for row in results)
        assert results[0]['evidence_seal']['status']=='PASS'
        assert finished.wait(10) and not errors
        assert seal.read(directory/'RECEIPT.json')['evidence_verified']
    finally:
        release.set();thread.join(10)


def test_no_transfer_before_gpu_closure_in_normal_path():
    # Every network evidence call is in the CPU finalizer, outside the live
    # run/dispatch loop. Behavioral handoff tests above enforce the proof gate.
    # The shared checkout may change after this module is imported. A cached
    # code object's firstlineno then addresses the wrong current source lines.
    # This is a structural check: parse one current file snapshot by name.
    import ast
    text=Path(launch.__file__).read_text()
    function=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run_campaign')
    source=ast.get_source_segment(text,function)
    before_cleanup=source.split('    finally:\n        for owned in reversed(remote_owned):')[0]
    assert "['tether', 'pull'" not in before_cleanup
    assert source.index('mps_daemon')<source.index('report = handoff(')


def test_structural_source_check_survives_stale_import_line_numbers(monkeypatch):
    # Reproduce a module loaded before another package inserts source lines.
    # The old inspect.getsource check reads an unrelated current function.
    namespace={}
    exec(compile('def run_campaign():\n    return None\n',launch.__file__,'exec'),namespace)
    monkeypatch.setattr(launch,'run_campaign',namespace['run_campaign'])
    test_no_transfer_before_gpu_closure_in_normal_path()
