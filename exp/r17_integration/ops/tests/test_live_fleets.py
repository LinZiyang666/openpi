"""CPU only: real installer/SHA stage, driver transport and supervisor paths."""
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from unittest import mock

import pytest

from exp.r17_integration.ops import live_fleets as live, live_fleet_worker as worker
from exp.r17_integration.ops import formal_eval as launch, formal_workers, formal_finalize
from exp.r17_integration.ops import local_workers, timan1_workers, driver_release
from exp.r17_integration.ops.formal_process import ticks, write_json
from exp.r17_integration.ops.build_client_bundle import build_client_bundle
from exp.r17_integration.ops.immutable_release import sha
from exp.r17_integration.ops.live_fleet_driver import configure
from exp.r17_integration.ops.tests.test_driver_group_commit import fixture, result, wait_until


def parent(tmp_path,domain='metaworld'):
    directory=tmp_path/'campaign';directory.mkdir();(directory/'assets').mkdir();(directory/'driver').mkdir()
    campaign=dict(run_name='cpu_live',model='fastwam',domain=domain,live_fleets=True,
        workers=2,worker_nodes={'timan107':2},server_node='h100',replicas=2,worker_egl_device=0,
        driver_host='fake',driver_port=1235,endpoint='fake:1234',public_port=1234,
        worker_memory_mib=1024,worker_min_free_gib=.01,worker_max_episodes=4,seconds=60,
        local_assets=str(directory/'assets'),runtime_identity='cpu_native',checkpoint='/sealed/native')
    write_json(directory/'campaign.json',campaign);write_json(directory/'assets/benchmark.json',{'identity':'sealed'})
    write_json(directory/'timan107.json',dict(campaign,worker_node='timan107',node_workers=2,worker_run='/unused'))
    build_client_bundle(launch.ROOT,['src/openpi/conductor/task.py',
        'exp/r17_integration/ops/live_fleet_worker.py',
        *['exp/r17_integration/ops/'+n+'.py' for n in
          ('build_client_bundle','sync_back','deploy','worker_installs','worker_memory','formal_process','remote_json',
           'formal_workers','worker_gpu','worker_disk','local_workers','timan1_workers','local_worker_entry')]],directory/'workers.tar')
    deployment={'timan107':dict(workers=2,worker_run='/unused',stage='/unused',payload=str(launch.ROOT),
                              env={},python=sys.executable)}
    registry=live.Registry(directory);registry.initialize(campaign,deployment)
    write_json(directory/'driver/LIVE_FLEET_DRIVER.json',dict(campaign_sha256=sha(directory/'campaign.json'),
        owner=dict(pid=os.getpid(),start_ticks=ticks(os.getpid()))))
    return directory,campaign,deployment,registry


def test_default_off_does_not_wrap_or_write(tmp_path):
    driver=mock.Mock();before=dict(driver.__dict__)
    assert configure(driver,{},tmp_path) is None
    assert driver.__dict__==before and not list(tmp_path.iterdir())


def test_old_frozen_campaign_refuses_without_writes(tmp_path):
    write_json(tmp_path/'campaign.json',{'workers':2})
    before=set(tmp_path.iterdir())
    with pytest.raises(ValueError,match='frozen campaign lacks'):
        live.Registry(tmp_path).request('attach',fleet_id='t124',node='timan1',workers=24,execute=True)
    assert set(tmp_path.iterdir())==before


@pytest.mark.parametrize('change',['seal','worker_archive','owner','closing','driver','bad_node','bad_name'])
def test_requests_fail_closed(parent_case,change):
    directory,campaign,deployment,registry=parent_case
    kwargs=dict(fleet_id='t124',node='timan1',workers=24,execute=True)
    if change=='seal':write_json(directory/'campaign.json',dict(campaign,checkpoint='/different'))
    elif change=='worker_archive':(directory/'workers.tar').write_bytes(b'changed')
    elif change=='owner':
        with registry.locked() as doc:
            doc['launcher_owner']['start_ticks']='wrong';registry.save(doc)
    elif change=='closing':
        with registry.locked() as doc:doc['state']='CLOSING';registry.save(doc)
    elif change=='driver':(directory/'driver/LIVE_FLEET_DRIVER.json').unlink()
    elif change=='bad_node':kwargs['node']='foreign'
    else:kwargs['fleet_id']='../foreign'
    with pytest.raises((ValueError,FileNotFoundError)):
        registry.request('attach',**kwargs)
    assert registry.read()['requests']=={}


@pytest.fixture
def parent_case(tmp_path):return parent(tmp_path)


def test_dry_run_duplicate_and_gpu3_authorization(parent_case):
    directory,campaign,_,registry=parent_case
    raw=registry.path.read_bytes()
    registry.request('attach',fleet_id='t124',node='timan1',workers=24)
    assert registry.path.read_bytes()==raw
    queued=registry.request('attach',fleet_id='t124',node='timan1',workers=24,execute=True)
    with pytest.raises(ValueError,match='already used'):
        registry.request('attach',fleet_id='t124',node='timan1',workers=24,execute=True)
    campaign['domain']='libero'
    doc=registry.read();doc['native_pin']=worker.native_pin(campaign)
    child=live.child_campaign(campaign,doc,queued,directory)
    lease=timan1_workers.authorize(child)
    assert lease['workers']==24 and lease['device']==3 and lease['uuid']==timan1_workers.UUID
    routed=timan1_workers.route(dict(child,worker_node='timan1',node_workers=24),child)
    assert routed['worker_egl_devices']==[3] and routed['worker_max_episodes']==4
    bad=dict(child,checkpoint='/changed')
    with pytest.raises(ValueError,match='native inputs'):timan1_workers.authorize(bad)
    plain=dict(child);plain.pop('live_fleet_authorization');plain['cache_study']='study'
    with pytest.raises(ValueError):timan1_workers.authorize(plain)


class CPUTransport:
    """No tether or renderer. Real bootstrap, filesystem/SHA and RAM code."""
    def __init__(self):self.calls=[]
    def transfer(self,argv,**kwargs):
        self.calls.append(argv)
        source=Path(argv[2]);target=Path(argv[3].split(':',1)[1])
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    def remote(self,node,argv,**kwargs):
        self.calls.append((node,argv))
        if '--import-probe' in argv:return json.dumps({'cpu_import_probe':True})
        if any('MetaWorldBenchmarkAdapter' in v for v in argv):return 'cpu benchmark probe'
        argv=list(argv);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
        if argv[0]=='env':
            argv.pop(0)
            while argv and '=' in argv[0]:
                key,value=argv.pop(0).split('=',1);env[key]=value
        if argv and argv[0].endswith('/bin/python'):argv[0]=sys.executable
        return subprocess.check_output(argv,env=env,text=True,stderr=subprocess.STDOUT,timeout=20)


def cpu_paths(tmp_path,campaign,node):
    root=tmp_path/'remote'
    return dict(stage=str(root/'stage'/campaign['run_name']/node),
        private=str(root/'private'/campaign['run_name']/node),
        run=str(root/'formal/runs'/(campaign['run_name']+'_'+node)))


def test_real_prepare_and_sha_staging_uses_exact_parent_archive(parent_case,tmp_path,monkeypatch):
    directory,campaign,_,registry=parent_case
    transport=CPUTransport()
    monkeypatch.setattr(launch,'remote',transport.remote);monkeypatch.setattr(launch,'transfer',transport.transfer)
    monkeypatch.setattr(launch,'worker_paths',lambda c,n:cpu_paths(tmp_path,c,n))
    row=registry.request('attach',fleet_id='extra',node='timan107',workers=2)
    child,item,key,manifest_sha=live.prepare_attachment(launch,campaign,directory,registry.read(),row)
    assert sha(directory/'workers.tar')==sha(directory/'fleets/extra/workers.tar')
    assert key=='fleet_extra' and child['endpoint']==campaign['endpoint']
    assert manifest_sha==sha(directory/'fleet_extra_launch.json')
    assert json.loads(Path(item['campaign_path']).read_text())==json.loads((directory/'fleet_extra_launch.json').read_text())
    assert sha(item['campaign_path'])==item['env']['R17_LIVE_FLEET_CONFIG_SHA256']
    assert item['campaign_transfer']['bytes']>0 and item['benchmark_transfer']['bytes']>0
    assert item['installed']['files']['src/openpi/conductor/task.py']
    # Concrete drain uses the same SHA-checked file transport and client helper.
    proof=live.drain_attachment(launch,item,'timan107')
    assert proof['drain_requested']
    worker_config=json.loads(Path(item['campaign_path']).read_text())
    assert worker.draining(worker_config)
    assert any('ram_admission' in str(call) or 'worker_memory import admission' in str(call) for call in transport.calls)


def test_resource_guard_failure_cannot_spawn(parent_case,tmp_path,monkeypatch):
    directory,campaign,_,registry=parent_case
    transport=CPUTransport();real_remote=transport.remote
    def remote(node,argv,**kwargs):
        if any('worker_memory import admission' in v for v in argv):
            return json.dumps(dict(admitted=False,reason='RAM exhausted'))
        return real_remote(node,argv,**kwargs)
    monkeypatch.setattr(launch,'remote',remote);monkeypatch.setattr(launch,'transfer',transport.transfer)
    monkeypatch.setattr(launch,'worker_paths',lambda c,n:cpu_paths(tmp_path,c,n))
    row=registry.request('attach',fleet_id='extra',node='timan107',workers=2)
    with pytest.raises(RuntimeError,match='RAM exhausted'):
        live.prepare_attachment(launch,campaign,directory,registry.read(),row)
    assert not any('formal_process' in str(call) and 'spawn' in str(call) for call in transport.calls)


def test_stock_tcp_driver_attach_dispatch_drain_complete_current(parent_case):
    from openpi.conductor import protocol as wire
    directory,campaign,_,registry=parent_case
    driver=fixture(directory/'driver',count=6)
    from exp.r17_integration.ops.group_commit import GroupCommit
    driver.evidence.close()
    driver.evidence=GroupCommit(directory/'driver',extra_streams=('fleet_dispatch.jsonl',))
    driver.scheduler.evidence=driver.evidence
    configure(driver,campaign,directory/'driver')
    wid='cpu_live_timan107_0'
    driver._shutdown_external_stop=lambda:False
    client,server=socket.socketpair();client.settimeout(5)
    thread=threading.Thread(target=driver._handle_conn,args=(server,lambda:False));thread.start()
    try:
        wire.send_message(client,wire.MSG_PULL,dict(worker_id=wid,server_host='fake:1234'))
        kind,payload=wire.recv_message(client);assert kind==wire.MSG_ASSIGN
        task=payload['task']
        # A new fleet registers against the same scheduler/driver, not a new
        # graph or scientific segment. Later it can dispatch remaining UIDs.
        with registry.locked() as doc:
            doc['fleets']['initial_timan107']['state']='DRAINING'
            doc['fleets']['new']=dict(fleet_id='new',node='timan1',workers=1,state='ACTIVE',
                evidence_key='fleet_new',worker_ids=['new_timan1_0'])
            registry.save(doc)
        wire.send_message(client,wire.MSG_REPORT_RESULT,result(task))
        wire.send_message(client,wire.MSG_PULL,dict(worker_id=wid,server_host='fake:1234'))
        kind,_=wire.recv_message(client);assert kind==wire.MSG_SHUTDOWN
    finally:
        client.close();thread.join(5);assert not thread.is_alive()
    wait_until(lambda:not driver.scheduler.committing or driver.evidence.failure)
    assert driver.evidence.failure is None, repr(driver.evidence.failure)
    driver._shutdown_local.worker_id='new_timan1_0'
    kind,payload=driver.handle_pull('fake:1234')
    assert payload['task']['task_uid']!=task['task_uid'] and payload['task']['attempt']==1
    assert driver.scheduler._eval_cc==1
    assert next(iter(driver.scheduler.endpoints.values())).capacity==1
    driver.handle_result(result(payload['task']));driver.close()
    census=driver.worker_census[wid]
    assert census['worker_node']=='timan107' and census['fleet_id']=='initial_timan107' and census['results']==1
    rows=[json.loads(s) for s in (directory/'driver/fleet_dispatch.jsonl').read_text().splitlines()]
    assert [r['worker_node'] for r in rows]==['timan107','timan1']
    accepted=[json.loads(s) for s in (directory/'driver/attempts.jsonl').read_text().splitlines()]
    assert len(accepted)==2 and all(r['accepted'] for r in accepted)
    with pytest.raises(ValueError,match='registered fleet'):
        driver._record_pull(dict(worker_id='foreign',server_host='fake:1234'),('127.0.0.1',1))


def test_owned_spawn_failure_records_cleanup_target_before_tether_call():
    recorded=[]
    api=mock.Mock()
    def remote(*a,**kw):
        assert recorded and recorded[0]['request']['receipt'].endswith('workers__fleet_extra.process.json')
        raise OSError('node_offline after remote spawn')
    api.remote=remote;api.pythonpath=lambda r:r
    with pytest.raises(OSError,match='node_offline'):
        live.owned_spawn(api,'timan107','workers__fleet_extra',['python','worker'],{},'/pin','python','/tmp/stage',
            seconds=60,lease_seconds=90,ownership=dict(kind='workers',fleet_key='fleet_extra',fleet_id='extra'),
            record=lambda o,first:recorded.append(o) if first else None)
    assert recorded[0]['receipt'] is None and recorded[0]['fleet_key']=='fleet_extra'


def test_launcher_finalization_fences_preparing_and_queued(parent_case,monkeypatch):
    directory,campaign,deployment,_=parent_case
    (directory/'LIVE_FLEETS.json').unlink()
    started=threading.Event();release=threading.Event();spawn=mock.Mock()
    from exp.r17_integration.ops.tether_retry import current_journal
    journal=object();token=current_journal.set(journal)
    def prepare(*args):
        assert current_journal.get() is journal
        started.set();assert release.wait(5)
        raise RuntimeError('synthetic preparation cancelled by campaign closure')
    monkeypatch.setattr(live,'prepare_attachment',prepare)
    owner=live.Launcher(launch,campaign,directory,deployment,spawn)
    owner.registry.request('attach',fleet_id='extra',node='timan107',workers=2,execute=True)
    owner.tick();current_journal.reset(token)
    assert started.wait(2);owner.freeze();release.set()
    owner.pool.shutdown(wait=True);owner.tick()
    assert not spawn.called
    assert list(owner.registry.read()['requests'].values())[0]['status']=='FAILED'
    with pytest.raises(ValueError,match='open pinned'):
        owner.registry.request('attach',fleet_id='late',node='timan107',workers=2,execute=True)


def test_real_supervisor_drain_does_not_replace_n4(tmp_path,monkeypatch):
    from exp.r17_integration.ops import worker_memory,worker_gpu
    from exp.r17_integration.qualification import archive_stage
    config=dict(run_name='cpu_drain',worker_node='timan107',worker_run=str(tmp_path/'run'),domain='metaworld',
        node_workers=1,seconds=10,endpoint='fake:1',driver_host='fake',driver_port=2,
        worker_egl_device=0,worker_max_episodes=4,worker_benchmark='/sealed',live_fleets=True)
    file=tmp_path/'campaign.json';write_json(file,config)
    real_popen=subprocess.Popen;children=[]
    def popen(argv,**kwargs):
        # Fake teacher episode, real process/supervisor: marker appears while
        # it runs; its completed N4 receipt is still valid, but no replacement.
        code=('import os,time; from exp.r17_integration.ops.formal_process import ticks,write_json; '
            'time.sleep(.15);write_json(os.environ["R17_WORKER_RECYCLE_RECEIPT"],dict('
            'schema="r17.worker_recycle.v1",pid=os.getpid(),start_ticks=ticks(os.getpid()),'
            'completed_episodes=4,maximum_episodes=4,reason="episode_boundary"));raise SystemExit(75)')
        env=dict(kwargs['env'],PYTHONPATH=launch.pythonpath(str(launch.ROOT)),CUDA_VISIBLE_DEVICES='')
        kwargs['env']=env
        child=real_popen([sys.executable,'-c',code],**kwargs);children.append(child)
        threading.Thread(target=worker.write_drain,args=(config,),daemon=True).start()
        return child
    monkeypatch.setattr(formal_workers.subprocess,'Popen',popen)
    monkeypatch.setattr(worker_gpu,'acquire',lambda *a:None)
    monkeypatch.setattr(worker_memory,'admission',lambda *a,**kw:dict(admitted=True))
    monkeypatch.setattr(formal_workers,'stage_snapshot',lambda *a,**kw:{})
    monkeypatch.setattr(sys,'argv',['formal_workers','--campaign',str(file)])
    with pytest.raises(SystemExit) as stopped:formal_workers.main()
    assert stopped.value.code==0 and len(children)==1 and children[0].returncode==75
    receipt=json.loads((tmp_path/'run/receipt.json').read_text())
    assert receipt['status']=='PASS' and receipt['reason']=='fleet_drained'
    assert receipt['fleet_drain']['no_replacements'] and receipt['pids'][0]['drained_boundary']
    assert (tmp_path/'run.tar').exists()


def test_same_node_fleets_need_distinct_closed_evidence():
    def row(label,key=None):
        value=dict(node='timan107',label=label,status='CLOSED',exit_code=0,alive=False,
            child_alive=False,supervisor_alive=False,owned_worker_survivors=[],closure_confirmed=True)
        if key:value.update(kind='workers',fleet_key=key)
        return value
    a,b=row('workers'),row('workers__fleet_more','fleet_more')
    report=dict(all_owned_closed=True,owned_local=[],owned_remote=[
        dict(node='timan107',label='workers'),dict(node='timan107',label=b['label'],kind='workers',fleet_key='fleet_more')],
        cleanup=[a,b])
    assert set(formal_finalize.closed_proof(report))=={'timan107','fleet_more'}
    report['cleanup'].pop()
    with pytest.raises(ValueError,match='closure missing'):formal_finalize.closed_proof(report)


def test_driver_pin_includes_hooks_only_when_enabled(parent_case,tmp_path,monkeypatch):
    import io,tarfile
    from exp.r17_integration.ops.immutable_release import archive_manifest,identity,install
    directory,campaign,_,registry=parent_case
    archive=directory/'runtime.tar'
    names=['exp/r17_integration/ops/formal_driver.py','exp/r17_integration/qualification/driver.py']
    with tarfile.open(archive,'w') as tar:
        for name in names:
            raw=(launch.ROOT/name).read_bytes();info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    campaign['runtime_identity']=identity(archive_manifest(archive))
    # Redirect immutable installs into this CPU fixture; never global releases.
    monkeypatch.setattr(driver_release,'install',lambda a,r,s,i:install(a,tmp_path/'driver_pin',s,i))
    driver_release.prepare(campaign,directory,launch.ROOT)
    selected=campaign['orchestration']['selected']['files']
    assert 'exp/r17_integration/ops/live_fleet_driver.py' in selected
    assert 'exp/r17_integration/ops/live_fleet_worker.py' in selected
    assert sha(directory/'workers.tar')==registry.read()['worker_archive_sha256']


def test_launcher_attach_and_detach_use_production_preparation(parent_case,tmp_path,monkeypatch):
    directory,campaign,deployment,_=parent_case
    transport=CPUTransport()
    monkeypatch.setattr(launch,'remote',transport.remote);monkeypatch.setattr(launch,'transfer',transport.transfer)
    monkeypatch.setattr(launch,'worker_paths',lambda c,n:cpu_paths(tmp_path,c,n))
    (directory/'LIVE_FLEETS.json').unlink()
    spawn=mock.Mock(return_value=dict(node='timan107',label='workers__fleet_extra',kind='workers',
                                     fleet_key='fleet_extra',fleet_id='extra'))
    owner=live.Launcher(launch,campaign,directory,deployment,spawn)
    queued=owner.registry.request('attach',fleet_id='extra',node='timan107',workers=2,execute=True)
    raw=(directory/'campaign.json').read_bytes();payload=(directory/'workers.tar').read_bytes()
    owner.tick();wait_until(lambda:owner.pending[1].done());owner.tick()
    doc=owner.registry.read()
    assert doc['requests'][queued['request_id']]['status']=='ATTACHED'
    assert set(deployment)=={'timan107','fleet_extra'} and spawn.call_count==1
    assert spawn.call_args.args[1]=='workers__fleet_extra'
    assert 'formal_workers' in ' '.join(spawn.call_args.args[2])
    assert doc['fleets']['extra']['worker_manifest_sha256']==sha(directory/'fleet_extra_launch.json')
    assert (directory/'campaign.json').read_bytes()==raw and (directory/'workers.tar').read_bytes()==payload
    detached=owner.registry.request('detach',fleet_id='extra',execute=True)
    owner.tick();doc=owner.registry.read()
    assert doc['requests'][detached['request_id']]['status']=='DRAINING'
    closed=dict(status='CLOSED',exit_code=0,alive=False,child_alive=False,supervisor_alive=False,
                owned_worker_survivors=[])
    assert owner.closed_fleet(spawn.return_value,closed)
    assert owner.registry.read()['fleets']['extra']['state']=='DRAINED'
    owner.freeze();owner.pool.shutdown(wait=True)


def test_finalizer_seals_every_same_node_fleet(tmp_path,monkeypatch):
    # Reuse the full native-protocol synthetic fixture; snapshots and final
    # archive installs are real. No remote command or process control is used.
    from exp.r17_integration.ops.tests.test_formal_finalize import complete
    from exp.r17_integration.ops.sync_back import snapshot_run
    directory,campaign,deployment,report=complete.__wrapped__(tmp_path)
    campaign['live_fleets']=True;write_json(directory/'campaign.json',campaign)
    (directory/'workers.tar').write_bytes(b'synthetic immutable worker fixture')
    deployment['timan107']['workers']=2
    registry=live.Registry(directory);doc=registry.initialize(campaign,deployment)
    evidence=tmp_path/'attached';evidence.mkdir()
    write_json(evidence/'manifest.json',{'run':'attached'})
    write_json(directory/'fleet_extra_launch.json',{'run':'attached'})
    write_json(evidence/'receipt.json',dict(status='PASS',worker_failures=[],pids=[]))
    snapshot=snapshot_run(evidence,directory/'fleet_extra.tar')
    row=dict(node='timan107',label='workers__fleet_extra',kind='workers',fleet_key='fleet_extra',
        fleet_id='extra',status='CLOSED',exit_code=0,alive=False,child_alive=False,
        supervisor_alive=False,owned_worker_survivors=[],closure_confirmed=True)
    report['owned_remote'].append({k:row[k] for k in ('node','label','kind','fleet_key','fleet_id')})
    report['cleanup'].append(row);report['transfers']['fleet_extra']=dict(remote=snapshot,verified={})
    deployment['fleet_extra']=dict(deployment['timan107'],node='timan107',worker_run='/scratch/attached_timan107')
    doc['fleets']['extra']=dict(fleet_id='extra',node='timan107',workers=1,state='CLOSED',initial=False,
        evidence_key='fleet_extra',worker_ids=['attached_timan107_0'],spawn_requested=True,closure=row)
    doc['fleets']['initial_timan107'].update(state='CLOSED',closure=report['cleanup'][0])
    doc['state']='CLOSED';registry.save(doc)
    report['live_fleets']=doc
    before=(directory/'EPISODE_PLAN.json').read_bytes()
    result=formal_finalize.finalize(campaign,directory,deployment,report)
    assert result['status']=='PASS' and set(result['live_fleet_seal']['fleet_ids'])=={'initial_timan107','extra'}
    assert set(result['transfers'])=={'timan107','fleet_extra'}
    assert (directory/'workers/fleet_extra/receipt.json').exists()
    assert (directory/'EPISODE_PLAN.json').read_bytes()==before
    pins=live.resume_pins(directory,result)
    assert str(directory/'LIVE_FLEETS.json') in pins
    del result['transfers']['fleet_extra']
    with pytest.raises(ValueError,match='evidence missing'):live.validate_seal(directory,deployment,result)


def test_local_live_formal_route_keeps_baseline_reservations(parent_case):
    directory,campaign,_,registry=parent_case
    campaign['domain']='libero'
    campaign.update(local_worker_memory_mib=4096,local_worker_vram_mib=1024)
    doc=registry.read();doc['native_pin']=worker.native_pin(campaign)
    child=live.child_campaign(campaign,doc,dict(fleet_id='localextra',node='local',workers=2),directory)
    proof=local_workers.authorize(child,today='2026-10-11')
    assert proof['live_fleet']['worker_nodes']=={'local':2}
    routed=local_workers.route(dict(child,worker_node='local',node_workers=2),child)
    assert routed['driver_host']=='127.0.0.1' and routed['policy_endpoint']==campaign['endpoint']
    assert routed['worker_memory_mib']==4096 and routed['worker_vram_mib']==1024
    assert routed['worker_server_vram_headroom_mib']==16384


def test_missing_supervisor_protocol_refuses_before_install(tmp_path):
    path=tmp_path/'old.tar'
    build_client_bundle(launch.ROOT,['src/openpi/conductor/task.py'],path)
    with pytest.raises(ValueError,match='protocol missing'):
        worker.check_payload_protocol(path)


def test_attached_supervisor_refuses_changed_config_before_admission(tmp_path,monkeypatch):
    from exp.r17_integration.ops import worker_gpu
    directory,campaign,_,registry=parent(tmp_path)
    row=dict(fleet_id='extra',node='timan107',workers=2)
    child=live.child_campaign(campaign,registry.read(),row,directory)
    child.update(worker_run=str(tmp_path/'unstarted'),worker_node='timan107',node_workers=2)
    file=tmp_path/'child.json';write_json(file,child)
    monkeypatch.setattr(sys,'argv',['formal_workers','--campaign',str(file)])
    monkeypatch.setenv('R17_LIVE_FLEET_CONFIG_SHA256','wrong')
    monkeypatch.setattr(worker_gpu,'acquire',lambda *a:pytest.fail('GPU admission must not happen'))
    with pytest.raises(ValueError,match='config SHA changed'):formal_workers.main()
    assert not (tmp_path/'unstarted').exists()


def test_periodic_sync_lock_blocks_closed_storage_cleanup(tmp_path,monkeypatch):
    import fcntl
    from exp.r17_integration.ops import worker_storage
    from exp.r17_integration.ops.tests.test_worker_storage import closed
    root,run,archive,directory,value,proof=closed.__wrapped__(tmp_path,monkeypatch)
    (run/'.incremental_sync.lock').touch()
    with (run/'.incremental_sync.lock').open('rb') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        with pytest.raises(ValueError,match='incremental sync still busy'):
            worker_storage.remove_closed_run(proof)
        assert run.exists() and archive.exists()
    result=worker_storage.remove_closed_run(proof)
    assert result['status']=='REMOVED' and (directory/'timan107.tar').exists()


def test_drain_marker_serializes_against_spawn(tmp_path):
    config=dict(live_fleets=True,worker_run=str(tmp_path/'run'),run_name='fleet',worker_node='timan107')
    started=threading.Event();finished=threading.Event()
    def drain():
        started.set();worker.write_drain(config);finished.set()
    with worker.lifecycle(config):
        thread=threading.Thread(target=drain);thread.start();assert started.wait(2)
        assert not finished.wait(.05)
        assert not worker.draining(config)
    thread.join(3);assert finished.is_set() and worker.draining(config)


def test_drain_before_start_needs_no_new_resource_admission(tmp_path,monkeypatch):
    from exp.r17_integration.ops import worker_gpu
    config=dict(run_name='drained',worker_node='timan107',worker_run=str(tmp_path/'run'),domain='metaworld',
        node_workers=1,seconds=10,endpoint='fake:1',driver_host='fake',driver_port=2,
        worker_egl_device=0,worker_max_episodes=4,worker_benchmark='/sealed',live_fleets=True)
    file=tmp_path/'campaign.json';write_json(file,config);worker.write_drain(config)
    monkeypatch.setattr(sys,'argv',['formal_workers','--campaign',str(file)])
    monkeypatch.setattr(worker_gpu,'acquire',lambda *a:pytest.fail('drained empty fleet must not reserve resources'))
    monkeypatch.setattr(formal_workers,'stage_snapshot',lambda *a,**kw:{})
    with pytest.raises(SystemExit) as stopped:formal_workers.main()
    assert stopped.value.code==0
    receipt=json.loads((tmp_path/'run/receipt.json').read_text())
    assert receipt['pids']==[] and receipt['status']=='PASS' and receipt['reason']=='fleet_drained'


def test_large_resume_attachment_keeps_configuration_out_of_argv(parent_case,tmp_path,monkeypatch):
    directory,campaign,_,registry=parent_case
    campaign['cache_resume_selection']={'missing_task_uids':['group:eval:0:'+str(i) for i in range(20000)]}
    write_json(directory/'campaign.json',campaign)
    doc=registry.read();doc['campaign_sha256']=sha(directory/'campaign.json');registry.save(doc)
    transport=CPUTransport()
    monkeypatch.setattr(launch,'remote',transport.remote);monkeypatch.setattr(launch,'transfer',transport.transfer)
    monkeypatch.setattr(launch,'worker_paths',lambda c,n:cpu_paths(tmp_path,c,n))
    row=dict(fleet_id='large',node='timan107',workers=2)
    child,item,key,manifest_sha=live.prepare_attachment(launch,campaign,directory,doc,row)
    assert item['campaign_transfer']['bytes']>128*1024
    for call in transport.calls:
        argv=call[1] if isinstance(call,tuple) else call
        assert len('\0'.join(argv).encode())<32*1024
    assert json.loads(Path(item['campaign_path']).read_text())['cache_resume_selection']==campaign['cache_resume_selection']
