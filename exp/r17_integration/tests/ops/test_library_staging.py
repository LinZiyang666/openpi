"""CPU-only real shards/snapshots; no tether, resident or GPU is started."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest

from exp.r17_integration.ops import library_staging as staging
from exp.r17_integration.ops.library_campaign import finish
from exp.r17_integration.ops.sync_back import install_snapshot, snapshot_run
from exp.r17_integration.debug.validate import digest, episode_identity
from exp.r17_integration.tests.search.library_fixture import synthetic_capture


@pytest.fixture
def campaign(tmp_path,monkeypatch):
    from exp.r17_integration.qualification import archive_stage
    monkeypatch.setattr(staging,'TRANSFER_ROOT',tmp_path/'tmp/r17_formal')
    monkeypatch.setattr(archive_stage,'STAGE_ROOT',tmp_path/'tmp/r17_Q_qualification')
    monkeypatch.setattr(staging.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*staging.GIB))
    return dict(library=True,model='fastwam',server_node='h100',run_name='CPU',suite='libero_spatial',
        h100_staging=dict(staging.DEFAULT_POLICY),server_run=str(tmp_path/'remote/servers'),smoke=True,synthetic=True)


def closed_shard(campaign,index=0,replica=0):
    budget = staging.ShardBudget(campaign)
    shard_id = digest(('CPU',index))
    leaf = budget.root/('replica_'+str(replica))/'capture/shards'/shard_id
    budget.reserve(shard_id,leaf)
    binding,episodes,audit,names = synthetic_capture(leaf/'binding_cpu',episodes=1,episode_offset=index)
    (leaf/'manifest.json').write_text(json.dumps(dict(shard_id=shard_id,campaign=campaign['run_name'])))
    size = sum(p.stat().st_size for p in leaf.rglob('*') if p.is_file())
    budget.charge(shard_id,2*size)
    budget.closed(shard_id)
    return shard_id,leaf,binding,episodes,audit,names


@pytest.mark.parametrize('filesystem',('data','tmp'))
def test_free_space_admission_rejects_before_any_capture_or_gpu(campaign,filesystem,tmp_path):
    def usage(path):
        low = str(path)==('/data' if filesystem=='data' else '/tmp')
        return SimpleNamespace(free=(28 if filesystem=='data' else 2)*staging.GIB if low else 100*staging.GIB)
    with pytest.raises(RuntimeError,match='admission'):
        staging.admit(campaign,usage=usage)
    assert not Path(campaign['server_run']).exists()
    assert staging.admit(campaign)['status']=='PASS'


@pytest.mark.parametrize('filesystem',('data','tmp'))
def test_during_run_free_space_guard_latches_failure(campaign,filesystem):
    free = dict(data=100*staging.GIB,tmp=100*staging.GIB)
    budget = staging.ShardBudget(campaign,usage=lambda p:SimpleNamespace(free=free['tmp' if str(p)=='/tmp' else 'data']))
    free[filesystem] = staging.GIB
    with pytest.raises(RuntimeError,match='free space'):
        budget.check()
    free[filesystem] = 100*staging.GIB
    with pytest.raises(RuntimeError,match='stopped'):
        budget.check()
    assert json.loads(budget.state_path.read_text())['failure']


def test_campaign_quota_is_shared_across_replicas_and_inflight_buffers(campaign):
    limits = staging.policy(campaign)
    capacity = (limits['limit_bytes']-limits['transfer_reserve_bytes']-limits['metadata_reserve_bytes'])//limits['shard_bytes']
    def reserve(index):
        budget = staging.ShardBudget(campaign)
        leaf = budget.root/('replica_'+str(index%3))/'capture/shards'/digest(index)
        budget.reserve(leaf.name,leaf,wait_seconds=0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(reserve,range(capacity)))
    budget = staging.ShardBudget(campaign)
    assert budget.check()['high_water_reserved_bytes']==capacity*limits['shard_bytes']+limits['transfer_reserve_bytes']+limits['metadata_reserve_bytes']
    with pytest.raises(RuntimeError,match='bound reached'):
        reserve(capacity)
    state = json.loads(budget.state_path.read_text())
    assert len(state['shards'])==capacity and state['high_water_reserved_bytes']<=20_000_000_000


def test_prewrite_charge_and_open_shard_transfer_fail_closed(campaign):
    budget = staging.ShardBudget(campaign)
    leaf = budget.root/'replica_0/capture/shards'/digest('open')
    budget.reserve(leaf.name,leaf)
    assert staging.pending(campaign)['shards']=={}
    assert staging.pending(campaign)['active']==1
    with pytest.raises(ValueError,match='open teacher shard'):
        staging.stage_shard(campaign,leaf.name)
    # No open capture was copied, and no quota credit was released.
    assert not staging.TRANSFER_ROOT.exists()
    assert len(json.loads(budget.state_path.read_text())['shards'])==1
    other = dict(campaign,run_name='oversize',server_run=str(budget.root.parent/'oversize'))
    quota = staging.ShardBudget(other)
    target = quota.root/'replica_0/capture/shards'/digest('oversize')
    quota.reserve(target.name,target)
    with pytest.raises(RuntimeError,match='exceeds bounded'):
        quota.charge(target.name,quota.limits['shard_bytes'])
    assert not target.exists()


@pytest.mark.parametrize('mutation',('archive','receipt','remote_changed'))
def test_corruption_never_deletes_h100_capture(campaign,tmp_path,mutation):
    shard_id,leaf,*_ = closed_shard(campaign)
    transfer = staging.stage_shard(campaign,shard_id)
    local = tmp_path/'local'; local.mkdir()
    archive = local/'pulled.tar'; shutil.copyfile(transfer['archive'],archive)
    if mutation=='archive':
        with archive.open('r+b') as stream: stream.write(b'bad')
        with pytest.raises(ValueError,match='SHA mismatch'):
            install_snapshot(archive,local/'installed',expected_sha256=transfer['archive_sha256'])
    else:
        verified = install_snapshot(archive,local/'installed',expected_sha256=transfer['archive_sha256'])
        receipt = dict(schema='r17.h100_shard_receipt.v1',status='verified',campaign=campaign['run_name'],
            shard_id=shard_id,archive_sha256=transfer['archive_sha256'],snapshot_digest=verified['digest'],
            partials=[],build_host='weilandserver')
        if mutation=='receipt': receipt['archive_sha256']='0'*64
        else: (leaf/'manifest.json').write_text('changed')
        with pytest.raises(ValueError,match='receipt|changed'):
            staging.acknowledge_shard(campaign,shard_id,receipt)
    assert leaf.exists() and Path(transfer['archive']).exists() and Path(transfer['source_archive']).exists()
    assert shard_id in json.loads((Path(campaign['server_run'])/'SHARD_BUDGET.json').read_text())['shards']


def test_midcampaign_verified_shards_sync_then_local_parent_build(campaign,tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_driver
    closed = [closed_shard(campaign,i,i) for i in (0,1)]
    local = tmp_path/'weilandserver'; local.mkdir()
    # Active residency marker remains throughout the transfer; no process calls.
    server_root = Path(campaign['server_run'])
    (server_root/'manifest.json').write_text('{"resident":"synthetic-running"}')
    received = threading.Event()
    def remote(action,*args):
        if action=='acknowledge_shard':
            shard_id,receipt = args
            assert (local/'shard_transfers'/(shard_id+'.receipt.json')).is_file()
            assert Path(receipt['destination']).is_dir()
            assert closed[int(shard_id==closed[1][0])][1].exists()
        result = getattr(staging,action)(campaign,*args)
        if action=='acknowledge_shard' and not staging.pending(campaign)['shards']: received.set()
        return result
    pump = staging.ShardPump(campaign,local,remote,lambda source,target:shutil.copyfile(source,target))
    pump.start()
    try:
        assert received.wait(5),repr(pump.error)
        summary = pump.finish()
    finally:
        pump.abort()
    assert len(summary['shards'])==2 and summary['high_water_reserved_bytes']<=20_000_000_000
    assert all(not row[1].exists() for row in closed)
    assert not list(staging.TRANSFER_ROOT.rglob('*.tar'))
    assert (server_root/'manifest.json').is_file()
    assert not (server_root/'bpool').exists()
    # Remaining closed server metadata uses the original SHA snapshot/install.
    snap = snapshot_run(server_root,tmp_path/'metadata.tar')
    verified = install_snapshot(tmp_path/'metadata.tar',local/'servers',expected_sha256=snap['archive_sha256'])
    binding,_,audit,names = closed[0][2:]
    episodes = [ep for row in closed for ep in row[3]]
    alignment = local/'ALIGNMENT.json'; alignment.write_text(json.dumps(audit))
    campaign.update(alignment=str(alignment),library_parent=str(local/'bpool'))
    (local/'RECEIPT.json').write_text(json.dumps(dict(status='PASS',all_owned_closed=True,
        transfers=dict(h100_shards=summary,h100_servers=dict(verified=verified)))))
    driver = local/'driver'; driver.mkdir()
    (driver/'journal.jsonl').write_text(''.join(json.dumps(dict(task_uid=ep['task_uid'],accepted=True,
        status='done',attempt=1,success=ep['success'],error=None))+'\n' for ep in episodes))
    (driver/'per_step.jsonl').write_text(''.join(json.dumps(dict(task_uid=ep['task_uid'],attempt=1,
        accepted=True,decision_seq=i))+'\n' for ep in episodes for i in (0,1)))
    tasks = [SimpleNamespace(extra=ep,task_uid=ep['task_uid'],orig_init_state_idx=ep['orig_init_idx']) for ep in episodes]
    monkeypatch.setattr(formal_driver,'plan',lambda c:(SimpleNamespace(identity=binding['benchmark_protocol'],ordered_tasks=names),binding['model'],tasks,[]))
    monkeypatch.setattr(staging.socket,'gethostname',lambda:'weilandserver')
    result = finish(campaign,local)
    assert result['status']=='PASS' and result['rows']==4 and result['failed_episodes']==1
    assert Path(result['parent']).is_relative_to(local)
    monkeypatch.setattr(staging.socket,'gethostname',lambda:'h100')
    with pytest.raises(ValueError,match='only on weilandserver'):
        finish(campaign,local)


def test_pump_failure_propagates_and_incomplete_episode_cannot_finish(campaign,tmp_path):
    def failed(action,*args): raise OSError('synthetic transfer failure')
    pump = staging.ShardPump(campaign,tmp_path/'local',failed,None)
    pump.start(); pump.thread.join(5)
    with pytest.raises(RuntimeError,match='transfer failed'): pump.check()
    budget = staging.ShardBudget(campaign)
    leaf = budget.root/'replica_0/capture/shards'/digest('incomplete')
    budget.reserve(leaf.name,leaf)
    other = staging.ShardPump(campaign,tmp_path/'other',lambda action,*args:getattr(staging,action)(campaign,*args),None)
    other.start()
    with pytest.raises(ValueError,match='unacknowledged'):
        other.finish()


def test_formal_h100_admission_precedes_model_load(campaign,tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_server
    from exp.r17_integration.qualification import model_runtime
    calls = []
    monkeypatch.setattr(model_runtime,'load',lambda *a:calls.append(a))
    monkeypatch.setattr(staging.shutil,'disk_usage',lambda p:SimpleNamespace(free=staging.GIB))
    with pytest.raises(RuntimeError,match='admission'):
        formal_server.serve(campaign,23341,tmp_path/'unused')
    assert calls==[] and not Path(campaign['server_run']).exists()


def test_deploy_admission_fails_before_any_runtime_or_transfer(campaign,tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_eval
    calls = []
    def remote(node,argv,*args,**kwargs):
        calls.append(argv)
        assert argv[-2]=='-c' and 'disk_usage' in argv[-1]
        raise RuntimeError('H100 free-space admission failed')
    monkeypatch.setattr(formal_eval,'remote',remote)
    monkeypatch.setattr(formal_eval,'transfer',lambda *a:pytest.fail('transfer before admission'))
    monkeypatch.setattr(staging.socket,'gethostname',lambda:'weilandserver')
    dest = tmp_path/'prepare'; dest.mkdir()
    with pytest.raises(RuntimeError,match='admission'):
        formal_eval.prepare(campaign,dest)
    assert len(calls)==1 and not (dest/'runtime.tar').exists()


def test_campaign_start_space_failure_never_spawns_owned_process(campaign,tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_eval
    campaign.update(runtime=str(tmp_path/'runtime'),seconds=60,vendor='unused',server_campaign=str(tmp_path/'campaign.json'))
    def remote(*args,**kwargs): raise RuntimeError('H100 free-space admission failed')
    monkeypatch.setattr(formal_eval,'remote',remote)
    monkeypatch.setattr(formal_eval.subprocess,'Popen',lambda *a,**k:pytest.fail('resident/driver started'))
    monkeypatch.setattr(staging.socket,'gethostname',lambda:'weilandserver')
    dest = tmp_path/'launch'; dest.mkdir()
    assert formal_eval.run_campaign(campaign,dest,{})==2
    receipt = json.loads((dest/'RECEIPT.json').read_text())
    assert receipt['status']=='FAIL' and receipt['owned_local']==receipt['owned_remote']==[]
    assert receipt['all_owned_closed'] is True and 'admission' in receipt['error']


def test_fastwam_l10_max_geometry_fits_one_episode_reservation(campaign):
    """Native-sized numeric fixture:104 decisions, fused49152 keys,2x256 RGB.

    No model/native sampler runs; this qualifies transport sizing/closure only.
    """
    import numpy as np
    from exp.r17_integration.tests.debug.fixtures import spec,record
    from exp.r17_integration.common.contracts import DecisionCapture,RequestContext
    from exp.r17_integration.debug.library_shards import ShardedLibraryCaptureFactory
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    model = spec('fastwam','libero',key_width=49152)
    ep = dict(record(0,task='libero_10/synthetic',n_decisions=104),
        benchmark_protocol=model.allowed_benchmark_protocols[0],task_id=30,episode_id=1500,
        init_pool='B',split='library',library_episode_id=0,suite_task_id=0,init_state_sha256='synthetic')
    config = SimpleNamespace(enabled=False,checkpoints={},recipe=SimpleNamespace(enabled=False),
        adapter_binding={'benchmark_protocol':ep['benchmark_protocol']})
    core = SimpleNamespace(spec=model,sampler=model.default_sampler,config=config,_episode=ep)
    budget = staging.ShardBudget(campaign)
    factory = ShardedLibraryCaptureFactory(budget.root/'replica_0/capture',budget)
    sink = factory(core)
    random = np.random.default_rng(17)
    images = {name:random.integers(0,256,(256,256,3),dtype=np.uint8)
        for name in ('observation/image','observation/wrist_image')}
    fields = {'vision_0':random.normal(size=49152).astype(np.float32),'robot_state':np.zeros(8,np.float32)}
    actions = np.zeros((32,7),np.float32)
    for seq in range(104):
        request = RequestContext(ep['campaign'],ep['benchmark_protocol'],ep['task_name'],0,seq,
            arm=ep['arm'],bundle_generation=ep['bundle_generation'],attempt=1,task_id=30)
        sink.bind_observation(images)
        sink(DecisionCapture(request,'policy',0,0,(1,1,10),10,actions,actions,fields))
        if seq==103:
            assert staging.pending(campaign)['shards']=={} and staging.pending(campaign)['active']==1
        sink.on_execution(dict(request=asdict(request),decision_seq=seq,executed_count=5,cursor=5,terminal=seq==103))
    ready = staging.pending(campaign)
    assert len(ready['shards'])==1 and ready['active']==0
    entry = next(iter(ready['shards'].values()))
    assert 40*1024**2<entry['actual_bytes']<entry['charged_bytes']<256*1024**2
    source = next((budget.root/entry['relative']).glob('binding_*'))
    rows = list(CaptureReader(source,max_cached_blocks=2).decisions())
    assert len(rows)==104 and rows[-1][0]['terminal'] is True
    assert rows[-1][0]['episode_id']==rows[-1][0]['library_episode_id']==0
    assert rows[-1][1]['image_0'].tobytes()==images['observation/image'].tobytes()
    assert rows[-1][2]['vision_0'].shape==(49152,)
    assert all(row['drained'] for row in factory.close())
