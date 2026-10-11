"""Frozen size-queue leases and admission for the requested concurrent split."""
import pytest

from exp.r17_integration.ops import local_workers,worker_gpu,worker_memory,cache_split_lanes
from exp.r17_integration.ops.immutable_release import sha
from exp.r17_integration.ops.formal_process import write_json


def frozen(tmp_path):
    study=tmp_path/'STUDY.json'
    write_json(study,dict(library_sizes=[50,100,200,300,400],artifacts_pending=[],digest='frozen'))
    assets=tmp_path/'assets';assets.mkdir()
    write_json(assets/'CACHE_STUDY.json',dict(source_study_digest='frozen'))
    campaign=dict(domain='libero',cache_study=str(study),local_assets=str(assets),model='dp',workers=32,
        worker_nodes={'timan107':32},execution_worker_nodes={'timan107':16,'local':16},cache_control='control',
        local_worker_server_vram_headroom_mib=8192,server_node='local',public_port=23400,driver_port=23419,
        endpoint='canonical:23400')
    campaign['cache_size_lane_authorization']=dict(schema='r17.cache_size_lane_authorization.v1',
        owner_authorized=True,study_date='2026-10-10',study=str(study),study_sha256=sha(study),study_digest='frozen',
        model='dp',workers=32,worker_nodes=campaign['execution_worker_nodes'],control='control',local_server_growth_mib=8192)
    return campaign


def test_frozen_size_lane_remains_authorized_after_midnight_and_in_closed_continuations(tmp_path):
    campaign=frozen(tmp_path)
    before=local_workers.authorize(campaign,today='2026-10-10')
    after=local_workers.authorize(dict(campaign,run_name='closed_continuation'),today='2026-10-13')
    assert before==after
    worker=local_workers.route(dict(campaign,worker_node='local'),campaign)
    assert worker['worker_memory_mib']==3584 and worker['worker_vram_mib']==717
    assert worker['worker_server_vram_headroom_mib']==8192
    assert worker['worker_memory_headroom_mib']==8192 and worker['policy_endpoint']=='127.0.0.1:23400'


@pytest.mark.parametrize('field,value',[('execution_worker_nodes',{'local':32}),('workers',31),
    ('model','fastwam'),('cache_control','different'),('local_worker_server_vram_headroom_mib',4096)])
def test_size_authorization_rejects_changed_fleet_or_resource_binding(tmp_path,field,value):
    campaign=frozen(tmp_path);campaign[field]=value
    with pytest.raises(ValueError,match='authorization changed'):local_workers.authorize(campaign,today='2026-10-11')


def test_size_authorization_rejects_changed_or_non_size_study(tmp_path):
    campaign=frozen(tmp_path)
    study=campaign['cache_study']
    write_json(study,dict(library_sizes=[500],artifacts_pending=[],digest='other'))
    with pytest.raises(ValueError):local_workers.authorize(campaign,today='2026-10-11')


def test_lower_server_reserve_is_scoped_to_frozen_size_lanes(tmp_path):
    campaign=frozen(tmp_path);campaign.pop('cache_size_lane_authorization')
    with pytest.raises(ValueError,match='reserve override'):local_workers.route(dict(campaign,worker_node='local'),campaign)
    with pytest.raises(ValueError):local_workers.authorize(campaign,today='2026-10-11')


def test_joint_local_fleets_fit_with_measured_reserve_and_retain_foreign_use_guard(tmp_path,monkeypatch):
    # One already admitted 16-worker fleet: 540 MiB observed each, its full
    # 1024 MiB reservations retained during recycle. Both DP replicas use2202.
    card=dict(device=0,uuid='local',total_mib=49140,used_mib=2202+16*540,usable=True)
    monkeypatch.setattr(worker_gpu,'reservations',lambda *a,**k:({0:16*(1024-540)},[]))
    campaign=dict(domain='libero',worker_node='local',worker_egl_device=0,worker_egl_devices=[0],
        node_workers=16,worker_vram_mib=1024,worker_server_vram_headroom_mib=8192)
    assert worker_gpu.admission(campaign,tmp_path/'new',observed=[card])['admitted']
    assert not worker_gpu.admission(dict(campaign,worker_server_vram_headroom_mib=16384),tmp_path/'new',observed=[card])['admitted']
    assert not worker_gpu.admission(campaign,tmp_path/'new',observed=[dict(card,used_mib=card['used_mib']+4096)])['admitted']
    mib=1024**2
    result=worker_memory.admission(dict(domain='libero',node_workers=32,worker_memory_mib=4096,
        worker_memory_headroom_mib=32768),tmp_path/'next',memory=(257792*mib,243155*mib))
    assert result['admitted'] and result['requested_bytes']==128*1024*mib


def test_fleet_totals_and_t1_concurrency_are_fixed():
    fleets=cache_split_lanes.FLEETS
    assert fleets['dp_libero']=={'timan107':16,'local':16}
    assert fleets['fastwam_libero']=={'timan107':16,'local':16,'timan1':16}
    assert fleets['cosmos_libero']=={'timan1':8}
    assert [sum(fleets[m].values()) for m in fleets]==[32,48,8]


def test_recovered_h100_preparation_can_clone_without_prior_server_campaign(tmp_path,monkeypatch):
    from exp.r17_integration.ops.tests.test_mps_queue_retry import prepared
    from exp.r17_integration.ops import queue_retry,formal_eval,formal_runtime,formal_prestage
    source,campaign=prepared(tmp_path)
    campaign['server_node']='h100'
    write_json(source/'campaign.json',campaign)
    formal_prestage.seal_prepared(campaign,source,{'timan107':{}})
    snapshot=queue_retry.capture(source,tmp_path/'snapshot')
    monkeypatch.setattr(formal_eval,'FORMAL_ROOT',tmp_path)
    monkeypatch.setattr(formal_runtime,'check_pins',lambda c:None)
    calls=[]
    monkeypatch.setattr(formal_eval,'remote',lambda node,argv,**k:calls.append(argv) or '{}')
    monkeypatch.setattr(formal_eval,'transfer',lambda argv:calls.append(argv))
    def workers(c,d):
        write_json(d/'WORKER_DEPLOYMENT.json',{'timan107':{}})
        write_json(d/'timan107.json',c)
        return {'timan107':{}}
    monkeypatch.setattr(formal_eval,'prepare_workers',workers)
    destination=queue_retry.clone(snapshot,'fixed',dict(cpu_only=True))
    candidate,_=formal_prestage.validate_prepared(destination)
    assert candidate['server_campaign']=='/tmp/r17_formal/fixed/server/campaign.json'
    assert any(candidate['server_campaign'] in str(call) for call in calls)
    assert all('spawn' not in call for call in calls)


def test_split_clone_updates_preflight_worker_summary(tmp_path,monkeypatch):
    from exp.r17_integration.ops.tests.test_mps_queue_retry import prepared
    from exp.r17_integration.ops import queue_retry,formal_eval,formal_runtime,formal_prestage,worker_topology
    source,_=prepared(tmp_path)
    snapshot=queue_retry.capture(source,tmp_path/'snapshot')
    monkeypatch.setattr(formal_eval,'FORMAL_ROOT',tmp_path)
    monkeypatch.setattr(formal_runtime,'check_pins',lambda c:None)
    counts={'timan107':12,'local':12}
    monkeypatch.setattr(worker_topology,'redistribute',lambda c,n,d,**kw:c.update(execution_worker_nodes=n))
    def workers(c,d):
        for node in counts:
            write_json(d/(node+'.json'),c)
            write_json(d/(node+'_MEMORY_ADMISSION.json'),dict(admitted=True))
        deployment={n:{} for n in counts};write_json(d/'WORKER_DEPLOYMENT.json',deployment)
        return deployment
    monkeypatch.setattr(formal_eval,'prepare_workers',workers)
    destination=queue_retry.clone(snapshot,'split',dict(cpu_only=True),worker_counts=counts)
    import json
    preflight=json.loads((destination/'PREFLIGHT.json').read_text())
    assert set(preflight['worker_nodes'])==set(counts) and preflight['execution_worker_nodes']==counts
    formal_prestage.validate_prepared(destination)


def test_armed_lane_keeps_waiting_for_repaired_predecessor_without_holding_its_lock(tmp_path,monkeypatch):
    import fcntl
    from exp.r17_integration.ops import cache_lane as lane
    directory=tmp_path/'old';directory.mkdir();(directory/'.lane.lock').touch()
    write_json(directory/'LANE_PLAN.json',dict(lane='4090'))
    write_json(directory/'LANE_STATE.json',dict(status='FAIL',results=[]))
    write_json(directory/'LANE_RECEIPT.json',dict(status='FAIL',all_owned_closed=True))
    monkeypatch.setattr(lane,'lane_processes',lambda d:[])
    waits=[]
    def repair(seconds):
        with (directory/'.lane.lock').open() as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            write_json(directory/'LANE_STATE.json',dict(status='PASS',results=[]))
            write_json(directory/'LANE_RECEIPT.json',dict(status='PASS',all_owned_closed=True))
        waits.append(seconds)
    monkeypatch.setattr(lane.time,'sleep',repair)
    record=lane.predecessor(directory,'4090',tmp_path/'new')
    gate=lane.wait_predecessor(record,interval=1,wait_for_pass=True);gate.close()
    assert waits==[1]
