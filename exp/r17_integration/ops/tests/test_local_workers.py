"""CPU checks for the explicit, today-only local LIBERO worker exception."""
import os
from pathlib import Path
from unittest import mock

import pytest

from exp.r17_integration.ops import formal_eval as launch, local_workers as local
from exp.r17_integration.ops import worker_memory, worker_gpu, cache_lane


def campaign(**extra):
    return dict(domain='libero',smoke=True,worker_nodes={'timan107':8,'local':16},
                server_node='local',public_port=23400,driver_port=23419,
                endpoint='ziyanglin.com:23400',worker_egl_device=7,**extra)


def test_explicit_counts_and_creation_scope():
    assert local.counts('timan107=8,local=16')=={'timan107':8,'local':16}
    # Larger explicit-capacity studies may parse up to40; T1's ordinary lease
    # still rejects more than16 in timan1_workers.authorize (tested separately).
    for value in ('timan1=41','local=0','local=2,local=3','local=x'):
        with pytest.raises(ValueError):local.counts(value)
    c=campaign()
    assert local.authorize(c,today='2026-10-10')['workers']==c['worker_nodes']
    for bad in (dict(c,domain='metaworld'),dict(c,smoke=False)):
        with pytest.raises(ValueError):local.authorize(bad,today='2026-10-10')
    with pytest.raises(ValueError):local.authorize(c,today='2026-10-11')


def test_local_routes_keep_canonical_task_key_and_reserve_server_headroom():
    c=campaign();w=dict(c,worker_node='local');local.route(w,c)
    assert w['endpoint']==c['endpoint'] and w['policy_endpoint']=='127.0.0.1:23400'
    assert w['driver_host']=='127.0.0.1' and w['worker_egl_devices']==[0]
    assert w['worker_memory_headroom_mib']==8192 and w['worker_server_vram_headroom_mib']==16384
    h=dict(c,server_node='h100',endpoint='broker.example:8000')
    assert local.route(dict(h,worker_node='local'),h)['policy_endpoint']==h['endpoint']
    remote=dict(c,worker_node='timan107');assert local.route(remote,c)==remote


def test_native_client_route_is_process_local_and_endpoint_fenced():
    from exp.r17_integration.ops.local_worker_entry import loopback
    from openpi_client import websocket_client_policy as module
    seen=[]
    class Client:
        def __init__(self,**kwargs):seen.append(kwargs)
    with mock.patch.object(module,'WebsocketClientPolicy',Client):
        with loopback('ziyanglin.com:23400','127.0.0.1:23400'):
            module.WebsocketClientPolicy(host='ziyanglin.com',port=23400)
            with pytest.raises(ValueError):module.WebsocketClientPolicy(host='other',port=23400)
        assert module.WebsocketClientPolicy is Client
    assert seen==[dict(host='127.0.0.1',port=23400)]


def test_local_transport_never_calls_tether(tmp_path,monkeypatch):
    monkeypatch.setattr(local,'ROOT',tmp_path)
    src=tmp_path/'source';src.write_bytes(b'preserved payload')
    dst=tmp_path/'release/data'
    with mock.patch.object(launch,'execute',return_value='direct') as execute:
        assert launch.remote('local',['python','-c','pass'])=='direct'
        assert execute.call_args.args[0][0]=='python'
        launch.transfer(['tether','push',str(src),'local:'+str(dst)])
        assert execute.call_count==1
    assert dst.read_bytes()==src.read_bytes()
    with pytest.raises(ValueError):local.copy(src,tmp_path.parent/'forbidden')


def test_local_memory_and_renderer_reservations_refuse_before_spawn(tmp_path):
    c=campaign();w=local.route(dict(c,worker_node='local',node_workers=4),c)
    mib=1024**2
    result=worker_memory.admission(w,tmp_path/'new',memory=(100*1024*mib,20*1024*mib))
    assert not result['admitted'] and result['headroom_bytes']==8192*mib
    card=dict(device=0,uuid='GPU-local',total_mib=49140,used_mib=3000,usable=True)
    result=worker_gpu.admission(dict(w,node_workers=36),tmp_path/'new',observed=[card])
    assert not result['admitted'] and result['cards'][0]['server_headroom_mib']==16384
    assert worker_gpu.admission(w,tmp_path/'new',observed=[card])['admitted']


def test_mixed_cache_lane_flags_leave_model_and_recycling_settings_unchanged():
    old=cache_lane.launch_args('dp_libero','/study.json','next')
    new=cache_lane.launch_args('dp_libero','/study.json','next',worker_counts='timan107=8,local=16')
    assert new[:-2]==old and new[-2:]==['--worker-counts','timan107=8,local=16']
    assert new[new.index('--worker-max-episodes')+1]=='4'
    jobs=cache_lane.jobs('h100','/study.json','next',worker_counts='timan107=4,local=8')
    assert all(j['argv'][-2:]==['--worker-counts','timan107=4,local=8'] for j in jobs)


def test_worker_wrapper_and_physical_device_receipt_configuration():
    from exp.r17_integration.ops.formal_workers import worker_command
    c=campaign();w=local.route(dict(c,worker_node='local',run_name='q',worker_benchmark='/sealed',worker_max_episodes=4),c)
    _,argv=worker_command(w,0)
    assert argv[argv.index('-m')+1]=='exp.r17_integration.ops.local_worker_entry'
    assert argv[argv.index('--server-key')+1]=='ziyanglin.com:23400'
    assert argv[argv.index('--egl-device')+1]=='0' and '--max-episodes' in argv
    assert launch.worker_paths(c|{'run_name':'q'},'local')['run'].startswith(str(local.ROOT))


def test_local_ops_overlay_preserves_every_native_client_and_simulator_byte(tmp_path):
    from exp.r17_integration.ops.build_client_bundle import build_client_bundle,verified_members
    root=tmp_path/'source';root.mkdir();directory=tmp_path/'run';directory.mkdir()
    for name in ('formal_workers','worker_gpu','worker_memory','worker_disk','local_worker_entry'):
        p=root/'exp/r17_integration/ops'/f'{name}.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('new ops')
    native='examples/libero/episode_runner.py';p=root/native;p.parent.mkdir(parents=True);p.write_text('frozen native')
    old=build_client_bundle(root,[native],directory/'workers.tar')
    c=campaign();c['local_worker_exception']=local.authorize(c,today='2026-10-10')
    local.payload(c,directory,root,old)
    manifest,_=verified_members(directory/'workers.tar','PAYLOAD.json')
    assert manifest['files'][native]==old['manifest']['files'][native]


def test_cache_exception_waives_only_fleet_topology_not_model_or_server_treatment():
    import copy,json
    from exp.r17_integration.ops import cache_control,cache_study
    source=cache_lane.RUN_ROOT/cache_study.CONTROLS['dp_libero']
    runtime=cache_lane.RUN_ROOT/cache_lane.RUNTIME_SOURCES['dp_libero']
    old=json.loads((source/'campaign.json').read_text())
    new=dict(old,workers=24,worker_nodes={'timan107':8,'local':16},cache_study='/sealed/cache/STUDY.json')
    new['local_worker_exception']=local.authorize(new,today='2026-10-10')
    proof=cache_control.control_proof(source,runtime,new)
    assert set(proof['worker_topology_changes'])=={'workers','worker_nodes'}
    for field in ('model','checkpoint','replicas','worker_memory_mib','worker_vram_mib','mps','dp_batching'):
        changed=copy.deepcopy(new);changed[field]='changed'
        with pytest.raises(ValueError,match=field):cache_control.control_proof(source,runtime,changed)


def test_direct_supervisor_is_receipt_owned_without_tether(tmp_path):
    import json,sys,time
    from exp.r17_integration.ops.formal_process import alive
    (tmp_path/'lease').touch()
    request=dict(argv=[sys.executable,'-c','pass'],env={'PYTHONPATH':str(launch.ROOT)},cwd=str(tmp_path),
        receipt=str(tmp_path/'process.json'),log=str(tmp_path/'process.log'),lease=str(tmp_path/'lease'),seconds=10)
    prefix=[sys.executable,'-m','exp.r17_integration.ops.formal_process']
    result=json.loads(launch.remote('local',prefix+['spawn','--request',json.dumps(request)]))
    assert result['request']==request
    until=time.monotonic()+10
    while True:
        report=json.loads(launch.remote('local',prefix+['status','--request',json.dumps(request)]))
        if not report['alive']:break
        assert time.monotonic()<until
        time.sleep(.1)
    assert report['exit_code']==0
