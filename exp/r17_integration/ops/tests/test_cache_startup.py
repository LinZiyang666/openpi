"""Fresh-port cache startup and worker liveness, without CUDA or remote I/O."""
import json
from unittest import mock

import pytest

from exp.r17_integration.ops import formal_eval as launch


@pytest.mark.parametrize('model,node', [('dp','local'),('fastwam','h100'),('cosmos','h100')])
def test_cache_fresh_ports_allocate_owned_policy_and_driver_routes(tmp_path,model,node):
    c=dict(model=model,server_node=node,cache_study='STUDY.json',run_name='cpu_cache',
           public_port=23400,driver_port=23419,endpoint_host='unforwarded.invalid')
    calls=[]
    def expose(argv,directory):
        calls.append(argv)
        return json.dumps(dict(node=argv[2],name=argv[-2],public_host='broker.invalid',port=32000+len(calls)))
    with mock.patch.object(launch,'tunnel_command',side_effect=expose):
        routes=launch.open_tunnels(c,tmp_path)
    assert len(routes)==2
    assert [a[2] for a in calls]==['weilandserver' if node=='local' else 'h100','weilandserver']
    assert c['endpoint']=='broker.invalid:32001'
    assert c['driver_host']=='broker.invalid' and c['driver_public_port']==32002
    assert launch.backend_control_host(c)==('127.0.0.1' if node=='local' else '149.165.153.233')
    assert json.loads((tmp_path/'TUNNELS.json').read_text())==routes


@pytest.mark.parametrize('model,node', [('dp','local'),('fastwam','h100')])
def test_default_off_native_routing_remains_identical(tmp_path,model,node):
    c=dict(model=model,server_node=node,endpoint_host='control.invalid')
    before=dict(c)
    with mock.patch.object(launch,'tunnel_command',side_effect=AssertionError('default hook invoked')):
        assert launch.open_tunnels(c,tmp_path)==[]
    assert c==before
    assert launch.backend_control_host(c)=='control.invalid'
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('status',[
    dict(status='CLOSED',exit_code=0,alive=False,child_alive=False),
    dict(status='CLOSED',exit_code=2,alive=False,child_alive=False),
    dict(status='FAIL',error='worker spawn failed',alive=True),
    dict(status='RUNNING',alive=True,child_alive=False),
])
def test_dead_worker_fleet_is_fatal_before_driver_even_with_lease(status):
    assert launch.worker_fleet_failed(status,dispatch_started=False)


def test_completed_fleet_close_and_new_probe_grace():
    assert not launch.worker_fleet_failed({},dispatch_started=False)
    assert not launch.worker_fleet_failed(dict(status='RUNNING',alive=True,child_alive=True),dispatch_started=False)
    assert not launch.worker_fleet_failed(dict(status='CLOSED',exit_code=0),dispatch_started=True)
    assert launch.worker_fleet_failed(dict(status='CLOSED',exit_code=2),dispatch_started=True)
    assert launch.worker_fleet_failed(dict(status='CLOSED'),dispatch_started=True)


def test_cache_gpu_exclusion_is_local_run_only(tmp_path):
    from exp.r17_integration.ops.cache_lane import gpu_launch_lock
    path=tmp_path/'gpu.lock'
    with gpu_launch_lock('local','run',path):
        with pytest.raises(RuntimeError,match='4090 GPU lock is held'):
            with gpu_launch_lock('local','run',path):pytest.fail('overlapping GPU launch')
        with gpu_launch_lock('h100','run',path):pass
        with gpu_launch_lock('local','prepare',path):pass
    with gpu_launch_lock('local','run',path):pass
    unused=tmp_path/'must_not_open'
    with gpu_launch_lock('h100','run',unused):pass
    assert not unused.exists()


@pytest.mark.parametrize('status',[
    dict(status='CLOSED',exit_code=0,alive=False,child_alive=False),
    dict(status='RUNNING',alive=True,child_alive=False),
])
def test_launcher_fails_before_backend_wait_for_lost_fleet(tmp_path,status):
    import sys
    import socket
    sockets=[socket.socket() for _ in range(3)]
    try:
        for sock in sockets:sock.bind(('127.0.0.1',0))
        ports=[sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:sock.close()
    c=dict(model='dp',server_node='local',cache_study='STUDY.json',runtime=str(launch.ROOT),
           run_name='cpu_dead_fleet',seconds=30,ports=ports,backend_ports=[ports[1]],
           public_port=ports[0],driver_port=ports[2],endpoint_host='broker.invalid',
           server_run=str(tmp_path/'servers'))
    item=dict(python=sys.executable,stage=str(tmp_path),env={},payload=str(launch.ROOT),worker_run=str(tmp_path/'fleet'))
    def remote(node,argv,**kwargs):
        if '-c' in argv:
            if 'read_text' in argv[-1]:return json.dumps(dict(reason='lost before dispatch',worker_failures=[]))
            return json.dumps(dict(free=c['ports'][:-1]))
        mode=argv[argv.index('exp.r17_integration.ops.formal_process')+1]
        if mode=='spawn':return json.dumps(dict(status='RUNNING',child=dict(pid=123),supervisor=dict(pid=456)))
        if mode=='stop':return json.dumps(dict(status='CLOSED',exit_code=0,alive=False,child_alive=False))
        raise AssertionError(argv)
    def refresh(owned,directory,cache,*args):
        cache['timan107','workers']=status
        return 0
    child=mock.Mock(pid=42,poll=mock.Mock(return_value=None),wait=mock.Mock(return_value=0))
    with mock.patch.object(launch,'host_python',return_value=sys.executable), \
         mock.patch.object(launch,'server_env',return_value={}), \
         mock.patch.object(launch,'remote',side_effect=remote), \
         mock.patch.object(launch,'refresh_remote_owned',side_effect=refresh), \
         mock.patch.object(launch.subprocess,'Popen',return_value=child), \
         mock.patch.object(launch,'ticks',return_value='cpu'), \
         mock.patch.object(launch,'stop_group',return_value=dict(status='closed')), \
         mock.patch.object(launch,'alive',return_value=False), \
         mock.patch.object(launch.socket,'create_connection',side_effect=AssertionError('must fail before port wait')):
        assert launch.run_campaign(c,tmp_path,{'timan107':item})==2
    receipt=json.loads((tmp_path/'RECEIPT.json').read_text())
    assert receipt['status']=='FAIL' and receipt['all_owned_closed']
    assert 'lost before dispatch' in receipt['error']
    assert receipt['worker_fleet_failure']['process_status']==status
