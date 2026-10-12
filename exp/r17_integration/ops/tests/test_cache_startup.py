"""Fresh-port cache startup and worker liveness, without CUDA or remote I/O."""
import json
from unittest import mock

import pytest

from exp.r17_integration.ops import formal_eval as launch


def test_local_server_with_local_workers_uses_loopback_without_broker(tmp_path):
    c=dict(model='dp',server_node='local',cache_study='STUDY.json',run_name='cpu_cache',
           worker_nodes={'local':48},public_port=23400,driver_port=23419)
    with mock.patch.object(launch.subprocess,'run',side_effect=AssertionError('no tether call')):
        assert launch.open_tunnels(c,tmp_path)==[]
    assert c['endpoint']=='127.0.0.1:23400' and c['driver_public_port']==23419
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('model',['fastwam','cosmos'])
def test_h100_server_with_remote_workers_connects_directly(tmp_path,model):
    c=dict(model=model,server_node='h100',cache_study='STUDY.json',run_name='cpu_cache',
           worker_nodes={'timan107':45,'timan1':40},public_port=23420,driver_port=23111)
    assert launch.open_tunnels(c,tmp_path)==[]
    assert c['endpoint']=='149.165.153.233:23420'
    assert (c['driver_host'],c['driver_public_port'])==('ziyanglin.com',23111)
    assert launch.backend_control_host(c)=='149.165.153.233'


@pytest.mark.parametrize('node,public,driver',[('h100',23420,23439),('local',23400,23111)])
def test_remote_workers_need_forwarded_weilandserver_ports(tmp_path,node,public,driver):
    c=dict(model='dp',server_node=node,worker_nodes={'timan107':45},public_port=public,driver_port=driver)
    with pytest.raises(ValueError,match='23100-23199'):
        launch.open_tunnels(c,tmp_path)


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
