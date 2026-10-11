"""Real CPU TCP regressions for consecutive campaign admission."""
import json
import socket
import threading
import time

import pytest

from exp.r17_integration.ops import formal_ports as ports


def listener():
    sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(('0.0.0.0',0));sock.listen(1)
    return sock,sock.getsockname()[1]


def test_time_wait_can_be_reused_by_actual_listener():
    server,port=listener()
    client=socket.create_connection(('127.0.0.1',port));accepted,_=server.accept()
    # The server actively closes; a plain bind reproduces C32's Errno98.
    accepted.close();client.close();server.close()
    with socket.socket() as plain,pytest.raises(OSError):plain.bind(('0.0.0.0',port))
    assert ports.probe_ports([port])==dict(free=True,ports=[port],busy=[])


def test_busy_listener_waits_and_is_never_signalled():
    server,port=listener()
    timer=threading.Timer(.15,server.close);timer.start()
    samples=ports.wait_release(lambda:ports.probe_ports([port]),seconds=2)
    timer.join()
    assert samples[0]['busy']==[port] and samples[-1]['free']


def test_busy_listener_timeout_preserves_owner():
    server,port=listener()
    try:
        with pytest.raises(TimeoutError):
            ports.wait_release(lambda:ports.probe_ports([port]),seconds=.02)
        assert not ports.probe_ports([port])['free']
        with socket.create_connection(('127.0.0.1',port),timeout=.2):pass
    finally:server.close()


def test_remote_by_value_admission_records_both_hosts(tmp_path):
    with socket.socket() as temporary:
        temporary.bind(('0.0.0.0',0));port=temporary.getsockname()[1]
    calls=[]
    def remote(node,argv,**kwargs):
        calls.append((node,argv,kwargs))
        return json.dumps(dict(free=True,ports=[23173,23174],busy=[]))
    result=ports.admit(dict(server_node='h100',ports=[23173,23174,port],driver_port=port),tmp_path,
        remote=remote,python='/frozen/python')
    assert result['status']=='PASS' and set(result['samples'])=={'h100','local_driver'}
    assert 'SO_REUSEADDR' in calls[0][1][-1] and calls[0][2]['replay_safe']
    assert json.loads((tmp_path/'PORT_RELEASE.json').read_text())['cpu_only']
