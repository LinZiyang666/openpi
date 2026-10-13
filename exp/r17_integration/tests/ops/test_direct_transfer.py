import json
from pathlib import Path
import socket
import subprocess

import pytest

from exp.r17_integration.ops import direct_transfer as sync


def test_rsync_daemon_roundtrip_and_owned_cleanup(tmp_path):
    source=tmp_path/'source'; source.write_bytes(bytes(range(256))*512)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    server=sync.daemon(str(tmp_path/'server'),'unit',port,str(source))
    output=tmp_path/'output'
    try:
        subprocess.run(['rsync','-a','--bwlimit=1024','--timeout=5',
            f'rsync://127.0.0.1:{port}/unit/payload',str(output)],check=True,capture_output=True)
        assert output.read_bytes()==source.read_bytes()
    finally:
        sync.close_daemon(server)
    assert not (tmp_path/'server').exists()
    assert source.exists()
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(('127.0.0.1',port),timeout=.2)


@pytest.mark.parametrize('node',('h100','timan107','timan1'))
@pytest.mark.parametrize('direction',('push','pull'))
def test_direct_path_and_bwlimit_on_every_transfer(tmp_path,monkeypatch,node,direction):
    source=tmp_path/'file'; source.write_bytes(b'cpu transport')
    calls=[]
    def remote(node,argv,**kwargs):
        calls.append((node,argv))
        if argv[0]=='python3':
            return json.dumps(dict(pid=123,root='/tmp/owned'))
        return ''
    monkeypatch.setattr(sync,'daemon',lambda *a:dict(pid=123,root='/tmp/owned'))
    monkeypatch.setattr(sync,'close_daemon',lambda *a:None)
    argv=['rsync',direction,str(source),node+':/scratch/zixuans8/file'] if direction=='push' else [
        'rsync',direction,node+':/scratch/zixuans8/file',str(source)]
    sync.transfer(argv,remote=remote,execute=lambda argv,**kw:calls.append(('local',argv)))
    payload=[argv for _,argv in calls if argv[0]=='rsync']
    assert len(payload)==1 and '--bwlimit=16384' in payload[0]
    url=next(v for v in payload[0] if v.startswith('rsync://'))
    assert ('149.165.153.233' if node=='h100' else 'ziyanglin.com') in url
    assert ':23118/' in url
    assert not any(argv[:2] in (['tether','push'],['tether','pull']) for _,argv in calls)
