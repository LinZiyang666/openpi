"""Bandwidth-limited rsync; tether carries remote commands only."""
from __future__ import annotations

import fcntl
import inspect
import json
import os
from pathlib import Path
import uuid

PORT = 23118  # Register on weilandserver's GPU-share ledger before collection.
BWLIMIT_KIBPS = 16384


def daemon(root, module, port, source=None):
    """Self-contained remote command; export only one transfer's private stage."""
    import os
    from pathlib import Path
    import socket
    import subprocess
    import time
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    data = root/'data'
    data.mkdir(mode=0o700)
    if source:
        os.link(source, data/'payload')
    config = root/'rsyncd.conf'
    # Non-root rsync must retain its groups; root's default would become nobody.
    identity = ('uid = '+str(os.getuid())+'\ngid = '+str(os.getgid())+'\n') if os.geteuid()==0 else ''
    config.write_text('use chroot = no\nlock file = '+str(root/'lock')+'\npid file = '+str(root/'pid')+
        '\nlog file = '+str(root/'log')+
        '\n'+identity+'['+module+']\npath = '+str(data)+'\nread only = '+('yes' if source else 'no')+
        '\nlist = no\nmax connections = 1\n')
    log = (root/'log').open('wb')
    subprocess.run(['rsync','--daemon','--port='+str(port),
        '--config='+str(config)],stdout=log,stderr=log,stdin=subprocess.DEVNULL,check=True,timeout=10)
    log.close()
    for _ in range(100):
        if (root/'pid').exists():
            break
        time.sleep(.05)
    pid = int((root/'pid').read_text())
    try:
        for _ in range(100):
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.1):
                    return dict(pid=pid,root=str(root))
            except OSError:
                time.sleep(.05)
        raise RuntimeError('rsync daemon did not listen')
    except BaseException:
        import signal
        os.kill(pid,signal.SIGTERM)
        raise


def close_daemon(server, destination=None):
    import os
    from pathlib import Path
    import signal
    import time
    root = Path(server['root'])
    cmdline = Path('/proc')/str(server['pid'])/'cmdline'
    def owns_daemon():
        try:
            return ('--config='+str(root/'rsyncd.conf')).encode() in cmdline.read_bytes().split(bytes([0]))
        except (FileNotFoundError, ProcessLookupError):
            return False
    if owns_daemon():
        try:
            os.kill(server['pid'],signal.SIGTERM)
        except ProcessLookupError:
            pass
        until = time.monotonic()+2
        while owns_daemon() and time.monotonic()<until:
            time.sleep(.05)
        if owns_daemon():
            try:
                os.kill(server['pid'],signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(.1)
        try:
            os.waitpid(server['pid'],0)
        except ChildProcessError:
            pass
    if destination:
        target = Path(destination)
        target.parent.mkdir(parents=True,exist_ok=True)
        os.replace(root/'data/payload',target)
    for file in (root/'data').iterdir():
        file.unlink()
    (root/'data').rmdir()
    for file in root.iterdir():
        file.unlink(missing_ok=True)
    root.rmdir()


def transfer(argv, *, remote, execute, heartbeat=lambda: None):
    """Accept rsync push/pull tokens, plus old internal tokens during migration.

    H100 exports an ephemeral daemon at its public IP. Worker nodes run rsync
    against weilandserver's forwarded port. No SSH credentials or broker data.
    """
    direction, source, destination = argv[1:4]
    if direction not in ('push','pull'):
        raise ValueError('transfer direction must be push or pull')
    node, path = str(destination if direction == 'push' else source).split(':',1)
    local = Path(source if direction == 'push' else destination).resolve()
    if node == 'local':
        from .local_workers import copy
        return copy(local,path) if direction == 'push' else copy(path,local)
    token = 'r17_'+uuid.uuid4().hex
    flags = ['rsync','-a','--timeout=90','--contimeout=15','--bwlimit='+str(BWLIMIT_KIBPS)]
    def call(function, *args):
        code = inspect.getsource(function)+'\nimport json; print(json.dumps('+function.__name__+'(*'+repr(args)+')))'
        return json.loads(remote(node,['python3','-c',code],seconds=60))
    with open('/tmp/r17_direct_rsync.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        heartbeat()
        if node == 'h100':
            from .formal_eval import h100_host  # R17_H100_HOST alias; remote() maps the node
            root = str(Path(path).parent/('.'+token))
            # Source hardlinks stay on their source filesystem; no extra archive copy.
            server = call(daemon,root,token,PORT,path if direction == 'pull' else None)
            url = f'rsync://{h100_host()}:{PORT}/{token}/payload'
            local.parent.mkdir(parents=True,exist_ok=True)
            complete = False
            try:
                # rsync --timeout catches stalls; the wall cap only bounds a slow but moving copy (shared ~33 MB/s ingress).
                execute([*flags,str(local),url] if direction == 'push' else [*flags,url,str(local)],seconds=7200)
                complete = True
            finally:
                call(close_daemon,server,path if complete and direction == 'push' else None)
        else:
            local.parent.mkdir(parents=True,exist_ok=True)
            root = str(local.parent/('.'+token))
            server = daemon(root,token,PORT,str(local) if direction == 'push' else None)
            url = f'rsync://ziyanglin.com:{PORT}/{token}/payload'
            complete = False
            try:
                if direction == 'push':
                    remote(node,['mkdir','-p',str(Path(path).parent)],replay_safe=True)
                remote(node,[*flags,url,path] if direction == 'push' else [*flags,path,url],seconds=7200)
                complete = True
            finally:
                close_daemon(server,str(local) if complete and direction == 'pull' else None)
        heartbeat()
        return dict(status='ok',lane='rsync',bwlimit_kibps=BWLIMIT_KIBPS)
