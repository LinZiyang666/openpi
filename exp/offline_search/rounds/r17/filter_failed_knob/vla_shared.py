"""Scoped legacy worker locks; native serving, admission and plans are unchanged."""
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import inspect
import os
from pathlib import Path
from unittest.mock import patch


def worker_lock(root, island):
    scope = hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:16]
    return str(Path(island)/'os_cl/locks'/('ffk_'+scope+'.lock'))


@contextmanager
def fleet_lock(root=None):
    from exp.offline_search.closed_loop.ops.h100 import control as c
    root = Path(root or os.environ['FFK_WORKER_FLEET_ROOT']).resolve()
    if not root.name.startswith('fa_'):
        raise ValueError('Shared legacy fleets require an explicit formal-A root')
    # Retain the maintenance gate and exclusive ownership of this run. A
    # different fleet's host-wide lock is neither acquired nor modified.
    with ExitStack() as stack:
        for path, mode in ((Path(str(c.CHAIN_LOCK)+'.gate'), fcntl.LOCK_SH),
                           (root/'state/h100_chain.lock', fcntl.LOCK_EX)):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, mode | fcntl.LOCK_NB)
        yield c.legacy_chains(ignore_pid=os.getpid())


def configure_node(n, lock, marker):
    """Executed in a fresh RPC process; never overwrite the deployed helper."""
    import inspect
    from pathlib import Path
    import os
    atomic = n.atomic
    def save(path, value):
        if isinstance(value, dict) and value.get('role') == 'driver' and 'cmd' in value:
            value = dict(value, env=dict(value['env'], FFK_WORKER_FLEET=marker))
            cmd = value['cmd']
            if cmd[:2] == ['bash', str(n.ISLAND/'os_cl/run_arm.sh')]:
                source = Path(cmd[1]).read_text()
                old = 'exec 9>"$R/os_cl/locks/fleet.lock"'
                if source.count(old) != 1:
                    raise ValueError('Unknown native run_arm fleet lock')
                source = source.replace(old, 'exec 9>"${FFK_WORKER_LOCK:?}"')
                value['env']['FFK_WORKER_LOCK'] = lock
                # $0 and ownership's argv needle remain the native script path.
                value['cmd'] = ['bash', '-c', source, cmd[1], *cmd[2:]]
        return atomic(path, value)
    def workers(proc_root=Path('/proc')):
        found = 0
        for p in proc_root.glob('[0-9]*'):
            try:
                if p.stat().st_uid != os.getuid(): continue
                argv = (p/'cmdline').read_bytes().split(b'\0')
                if b'examples.libero.worker_entry' not in argv: continue
                if ('FFK_WORKER_FLEET='+marker).encode() in (p/'environ').read_bytes().split(b'\0'):
                    found += 1
            except (FileNotFoundError, ProcessLookupError, PermissionError): pass
        return found
    # Native cleanup still waits for its own fleet lock and workers only.
    source = inspect.getsource(n.wait_fleet)
    source = source.replace('path = ISLAND / "os_cl/locks/fleet.lock"', 'path = Path('+repr(lock)+')')
    exec(source, n.__dict__)
    n.atomic, n.island_workers = save, workers


@contextmanager
def r15(root):
    from exp.offline_search.closed_loop.ops.h100 import control as c
    native_remote = c.remote
    source = inspect.getsource(configure_node)
    def boot(host):
        # Every fleet of the arm (driver and agent-only ones): this root's own lock on that fleet's island.
        return ('import importlib.util\n'
                's=importlib.util.spec_from_file_location("ffk_native_node",'+repr(c.REMOTE_NODE[host])+')\n'
                'n=importlib.util.module_from_spec(s);s.loader.exec_module(n)\n'+source+'\n'
                'configure_node(n,'+repr(worker_lock(root, c.island(host)))+','+repr(Path(root).name)+')\nn.main()\n')
    def remote(node, argv, *a, **kw):
        argv = list(map(str, argv))
        if node in c.FLEETS and argv[:2] == ['python3', c.REMOTE_NODE[node]]:
            argv = ['/usr/bin/python3', '-B', '-S', '-c', boot(node), *argv[2:]]
        return native_remote(node, argv, *a, **kw)
    with patch.object(c, 'fleet_lock', fleet_lock), patch.object(c, 'remote', remote):
        yield
