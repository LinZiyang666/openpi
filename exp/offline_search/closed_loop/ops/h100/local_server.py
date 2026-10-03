"""Detached local policy servers. Only receipt-authenticated PIDs may be signalled.

There is one process per public port (pi05 --replicas 1); no internal ports or
tmux sessions. The supervisor records exits even when a controller disappears.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from . import node

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[4]
LOCKS = HERE / '.local_locks'
CPUS = '22-37,66-81'
GROOT = Path('/home/weiland/projects/openpi_ext/third_party/gr00t_n15')
GROOT_PY = '/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python'
PI05_CKPT = '/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch'
PORT_RELEASE_TIMEOUT = 60


def cpu_affinity():
    value = os.environ.get('LOCAL_SERVER_CPUS', CPUS)
    selected = set()
    try:
        for part in value.split(','):
            ends = [int(v) for v in part.split('-')]
            if len(ends) == 1:
                selected.add(ends[0])
            elif len(ends) == 2 and ends[0] <= ends[1]:
                selected.update(range(ends[0], ends[1] + 1))
            else:
                raise ValueError
    except ValueError:
        raise ValueError('invalid LOCAL_SERVER_CPUS') from None
    allowed = set(range(22, 38)) | set(range(66, 82))
    if not selected or not selected <= allowed:
        raise ValueError('LOCAL_SERVER_CPUS must be within 22-37,66-81')
    return value


def listeners():
    output = node.call(['ss', '-tlnpH'])
    result = {}
    for line in output.splitlines():
        columns = line.split()
        if len(columns) < 5:
            continue
        port = int(columns[3].rsplit(':', 1)[1])
        result.setdefault(port, set()).update(map(int, re.findall(r'pid=(\d+)', line)))
    return result


def port_block(port, replicas=1):
    """Include proxy child ports if a future stock topology uses replicas."""
    return list(range(port, port + (replicas + 1 if replicas > 1 else 1)))


def can_bind(port):
    try:
        with socket.socket() as sock:
            sock.bind(('0.0.0.0', port))
        return True
    except OSError:
        return False


def choose_ports(count=4, requested=None, replicas=1):
    if count < 1 or replicas < 1:
        raise ValueError('positive server count and replicas required')
    busy = listeners()
    def available(ports):
        all_ports = [p for public in ports for p in port_block(public, replicas)]
        if len(set(all_ports)) != len(all_ports) or any(not 23100 <= p <= 23199 for p in all_ports):
            raise ValueError('PORTS must be distinct, nonoverlapping ports in 23100..23199')
        for port in all_ports:
            if port in busy or not can_bind(port):
                return False
        return True
    if requested is not None:
        if not requested:
            raise ValueError('PORTS must not be empty')
        if not available(requested):
            raise RuntimeError('LOCAL_PORT_BUSY: requested listener belongs to another process')
        return list(requested)
    width = replicas + 1 if replicas > 1 else 1
    # Leave the default sync/export ports available for the other server host.
    for first in range(23100, 23197 - count * width + 1):
        ports = [first + i * width for i in range(count)]
        if available(ports):
            return ports
    raise RuntimeError('LOCAL_PORT_BUSY: no free block in 23100..23196')


def gpu_free():
    return int(node.call(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                          '--format=csv,noheader,nounits']).splitlines()[0])


def command(spec):
    env = dict(spec['env'])
    env.setdefault('PI05_CKPT', PI05_CKPT)
    env.setdefault('GROOT_CKPT', f"/data/ckpt/n15_{spec['suite']}")
    env.setdefault('OMP', '1')
    argv, env = node.command(spec['model'], spec['suite'], spec['port'], spec['yaml'],
                             spec['out'], spec['tag'], spec['plugin'], env)
    argv[0] = str(TREE / '.venv/bin/python') if spec['model'] == 'pi05' else GROOT_PY
    argv[1] = str(TREE / Path(argv[1]).relative_to(node.TREE))
    out = Path(spec['out'])
    runtime = out / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    env.update(HOME='/home/weiland', PYTHONPATH=f'{GROOT}:{GROOT}/examples/Libero:{TREE}:{TREE}/src:{TREE}/packages/openpi-client/src',
               TMPDIR=str(runtime), TORCHINDUCTOR_CACHE_DIR=str(runtime / 'inductor'),
               TRITON_CACHE_DIR=str(runtime / 'triton'), PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return ['taskset', '-c', cpu_affinity(), *argv], env


def paths(spec):
    if spec.get('role') != 'server' or spec.get('server_host') != 'local':
        raise ValueError('local backend accepts only local server specs')
    root, out = Path(spec['run_root']).resolve(), Path(spec['out']).resolve()
    if not node.within(out, root / 'runs'):
        raise ValueError('local output must be under run_root/runs')
    if not re.fullmatch(r'[A-Za-z0-9_]+', spec['tag']):
        raise ValueError('invalid server tag')
    name = 'server_' + spec['tag']
    return out, out / (name + '.owner.json'), out / (name + '.log')


@contextmanager
def locked(spec):
    paths(spec)
    port = int(spec['port'])
    if not 23100 <= port <= 23199:
        raise ValueError('local port must be in 23100..23199')
    LOCKS.mkdir(parents=True, exist_ok=True)
    # Serialize starts/free-memory checks on this GPU as well as each port.
    with open(LOCKS / 'gpu.lock', 'a') as gpu, open(LOCKS / f'{port}.lock', 'a') as lock:
        fcntl.flock(gpu, fcntl.LOCK_EX)
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def proc_identity(pid):
    try:
        tail = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if tail[0] == 'Z' else tail[19]
    except FileNotFoundError:
        return None


def process_present(process):
    """Keep waiting through an empty exit-time cmdline, without accepting PID reuse."""
    return bool(process and process.get('starttime') is not None
                and proc_identity(process['pid']) == process['starttime'])


def authenticated(process, *, exiting=False):
    if not process_present(process):
        return False
    try:
        raw = Path(f"/proc/{process['pid']}/cmdline").read_bytes().hex()
    except FileNotFoundError:
        return False
    if raw != process['cmdline']:
        # During exit Linux clears cmdline before the task is reaped or marked Z.
        if not raw:
            # An unchanged birth time authenticates an exiting receipt for waiting
            # only. Signalling still requires the recorded command line.
            return exiting and process_present(process)
        if not process_present(process):
            return False
        raise RuntimeError('local PID command differs from ownership receipt; refusing')
    return True


def process_receipt(pid):
    return dict(pid=pid, starttime=proc_identity(pid),
                cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().hex())


def matching(spec):
    _, receipt_path, _ = paths(spec)
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt['spec'] != spec:
        raise RuntimeError('local receipt does not match requested launch')
    return receipt


def previous_launches(spec):
    """Find receipts for this port, confined to this chain's own run root."""
    root = Path(spec['run_root']).resolve()
    receipts = []
    for path in (root / 'runs').rglob('server_*.owner.json'):
        receipt = json.loads(path.read_text())
        old = receipt.get('spec', {})
        if (old.get('role') != 'server' or old.get('server_host') != 'local'
                or old.get('port') != spec['port']
                or Path(old.get('run_root', '')).resolve() != root):
            continue
        # A copied or misplaced receipt cannot claim a listener.
        if paths(old)[1].resolve() == path.resolve():
            receipts.append(receipt)
    return receipts


def wait_port_release(spec, receipts, deadline):
    processes = [receipt[key] for receipt in receipts for key in ('child', 'supervisor')
                 if receipt.get(key)]
    children = [receipt['child'] for receipt in receipts if receipt.get('child')]
    while True:
        live = [process for process in processes if authenticated(process, exiting=True)]
        owned_pids = {process['pid'] for process in children if process in live}
        busy = listeners().get(spec['port'])
        if busy is not None and (not busy or not busy <= owned_pids):
            raise RuntimeError('LOCAL_PORT_BUSY: requested listener belongs to another process')
        released = busy is None and can_bind(spec['port'])
        if not live and released:
            return
        if not receipts:
            raise RuntimeError('LOCAL_PORT_BUSY: requested listener belongs to another process')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"LOCAL_PORT_RELEASE_TIMEOUT: owned launch/port {spec['port']} did not release")
        time.sleep(min(.1, remaining))


def start(spec):
    with locked(spec):
        out, rp, log = paths(spec)
        if rp.exists():
            old = json.loads(rp.read_text())
            live = authenticated(old.get('supervisor')) or authenticated(old.get('child'))
            if live:
                if old['spec'] == spec:
                    return 'ALREADY_STARTED local'
                if not (old.get('finished') or old.get('stopped')):
                    raise RuntimeError('occupied local launch; refusing')
            if old['spec'] == spec and (old.get('finished') or old.get('stopped')):
                return 'ALREADY_FINISHED local'
        deadline = time.monotonic() + PORT_RELEASE_TIMEOUT
        wait_port_release(spec, previous_launches(spec), deadline)
        choose_ports(requested=[spec['port']])
        if gpu_free() < spec['need_mb']:
            raise RuntimeError(f"GPU_TIGHT server_host=local need={spec['need_mb']}")
        if not Path(spec['yaml']).is_file():
            raise FileNotFoundError(spec['yaml'])
        out.mkdir(parents=True, exist_ok=True)
        argv, env = command(spec)
        if log.exists():
            log.replace(log.with_name(log.name + f'.prev.{time.time_ns()}'))
        job = out / ('server_' + spec['tag'] + '.job.json')
        node.atomic(job, dict(spec=spec, cmd=argv, env=env))
        node.atomic(rp, dict(spec=spec))
        launch = ['taskset', '-c', cpu_affinity(), str(TREE / '.venv/bin/python'), '-B',
                  '-m', 'exp.offline_search.closed_loop.ops.h100.local_server', 'supervise', str(job)]
        with open(log, 'a') as stream:
            # setsid(2), stdin detached, all output retained under the run root.
            proc = subprocess.Popen(launch, cwd=TREE, env={**os.environ, **env}, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=stream, start_new_session=True, close_fds=True)
        try:
            # Supervisor writes its own post-exec identity, then child's post-exec identity.
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                receipt = matching(spec)
                if receipt.get('child') or receipt.get('finished'):
                    return f"STARTED local supervisor_pid={proc.pid} port={spec['port']}"
                if proc.poll() is not None:
                    raise RuntimeError('local supervisor failed: ' + log.read_text()[-3000:])
                time.sleep(.05)
            raise RuntimeError('local supervisor did not acknowledge launch')
        except BaseException:
            # This Popen object owns the PID, including failures before a receipt exists.
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=15)
            raise


def supervise(job_path):
    job = json.loads(Path(job_path).read_text())
    spec = job['spec']
    _, rp, log = paths(spec)
    receipt = dict(spec=spec, supervisor=process_receipt(os.getpid()))
    node.atomic(rp, receipt)
    child = None
    def forward(sig, frame):
        if child is not None and child.poll() is None:
            child.terminate()  # only this supervisor's child PID
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    rc = 1
    try:
        with open(log, 'a', buffering=1) as stream:
            child = subprocess.Popen(job['cmd'], cwd=TREE, env={**os.environ, **job['env']},
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                     start_new_session=True, close_fds=True)
            # taskset execs the serving interpreter. Wait for that exec before recording argv.
            deadline = time.monotonic() + 5
            while child.poll() is None and time.monotonic() < deadline:
                argv = Path(f'/proc/{child.pid}/cmdline').read_bytes().split(b'\0')
                if b'taskset' not in argv[0]:
                    break
                time.sleep(.01)
            if child.poll() is None:
                receipt['child'] = process_receipt(child.pid)
                node.atomic(rp, receipt)
            rc = child.wait()
    finally:
        with open(log, 'a') as stream:
            stream.write(f'SERVER_EXIT={rc}\n')
        receipt.update(finished=True, returncode=rc)
        node.atomic(rp, receipt)


def status(spec):
    _, _, log = paths(spec)
    receipt = matching(spec)
    running = bool(receipt and authenticated(receipt.get('child')))
    tail = log.read_text(errors='replace')[-3000:] if log.exists() else ''
    listening = running and receipt['child']['pid'] in listeners().get(spec['port'], set())
    return dict(running=running, listening=bool(listening),
                dead=bool(receipt and not running and (receipt.get('finished') or not authenticated(receipt.get('supervisor')))), tail=tail)


def signal_owned(process, sig):
    if authenticated(process):
        try:
            os.kill(process['pid'], sig)
        except ProcessLookupError:
            pass  # It exited after authentication and before the signal.


def stop(spec, timeout_s=PORT_RELEASE_TIMEOUT):
    with locked(spec):
        receipt = matching(spec)
        if not receipt:
            return 'NO_OWNED_LAUNCH'
        budget = max(0, min(timeout_s, PORT_RELEASE_TIMEOUT))
        deadline = time.monotonic() + budget
        graceful_deadline = deadline - min(10, budget / 2)
        for key in ('child', 'supervisor'):
            process = receipt.get(key)
            signal_owned(process, signal.SIGTERM)
            while process_present(process) and time.monotonic() < graceful_deadline:
                time.sleep(max(0, min(.1, graceful_deadline - time.monotonic())))
            if process_present(process):
                signal_owned(process, signal.SIGKILL)
            while process_present(process) and time.monotonic() < deadline:
                time.sleep(max(0, min(.1, deadline - time.monotonic())))
            if process_present(process):
                raise RuntimeError('LOCAL_PORT_RELEASE_TIMEOUT: owned local process did not exit')
        receipt = matching(spec)
        wait_port_release(spec, [receipt], deadline)
        receipt['stopped'] = True
        _, rp, _ = paths(spec)
        node.atomic(rp, receipt)
        return 'STOPPED local'


def manifest():
    digest, count = hashlib.sha256(), 0
    for folder in ('src', 'scripts', 'packages/openpi-client/src', 'exp/offline_search', 'exp/libero_groot'):
        for path in sorted((TREE / folder).rglob('*')):
            if path.suffix not in ('.py', '.yaml', '.json') or '__pycache__' in path.parts or '.local_locks' in path.parts:
                continue
            if path.is_file():
                digest.update(f'{path.relative_to(TREE)}\0{node.sha(path)}\n'.encode())
                count += 1
    return dict(root=str(TREE), sha256=digest.hexdigest(), files=count,
                rule='local serving source: sorted relative-path NUL file-sha256 LF')


def checkpoints(expected):
    results = {}
    for name, want in expected.items():
        if name == 'rule':
            continue
        if name == 'pi05_libero_pytorch':
            root = Path(os.environ.get('PI05_CKPT', PI05_CKPT))
        elif name in ('n15_libero_10', 'n15_libero_spatial'):
            root = Path('/data/ckpt') / name
        else:
            raise ValueError('unknown local checkpoint ' + name)
        files = sorted(root.glob('*.safetensors'), key=lambda p: p.name)
        digest = hashlib.sha256()
        for path in files:
            digest.update(f'{path.name}\0{node.sha(path)}\n'.encode())
        actual = digest.hexdigest() if files else None
        results[name] = dict(path=str(root), actual=actual, expected=want, match=actual == want)
    if not all(r['match'] for r in results.values()):
        raise RuntimeError('local checkpoint missing/mismatch: ' + json.dumps(results))
    return json.dumps(results, indent=1)


def rpc(action, spec):
    if action == 'start':
        return start(spec)
    if action == 'stop':
        return stop(spec)
    if action == 'status':
        return json.dumps(status(spec))
    if action == 'manifest':
        return json.dumps(manifest())
    if action == 'checkpoints':
        return checkpoints(spec)
    raise ValueError('unsupported local server action: ' + action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'status', 'stop', 'supervise'])
    parser.add_argument('spec')
    args = parser.parse_args()
    if args.action == 'supervise':
        supervise(args.spec)
    else:
        print(rpc(args.action, json.loads(Path(args.spec).read_text())))


if __name__ == '__main__':
    main()
