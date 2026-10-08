"""Receipt/exec races with fake processes and time; no launches or signals."""
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from exp.offline_search.closed_loop.ops.h100 import local_server as local, node


@pytest.fixture
def launch(tmp_path, monkeypatch):
    spec = dict(role='server', server_host='local', run_root=str(tmp_path),
                out=str(tmp_path / 'runs/arm/server'), tag='arm_23150', port=23150)
    argv = ['/fake/venv/bin/python', '/fake/serve.py', '--port', '23150']
    cmd = ['taskset', '-c', '22-23,66', *argv]
    job_path = tmp_path / 'job.json'
    job_path.write_text(json.dumps(dict(spec=spec, cmd=cmd, env={})))
    _, receipt_path, _ = local.paths(spec)
    receipt_path.parent.mkdir(parents=True)
    fake = SimpleNamespace(now=0., pid=99999999, birth='child_birth', delay=6.,
                           cmd=cmd, argv=argv, spawned=[], affinity=[], waits=0,
                           signals=[], command_override=None)

    def cmdline():
        if fake.command_override is not None:
            return fake.command_override
        current = fake.cmd if fake.now < fake.delay else argv
        return b''.join(os.fsencode(arg) + b'\0' for arg in current)

    read_bytes = Path.read_bytes
    def read(path, *args, **kwargs):
        if str(path) == f'/proc/{fake.pid}/cmdline':
            return cmdline()
        return read_bytes(path, *args, **kwargs)

    identity = local.proc_identity
    monkeypatch.setattr(local, 'proc_identity', lambda pid: fake.birth if pid == fake.pid else identity(pid))
    monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(local.time, 'monotonic', lambda: fake.now)
    monkeypatch.setattr(local.time, 'sleep', lambda seconds: setattr(fake, 'now', fake.now + seconds))
    monkeypatch.setattr(local.os, 'sched_setaffinity', lambda pid, cpus: fake.affinity.append((pid, cpus)))
    monkeypatch.setattr(local.os, 'kill', lambda pid, sig: fake.signals.append((pid, sig)))
    monkeypatch.setattr(local.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(node, 'call', lambda *args, **kwargs: pytest.fail('no external operations'))
    monkeypatch.setattr(local, 'listeners', lambda: pytest.fail('no listener checks'))
    monkeypatch.setattr(local, 'gpu_free', lambda: pytest.fail('no GPU probes'))

    def wait():
        fake.waits += 1
        # Deterministically exec after the old five-second polling deadline.
        fake.now = fake.delay + 1
        receipt = json.loads(receipt_path.read_text())['child']
        assert local.authenticated(receipt)
        assert receipt['cmdline'] == cmdline().hex()
        local.signal_owned(receipt, signal.SIGTERM)
        assert fake.signals == [(fake.pid, signal.SIGTERM)]
        fake.signals.clear()
        fake.command_override = b'foreign\0'
        with pytest.raises(RuntimeError, match='command differs'):
            local.signal_owned(receipt, signal.SIGTERM)
        fake.command_override = None
        fake.birth = 'reused_pid_birth'
        local.signal_owned(receipt, signal.SIGTERM)
        assert not fake.signals
        return 0

    def popen(argv, **kwargs):
        fake.spawned.append((argv, kwargs))
        # Popen acknowledges exec only after preexec_fn has succeeded. A
        # taskset wrapper would still be alive for fake.delay seconds instead.
        if kwargs.get('preexec_fn') is not None:
            kwargs['preexec_fn']()
        fake.cmd = argv
        return SimpleNamespace(pid=fake.pid, poll=lambda: None, wait=wait)

    monkeypatch.setattr(local.subprocess, 'Popen', popen)
    return fake, spec, job_path


@pytest.mark.parametrize('launcher_delay', [.25, 6.])
def test_supervise_records_final_exec_even_with_delayed_launcher(launch, launcher_delay):
    fake, spec, job_path = launch
    fake.delay = launcher_delay
    local.supervise(job_path)
    assert len(fake.spawned) == fake.waits == 1
    argv, kwargs = fake.spawned[0]
    assert argv == fake.argv
    assert fake.affinity == [(0, {22, 23, 66})]
    assert kwargs['start_new_session'] and kwargs['close_fds']
    assert local.matching(spec)['finished']
    assert local.matching(spec)['returncode'] == 0


def test_supervise_rejects_reserved_affinity_before_launch(launch):
    fake, spec, job_path = launch
    job = json.loads(job_path.read_text())
    job['cmd'][2] = '38'
    job_path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match='LOCAL_SERVER_CPUS'):
        local.supervise(job_path)
    receipt = local.matching(spec)
    assert receipt['finished'] and 'child' not in receipt
    assert not fake.spawned and not fake.signals


def test_supervise_affinity_failure_never_records_child(launch, monkeypatch):
    fake, spec, job_path = launch
    def failed_affinity(pid, cpus):
        raise OSError('affinity unavailable')
    monkeypatch.setattr(local.os, 'sched_setaffinity', failed_affinity)
    with pytest.raises(OSError, match='affinity unavailable'):
        local.supervise(job_path)
    receipt = local.matching(spec)
    assert receipt['finished'] and 'child' not in receipt
    assert fake.waits == 0 and not fake.signals


def test_supervise_accepts_direct_interpreter_job(launch):
    fake, spec, job_path = launch
    job = json.loads(job_path.read_text())
    job['cmd'] = fake.argv
    job_path.write_text(json.dumps(job))
    local.supervise(job_path)
    assert fake.spawned[0][0] == fake.argv
    assert not fake.affinity and fake.waits == 1
    assert local.matching(spec)['finished']


@pytest.mark.parametrize('model,suite', [('pi05', 'libero_10'), ('groot', 'libero_10'),
                                        ('groot', 'libero_spatial')])
@pytest.mark.parametrize('stock', ['0', '1'])
def test_h100_server_command_starts_interpreter_directly(model, suite, stock):
    argv, _ = node.command(model, suite, 23250, '/fake/config.yaml', '/fake/out', 'arm', [],
                           dict(STOCK=stock))
    assert argv[0].endswith('/bin/python')
    assert argv[1].endswith('.py') and '/fake/config.yaml' in argv
    assert 'taskset' not in argv
