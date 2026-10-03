"""Port handoff races with fake time/processes; no listeners, GPU probes or launches."""
import copy
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from exp.offline_search.closed_loop.ops.h100 import local_server as local, node


class FakeSystem:
    def __init__(self):
        self.now = 0.0
        self.processes = {}
        self.signals = []
        self.spawns = []
        self.busy = lambda: {}
        self.bindable = lambda port: True

    def add(self, pid, *, exit_at=float('inf'), clear_at=float('inf')):
        receipt = dict(pid=pid, starttime=f'birth_{pid}', cmdline=b'owned\0'.hex())
        self.processes[pid] = dict(receipt=receipt, exit_at=exit_at, clear_at=clear_at,
                                   identity=receipt['starttime'], command=b'owned\0')
        return receipt

    def identity(self, pid):
        process = self.processes.get(pid)
        return process['identity'] if process and self.now < process['exit_at'] else None

    def cmdline(self, pid):
        process = self.processes[pid]
        return b'' if self.now >= process['clear_at'] else process['command']

    def sleep(self, seconds):
        assert seconds > 0
        self.now += seconds

    def kill(self, pid, sig):
        self.signals.append((self.now, pid, sig))


@pytest.fixture
def system(tmp_path, monkeypatch):
    fake = FakeSystem()
    monkeypatch.setattr(local, 'LOCKS', tmp_path / 'locks')
    monkeypatch.setattr(local.time, 'monotonic', lambda: fake.now)
    monkeypatch.setattr(local.time, 'sleep', fake.sleep)
    monkeypatch.setattr(local, 'proc_identity', fake.identity)
    read_bytes = Path.read_bytes
    def read(path, *args, **kwargs):
        if str(path).startswith('/proc/'):
            assert path.name == 'cmdline'
            return fake.cmdline(int(path.parent.name))
        return read_bytes(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(local, 'listeners', lambda: fake.busy())
    monkeypatch.setattr(local, 'can_bind', lambda port: fake.bindable(port))
    monkeypatch.setattr(local.os, 'kill', fake.kill)
    monkeypatch.setattr(node, 'call', lambda *a, **kw: pytest.fail('no external commands'))
    monkeypatch.setattr(local, 'gpu_free', lambda: 49000)
    monkeypatch.setattr(local, 'command', lambda spec: (['fake-server'], {}))
    monkeypatch.setattr(local.subprocess, 'Popen', lambda *a, **kw: pytest.fail('no real launches'))
    return fake


def spec_for(root, arm='next', port=23150):
    return dict(role='server', server_host='local', run_root=str(root),
                out=str(root / 'runs' / arm / f'server_{port}'), tag=f'{arm}_{port}',
                port=port, model='pi05', suite='libero_10', yaml=str(root / 'cfg.yaml'),
                plugin=[], env={}, need_mb=9000, launch_id=arm)


def record(spec, **fields):
    node.atomic(local.paths(spec)[1], dict(spec=spec, **fields))


def fake_launch(system, spec, monkeypatch):
    Path(spec['yaml']).write_text('fake')
    def spawn(*args, **kwargs):
        system.spawns.append(system.now)
        record(spec, child=dict(pid=999, starttime='new', cmdline='new'))
        return SimpleNamespace(pid=998, poll=lambda: None)
    monkeypatch.setattr(local.subprocess, 'Popen', spawn)


def test_stop_waits_past_empty_cmdline_for_child_and_port(tmp_path, monkeypatch, system):
    spec = spec_for(tmp_path, 'previous')
    child = system.add(101, clear_at=.2, exit_at=1)
    supervisor = system.add(102, exit_at=1.5)
    record(spec, child=child, supervisor=supervisor)
    system.busy = lambda: {23150: {101}} if system.now < 1 else {}
    # The listener is gone before the port can be rebound (e.g. TCP cleanup).
    system.bindable = lambda port: system.now >= 2
    original_sleep = system.sleep
    def sleep(seconds):
        assert not local.matching(spec).get('stopped')
        original_sleep(seconds)
    monkeypatch.setattr(local.time, 'sleep', sleep)
    assert local.stop(spec) == 'STOPPED local'
    assert 2 <= system.now < 2.2
    assert not local.process_present(child) and not local.process_present(supervisor)
    assert local.matching(spec)['stopped']
    assert [(pid, sig) for _, pid, sig in system.signals] == [(101, signal.SIGTERM), (102, signal.SIGTERM)]


@pytest.mark.parametrize('clear_at', [0, .2])
@pytest.mark.parametrize('stopped', [False, True])
def test_start_waits_for_authenticated_previous_arm(tmp_path, monkeypatch, system, clear_at, stopped):
    previous, spec = spec_for(tmp_path, 'previous'), spec_for(tmp_path)
    child = system.add(101, clear_at=clear_at, exit_at=1)
    supervisor = system.add(102, exit_at=1.5)
    record(previous, child=child, supervisor=supervisor, stopped=stopped)
    system.busy = lambda: {23150: {101}} if system.now < 1 else {}
    system.bindable = lambda port: system.now >= 2
    fake_launch(system, spec, monkeypatch)
    assert local.start(spec).startswith('STARTED local')
    assert 2 <= system.spawns[0] < 2.2
    assert not system.signals
    assert local.matching(previous)['stopped'] == stopped


def test_start_waits_for_stopped_launch_in_same_output(tmp_path, monkeypatch, system):
    spec = spec_for(tmp_path)
    previous = {**spec, 'launch_id': 'previous'}
    record(previous, child=system.add(101, exit_at=1), stopped=True)
    system.busy = lambda: {23150: {101}} if system.now < 1 else {}
    fake_launch(system, spec, monkeypatch)
    assert local.start(spec).startswith('STARTED local')
    assert system.spawns[0] >= 1
    assert not system.signals


@pytest.mark.parametrize('kind', ['foreign', 'unknown', 'mixed', 'other_root', 'recycled', 'misplaced', 'wrong_port'])
def test_start_refuses_foreign_listener_despite_previous_receipt(tmp_path, monkeypatch, system, kind):
    previous, spec = spec_for(tmp_path, 'previous'), spec_for(tmp_path)
    child = system.add(101)
    if kind == 'other_root':
        previous['run_root'] = str(tmp_path / 'other_root')
    if kind == 'wrong_port':
        previous['port'] = 23151
    receipt = dict(spec=previous, child=child, stopped=True)
    # Write manually so an invalid/cross-root receipt cannot grant ownership.
    path = Path(previous['out']) / f"server_{previous['tag']}.owner.json"
    if kind == 'misplaced':
        path = tmp_path / 'runs/copied/server/server_copy.owner.json'
    node.atomic(path, receipt)
    if kind == 'recycled':
        system.processes[101]['identity'] = 'different_birth'
    pids = {'foreign': {404}, 'unknown': set(), 'mixed': {101, 404}}.get(kind, {101})
    system.busy = lambda: {23150: pids}
    with pytest.raises(RuntimeError, match='LOCAL_PORT_BUSY'):
        local.start(spec)
    assert system.now == 0 and not system.spawns and not system.signals


def test_start_refuses_changed_command(tmp_path, system):
    previous, spec = spec_for(tmp_path, 'previous'), spec_for(tmp_path)
    record(previous, child=system.add(101), stopped=True)
    system.processes[101]['command'] = b'foreign\0'
    system.busy = lambda: {23150: {101}}
    with pytest.raises(RuntimeError, match='command differs'):
        local.start(spec)
    assert system.now == 0 and not system.spawns and not system.signals


def test_start_refuses_foreign_process_taking_port_during_wait(tmp_path, system):
    previous, spec = spec_for(tmp_path, 'previous'), spec_for(tmp_path)
    record(previous, child=system.add(101, exit_at=1), stopped=True)
    system.busy = lambda: {23150: {101 if system.now < .5 else 404}}
    with pytest.raises(RuntimeError, match='LOCAL_PORT_BUSY'):
        local.start(spec)
    assert .5 <= system.now < .7 and not system.spawns and not system.signals


@pytest.mark.parametrize('held', ['child', 'supervisor', 'bind'])
def test_start_release_wait_is_bounded(tmp_path, system, held):
    previous, spec = spec_for(tmp_path, 'previous'), spec_for(tmp_path)
    fields = {held: system.add(101)} if held != 'bind' else {'child': system.add(101, exit_at=0)}
    record(previous, **fields, stopped=True)
    if held == 'child':
        system.busy = lambda: {23150: {101}}
    if held == 'bind':
        system.bindable = lambda port: False
    with pytest.raises(RuntimeError, match='LOCAL_PORT_RELEASE_TIMEOUT'):
        local.start(spec)
    assert system.now == 60 and not system.spawns and not system.signals


@pytest.mark.parametrize('held', ['child', 'supervisor', 'bind'])
def test_stop_release_wait_is_bounded_and_does_not_mark_stopped(tmp_path, system, held):
    spec = spec_for(tmp_path)
    fields = {held: system.add(101)} if held != 'bind' else {'child': system.add(101, exit_at=0)}
    record(spec, **fields)
    if held == 'child':
        system.busy = lambda: {23150: {101}}
    if held == 'bind':
        system.bindable = lambda port: False
    with pytest.raises(RuntimeError, match='LOCAL_PORT_RELEASE_TIMEOUT'):
        local.stop(spec, timeout_s=90)
    assert system.now == 60 and not local.matching(spec).get('stopped')
    if held != 'bind':
        assert [sig for _, _, sig in system.signals] == [signal.SIGTERM, signal.SIGKILL]


def test_stop_does_not_signal_recycled_pid_or_foreign_listener(tmp_path, system):
    spec = spec_for(tmp_path)
    record(spec, child=system.add(101))
    system.processes[101]['identity'] = 'different_birth'
    system.busy = lambda: {23150: {101}}
    with pytest.raises(RuntimeError, match='LOCAL_PORT_BUSY'):
        local.stop(spec)
    assert not system.signals and not local.matching(spec).get('stopped')


def test_stop_tolerates_exit_between_authentication_and_signal(tmp_path, monkeypatch, system):
    spec = spec_for(tmp_path)
    record(spec, child=system.add(101))
    def vanished(pid, sig):
        system.processes[pid]['exit_at'] = 0
        raise ProcessLookupError
    monkeypatch.setattr(local.os, 'kill', vanished)
    assert local.stop(spec) == 'STOPPED local'
    assert system.now == 0


def test_duplicate_completed_launch_stays_finished(tmp_path, system):
    spec = spec_for(tmp_path)
    record(spec, child=system.add(101, exit_at=0), stopped=True)
    original = copy.deepcopy(local.matching(spec))
    assert local.start(spec) == 'ALREADY_FINISHED local'
    assert local.matching(spec) == original and not system.spawns
