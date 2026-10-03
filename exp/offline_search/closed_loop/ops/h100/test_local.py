"""Local topology tests: fake servers, real flock, no GPU or fleet mutations."""
import contextlib
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

import pytest

from exp.offline_search.closed_loop.ops.h100 import assets, control, local_server as local, node
from exp.offline_search.closed_loop.ops.h100.server_host import endpoint, server_host
from exp.offline_search.closed_loop.ops.h100.test_h100 import fixture_run
from exp.offline_search.closed_loop.ops.h100.test_robustness import mock_chain


@pytest.fixture(autouse=True)
def topology(monkeypatch):
    monkeypatch.delenv('SERVER_HOST', raising=False)
    monkeypatch.delenv('LOCAL_SERVER_CPUS', raising=False)
    monkeypatch.setattr(control, 'remote', lambda *a, **kw: pytest.fail('unmocked remote operation'))
    monkeypatch.setattr(control, 'push', lambda *a, **kw: pytest.fail('unmocked remote push'))


def test_selector_default_and_public_endpoints(monkeypatch):
    assert server_host() == 'h100'
    assert endpoint(23230) == '149.165.153.233:23230'
    monkeypatch.setenv('SERVER_HOST', 'local')
    assert endpoint(23150) == 'ziyanglin.com:23150'
    monkeypatch.setenv('SERVER_HOST', 'unknown')
    with pytest.raises(ValueError, match='SERVER_HOST'):
        control.ports_workers()


def test_free_block_accounts_for_listeners_and_replica_ports(monkeypatch):
    monkeypatch.setattr(local, 'can_bind', lambda p: True)
    monkeypatch.setattr(local, 'listeners', lambda: {23100: {101}, 23103: {102}})
    assert local.choose_ports(4) == [23104, 23105, 23106, 23107]
    assert local.choose_ports(2, replicas=2) == [23104, 23107]
    assert local.port_block(23150, 1) == [23150]
    assert local.port_block(23150, 3) == [23150, 23151, 23152, 23153]
    with pytest.raises(RuntimeError, match='LOCAL_PORT_BUSY'):
        local.choose_ports(requested=[23103])
    for ports in ([23200], [23099], [23150, 23150], [23199]):
        with pytest.raises(ValueError, match='23100..23199'):
            local.choose_ports(requested=ports, replicas=2)
    with pytest.raises(ValueError, match='nonoverlapping'):
        local.choose_ports(requested=[23150, 23151], replicas=2)


def test_all_ports_busy_and_timan107_capacity(monkeypatch):
    monkeypatch.setattr(local, 'can_bind', lambda p: True)
    monkeypatch.setattr(local, 'listeners', lambda: {p: set() for p in range(23100, 23200)})
    with pytest.raises(RuntimeError, match='no free block'):
        local.choose_ports()
    monkeypatch.setattr(local, 'listeners', lambda: {})
    monkeypatch.setenv('SERVER_HOST', 'local')
    monkeypatch.setenv('WORKER_HOST', 'timan107')
    monkeypatch.delenv('PORTS', raising=False)
    monkeypatch.setenv('WPS', '16')
    assert control.ports_workers() == ([23100, 23101, 23102, 23103], 16)
    monkeypatch.setenv('PORTS', '23150,23151,23152,23153')
    assert control.ports_workers()[0] == [23150, 23151, 23152, 23153]
    monkeypatch.setenv('WPS', '17')
    with pytest.raises(ValueError, match='1..64'):
        control.ports_workers()


def test_ss_parser_includes_ipv6_and_foreign_pids(monkeypatch):
    output = ('LISTEN 0 128 0.0.0.0:23150 0.0.0.0:* users:(("python",pid=123,fd=9))\n'
              'LISTEN 0 128 [::]:23151 [::]:*\n')
    monkeypatch.setattr(node, 'call', lambda argv: output)
    assert local.listeners() == {23150: {123}, 23151: set()}


def test_selection_skips_bind_race_and_time_wait(monkeypatch):
    monkeypatch.setattr(local, 'listeners', lambda: {})
    monkeypatch.setattr(local, 'can_bind', lambda p: p != 23100)
    assert local.choose_ports(2) == [23101, 23102]
    with pytest.raises(RuntimeError, match='LOCAL_PORT_BUSY'):
        local.choose_ports(requested=[23100])


@pytest.mark.parametrize('cpus', ['38', '82-87', '21', '22-43', '', 'foo', '37-22'])
def test_local_affinity_refuses_reserved_and_invalid_cpus(monkeypatch, cpus):
    monkeypatch.setenv('LOCAL_SERVER_CPUS', cpus)
    with pytest.raises(ValueError, match='LOCAL_SERVER_CPUS'):
        local.cpu_affinity()


def spec_for(root, port=23150):
    return dict(role='server', server_host='local', run_root=str(root), out=str(root / 'runs/arm/server'),
                tag='arm_23150', port=port, model='pi05', suite='libero_10', yaml=str(root / 'cfg.yaml'),
                plugin=['--os-method', 'native'], env=dict(STAGE1_ONLY='0', STOCK='0'), need_mb=9000,
                launch_id='owned')


@pytest.mark.parametrize('model,stock,stage1', [('pi05', False, False), ('pi05', True, True),
                                              ('groot', False, False), ('groot', True, True)])
def test_local_command_preserves_serving_flags_and_maps_islands(tmp_path, model, stock, stage1):
    spec = spec_for(tmp_path)
    spec['model'] = model
    spec['env'].update(STOCK='1' if stock else '0', STAGE1_ONLY='1' if stage1 else '0',
                       GROOT_DENOISING_STEPS='7', GPU_LOCK='1', OMP='2')
    spec['plugin'] = [] if stock else ['--os-method', 'method:M', '--os-fit-artifact', '/local/fit.pkl']
    cmd, env = local.command(spec)
    assert cmd[:3] == ['taskset', '-c', local.CPUS]
    assert cmd[3] == (str(local.TREE / '.venv/bin/python') if model == 'pi05' else local.GROOT_PY)
    assert str(local.TREE) in cmd[4]
    assert '/home/exouser/' not in json.dumps([cmd, env])
    assert env['OPENPI_SERVER_GPU_MEMORY_LOCK'] == '1'
    assert env['OMP_NUM_THREADS'] == env['MKL_NUM_THREADS'] == '2'
    assert env['OPENBLAS_NUM_THREADS'] == '1'
    assert env['TMPDIR'].startswith(str(tmp_path))
    if model == 'pi05':
        assert cmd[cmd.index('--policy.dir') + 1] == local.PI05_CKPT
        assert cmd[cmd.index('--replicas') + 1] == '1'
    else:
        assert cmd[cmd.index('--checkpoint') + 1] == '/data/ckpt/n15_libero_10'
        assert cmd[cmd.index('--denoising-steps') + 1] == '7'
    assert ('--os-fit-artifact' in cmd) == (not stock)


def test_default_plan_bytes_match_frozen_prelocal_builder(tmp_path, monkeypatch, capsys):
    root, store, base, row = fixture_run(tmp_path)
    old = types.ModuleType('prelocal_assets')
    old.__package__ = assets.__package__
    baseline = Path(assets.__file__).with_name('fixtures') / 'assets_prelocal.txt'
    exec(compile(baseline.read_text(), str(baseline), 'exec'), old.__dict__)
    outcomes = []
    for module in (old, assets):
        remapper = module.remap
        def mapper(v, r, s=store, b=base, original=remapper):
            if isinstance(v, str) and v == str(tmp_path / 'native.pkl'):
                return str(base / 'native.pkl')
            return original(v, r, s, b)
        monkeypatch.setattr(module, 'remap', mapper)
        plan = module.build_plan(root, ['sample'], root / 'plan', store=store, base=base)
        outcomes.append((json.dumps(plan).encode(), (root / 'plan/plan.json').read_bytes(), capsys.readouterr().out,
                         {str(p): p.read_bytes() for p in (root / 'plan/generated').rglob('*') if p.is_file()}))
    assert outcomes[0] == outcomes[1]
    assert 'server_host' not in plan


def test_historical_h100_path_and_pickle_walkers_are_unchanged():
    import inspect
    source = (Path(assets.__file__).with_name('fixtures') / 'assets_prelocal.txt').read_text()
    for function, following in ((assets.remap, 'strings'), (assets.relocated_fit, 'build_plan')):
        start = source.index('def ' + function.__name__ + '(')
        end = source.index('def ' + following + '(', start)
        assert inspect.getsource(function).strip() == source[start:end].strip()


def test_default_server_spec_byte_identity(tmp_path, monkeypatch):
    row = dict(arm='arm', model='pi05', suite='libero_10', mode='stock', yaml='cfg.yaml')
    default = control.server_spec(tmp_path, row, 23240)
    expected = dict(role='server', out=str(control.BASE / 'runs' / tmp_path.name / 'runs/arm/server_23240'),
                    tag='arm_23240', model='pi05', suite='libero_10', port=23240, yaml='cfg.yaml', plugin=[],
                    env={'STAGE1_ONLY': '1', 'STOCK': '1'}, need_mb=3000)
    assert json.dumps(default).encode() == json.dumps(expected).encode()
    monkeypatch.setenv('SERVER_HOST', 'h100')
    assert json.dumps(default).encode() == json.dumps(control.server_spec(tmp_path, row, 23240)).encode()


def test_local_plan_keeps_original_prefit_and_payloads(tmp_path):
    root, store, _, row = fixture_run(tmp_path)
    plan = assets.build_plan(root, ['sample'], root / 'plan', store=store, local=True)
    assert plan['server_host'] == 'local' and plan['server_transfer_bytes'] == 0
    assert plan['remote_run'] == str(root)
    ready = plan['arms'][0]
    assert ready['kwargs'] == row['kwargs']
    assert assets.flag(ready['plugin_args'], '--os-fit-artifact') == str(root / 'fits/sample.pkl')
    fit = next(x for x in plan['files'] if x['original'].endswith('sample.pkl'))
    assert fit['source_sha256'] == fit['sha256']
    assert '/data/oscl_h100' not in json.dumps(plan)
    import yaml
    config = yaml.safe_load(Path(ready['yaml']).read_text())
    assert config['backend']['in_memory']['preload_path'] == str(tmp_path / 'native.pkl')


def test_local_sync_concurrent_with_h100_never_transfers_to_h100(tmp_path, monkeypatch):
    root, store, _, row = fixture_run(tmp_path)
    monkeypatch.setattr(control, 'CHAIN_LOCK', tmp_path / 'h100_chain.lock')
    monkeypatch.setattr(control, 'build_plan', lambda r, n, w, **kw: assets.build_plan(r, n, w, store=store, **kw))
    monkeypatch.setattr(control, 'daemon', lambda *a: pytest.fail('local sync must not export fit files'))
    monkeypatch.setattr(control, 'rpc', lambda *a, **kw: pytest.fail('local sync must not use server RPCs'))
    bundles = []
    monkeypatch.setattr(control, 'bundle', lambda h, d, p: bundles.append((h, d, p)))
    monkeypatch.setenv('WORKER_HOST', 'timan108')
    monkeypatch.setenv('SERVER_HOST', 'h100')
    with control.fleet_lock(tmp_path / 'h100_chain'):
        monkeypatch.setattr(control, 'legacy_chains', lambda **kw: [dict(root=tmp_path / 'h100_chain', worker_host='timan108')])
        monkeypatch.setenv('WORKER_HOST', 'timan107')
        monkeypatch.setenv('SERVER_HOST', 'local')
        plan = control.sync(root, ['sample'])
        assert plan['worker_island'] == str(control.island('timan107'))
        monkeypatch.setattr(control, '_setup', lambda **kw: kw)
        # Worker setup is scoped too; a held local fleet lock must exclude it.
        with control.fleet_lock(root):
            with pytest.raises(RuntimeError, match='H100_CHAIN_BUSY'):
                control.setup()
    assert [b[0] for b in bundles] == ['timan107']
    assert json.loads((root / 'h100_sync/synced.json').read_text())['server_host'] == 'local'


@pytest.mark.parametrize('failure', [False, True])
def test_local_chain_runs_under_h100_lock_uses_local_gpu_and_cleans_up(tmp_path, monkeypatch, failure):
    root, events, sleeps = mock_chain(tmp_path, monkeypatch)
    # Hold the other fleet throughout local preflight, launch, polling and teardown.
    monkeypatch.setenv('WORKER_HOST', 'timan108')
    with control.fleet_lock(tmp_path / 'other_chain'):
        monkeypatch.setattr(control, 'legacy_chains', lambda **kw: [dict(root=tmp_path / 'other_chain', worker_host='timan108')])
        path = root / 'h100_sync/synced.json'
        plan = json.loads(path.read_text())
        plan.update(server_host='local', worker_host='timan107')
        path.write_text(json.dumps(plan))
        monkeypatch.setenv('SERVER_HOST', 'local')
        monkeypatch.setenv('WORKER_HOST', 'timan107')
        monkeypatch.setenv('PORTS', '23150')
        monkeypatch.setattr(local, 'choose_ports', lambda count, requested=None: requested)
        previous_rpc, previous_remote = control.rpc, control.remote
        calls = []
        def rpc(host, action, spec, **kw):
            calls.append((host, action, copy.deepcopy(spec)))
            assert host in ('local', 'timan107')
            if failure and action == 'status' and spec['role'] == 'server':
                return json.dumps(dict(running=False, listening=False, dead=True))
            return previous_rpc(host, action, spec, **kw)
        def remote(host, cmd, **kw):
            assert host in ('local', 'timan107')
            return '49000' if host == 'local' else previous_remote(host, cmd, **kw)
        monkeypatch.setattr(control, 'rpc', rpc)
        monkeypatch.setattr(control, 'remote', remote)
        if failure:
            with pytest.raises(RuntimeError, match='SERVER_DIED_AT_BOOT'):
                control.chain(root, ['first'])
        else:
            control.chain(root, ['first'])
            assert (root / 'state/CHAIN.DONE').exists()
        assert any(h == 'local' and a == 'stop' for h, a, _ in calls)
        saved = json.loads((root / 'runs/first/h100_launch.json').read_text())
        assert saved['server_host'] == 'local'
        assert saved['driver']['servers'] == ['ziyanglin.com:23150']
        assert saved['servers'][0]['out'].startswith(str(root / 'runs'))


def test_abort_uses_saved_server_host_regardless_of_environment(tmp_path, monkeypatch):
    path = tmp_path / 'runs/arm/h100_launch.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(server_host='local', worker_host='timan107', driver={}, servers=[{}])))
    calls = []
    monkeypatch.setattr(control, 'rpc', lambda host, action, spec, **kw: calls.append((host, action)) or 'OK')
    control.abort(tmp_path, ['arm'])
    assert calls == [('timan107', 'stop'), ('local', 'stop')]
    path.write_text(json.dumps(dict(driver={}, servers=[{}])))
    calls.clear()
    monkeypatch.setenv('SERVER_HOST', 'local')
    control.abort(tmp_path, ['arm'])
    assert calls == [('timan108', 'stop'), ('h100', 'stop')]


def test_detached_fake_process_lifecycle_idempotency_and_foreign_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(local, 'LOCKS', tmp_path / 'locks')
    monkeypatch.setattr(local, 'gpu_free', lambda: 49000)
    port = local.choose_ports(1)[0]
    spec = spec_for(tmp_path, port)
    (tmp_path / 'cfg.yaml').write_text('fake')
    # A real detached stdlib-only fake listener exercises supervisor/PID receipts.
    argv = [sys.executable, '-B', '-c',
            f'import socket,time; s=socket.socket(); s.bind(("0.0.0.0",{port})); s.listen(); time.sleep(60)']
    monkeypatch.setattr(local, 'command', lambda s: (argv, dict(PYTHONDONTWRITEBYTECODE='1')))
    try:
        assert local.start(spec).startswith('STARTED local')
        deadline = time.monotonic() + 10
        while not local.status(spec)['listening'] and time.monotonic() < deadline:
            time.sleep(.05)
        assert local.status(spec)['running'] and local.status(spec)['listening']
        assert local.start(spec) == 'ALREADY_STARTED local'
        wrong = {**spec, 'launch_id': 'foreign'}
        for action in (local.start, local.stop, local.status):
            with pytest.raises(RuntimeError, match='receipt|occupied local launch'):
                action(wrong)
        receipt = local.matching(spec)
        assert os.getsid(receipt['supervisor']['pid']) == receipt['supervisor']['pid']
        assert os.getsid(receipt['child']['pid']) == receipt['child']['pid']
    finally:
        local.stop(spec, timeout_s=5)
    assert local.stop(spec) == 'STOPPED local'
    assert local.status(spec)['dead'] and not local.status(spec)['running']
    assert not local.status(spec)['listening']
    assert local.start(spec) == 'ALREADY_FINISHED local'
    assert 'SERVER_EXIT=' in local.paths(spec)[2].read_text()


def test_stop_refuses_recycled_pid_and_command_mismatch(tmp_path, monkeypatch):
    spec = spec_for(tmp_path)
    monkeypatch.setattr(local, 'LOCKS', tmp_path / 'locks')
    _, rp, _ = local.paths(spec)
    process = dict(pid=99999999, starttime='original', cmdline=b'owned\0'.hex())
    node.atomic(rp, dict(spec=spec, child=process))
    monkeypatch.setattr(local, 'proc_identity', lambda pid: 'recycled')
    monkeypatch.setattr(local.os, 'kill', lambda *a: pytest.fail('recycled PID must not be signalled'))
    assert local.stop(spec) == 'STOPPED local'
    monkeypatch.setattr(local, 'proc_identity', lambda pid: 'original')
    read = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda p: b'foreign\0' if str(p).startswith('/proc/') else read(p))
    with pytest.raises(RuntimeError, match='command differs'):
        local.stop(spec)


def test_health_does_not_count_foreign_listener(tmp_path, monkeypatch):
    spec = spec_for(tmp_path)
    monkeypatch.setattr(local, 'matching', lambda s: dict(spec=s, child=dict(pid=123)))
    monkeypatch.setattr(local, 'authenticated', lambda p: bool(p))
    monkeypatch.setattr(local, 'listeners', lambda: {23150: {456}})
    assert local.status(spec)['running'] and not local.status(spec)['listening']


def test_local_rpc_never_uses_tether(monkeypatch):
    monkeypatch.setattr(local, 'rpc', lambda action, spec: 'LOCAL')
    monkeypatch.setattr(control, 'remote', lambda *a, **kw: pytest.fail('local RPC must not contact h100'))
    assert control.rpc('local', 'status', {}) == 'LOCAL'


def test_local_lock_holder_authenticates_server_selector(tmp_path, monkeypatch):
    lock = tmp_path / 'h100_chain.lock'
    lock.touch()
    monkeypatch.setattr(control, 'CHAIN_LOCK', lock)
    root = tmp_path / 'local_root'
    (root / 'state').mkdir(parents=True)
    (root / 'state/h100_chain.lock').touch()
    pid = 99999999
    st = lock.stat()
    table = f'1: FLOCK ADVISORY WRITE {pid} {os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino} 0 EOF\n'
    argv = b'python\0-m\0exp.offline_search.closed_loop.ops.h100.control\0chain\0' + str(root).encode() + b'\0arm\0'
    receipt = root / 'state/h100_chain.owner.json'
    node.atomic(receipt, dict(pid=pid, starttime='time', cmdline=argv.hex(), server_host='local'))
    read_text, read_bytes = Path.read_text, Path.read_bytes
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **kw: table if str(p) == '/proc/locks' else read_text(p, *a, **kw))
    def proc_bytes(p, *a, **kw):
        if str(p) == f'/proc/{pid}/cmdline':
            return argv
        if str(p) == f'/proc/{pid}/environ':
            return b'WORKER_HOST=timan107\0SERVER_HOST=local\0'
        return read_bytes(p, *a, **kw)
    monkeypatch.setattr(Path, 'read_bytes', proc_bytes)
    monkeypatch.setattr(control, 'identity', lambda pid: 'time')
    assert control.legacy_chains() == [dict(root=root, worker_host='timan107', server_host='local')]
    saved = json.loads(receipt.read_text())
    saved.pop('server_host')
    node.atomic(receipt, saved)
    with pytest.raises(RuntimeError, match='unauthenticated server host'):
        control.legacy_chains()


def test_h100_sync_does_not_protect_local_chain_asset_namespace(tmp_path, monkeypatch):
    root, store, base, row = fixture_run(tmp_path)
    file = dict(rel='cfg.yaml', sha256='a' * 64, size=1, original=row['yaml'], source=row['yaml'])
    monkeypatch.setattr(control, 'build_plan', lambda *a: dict(arms=[row], files=[file]))
    @contextlib.contextmanager
    def daemon(*a):
        yield 'rsync://mock'
    monkeypatch.setattr(control, 'daemon', daemon)
    calls = []
    monkeypatch.setattr(control, 'rpc', lambda host, action, spec, **kw: calls.append((host, action, spec)) or 'OK')
    monkeypatch.setattr(control, 'bundle', lambda *a: None)
    # No synced.json is needed for the local chain: it owns no h100 assets.
    control._sync(root, ['sample'], concurrent=True, active=[dict(root=tmp_path / 'local_chain', server_host='local')])
    assert calls[0][:2] == ('h100', 'pull-new')
    assert calls[0][2]['protected'] == {}


def test_wrong_synced_server_host_refused_before_launch(tmp_path, monkeypatch):
    root, events, _ = mock_chain(tmp_path, monkeypatch)
    monkeypatch.setenv('SERVER_HOST', 'local')
    monkeypatch.setattr(local, 'choose_ports', lambda count, requested=None: [23150])
    with pytest.raises(RuntimeError, match='synced server host differs'):
        control.chain(root, ['first'])
    assert events == []


def test_local_collect_keeps_logs_and_only_pulls_worker_archive(tmp_path, monkeypatch):
    import tarfile
    from exp.offline_search.closed_loop.ops import collect as collector
    path = tmp_path / 'runs/arm/h100_launch.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(server_host='local', worker_host='timan107')))
    log = path.parent / 'server_23150/decisions_arm.jsonl'
    log.parent.mkdir()
    log.write_text('server log\n')
    journal = tmp_path / 'journal'
    journal.write_text('worker log\n')
    archive = tmp_path / 'worker.tar'
    with tarfile.open(archive, 'w') as tf:
        tf.add(journal, arcname='client/journal.jsonl')
    contacts = []
    monkeypatch.setattr(control, 'rpc', lambda host, action, spec, **kw: contacts.append(host) or json.dumps(dict(sha256=node.sha(archive))))
    def pull(cmd, **kw):
        assert cmd[3].startswith('timan107:')
        Path(cmd[-1]).write_bytes(archive.read_bytes())
    monkeypatch.setattr(control, 'run', pull)
    monkeypatch.setattr(control, 'remote', lambda host, *a: contacts.append(host))
    monkeypatch.setattr(collector, 'summarize', lambda *a, **kw: dict(complete=1, success=1, sr=1.0))
    control.collect(tmp_path, 'arm')
    assert contacts == ['timan107', 'timan107']
    assert log.read_text() == 'server log\n'
    assert (path.parent / 'client/journal.jsonl').read_text() == 'worker log\n'


def test_local_start_refuses_foreign_listener_and_low_local_memory(tmp_path, monkeypatch):
    spec = spec_for(tmp_path)
    monkeypatch.setattr(local, 'LOCKS', tmp_path / 'locks')
    monkeypatch.setattr(local, 'listeners', lambda: {23150: {123}})
    monkeypatch.setattr(local, 'command', lambda s: pytest.fail('must refuse before spawning'))
    monkeypatch.setattr(local, 'gpu_free', lambda: pytest.fail('busy port must refuse before GPU check'))
    with pytest.raises(RuntimeError, match='LOCAL_PORT_BUSY'):
        local.start(spec)
    monkeypatch.setattr(local, 'listeners', lambda: {})
    monkeypatch.setattr(local, 'gpu_free', lambda: 1)
    with pytest.raises(RuntimeError, match='GPU_TIGHT server_host=local'):
        local.start(spec)
