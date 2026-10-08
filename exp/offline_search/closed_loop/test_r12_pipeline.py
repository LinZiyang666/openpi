"""Real emitted arms -> planner -> dry transport -> exact serving argv tests."""
from __future__ import annotations

import contextlib
import copy
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
from types import SimpleNamespace

import pytest

from .execution_worker import audited_run
from .lookhalf_config import METHODS, validate_execution, validate_fitted, value
from .ops.h100 import assets, control, local_server, node
from .plugin import parse_cli
from .stage_overrides import parse_flags

R12 = Path(__file__).resolve().parents[1] / 'rounds/r12'
_DIGESTS = {}
_SHA = assets.sha


@pytest.mark.parametrize('group,local', [('lookhalf', True), ('guard', True), ('lookhalf', False), ('guard', False)])
def test_emitted_plan_dry_sync_exact_argv(group, local, tmp_path, monkeypatch):
    """Exercise all 32 local arms and all seven method variants remotely too."""
    def digest(path):
        # Every real asset is hashed once; repeated plans reuse the digest only
        # while inode, byte length, mtime and ctime are unchanged.
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in _DIGESTS:
            _DIGESTS[key] = _SHA(path)
        return _DIGESTS[key]
    monkeypatch.setattr(assets, 'sha', digest)
    records = json.loads((R12 / group / 'pilot_roots.json').read_text())
    if not local:
        records = records[:1]
    monkeypatch.setenv('SERVER_HOST', 'local' if local else 'h100')
    monkeypatch.setenv('WORKER_HOST', 'timan107')
    for flag in ('OSCL_MANIFEST', 'OSCL_INIT_POOL', 'OSCL_TASKS', 'OSCL_EPISODES'):
        monkeypatch.delenv(flag, raising=False)
    bundles = []
    monkeypatch.setattr(control, 'bundle', lambda host, dest, pairs: bundles.append((host, dest, pairs)))

    @contextlib.contextmanager
    def daemon(*args):
        yield 'dry://no-network'

    monkeypatch.setattr(control, 'daemon', daemon)
    monkeypatch.setattr(control, 'rpc', lambda host, action, spec, **kw: 'DRY_TRANSFER')
    for record in records:
        root = Path(record['root'])
        emitted = json.loads((root / 'arms.json').read_text())
        before = (root / 'arms.json').read_bytes()
        work = tmp_path / root.name
        plan = control.sync(root, record['arms'], plan_only=True, workdir=work)
        # Execute the same sync implementation with ONLY its external transport
        # mocked. This writes receipts in test scratch, never a real pilot root.
        synced = control._sync(root, record['arms'], workdir=work)
        assert json.loads((work / 'synced.json').read_text()) == synced
        assert plan == synced
        assert before == (root / 'arms.json').read_bytes()
        for original, ready in zip(emitted, synced['arms'], strict=True):
            selected = original['kwargs']['method']
            mapped = assets.local_remap if local else assets.remap
            expected_args = mapped(copy.deepcopy(original['plugin_args']), root)
            if local:
                fitted = value(expected_args, '--os-fit-artifact')
                source = next(f['source'] for f in synced['files'] if f['original'] == fitted)
                expected_args = [source if v == fitted else v for v in expected_args]
            contract = next(f for f in synced['files'] if original['arm'] + ': B recording contract' in f['reasons'])
            expected_contract = contract['source'] if local else str(assets.BASE / contract['rel'])
            expected_args += ['--os-look-contract', expected_contract, '--os-look-contract-sha256', contract['sha256']]
            assert ready['plugin_args'] == expected_args
            spec = control.server_spec(root, ready, 23160 if local else 23260)
            expected_plugin = ['--os-method', original['method'], '--os-cell', original['cell'],
                               *expected_args, '--os-kwargs', json.dumps(mapped(original['kwargs'], root))]
            assert spec['plugin'] == expected_plugin
            if local:
                # command() creates a runtime directory even without a launch.
                spec['run_root'] = str(work / 'argv')
                spec['out'] = str(work / 'argv/runs' / original['arm'] / 'server_23160')
            argv, env = local_server.command(spec) if local else node.command(
                spec['model'], spec['suite'], spec['port'], spec['yaml'], spec['out'], spec['tag'],
                spec['plugin'], dict(spec['env']))
            offset = 5 if local else 2
            stock = (['--port', str(spec['port']), '--replicas', '1', '--cache-config', spec['yaml'],
                      'policy:checkpoint', '--policy.config', 'pi05_libero', '--policy.dir',
                      local_server.PI05_CKPT if local else '/home/exouser/ckpt/pi05_libero_pytorch']
                     if original['model'] == 'pi05' else
                     ['--checkpoint', '/data/ckpt/n15_' + original['suite'] if local else
                      '/home/exouser/ckpt/n15_' + original['suite'], '--port', str(spec['port']),
                      '--denoising-steps', '8', '--concurrent', '--allow-dynamic-bundles', '--cache-config', spec['yaml']])
            assert argv[offset:] == [*expected_plugin, '--os-log-dir', spec['out'], '--os-tag', spec['tag'], *stock]
            _, remaining = parse_flags(argv[offset:])
            opts, rest = parse_cli(remaining)
            assert rest == stock
            assert opts.os_lookhalf == selected
            assert opts.os_request_cameras == (selected not in ('off', 'h2'))
            assert opts.os_record_looks and opts.os_record_look_images == 56
            assert opts.os_look_contract == expected_contract
            assert env['STAGE1_ONLY'] == '0'
    assert bundles and all(host == 'timan107' for host, _, _ in bundles)


@pytest.mark.parametrize('spec,choices', METHODS.items())
def test_bad_selection_fails_before_model_loading(spec, choices):
    base = ['--os-method', spec, '--os-kwargs', json.dumps({'method': choices[1]}),
            '--os-cell', 'pi05_l10_cache', '--os-log-dir', '/unused', '--os-blind', '--os-judge', 'guard_only']
    for bad in ([], ['--os-lookhalf'], ['--os-lookhalf', '--os-look-contract', '/path'],
                ['--os-lookhalf', '/path/look_contract.json'], ['--os-lookhalf='],
                ['--os-lookhalf', 'typo'], ['--os-lookhalf', 'off'],
                ['--os-lookhalf', choices[1], '--os-lookhalf=' + choices[1]]):
        with pytest.raises(SystemExit):
            parse_cli(base + bad)
        with pytest.raises(ValueError):
            node.command('pi05', 'libero_10', 23160, 'x', '/unused', 'bad', base + bad, {})
    good = base + ['--os-lookhalf=' + choices[1], '--os-no-shadow-native', '--os-tokens', 'off']
    assert parse_cli(good)[0].os_lookhalf == choices[1]


def test_actual_execution_receipts_and_default_unchanged():
    sent = []

    class Environment:
        def step(self, action):
            if action == 'error':
                raise RuntimeError('simulator failed')
            return action

    class Client:
        def infer(self, observation):
            sent.append(observation)
            return {'actions': [1] * 5}

    def rollout(env, client, *args, **kwargs):
        for _ in range(10):
            env.step('settle')
        client.infer({'same': 1})
        for _ in range(5):
            env.step('control')
        client.infer({'same': 1})
        for _ in range(2):
            env.step('partial')
        try:
            env.step('error')
        except RuntimeError:
            pass
        client.infer({'same': 1})
        return 'unchanged'

    arguments = (Environment(), Client(), None, 'task', SimpleNamespace(replan_steps=5), 20)
    assert audited_run(rollout, *arguments) == 'unchanged'
    assert [x['__extra__'] for x in sent] == [
        dict(executed_steps=0, exec_ok=True, decision_id=0),
        dict(executed_steps=5, exec_ok=True, decision_id=1),
        dict(executed_steps=2, exec_ok=False, decision_id=2)]
    for step, observation in enumerate(sent):
        validate_execution(observation['__extra__'], step)
    sent.clear()
    assert rollout(*arguments) == 'unchanged'
    assert sent == [{'same': 1}] * 3
    for bad in ({}, dict(executed_steps=True, exec_ok=True, decision_id=0),
                dict(executed_steps=5, exec_ok=True, decision_id=2)):
        with pytest.raises(ValueError, match='R12_EXECUTION_AUDIT_REQUIRED'):
            validate_execution(bad, 0)


@pytest.mark.parametrize('pool', ['A', 'B'])
def test_worker_module_selected_only_for_attested_opt_in(pool, monkeypatch):
    from . import worker_pool, devset
    monkeypatch.setattr(worker_pool, 'validate', lambda *a: (pool, {'dev': True} if pool == 'B' else None))
    monkeypatch.setattr(worker_pool, 'install_journal', lambda pool: None)
    monkeypatch.setattr(devset, 'pool_record', lambda *a: (Path('/unused'), {'suite': 'libero_spatial', 'rollup_sha256': 'test'}))
    monkeypatch.delenv('OSCL_RECORD_LOOKS', raising=False)
    calls = []

    def spec(**kwargs):
        calls.append(kwargs)
        return kwargs

    driver = SimpleNamespace(WorkerSpec=spec, load_apool_digest=lambda *a, **k: None)
    monkeypatch.delenv('OSCL_EXECUTION_AUDIT', raising=False)
    worker_pool.install(driver, ['--task-suite', 'libero_spatial'], None)
    assert driver.WorkerSpec is spec
    monkeypatch.setenv('OSCL_EXECUTION_AUDIT', '1')
    worker_pool.install(driver, ['--task-suite', 'libero_spatial'], None)
    assert driver.WorkerSpec(worker_id='worker')['worker_module'] == 'exp.offline_search.closed_loop.execution_worker'
    def refuse(*args):
        raise ValueError('POOL_MISMATCH')
    monkeypatch.setattr(worker_pool, 'validate', refuse)
    with pytest.raises(ValueError, match='POOL_MISMATCH'):
        worker_pool.install(driver, [], None)


def test_fitted_alias_and_method_mismatch_fail():
    for family in ('r12_lookhalf', 'r12_guard'):
        fitted = SimpleNamespace(family=family, lookhalf_method='off', guard_method='off')
        for selected in (None, 'm2', 'opus_m1', '/path'):
            with pytest.raises(ValueError, match='fitted method differs'):
                validate_fitted(fitted, selected)
        validate_fitted(fitted, 'off')
        delattr(fitted, 'guard_method' if family == 'r12_guard' else 'lookhalf_method')
        with pytest.raises(ValueError, match='fitted method differs'):
            validate_fitted(fitted, 'off')
    validate_fitted(SimpleNamespace(family='legacy'), None)


@pytest.mark.parametrize('module,class_name,selected', [
    ('guard', 'R12Guard', 'h2'), ('lookhalf', 'R12LookHalf', 'opus_m1')])
def test_aliased_r12_classes_require_matching_explicit_selection(module, class_name, selected, monkeypatch):
    """Family validation must also apply to classes loaded through a module alias."""
    monkeypatch.syspath_prepend(str(R12.parents[1]))
    cls = getattr(importlib.import_module(f'closed_loop.{module}'), class_name)
    canonical = getattr(importlib.import_module(f'exp.offline_search.closed_loop.{module}'), class_name)
    assert cls is not canonical
    fitted = cls(method=selected)
    validate_fitted(fitted, selected)
    for bad in (None, 'off', '/path'):
        with pytest.raises(ValueError, match='fitted method differs'):
            validate_fitted(fitted, bad)


@pytest.mark.parametrize('artifact', sorted(
    [path for group in ('guard', 'lookhalf') for path in (R12 / group / 'artifacts').glob('*/*.pkl')]),
    ids=lambda path: str(path.relative_to(R12)))
def test_frozen_r12_pickles_match_fix1_arm_selection(artifact):
    """Validate every frozen fit with the explicit selection in the fix1 arms."""
    group = artifact.relative_to(R12).parts[0]
    arms_files = sorted((R12 / group / 'pilots').glob(f'r12_{group}_fix1_*/arms.json'))
    assert arms_files, f'no fix1 arms for {group}'
    rows = [row for path in arms_files for row in json.loads(path.read_text())]
    references = [row for row in rows
                  if Path(value(row['plugin_args'], '--os-fit-artifact')).resolve() == artifact.resolve()]
    if not references:
        # The extra guard Spatial-200 fits have no emitted pilot arms. Validate
        # them against the corresponding explicit selections in the fix1 arms.
        assert group == 'guard' and artifact.parent.name == 'pi05_spatial_200'
        references = [row for row in rows if row['kwargs']['method'] == artifact.stem]
    selections = {value(row['plugin_args'], '--os-lookhalf') for row in references}
    assert len(selections) == 1
    selected = selections.pop()
    assert selected is not None
    with artifact.open('rb') as stream:
        blob = pickle.load(stream)
    assert blob['kwargs']['method'] == selected
    assert all(row['method'] == blob['spec'] and row['kwargs']['method'] == selected for row in references)
    validate_fitted(blob['method'], selected)


def test_frozen_h2_pickle_refuses_m2_selection():
    """A real guard fit must reject another explicit guard selection."""
    with (R12 / 'guard/artifacts/pi05_l10_50/h2.pkl').open('rb') as stream:
        fitted = pickle.load(stream)['method']
    assert fitted.guard_method == 'h2' and fitted.lookhalf_method == 'off'
    for selected in ('m2', 'off', None):
        with pytest.raises(ValueError, match='fitted method differs'):
            validate_fitted(fitted, selected)


@pytest.mark.parametrize('group,start', [('lookhalf', 23110), ('guard', 23120)])
def test_coordinator_count_ports_and_wps_environment(group, start, tmp_path):
    original_script = R12 / group / 'coordinator_commands.sh'
    base = R12 / group / 'pilots'
    scratch_base = tmp_path / 'pilots'
    for record in json.loads((R12 / group / 'pilot_roots.json').read_text()):
        source = Path(record['root'])
        target = scratch_base / source.name
        target.mkdir(parents=True)
        (target / 'arms.json').write_bytes((source / 'arms.json').read_bytes())
    script = tmp_path / 'coordinator_commands.sh'
    script.write_text(original_script.read_text().replace(f'BASE={base}', f'BASE={scratch_base}'))
    clean = {k: v for k, v in os.environ.items() if k not in ('PORTS', 'LOCAL_SERVER_COUNT', 'WPS')}
    # Only reports run; shell trace exposes the effective exported environment.
    result = subprocess.run(['bash', '-x', str(script), 'report'], env=clean, capture_output=True, text=True, check=True)
    assert '+ LOCAL_SERVER_COUNT=4' in result.stderr
    assert '+ WPS=16' in result.stderr
    assert '+ PORTS=' + ','.join(str(start + i) for i in range(4)) in result.stderr
    custom = {**clean, 'LOCAL_SERVER_COUNT': '2', 'PORTS': '23150,23151', 'WPS': '7'}
    result = subprocess.run(['bash', '-x', str(script), 'report'], env=custom, capture_output=True, text=True, check=True)
    assert '+ LOCAL_SERVER_COUNT=2' in result.stderr
    # Inherited PORTS needs no assignment; the derivation branch sees it and
    # stays closed, so the preflight and reports receive the supplied list.
    assert '+ [[ -z 23150,23151 ]]' in result.stderr
    assert '+ WPS=7' in result.stderr
    result = subprocess.run(['bash', str(script), 'report'], env={**custom, 'PORTS': '23150'}, capture_output=True, text=True)
    assert result.returncode and 'exactly LOCAL_SERVER_COUNT' in result.stderr
