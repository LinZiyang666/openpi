"""CPU-only deployed-off replica checks with every external transport mocked."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp.offline_search.rounds.r12 import prepare_test_a_rep as rep

from . import plugin, worker_pool
from .lookhalf_config import value
from .ops.h100 import assets, control, local_server, node
from .ops.remote.run_gtp_subset import load_manifest
from .stage_overrides import parse_flags

_DIGESTS = {}
_SHA = assets.sha


def _read(path):
    return json.loads(path.read_text())


def _snapshot(root):
    files = [root / name for name in ('arms.json', 'arms_in.json', 'ROOT.json', 'eval500.json')]
    files += list((root / 'config').iterdir())
    return {str(path): path.read_bytes() for path in files}


@pytest.fixture(autouse=True)
def no_launches(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('process launch, live transport or chain attempted in CPU replica test')

    for module, names in ((control, ('run', 'remote', 'chain')),
                          (local_server, ('start', 'rpc')), (node, ('launch',))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


@pytest.fixture
def dry_transport(monkeypatch):
    def digest(path):
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in _DIGESTS:
            _DIGESTS[key] = _SHA(path)
        return _DIGESTS[key]

    monkeypatch.setattr(assets, 'sha', digest)
    for flag in ('OSCL_MANIFEST', 'OSCL_INIT_POOL', 'OSCL_TASKS', 'OSCL_EPISODES',
                 'OSCL_POOL_CONTRACT', 'OSCL_POOL_CONTRACT_SHA256', 'OSCL_RECORD_LOOKS',
                 'OSCL_EXECUTION_AUDIT'):
        monkeypatch.delenv(flag, raising=False)
    calls = {'bundles': [], 'rpc': []}
    monkeypatch.setattr(control, 'bundle', lambda *args: calls['bundles'].append(args))

    @contextlib.contextmanager
    def daemon(*args):
        yield 'dry://no-network'

    def rpc(host, action, spec, **kwargs):
        assert host == 'h100' and action == 'pull'
        calls['rpc'].append((host, action, spec))
        return 'DRY_TRANSFER'

    monkeypatch.setattr(control, 'daemon', daemon)
    monkeypatch.setattr(control, 'rpc', rpc)
    return calls


@pytest.fixture(scope='module')
def installed(tmp_path_factory):
    destination = tmp_path_factory.mktemp('r12_rep_install')
    return {Path(path).name: Path(path) for path in rep.install(destination)}


@pytest.mark.parametrize('cell', rep.CELLS)
def test_staged_definitions_equal_deployed_off(cell):
    source = rep.TRACE / f'r12_test_{cell}'
    root = rep.STAGED / f'r12_test_rep_{cell}'
    row = _read(root / 'arms.json')
    original = next(r for r in _read(source / 'arms.json') if r['arm'] == source.name + '_off')
    assert row == [rep._rewrite(original, source, root)]
    specs = [r for r in _read(source / 'arms_in.json') if r['name'] == original['arm']]
    assert _read(root / 'arms_in.json') == rep._rewrite(specs, source, root)
    record = _read(root / 'ROOT.json')
    assert record['arms'] == [root.name + '_off'] and record['methods'] == ['off']
    assert record['arm_count'] == 1 and record['launches'] == 0
    assert record['init_pool'] == 'A' and record['episodes_per_arm'] == 500
    assert (root / 'eval500.json').read_bytes() == (source / 'eval500.json').read_bytes()
    assert rep.devset.sha(root / 'eval500.json') == record['selection_sha256']
    frozen = record['frozen_artifacts'][row[0]['arm']]
    assert frozen == _read(source / 'ROOT.json')['frozen_artifacts'][original['arm']]
    assert rep.devset.sha(frozen['path']) == frozen['sha256']
    assert row[0]['kwargs']['episode_subset'] == original['kwargs']['episode_subset']
    assert sum(map(len, row[0]['kwargs']['episode_subset'].values())) == (200 if cell.endswith('200') else 50)
    assert {p.name for p in (root / 'config').iterdir()} == {
        root.name + '_off.yaml', 'matrix_' + root.name + '_off.yaml'}
    assert Path(row[0]['yaml']).read_bytes() == Path(original['yaml']).read_bytes()
    assert Path(row[0]['matrix']).read_bytes() == Path(original['matrix']).read_bytes().replace(
        source.name.encode(), root.name.encode())
    assert {p.name for p in root.iterdir()} == {
        'arms.json', 'arms_in.json', 'ROOT.json', 'eval500.json', 'config'}


@pytest.mark.parametrize('placement', ['staged', 'installed'])
@pytest.mark.parametrize('local', [True, False], ids=['local', 'h100_timan108'])
@pytest.mark.parametrize('cell', rep.CELLS)
def test_real_plan_sync_and_complete_source_argv(
        cell, local, placement, installed, tmp_path, monkeypatch, dry_transport):
    """Real builders on both source and replica; only external transport is mocked."""
    source = rep.TRACE / f'r12_test_{cell}'
    name = f'r12_test_rep_{cell}'
    root = rep.STAGED / name if placement == 'staged' else installed[name]
    monkeypatch.setenv('SERVER_HOST', 'local' if local else 'h100')
    monkeypatch.setenv('WORKER_HOST', 'timan108')
    before = {path: _snapshot(path) for path in (source, root)}
    argv_results, plans, env_results = [], [], []
    work = tmp_path / 'transport'
    for current in (source, root):
        arm = current.name + '_off'
        plan = control.sync(current, [arm], plan_only=True, workdir=work)
        synced = control._sync(current, [arm], workdir=work)
        assert plan == synced == _read(work / 'synced.json')
        assert len(synced['arms']) == 1 and 'dev_contracts' not in synced
        ready = synced['arms'][0]
        original = next(r for r in _read(current / 'arms.json') if r['arm'] == arm)
        count, manifest_path, manifest_sha = control.selection(current, original)
        assert count == 500
        manifest = load_manifest(manifest_path)
        assert manifest_sha == manifest['sha256'] == load_manifest(source / 'eval500.json')['sha256']
        assert all(sum(t == task for t, _ in manifest['selected']) == 50 for task in range(10))
        fit = next(f for f in synced['files'] if arm + ': fitted method' in f['reasons'])
        frozen = _read(current / 'ROOT.json')['frozen_artifacts'][arm]
        assert fit['source_sha256'] == frozen['sha256'] == assets.sha(frozen['path'])
        if local:
            assert fit['sha256'] == frozen['sha256'] and fit['source'] == frozen['path']
        spec = control.server_spec(current, ready, 23160 if local else 23260)
        if local:
            # command() creates an empty runtime directory; keep it in scratch.
            scratch = tmp_path / 'argv' / current.name
            spec.update(run_root=str(scratch), out=str(scratch / 'runs' / arm / 'server_23160'))
            argv, env = local_server.command(spec)
        else:
            argv, env = node.command(spec['model'], spec['suite'], spec['port'], spec['yaml'],
                                     spec['out'], spec['tag'], spec['plugin'], dict(spec['env']))
        _, remaining = parse_flags(argv[5 if local else 2:])
        opts, _ = plugin.parse_cli(remaining)
        assert opts.os_lookhalf == 'off' and opts.os_method == original['method']
        assert not opts.os_request_cameras and not opts.os_record_looks and not opts.os_look_contract
        assert env['STAGE1_ONLY'] == '0'
        driver_env = control.execution_env(ready)
        assert driver_env == {'OSCL_REPLAN_STEPS': '5'}
        _assert_legacy_worker(ready, manifest, driver_env, tmp_path, monkeypatch)
        plans.append(synced)
        argv_results.append(argv)
        env_results.append(env)
    assert argv_results[1] == rep._rewrite(argv_results[0], source, root)
    assert env_results[1] == rep._rewrite(env_results[0], source, root)
    # The same served fit and runtime/native payload bytes are required, even
    # when h100 relocation changes the source pickle's path metadata.
    payloads = []
    for plan in plans:
        payloads.append({f['original']: (f['source_sha256'], f['sha256'], f['size'])
                         for f in plan['files'] if not any(
                             reason.endswith((': server config', ': client matrix')) for reason in f['reasons'])})
    assert payloads[0] == payloads[1]
    assert dry_transport['bundles'] and all(call[0] == 'timan108' for call in dry_transport['bundles'])
    assert bool(dry_transport['rpc']) is (not local)
    assert before == {path: _snapshot(path) for path in (source, root)}


def _assert_legacy_worker(row, manifest, env, scratch, monkeypatch):
    from openpi.conductor import WorkerSpec
    from openpi.conductor import driver as journal_driver
    from exp.ablation_study.cache_size.run_size_eval import load_apool_digest

    driver = SimpleNamespace(WorkerSpec=WorkerSpec, load_apool_digest=load_apool_digest)
    with monkeypatch.context() as patch:
        patch.setattr(journal_driver, 'Journal', journal_driver.Journal)
        for key, val in env.items():
            patch.setenv(key, val)
        worker_pool.install(driver, ['--task-suite', row['suite'], '--journal', str(scratch / 'journal.jsonl')], manifest)
        spec = driver.WorkerSpec(worker_id='cpu_check', server_key='unused:0', gpu_id='', replan_steps=5)
        assert spec.worker_module == 'examples.libero.worker_entry'


def test_install_only_definitions(installed):
    for cell in rep.CELLS:
        name = f'r12_test_rep_{cell}'
        source, target = rep.STAGED / name, installed[name]
        assert (target / 'eval500.json').read_bytes() == (source / 'eval500.json').read_bytes()
        assert _read(target / 'arms.json') == rep._rewrite(_read(source / 'arms.json'), source, target)
        assert _read(target / 'ROOT.json')['root'] == _read(target / 'ROOT.json')['final_root'] == str(target)
        assert {p.relative_to(target) for p in target.rglob('*') if p.is_file()} == {
            p.relative_to(source) for p in source.rglob('*') if p.is_file()}
        assert not any((target / name).exists() for name in ('runs', 'state', 'h100_sync'))


@pytest.mark.parametrize('collision', ['first', 'last', 'dangling_symlink'])
def test_install_refuses_all_collisions_before_any_copy(collision, tmp_path):
    destination = tmp_path / 'trace'
    destination.mkdir()
    cell = rep.CELLS[0] if collision == 'first' else rep.CELLS[-1]
    target = destination / f'r12_test_rep_{cell}'
    if collision == 'dangling_symlink':
        target.symlink_to(tmp_path / 'missing')
    else:
        target.mkdir()
        (target / 'sentinel').write_bytes(b'preserve')
    with pytest.raises(FileExistsError, match='fresh destination'):
        rep.install(destination)
    assert list(destination.iterdir()) == [target]
    if collision != 'dangling_symlink':
        assert (target / 'sentinel').read_bytes() == b'preserve'


def test_prepare_refuses_existing_roots_before_any_emission(tmp_path):
    stage = tmp_path / 'stage'
    collision = stage / f'r12_test_rep_{rep.CELLS[-1]}'
    collision.mkdir(parents=True)
    with pytest.raises(FileExistsError, match='fresh staging'):
        rep.prepare(stage)
    assert list(stage.iterdir()) == [collision]


@pytest.mark.parametrize('changed', ['eval500.json', 'arms.json', 'ROOT.json', 'yaml', 'matrix', 'extra_config'])
def test_install_rejects_staged_drift_before_any_copy(changed, tmp_path):
    stage = tmp_path / 'stage'
    record = rep.prepare(stage)[-1]
    root = Path(record['root'])
    row = _read(root / 'arms.json')[0]
    if changed == 'extra_config':
        path = root / 'config' / 'active.yaml'
    else:
        path = Path(row[changed]) if changed in ('yaml', 'matrix') else root / changed
    path.write_bytes(path.read_bytes() + b' ' if path.exists() else b'active')
    destination = tmp_path / 'trace'
    with pytest.raises(ValueError, match='(differs|only off)'):
        rep.install(destination, stage)
    assert not destination.exists()
