"""CPU integration checks on emitted R12 A roots and the Spatial-200 B screen.

Real planning, sync, startup argv and pool attestation run locally. Only sync's
external transport is mocked; no policy server, simulator or driver is started.
"""
from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import plugin, worker_pool
from .lookhalf_cameras import ledger
from .lookhalf_config import value
from .ops import collect
from .ops.h100 import assets, control, local_server, node
from .ops.remote.run_gtp_subset import load_manifest
from .stage_overrides import parse_flags
from .test_r12_pipeline import R12, _DIGESTS, _SHA

_A = json.loads((R12 / 'test_a_roots.json').read_text())
_B = json.loads((R12 / 'guard/fix2_roots.json').read_text())


@pytest.fixture
def dry_transport(monkeypatch):
    """Mock transport only and cache real hashes while all file stats match."""
    def digest(path):
        path = Path(path)
        st = path.stat()
        key = (str(path.resolve()), st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if key not in _DIGESTS:
            _DIGESTS[key] = _SHA(path)
        return _DIGESTS[key]

    monkeypatch.setattr(assets, 'sha', digest)
    for flag in ('OSCL_MANIFEST', 'OSCL_INIT_POOL', 'OSCL_TASKS', 'OSCL_EPISODES',
                 'OSCL_POOL_CONTRACT', 'OSCL_POOL_CONTRACT_SHA256', 'OSCL_RECORD_LOOKS',
                 'OSCL_EXECUTION_AUDIT'):
        monkeypatch.delenv(flag, raising=False)
    bundles = []
    monkeypatch.setattr(control, 'bundle', lambda host, dest, pairs: bundles.append((host, dest, pairs)))

    @contextlib.contextmanager
    def daemon(*args):
        yield 'dry://no-network'

    monkeypatch.setattr(control, 'daemon', daemon)
    monkeypatch.setattr(control, 'rpc', lambda *args, **kwargs: 'DRY_TRANSFER')
    return bundles


@pytest.mark.parametrize('local', [True, False], ids=['local_timan107', 'h100_timan108'])
@pytest.mark.parametrize('record', [*_A, *_B], ids=lambda r: Path(r['root']).name)
def test_actual_emitted_plan_sync_server_argv(record, local, tmp_path, monkeypatch, dry_transport):
    """Verify every new arm through the real dependency and argv builders."""
    root = Path(record['root'])
    monkeypatch.setenv('SERVER_HOST', 'local' if local else 'h100')
    host = 'timan107' if local else 'timan108'
    monkeypatch.setenv('WORKER_HOST', host)
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    runtime_dirs = {name: (root / name).exists() for name in ('runs', 'state')}
    original = json.loads((root / 'arms.json').read_text())
    work = tmp_path / 'plan'
    plan = control.sync(root, record['arms'], plan_only=True, workdir=work)
    synced = control._sync(root, record['arms'], workdir=work)
    assert plan == synced == json.loads((work / 'synced.json').read_text())
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert dry_transport and all(h == host for h, _, _ in dry_transport)
    mapper = assets.local_remap if local else assets.remap
    for row, ready in zip(original, synced['arms'], strict=True):
        selected = row['kwargs']['method']
        manifest = load_manifest(row['manifest'])
        assert len(manifest['selected']) == record['episodes_per_arm']
        assert all(sum(t == task for t, _ in manifest['selected']) == record['episodes_per_arm'] // 10
                   for task in range(10))
        assert control.selection(root, row)[0] == record['episodes_per_arm']
        frozen = record['frozen_artifacts'][row['arm']]
        assert value(row['plugin_args'], '--os-fit-artifact') == frozen['path']
        fit = next(f for f in synced['files'] if row['arm'] + ': fitted method' in f['reasons'])
        assert fit['original'] == frozen['path']
        assert fit['source_sha256'] == frozen['sha256'] == assets.sha(frozen['path'])
        expected_args = mapper(copy.deepcopy(row['plugin_args']), root)
        mapped_fit = value(expected_args, '--os-fit-artifact')
        if local:
            assert fit['sha256'] == frozen['sha256']
            assert fit['source'] == frozen['path']
            expected_args = [fit['source'] if v == mapped_fit else v for v in expected_args]
        else:
            assert str(assets.BASE / fit['rel']) == mapped_fit
        if record['init_pool'] == 'B':
            contract = next(f for f in synced['files'] if row['arm'] + ': B recording contract' in f['reasons'])
            expected_args += ['--os-look-contract', contract['source'] if local else str(assets.BASE / contract['rel']),
                              '--os-look-contract-sha256', contract['sha256']]
        else:
            assert row['manifest'] == str(root / 'eval500.json')
            assert 'dev_contracts' not in synced
            assert not any('recording contract' in reason for f in synced['files'] for reason in f['reasons'])
        assert ready['plugin_args'] == expected_args
        spec = control.server_spec(root, ready, 23160 if local else 23260)
        expected_plugin = ['--os-method', row['method'], '--os-cell', row['cell'], *expected_args,
                           '--os-kwargs', json.dumps(mapper(row['kwargs'], root))]
        assert spec['plugin'] == expected_plugin
        if local:
            assert spec['out'] == str(root / 'runs' / row['arm'] / 'server_23160')
            # The local argv builder also creates its empty runtime directory.
            # Keep that side effect in scratch, with all serving inputs intact.
            spec['run_root'] = str(tmp_path / 'argv')
            spec['out'] = str(tmp_path / 'argv/runs' / row['arm'] / 'server_23160')
        argv, env = local_server.command(spec) if local else node.command(
            spec['model'], spec['suite'], spec['port'], spec['yaml'], spec['out'], spec['tag'],
            spec['plugin'], dict(spec['env']))
        stock = (['--port', str(spec['port']), '--replicas', '1', '--cache-config', spec['yaml'],
                  'policy:checkpoint', '--policy.config', 'pi05_libero', '--policy.dir',
                  local_server.PI05_CKPT if local else '/home/exouser/ckpt/pi05_libero_pytorch']
                 if row['model'] == 'pi05' else
                 ['--checkpoint', '/data/ckpt/n15_' + row['suite'] if local else
                  '/home/exouser/ckpt/n15_' + row['suite'], '--port', str(spec['port']),
                  '--denoising-steps', '8', '--concurrent', '--allow-dynamic-bundles', '--cache-config', spec['yaml']])
        offset = 5 if local else 2
        assert argv[offset:] == [*expected_plugin, '--os-log-dir', spec['out'], '--os-tag', spec['tag'], *stock]
        _, remaining = parse_flags(argv[offset:])
        opts, rest = plugin.parse_cli(remaining)
        assert rest == stock and opts.os_lookhalf == selected
        assert opts.os_fit_artifact == (fit['source'] if local else mapped_fit)
        assert opts.os_request_cameras == (selected not in ('off', 'h2'))
        assert env['STAGE1_ONLY'] == '0'
        driver_env = control.execution_env(ready)
        assert driver_env.get('OSCL_EXECUTION_AUDIT') == ('1' if selected != 'off' else None)
        assert driver_env.get('OSCL_RECORD_LOOKS') == ('1' if record['init_pool'] == 'B' else None)
        assert driver_env['OSCL_REPLAN_STEPS'] == '5'
        if record['init_pool'] == 'A':
            assert not opts.os_record_looks and opts.os_record_look_images == 0
            assert not opts.os_look_contract and opts.os_groot_path_check_every == 0
            assert not any(v.split('=', 1)[0].startswith('--os-record-look') for v in argv)
            _assert_a_worker(row, manifest, driver_env, tmp_path, monkeypatch)
        else:
            assert opts.os_record_looks and opts.os_record_look_images == 56
            assert Path(record['source_selection']).read_bytes() == Path(row['manifest']).read_bytes()
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert runtime_dirs == {name: (root / name).exists() for name in runtime_dirs}


def _assert_a_worker(row, manifest, env, scratch, monkeypatch):
    from openpi.conductor import WorkerSpec
    from openpi.conductor import driver as journal_driver
    from exp.ablation_study.cache_size.run_size_eval import load_apool_digest

    driver = SimpleNamespace(WorkerSpec=WorkerSpec, load_apool_digest=load_apool_digest)
    # Real pool validation, file-content attestation and journal installation;
    # constructing a WorkerSpec does not start any driver or worker process.
    with monkeypatch.context() as patch:
        patch.setattr(journal_driver, 'Journal', journal_driver.Journal)
        for key, val in env.items():
            patch.setenv(key, val)
        worker_pool.install(driver, ['--task-suite', row['suite'], '--journal', str(scratch / 'journal.jsonl')], manifest)
        spec = driver.WorkerSpec(worker_id='cpu_check', server_key='unused:0', gpu_id='', replan_steps=5)
        expected = ('examples.libero.worker_entry' if row['kwargs']['method'] == 'off'
                    else 'exp.offline_search.closed_loop.execution_worker')
        assert spec.worker_module == expected


def test_frozen_selection_templates_and_pending_screen():
    """Require exactly the requested cells, frozen subsets, and separate hold root."""
    from exp.offline_search.rounds.r12.prepare_test_a import CELLS, TRACE

    assert len(_A) == 7 and sum(r['arm_count'] for r in _A) == 23
    assert len(_B) == 1 and _B[0]['methods'] == ['h2', 'm2'] and _B[0]['arm_count'] == 2
    assert [r['methods'] for r in _A] == [list(methods) for _, methods, _ in CELLS]
    assert sum(r['pending_B_screen'] for r in _A) == 1
    for record in _A:
        root = Path(record['root'])
        assert Path(record['final_root']).parent == TRACE
        for n in range(1, 5):
            assert (root / 'eval500.json').read_bytes() == (TRACE / f'r11_knob_{n}/eval500.json').read_bytes()
        for row in json.loads((root / 'arms.json').read_text()):
            frozen = record['frozen_artifacts'][row['arm']]
            source = Path(frozen['source'])
            if source.name == 'arms.json':
                pilot = next(r for r in json.loads(source.read_text()) if r['kwargs']['method'] == row['kwargs']['method'])
                assert row['kwargs'] == pilot['kwargs']
                assert row['src_yaml'] == pilot['src_yaml']
                assert row['yaml_sha256'] == pilot['yaml_sha256']
            assert row['kwargs']['library'] == ('bpool_cs' if row['model'] == 'pi05' else 'bpool_all')
            assert sum(map(len, row['kwargs']['episode_subset'].values())) == (200 if '200' in root.name else 50)
            assert row['full_model'] and row['cost_ledger'] and row['replan_steps'] == 5
            assert value(row['plugin_args'], '--os-policy-tail-blocks') == '1'
            assert value(row['plugin_args'], '--os-judge') == 'guard_only'


def test_spatial_200_frozen_theta_provenance():
    """Read existing frozen fits and verify the original dev calibration values."""
    import pickle
    risk = json.loads((R12 / 'guard/risk_model.json').read_text())
    reference = json.loads((R12 / 'opus/out/guard_m2.json').read_text())['thetas']
    for selected in ('h2', 'm2'):
        with (R12 / f'guard/artifacts/pi05_spatial_200/{selected}.pkl').open('rb') as stream:
            fitted = pickle.load(stream)['method']
        assert fitted.risk_fit['threshold_source_arm'] == 'r11_dev_pi05_spatial_200'
        assert 'B-dev calibration' in fitted.fit_info['guard_thresholds_provenance']
        for key in ('theta_a', 'theta_b'):
            assert fitted.risk_fit[key] == reference['r11_dev_pi05_spatial_200'][key] == risk['cells']['pi05_spatial_200'][key]


def test_staged_install_rewrites_only_owned_paths(tmp_path):
    """Exercise final placement in scratch, preserving selections and config bytes."""
    from exp.offline_search.rounds.r12.prepare_test_a import install_test_a

    destination = tmp_path / 'trace_runs'
    installed = install_test_a(destination)
    assert len(installed) == 7
    for record, target in zip(_A, installed, strict=True):
        source, target = Path(record['root']), Path(target)
        assert (target / 'eval500.json').read_bytes() == (source / 'eval500.json').read_bytes()
        for row in json.loads((target / 'arms.json').read_text()):
            assert row['manifest'] == str(target / 'eval500.json')
            assert Path(row['yaml']).read_bytes() == (source / 'config' / Path(row['yaml']).name).read_bytes()
            assert Path(row['matrix']).is_file()
            assert value(row['plugin_args'], '--os-fit-artifact') == record['frozen_artifacts'][row['arm']]['path']
            assert control.selection(target, row)[0] == 500
        assert not (target / 'runs').exists() and not (target / 'state').exists()
    with pytest.raises(FileExistsError):
        install_test_a(destination)


@pytest.mark.parametrize('selected', ['opus_m1', 'h2', 'm2'])
def test_active_a_request_without_receipt_fails_before_encoding(selected, tmp_path):
    """A serving session refuses missing/malformed receipts without fallback."""
    from .test_look_recording import connection

    row = next(r for r in json.loads((Path(_A[0]['root']) / 'arms.json').read_text())
               if r['kwargs']['method'] == selected)
    opts, _ = plugin.parse_cli(['--os-method', row['method'], '--os-cell', row['cell'],
                               '--os-kwargs', json.dumps(row['kwargs']), *row['plugin_args'],
                               '--os-log-dir', str(tmp_path / 'server')])
    assert opts.os_lookhalf == selected and not opts.os_record_looks
    conn, session, obs, _ = connection(plugin, opts)
    session.rt.lookhalf = selected
    conn._osp_adapter = SimpleNamespace()
    before = vars(session.method).copy()
    for extra in ({}, {'executed_steps': 5, 'exec_ok': True, 'decision_id': 2}):
        with pytest.raises(ValueError, match='R12_EXECUTION_AUDIT_REQUIRED'):
            conn.infer({**obs, '__extra__': extra})
        assert session.rt.decision_count == 0
        assert vars(session.method) == before
        assert session.step == 0


@pytest.mark.parametrize('selected', ['off', 'opus_m1', 'h2', 'm2'])
def test_a_collection_retains_ledger_and_first_guard_per_episode(selected, tmp_path):
    """Exclude retries, duplicates and incomplete episodes, and retain no-call nulls."""
    row = next(r for r in json.loads((Path(_A[0]['root']) / 'arms.json').read_text()) if r['kwargs']['method'] == selected)
    (tmp_path / 'arms.json').write_text(json.dumps([row]))
    arm = row['arm']
    client = tmp_path / 'runs' / arm / 'client'
    client.mkdir(parents=True)
    uids = [f'{arm}:eval:0:{n}' for n in range(4)]
    journal = [dict(task_uid=u, yaml_id=arm, phase='eval', status='failed', accepted=True,
                    success=False, attempt=2 if n == 0 else 1, init_pool='A') for n, u in enumerate(uids[:3])]
    (client / 'journal.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in journal))
    rows = []
    for uid, step, attempt, reason in [(uids[0], 8, 2, 4), (uids[0], 4, 2, 4), (uids[0], 0, 1, 4),
                                       (uids[1], 0, 1, 62), (uids[3], 0, 1, 4), (uids[0], 4, 2, 4)]:
        rows.append(dict(ev='dec', uid=uid, step=step, attempt=attempt, ts=float(step), tag='fixture',
                         hit=False, vision=True, ok=True, extras=dict(os_reason=reason),
                         lookhalf_cost=ledger('pi05', vision=True, wrist_first=False, completions=0, miss=True)))
    server = tmp_path / 'runs' / arm / 'server_fixture'
    server.mkdir()
    (server / 'decisions_fixture.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = collect.summarize(tmp_path, arm, write=False)
    assert summary['complete'] == 3
    assert summary['first_guard_call'] == {uids[0]: 4, uids[1]: None, uids[2]: None}
    cost = summary['lookhalf_cost_ledger']
    assert cost['decisions'] == 3 and cost['ir_per_five_controls'] == 1
    assert cost['first_guard_call'] == {uids[0]: 4}
    assert cost['per_episode'][uids[0]]['first_guard_call'] == 4
    assert cost['per_episode'][uids[1]]['first_guard_call'] is None
    assert not (client.parent / 'looks').exists()
