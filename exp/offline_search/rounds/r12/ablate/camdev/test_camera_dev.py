"""CPU checks of all new roots through real planning/sync and mocked transport."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp.offline_search.closed_loop import devset, plugin, worker_pool
from exp.offline_search.closed_loop.cache_ablate_cameras import CameraRequest, ledger, receipt
from exp.offline_search.closed_loop.cache_ablate_config import SPEC
from exp.offline_search.closed_loop.lookhalf_config import value
from exp.offline_search.closed_loop.ops import collect
from exp.offline_search.closed_loop.ops.h100 import assets, control, local_server, node
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
from exp.offline_search.closed_loop.test_r12_test_a import dry_transport  # noqa: F401
from .prepare import ABLATE, ARMS, CELLS, HERE, REQUESTED, prepare, selection

RECORDS = json.loads((HERE / 'roots.json').read_text())


@pytest.fixture(scope='session')
def argv_evidence():
    records = []
    yield records
    destination = os.environ.get('CAMDEV_ARGV_EVIDENCE')
    if destination:
        Path(destination).write_text(json.dumps(records, indent=2) + '\n')


@pytest.mark.parametrize('record', RECORDS, ids=lambda r: r['cell'])
def test_selection_is_maximal_unused_B_and_hash_bound(record):
    root = Path(record['root'])
    manifest = load_manifest(root / 'dev_selection.json')
    previous = load_manifest(record['previous_selection']['path'])
    previous_pairs = set()
    assert len(record['previous_selections']) == 2
    for old_record in record['previous_selections']:
        old = load_manifest(old_record['path'])
        assert old['sha256'] == old_record['selection_sha256']
        assert devset.sha(old['path']) == old_record['file_sha256']
        assert old['data']['init_pool'] == 'B' and old['data']['dev'] is True
        assert not (set(manifest['selected']) & set(old['selected']))
        previous_pairs.update(old['selected'])
    assert {tuple(p) for p in record['excluded_K10_pairs']} == previous_pairs
    assert devset.root_pool(root) == 'B'
    assert tuple(record['methods']) == ARMS and record['arm_count'] == 2
    assert not record['a_used'] and record['refits'] == record['launches'] == record['remote_operations'] == 0
    assert not record['look_recording'] and record['cost_ledger'] and record['per_episode_journal']
    assert previous['sha256'] == record['previous_selection']['selection_sha256']
    assert devset.sha(previous['path']) == record['previous_selection']['file_sha256']
    assert manifest['sha256'] == record['selection_sha256']
    assert devset.sha(root / 'dev_selection.json') == record['selection_file_sha256']
    assert len(manifest['selected']) == len(manifest['data']['pairs']) == record['episodes_per_arm']
    assert not (set(manifest['selected']) & set(previous['selected']))
    rows = json.loads((root / 'arms.json').read_text())
    assert [r['arm'] for r in rows] == record['arms']
    for row in rows:
        contract = devset.arm_contract(root, row, manifest)
        assert contract['subset'] == record['library_subset']
        assert contract['manifest_sha256'] == record['selection_sha256']
        for task in range(10):
            library = set(contract['subset']['excluded_inits_by_task'][str(task)])
            old = {i for t, i in previous_pairs if t == task}
            available = set(range(50)) - library - old
            selected = {i for t, i in manifest['selected'] if t == task}
            assert not (selected & library) and not (selected & old)
            assert len(selected) == min(REQUESTED, len(available)) == record['per_task_counts'][str(task)]
            assert available == set(record['available_inits_by_task'][str(task)])
            assert selected == available  # All current cells have fewer than K40.
            expected_counts = ([15, 15, 18, 16, 19, 16, 15, 17, 16, 16] if record['cell'].endswith('_200')
                               else [27, 28, 31, 28, 32, 27, 28, 29, 28, 28])
            assert len(selected) == expected_counts[task]


@pytest.mark.parametrize('local', [True, False], ids=['local_timan107', 'h100_timan108'])
@pytest.mark.parametrize('record', RECORDS, ids=lambda r: r['cell'])
def test_every_arm_real_plan_sync_full_argv_worker(record, local, tmp_path, monkeypatch,
                                                  dry_transport, argv_evidence):
    root = Path(record['root'])
    host = 'timan107' if local else 'timan108'
    monkeypatch.setenv('SERVER_HOST', 'local' if local else 'h100')
    monkeypatch.setenv('WORKER_HOST', host)
    monkeypatch.setenv('LOCAL_SERVER_CPUS', '22-37,66-81')
    for key in ('PI05_CKPT', 'GROOT_CKPT', 'GROOT_DENOISING_STEPS'):
        monkeypatch.delenv(key, raising=False)
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    work = tmp_path / 'plan'
    plan = control.sync(root, record['arms'], plan_only=True, workdir=work)
    synced = control._sync(root, record['arms'], workdir=work)
    assert plan == synced == json.loads((work / 'synced.json').read_text())
    assert synced['init_pool'] == 'B' and synced['worker_host'] == host
    assert not any('recording contract' in reason for f in synced['files'] for reason in f['reasons'])
    mapper = assets.local_remap if local else assets.remap
    originals = json.loads((root / 'arms.json').read_text())
    for original, row in zip(originals, synced['arms'], strict=True):
        selected = original['kwargs']['method']
        assert original['cost_ledger'] is True
        manifest = load_manifest(original['manifest'])
        assert control.selection(root, original) == (record['episodes_per_arm'], manifest['path'], record['selection_sha256'])
        contract = synced['dev_contracts'][original['arm']]
        assert contract['manifest_sha256'] == record['selection_sha256']
        fit = next(f for f in synced['files'] if original['arm'] + ': fitted method' in f['reasons'])
        frozen = record['frozen_artifacts'][original['arm']]
        assert fit['original'] == frozen['path']
        assert fit['source_sha256'] == frozen['sha256'] == assets.sha(frozen['path'])
        args = mapper(copy.deepcopy(original['plugin_args']), root)
        mapped_fit = value(args, '--os-fit-artifact')
        if local:
            args = [fit['source'] if v == mapped_fit else v for v in args]
            assert fit['sha256'] == frozen['sha256']
        else:
            assert mapped_fit == str(assets.BASE / fit['rel'])
        args[args.index('--os-cache-ablate-sha256') + 1] = fit['sha256']
        assert row['plugin_args'] == args
        port = 23150 if local else 23250
        spec = control.server_spec(root, row, port)
        expected_plugin = ['--os-method', SPEC, '--os-cell', original['cell'], *args,
                           '--os-kwargs', json.dumps(mapper(original['kwargs'], root))]
        assert spec['plugin'] == expected_plugin
        if local:
            spec.update(run_root=str(tmp_path / 'argv'), out=str(tmp_path / 'argv/runs' / original['arm'] / 'server'))
        argv, env = local_server.command(spec) if local else node.command(
            spec['model'], spec['suite'], spec['port'], spec['yaml'], spec['out'], spec['tag'],
            spec['plugin'], dict(spec['env']))
        if original['model'] == 'pi05':
            prefix = (['taskset', '-c', '22-37,66-81', str(local_server.TREE / '.venv/bin/python'),
                       str(local_server.TREE / 'exp/offline_search/closed_loop/serve_pi05.py')] if local else
                      ['/home/exouser/openpi/.venv/bin/python', str(node.TREE / 'exp/offline_search/closed_loop/serve_pi05.py')])
            stock = ['--port', str(port), '--replicas', '1', '--cache-config', spec['yaml'],
                     'policy:checkpoint', '--policy.config', 'pi05_libero', '--policy.dir',
                     local_server.PI05_CKPT if local else '/home/exouser/ckpt/pi05_libero_pytorch']
        else:
            prefix = (['taskset', '-c', '22-37,66-81', local_server.GROOT_PY,
                       str(local_server.TREE / 'exp/offline_search/closed_loop/serve_groot.py')] if local else
                      ['/home/exouser/gr00t_n15_venv/.venv/bin/python', str(node.TREE / 'exp/offline_search/closed_loop/serve_groot.py')])
            stock = ['--checkpoint', ('/data/ckpt/n15_' if local else '/home/exouser/ckpt/n15_') + original['suite'],
                     '--port', str(port), '--denoising-steps', '8', '--concurrent',
                     '--allow-dynamic-bundles', '--cache-config', spec['yaml']]
        assert argv == [*prefix, *expected_plugin, '--os-log-dir', spec['out'], '--os-tag', spec['tag'], *stock]
        assert not any(a.startswith(('--os-record-look', '--os-look-contract')) for a in argv)
        opts, rest = plugin.parse_cli(argv[len(prefix):])
        assert rest == stock and opts.os_cache_ablate == selected
        assert opts.os_cache_ablate_sha256 == fit['sha256']
        assert opts.os_fit_artifact == (fit['source'] if local else mapped_fit)
        assert not opts.os_record_looks and opts.os_record_look_images == 0
        assert not opts.os_record_look_shadow and not opts.os_look_contract
        assert opts.os_blind and opts.os_request_cameras and opts.os_tokens == 'off'
        assert opts.judge is None and opts.os_lookhalf is None and not opts.os_policy_tail
        assert env['STAGE1_ONLY'] == '0'
        driver_env = control.execution_env(row)
        assert driver_env == dict(OSCL_EXECUTION_AUDIT='1', OSCL_REPLAN_STEPS='5')
        worker = install_worker(root, original, manifest, tmp_path / selected, monkeypatch)
        assert worker.worker_module == record['execution_worker']
        argv_evidence.append(dict(root=str(root), arm=original['arm'], server_host='local' if local else 'h100',
                                  worker_host=host, argv=argv, execution_env=driver_env,
                                  worker_module=worker.worker_module, source_fit_sha256=fit['source_sha256'],
                                  served_fit_sha256=fit['sha256'], selection_sha256=manifest['sha256'],
                                  look_recording=False, transport_mocked=True))
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert dry_transport and all(h == host for h, _, _ in dry_transport)
    assert not (root / 'runs').exists() and not (root / 'state').exists()


def install_worker(root, row, manifest, scratch, monkeypatch):
    from exp.ablation_study.cache_size.run_size_eval import load_apool_digest
    from openpi.conductor import WorkerSpec, driver as journal_driver

    scratch.mkdir(parents=True, exist_ok=True)
    contract_path = scratch / 'pool_contract.json'
    contract_path.write_text(json.dumps(devset.arm_contract(root, row, manifest)))
    driver = SimpleNamespace(WorkerSpec=WorkerSpec, load_apool_digest=load_apool_digest)
    args = ['--task-suite', row['suite'], '--journal', str(scratch / 'journal.jsonl'), '--arm-matrix', row['matrix']]
    monkeypatch.setattr(journal_driver, 'Journal', journal_driver.Journal)
    monkeypatch.setenv('OSCL_INIT_POOL', 'B')
    monkeypatch.setenv('OSCL_POOL_CONTRACT', str(contract_path))
    monkeypatch.setenv('OSCL_POOL_CONTRACT_SHA256', devset.sha(contract_path))
    monkeypatch.delenv('OSCL_RECORD_LOOKS', raising=False)
    for key, val in control.execution_env(row).items():
        monkeypatch.setenv(key, val)
    worker_pool.install(driver, args, manifest)  # Real B init digests and matrix/contract validation.
    return driver.WorkerSpec(worker_id='cpu', server_key='unused:0', gpu_id='', replan_steps=5)


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('record', RECORDS, ids=lambda r: r['cell'])
def test_B_journal_and_compute_ledger_survive_recording_off(record, arm, tmp_path, monkeypatch):
    from openpi.conductor import driver as journal_driver

    root = Path(record['root'])
    row = next(r for r in json.loads((root / 'arms.json').read_text()) if r['kwargs']['method'] == arm)
    manifest = load_manifest(row['manifest'])
    install_worker(root, row, manifest, tmp_path / 'worker', monkeypatch)
    journal = journal_driver.Journal(tmp_path / 'worker/journal.jsonl')
    task, init = min(manifest['selected'])
    uid = f'{row["arm"]}:eval:{task}:{init}'
    journal.record(task_uid=uid, yaml_id=row['arm'], phase='eval', status='failed', success=False,
                   accepted=True, attempt=2, duration_s=1.23456)
    entry = json.loads((tmp_path / 'worker/journal.jsonl').read_text())
    assert entry['init_pool'] == 'B' and entry['duration_s'] == 1.235
    assert journal.replay_done_uids() == {uid}
    # Collection gets ordinary episode journals and decision ledgers, no looks.
    collected_root = tmp_path / 'collection'
    client = collected_root / 'runs' / row['arm'] / 'client'
    client.mkdir(parents=True)
    (collected_root / 'arms.json').write_text(json.dumps([row]))
    (client / 'journal.jsonl').write_text(json.dumps(entry) + '\n')
    server = client.parent / 'server_fixture'
    server.mkdir()
    work = {}
    receipt(CameraRequest(None, 'full' if arm == 'both' else 'wrist_only', work))
    decisions = []
    for attempt, step, vision in ((1, 0, True), (2, 0, True), (2, 1, False), (2, 0, True)):
        decisions.append(dict(ev='dec', uid=uid, attempt=attempt, step=step, vision=vision, hit=True,
                              ts=float(step), ok=True, camera_work=work if vision else {},
                              cache_ablate_cost=ledger(row['model'], arm, vision=vision, hit=True,
                                                       work=work if vision else {})))
    (server / 'decisions_fixture.jsonl').write_text(''.join(json.dumps(d) + '\n' for d in decisions))
    result = collect.summarize(collected_root, row['arm'], write=False)
    cost = result['cache_ablate_cost_ledger']
    assert result['cost_ledger'] == cost
    assert result['complete'] == 1 and cost['decisions'] == 2 and cost['engagement_verified']
    assert cost['one_camera_looks'] == int(arm == 'wrist_sub')
    assert cost['two_camera_looks'] == int(arm == 'both')
    assert cost['encoded_cameras'] == (2 if arm == 'both' else 1)
    assert cost['work'] == pytest.approx((.152 if row['model'] == 'pi05' else .148) / (1 if arm == 'both' else 2))
    assert all(cost[field] == expected for field, expected in record['engagement_requirements'][row['arm']].items())
    assert not list(collected_root.rglob('looks*'))


def test_emission_reproducible_and_refuses_existing_roots(tmp_path, monkeypatch):
    opened = []
    original_open = Path.open

    def B_only_open(path, *args, **kwargs):
        name = str(path)
        assert not any(part in name for part in ('/queries/', '/test_a', '_apool', 'eval500.json', '/r11_knob_'))
        opened.append(name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', B_only_open)
    destination = tmp_path / 'roots'
    regenerated = prepare(destination)
    assert tuple(r['cell'] for r in regenerated) == CELLS
    for actual, fresh in zip(RECORDS, regenerated, strict=True):
        a, b = Path(actual['root']), Path(fresh['root'])
        assert (a / 'dev_selection.json').read_bytes() == (b / 'dev_selection.json').read_bytes()
        assert fresh['selection_sha256'] == actual['selection_sha256']
        assert sorted((v['path'], v['sha256']) for v in fresh['frozen_artifacts'].values()) == sorted(
            (v['path'], v['sha256']) for v in actual['frozen_artifacts'].values())
    before = {str(p): p.read_bytes() for p in destination.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError, match='fresh root required'):
        prepare(destination)
    assert before == {str(p): p.read_bytes() for p in destination.rglob('*') if p.is_file()}
    assert any('/r11_dev_k10_size50/' in path for path in opened)
    assert any('/r11_dev_k10_size200/' in path for path in opened)


@pytest.mark.parametrize('bad', ['A', 'wrong_previous_count', 'changed_screen'])
def test_selector_refuses_wrong_pool_or_prior_screen(bad, tmp_path):
    source = json.loads((ABLATE / 'pilot_roots.json').read_text())[0]
    base = json.loads((Path(source['root']) / 'arms.json').read_text())[0]
    prior = json.loads(Path(source['source_selection']).read_text())
    path = tmp_path / 'r11_dev_k10_size50' / 'dev_pi05_l10.json'
    path.parent.mkdir()
    if bad == 'A':
        prior.update(init_pool='A', dev=False)
    elif bad == 'wrong_previous_count':
        prior['pairs'].pop()
    else:
        prior['seed'] += 1
    path.write_text(json.dumps(prior))
    source = dict(source, source_selection=str(path))
    with pytest.raises(ValueError):
        selection('pi05_l10_50', source, base)


def test_existing_definitions_artifacts_and_methods_preserved():
    snapshot = json.loads((HERE / 'PRESERVED_INPUTS.json').read_text())
    assert snapshot and all(devset.sha(path) == digest for path, digest in snapshot.items())
