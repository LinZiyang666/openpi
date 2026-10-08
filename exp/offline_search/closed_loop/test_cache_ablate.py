"""Pure-cache ablation: actual roots, dry transport, camera work and B-only math."""
import copy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace as NS

import numpy as np
import pytest

from . import plugin, worker_pool, devset
from .cache_ablate import ARMS, SPEC
from .cache_ablate_cameras import CameraRequest, ledger, receipt
from .cache_ablate_config import validate_fitted
from .cache_ablate_ledger import summarize
from .lookhalf_config import value
from .ops.h100 import assets, control, local_server, node
from .ops.remote.run_gtp_subset import load_manifest
from .test_r12_test_a import dry_transport

HERE = Path(__file__).resolve().parents[1] / 'rounds/r12/ablate'
RECORDS = [r for file in ('pilot_roots.json', 'test_a_roots.json') for r in json.loads((HERE/file).read_text())]


@pytest.mark.parametrize('local', [True, False])
@pytest.mark.parametrize('record', RECORDS, ids=lambda r: Path(r['root']).name)
def test_emitted_plan_sync_complete_argv(record, local, tmp_path, monkeypatch, dry_transport):
    root = Path(record['root'])
    monkeypatch.setenv('SERVER_HOST', 'local' if local else 'h100')
    host = 'timan107' if local else 'timan108'
    monkeypatch.setenv('WORKER_HOST', host)
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    work = tmp_path/'plan'
    plan = control.sync(root, record['arms'], plan_only=True, workdir=work)
    synced = control._sync(root, record['arms'], workdir=work)
    assert plan == synced == json.loads((work/'synced.json').read_text())
    mapper = assets.local_remap if local else assets.remap
    for original, row in zip(json.loads((root/'arms.json').read_text()), synced['arms'], strict=True):
        selected = original['kwargs']['method']
        manifest = load_manifest(original['manifest'])
        assert control.selection(root, original)[0] == record['episodes_per_arm']
        assert Path(record['source_selection']).read_bytes() == Path(original['manifest']).read_bytes()
        assert all(sum(t == task for t, _ in manifest['selected']) == record['episodes_per_arm']//10 for task in range(10))
        fit = next(f for f in synced['files'] if original['arm']+': fitted method' in f['reasons'])
        frozen = record['frozen_artifacts'][original['arm']]
        assert fit['original'] == frozen['path']
        assert fit['source_sha256'] == frozen['sha256'] == assets.sha(frozen['path'])
        args = mapper(copy.deepcopy(original['plugin_args']), root)
        mapped_fit = value(args, '--os-fit-artifact')
        if local:
            args = [fit['source'] if v == mapped_fit else v for v in args]
            assert fit['sha256'] == frozen['sha256']
        else:
            assert mapped_fit == str(assets.BASE/fit['rel'])
        args[args.index('--os-cache-ablate-sha256')+1] = fit['sha256']
        if record['init_pool'] == 'B':
            c = next(f for f in synced['files'] if original['arm']+': B recording contract' in f['reasons'])
            args += ['--os-look-contract', c['source'] if local else str(assets.BASE/c['rel']),
                     '--os-look-contract-sha256', c['sha256']]
        else:
            assert 'dev_contracts' not in synced
        assert row['plugin_args'] == args
        spec = control.server_spec(root, row, 23160 if local else 23260)
        expected = ['--os-method', SPEC, '--os-cell', original['cell'], *args,
                    '--os-kwargs', json.dumps(mapper(original['kwargs'], root))]
        assert spec['plugin'] == expected
        if local:
            spec.update(run_root=str(tmp_path/'argv'), out=str(tmp_path/'argv/runs'/original['arm']/'server'))
        argv, env = local_server.command(spec) if local else node.command(spec['model'], spec['suite'], spec['port'],
            spec['yaml'], spec['out'], spec['tag'], spec['plugin'], dict(spec['env']))
        stock = (['--port', str(spec['port']), '--replicas', '1', '--cache-config', spec['yaml'],
                  'policy:checkpoint', '--policy.config', 'pi05_libero', '--policy.dir',
                  local_server.PI05_CKPT if local else '/home/exouser/ckpt/pi05_libero_pytorch']
                 if original['model'] == 'pi05' else
                 ['--checkpoint', '/data/ckpt/n15_'+original['suite'] if local else '/home/exouser/ckpt/n15_'+original['suite'],
                  '--port', str(spec['port']), '--denoising-steps', '8', '--concurrent', '--allow-dynamic-bundles',
                  '--cache-config', spec['yaml']])
        assert argv[5 if local else 2:] == [*expected, '--os-log-dir', spec['out'], '--os-tag', spec['tag'], *stock]
        opts, rest = plugin.parse_cli(argv[5 if local else 2:])
        assert rest == stock and opts.os_cache_ablate == selected
        assert opts.os_fit_artifact == (fit['source'] if local else mapped_fit)
        assert opts.os_cache_ablate_sha256 == fit['sha256']
        assert opts.judge is None and opts.os_lookhalf is None and not opts.os_policy_tail
        assert opts.os_request_cameras and opts.os_blind and opts.os_tokens == 'off'
        assert opts.os_record_looks == (record['init_pool'] == 'B')
        assert opts.os_record_look_images == (56 if record['init_pool'] == 'B' else 0)
        assert not opts.os_record_look_shadow and opts.os_groot_path_check_every == 0
        assert env['STAGE1_ONLY'] == '0'
        driver_env = control.execution_env(row)
        assert driver_env['OSCL_EXECUTION_AUDIT'] == '1'
        assert driver_env.get('OSCL_RECORD_LOOKS') == ('1' if record['init_pool'] == 'B' else None)
        _worker_check(root, original, manifest, driver_env, tmp_path, monkeypatch)
        with Path(frozen['path']).open('rb') as stream:
            method = pickle.load(stream)['method']
        validate_fitted(method, selected)
        assert method.next_camera_mode == ('full' if selected == 'both' else 'wrist_only' if selected.startswith('wrist') else 'agent_only')
        assert not hasattr(method.inner, 'heads') and method.inner.budget == 1 and method.inner.gates == 'budget_only'
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert dry_transport and all(host0 == host for host0, _, _ in dry_transport)
    assert not (root/'runs').exists() and not (root/'state').exists()


def _worker_check(root, row, manifest, env, scratch, monkeypatch):
    from openpi.conductor import WorkerSpec, driver as journal_driver
    from exp.ablation_study.cache_size.run_size_eval import load_apool_digest
    driver = NS(WorkerSpec=WorkerSpec, load_apool_digest=load_apool_digest)
    args = ['--task-suite', row['suite'], '--journal', str(scratch/'journal.jsonl')]
    # Actual pool/file attestation on both pools; no conductor process starts.
    with monkeypatch.context() as patch:
        patch.setattr(journal_driver, 'Journal', journal_driver.Journal)
        patch.setenv('OSCL_INIT_POOL', row['init_pool'])
        for key, v in env.items():
            if key != 'OSCL_RECORD_LOOKS':  # stamp_tasks needs run_gtp's Strategy class, not WorkerSpec.
                patch.setenv(key, v)
        patch.delenv('OSCL_RECORD_LOOKS', raising=False)
        if row['init_pool'] == 'B':
            c = devset.arm_contract(root, row, manifest)
            p = scratch/'worker_contract.json'
            p.write_text(json.dumps(c))
            patch.setenv('OSCL_POOL_CONTRACT', str(p))
            patch.setenv('OSCL_POOL_CONTRACT_SHA256', devset.sha(p))
            args += ['--arm-matrix', row['matrix']]
        worker_pool.install(driver, args, manifest)
        worker = driver.WorkerSpec(worker_id='cpu', server_key='unused:0', gpu_id='', replan_steps=5)
        assert worker.worker_module == 'exp.offline_search.closed_loop.execution_worker'


@pytest.mark.parametrize('model', ['pi05', 'groot'])
@pytest.mark.parametrize('selected', ARMS)
def test_ledger_checks_engagement_and_no_calls(model, selected):
    mode = 'full' if selected == 'both' else 'wrist_only' if selected.startswith('wrist') else 'agent_only'
    work = {}
    receipt(CameraRequest(None, mode, work))
    c = ledger(model, selected, vision=True, work=work, hit=True)
    blind = ledger(model, selected, vision=False, work={}, hit=True)
    result = summarize(model, selected, [dict(vision=True, hit=True, camera_work=work, cache_ablate_cost=c),
                                        dict(vision=False, hit=True, cache_ablate_cost=blind)])
    assert result['calls'] == 0 and result['encoded_cameras'] == (2 if selected == 'both' else 1)
    assert result['one_camera_looks'] == int(selected != 'both')
    assert result['two_camera_looks'] == int(selected == 'both')
    assert result['ir_per_five_controls'] == pytest.approx((.152 if model == 'pi05' else .148)/(2 if selected == 'both' else 4))
    with pytest.raises(RuntimeError, match='CAMERA_RECEIPT_REQUIRED'):
        ledger(model, selected, vision=True, work={}, hit=True)
    with pytest.raises(RuntimeError, match='POLICY_CALL_REFUSED'):
        ledger(model, selected, vision=True, work=work, hit=False)


@pytest.mark.parametrize('selected', ARMS)
def test_missing_execution_receipt_fails_before_encoding(selected, tmp_path):
    from .test_look_recording import connection
    row = next(r for r in json.loads((Path(RECORDS[5]['root'])/'arms.json').read_text()) if r['kwargs']['method'] == selected)
    opts, _ = plugin.parse_cli(['--os-method', SPEC, '--os-cell', row['cell'], '--os-kwargs', json.dumps(row['kwargs']),
                               *row['plugin_args'], '--os-log-dir', str(tmp_path)])
    conn, session, obs, _ = connection(plugin, opts)
    session.rt.cache_ablate = selected
    conn._osp_adapter = NS()
    for extra in ({}, dict(executed_steps=5, exec_ok=True, decision_id=2)):
        with pytest.raises(ValueError, match='R12_EXECUTION_AUDIT_REQUIRED'):
            conn.infer({**obs, '__extra__': extra})
        assert session.rt.decision_count == session.step == 0


def test_installer_preserves_configs_and_selection(tmp_path):
    from exp.offline_search.rounds.r12.ablate.prepare import install_test_a
    installed = install_test_a(tmp_path/'A')
    assert len(installed) == 5
    for record, path in zip(RECORDS[5:], installed, strict=True):
        source, target = Path(record['root']), Path(path)
        assert (target/'eval500.json').read_bytes() == (source/'eval500.json').read_bytes()
        for row in json.loads((target/'arms.json').read_text()):
            assert row['manifest'] == str(target/'eval500.json')
            for field in ('yaml', 'matrix'):
                assert Path(row[field]).read_bytes() == (source/'config'/Path(row[field]).name).read_bytes()
        assert not (target/'runs').exists()
    with pytest.raises(FileExistsError):
        install_test_a(tmp_path/'A')


@pytest.mark.parametrize('selected', ARMS)
def test_configuration_refuses_mixed_or_corrected_paths(selected, tmp_path):
    row = next(r for r in json.loads((Path(RECORDS[5]['root'])/'arms.json').read_text()) if r['kwargs']['method'] == selected)
    args = ['--os-method', SPEC, '--os-cell', row['cell'], '--os-kwargs', json.dumps(row['kwargs']),
            *row['plugin_args'], '--os-log-dir', str(tmp_path)]
    for flag in ('--os-judge=always', '--os-lookhalf=off', '--os-policy-tail', '--os-record-look-shadow',
                 '--os-groot-path-check-every=1', '--os-cache-ablate=both'):
        with pytest.raises(SystemExit):
            plugin.parse_cli([*args, flag])
    path = HERE/'artifacts/pi05_l10_50'/f'{selected}.pkl'
    with path.open('rb') as stream:
        m = pickle.load(stream)['method']
    with pytest.raises(ValueError, match='fitted ablation differs'):
        validate_fitted(m, None)


@pytest.mark.parametrize('cell', [r['cell'] for r in RECORDS[:5]])
def test_both_layer1_byte_equivalence_and_reduced_metric_definition(cell):
    from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
    from exp.offline_search.rounds.r10.recipe.fitting import _Library, library_boundary
    from exp.offline_search.rounds.r02.g3_recovery.g3_core import PseudoQuery, episode_rows
    from exp.offline_search.rounds.r12.ablate.build import STORE, R10, fitted_data, refit_metric
    model, suite, size = cell.split('_')
    with (R10/'recipe/artifacts'/f'r10_recipe_{cell}.pkl').open('rb') as stream:
        original = pickle.load(stream)['method']
    reference = BlindAWM.__new__(BlindAWM)
    vars(reference).update(vars(original.inner.base))
    with (HERE/'artifacts'/cell/'both.pkl').open('rb') as stream:
        both = pickle.load(stream)['method']
    with library_boundary(STORE, Path(STORE)/'library'/f'{model}_{suite}'/original.library):
        lib = _Library(STORE, f'{model}_{suite}', original.library, original.episode_subset)
        C = fitted_data(original, None, lib)
        rows = next(iter(episode_rows(original.inner.C).values()))
        for pos in (0, 2, 4):
            q = PseudoQuery(lib, original.inner.C, rows, pos, model, True)
            q.hist_has_vision = np.ones(pos, bool)
            reference.reset(q.episode); both.reset(q.episode)
            a, b = reference.query(q), both.query(q)
            np.testing.assert_array_equal(b.topk, original.row_subset[a.topk])
            for attr in ('scores', 'action'):
                np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
            assert a.confidence == b.confidence and a.extras == b.extras
            # Also exercise the actual deployed wrapper with only its
            # corrective base replaced; guard diagnostics cannot alter picks.
            original.inner.base = reference
            original.reset(q.episode)
            served = original.query(q)
            np.testing.assert_array_equal(served.topk, b.topk)
            np.testing.assert_array_equal(served.action, b.action)
            from .blind import BlindQueryView, BlindResult
            bq = BlindQueryView(pos+1, q.task_id, q.episode, q.rs, q.rs, True, b.action,
                               np.array([b.action]*(pos+1)), np.ones(pos+1), np.array([q.rs]*(pos+1)),
                               np.ones(pos+1, bool), 0)
            ba, bb = reference.blind_step(bq), both.blind_step(bq)
            assert isinstance(ba, BlindResult) and isinstance(bb, BlindResult)
            np.testing.assert_array_equal(ba.action, bb.action)
            np.testing.assert_array_equal(original.row_subset[ba.rows], bb.rows)
        for camera, name in ((0, 'agent'), (1, 'wrist')):
            for variant in ('sub', 'refit'):
                with (HERE/'artifacts'/cell/f'{name}_{variant}.pkl').open('rb') as stream:
                    reduced = pickle.load(stream)['method']
                T = original.inner.base.tasks[0]
                ix = np.r_[np.arange(camera*64, (camera+1)*64), np.arange(128, 136)]
                X = np.c_[C['P'], C['rs']].astype(np.float64)[T.rows][:, ix]
                m = reduced.inner.camera_metrics[0]
                if variant == 'sub':
                    delta = X[0]-X
                    for initial, W in ((False, T.Wf), (True, T.W0f)):
                        Q = W[ix].astype(np.float64) @ W[ix].astype(np.float64).T
                        expected = np.sqrt(np.maximum(np.einsum('ni,ij,nj->n', delta, Q, delta), 0))
                        np.testing.assert_allclose(m.distance(X[0].astype(np.float32), initial), expected, atol=1e-4, rtol=1e-5)
                else:
                    heads = (np.asarray(lib.action[:, :5, :7], np.float64)/original.inner.base.sig.astype(np.float64)).reshape(lib.L, 35)
                    fitted = refit_metric(X, heads[T.rows], C['ep'][T.rows], C['step'][T.rows])
                    for field in vars(fitted):
                        np.testing.assert_array_equal(getattr(fitted, field), getattr(m, field))
                # Deleting the other camera proves retrieval never reads it.
                class Query:
                    def __getattr__(self, field):
                        if field == f'key_v{1-camera}' or field == f'hist_key_v{1-camera}':
                            raise AssertionError('unused camera accessed')
                        return getattr(q, field)
                reduced.reset(q.episode)
                result = reduced.query(Query())
                assert np.isfinite(result.action).all()
                assert reduced.fit_info['library_only'] and all('/library/' in p for p in reduced.fit_info['fit_reads'])


@pytest.mark.parametrize('mode', ['full', 'wrist_only', 'agent_only'])
@pytest.mark.parametrize('batched', [False, True])
def test_installed_pi05_encodes_only_selected_real_cameras(mode, batched, monkeypatch):
    import torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.serving.batching_coordinator import Pi05StageBatcher
    from openpi.serving import stage_io
    from openpi.models.model import Observation
    from exp.offline_search.rounds.r07.c2_wrist.test_dispatch import Model, stock, raw, keys
    from .cache_ablate_cameras import install
    monkeypatch.setattr(PI0Pytorch, 'run_stage1', stock)
    monkeypatch.setattr(PI0Pytorch, '_r4_override', None, raising=False)
    model = Model()
    def full_batch(batcher, observations):
        return [stock(batcher._model, Observation.from_dict(stage_io.stack_observation([obs], device='cpu')))
                for obs in observations]
    monkeypatch.setattr(Pi05StageBatcher, 'run_stage1_batch', full_batch)
    install('pi05')
    audit = {}
    if batched:
        batcher = NS(_model=model, _device='cpu')
        out = Pi05StageBatcher.run_stage1_batch(batcher, [CameraRequest(raw(3.), mode, audit)])[0]
    else:
        obs = Observation.from_dict(stage_io.stack_observation([raw(3.)], device='cpu'))
        out = model.run_stage1(CameraRequest(obs, mode, audit))
    encoded_values = [values[0] for n, values in model.paligemma_with_expert.calls]
    # The stock control also computes its historical masked dummy; count real
    # cameras explicitly, and require single-camera dispatch has no dummy call.
    assert [x for x in encoded_values if x != -1.] == ([3., 6.] if mode == 'full' else [6.] if mode == 'wrist_only' else [3.])
    if mode != 'full':
        assert len(encoded_values) == 1
    k = keys(out)
    assert torch.all(k['vision_0'] == (0. if mode == 'wrist_only' else 3.))
    assert torch.all(k['vision_1'] == (0. if mode == 'agent_only' else 6.))
    assert audit['initial_cameras'] == (2 if mode == 'full' else 1) and audit['completions'] == 0
    assert not hasattr(out, 'deferred_base')


@pytest.mark.parametrize('mode', ['full', 'wrist_only', 'agent_only'])
def test_installed_groot_encodes_only_selected_cameras(mode, monkeypatch):
    import torch
    from openpi.cache.groot.staged import GrootStagedRunner, GrootStage1Output
    from .cache_ablate_cameras import install
    ids = torch.cat([torch.ones((1, 2), dtype=torch.long), torch.full((1, 256), 7),
                     torch.ones((1, 1), dtype=torch.long), torch.full((1, 256), 7)], 1)
    pixels = torch.tensor([3., 6.]).reshape(2, 1, 1, 1)
    calls = []
    def encode(x):
        calls.append(x[:, 0, 0, 0].tolist())
        return x[:, 0, 0, 0][:, None, None].expand(-1, 256, 4).clone()
    eagle = NS(image_token_index=7, extract_feature=encode,
               language_model=NS(get_input_embeddings=lambda: lambda ids: torch.zeros((*ids.shape, 4))))
    inputs = dict(eagle_input_ids=ids, eagle_attention_mask=torch.ones_like(ids), eagle_pixel_values=pixels)
    runner = NS(_eagle=eagle, _require_session=lambda *a: None,
                _model=NS(prepare_input=lambda obs: (inputs, {})))
    def full(runner, obs):
        features = runner._eagle.extract_feature(pixels)
        embeds = torch.zeros((*ids.shape, 4))
        embeds[ids == 7] = features.reshape(-1, 4)
        return GrootStage1Output(embeds, torch.ones_like(ids), ids == 7, {})
    monkeypatch.setattr(GrootStagedRunner, 'run_stage1', full)
    monkeypatch.setattr(GrootStagedRunner, '_r12_override', None, raising=False)
    install('groot')
    audit = {}
    out = GrootStagedRunner.run_stage1(runner, CameraRequest({}, mode, audit))
    assert calls == ([[3., 6.]] if mode == 'full' else [[6.]] if mode == 'wrist_only' else [[3.]])
    assert audit['initial_cameras'] == (2 if mode == 'full' else 1) and audit['completions'] == 0
    assert torch.all(out.input_embeds[0, 2:258] == (0. if mode == 'wrist_only' else 3.))
    assert torch.all(out.input_embeds[0, 259:] == (0. if mode == 'agent_only' else 6.))
    assert not hasattr(out, 'deferred_base')


@pytest.mark.parametrize('selected', ARMS)
def test_collection_retains_accepted_camera_ledger(selected, tmp_path):
    from .ops import collect
    row = next(r for r in json.loads((Path(RECORDS[5]['root'])/'arms.json').read_text()) if r['kwargs']['method'] == selected)
    (tmp_path/'arms.json').write_text(json.dumps([row]))
    arm = row['arm']
    client = tmp_path/'runs'/arm/'client'
    client.mkdir(parents=True)
    uid = f'{arm}:eval:0:0'
    (client/'journal.jsonl').write_text(json.dumps(dict(task_uid=uid, yaml_id=arm, phase='eval', status='failed',
        accepted=True, success=False, attempt=2, init_pool='A'))+'\n')
    mode = 'full' if selected == 'both' else 'wrist_only' if selected.startswith('wrist') else 'agent_only'
    work = {}; receipt(CameraRequest(None, mode, work))
    server = tmp_path/'runs'/arm/'server_fixture'
    server.mkdir()
    rows = []
    for attempt, step, vision in [(1, 0, True), (2, 0, True), (2, 1, False), (2, 0, True)]:
        rows.append(dict(ev='dec', uid=uid, attempt=attempt, step=step, vision=vision, hit=True,
            ts=float(step), ok=True, camera_work=work if vision else {},
            cache_ablate_cost=ledger('pi05', selected, vision=vision, work=work, hit=True)))
    (server/'decisions_fixture.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    collected = collect.summarize(tmp_path, arm, write=False)
    result = collected['cache_ablate_cost_ledger']
    assert collected['cost_ledger'] == result
    assert result['decisions'] == 2 and result['vision_decisions'] == 1 and result['calls'] == 0
    assert result['two_camera_looks'] == int(selected == 'both')
    assert result['one_camera_looks'] == int(selected != 'both')


def test_changed_metric_sha_refused_before_pickle_load(tmp_path):
    row = json.loads((Path(RECORDS[5]['root'])/'arms.json').read_text())[0]
    args = ['--os-method', SPEC, '--os-cell', row['cell'], '--os-kwargs', json.dumps(row['kwargs']),
            *row['plugin_args'], '--os-log-dir', str(tmp_path)]
    args[args.index('--os-cache-ablate-sha256')+1] = '0'*64
    opts, _ = plugin.parse_cli(args)
    runtime = NS(api=None, store=None, opts=opts, cell=row['cell'])
    with pytest.raises(SystemExit, match='artifact SHA256 differs'):
        plugin.PluginRuntime._load_and_fit(runtime)


@pytest.mark.parametrize('selected', ARMS)
def test_pure_cache_judge_always_full_hit(selected):
    from openpi.cache.components.judge import HitType
    session = NS(rt=NS(judge=None), _dec=dict(hit=True, extras=dict(os_force_miss=1)), wire_diag=lambda: {})
    result = plugin.PluginJudge(session)([NS(id='method_pick')], 'CP1', None)
    assert result.hit_type == HitType.FULL_HIT and result.winner_id == 'method_pick'


@pytest.mark.parametrize('selected', ARMS)
def test_serving_initial_look_uses_fixed_camera_and_records_work(selected, tmp_path):
    from .test_look_recording import connection, FakePolicy
    row = next(r for r in json.loads((Path(RECORDS[5]['root'])/'arms.json').read_text()) if r['kwargs']['method'] == selected)
    opts, _ = plugin.parse_cli(['--os-method', SPEC, '--os-cell', row['cell'], '--os-kwargs', json.dumps(row['kwargs']),
                               *row['plugin_args'], '--os-log-dir', str(tmp_path)])
    conn, s, obs, records = connection(plugin, opts)
    mode = 'full' if selected == 'both' else 'wrist_only' if selected.startswith('wrist') else 'agent_only'
    s.rt.cache_ablate = selected
    s.rt.request_cameras = s.rt.blind = s.rt.r4 = True
    s.method.next_camera_mode = mode
    s.method.set_camera_mode = lambda received: None if received == mode else pytest.fail('camera changed')
    s.method.invalidate_anchor = lambda: None
    class Policy(FakePolicy):
        def stage1(self, observation):
            receipt(CameraRequest(observation, self.s._camera_mode, self.s._camera_work))
            return observation
    conn._osp_inner = Policy(s)
    conn._osp_adapter = plugin._BlindAdapter(conn._osp_inner, s)
    conn.infer({**obs, '__extra__': dict(executed_steps=0, exec_ok=True, decision_id=0)})
    d = next(r for r in records if r['ev'] == 'dec')
    assert d['hit'] and d['vision'] and d['camera_mode'] == mode
    assert d['camera_encoded_count'] == (2 if selected == 'both' else 1)
    assert d['cache_ablate_cost']['policy_cost'] == d['cache_ablate_cost']['completion_cost'] == 0
    assert d['owner_cost'] == (.152 if selected == 'both' else .076)
