from dataclasses import asdict, replace
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
import yaml

from exp.r17_integration.conductor.manifest import EndpointSpec, episode_tasks
from exp.r17_integration.conductor.strategy import MetaWorldStrategy, PerTaskDescent, MetaWorldDescentStrategy
from exp.r17_integration.conductor.driver import MetaWorldConductorDriver
from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, R17MetaWorldEpisodeRunner
from exp.r17_integration.tests.benchmarks.test_metaworld import Env, Client, task
from exp.r17_integration.tests.benchmarks.test_libero import spec as libero_spec
from openpi.conductor.task import EpisodeResult, ServerEndpoint
from openpi.conductor.protocol import result_to_wire, task_from_wire


def test_config_identity_prevents_queueing_same_geometry_different_recipe(tmp_path):
    endpoint = EndpointSpec(ServerEndpoint('local', 1), 'm', 'n', 'l', FORMAL_SPEC.identity,
                            bundle_identities=('recipe_gap1',))
    t = episode_tasks(campaign='c', arm='x', bundle_id='b', endpoint=endpoint.endpoint,
                      model_identity='m', norm_identity='n', layout_identity='l', names=['reach-v3'], indices=[0],
                      bundle_identity='recipe_gap1')[0]
    assert endpoint.accepts(t)
    assert not endpoint.accepts(replace(t, extra={**t.extra, 'bundle_identity': 'recipe_gap2'}))
    unknown = replace(t, extra={k: v for k, v in t.extra.items() if k != 'bundle_identity'})
    assert not endpoint.accepts(unknown)
    with pytest.raises(ValueError, match='compatible'):
        MetaWorldStrategy([unknown], endpoints=[endpoint])


def test_mw_cost_final_ack_and_physical_request_generation_match():
    c, environment, events = Client(), Env(success=7), []
    infer = c.infer
    def timed(obs):
        result = infer(obs)
        result['server_timing'] = {'infer_ms': 3.2, 'prev_total_ms': 4.5}
        result['__hit_meta__'] = {'source': 'policy', 'cursor': 0, 'actual_nfe': 2, 'stage_calls': [1, 1, 2],
                                'capture_enabled': False, 'pricing_eligible': True}
        return result
    c.infer = timed
    t = task(attempt=2)
    t = replace(t, extra={**t.extra, 'bundle_generation': 'distinct_generation'})
    runner = R17MetaWorldEpisodeRunner(env_factory=lambda *a: environment, client_factory=lambda _: c,
                                     capture_sink=events.append, framework_context=True)
    result = runner.run(t, lambda *a: None)
    costs = [r for r in result.per_step_rows if not r.get('_kind')]
    assert len(costs) == 2 and [r['decision_seq'] for r in costs] == [0, 1]
    assert all(r['cost_ms'] == 3.2 and r['server_timing']['prev_total_ms'] == 4.5 for r in costs)
    assert all(r['attempt'] == 2 and r['bundle_generation'] == 'distinct_generation' for r in costs)
    assert c.calls[-1][1]['execution'] == {'decision_seq': 1, 'executed_count': 2, 'terminal': True}
    assert {r['bundle_generation'] for r in events} == {r['__request__']['bundle_generation'] for r in c.requests}
    assert c.calls[1][1]['extra_metadata']['ack_protocol'] == 'r17.execution.v1'
    assert result.per_step_rows[-1]['cursor_end'] == result.per_step_rows[-1]['last_cursor'] + 1


def test_terminal_ack_transport_failure_cannot_be_accepted_as_success():
    c = Client()
    c.episode_end = lambda **kwargs: (_ for _ in ()).throw(ConnectionError('lost final ack'))
    runner = R17MetaWorldEpisodeRunner(env_factory=lambda *a: Env(success=1), client_factory=lambda _: c,
                                     framework_context=True)
    with pytest.raises(RuntimeError, match='final execution'):
        runner.run(task(), lambda *a: None)


def test_descent_terminal_error_finishes_with_diagnostic_freeze_and_resume(tmp_path):
    endpoint = EndpointSpec(ServerEndpoint('local', 1), 'm', 'n', 'l', FORMAL_SPEC.identity)
    def make():
        controller = PerTaskDescent(['reach-v3'], ['k2', 'k1'], {'reach-v3': {30: True, 31: True}}, n_dev=2)
        episodes = []
        for level in controller.levels:
            group = episode_tasks(campaign='c', arm=level, bundle_id=level, endpoint=endpoint.endpoint,
                                  model_identity='m', norm_identity='n', layout_identity='l', num_steps=2,
                                  partition='dev', names=['reach-v3'], indices=[30, 31])
            episodes.extend(replace(t, extra={**t.extra, 'descent_level': level}) for t in group)
        strategy = MetaWorldDescentStrategy(episodes, endpoints=[endpoint], controller=controller)
        driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp_path/'dispatch.jsonl',
                    yaml_weights={t.yaml_id: 2 for t in episodes}, servers=[endpoint.endpoint],
                    journal_path=str(tmp_path/'journal.jsonl'), ctl_factory=lambda e: NS(close=lambda: None))
        return controller, strategy, driver
    controller, strategy, driver = make()
    driver.drive_stages_once()
    _, payload = driver.handle_pull('local:1')
    dispatched = task_from_wire(payload['task'])
    driver.handle_result(result_to_wire(EpisodeResult(dispatched.task_uid, False, 0,
                         'FatalEpisodeError: native error', attempt=dispatched.attempt)))
    for _ in range(10):
        driver.drive_stages_once()
    assert driver.scheduler.all_done()
    assert 'task' not in driver.handle_pull('local:1')[1]
    receipt = strategy.freeze({'model': 'm'})
    assert receipt['status'] == 'diagnostic_error' and receipt['errors']['reach-v3']['level'] == 'k2'
    assert all(not cohort for cohort in receipt['cohorts'].values())
    with pytest.raises(ValueError, match='terminal dev errors'):
        controller.authorize_test(receipt['freeze_hash'], receipt['identities'])
    driver.close()
    resumed, strategy, driver = make()
    for _ in range(10):
        driver.drive_stages_once()
    assert driver.scheduler.all_done() and resumed.errors == controller.errors
    driver.close()


def test_bundle_restoration_after_registry_loss_and_socket_restart(tmp_path):
    endpoint = EndpointSpec(ServerEndpoint('local', 1), 'm', 'n', 'l', FORMAL_SPEC.identity)
    tasks = episode_tasks(campaign='c', arm='full', bundle_id='full', endpoint=endpoint.endpoint,
                          model_identity='m', norm_identity='n', layout_identity='l', names=['reach-v3'], indices=[0, 1])
    registry, loads = {}, []
    class Control:
        def __init__(self):
            self.dead = False
        def close(self):
            self.dead = True
        def load_cache_config(self, path, **kwargs):
            if self.dead:
                raise ConnectionError('cached control socket died before stage setup')
            registry[kwargs['bundle_id']] = path
            loads.append(kwargs['bundle_id'])
        def select_bundle(self, bundle):
            if self.dead:
                raise ConnectionError('server restarted')
            if bundle not in registry:
                raise ValueError('unknown bundle')
    strategy = MetaWorldStrategy(tasks, endpoints=[endpoint], bundles={'full': tmp_path/'full.yaml'})
    driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp_path/'dispatch.jsonl',
                 yaml_weights={tasks[0].yaml_id: 2}, servers=[endpoint.endpoint],
                 journal_path=str(tmp_path/'journal.jsonl'), ctl_factory=lambda e: Control())
    registry.clear()
    driver._ctls['local:1'].dead = True
    driver.drive_stages_once()
    assert 'full' in registry  # stage start restores lost bundles
    _, p = driver.handle_pull('local:1')
    driver.handle_result(result_to_wire(EpisodeResult(p['task']['task_uid'], True, 1, attempt=p['task']['attempt'])))
    registry.clear()
    driver._ctls['local:1'].dead = True
    _, p = driver.handle_pull('local:1')
    assert 'task' in p and 'full' in registry and len(loads) >= 3
    driver.close()


def test_model_neutral_libero_driver_manifest_and_plan(tmp_path):
    from exp.r17_integration.common.fake_adapter import fake_spec, fake_provider
    from exp.r17_integration.common.provider import binding_for
    from exp.r17_integration.common.contracts import digest
    from exp.r17_integration.conductor.driver import main
    from exp.r17_integration.models.pi05.provider import config_document
    s = libero_spec()
    model = replace(fake_spec(horizon=32, action_dim=7, views=2), allowed_benchmark_protocols=(s.identity,))
    provider = fake_provider('p2_plan_' + tmp_path.name, model)
    doc = config_document(provider, k=2, commit=15, evidence_dir=tmp_path/'miss')
    yaml_path = tmp_path/'full.yaml'
    yaml_path.write_text(yaml.safe_dump(doc))
    benchmark_path, manifest = tmp_path/'benchmark.json', tmp_path/'model.json'
    benchmark_path.write_text(json.dumps(s.manifest()))
    manifest.write_text(json.dumps({'r17_model_spec': asdict(model)}))
    main(['--manifest', str(manifest), '--benchmark-spec', str(benchmark_path), '--endpoint', 'never_connect:9',
          '--arm', f'full:{yaml_path}:2', '--output', str(tmp_path/'plan'), '--plan-only'])
    plan = json.loads((tmp_path/'plan/episode_manifest.json').read_text())
    assert plan['expected_episodes'] == 400 and plan['model']['layout']['horizon'] == 32
    assert {t['task_id'] for t in plan['tasks']} == set(range(40))
    assert {t['experiment'] for t in plan['tasks']} == set(SUITES := ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'))
    assert all(t['extra']['bundle_identity'] for t in plan['tasks'])
    assert all(t['extra']['commit_horizon'] == 15 for t in plan['tasks'])


@pytest.mark.parametrize('commit', [7, 12])
def test_mw_nondivisible_commit_rejected_at_plan_boundary(commit):
    with pytest.raises(ValueError, match='divisible'):
        episode_tasks(campaign='c', arm='full', bundle_id='full', endpoint=ServerEndpoint('local', 1),
                      model_identity='m', norm_identity='n', layout_identity='l', names=['reach-v3'], indices=[0],
                      commit_horizon=commit)


def test_worker_debug_selects_native_sampler_per_task_and_closes_writers(tmp_path, monkeypatch):
    from exp.mw_finetune import rendering
    from exp.r17_integration.conductor import worker_entry
    from exp.r17_integration.models.pi05.adapter import model_spec, sampler_for_steps
    from openpi_client import websocket_client_policy
    from openpi.conductor import worker
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    metadata = {'r17_model_spec': asdict(model_spec('m', 'n')),
                'r17_samplers': {str(k): asdict(sampler_for_steps(k)) for k in (2, 10)}}
    monkeypatch.setattr(rendering, 'configure', lambda *a: None)
    monkeypatch.setattr(rendering, 'make_env', lambda *a: Env(success=3))
    monkeypatch.setattr(websocket_client_policy, 'WebsocketClientPolicy', lambda *a: NS(
        get_server_metadata=lambda: metadata, close=lambda: None))
    monkeypatch.setattr(worker_entry, 'attest_worker', lambda *a: {'fixture_only': True})
    class Loop:
        def __init__(self, worker_id, server_key, runner, **kwargs):
            self.runner = runner
        def run_forever(self):
            self.runner._client_factory = lambda _: Client()
            for k in (2, 10):
                t = task(attempt=1)
                from exp.r17_integration.common.contracts import digest
                t = replace(t, task_uid=t.task_uid+f'_k{k}', extra={**t.extra, 'num_steps': k, 'task_data_hash': 'fixture',
                    'model_identity': 'm', 'norm_identity': 'n', 'layout_identity': digest(metadata['r17_model_spec']['layout'])})
                self.runner.run(t, lambda *a: None)
    monkeypatch.setattr(worker, 'WorkerLoop', Loop)
    directory = tmp_path/'worker_debug'
    worker_entry.main(['--worker-id', 'cpu_owned', '--server-key', 'unused:1', '--driver-host', 'unused',
                       '--driver-port', '1', '--debug-directory', str(directory)])
    manifests = list(directory.glob('binding_*/r17_capture.json'))
    assert len(manifests) == 2
    assert {json.loads(p.read_text())['binding']['sampler']['num_steps'] for p in manifests} == {2, 10}
    assert all(json.loads(p.read_text())['status'] == 'closed' for p in manifests)
    for path in manifests:
        reader = CaptureReader(path.parent)
        rows = [(r, a) for r, a in reader.physical() if r['event'] == 'control']
        assert len(rows) == 3
        assert all(arrays['qpos'].dtype == np.float64 for record, arrays in rows)
        assert all(record['bundle_generation'] == 'full' for record, arrays in rows)


def test_libero_worker_entry_uses_original_setup_local_ids_and_native_kernel(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from exp.r17_integration.conductor import worker_entry
    from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec
    from exp.r17_integration.tests.benchmarks.test_libero import Env as LiberoEnv, TailClient, task as libero_task, native_kernel, args
    from examples.libero import worker_entry as original_worker
    from openpi.conductor import worker
    s = libero_spec()
    benchmark_path = tmp_path/'benchmark.json'
    benchmark_path.write_text(json.dumps(s.manifest()))
    native = ModuleType('examples.libero.main')
    native.Args = lambda **kw: NS(**{**vars(args(s)), 'task_suite_name': 'libero_spatial', **kw})
    native._run_episode = native_kernel()
    monkeypatch.setitem(sys.modules, 'examples.libero.main', native)
    monkeypatch.setattr(LiberoBenchmarkSpec, 'from_official', classmethod(lambda cls, **kw: s))
    seen, outcomes, clients = [], [], []
    def setup_factory(worker_args, seed, pools, mode):
        assert seed == 7 and mode == 'orig' and pools == 'libero_object=fixture_pool'
        def setup(t):
            seen.append((t.experiment, t.task_id, t.orig_init_state_idx))
            global_id = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10').index(t.experiment)*10 + t.task_id
            return LiberoEnv(done_at=7), np.array([t.orig_init_state_idx]), s.tasks[global_id][1], 220
        return setup
    monkeypatch.setattr(original_worker, '_build_episode_setup', setup_factory)
    monkeypatch.setattr(original_worker, 'build_probe', lambda *a: {'libero_code_digest': 'fixture'})
    class Loop:
        def __init__(self, worker_id, server_key, runner, **kwargs):
            self.runner = runner
            assert kwargs['probe']['benchmark_protocol'] == s.identity
            assert kwargs['probe']['task_data']['original_init_idx'] == 0
        def run_forever(self):
            c = TailClient(32)
            clients.append(c)
            self.runner._client_factory = lambda endpoint: c
            outcomes.append(self.runner.run(libero_task(s, global_id=13, idx=7), lambda *a: None))
    monkeypatch.setattr(worker, 'WorkerLoop', Loop)
    worker_entry.main(['--worker-id', 'cpu_owned', '--server-key', 'unused:1', '--driver-host', 'unused',
                       '--driver-port', '1', '--benchmark', 'libero', '--benchmark-spec', str(benchmark_path),
                       '--init-states-dir', 'libero_object=fixture_pool'])
    assert seen == [('libero_spatial', 0, 0), ('libero_object', 3, 7)]
    assert outcomes[0].success and outcomes[0].n_steps == 7
    assert clients[0].starts[0]['extra_metadata']['task_id'] == 13
    assert clients[0].ends[0]['execution']['executed_count'] == 2
    command = worker_entry.worker_command(NS(worker_id='w', server_key='h:1', gpu_id=0), 'driver', 2,
                worker_python='python3.11', benchmark='libero', benchmark_spec=benchmark_path,
                init_states_dir='libero_object=fixture_pool')
    assert '--benchmark-spec' in command and command[command.index('--benchmark') + 1] == 'libero'


def test_worker_capture_rejects_wrong_checkpoint_before_writing():
    from exp.r17_integration.conductor.worker_entry import validate_capture_identity
    from exp.r17_integration.models.pi05.adapter import model_spec
    from exp.r17_integration.common.contracts import digest
    metadata = {"r17_model_spec": asdict(model_spec("actual", "norm"))}
    event = {"model_identity": "actual", "norm_identity": "norm", "layout_identity": digest(metadata["r17_model_spec"]["layout"])}
    validate_capture_identity(metadata, event)
    for field in event:
        with pytest.raises(ValueError, match="model binding mismatch"):
            validate_capture_identity(metadata, {**event, field: "wrong"})


def test_concurrent_dispatch_never_overlaps_control_rpc_on_one_socket(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    endpoint = EndpointSpec(ServerEndpoint('local', 1), 'm', 'n', 'l', FORMAL_SPEC.identity)
    tasks = episode_tasks(campaign='c', arm='full', bundle_id='full', endpoint=endpoint.endpoint,
                          model_identity='m', norm_identity='n', layout_identity='l', names=['reach-v3'], indices=[0])
    gate, calls, created = threading.Lock(), [], []
    class Control:
        def select_bundle(self, bundle):
            if not gate.acquire(blocking=False):
                raise RuntimeError('recv called while another thread is receiving')
            try:
                time.sleep(.002)
                calls.append(bundle)
            finally:
                gate.release()
        def load_cache_config(self, *a, **kw):
            pass
        def close(self):
            pass
    strategy = MetaWorldStrategy(tasks, endpoints=[endpoint], bundles={'full': tmp_path/'full.yaml'})
    controls = {}
    def ctl(target):
        if target.key not in controls:
            controls[target.key] = Control()
            created.append(target.key)
        return controls[target.key]
    strategy.driver = NS(_ctl=ctl, _ctls=controls, _ctl_lock=threading.Lock())
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: strategy.ensure_bundle(endpoint.endpoint, 'full'), range(50)))
    assert calls == ['full'] * 50 and created == ['local:1']
