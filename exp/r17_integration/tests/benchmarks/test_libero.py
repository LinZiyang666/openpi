"""Reviewer B1/B2 regressions: actual original LIBERO kernel, CPU env/model fixtures."""
import ast
import collections
from dataclasses import asdict, replace
import inspect
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from exp.r17_integration.benchmarks.libero_bridge import (
    SUITES, LiberoBenchmarkSpec, R17LiberoEpisodeRunner, decision_seed,
)
from exp.r17_integration.conductor.manifest import episode_tasks, EndpointSpec, load_benchmark_spec, validate_worker_probe
from openpi.conductor.task import ServerEndpoint


def spec():
    maps = {s: {'suite': s, 'tasks': [{'task_id': i, 'name': f'native_task_{i}', 'language': f'{s} instruction {i}'}
                                    for i in range(10)]} for s in SUITES}
    benchmark = LiberoBenchmarkSpec.from_task_maps(maps)
    # The real server loader explicitly registers its sealed benchmark descriptor.
    from exp.r17_integration.common.benchmarks import register_benchmark
    register_benchmark(benchmark)
    return benchmark


def native_kernel():
    """Compile the unchanged original function; isolate its eager vendor imports.

    Image operations and collector helpers are original. Only quaternion->axis
    conversion uses the exact identity-quaternion result for our CPU fixture.
    """
    from openpi_client import image_tools
    from examples.libero.collect_util import encode_collect_meta
    source = Path('examples/libero/main.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_run_episode')
    scope = {'np': np, 'time': time, 'collections': collections, 'logging': logging, 'image_tools': image_tools,
             'LIBERO_DUMMY_ACTION': [0.] * 6 + [-1.], '_quat2axisangle': lambda q: np.zeros(3),
             '_encode_collect_meta': encode_collect_meta, 'Callable': object, 'Any': object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'examples/libero/main.py', 'exec', flags=__import__('__future__').annotations.compiler_flag), scope)
    return scope['_run_episode']


class Env:
    def __init__(self, done_at=34, fail_at=None):
        self.done_at, self.fail_at = done_at, fail_at
        self.actions = []
        self.closed = False
        self.t = 0
    def obs(self):
        image = np.full((256, 256, 3), self.t, np.uint8)
        image[0, 0] = [1, 2, 3]
        return {'agentview_image': image, 'robot0_eye_in_hand_image': image + np.uint8(10),
                'robot0_eef_pos': np.array([.1, .2, .3]), 'robot0_eef_quat': np.array([0., 0., 0., 1.]),
                'robot0_gripper_qpos': np.array([.4, .5])}
    def reset(self):
        self.t = 0
    def set_init_state(self, initial):
        self.initial = np.array(initial)
        return self.obs()
    def step(self, action):
        control = self.t - 10 + 1
        if control == self.fail_at:
            raise ValueError('fixture native error')
        self.actions.append(np.asarray(action))
        self.t += 1
        return self.obs(), 0., control == self.done_at, {}
    def close(self):
        self.closed = True


def args(s):
    return NS(resize_size=s.wire_size, replan_steps=s.replan_steps, num_steps_wait=s.settle_steps,
              seed=s.seed, save_trajectory=False, save_video=False, display=False, collect_gate_dir='')


def task(s, *, global_id=13, idx=7):
    return replace(episode_tasks(campaign='libero_cpu', arm='full', bundle_id='bundle', endpoint=ServerEndpoint('local', 1),
                   model_identity='m', norm_identity='n', layout_identity='l', num_steps=2, partition='test', spec=s,
                   names=[s.tasks[global_id][0]], indices=[idx], bundle_generation='actual_generation')[0], attempt=1)


class TailClient:
    """Small controller oracle: refuse tail reuse without actual-count acks."""
    def __init__(self, commit):
        self.commit = commit
        self.requests, self.ends, self.starts = [], [], []
        self.cursor = self.seq = self.calls = 0
        self.chunk = np.arange(32 * 7, dtype=np.float32).reshape(32, 7) / 300
    def select_bundle(self, bundle):
        pass
    def episode_start(self, **kwargs):
        self.starts.append(kwargs)
    def infer(self, obs):
        self.requests.append(obs)
        ack = obs.get('__execution__')
        if ack:
            assert ack['decision_seq'] == self.seq - 1
            self.cursor += ack['executed_count']
        else:
            self.cursor = 0
        if self.cursor >= self.commit:
            self.cursor = 0
        source = 'policy_tail' if self.cursor else 'policy'
        if source == 'policy':
            self.calls += 1
        self.seq += 1
        return {'actions': self.chunk[self.cursor:self.commit].copy(), 'server_timing': {'infer_ms': .7},
                '__hit_meta__': {'source': source, 'cursor': self.cursor, 'actual_nfe': 2 if source == 'policy' else 0,
                                 'stage_calls': [1, 1, 2] if source == 'policy' else [0, 0, 0], 'sampler_identity': 'fixture'}}
    def episode_end(self, **kwargs):
        self.ends.append(kwargs)
    def close(self):
        pass


@pytest.mark.parametrize('commit,counts,calls', [(10, [5, 5, 5, 5, 5, 5, 4], 4),
                                                (15, [5, 5, 5, 5, 5, 5, 4], 3),
                                                (32, [5, 5, 5, 5, 5, 5, 2, 2], 2)])
def test_original_kernel_executes_extended_commit_and_partial_terminal_tail(commit, counts, calls):
    s, environment, client = spec(), Env(), TailClient(commit)
    t = task(s)
    runner = R17LiberoEpisodeRunner(args(s), lambda _: (environment, np.array([7., 8.]), s.tasks[t.task_id][1], 220),
                                  spec=s, client_factory=lambda _: client, run_episode_fn=native_kernel())
    result = runner.run(t, lambda *args: None)
    assert result.success and result.n_steps == 34 and result.attempt == 1
    receipts = [r for r in result.per_step_rows if r.get('_kind') == 'execution_ack']
    assert [r['executed_count'] for r in receipts] == counts
    assert sum(r['executed_count'] for r in receipts) == 34
    assert client.calls == calls
    assert client.ends[-1]['execution'] == {'decision_seq': len(counts) - 1, 'executed_count': counts[-1], 'terminal': True}
    assert all(r['cursor_end'] == r['last_cursor'] + 1 for r in receipts)
    assert client.starts[0]['extra_metadata']['original_init_idx'] == 7
    assert client.starts[0]['extra_metadata']['task_id'] == 13
    for seq, obs in enumerate(client.requests):
        request = obs['__request__']
        assert request['task_name'] == 'libero_object/native_task_3'
        assert request['original_init_idx'] == 7 and request['task_id'] == 13
        assert request['bundle_generation'] == 'actual_generation'
        assert request['explicit_seed'] == decision_seed(request['task_name'], 7, seq)
        assert obs['observation/image'].shape == obs['observation/wrist_image'].shape == (256, 256, 3)
        assert not np.array_equal(obs['observation/image'], obs['observation/wrist_image'])
    verdicts = [r for r in result.per_step_rows if not r.get('_kind')]
    assert [r['decision_seq'] for r in verdicts] == list(range(len(counts)))
    assert all(r['cost_ms'] == .7 and r['timing_scope'] == 'server_timing.infer_ms' for r in verdicts)
    assert all(r['bundle_generation'] == 'actual_generation' and r['task_data_hash'] for r in receipts)
    runner.close()


def test_original_libero_default_off_bytes_and_early_done():
    from examples.libero.episode_runner import LiberoEpisodeRunner
    s = spec()
    environments, clients = [Env(done_at=3), Env(done_at=3)], [TailClient(5), TailClient(5)]
    t = task(s)
    setup = lambda i: lambda _: (environments[i], np.array([7., 8.]), s.tasks[t.task_id][1], 220)
    old = LiberoEpisodeRunner(args(s), setup(0), client_factory=lambda _: clients[0], run_episode_fn=native_kernel())
    new = R17LiberoEpisodeRunner(args(s), setup(1), spec=s, client_factory=lambda _: clients[1],
                                run_episode_fn=native_kernel(), framework_context=False)
    results = [r.run(t, lambda *a: None) for r in (old, new)]
    assert results[0].success == results[1].success
    assert [a.tobytes() for a in environments[0].actions] == [a.tobytes() for a in environments[1].actions]
    assert clients[0].starts == clients[1].starts and clients[0].ends == clients[1].ends
    for a, b in zip(clients[0].requests, clients[1].requests, strict=True):
        assert a.keys() == b.keys()
        assert all(np.array_equal(a[k], b[k]) for k in a)
    # Enabled path counts early success accurately (legacy final timestep is off by one).
    client = TailClient(32)
    enabled = R17LiberoEpisodeRunner(args(s), lambda _: (Env(done_at=3), np.zeros(2), s.tasks[t.task_id][1], 220),
                                    spec=s, client_factory=lambda _: client, run_episode_fn=native_kernel())
    result = enabled.run(t, lambda *a: None)
    assert result.n_steps == client.ends[-1]['execution']['executed_count'] == 3


def test_global_ids_partitions_context_seeds_and_worker_probe(tmp_path):
    s = spec()
    tasks = episode_tasks(campaign='libero_cpu', arm='full', bundle_id='bundle', endpoint=ServerEndpoint('local', 1),
                          model_identity='m', norm_identity='n', layout_identity='l', spec=s)
    assert len(tasks) == len({t.task_uid for t in tasks}) == 400
    assert {t.task_id for t in tasks} == set(range(40))
    assert all(t.experiment == SUITES[t.task_id // 10] and t.extra['suite_task_id'] == t.task_id % 10 for t in tasks)
    seeds = {decision_seed(s.tasks[tid][0], idx, 0) for tid in (3, 13, 23, 33) for idx in (0, 7)}
    assert len(seeds) == 8
    assert s.identity != replace(s, wire_size=96).identity
    assert s.identity != replace(s, seed=42, replan_steps=32, max_steps_by_suite=(280, 280, 300, 520)).identity
    with pytest.raises(ValueError):
        replace(s, wire_size=224)
    path = tmp_path / 'benchmark.json'
    path.write_text(json.dumps(s.manifest()))
    assert load_benchmark_spec(path) == s
    from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkAdapter
    probe = LiberoBenchmarkAdapter(s).worker_probe({'task_data': {'state_sha256': 'fixture'}, 'libero_code_digest': 'fixture'})
    validate_worker_probe(probe, spec=s)
    with pytest.raises(ValueError):
        validate_worker_probe({**probe, 'resize_size': 224}, spec=s)


def test_native_error_cannot_be_reported_as_sr_failure():
    s, client = spec(), TailClient(32)
    t = task(s)
    runner = R17LiberoEpisodeRunner(args(s), lambda _: (Env(fail_at=3), np.zeros(2), s.tasks[t.task_id][1], 220),
                                  spec=s, client_factory=lambda _: client, run_episode_fn=native_kernel())
    with pytest.raises(RuntimeError, match='native env step'):
        runner.run(t, lambda *a: None)
    assert client.ends[-1]['execution']['executed_count'] == 2


@pytest.mark.parametrize('commit', [10, 15, 32])
def test_real_framework_extended_commit_native_kernel_and_terminal_capture(tmp_path, commit):
    from exp.r17_integration.common.fake_adapter import FakeAdapter, fake_spec, fake_sampler
    from exp.r17_integration.common.provider import AdapterProvider, register_provider, binding_for
    from exp.r17_integration.common.contracts import FeatureSpec
    from exp.r17_integration.models.pi05.provider import config_document, make_connection_factory
    from exp.r17_integration.common.config import config_from_dict, build_shared_storage
    from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
    from exp.r17_integration.tests.models.pi05.test_pi05_fix_round1 import server_for
    s = spec()
    model = replace(fake_spec(horizon=32, action_dim=7, views=2), allowed_benchmark_protocols=(s.identity,))
    # Native robot observation is state8, independent of the action7 geometry.
    model = replace(model, feature_spec=replace(model.feature_spec, fields={**model.feature_spec.fields, 'robot_state': 8}))
    class LiberoAdapter(FakeAdapter):
        def extract_state_only(self, obs):
            return super().extract_state_only({'state': obs.get('state', obs.get('observation/state'))})
        def prepare(self, obs, request):
            return super().prepare({**obs, 'state': obs['observation/state']}, request)
    resident = LiberoAdapter(model)
    provider = AdapterProvider('p2_libero_core_' + tmp_path.name, model, lambda: resident, fake_sampler)
    register_provider(provider)
    doc = config_document(provider, k=2, commit=15, evidence_dir=tmp_path/'miss')
    doc['keys']['vision_1'] = {'enabled': True}
    doc['adapter_binding'] = binding_for(provider, s.identity, num_steps=2, commit_horizon=commit,
                                          allow_partial_commit=commit % 5 != 0)
    config = config_from_dict(doc)
    captures, acks = [], []
    class Sink:
        def __call__(self, record):
            captures.append(record)
        def on_execution(self, ack):
            acks.append(ack)
    factory = make_connection_factory(provider, config, build_shared_storage(config), capture_sink=Sink())
    with server_for(factory, provider) as (port, server):
        environment = Env()
        t = replace(task(s), server_host='127.0.0.1', server_port=port, bundle_id='default',
                    extra={**task(s).extra, 'bundle_generation': 'default'})
        runner = R17LiberoEpisodeRunner(args(s), lambda _: (environment, np.array([7., 8.]), s.tasks[t.task_id][1], 220),
                                      spec=s, run_episode_fn=native_kernel())
        try:
            result = runner.run(t, lambda *a: None)
            assert result.n_steps == 34 and environment.closed
            assert sum(r['executed_count'] for r in acks) == 34
            assert acks[-1]['terminal'] and acks[-1]['request'].original_init_idx == 7
            assert acks[-1]['request'].task_id == 13 and acks[-1]['request'].task_name == s.tasks[13][0]
            expected_misses = {10: 4, 15: 3, 32: 2}[commit]
            assert resident.counts['stage1'] == resident.counts['stage2'] == expected_misses
            assert resident.counts['nfe'] == expected_misses * 2
            assert len(captures) == len(acks)
            costs = [r for r in result.per_step_rows if not r.get('_kind')]
            assert all(r['cost_ms'] >= 0 and r['server_timing']['infer_ms'] >= 0 for r in costs)
            assert all(r['capture_enabled'] and not r['pricing_eligible'] for r in costs)
        finally:
            runner.close()


def test_library_mapping_never_merges_suite_local_task_ids(tmp_path):
    from exp.r17_integration.common.fake_adapter import fake_spec
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.search.store import write_store
    from exp.r17_integration.search.fit import library_data
    s = spec()
    model = replace(fake_spec(horizon=32, action_dim=7, views=2), allowed_benchmark_protocols=(s.identity,))
    rows = [{'campaign': 'cpu_library', 'arm': 'teacher', 'bundle_generation': 'teacher',
             'benchmark_protocol': s.identity, 'task_name': s.tasks[tid][0], 'orig_init_idx': 10,
             'task_data_hash': f'fixture_task_{tid}', 'task_uid': f'task_{tid}_10', 'attempt': 1,
             'decision_seq': 0, 'split': 'library'} for tid in (3, 13, 23, 33)]
    arrays = {'key_vision_0': np.arange(16, dtype=np.float32).reshape(4, 4),
              'key_vision_1': np.arange(16, dtype=np.float32).reshape(4, 4)+1,
              'state': np.ones((4, 8), np.float32), 'normalized_actions': np.ones((4, 32, 7), np.float32)}
    store = write_store(tmp_path/'teacher', model_binding(model, s.identity), rows, arrays)
    data = library_data(store, task_names=[r[0] for r in s.tasks])
    assert data.task_id.tolist() == [3, 13, 23, 33]
    assert len(set(data.episode.tolist())) == 4
    assert set(data.features) == {'vision_0', 'vision_1'}


def test_libero_start_failure_closes_created_native_environment():
    s, environment, client = spec(), Env(), TailClient(32)
    t = task(s)
    client.episode_start = lambda **kw: (_ for _ in ()).throw(ConnectionError('fixture start failure'))
    runner = R17LiberoEpisodeRunner(args(s), lambda _: (environment, np.zeros(2), s.tasks[t.task_id][1], 220),
                                  spec=s, client_factory=lambda _: client, run_episode_fn=native_kernel())
    with pytest.raises(ConnectionError, match='start failure'):
        runner.run(t, lambda *a: None)
    assert environment.closed
