"""CPU native stack/split, private noise, capture and warm continuation."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import numpy as np
import pytest
import torch

from exp.r17_integration.common.batch_runtime import BatchedAdapter, create_runtime
from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.models.pi05.adapter import Pi05Adapter, sampler_for_steps
from exp.r17_integration.tests.models.pi05 import test_pi05, test_pi05_libero


def batch_fixture(libero=False):
    fixtures = test_pi05_libero if libero else test_pi05
    adapter, model = fixtures.fixture_adapter()
    original = model.run_stage1
    def stage1(obs):
        import jax
        from openpi.serving.stage_io import stack_stage1_output
        # Only the neural tower is a fixture; use the real native stack/split.
        return stack_stage1_output([original(jax.tree.map(lambda x: x[i:i+1], obs))
                                    for i in range(obs.state.shape[0])])
    model.run_stage1 = stage1
    def stage2(s1):
        from openpi.models_pytorch.pi0_pytorch import Stage2Output
        return Stage2Output(s1, (), s1.prefix_embs.clone())
    model.run_stage2 = model.run_stage2_capture = stage2
    return adapter, fixtures.observation()


@pytest.mark.parametrize('libero', [False, True])
def test_batched_native_rows_rng_capture_and_warm(libero):
    adapter, obs = batch_fixture(libero)
    n, sampler = 3, sampler_for_steps(2)
    protocol = adapter.metadata().allowed_benchmark_protocols[0]
    from exp.r17_integration.common.contracts import RequestContext
    requests = [RequestContext('batch', protocol, 'assembly-v3', i, 0, explicit_seed=130+i) for i in range(n)]
    observations = [dict(obs, **{'observation/state':obs['observation/state']+i*.1,
                                'observation/image':np.full_like(obs['observation/image'], 10+30*i)}) for i in range(n)]
    def run(client, i, barrier=None):
        prepared = client.prepare(observations[i], requests[i])
        if barrier: barrier.wait(timeout=15)
        s1 = client.run_stage1(prepared)
        keys = client.retrieval_fields(s1)
        if barrier: barrier.wait(timeout=15)
        s2 = client.run_stage2(s1, capture=True)
        if barrier: barrier.wait(timeout=15)
        result = client.sample(s2, sampler, RequestRNG(requests[i]), CaptureSpec(True, True))
        capture = client.capture_fields(s1, s2, result)
        if barrier: barrier.wait(timeout=15)
        warm = client.resume(s2, result.snapshots[0], sampler, RequestRNG(requests[i]))
        return keys, result, capture, warm
    references = [run(adapter, i) for i in range(n)]
    runtime = create_runtime(adapter, n, 50, device='cpu')
    clients = [BatchedAdapter(Pi05Adapter(adapter.policy, adapter.spec), runtime) for _ in range(n)]
    try:
        barrier = threading.Barrier(n)
        with ThreadPoolExecutor(n) as pool:
            replies = list(pool.map(lambda i:run(clients[i], i, barrier), range(n)))
        for i, (reference, reply) in enumerate(zip(references, replies, strict=True)):
            keys, result, capture, warm = reply
            for name in keys:
                assert torch.equal(keys[name], reference[0][name])
            np.testing.assert_array_equal(result.capture['initial_noise'], RequestRNG(requests[i]).normal((adapter.spec.layout.horizon, 32)))
            np.testing.assert_array_equal(result.normalized_chunk, reference[1].normalized_chunk)
            np.testing.assert_array_equal(warm.normalized_chunk, reference[3].normalized_chunk)
            assert result.rng_receipt == reference[1].rng_receipt
            for name in capture:
                np.testing.assert_array_equal(capture[name], reference[2][name])
            assert clients[i].last_batch_sizes == dict(stage1=n, stage2=n, stage3=n)
        # Rows remain independent after split, including stage2 capture output.
        replies[0][2]['prefix_out'].fill(-999)
        assert np.all(replies[1][2]['prefix_out'] != -999)
    finally:
        runtime.close()


def test_runtime_bucket_keys_include_schedule_shape_and_resume():
    from openpi.serving.batching_core import Stage3MissPayload, Stage3WarmStartPayload
    adapter, obs = batch_fixture()
    from exp.r17_integration.common.contracts import RequestContext
    r = RequestContext('b', adapter.spec.allowed_benchmark_protocols[0], 'assembly-v3', 0, 0)
    s1 = adapter.run_stage1(adapter.prepare(obs, r))
    s2 = adapter.run_stage2(s1)
    runtime = create_runtime(adapter, 2, 0, device='cpu')
    client = BatchedAdapter(Pi05Adapter(adapter.policy, adapter.spec), runtime)
    try:
        item = client._input(s2.data)
        a = Stage3MissPayload(item, torch.zeros(5,32), 2, ())
        key = runtime.core._batcher.bucket_key
        assert key(a) != key(replace(a, num_steps=10))
        assert key(a) != key(Stage3WarmStartPayload(item, a.noise, .5, 2))
        assert key(a) != key(replace(a, noise=a.noise.double()))
    finally:
        runtime.close()


@pytest.mark.parametrize('capture', [False, True])
def test_formal_resident_serving_rows_capture_and_yaml_switch(tmp_path, monkeypatch, capture):
    from dataclasses import asdict
    import json
    from types import SimpleNamespace
    import uuid
    import yaml
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    from exp.r17_integration.common import websocket_policy_server as ws
    from exp.r17_integration.common.config import build_shared_storage, load_cache_config
    from exp.r17_integration.models.pi05.provider import register_provider
    from exp.r17_integration.ops import formal_eval, formal_server
    from exp.r17_integration.qualification import model_runtime
    adapter, obs = batch_fixture()
    provider = register_provider(adapter, provider_id='pi_batch_'+uuid.uuid4().hex)
    campaign = dict(model='pi05',domain='metaworld',assets=str(tmp_path),device='cpu',
        memory_budget_gib=10,replicas=2,server_node='local',library=capture,
        stage_batching=dict(max_batch=2,batch_wait_ms=50))
    cfg = formal_eval.full_config(provider,campaign)
    (tmp_path/'full.yaml').write_text(yaml.safe_dump(asdict(cfg)))
    (tmp_path/'benchmark.json').write_text(json.dumps(asdict(FORMAL_SPEC)))
    switched = formal_eval.full_config(provider,dict(campaign,library=False,denoising_steps=2))
    (tmp_path/'k2.yaml').write_text(yaml.safe_dump(asdict(switched)))
    switched = load_cache_config(tmp_path/'k2.yaml')
    import openpi.serving.websocket_policy_server as stock_ws
    bundle = SimpleNamespace(cache_config=switched,shared_storage=build_shared_storage(switched),yaml_id='k2')
    monkeypatch.setattr(stock_ws,'get_current_cache_bundle',lambda bundle_id:bundle)
    monkeypatch.setattr(model_runtime,'load',lambda *args:(adapter,provider))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *args:SimpleNamespace(total_memory=20*2**30))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    records = []
    class Sink:
        def bind_observation(self, observation): self.observation = observation
        def __call__(self, record): records.append(record)
    class Captures:
        def __call__(self, connection): return Sink()
        def close(self): return []
    from exp.r17_integration.debug import library_capture
    monkeypatch.setattr(library_capture,'campaign_capture_factory',lambda *args:Captures())
    responses = []
    class Server:
        def __init__(self, policy, **kwargs): self.kwargs = kwargs
        def request_stop(self): pass
        def serve_forever(self, **kwargs):
            assert self.kwargs['metadata']['resident_execution'] == 'cross-connection native stage batches'
            clients = [self.kwargs['connection_policy_factory'](bundle_id='k2') for _ in range(2)]
            for i, client in enumerate(clients):
                client.on_episode_start(extra_metadata=dict(campaign='cpu',task_name='assembly-v3',task_id=0,
                    original_init_idx=i,seed=7,explicit_seed=190+i,ack_protocol='r17.execution.v1'))
            with ThreadPoolExecutor(2) as pool:
                responses.extend(pool.map(lambda i:clients[i].infer(obs),range(2)))
            for client in clients:
                client.on_episode_end(execution=dict(decision_seq=0,executed_count=5,terminal=True))
    monkeypatch.setattr(ws,'WebsocketPolicyServer',Server)
    formal_server.serve(campaign,23180,tmp_path/'server')
    assert all(r['__hit_meta__']['actual_nfe']==2 for r in responses)
    rows = [json.loads(line) for line in (tmp_path/'server/serving.jsonl').read_text().splitlines()]
    assert len(rows)==2 and all(row['batching']['stages']['stage3']['size']==2 for row in rows)
    assert len(records)==(2 if capture else 0)
    assert (tmp_path/'server/CAPTURE_CLOSED.json').exists() is capture


def test_library_hits_and_periodic_miss_keep_native_stage_counts(tmp_path):
    import uuid
    import yaml
    from exp.r17_integration.common.config import build_shared_storage, load_cache_config
    from exp.r17_integration.common.serving import make_connection_factory
    from exp.r17_integration.models.pi05.provider import config_document, register_provider
    from exp.r17_integration.tests.models.pi05.test_pi05_e2e import tiny_store
    from exp.metaworld.tasks import task_id_of
    adapter, obs = batch_fixture()
    provider = register_provider(adapter, provider_id='pi_batch_library_'+uuid.uuid4().hex)
    _, recipe = tiny_store(adapter,tmp_path/'library')
    path = tmp_path/'library.yaml'
    path.write_text(yaml.safe_dump(config_document(provider,k=2,evidence_dir=tmp_path/'miss',recipe=recipe)))
    cfg = load_cache_config(path)
    runtime = create_runtime(adapter,2,50,device='cpu')
    try:
        factory = runtime.wrap_connection_factory(make_connection_factory(provider,cfg,build_shared_storage(cfg),
            adapter_factory=lambda:BatchedAdapter(provider.load_adapter(),runtime)))
        clients = [factory() for _ in range(2)]
        for i, client in enumerate(clients):
            client.on_episode_start(extra_metadata=dict(campaign='batch_library',task_name='reach-v3',task_id=task_id_of('reach-v3'),
                original_init_idx=i,seed=7,explicit_seed=200+i,ack_protocol='r17.execution.v1'))
        rows = []
        for seq in range(4):
            with ThreadPoolExecutor(2) as pool:
                results = list(pool.map(lambda c:c.infer(dict(obs,**({'__execution__':dict(decision_seq=seq-1,executed_count=5)}
                                                      if seq else {}))),clients))
            rows.extend(r['__hit_meta__'] for r in results)
        assert {'library','policy'} <= {row['source'] for row in rows}
        for row in rows:
            stages = row['batching']['stages']
            if row['source'] == 'library':
                assert set(stages)=={'stage1'} and row['actual_nfe']==0
            else:
                assert set(stages)=={'stage1','stage2','stage3'} and row['actual_nfe']==2
    finally:
        runtime.close()
