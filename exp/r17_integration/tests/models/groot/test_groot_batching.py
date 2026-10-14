"""CPU all-stage batching, suite routing and native per-request RNG."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import numpy as np
import pytest
import torch

from exp.r17_integration.common.batch_runtime import BatchedAdapter, create_runtime
from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.models.groot.provider import GrootMetaWorldAdapter, make_provider
from exp.r17_integration.qualification.model_runtime import GrootAliases
from .test_libero import LiberoBackend, libero_manifest, observation


@pytest.mark.parametrize('libero', [False, True])
def test_all_stages_private_noise_capture_warm_and_suites(guarded_stub, manifest, obs, context, libero):
    backend = guarded_stub()
    if libero:
        from exp.r17_integration.ops.formal_eval import benchmark
        manifest = libero_manifest(manifest)
        backend = {'libero_spatial':LiberoBackend(), 'libero_10':LiberoBackend()}
        observations = [observation() for _ in range(4)]
        b = benchmark(True)
        contexts = [RequestContext('b', b.identity, b.ordered_tasks[0 if i<2 else 30], i, 0,
                                  task_id=0 if i<2 else 30, explicit_seed=140+i) for i in range(4)]
    else:
        observations = [dict(obs) for _ in range(4)]
        contexts = [replace(context, original_init_idx=i, explicit_seed=140+i) for i in range(4)]
    for i, wire in enumerate(observations):
        key = 'observation/image' if libero else 'image'
        wire[key] = np.full_like(wire[key], 3+20*i)
    adapter = GrootMetaWorldAdapter(backend, manifest)
    provider = make_provider(backend, manifest)
    sampler = adapter.metadata().default_sampler
    def run(client, i, barrier=None):
        prepared = client.prepare(observations[i], contexts[i])
        if barrier: barrier.wait(timeout=15)
        s1 = client.run_stage1(prepared)
        keys = client.retrieval_fields(s1)
        if barrier: barrier.wait(timeout=15)
        s2 = client.run_stage2(s1, capture=True)
        if barrier: barrier.wait(timeout=15)
        result = client.sample(s2, sampler, RequestRNG(contexts[i]), CaptureSpec(True, True))
        captured = client.capture_fields(s1, s2, result)
        if barrier: barrier.wait(timeout=15)
        warm = client.resume(s2, result.snapshots[0], sampler, RequestRNG(contexts[i]), CaptureSpec(True, False))
        return keys, result, captured, warm
    references = [run(adapter, i) for i in range(4)]
    runtime = create_runtime(adapter, 4, 50, device='cpu')
    clients = [BatchedAdapter(GrootAliases(provider.load_adapter()), runtime) for _ in range(4)]
    try:
        barrier = threading.Barrier(4)
        with ThreadPoolExecutor(4) as pool:
            replies = list(pool.map(lambda i:run(clients[i], i, barrier), range(4)))
        for i, (ref, reply) in enumerate(zip(references, replies, strict=True)):
            keys, result, captured, warm = reply
            for name in keys:
                assert torch.equal(keys[name], ref[0][name])
            assert torch.equal(result.capture['initial_noise'], ref[1].capture['initial_noise'])
            assert result.rng_receipt == ref[1].rng_receipt
            assert torch.equal(result.normalized_chunk, ref[1].normalized_chunk)
            assert torch.equal(warm.normalized_chunk, ref[3].normalized_chunk)
            for name in captured:
                assert torch.equal(captured[name], ref[2][name])
            for actual, expected in zip(result.snapshots, ref[1].snapshots, strict=True):
                assert actual.loop_index == expected.loop_index and torch.equal(actual.x_t, expected.x_t)
            n = 2 if libero else 4
            assert clients[i].last_batch_sizes == dict(stage1=n, stage2=n, stage3=n)
    finally:
        runtime.close()


def test_early_stack_split_and_different_prompt_buckets(guarded_stub, manifest, obs, context):
    adapter = GrootMetaWorldAdapter(guarded_stub(), manifest)
    prepared = [adapter.prepare(dict(obs, prompt=p), context).data for p in ('reach', 'reach', 'longer task')]
    batcher = adapter.make_stage_batcher()
    calls = []
    original = adapter.runner.run_stage1
    def run(data):
        calls.append(data['eagle_input_ids'].shape[0])
        return original(data)
    adapter.runner.run_stage1 = run
    s1 = batcher.run_stage1_batch(prepared)
    assert calls == [2, 1]
    s2 = batcher.run_stage2_batch(s1, capture=False)
    for native, p, features in zip(s1, prepared, s2, strict=True):
        with adapter.runner.session():
            ref = original(p)
            ref2 = adapter.runner.run_stage2_llm(ref)
        assert torch.equal(native.input_embeds, ref.input_embeds)
        assert torch.equal(native.state, ref.state)
        assert torch.equal(features.backbone_features, ref2.backbone_features)
    from openpi.cache.groot.batcher import stage3_input
    from openpi.serving.batching_core import Stage3MissPayload, Stage3WarmStartPayload
    a = Stage3MissPayload(stage3_input(adapter.runner, s2[0]), torch.zeros(16,32), 4)
    b = replace(a, stage2_out=stage3_input(adapter.runner, s2[2]))
    assert batcher.bucket_key(a) != batcher.bucket_key(b)
    assert batcher.bucket_key(a) != batcher.bucket_key(replace(a, num_steps=2))
    assert batcher.bucket_key(a) != batcher.bucket_key(Stage3WarmStartPayload(a.stage2_out, a.noise, .25, 4))
