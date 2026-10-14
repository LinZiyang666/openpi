"""Manual GPU parity and native stage timings; never launches a campaign.

Run once per model in its existing Python island. Descriptors/inputs are read
only. Same real observation, distinct private seeds, B1 versus queued B8.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch


def array(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def load(args):
    descriptor = args.descriptor or Path('/data/mw_finetune/eval') / (
        'pimk1_ful/assets/load.json' if args.model == 'pi05' else 'grmk1_red/assets/load.json')
    desc = json.loads(descriptor.read_text())
    if args.model == 'pi05':
        from exp.r17_integration.models.pi05.adapter import load_adapter
        from exp.r17_integration.models.pi05.provider import register_provider
        desc['device'] = args.device
        adapter = load_adapter(desc)
        return adapter, register_provider(adapter)
    vendor = args.vendor_root.resolve()
    sys.path.insert(0, str(vendor))
    # The saved GR00T descriptor names the H100 checkout. Select the local
    # equivalents in memory; never modify the campaign's descriptor.
    for key in ('code_files', 'tokenizer_files'):
        desc[key] = [str(vendor / p.split('/gr00t_n15/', 1)[1])
                     if not Path(p).exists() and '/gr00t_n15/' in p else p for p in desc[key]]
    from exp.r17_integration.models.groot.loader import load_checkpoint
    from exp.r17_integration.models.groot.provider import GrootMetaWorldAdapter, make_provider
    from exp.r17_integration.qualification.model_runtime import GrootAliases
    backend, manifest = load_checkpoint(desc['checkpoint'], device=args.device,
        denoising_steps=desc['live_k'], tokenizer_files=desc['tokenizer_files'], code_files=desc['code_files'],
        benchmark=desc.get('benchmark', 'metaworld'), benchmark_protocol=desc.get('benchmark_protocol'))
    from dataclasses import replace
    provider = make_provider(backend, manifest)
    original = provider.load_adapter
    provider = replace(provider, load_adapter=lambda: GrootAliases(original()))
    return GrootAliases(GrootMetaWorldAdapter(backend, manifest)), provider


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('pi05', 'groot'), required=True)
    p.add_argument('--descriptor', type=Path)
    p.add_argument('--observation', type=Path,
        default=Path('exp/r17_integration/profile/inputs_mt50/assembly_seed8_idx0.npz'))
    p.add_argument('--vendor-root', type=Path,
        default=Path('/home/weiland/projects/openpi_ext/third_party/gr00t_n15'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, help='pi05 K; GR00T must equal descriptor live_k')
    p.add_argument('--iterations', type=int, default=5)
    args = p.parse_args()
    from exp.r17_integration.common.batch_runtime import BatchedAdapter, create_runtime
    from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
    from exp.r17_integration.common.request_rng import RequestRNG
    from openpi.serving.batching_core import Stage3MissPayload
    adapter, provider = load(args)
    sampler = provider.sampler_for_steps(args.steps or adapter.metadata().default_sampler.num_steps)
    with np.load(args.observation, allow_pickle=True) as data:
        obs = {k: data[k][0] for k in ('observation/image', 'observation/state')}
        obs['prompt'] = str(data['prompt'][0])
    requests = [RequestContext('gpu_batch_parity', adapter.metadata().allowed_benchmark_protocols[0],
        'assembly-v3', i, 0, task_id=0, explicit_seed=195+i) for i in range(16)]
    capture = CaptureSpec(enabled=True, snapshots=False)
    def decision(client, i, barrier=None):
        prepared = client.prepare(obs, requests[i])
        if barrier: barrier.wait(timeout=300)
        s1 = client.run_stage1(prepared)
        keys = client.retrieval_fields(s1)
        if barrier: barrier.wait(timeout=300)
        s2 = client.run_stage2(s1)
        if barrier: barrier.wait(timeout=300)
        result = client.sample(s2, sampler, RequestRNG(requests[i]), capture)
        wire = client.to_wire(result.normalized_chunk, prepared)
        return result, keys, wire.actions
    references = [decision(provider.load_adapter(), i) for i in range(16)]
    runtime = create_runtime(adapter, 8, 50, device=args.device)
    try:
        clients = [BatchedAdapter(provider.load_adapter(), runtime) for _ in range(8)]
        barrier = threading.Barrier(8)
        with ThreadPoolExecutor(8) as pool:
            replies = list(pool.map(lambda i: decision(clients[i], i, barrier), range(8)))
        noise_equal = all(np.array_equal(array(ref[0].capture['initial_noise']), array(reply[0].capture['initial_noise']))
                          for ref, reply in zip(references, replies))
        print(json.dumps(dict(model=args.model, K=sampler.num_steps, B=8, noise_equal=noise_equal,
            max_action_delta=max(float(np.max(np.abs(array(a[0].normalized_chunk)-array(b[0].normalized_chunk))))
                                 for a,b in zip(references,replies)),
            max_wire_action_delta=max(float(np.max(np.abs(array(a[2])-array(b[2])))) for a,b in zip(references,replies)),
            max_key_delta={k:max(float(np.max(np.abs(array(a[1][k])-array(b[1][k]))))
                                for a,b in zip(references,replies)) for k in replies[0][1]},
            actual_stage_batch_sizes=[c.last_batch_sizes for c in clients])), flush=True)
        assert noise_equal, 'per-request native noise changed'
    finally:
        runtime.close()

    batcher = adapter.make_stage_batcher()
    prepared = [adapter.prepare(obs, request) for request in requests]
    inputs = [adapter.stage1_batch_input(p) for p in prepared]
    def timed(fn):
        fn()
        torch.cuda.synchronize(args.device)
        times = []
        for _ in range(args.iterations):
            begin = time.perf_counter()
            fn()
            torch.cuda.synchronize(args.device)
            times.append(1000*(time.perf_counter()-begin))
        return float(np.median(times))
    with torch.no_grad():
        for size in (1,2,4,8,16):
            s1 = batcher.run_stage1_batch(inputs[:size])
            s2 = batcher.run_stage2_batch(s1, capture=False)
            if args.model == 'groot':
                from openpi.cache.groot.batcher import stage3_input
                conditioning = [stage3_input(adapter.runner, value) for value in s2]
                points = None
            else:
                conditioning, points = s2, ()
            payloads = [Stage3MissPayload(value,
                torch.as_tensor(references[i][0].capture['initial_noise'], device=args.device), sampler.num_steps, points)
                for i,value in enumerate(conditioning)]
            print(json.dumps(dict(model=args.model, K=sampler.num_steps, B=size,
                timing_scope='native StageBatcher host wall, synchronized; no queue wait',
                stage1_ms=timed(lambda:batcher.run_stage1_batch(inputs[:size])),
                stage2_ms=timed(lambda:batcher.run_stage2_batch(s1,capture=False)),
                stage3_ms=timed(lambda:batcher.run_stage3_miss(payloads,num_steps=sampler.num_steps,
                    save_timesteps_per_request=[points]*size)))), flush=True)


if __name__ == '__main__':
    main()
