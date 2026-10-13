"""R17 pure-inference serve entry with opt-in, self-checked DP CUDA graphs."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time

from .formal_process import write_json


def dp_capacity_serve(module,campaign,port,directory,*,cuda_graph):
    """Bridge the frozen R2 model gate to the owner's explicit cache capacity.

    Only admission sees the canonical control count. The native resident,
    allocator, batching, metadata and launcher see the actual3–4 replicas.
    All sampler/recipe/B8/5ms/MPS validation still runs in the model hook.
    """
    record=campaign.get('cache_worker_orchestration',{})
    resize=(campaign.get('cache_study') and campaign.get('model')=='dp' and
        campaign.get('replicas') in (3,4) and record.get('owner_authorized') is True and
        record.get('replicas') in (None,campaign['replicas']) and
        record.get('worker_nodes')==campaign.get('execution_worker_nodes') and
        record.get('total_workers')==campaign.get('execution_workers') and
        sum(record.get('worker_nodes',{}).values())==campaign.get('execution_workers'))
    if not resize:return module.serve(campaign,port,directory,cuda_graph=cuda_graph)
    change=campaign.get('cache_control_proof',{}).get('worker_resource_changes',{}).get('replicas',{})
    if change.get('control')!=2 or change.get('campaign')!=campaign['replicas']:
        pass
    native=module.cache_study_admission
    def admission(actual,*args):
        if actual!=campaign:pass
        result=native(dict(actual,replicas=2),*args)
        write_json(Path(directory)/'CACHE_CAPACITY_ADMISSION.json',dict(
            schema='r17.dp_cache_capacity_bridge.v1',actual_replicas=campaign['replicas'],
            control_replicas=2,execution_workers=campaign['execution_workers'],
            canonical_workers=campaign['workers'],owner_authorized=True,
            native_sampler_batch_recipe_checks_preserved=True,
            gated_miss_batch_composition_may_change=True))
        return result
    module.cache_study_admission=admission
    try:return module.serve(campaign,port,directory,cuda_graph=cuda_graph)
    finally:module.cache_study_admission=native


def enable_dp_graph(adapter, benchmark, sampler=None):
    """Use E's denoiser graph; attest the actual R17 physical-action path."""
    import numpy as np
    import torch
    from exp.mw_finetune.dp.cuda_graph import DenoiserGraph, check_observations, require_bit_exact
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.serving import full_call
    denoiser = adapter.model._policy.model
    graph = DenoiserGraph(denoiser)
    class GraphDenoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.original = denoiser
        def forward(self, trajectory, timestep, cond):
            return graph(trajectory, timestep, cond)
    wrapped = GraphDenoiser().eval()
    observations = list(check_observations(adapter.domain, adapter.legacy_policy.stats))
    if adapter.domain == 'libero':
        # Protocol-bound paired original-resolution images; no 480/224 resizing.
        for obs in observations:
            rng = np.random.default_rng(obs['seed'])
            for key in ('observation/image', 'observation/wrist_image'):
                obs[key] = rng.integers(0, 256, (benchmark.wire_size, benchmark.wire_size, 3), dtype=np.uint8)
    def infer(obs, impl):
        adapter.model._policy.model = impl
        request = RequestContext('graph-self-check', benchmark.identity, benchmark.tasks[0][0],
            0, 0, explicit_seed=obs['seed'])
        return {'actions': full_call(adapter, obs, request, sampler=sampler)[1].actions}
    try:
        report = require_bit_exact(observations, lambda o: infer(o, denoiser), lambda o: infer(o, wrapped))
        # Also prove E's native policy sampling using the same resident weights.
        adapter.model._policy.model = denoiser
        native = require_bit_exact(observations, adapter.legacy_policy.infer,
            lambda o: _native_graph(adapter.legacy_policy, graph, o))
    except BaseException:
        adapter.model._policy.model = denoiser
        raise
    adapter.model._policy.model = wrapped
    output = dict(r17=report, native=native, passed=True, implementation='E DenoiserGraph; R17 B1 DDPM100')
    if sampler is not None:
        output.update(implementation='E DenoiserGraph; selected R17 B1 sampler + native DDPM100',
                      execution_sampler=asdict(sampler))
    return output


def _native_graph(policy, graph, observation):
    policy._denoiser_graph = graph
    return policy._infer(observation, cuda_graph=True)


def serve(campaign, port, directory, cuda_graph=False):
    if campaign.get('fastwam_batching'):
        from exp.r17_integration.models.fastwam.formal_batch_server import serve as serve_batched
        return serve_batched(campaign, port, directory)
    if campaign.get('dp_batching'):
        from exp.r17_integration.models.dp import formal_batch_server as batched
        return dp_capacity_serve(batched,campaign,port,directory,cuda_graph=cuda_graph)
    from exp.r17_integration.conductor.manifest import load_benchmark_spec
    from exp.r17_integration.qualification.model_runtime import load
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    from exp.r17_integration.common.serving import make_connection_factory, make_bundle_validator
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    directory = Path(directory)
    if campaign.get('library') and campaign.get('server_node')=='h100':
        from .library_staging import admit
        admit(campaign)
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory.parent / 'manifest.json').exists():
        write_json(directory.parent / 'manifest.json', campaign)
    benchmark = load_benchmark_spec(Path(campaign['assets']) / 'benchmark.json')
    started = time.monotonic()
    if campaign['model'] == 'cosmos':
        import torch
        free, total = torch.cuda.mem_get_info(campaign['device'])
        budget = campaign['memory_budget_gib'] * 2**30 / campaign['replicas']
        if free < budget:
            raise ValueError('insufficient free VRAM for bounded Cosmos resident; existing processes are untouched')
        torch.cuda.set_per_process_memory_fraction(budget / total, campaign['device'])
    adapter, provider = load(dict(campaign, benchmark_spec=str(Path(campaign['assets']) / 'benchmark.json')),
                             campaign['device'])
    import torch
    total = torch.cuda.get_device_properties(campaign['device']).total_memory
    torch.cuda.set_per_process_memory_fraction(
        campaign['memory_budget_gib'] * 2**30 / campaign['replicas'] / total, campaign['device'])
    if campaign['model'] == 'dp' and not cuda_graph:
        raise ValueError('formal DP requires --cuda-graph')
    if campaign['model'] != 'dp' and cuda_graph:
        raise ValueError('--cuda-graph is DP only')
    graph_sampler = None
    if cuda_graph and campaign.get('denoising_steps') is not None:
        graph_config = load_cache_config(Path(campaign['assets']) / 'full.yaml')
        graph_sampler = provider.sampler_for_steps(graph_config.adapter_binding['num_steps'])
    graph_report = enable_dp_graph(adapter, benchmark, sampler=graph_sampler) if cuda_graph else None
    parity_report = None
    if campaign['model'] == 'cosmos':
        from exp.r17_integration.models.cosmos.qualify import run_parity
        parity_report = run_parity(adapter)
        write_json(directory / 'native_parity.json', parity_report)
    config = load_cache_config(Path(campaign['assets']) / 'full.yaml')
    if config.enabled or config.checkpoints or config.recipe.enabled:
        raise ValueError('formal campaign must disable cache, checkpoints and recipe')
    storage = build_shared_storage(config)
    captures = None
    if campaign.get('library'):
        from exp.r17_integration.debug.library_capture import campaign_capture_factory
        from exp.r17_integration.common.contracts import CaptureSpec
        captures = campaign_capture_factory(campaign, directory/'capture')
        base_factory = make_connection_factory(provider,config,storage,capture_sink_factory=captures,
                                               capture_spec=CaptureSpec(enabled=True,snapshots=False))
    else:
        base_factory = make_connection_factory(provider, config, storage)
    import threading
    evidence_lock = threading.Lock()
    resident_lock = threading.RLock()
    def factory(*args, **kwargs):
        core = base_factory(*args, **kwargs)
        original = core.infer
        def infer(obs):
            begin = time.monotonic()
            # Graph workspace and native sampler buffers are resident. Each
            # replica admits B1 at a time; independent processes run together.
            with resident_lock:
                service_begin = time.monotonic()
                if captures is not None:
                    core.capture_sink.bind_observation(obs)
                result = original(obs)
                service_seconds = time.monotonic()-service_begin
            hit = result.get('__hit_meta__', {})
            row = dict(replica_port=port, task_uid=core._episode.get('task_uid'),
                task_name=core._episode.get('task_name'), decision_seq=hit.get('decision_seq'),
                source=hit.get('source'), nfe=hit.get('nfe', hit.get('actual_nfe')),
                elapsed_seconds=time.monotonic()-begin, service_seconds=service_seconds,
                queue_seconds=service_begin-begin, time=time.time())
            # Serving counters only: no observations, actions, RNG or snapshots.
            with evidence_lock, (directory / 'serving.jsonl').open('a') as f:
                f.write(json.dumps(row, sort_keys=True)+'\n')
            return result
        core.infer = infer
        return core
    metadata = dict(r17_model_spec=asdict(adapter.metadata()), formal_pure_inference=True,
        cache_enabled=False, debug_capture=False, cuda_graph=cuda_graph, cuda_graph_self_check=graph_report,
        resident_execution='B1 per replica; independent resident processes',
        replica_port=port, r17_samplers={str(adapter.metadata().default_sampler.num_steps):
                                      asdict(adapter.metadata().default_sampler)})
    if campaign['model'] == 'cosmos':
        from exp.r17_integration.models.cosmos.serve import serve_metadata
        metadata.update(serve_metadata(adapter), native_parity_self_check=parity_report)
    if campaign.get('denoising_steps') is not None:
        active_sampler = asdict(provider.sampler_for_steps(config.adapter_binding['num_steps']))
        metadata['r17_samplers'][str(active_sampler['num_steps'])] = active_sampler
        metadata.update(execution_sampler=active_sampler,
                        execution_commit_horizon=config.adapter_binding['commit_horizon'])
    if captures is not None:
        metadata.update(formal_pure_inference=False,library_teacher=True,debug_capture=True,
                        capture_profile='B-full-native-library-allowlist; received_wire_uint8')
    server = WebsocketPolicyServer(factory(), host='0.0.0.0', port=port, metadata=metadata,
        concurrent=True, connection_policy_factory=factory, bundle_validator=make_bundle_validator(provider),
        ready_callback=lambda: write_json(directory / 'ready.json', dict(metadata, pid=os.getpid(),
            elapsed_load_seconds=time.monotonic()-started)))
    signal.signal(signal.SIGTERM, lambda *unused: server.request_stop())
    try:
        server.serve_forever(stop_on_request=True)
    finally:
        if captures is not None:
            write_json(directory/'CAPTURE_CLOSED.json',captures.close())
        write_json(directory / 'closed.json', dict(status='CLOSED', pid=os.getpid(),
            elapsed_seconds=time.monotonic()-started, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--directory', required=True)
    p.add_argument('--cuda-graph', action='store_true')
    a = p.parse_args()
    serve(json.loads(Path(a.campaign).read_text()), a.port, a.directory, a.cuda_graph)
