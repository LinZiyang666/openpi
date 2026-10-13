"""Thin qualification composition over P-owned loaders, factories and writers."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path


class GrootAliases:
    """Map P2 canonical image/state keys into P5's documented native wire keys."""
    def __init__(self,adapter): self.adapter=adapter
    def __getattr__(self,key): return getattr(self.adapter,key)
    @staticmethod
    def wire(obs):
        result=dict(obs)
        for native,canonical in (('state','observation/state'),('image','observation/image')):
            if canonical in obs:
                if native in obs:
                    import numpy as np
                    if not np.array_equal(obs[native],obs[canonical]): raise ValueError('conflicting GR wire aliases: '+native)
                result[native]=obs[canonical]
        return result
    def prepare(self,obs,request): return self.adapter.prepare(self.wire(obs),request)
    def extract_state_only(self,obs): return self.adapter.extract_state_only(self.wire(obs))


def load(job, device='cuda:0'):
    from .benchmark_binding import benchmark_spec
    benchmark_spec(job)
    assets = Path(job['assets'])
    desc = json.loads((assets / 'load.json').read_text())
    model = job['model']
    if model == 'pi05':
        from exp.r17_integration.models.pi05.adapter import load_adapter
        from exp.r17_integration.models.pi05.provider import register_provider
        desc['device'] = device
        adapter = load_adapter(desc)
        provider = register_provider(adapter)
    elif model == 'groot':
        from exp.r17_integration.models.groot.loader import load_checkpoint
        from exp.r17_integration.models.groot.provider import GrootMetaWorldAdapter, make_provider, register_provider
        backend, manifest = load_checkpoint(desc['checkpoint'], device=device, denoising_steps=desc['live_k'],
            tokenizer_files=desc['tokenizer_files'], code_files=desc['code_files'])
        adapter = GrootMetaWorldAdapter(backend, manifest)
        provider = make_provider(backend, manifest)
        from dataclasses import replace
        original=provider.load_adapter
        provider=replace(provider,load_adapter=lambda:GrootAliases(original()))
        provider=register_provider(provider)
        adapter=GrootAliases(adapter)
    elif model == 'dp':
        from exp.r17_integration.models.dp.loader import load_adapter
        from exp.r17_integration.models.dp.provider import register_provider
        desc['device'] = device
        adapter = load_adapter(desc)
        provider = register_provider(adapter)
    elif model == 'fastwam':
        from exp.r17_integration.models.fastwam.loader import load_adapter, LoadSpec
        from exp.r17_integration.models.fastwam.provider import register_provider
        adapter = load_adapter(LoadSpec(**desc), device=device)
        provider = register_provider(adapter)
    elif model == 'cosmos':
        from exp.r17_integration.models.cosmos.loader import load_adapter
        from exp.r17_integration.models.cosmos.provider import register_provider
        adapter = load_adapter(desc, device=device)
        provider = register_provider(adapter)
    else:
        raise ValueError(model)
    from exp.r17_integration.debug.validate import canonical
    if canonical(asdict(adapter.metadata())) != canonical(json.loads((assets / 'model_spec.json').read_text())):
        pass
    return adapter, provider


def wire_observations(job):
    import numpy as np
    path = Path(job.get('observations_dir',job['assets'])) / ('fastwam_observations.npz' if job['model'] == 'fastwam' else 'wire_observations.npz')
    with np.load(path, allow_pickle=False) as data:
        obs = [{name: str(data[name][i]) if name == 'prompt' else data[name][i].copy()
                for name in data.files} for i in range(3)]
    if job['domain'] == 'libero' and job['model'] == 'dp':
        for row in obs:
            if 'observation/wrist_image' not in row:
                raise ValueError('recorded paired LIBERO wrist view is required')
    return obs


def make_capture_factory(provider, config, storage, directory, recipe_factory=None):
    # P2's core-aware callback also works with all P1 ConnectionPolicy adapters.
    from exp.r17_integration.models.pi05.provider import make_capture_sink_factory
    from exp.r17_integration.common.serving import make_connection_factory
    from exp.r17_integration.common.contracts import CaptureSpec
    capture = make_capture_sink_factory(directory)
    base = make_connection_factory(provider, config, storage, recipe_factory=recipe_factory)
    def factory(base_policy=None, bundle_id='default'):
        core = base(base_policy, bundle_id)
        if not bundle_id.startswith('plain_'):
            core.capture = CaptureSpec(True, True)
            from .capture_portable import PortableSink
            core.capture_sink = PortableSink(capture(core))
        return core
    return factory, capture


def serve(job, run, *, device='cuda:0', server_started=None):
    import signal
    import time
    from exp.r17_integration.common.serving import make_bundle_validator
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from exp.r17_integration.debug.validate import model_binding
    from .assets import write_json
    started = time.monotonic()
    adapter, provider = load(job,device)
    assets = Path(job['assets'])
    spec = adapter.metadata()
    config = load_cache_config(assets / ('k'+str(spec.default_sampler.num_steps)+'.yaml'))
    storage = build_shared_storage(config)
    from exp.r17_integration.search.store import Store
    from exp.r17_integration.debug.validate import digest
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    from exp.r17_integration.common.recipe import make_recipe_factory
    store=Store(assets/'synthetic_teacher_store',expected_binding=model_binding(spec,spec.allowed_benchmark_protocols[0]))
    # Two same-K GR bundles differ in recipe; other models also exercise K2.
    retrieval = run / 'retrieval.yaml'
    document = json.loads(json.dumps(asdict(load_cache_config(assets / ('k'+str(spec.default_sampler.num_steps if job['model']=='groot' else 2)+'.yaml')))))
    document['checkpoints']['cp1']['gate']['type'] = 'always_search'
    from .benchmark_binding import recipe_settings
    document['recipe'].update(enabled=True,store_path=str(assets/'synthetic_teacher_store'),
        store_digest=store.manifest['content_digest'],teacher_sampler_identity=digest(store.manifest['binding']['sampler']),
        corrector=True,k=2,kref=2,early=False,
        **recipe_settings(spec.allowed_benchmark_protocols[0]))
    import yaml
    retrieval.write_text(yaml.safe_dump(document))
    recipe_factory=make_recipe_factory(provider)
    full_doc = asdict(config)
    full_doc['checkpoints']['cp1']['gate']['type'] = 'always_skip'
    (run / 'full.yaml').write_text(yaml.safe_dump(full_doc))
    config = load_cache_config(run / 'full.yaml')
    factory, capture = make_capture_factory(provider, config, storage, run / 'model_capture', recipe_factory)
    metadata = dict(r17_model_spec=asdict(spec), qualification=True, synthetic_library=True,
                    quality_claim=False, full_yaml=str(run/'full.yaml'), retrieval_yaml=str(retrieval))
    if job['model'] == 'fastwam':
        from .manifest_binding import validate_manifest
        metadata['model_manifest'] = validate_manifest(asdict(adapter.manifest), asdict(spec))
    ks=(job.get('live_k',4),) if job['model']=='groot' else ((spec.default_sampler.num_steps,2) if job['model']=='dp' else (spec.default_sampler.num_steps,2,1))
    metadata['r17_samplers']={str(k):asdict(provider.sampler_for_steps(k)) for k in dict.fromkeys(ks)}
    server = WebsocketPolicyServer(factory(None, 'default'), host='0.0.0.0', port=job['server_port'],
        metadata=metadata, concurrent=True, connection_policy_factory=factory,
        bundle_validator=make_bundle_validator(provider),
        ready_callback=lambda: write_json(run / 'SERVER_READY.json', dict(metadata, pid=__import__('os').getpid(),
                     elapsed_load_seconds=time.monotonic()-started)))
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *args: server.request_stop())
    if server_started is not None: server_started(server)
    try:
        server.serve_forever(stop_on_request=True)
    finally:
        closed = capture.close()
        write_json(run / 'SERVER_CLOSED.json', dict(status='PASS' if all(r['drained'] for r in closed) else 'FAIL',
            writers=closed, elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--run', required=True)
    a = p.parse_args()
    serve(json.loads(Path(a.job).read_text()), Path(a.run))
