"""CPU sealing and library preparation using the published P1–P6 APIs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.replace(part, path)


def sealed_dp_adapter(job, benchmark):
    """Discover pins on CPU, then require the formal export checks when selected."""
    from dataclasses import replace
    from exp.r17_integration.models.dp.loader import load_adapter
    from exp.r17_integration.models.dp.config import LoadSpec
    descriptor = LoadSpec(job['checkpoint'], job['domain'], device='cpu', diagnostic=True,
        libero_benchmark_specs=[benchmark.manifest()] if job['domain']=='libero' else None)
    adapter = load_adapter(descriptor)
    if job.get('diagnostic', True) is False:
        pins = dict(expected_weight_sha256=adapter.weight_identity,
            expected_norm_identity=adapter.metadata().norm_identity,
            expected_model_identity=adapter.metadata().checkpoint_identity)
        del adapter
        adapter = load_adapter(replace(descriptor, diagnostic=False, **pins))
    return adapter


def model_assets(job, output):
    """No CUDA calls. DP's normalization requires an explicit CPU weight load."""
    from exp.r17_integration.common.provider import AdapterProvider
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    model, domain, checkpoint = job['model'], job['domain'], job['checkpoint']
    from .benchmark_binding import benchmark_spec
    benchmark=benchmark_spec(job)
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_symlink() or 'latest' in checkpoint_path.name:
        raise ValueError('immutable directory required')
    if model == 'pi05':
        from exp.r17_integration.models.pi05.adapter import content_manifest, model_spec, sampler_for_steps
        manifest = content_manifest(checkpoint)
        if not manifest['tokenizer']['sha256']:
            raise ValueError('missing sealed local PaliGemma tokenizer')
        spec = model_spec(manifest['checkpoint_identity'], manifest['norm_identity'], code_digest=manifest['code_digest'])
        load = dict(checkpoint=checkpoint, checkpoint_identity=spec.checkpoint_identity, device='cuda:0')
        provider = AdapterProvider('r17_pi05_mt50_v1', spec, lambda: None, sampler_for_steps)
    elif model == 'groot':
        from exp.r17_integration.models.groot.manifest import checkpoint_manifest
        vendor = Path(job['vendor'])
        eagle = vendor / 'gr00t/model/backbone/eagle2_hg_model'
        tokenizer = sorted(str(p) for p in eagle.iterdir() if p.is_file() and p.suffix in ('.json', '.txt', '.model'))
        code = sorted({str(p) for root in (vendor / 'gr00t', Path('exp/mw_finetune/groot'),
                    Path('exp/r17_integration/models/groot'), Path('src/openpi/cache/groot'))
                    for p in root.rglob('*.py')})
        manifests = {k: checkpoint_manifest(checkpoint, denoising_steps=k, tokenizer_files=tokenizer,
                                           code_files=code) for k in (4, 3, 2, 1)}
        m = manifests[job.get('live_k', 4)]
        spec = m.model_spec()
        manifest = dict(schema='r17.model_manifest.v1', spec=asdict(spec), groot=m.to_dict(),
                        assets=dict(tokenizer_files=tokenizer, code_files=code))
        load = dict(checkpoint=checkpoint, tokenizer_files=tokenizer, code_files=code, live_k=m.denoising_steps)
        provider = AdapterProvider('r17_groot_mw_v1', spec, lambda: None,
                                   lambda k: manifests[k].model_spec().default_sampler)
    elif model == 'fastwam':
        from exp.r17_integration.models.fastwam.loader import LoadSpec, inspect_config, read_stats, seal_descriptor
        from exp.r17_integration.models.fastwam.manifest import make_manifest, make_sampler
        import importlib.util
        scheduler_path = '/data/mw_finetune/vendor/FastWAM/src/fastwam/models/wan22/schedulers/scheduler_continuous.py'
        module_spec = importlib.util.spec_from_file_location('q_native_scheduler', scheduler_path)
        native = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(native)
        # The scheduler is a CPU-only object; never construct the policy here.
        descriptor = LoadSpec(domain, checkpoint, '/data/xwam/wan22_5b',
            base='/data/mw_finetune/models/fastwam_libero', diagnostic=True,
            benchmark_protocols=(benchmark.identity,), benchmark_specs=(benchmark.manifest(),))
        config, stats = inspect_config(descriptor), read_stats(descriptor)
        sealed = seal_descriptor(descriptor)
        load = dict(asdict(descriptor), expected_digests=sealed)
        scheduler = native.WanContinuousFlowMatchScheduler(shift=5.0)
        sampler_for_steps = lambda k: make_sampler(scheduler, k)
        m = make_manifest(domain, config, stats, sampler_for_steps(10), sealed,
                          benchmark_protocols=descriptor.benchmark_protocols)
        spec, manifest = m.spec, asdict(m)
        provider = AdapterProvider(spec.model_id, spec, lambda: None, sampler_for_steps)
    elif model == 'cosmos':
        from exp.r17_integration.models.cosmos.loader import LoadSpec, cpu_manifest
        from exp.r17_integration.models.cosmos.manifest import make_sampler
        descriptor = LoadSpec(checkpoint, vendor_root=job['vendor'],
                              benchmark_protocols=(benchmark.identity,))
        m = cpu_manifest(descriptor)
        spec, manifest = m.spec, asdict(m)
        load = dict(asdict(descriptor), expected_digests=m.content_digests)
        provider = AdapterProvider(spec.model_id, spec, lambda: None, make_sampler)
    elif model == 'dp':
        from exp.r17_integration.models.dp.provider import register_provider
        adapter = sealed_dp_adapter(job, benchmark)
        provider = register_provider(adapter)
        spec, manifest = adapter.metadata(), adapter.manifest()
        load = dict(adapter.load_spec, device='cuda:0', expected_weight_sha256=adapter.weight_identity,
                    expected_norm_identity=spec.norm_identity, expected_model_identity=spec.checkpoint_identity)
    else:
        raise ValueError(model)
    from exp.r17_integration.common.provider import register_provider, get_provider
    try:
        get_provider(provider.provider_id)
    except ValueError:
        register_provider(provider)
    write_json(output / 'model_manifest.json', manifest)
    write_json(output / 'model_spec.json', asdict(spec))
    write_json(output / 'load.json', load)
    return spec, provider


def synthetic_store(spec, provider, output, *, task_name=None):
    """Production geometry and identity, explicitly fabricated arrays, no teacher claim."""
    import numpy as np
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.debug.capture_bridge import CaptureBridge, export_h5
    from exp.r17_integration.search.build_library import build_library, save_stock_artifact
    from exp.r17_integration.search.fit import fit_recipe, save_fitted_recipe
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    protocol = spec.allowed_benchmark_protocols[0]
    task_name=task_name or ('reach-v3' if len(spec.layout.valid_state_indices)==4 else 'libero_mechanics')
    b = model_binding(spec, protocol)
    bridge = CaptureBridge(output / 'synthetic_capture', enabled=True, binding=b)
    episodes = []
    for ep in range(3):
        row = dict(campaign='Q-synthetic', arm='synthetic', bundle_generation='synthetic-v1',
            benchmark_protocol=protocol, task_name=task_name,
            orig_init_idx=10+ep, task_data_hash='synthetic-task-data-'+str(ep), task_uid='Q-synthetic-'+str(ep),
            attempt=1, split='library', closed_ok=True, success=False, n_decisions=3, synthetic=True,
            real_teacher=False, actual_gpu_executed=False)
        episodes.append(row)
        for seq in range(3):
            rng = np.random.default_rng(100+ep*3+seq)
            state = rng.normal(0, .1, len(spec.layout.valid_state_indices)).astype(np.float32)
            actions = rng.normal(0, .1, (spec.layout.horizon, spec.layout.action_width)).astype(np.float32)
            keys = {key: (state.copy() if key == 'robot_state' else rng.normal(0, .1, size).astype(np.float32))
                    for key, size in spec.feature_spec.fields.items()}
            bridge.capture(dict(row, decision_seq=seq, source='synthetic', actual_nfe=None,
                sampler_receipt={'identity': spec.default_sampler.identity}),
                dict(state=state, normalized_actions=actions,
                     native_grid=np.asarray(spec.default_sampler.native_timesteps)), rawkeys=keys)
    assert bridge.close()['drained']
    h5 = export_h5(output / 'synthetic_capture', output / 'synthetic.h5')
    store = build_library([h5], output / 'synthetic_teacher_store', b, episodes,
        key_fields=spec.feature_spec.fields, split_policy='mt50_init_v1' if protocol == FORMAL_SPEC.identity else 'declared_partition_v1')
    save_stock_artifact(store, output / 'synthetic_library.pkl')
    from .benchmark_binding import recipe_table
    names = recipe_table(protocol)['ordered_tasks']
    recipe, receipt = fit_recipe(store, task_names=names, corrector=True, periodic_gap=2,
        early=False, commit_horizon=spec.layout.replan_steps, k=2, kref=2)
    save_fitted_recipe(recipe, receipt, output / 'synthetic_fit')
    write_json(output / 'SYNTHETIC.json', dict(synthetic=True, real_teacher=False, quality_claim=False,
        episodes=3, decisions=9, store_digest=store.manifest['content_digest'], fit_digest=receipt['digest']))


def configs_and_plan(job, spec, provider, output):
    import yaml
    from exp.r17_integration.common.qualification import config_for_probe
    from exp.r17_integration.common.provider import validate_config
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    from exp.r17_integration.conductor.manifest import EndpointSpec, episode_tasks
    from openpi.conductor.task import ServerEndpoint
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    ks = (spec.default_sampler.num_steps, 2, 1)
    if job['model']=='dp':
        # K1 is a labelled lossy native study, excluded from live Q recipes/bundles.
        ks = (spec.default_sampler.num_steps, 2)
    if job['model'] == 'groot':
        ks = (job.get('live_k', 4),)
    for k in dict.fromkeys(ks):
        config = config_for_probe(provider, k, spec.layout.replan_steps)
        document = asdict(config)
        from .benchmark_binding import recipe_table
        document['recipe'].update(recipe_table(spec.allowed_benchmark_protocols[0]))
        # Leave inactive trace/routing defaults out, retain strict binding.
        path = output / f'k{k}.yaml'
        path.write_text(yaml.safe_dump(document))
        actual = load_cache_config(path)
        validate_config(actual)
        build_shared_storage(actual)
    endpoint = ServerEndpoint(job['endpoint_host'], job['server_port'])
    tasks = []
    if job['domain'] == 'metaworld':
        for arm, k, idx in [('full', ks[0], 0), ('retrieval', ks[0] if job['model'] == 'groot' else 2, 1)]:
            ypath = output / f'k{k}.yaml'
            tasks.extend(episode_tasks(campaign='Q-'+job['asset_id'], arm=arm, bundle_id=arm+'_gen1',
                endpoint=endpoint, model_identity=spec.checkpoint_identity, norm_identity=spec.norm_identity,
                layout_identity=spec.layout.identity, num_steps=k, partition='smoke', names=['reach-v3'], indices=[idx],
                yaml_identity=hashlib.sha256(ypath.read_bytes()).hexdigest(), bundle_generation=arm+'_gen1'))
        e = EndpointSpec(endpoint, spec.checkpoint_identity, spec.norm_identity, spec.layout.identity,
                         FORMAL_SPEC.identity, allowed_k=tuple(ks))
        assert all(e.accepts(t) for t in tasks)
    from .benchmark_binding import benchmark_spec
    write_json(output / 'episode_manifest.json', dict(schema='r17.episode_plan.v1', plan_only=True,
        quality_sr_claim=False, model=asdict(spec), expected_episodes=len(tasks), tasks=[asdict(t) for t in tasks],
        endpoint=asdict(endpoint), driver_address=f"{job['endpoint_host']}:{job['driver_port']}",
        worker_node=job['worker_node'], benchmark=benchmark_spec(job).manifest()))


def observations(job, output):
    import numpy as np
    if job['domain']=='libero':
        import shutil
        from .recorded_libero import validate
        source=Path(job['recorded_observations'])
        validate(source)
        for name in ('wire_observations.npz','fastwam_observations.npz','OBSERVATIONS.json'):
            shutil.copyfile(source/name,output/name)
        validate(output)
        return
    image = np.arange(480*480*3, dtype=np.uint8).reshape(480,480,3)
    domain = job['domain']
    nstate = 4 if domain == 'metaworld' else 8
    # MW y/z/gripper near the native training range. Zero y/gripper clips to
    # identical GR values; distinct raw rows must survive native normalization.
    base=np.array([0.,.60,.20,.60] if domain=='metaworld' else [.45,0.,.35,0.,0.,0.,.04,.04],np.float32)
    delta=np.array([.04,-.04,.03,.10] if domain=='metaworld' else [.01,.02,-.01,.01,-.02,.01,.002,-.002],np.float32)
    states = np.stack([base+i*delta for i in range(3)])
    prompts = np.array(['reach the object' if domain == 'metaworld' else job.get('mechanics_task_name','pick up the black bowl and place it on the plate')]*3)
    images = np.stack([np.roll(image, i*7, axis=1) for i in range(3)])
    np.savez_compressed(output / 'groot_observations.npz', images=images, states=states, prompts=prompts)
    args = dict(image=images, state=states, prompt=prompts)
    np.savez_compressed(output / 'fastwam_observations.npz', **args)
    np.savez_compressed(output / 'wire_observations.npz', **{'observation/image': images,
        'observation/state': states, 'prompt': prompts})
    write_json(output / 'OBSERVATIONS.json', dict(synthetic=True, real_reset=False, quality_claim=False,
        rows=3, image_shape=list(images.shape), state_shape=list(states.shape)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error("CPU preparation requires CUDA_VISIBLE_DEVICES=''")
    job = json.loads(Path(a.job).read_text())
    out = Path(a.output)
    if out != Path(job['assets']):
        parser.error('--output must equal the concrete job assets path; publish a new job for a new generation')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'job.json', job)
    report = dict(status='PARTIAL', cpu_only=True, actual_gpu_executed=False)
    try:
        spec, provider = model_assets(job, out)
        configs_and_plan(job, spec, provider, out)
        observations(job, out)
        synthetic_store(spec, provider, out, task_name=job.get('mechanics_task_name'))
        report.update(status='PASS', manifest=True, configs=True, plan_only=True, synthetic_store=True,
            observation_manifest_sha256=hashlib.sha256((out/'OBSERVATIONS.json').read_bytes()).hexdigest())
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out / 'CPU_RECEIPT.json', report)


if __name__ == '__main__':
    main()
