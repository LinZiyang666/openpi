"""One resident, serial B1 replay of a sealed R17 adapter and fitted cache recipe.

No sockets, workers, lane mutations, or checkpoint writes. Capture/plain parity
is checked. This isolates model arithmetic; it does not certify dynamic batching.
"""
from __future__ import annotations
import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from artifacts import array, digest, save_tree, sha, tensor_leaves, write_json
from capture import read_inputs
from inventory import collect


def verify_runtime(root, expected):
    root = Path(root).resolve()
    doc = json.loads((root / 'RUNTIME_RELEASE.json').read_text())
    actual = hashlib.sha256((json.dumps(doc, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest()
    if actual != expected or not doc.get('files'):
        raise ValueError('runtime release identity mismatch')
    for name, wanted in doc['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or sha(path) != wanted:
            raise ValueError('runtime source differs from release: ' + name)
    return actual


def apply_profile(name):
    import torch
    if name == 'native':
        return
    # Deliberately cumulative, applied AGAIN after the model constructor.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if name in ('highest', 'eager', 'math', 'deterministic', 'full_reduction'):
        torch.set_float32_matmul_precision('highest')
    if name in ('math', 'deterministic', 'full_reduction'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
            torch.backends.cuda.enable_cudnn_sdp(False)
    if name in ('deterministic', 'full_reduction'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    if name == 'full_reduction':
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


@contextlib.contextmanager
def draws(torch):
    """Observe actual torch.randn draws, preserving generator/device/dtype/order."""
    original = torch.randn
    recorded = []
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        recorded.append(array(result))
        return result
    torch.randn = observed
    try:
        yield recorded
    finally:
        torch.randn = original


def native_tensors(value, prefix='', depth=0):
    """Only stage data, never model parameters or callable closures/KV caches."""
    if depth > 5:
        return {}
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return {prefix: array(value)}
    if dataclasses.is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k not in ('video_kv_cache', 'past_key_values', 'prefix_kv_cache', 'stage1', 'x0_fn'):
                result.update(native_tensors(v, prefix + '/' + str(k), depth + 1))
        return result
    return {}


def run_stages(adapter, observation, request, sampler, model):
    import torch
    from exp.r17_integration.common.contracts import CaptureSpec
    from exp.r17_integration.common.request_rng import RequestRNG
    prepared = adapter.prepare(observation, request)
    s1 = adapter.run_stage1(prepared)
    fields = {k: array(v) for k, v in adapter.retrieval_fields(s1).items()}
    # pi05's capture entry retains the already-computed prefix output. Its
    # forward arguments are unchanged; plain/capture action parity is mandatory.
    s2 = adapter.run_stage2(s1, capture=model == 'pi05')
    stages = dict(stage1=native_tensors(s1.data), stage2=native_tensors(s2.data), query=fields)
    cosmos_steps = []
    cosmos_noise = []
    original = None
    if model == 'cosmos':
        from cosmos_policy._src.imaginaire.utils import misc
        original_rand = misc.arch_invariant_rand
        def observed_rand(*args, **kwargs):
            value = original_rand(*args, **kwargs)
            cosmos_noise.append(array(value))
            return value
        misc.arch_invariant_rand = observed_rand
        # The native stage2 is a callable closure, with no exposed prefix tensor.
        # Record its actual latent conditioning, not a fabricated hidden state.
        stages['stage2']['/latent_conditioning'] = array(s1.data['latent'])
        original = s2.data['x0_fn']
        def observed(x, sigma):
            y = original(x, sigma)
            cosmos_steps.append(dict(x_t=array(x), sigma=array(sigma), prediction=array(y)))
            return y
        s2.data['x0_fn'] = observed
    try:
        with draws(torch) as noises:
            result = adapter.sample(s2, sampler, RequestRNG(request),
                CaptureSpec(enabled=model != 'cosmos', snapshots=model != 'cosmos'))
    finally:
        if original is not None:
            s2.data['x0_fn'] = original
            misc.arch_invariant_rand = original_rand
    stages['noise'] = {str(i): v for i, v in enumerate(noises)}
    stages['noise'].update({f'arch_invariant_{i}': v for i, v in enumerate(cosmos_noise)})
    for name in ('initial_noise', 'first_step_input'):
        if name in result.capture:
            stages['noise'][name] = array(result.capture[name])
    # Fast-WAM keeps its first draw in native_steps; pi05/GR00T expose initial_noise.
    if result.capture.get('native_steps'):
        stages['noise']['initial_noise'] = array(result.capture['native_steps'][0]['x_t'])
    stages['denoise'] = {f'{s.loop_index:03d}': array(s.x_t) for s in result.snapshots}
    if cosmos_steps:
        stages['denoise'] = {f'{i:03d}': v['x_t'] for i, v in enumerate(cosmos_steps)}
        stages['cosmos_predictions'] = cosmos_steps
    if hasattr(adapter, 'capture_fields'):
        stages['capture'] = adapter.capture_fields(s1, s2, result)
    else:
        stages['capture'] = result.capture
    stages['final_normalized'] = array(result.normalized_chunk)
    stages['final_wire'] = array(adapter.to_wire(result.normalized_chunk, prepared).actions)
    # The final update + K-1 snapshots cover every denoising step.
    expected = result.actual_nfe if model == 'cosmos' else sampler.num_steps - 1
    if len(stages['denoise']) != expected or not stages['noise']:
        raise ValueError(f'incomplete denoise/noise capture: {len(stages["denoise"])} != {expected}')
    return stages, fields, array(adapter.extract_state_only(observation).state), result


def make_cache(provider, yaml_path):
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    from exp.r17_integration.common.recipe import make_recipe_factory, recipe_identity
    config = load_cache_config(yaml_path)
    if not config.recipe.enabled or not config.recipe.fitted_path:
        raise ValueError('cache replay requires an existing fitted recipe; never fit a library in this probe')
    fit = Path(config.recipe.fitted_path)
    ready = json.loads((fit / 'READY.json').read_text())
    if (ready.get('digest') != config.recipe.fitted_digest or
        digest({k: v for k, v in ready.items() if k != 'digest'}) != config.recipe.fitted_digest or
        sha(fit / 'fit.npz') != ready['arrays_sha256']):
        raise ValueError('fitted cache content seal mismatch')
    storage = build_shared_storage(config)
    recipe = make_recipe_factory(provider)(config, storage)
    binding = config.adapter_binding
    spec = provider.model_spec
    for name, expected in [('checkpoint_identity', spec.checkpoint_identity), ('norm_identity', spec.norm_identity),
                           ('layout_identity', spec.layout.identity), ('feature_identity', spec.feature_spec.identity)]:
        if binding[name] != expected:
            raise ValueError('cache/model binding mismatch: ' + name)
    seal = dict(recipe=recipe_identity(config), store=config.recipe.store_digest,
                fit=config.recipe.fitted_digest, stall=config.recipe.stall_digest,
                execution_binding=dict(binding))
    return recipe, seal


def select_sampler(provider, fallback_steps, cache_seal):
    binding = cache_seal['execution_binding'] if cache_seal else None
    sampler = provider.sampler_for_steps(binding['num_steps'] if binding else fallback_steps)
    if binding and sampler.identity != binding['sampler_identity']:
        raise ValueError('cache/native sampler identity mismatch')
    return sampler


def cache_query(recipe, row, request, features, state):
    from exp.r17_integration.common.query_view import QueryView
    if row['episode_start']:
        recipe.reset(dict(request=dataclasses.asdict(request), task_name=request.task_name,
                          original_init_idx=request.original_init_idx, task_id=request.task_id))
    count = row.get('executed_controls_since_last_query', 0)
    if type(count) is not int or count < 0:
        raise ValueError('invalid recorded execution count')
    if count:
        hook = getattr(recipe, 'on_execution', None)
        if hook is not None:
            hook(count)
    allowed = {'hist_has_vision', 'hist_hit', 'hist_state', 'previous_action', 'prev_hit', 'executed_count'}
    if set(row['history']) - allowed:
        raise ValueError('unknown query history fields')
    result = recipe.query(QueryView(request, features, state, **row['history']))
    neighbours = array(result.rows).reshape(-1)
    distances = array(result.distances).reshape(-1)
    extra = result.extras
    base = getattr(recipe, 'base', recipe)
    ratio = (extra['raw_anchor_distance'] / base.distance_scale
             if 'raw_anchor_distance' in extra and hasattr(base, 'distance_scale') else None)
    return dict(hit=not bool(result.force_miss),
        top1=int(neighbours[0]) if len(neighbours) else None,
        corrector=extra.get('corrector_applied'), stall=extra.get('stall_call'),
        distances=distances, neighbours=neighbours, weights=array(result.weights),
        corrected_action=array(result.action), diagnostics=extra,
        # Distances may be weighted/scored in a model-specific way: do not invent
        # a threshold margin from arbitrary fields. Comparator marks it unavailable.
        score_margin=extra.get('score_margin'), neighbour_margin=(float(distances[1] - distances[0])
            if len(distances) > 1 else None),
        corrector_ratio=ratio, corrector_margin=min(abs(ratio - .75), abs(ratio - 2.)) if ratio is not None else None,
        stall_margin=extra.get('stall_margin'),
        threshold=extra.get('threshold'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'runtime', 'inputs', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--runtime-identity', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--memory-gib', type=float, default=4)
    p.add_argument('--deadline-seconds', type=int, default=300)
    p.add_argument('--cache-yaml')
    p.add_argument('--profile', choices=('native', 'tf32_off', 'highest', 'eager', 'math', 'deterministic', 'full_reduction'), default='native')
    p.add_argument('--vendor-path', action='append', default=[])
    a = p.parse_args()
    if a.repeats < 3 or a.memory_gib <= 0 or not 0 < a.deadline_seconds <= 900:
        p.error('require >=3 repeats, positive memory and deadline <=900s')
    if a.profile in ('deterministic', 'full_reduction'):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if a.profile in ('eager', 'math', 'deterministic', 'full_reduction'):
        os.environ['TORCHDYNAMO_DISABLE'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # These five serving adapters are PyTorch. JAX is imported for shared tree /
    # typing helpers only; do not let its backend allocate an unrelated GPU pool.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    sys.dont_write_bytecode = True
    root = Path(a.runtime).resolve()
    verify_runtime(root, a.runtime_identity)
    sys.path[:0] = [str(root), str(root / 'src'), str(root / 'packages/openpi-client/src'), *a.vendor_path]
    manifest, inputs = read_inputs(a.inputs)
    job = json.loads(Path(a.job).read_text())
    job['benchmark_spec'] = str(Path(job['assets']) / 'benchmark.json')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='xgpu.replay.v1', status='INCOMPLETE', input_identity=manifest['identity'],
        runtime_identity=a.runtime_identity, profile=a.profile, model=job['model'],
        scope='serial B1 adapter + fitted recipe; not live dynamic batch/scheduling',
        rows=[], started=time.time(), cache_identity=None)
    write_json(out / 'REPLAY.json', report)
    start = time.monotonic()
    # A watchdog caps total wall time, including hung model loads. Writes remain
    # INCOMPLETE on forced exit. External timeout is an additional coordinator guard.
    import threading
    done = threading.Event()
    def deadline():
        if not done.wait(a.deadline_seconds):
            os._exit(124)
    threading.Thread(target=deadline, daemon=True).start()
    try:
        import torch
        if a.device.startswith('cuda'):
            free, total = torch.cuda.mem_get_info(a.device)
            budget = int(a.memory_gib * 2**30)
            if free < budget + 512 * 2**20:
                raise RuntimeError('insufficient free memory; coordinator must admit this probe')
            original_cap = torch.cuda.set_per_process_memory_fraction
            def bounded_cap(fraction, device=None):
                # Some frozen loaders request 8 GiB internally. They must not
                # override the coordinator's smaller diagnostic allocation.
                return original_cap(min(fraction, budget / total), device)
            torch.cuda.set_per_process_memory_fraction = bounded_cap
            bounded_cap(budget / total, a.device)
        apply_profile(a.profile)
        from exp.r17_integration.qualification.model_runtime import load
        from exp.r17_integration.common.contracts import RequestContext
        from exp.r17_integration.common.serving import full_call
        from exp.r17_integration.models.pi05.gpu_diagnostic import action_difference
        if job['model'] == 'dp':
            # Frozen DP loader validates descriptor text_encoder, but its legacy
            # policy constructor otherwise uses a hard-coded absolute default.
            # Pass the same sealed content via the documented loader argument.
            from exp.mw_finetune.dp import policy as dp_policy
            desc = json.loads((Path(job['assets']) / 'load.json').read_text())
            native_loader = dp_policy.load_policy
            def relocated_loader(checkpoint, device, **kw):
                return native_loader(checkpoint, device, **dict(kw, text_encoder=desc['text_encoder']))
            dp_policy.load_policy = relocated_loader
            try:
                adapter, provider = load(job, device=a.device)
            finally:
                dp_policy.load_policy = native_loader
            report['path_only_loader_override'] = 'DP text_encoder from sealed load.json; no arithmetic changes'
        else:
            adapter, provider = load(job, device=a.device)
        expected_spec = json.loads((Path(job['assets']) / 'model_spec.json').read_text())
        actual_spec = dataclasses.asdict(adapter.metadata())
        if digest(actual_spec) != digest(expected_spec):
            raise ValueError('loaded model spec differs from sealed job')
        report['model_spec'] = actual_spec
        report['binding_identity'] = digest(actual_spec)
        apply_profile(a.profile)
        report['environment'] = collect(with_torch=True)
        report['environment']['scope'] = 'after model load and profile; actual replay process'
        if 'jax' in sys.modules:
            report['environment']['jax_default_matmul_precision'] = sys.modules['jax'].config.jax_default_matmul_precision
        if a.device.startswith('cuda'):
            report['gpu'] = dict(name=torch.cuda.get_device_name(a.device),
                capability=list(torch.cuda.get_device_capability(a.device)))
        spec = adapter.metadata()
        recipe = None
        if a.cache_yaml:
            recipe, report['cache_identity'] = make_cache(provider, a.cache_yaml)
        sampler = select_sampler(provider, job.get('denoising_steps') or spec.default_sampler.num_steps,
                                 report['cache_identity'])
        report['sampler'] = dataclasses.asdict(sampler)
        binding = report['cache_identity']['execution_binding'] if recipe is not None else None
        report['commit_horizon'] = binding['commit_horizon'] if binding else spec.layout.replan_steps
        for repeat in range(a.repeats):
            if recipe is not None:
                recipe.reset({})
            for i, row in enumerate(inputs):
                request = RequestContext(**row['request'])
                if request.benchmark_protocol not in spec.allowed_benchmark_protocols:
                    raise ValueError('input benchmark outside model protocols')
                if binding and request.benchmark_protocol != binding['benchmark_protocol']:
                    raise ValueError('input/cache benchmark protocol mismatch')
                with torch.inference_mode():
                    stages, features, state, result = run_stages(adapter, row['observation'], request, sampler, job['model'])
                    # Plain-vs-capture is a required observer-effect control, not a
                    # relaxation of native parity. Repeat zero covers every input.
                    parity = None
                    if repeat == 0:
                        plain, wire = full_call(adapter, row['observation'], request, sampler=sampler)
                        parity = action_difference(array(wire.actions), stages['final_wire'])
                        if not parity['exact']:
                            raise ValueError('capture changed native actions: ' + str(parity))
                    cache = cache_query(recipe, row, request, features, state) if recipe is not None else None
                name = f'row_{i:04d}_repeat_{repeat:02d}'
                seal = save_tree(out / name, dict(stages=stages, cache=cache,
                    rng_receipt=result.rng_receipt, request=row['request']))
                report['rows'].append(dict(input=i, repeat=repeat, file=name, sha256=seal,
                    actual_nfe=result.actual_nfe, capture_plain_parity=parity))
                report['elapsed_seconds'] = time.monotonic() - start
                write_json(out / 'REPLAY.json', report)
        report['status'] = 'COMPLETE'
        report['environment_after_replay'] = collect(with_torch=True)
        if a.device.startswith('cuda'):
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated(a.device) / 2**30
            report['peak_reserved_gib'] = torch.cuda.max_memory_reserved(a.device) / 2**30
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        done.set()
        report['elapsed_seconds'] = time.monotonic() - start
        write_json(out / 'REPLAY.json', report)
    print(json.dumps(dict(status=report['status'], rows=len(report['rows']), output=str(out))))


if __name__ == '__main__':
    main()
