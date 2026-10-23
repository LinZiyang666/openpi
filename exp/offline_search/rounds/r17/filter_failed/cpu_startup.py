"""CPU descriptor/identity half of the frozen resident startup, without serving.

DP uses the complete strict CPU loader. Large VLA/video policies use the same
content-descriptor and ModelSpec builders called by their loaders; no backend
or CUDA policy is constructed. Execute by run_path in the selected frozen tree.
"""
from dataclasses import asdict
import importlib.util
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import sys
import traceback


def probe(campaign_path):
    c = json.loads(Path(campaign_path).read_text())
    assets = Path(c['assets'])
    desc = json.loads((assets/'load.json').read_text())
    sealed = json.loads((assets/'model_spec.json').read_text())
    manifest = json.loads((assets/'model_manifest.json').read_text())
    result = dict(run=c['run_name'], python=sys.executable, prefix=sys.prefix, status='FAIL',
                  assets=str(assets), model=c['model'], identity_path='', package_versions={})
    for package in ('torch','torchvision','diffusers','transformers','numpy','safetensors'):
        try: result['package_versions'][package] = version(package)
        except PackageNotFoundError: pass
    try:
        pinned = manifest.get('dependency_versions', {})
        result['pinned_packages_equal'] = all(result['package_versions'].get(k) == v for k,v in pinned.items())
        from exp.r17_integration.qualification.benchmark_binding import benchmark_spec
        benchmark_spec(dict(c, benchmark_spec=str(assets/'benchmark.json')))
        if c['model'] == 'dp':
            from exp.r17_integration.qualification.model_runtime import load
            adapter, provider = load(dict(c, benchmark_spec=str(assets/'benchmark.json')), 'cpu')
            spec = asdict(adapter.metadata())
            result['identity_path'] = 'qualification.model_runtime.load -> dp.loader.load_adapter(cpu)'
        elif c['model'] == 'pi05':
            from exp.r17_integration.models.pi05.adapter import content_manifest, model_spec
            m = content_manifest(desc['checkpoint'])
            spec = asdict(model_spec(m['checkpoint_identity'], m['norm_identity'], code_digest=m['code_digest']))
            result['identity_path'] = 'pi05.adapter.content_manifest/model_spec (loader descriptor path)'
        elif c['model'] == 'groot':
            from exp.r17_integration.models.groot.loader import checkpoint_manifest
            m = checkpoint_manifest(desc['checkpoint'], denoising_steps=desc['live_k'],
                                    tokenizer_files=desc['tokenizer_files'], code_files=desc['code_files'])
            spec = asdict(m.model_spec())
            result['identity_path'] = 'groot.loader.checkpoint_manifest (load_checkpoint descriptor path)'
        elif c['model'] == 'fastwam':
            from exp.r17_integration.models.fastwam.loader import LoadSpec, inspect_config, read_stats, seal_descriptor, raw_wan_code_root
            from exp.r17_integration.models.fastwam.protocols import validate_benchmark_specs
            from exp.r17_integration.models.fastwam.manifest import make_manifest, make_sampler
            descriptor = LoadSpec(**desc)
            validate_benchmark_specs(descriptor)
            actual = seal_descriptor(descriptor)
            result['content_seal_equal'] = actual == descriptor.expected_digests
            path = raw_wan_code_root()/'schedulers/scheduler_continuous.py'
            module_spec = importlib.util.spec_from_file_location('ff_cpu_scheduler', path)
            scheduler_module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(scheduler_module)
            sampler = make_sampler(scheduler_module.WanContinuousFlowMatchScheduler(shift=5.0), 10)
            m = make_manifest(descriptor.benchmark, inspect_config(descriptor), read_stats(descriptor), sampler, actual,
                              benchmark_protocols=descriptor.benchmark_protocols, execution_protocol=descriptor.execution_protocol)
            spec = asdict(m.spec)
            result['identity_path'] = 'fastwam.loader.seal_descriptor/inspect_config/read_stats -> native CPU scheduler -> make_manifest'
        elif c['model'] == 'cosmos':
            from exp.r17_integration.models.cosmos.loader import LoadSpec, cpu_manifest
            descriptor = LoadSpec(**desc)
            m = cpu_manifest(descriptor)
            result['content_seal_equal'] = dict(m.content_digests) == descriptor.expected_digests
            spec = asdict(m.spec)
            result['identity_path'] = 'cosmos.loader.cpu_manifest (strict loader descriptor path)'
        else:
            raise ValueError(c['model'])
        from exp.r17_integration.debug.validate import canonical
        result['actual_spec'] = spec
        result['spec_equal'] = canonical(spec) == canonical(sealed)
        # Config parsing needs the registered identity/sampler catalogue, but
        # no heavyweight backend. Use the same metadata-only registration as
        # the standard study builder, after verifying the actual descriptor.
        import yaml
        from types import SimpleNamespace
        from exp.r17_integration.ops.cache_study import model_spec as decode_spec
        from exp.r17_integration.common.provider import AdapterProvider, get_provider, register_provider
        study = json.loads((assets/'CACHE_STUDY.json').read_text())
        documents = [yaml.safe_load((assets/'full.yaml').read_text())]
        documents += [yaml.safe_load((assets/row['file']).read_text()) for row in study['groups']]
        samplers = {}
        for doc in documents:
            binding = doc['adapter_binding']
            samplers.setdefault(binding['provider'], {})[binding['num_steps']] = binding['sampler_identity']
        for name, profiles in samplers.items():
            try:get_provider(name)
            except ValueError:
                decoded = decode_spec(spec)
                register_provider(AdapterProvider(name, decoded, lambda:None,
                    lambda k, profiles=profiles:SimpleNamespace(num_steps=k, identity=profiles[k], family=decoded.default_sampler.family)))
        from exp.r17_integration.common.config import load_cache_config
        full = load_cache_config(assets/'full.yaml')
        assert not full.recipe.enabled, 'startup must use the source plain full.yaml'
        if c['model']=='dp' and c['domain']=='libero' and c.get('dp_batching'):
            from exp.r17_integration.models.dp.batch_runtime import BatchConfig
            from exp.r17_integration.models.dp.formal_batch_server import cache_study_admission
            batch=c['dp_batching']
            config=BatchConfig(batch['max_batch'],batch['batch_wait_ms'],batch['numerics_policy'])
            assert config.metadata()==batch
            admission_c=dict(c)
            if c['replicas'] in (3,4):
                # Exactly the existing source R4 admission bridge in server_cli;
                # no identity, backend, sampler or actual topology is changed.
                assert (Path(c['filter_failed_source'])/'servers/replica_0/CACHE_CAPACITY_ADMISSION.json').exists()
                admission_c['replicas']=2
            cache_study_admission(admission_c,full,config,provider)
            result['dp_startup_admission_checked']=True
        for row in study['groups']:
            cfg = load_cache_config(assets/row['file'])
            assert row['group'] == 'c' and cfg.recipe.miss_mode == 'never'
            assert not cfg.recipe.corrector and not cfg.recipe.guard and cfg.recipe.periodic_gap is None
            if c['model']=='dp':provider.config_validator(cfg)
            for path in (Path(cfg.recipe.store_path)/'VIEW.json', Path(cfg.recipe.fitted_path)/'READY.json',
                         Path(cfg.recipe.fitted_path)/'fit.npz'):
                assert path.is_file(), str(path)
        for key in ('dp_batching', 'fastwam_batching', 'stage_batching'):
            if c.get(key):
                result[key] = c[key]
        result.update(status='PASS', cache_groups=len(study['groups']), startup_config_checked=True)
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
    if 'torch' in sys.modules:
        result['cuda_initialized'] = sys.modules['torch'].cuda.is_initialized()
        assert not result['cuda_initialized']
    return result
