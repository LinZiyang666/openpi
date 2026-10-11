"""Framework-native provider; reject unqualified Cosmos combinations explicitly."""
from .key_builder import CosmosKeyBuilder


def validate_config(config):
    if config.trace.enabled or config.warm_reset is not None:
        raise ValueError("Cosmos model trace/warm reset is unsupported")
    if config.recipe.enabled:
        r, b = config.recipe, config.adapter_binding
        from .manifest import make_sampler
        if (not config.enabled or config.checkpoints or r.miss_mode not in ('never', 'stall_only') or
                b['num_steps'] != 5 or b['sampler_identity'] != make_sampler(5).identity or
                b['commit_horizon'] != 16 or b.get('allow_partial_commit', False) or
                r.teacher_sampler_identity != make_sampler(5).identity or
                r.guard or r.target_ir is not None or r.ir_costs is not None or
                (r.k, r.kref, r.early) != (16, 8, True) or not r.fitted_path or not r.fitted_digest or
                (r.miss_mode == 'stall_only') != bool(r.stall_path and r.stall_digest)):
            raise ValueError('Cosmos recipe supports only the sealed native K5/C16 cache study; ordinary MISS unsupported')


def configure_library_provider(provider):
    from .library_admission import configure_library_provider as configure
    return configure(provider)


def register_provider(adapter):
    from exp.r17_integration.common.provider import AdapterProvider, register_provider as register
    from .staged import CosmosAdapter
    spec = adapter.metadata()
    def clone():
        return CosmosAdapter(adapter.model, adapter.manifest, adapter.cfg, adapter.stats, adapter.embeddings,
                             adapter.cu, lock=adapter.lock)
    provider = AdapterProvider(spec.model_id, spec, clone, adapter.sampler_for_steps,
        key_builder_factory=lambda config: CosmosKeyBuilder(spec.feature_spec), config_validator=validate_config)
    provider = configure_library_provider(provider)
    register(provider)
    return provider
