"""Explicit registration + shared factory. No model-specific server/cache loop."""
import copy

from exp.r17_integration.common.provider import AdapterProvider

from .key_builder import DPKeyBuilder
from .sampler import SamplerProfile
from .trace import DPTraceAdapter
from .admission import admission_validators, artifact_validator, recipe_factory as validate_recipe_factory
from .library import configure_library_provider


def register_provider(adapter, *, provider_id=None):
    if adapter.batch_diagnostic_label:
        raise ValueError("DP labelled batch diagnostic cannot be registered for production")
    from exp.r17_integration.common.provider import register_provider as register, register_trace_provider
    spec = adapter.metadata()
    sampler_for_steps = lambda k: SamplerProfile("auto" if spec.default_sampler.family == "ddpm" else "ddim", k,
                             spec.default_sampler.config.get("eta", 0.0)).to_spec(adapter.model.cfg)
    binding_validator, config_validator = admission_validators(spec, sampler_for_steps)
    provider = AdapterProvider(
        provider_id=provider_id or f"r17_dp_{adapter.domain}_{spec.default_sampler.family}_{spec.identity[-12:]}_v1",
        model_spec=spec, load_adapter=lambda: copy.copy(adapter),
        sampler_for_steps=sampler_for_steps,
        key_builder_factory=lambda config: DPKeyBuilder(adapter),
        trace_factory=DPTraceAdapter, artifact_validator=artifact_validator,
        binding_validator=binding_validator, config_validator=config_validator)
    provider = configure_library_provider(provider)
    register(provider)
    register_trace_provider(spec.model_id, provider.trace_factory)
    return provider


def make_connection_factory(adapter, *, provider, default_config, default_storage,
                            capture_sink=None, recipe_factory=None, stage_core=None, capture_spec=None, capture_sink_factory=None):
    from exp.r17_integration.common.serving import make_connection_factory as make
    from exp.r17_integration.common.recipe import make_recipe_factory
    if adapter.batch_diagnostic_label:
        raise ValueError("DP labelled batch diagnostic cannot serve production connections")
    check_production_core(stage_core, adapter)
    if provider.model_spec.identity != adapter.metadata().identity:
        pass
    provider.config_validator(default_config)
    def new_adapter():
        own = copy.copy(adapter)
        return own if stage_core is None else CoordinatedAdapter(own, stage_core)
    return make(provider, default_config, default_storage, capture_sink=capture_sink,
                recipe_factory=validate_recipe_factory(provider, recipe_factory or make_recipe_factory(provider)),
                adapter_factory=new_adapter, capture_spec=capture_spec, capture_sink_factory=capture_sink_factory)


def check_production_core(core, adapter=None):
    # Stock core has no read-only public batch-size query. Fail closed on
    # unknown coordinators and recheck before every submit; stage hooks also
    # reject B>1 if dispatcher parameters change between check and dispatch.
    limit = adapter.batch_config.max_batch if adapter is not None and adapter.batch_config is not None else 1
    if core is not None and not 1 <= getattr(core, "_max_batch_size", 0) <= limit:
        raise ValueError("DP production BatchingCore requires max_batch_size=1; GPU batch composition changes outputs")


class CoordinatedAdapter:
    """Route the same shared connection executor through stock BatchingCore."""
    def __init__(self, adapter, core):
        if adapter.batch_diagnostic_label:
            raise ValueError("DP labelled batch diagnostic cannot serve production connections")
        check_production_core(core, adapter)
        self.adapter, self.core = adapter, core
        self.batcher = adapter.make_stage_batcher()
        import uuid
        self.connection_id = uuid.uuid4().hex
        self.last_batch_receipt = {}

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def prepare(self, observation, request):
        prepared = self.adapter.prepare(observation, request)
        if self.adapter.batch_ledger is not None:
            from dataclasses import asdict
            import uuid
            prepared.data["batch_member"] = {"connection_id": self.connection_id,
                "request_id": uuid.uuid4().hex, "request": asdict(request)}
        return prepared

    def run_stage1(self, prepared):
        check_production_core(self.core, self.adapter)
        out = self.core.submit_to_stage(1, "dp_shared", prepared)
        self.last_batch_receipt = copy.deepcopy(out.data.get("batching", {}))
        return out

    def run_stage2(self, s1, *, capture=False):
        check_production_core(self.core, self.adapter)
        out = self.core.submit_to_stage(2, "dp_shared", s1, requires_stage2_capture=capture)
        self.last_batch_receipt = copy.deepcopy(out.data.get("batching", {}))
        return out

    def sample(self, s2, sampler, rng, capture):
        check_production_core(self.core, self.adapter)
        p = self.batcher.miss_payload(s2, sampler, rng, capture)
        out = self.core.submit_to_stage(3, "dp_shared", p).result
        self.last_batch_receipt = copy.deepcopy(out.capture.get("batching", {}))
        return out

    def resume(self, s2, snapshot, sampler, rng, capture):
        check_production_core(self.core, self.adapter)
        p = self.batcher.warm_payload(s2, snapshot, sampler, rng, capture)
        out = self.core.submit_to_stage(3, "dp_shared", p).result
        self.last_batch_receipt = copy.deepcopy(out.capture.get("batching", {}))
        return out
