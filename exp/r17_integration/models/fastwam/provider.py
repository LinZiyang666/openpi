"""Explicit R17 provider registration; all mutable wrappers are connection-local."""
from __future__ import annotations

from .key_builder import FastWAMKeyBuilder


def configure_library_provider(provider, *, manifest=None):
    """Install the same lazy, content-bound B500 admission hook on CPU/live paths.

    P6's CPU sealer supplies a frozen metadata-only provider. Registration must
    keep its ID/spec; no adapter or weights are loaded by this operation.
    """
    from dataclasses import replace
    from .ops.library_compatibility import library_validator
    return replace(provider, library_validator=library_validator(provider.model_spec, manifest=manifest))


def register_provider(adapter, *, provider_id=None, coordinator=None):
    from exp.r17_integration.common.provider import AdapterProvider, register_provider as register
    from .staged import FastWAMAdapter
    spec = adapter.metadata()
    if coordinator is not None and not adapter.allow_unqualified_batch and adapter.batch_config is None:
        raise ValueError("Fast-WAM production batch coordinator disabled until GPU parity qualification")
    def clone():
        local = FastWAMAdapter(adapter.model, adapter.manifest, adapter.wire, adapter.config,
            stage1_lock=adapter.stage1_lock, allow_unqualified_batch=adapter.allow_unqualified_batch,
            batch_config=adapter.batch_config, batch_ledger=adapter.batch_ledger)
        if coordinator is not None:
            from .batcher import FastWAMBatchedAdapter
            return FastWAMBatchedAdapter(local, coordinator)
        return local
    p = AdapterProvider(provider_id or spec.model_id, spec, clone, adapter.sampler_for_steps,
        key_builder_factory=lambda config: FastWAMKeyBuilder(spec.feature_spec),
        trace_factory=lambda a: a.make_trace_adapter())
    p = configure_library_provider(p, manifest=adapter.manifest)
    register(p)
    return p


def make_connection_factory(provider, config, storage, **kwargs):
    from exp.r17_integration.common.serving import make_connection_factory as factory
    return factory(provider, config, storage, **kwargs)


def make_stage_batcher(adapter):
    return adapter.make_stage_batcher()


def make_trace_adapter(adapter):
    return adapter.make_trace_adapter()
