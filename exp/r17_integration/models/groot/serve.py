"""Thin assembly for the shared WS/cache paths; no independent server protocol."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

from openpi.cache.groot.interceptor import GrootCacheInterceptor
from openpi.cache.groot.staged import GrootStagedRunner
from .manifest import validate_library
from .wire_policy import GrootMetaWorldWirePolicy


def make_bundle_validator(provider):
    """P1 validates the sealed recipe/artifact before the WS publishes a bundle."""
    from exp.r17_integration.common.serving import make_bundle_validator as shared_validator
    return shared_validator(provider)


def make_connection_factory(provider, *args, **kwargs):
    """P1's layout-aware full/CP1/K/five-layer/cursor executor, on the original WS server."""
    from exp.r17_integration.common.serving import make_connection_factory as make
    return make(provider, *args, **kwargs)


@dataclass(frozen=True)
class StockBundle:
    """Local certified-factory shim for stock GR advanced hooks. Registry is explicit."""
    manifest: Any
    miss_num_steps: int
    make_orchestrator: Any = None
    trace_factory: Any = None
    warm_reset_factory: Any = None
    library_meta: dict | None = None
    storage: Any = None


def make_stock_connection_factory(backend, manifest, bundles, *, runner_factory=GrootStagedRunner, coordinator=None):
    """Preserve original GR interceptor branches and per-connection ownership.

    A bundle supplies an already validated stock component/orchestrator factory.
    P1 owns YAML registration. This shim is never installed by import.
    """
    model_lock = threading.RLock()

    def factory(shared_base_policy, bundle_id="default"):
        del shared_base_policy
        try:
            bundle = bundles[bundle_id]
        except KeyError:
            raise ValueError(f"unknown GR00T-MW bundle {bundle_id!r}") from None
        if bundle.manifest != manifest:
            raise ValueError("bundle model/normalization/geometry differs from endpoint")
        if bundle.miss_num_steps != manifest.denoising_steps:
            raise ValueError("bundle MISS K differs from endpoint")
        if bundle.library_meta is not None:
            validate_library(bundle.library_meta, manifest)
        runner = runner_factory(backend.model)
        if runner.live_schedule().num_steps != manifest.denoising_steps:
            raise RuntimeError("live GR00T endpoint K changed")
        orchestrator = bundle.make_orchestrator(bundle.storage) if bundle.make_orchestrator else None
        trace = bundle.trace_factory(runner, orchestrator, bundle_id) if bundle.trace_factory else None
        warm = bundle.warm_reset_factory(runner, orchestrator) if bundle.warm_reset_factory else None
        kwargs = {}
        if trace is not None:
            kwargs.update(trace=trace, coordinator=coordinator, model_lock=model_lock,
                          bundle_id=bundle_id, trace_vision_fields=("vision_0",))
        elif warm is not None:
            kwargs.update(warm.interceptor_kwargs() if hasattr(warm, "interceptor_kwargs") else {"warm_reset": warm})
        else:
            kwargs.update(miss_num_steps=bundle.miss_num_steps)
        interceptor = GrootCacheInterceptor(backend, runner, orchestrator=orchestrator, **kwargs)
        inner = warm.wrap(interceptor) if warm is not None and hasattr(warm, "wrap") else interceptor
        # Trace holds the lock around stage1/2 internally and releases before S3.
        return GrootMetaWorldWirePolicy(inner, manifest, lock=None if trace is not None else model_lock)

    return factory


def serve(backend, manifest, *, port, default_config, default_storage, host="0.0.0.0",
          capture_sink=None, capture_sink_factory=None, debug_directory=None, recipe_factory=None, provider=None):
    from exp.r17_integration.common.recipe import make_recipe_factory
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from .provider import make_provider, register_provider
    provider = register_provider(provider or make_provider(backend, manifest))
    if debug_directory is not None and (capture_sink is not None or capture_sink_factory is not None):
        raise ValueError("debug_directory owns its capture factory; choose one capture option")
    owned_capture = None
    if debug_directory is not None:
        from exp.r17_integration.debug.capture_bridge import make_capture_sink_factory
        owned_capture = make_capture_sink_factory(provider, debug_directory)
        capture_sink_factory = owned_capture
    # Recipe choice is entirely in each versioned YAML bundle, independent of MISS K.
    recipe_factory = recipe_factory or make_recipe_factory(provider)
    factory = make_connection_factory(provider, default_config, default_storage, capture_sink=capture_sink,
                                      capture_sink_factory=capture_sink_factory, recipe_factory=recipe_factory)
    base = GrootMetaWorldWirePolicy(backend, manifest)
    server = WebsocketPolicyServer(base, host=host, port=port, metadata=base.get_metadata(), concurrent=True,
                                   connection_policy_factory=factory,
                                   bundle_validator=make_bundle_validator(provider))
    # Stop only this explicitly started server and seal its own debug writers.
    import signal
    previous_signal = None
    if threading.current_thread() is threading.main_thread():
        previous_signal = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *args: server.request_stop())
    try:
        server.serve_forever(stop_on_request=True)
    finally:
        if previous_signal is not None:
            signal.signal(signal.SIGTERM, previous_signal)
        if owned_capture is not None:
            owned_capture.close()


def default_config(provider, *, commit_horizon=5):
    """Fresh full-policy default on an empty stock backend; no library or debug work."""
    from exp.r17_integration.common.provider import binding_for
    from exp.r17_integration.common.config import (BackendConfig, CacheConfig, CheckpointConfig, GateConfig, JudgeConfig,
                                    KeyBuilderConfig, KeysConfig, RecipeConfig, SearchStrategyConfig, TimerConfig,
                                    WritePolicyConfig)
    from exp.r17_integration.common.benchmarks import get_benchmark, task_table_identity
    keys = KeysConfig()
    for name, value in vars(keys).items():
        value.enabled = name in provider.model_spec.feature_spec.fields
    protocol = provider.model_spec.allowed_benchmark_protocols[0]
    # Inert (disabled) recipe still carries the complete canonical task table so
    # every request's task_name/task_id is checked against it before model work.
    table = RecipeConfig(ordered_tasks=list(get_benchmark(protocol).ordered_tasks),
                         task_table_identity=task_table_identity(protocol))
    return CacheConfig(enabled=True, timer=TimerConfig(enabled=False), keys=keys, recipe=table,
        key_builder=KeyBuilderConfig(type="adapter_native"),
        checkpoints={"cp1": CheckpointConfig(gate=GateConfig(type="always_search"), judge=JudgeConfig(type="always_hit"),
                            search_strategy=SearchStrategyConfig(type="weighted_rrf_knn", top_k=1))},
        backend=BackendConfig(type="in_memory", vector_dims=dict(provider.model_spec.feature_spec.fields)),
        write_policy=WritePolicyConfig(type="never"),
        adapter_binding=binding_for(provider, provider.model_spec.allowed_benchmark_protocols[0], commit_horizon=commit_horizon))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Opt-in GR00T-MW staged server on the shared WS/cache path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", choices=("metaworld", "libero"), default="metaworld")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--denoising-steps", type=int, choices=(4, 3, 2, 1), default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-config")
    parser.add_argument("--debug-directory", help="P6 model debug capture root; default off, bound per connection/bundle")
    parser.add_argument("--commit-horizon", type=int, default=5)
    parser.add_argument("--tokenizer-file", action="append", required=True)
    parser.add_argument("--code-file", action="append", required=True)
    args = parser.parse_args(argv)
    from .loader import load_checkpoint, checkpoint_paths
    from .provider import make_provider, register_provider
    from exp.r17_integration.common.config import build_shared_storage, load_cache_config, validate_cache_config
    checkpoint = checkpoint_paths(args.checkpoint) if args.benchmark == "libero" else args.checkpoint
    backend, manifest = load_checkpoint(checkpoint, device=args.device, denoising_steps=args.denoising_steps,
                                        tokenizer_files=args.tokenizer_file, code_files=args.code_file, benchmark=args.benchmark)
    provider = register_provider(make_provider(backend, manifest))
    config = load_cache_config(args.cache_config) if args.cache_config else default_config(provider, commit_horizon=args.commit_horizon)
    validate_cache_config(config)
    storage = build_shared_storage(config)
    serve(backend, manifest, host=args.host, port=args.port, default_config=config, default_storage=storage,
          provider=provider, debug_directory=args.debug_directory)


if __name__ == "__main__":
    main()
