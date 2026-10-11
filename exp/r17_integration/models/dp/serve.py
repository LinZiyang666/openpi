"""Explicit DP entrypoint using P1's factory + stock WebSocket/BatchingCore."""
from dataclasses import asdict


def callback(name):
    """Explicit deployment hook; never imported on the default-off path."""
    if name is None:
        return None
    import importlib
    module, attr = name.rsplit(":", 1)
    value = getattr(importlib.import_module(module), attr)
    if not callable(value):
        raise TypeError(f"DP hook {name!r} must be callable")
    return value


def plain_config(provider, *, num_steps=100, commit_horizon=5, retrieval=False,
                 allow_lossy_sampler=False, sampler_study_label=None, allow_partial_commit=False):
    from exp.r17_integration.common.provider import binding_for
    from exp.r17_integration.common.config import (BackendConfig, CacheConfig, CheckpointConfig, GateConfig,
                                     JudgeConfig, KeyBuilderConfig, KeysConfig, SearchStrategyConfig,
                                     WritePolicyConfig, validate_cache_config)
    keys = KeysConfig()
    for name, key in vars(keys).items():
        key.enabled = name in provider.model_spec.feature_spec.fields
    checkpoints = {}
    if retrieval:
        checkpoints = {"cp1": CheckpointConfig(gate=GateConfig(type="always_search"),
                         judge=JudgeConfig(type="always_hit"),
                         search_strategy=SearchStrategyConfig(type="weighted_rrf_knn", top_k=1))}
    config = CacheConfig(enabled=True, keys=keys, key_builder=KeyBuilderConfig(type="adapter_native"),
                         checkpoints=checkpoints,
                         backend=BackendConfig(type="in_memory", vector_dims=dict(provider.model_spec.feature_spec.fields)),
                         write_policy=WritePolicyConfig(type="never"),
                         adapter_binding=binding_for(provider, provider.model_spec.allowed_benchmark_protocols[0],
                                                     num_steps=num_steps, commit_horizon=commit_horizon,
                                                     allow_partial_commit=allow_partial_commit,
                                                     allow_lossy_sampler=allow_lossy_sampler,
                                                     sampler_study_label=sampler_study_label))
    provider.config_validator(config)
    validate_cache_config(config)
    return config


def main():
    import argparse
    import json
    from pathlib import Path
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-spec", type=Path, required=True, help="Pinned LoadSpec JSON; device is explicit")
    parser.add_argument("--cache-config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--batch-size", "--max-batch", dest="batch_size", type=int, default=1,
                        help="Opt-in DP-LIBERO matched batch configuration, 1..32")
    parser.add_argument("--batch-wait-ms", type=float, default=5)
    parser.add_argument("--batch-ledger", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--yaml-out", type=Path)
    parser.add_argument("--recipe-factory", help="P1/P6 module:function(config,storage); returns a connection-local recipe")
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument("--capture-sink", help="P6 module:function(DecisionCapture); observe-only capture")
    capture.add_argument("--debug-directory", type=Path, help="P6 native closed capture, bound to each connection's actual sampler")
    args = parser.parse_args()
    from .batch_runtime import BatchConfig, create_runtime
    try:
        BatchConfig(args.batch_size, args.batch_wait_ms)
        if args.batch_size > 1 and args.batch_ledger is None:
            raise ValueError("DP B>1 requires --batch-ledger and matched control/arm configuration")
    except ValueError as exc:
        parser.error(str(exc))
    from .loader import load_adapter
    from .provider import make_connection_factory, register_provider
    from .wire_policy import WirePolicy
    from exp.r17_integration.common.serving import make_bundle_validator
    from exp.r17_integration.common.config import build_shared_storage, load_cache_config
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    adapter = load_adapter(json.loads(args.load_spec.read_text()))
    provider = register_provider(adapter)
    config = load_cache_config(args.cache_config) if args.cache_config else plain_config(provider)
    if args.manifest_out:
        args.manifest_out.write_text(json.dumps(adapter.manifest(), indent=2, allow_nan=False) + "\n")
    if args.yaml_out:
        args.yaml_out.write_text(yaml.safe_dump(asdict(config)))
    storage = build_shared_storage(config)
    capture_factory = None
    if args.debug_directory:
        from exp.r17_integration.debug.capture_bridge import make_capture_sink_factory
        capture_factory = make_capture_sink_factory(provider, args.debug_directory)
    runtime = None
    try:
        runtime = create_runtime(adapter, args.batch_size, args.batch_wait_ms, args.batch_ledger)
        factory = make_connection_factory(adapter, provider=provider, default_config=config,
                                          default_storage=storage,
                                          stage_core=runtime.core if runtime else None,
                                          recipe_factory=callback(args.recipe_factory),
                                          capture_sink=callback(args.capture_sink), capture_sink_factory=capture_factory)
        if runtime:
            factory = runtime.wrap_connection_factory(factory)
        policy = WirePolicy(adapter)
        server = WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.get_metadata(),
                                       concurrent=True, connection_policy_factory=factory,
                                       bundle_validator=make_bundle_validator(provider))
        server.serve_forever()
    finally:
        if runtime:
            runtime.close()
        if capture_factory is not None:
            capture_factory.close()


if __name__ == "__main__":
    main()
