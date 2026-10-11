"""Opt-in entry point: use the shared factory and existing WebSocket server."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .loader import LoadSpec, inspect_config, load_adapter, read_stats, seal_descriptor


def serve_metadata(adapter):
    metadata={"r17_manifest": asdict(adapter.manifest), "r17_model_spec": asdict(adapter.metadata()),
        "r17_samplers": {str(k):asdict(adapter.sampler_for_steps(k)) for k in
            ((10,) if adapter.wire.protocol.fixed_seed is not None else (10,2,1))},
        "r17_execution_protocol": asdict(adapter.wire.protocol),
        "batch_policy": "B1_only; batch-composition GPU parity unqualified",
        "loader_parity": "raw-Wan vs published sidecar unverified"}
    if adapter.batch_config is not None:
        metadata.update(batch_policy=adapter.batch_config.numerics_policy,
            r17_batching=adapter.batch_config.metadata(),
            batch_runtime_capabilities={f'batch_stage{i}':'enabled_study' for i in (1,2,3)})
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", required=True, help="JSON LoadSpec with immutable content digests")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="caller-selected free owned port")
    parser.add_argument("--cache-config")
    parser.add_argument("--manifest-output")
    parser.add_argument("--max-batch", "--max-batch-size", dest="max_batch_size", type=int, default=1,
        help="default-off LIBERO native BF16 batch study; maximum 16")
    parser.add_argument("--batch-wait-ms", type=float, default=5)
    parser.add_argument("--batch-ledger",type=Path,help="required for B>1: stage and decision composition JSONL")
    parser.add_argument("--recipe-factory", help="optional P1/P6 factory module:function(config,storage)")
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument("--capture-sink", help="optional P6 sink module:function(DecisionCapture)")
    capture.add_argument("--debug-directory", type=Path,
        help="P6 per-connection capture bound to the actual bundle/model/sampler")
    parser.add_argument("--seal-output", help="offline content sealing; does not load model")
    parser.add_argument("--inspect", action="store_true", help="small config/norm probe only; no weight identity claim")
    parser.add_argument("--protocol-info", action="store_true", help="labelled execution protocol only; no model load")
    args = parser.parse_args()
    spec = LoadSpec(**json.loads(Path(args.descriptor).read_text()))
    if args.protocol_info:
        from .protocols import execution_protocol
        print(json.dumps(asdict(execution_protocol(spec.benchmark, spec.execution_protocol)), indent=2))
        return
    if args.inspect:
        print(json.dumps({"config": inspect_config(spec), "stats": read_stats(spec),
            "status": "diagnostic_config_only; weights not loaded/hashed"}, indent=2))
        return
    if args.seal_output:
        Path(args.seal_output).write_text(json.dumps({**asdict(spec), "expected_digests": seal_descriptor(spec)}, indent=2)+"\n")
        return
    if not args.cache_config or not args.port:
        parser.error("serving requires --cache-config and a free owned --port")
    from .batch_runtime import BatchConfig,create_runtime
    try:
        BatchConfig(args.max_batch_size,args.batch_wait_ms)
        if args.max_batch_size>1 and (spec.benchmark!='libero' or args.batch_ledger is None):
            raise ValueError("Fast-WAM requires --max-batch-size=1 unless LIBERO --batch-ledger is supplied for an explicit BF16 batching study")
    except ValueError as exc:
        parser.error(str(exc))
    adapter = load_adapter(spec, device=args.device)
    from .provider import register_provider, make_connection_factory
    from exp.r17_integration.common.serving import make_bundle_validator
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    runtime=create_runtime(adapter,args.max_batch_size,args.batch_wait_ms,args.batch_ledger)
    def callback(name):
        if name is None:
            return None
        import importlib
        module, attr = name.rsplit(":", 1)
        return getattr(importlib.import_module(module), attr)
    capture_factory = None
    try:
        provider = register_provider(adapter,coordinator=runtime.core) if runtime else register_provider(adapter)
        config = load_cache_config(args.cache_config)
        storage = build_shared_storage(config)
        if args.debug_directory:
            from exp.r17_integration.debug.capture_bridge import make_capture_sink_factory
            capture_factory = make_capture_sink_factory(provider, args.debug_directory)
        factory = make_connection_factory(provider, config, storage,
            recipe_factory=callback(args.recipe_factory), capture_sink=callback(args.capture_sink),
            capture_sink_factory=capture_factory)
        if runtime is not None:
            factory=runtime.wrap_connection_factory(factory)
        if args.manifest_output:
            Path(args.manifest_output).write_text(json.dumps(asdict(adapter.manifest), indent=2)+"\n")
        server = WebsocketPolicyServer(factory(), host=args.host, port=args.port,
            metadata=serve_metadata(adapter), concurrent=True,
            connection_policy_factory=factory, bundle_validator=make_bundle_validator(provider))
        server.serve_forever()
    finally:
        if runtime is not None:
            runtime.close()
        if capture_factory is not None:
            capture_factory.close()


if __name__ == "__main__":
    main()
