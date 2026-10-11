"""Explicit R17 entry; original WebsocketPolicyServer and P1 factories."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from .adapter import content_manifest, load_adapter, model_spec, sampler_for_steps
from .provider import register_provider, config_document, make_connection_factory, make_capture_sink_factory


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--config", help="default YAML bundle")
    ap.add_argument("--manifest-only", action="store_true", help="content hashes only; never load model")
    ap.add_argument("--emit-configs", type=Path, help="emit content-bound K10/K2/K1 YAMLs without loading model")
    ap.add_argument("--recipe-store", type=Path, help="deprecated; specify recipe in each YAML bundle")
    ap.add_argument("--periodic-gap", type=float, help="deprecated; use YAML recipe.periodic_gap or calibrated target_ir")
    ap.add_argument("--debug-directory", type=Path, help="explicit P6 model capture directory; default off")
    a = ap.parse_args(argv)
    if a.manifest_only:
        print(json.dumps(content_manifest(a.checkpoint), sort_keys=True))
        return
    if a.emit_configs:
        import yaml
        from exp.r17_integration.common.provider import AdapterProvider
        manifest = content_manifest(a.checkpoint)
        spec = model_spec(manifest["checkpoint_identity"], manifest["norm_identity"], code_digest=manifest["code_digest"])
        provider = AdapterProvider("r17_pi05_mt50_v1", spec, lambda: None, sampler_for_steps)
        a.emit_configs.mkdir(parents=True, exist_ok=True)
        for k in (10, 2, 1):
            (a.emit_configs / f"k{k}.yaml").write_text(yaml.safe_dump(config_document(provider, k=k, evidence_dir=a.emit_configs / "miss_evidence")))
        (a.emit_configs / "model_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        return
    if not a.config:
        ap.error("--config required for explicit R17 serving")
    if a.recipe_store or a.periodic_gap is not None:
        ap.error("five-layer recipe selection belongs in each YAML's r17.recipe.v1 section")
    adapter = load_adapter({"checkpoint": a.checkpoint, "device": a.device})
    provider = register_provider(adapter)
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    from exp.r17_integration.common.serving import make_bundle_validator
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    config = load_cache_config(a.config)
    storage = build_shared_storage(config)
    from exp.r17_integration.common.recipe import make_recipe_factory
    sink_factory = None
    if a.debug_directory:
        sink_factory = make_capture_sink_factory(a.debug_directory)
    factory = make_connection_factory(provider, config, storage,
                                      recipe_factory=make_recipe_factory(provider),
                                      capture_sink_factory=sink_factory)
    # Base policy is built through the same connection factory and the server
    # resolves subsequent select_bundle/load_cache_config via stock controls.
    policy = factory(None, "default")
    metadata = {"r17_model_spec": asdict(adapter.metadata()), "benchmark": "metaworld_mt50",
                "r17_samplers": {str(k): asdict(sampler_for_steps(k)) for k in (10, 2, 1)},
                "r17_execution_protocol": "r17.execution.v1"}
    server = WebsocketPolicyServer(policy, host=a.host, port=a.port, metadata=metadata,
                                   concurrent=True, connection_policy_factory=factory,
                                   bundle_validator=make_bundle_validator(provider))
    import signal
    signal.signal(signal.SIGTERM, lambda *args: server.request_stop())
    try:
        server.serve_forever(stop_on_request=True)
    finally:
        if sink_factory:
            sink_factory.close()


if __name__ == "__main__":
    main()
