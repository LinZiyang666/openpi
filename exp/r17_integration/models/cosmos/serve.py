"""Official Cosmos LIBERO resident on the shared R17 factory and stock wire."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path


def serve_metadata(adapter):
    metadata = {"r17_model_spec": asdict(adapter.metadata()), "r17_manifest": asdict(adapter.manifest),
        "r17_samplers": {str(k): asdict(adapter.sampler_for_steps(k)) for k in (1, 2, 3, 5)},
        "official_config": adapter.cfg.config,
        "num_samples": 1, "value_seed_selection": False, "current_observation_only": True,
        "batch_policy": "B1 serialized per resident; stock proxy replicas",
        "unsupported": [name for name, cap in adapter.metadata().capabilities.features.items() if cap.status == "unsupported"]}
    if adapter.metadata().model_id == 'cosmos_mw':
        metadata.update(protocol_source=getattr(adapter.cfg, 'protocol_source',
                            'reconstructed; author runner and training config unavailable'),
                        predicted_chunk=16, execution_steps=5, trained_execution_horizon=None,
                        state_t=7, conditioning_frames=3, cameras=['corner2'],
                        image_resize=getattr(adapter.cfg, 'image_resize', 'bicubic'),
                        use_jpeg_compression=adapter.cfg.use_jpeg_compression)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--descriptor", required=True)
    p.add_argument("--cache-config", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    from .loader import load_adapter
    from .provider import register_provider
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    from exp.r17_integration.common.serving import make_bundle_validator, make_connection_factory
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    adapter = load_adapter(json.loads(Path(a.descriptor).read_text()), device=a.device)
    provider = register_provider(adapter)
    config = load_cache_config(a.cache_config)
    factory = make_connection_factory(provider, config, build_shared_storage(config))
    server = WebsocketPolicyServer(factory(), host=a.host, port=a.port, metadata=serve_metadata(adapter),
        concurrent=True, connection_policy_factory=factory, bundle_validator=make_bundle_validator(provider))
    server.serve_forever()


if __name__ == "__main__":
    main()
