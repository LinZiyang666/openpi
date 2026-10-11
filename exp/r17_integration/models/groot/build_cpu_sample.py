"""Build explicitly diagnostic P5 handoff tensors without vendor/checkpoint/GPU use."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch


def main():
    import numpy as np
    import yaml
    from exp.r17_integration.common.contracts import CaptureSpec, DecisionCapture, ModelManifest, RequestContext
    from exp.r17_integration.common.provider import binding_for
    from exp.r17_integration.common.request_rng import RequestRNG
    from exp.r17_integration.tests.models.groot.conftest import Backend, Eagle, Head
    from .manifest import GrootManifest, digest, library_metadata
    from .provider import GrootMetaWorldAdapter, make_provider
    backend = Backend()
    manifest = GrootManifest(*(digest(name) for name in ("stub-weights", "stub-norm", "stub-code", "stub-transform", "stub-tokenizer")), hidden_width=8)
    # ONLY test vendor pins are replaced in this diagnostic process; production guards remain on.
    with patch("openpi.cache.groot.staged.UPSTREAM_FORWARD_SHA256", hashlib.sha256(inspect.getsource(Eagle.forward).encode()).hexdigest()), patch("openpi.cache.groot.staged.UPSTREAM_ACTION_HEAD_SHA256", hashlib.sha256(Path(inspect.getsourcefile(Head)).read_bytes()).hexdigest()):
        adapter = GrootMetaWorldAdapter(backend, manifest)
        obs = {"image": np.full((480, 480, 3), 2, np.uint8), "state": np.array([.1, .2, .3, .4], np.float32), "prompt": "reach"}
        request = RequestContext("P5-CPU-stub", manifest.benchmark_protocol, "reach-v3", 10, 0)
        prepared = adapter.prepare(obs, request)
        s1 = adapter.run_stage1(prepared)
        s2 = adapter.run_stage2(s1)
        result = adapter.sample(s2, adapter.metadata().default_sampler, RequestRNG(request), CaptureSpec(True, True))
        wire = adapter.to_wire(result.normalized_chunk, prepared)
        decision = DecisionCapture(request, "policy", 0, 0, (1, 1, 4), 4, result.normalized_chunk,
                                   wire.actions, adapter.retrieval_fields(s1), native_snapshots=result.snapshots)
        capture = adapter.make_trace_adapter().capture(decision)
        arrays = {**capture["fields"], "normalized_actions": capture["normalized_chunk"], "wire_actions": capture["wire_actions"],
                  "initial_noise": result.capture["initial_noise"].numpy()}
        for snapshot in result.snapshots:
            arrays[f"snapshot_{snapshot.loop_index}"] = snapshot.x_t.numpy()
        output = Path(__file__).parent / "samples"
        output.mkdir(exist_ok=True)
        np.savez_compressed(output / "cpu_stub_capture.npz", **arrays)
        meta = {"diagnostic_fixture": True, "real_checkpoint": False, "gpu_qualified": False,
                "model_manifest": asdict(ModelManifest(manifest.model_spec(), {"stub_source": hashlib.sha256(Path(inspect.getsourcefile(Backend)).read_bytes()).hexdigest()})),
                "library_metadata": library_metadata(manifest), "request": asdict(request),
                "native_grid": manifest.sampler_spec, "actual_nfe": result.actual_nfe,
                "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
                "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
                "snapshots": [{name: value for name, value in vars(snapshot).items() if name != "x_t"} for snapshot in result.snapshots]}
        (output / "cpu_stub_capture.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        provider = make_provider(backend, manifest)
        yaml_config = {"enabled": True, "timer": {"enabled": False},
                       "adapter_binding": binding_for(provider, manifest.benchmark_protocol, commit_horizon=10),
                       "key_builder": {"type": "adapter_native"},
                       "keys": {name: {"enabled": name in ("vision_0", "robot_state")} for name in
                                ("vision_0", "vision_1", "vision_2", "robot_state", "prompt_emb", "vlm_out")},
                       "backend": {"type": "in_memory", "vector_dims": dict(provider.model_spec.feature_spec.fields)},
                       "write_policy": {"type": "never"},
                       "checkpoints": {"cp1": {"gate": {"type": "always_search"}, "judge": {"type": "always_hit"},
                                               "search_strategy": {"type": "weighted_rrf_knn", "top_k": 1}}}}
        (output / "cpu_stub_bundle.yaml").write_text(yaml.safe_dump(yaml_config, sort_keys=False))
        print(json.dumps({"output": str(output), "actual_nfe": result.actual_nfe, "diagnostic_fixture": True}))


if __name__ == "__main__":
    main()
