"""Reproduce explicitly synthetic CPU samples; never load a checkpoint/renderer."""
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path


def main():
    if os.getenv("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("sample generation requires CUDA_VISIBLE_DEVICES='' ")
    import numpy as np
    from exp.r17_integration.common.contracts import CaptureSpec, DecisionCapture, RequestContext
    from exp.r17_integration.common.request_rng import RequestRNG
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, MetaWorldBenchmarkAdapter, decision_seed
    from exp.r17_integration.benchmarks.physical_mt50 import NativeMetaWorldPhysicalAdapter
    from exp.r17_integration.debug.capture_bridge import CaptureBridge, DecisionSink
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    from exp.r17_integration.debug.validate import model_binding, plain
    from exp.r17_integration.models.pi05.adapter import sampler_for_steps
    from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter, observation
    from exp.r17_integration.tests.benchmarks.test_metaworld import Env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "samples/native_k2")
    directory = parser.parse_args().output
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "sample_manifest.json").exists():
        existing = json.loads((directory / "sample_manifest.json").read_text())
        assert existing["synthetic_fixture"] and not existing["checkpoint_loaded"] and not existing["renderer_used"]
        readback = list(CaptureReader(directory / "writer").decisions())
        assert len(readback) == 1 and readback[0][0]["executed_count"] == 3
        assert len(json.loads((directory / "formal_500.json").read_text())) == 500
        print(json.dumps({"samples": str(directory), "native_nfe": existing["decision"]["actual_nfe"],
                          "model_capture_readback": "passed", "synthetic_fixture": True, "reused_verified": True}))
        return
    req = RequestContext("p2_synthetic_cpu_sample", FORMAL_SPEC.identity, "reach-v3", 0, 0,
                         arm="full_fixture", attempt=1, bundle_generation="sample_gen1", task_id=43,
                         explicit_seed=decision_seed("reach-v3", 0, 0))
    adapter, model = fixture_adapter()
    obs = observation()
    prepared = adapter.prepare(obs, req)
    s1 = adapter.run_stage1(prepared)
    s2 = adapter.run_stage2(s1, capture=True)
    result = adapter.sample(s2, sampler_for_steps(2), RequestRNG(req), CaptureSpec(True, True))
    wire = adapter.to_wire(result.normalized_chunk, prepared)
    fields = adapter.retrieval_fields(s1)
    capture = DecisionCapture(req, "full", 0, 0, (1, 1, 2), result.actual_nfe,
        result.normalized_chunk, wire.actions, fields, result.rng_receipt,
        diagnostics={"synthetic_fixture": True, "neural_velocity": "ToyModel", "physics": "Env fixture"},
        native_snapshots=result.snapshots, model_capture=adapter.capture_fields(s1, s2, result))
    identity = {"campaign": req.campaign, "arm": req.arm, "bundle_generation": req.bundle_generation,
        "benchmark_protocol": FORMAL_SPEC.identity, "task_name": "reach-v3", "task_id": 43,
        "orig_init_idx": 0, "task_data_hash": "synthetic_fixture_not_native_MT1",
        "task_uid": "p2_sample:eval:43:0", "attempt": 1}
    bridge = CaptureBridge(directory / "writer", enabled=True,
                           binding=model_binding(adapter.metadata(), FORMAL_SPEC.identity, sampler=result.sampler))
    sink = DecisionSink(bridge, lambda request: identity)
    try:
        sink(capture)
        sink.on_execution({"request": req, "decision_seq": 0, "cursor": 3, "executed_count": 3, "terminal": True})
    finally:
        assert bridge.close()["drained"]
    readback = list(CaptureReader(directory / "writer").decisions())
    assert len(readback) == 1 and readback[0][0]["executed_count"] == 3
    environment = Env()
    physical = NativeMetaWorldPhysicalAdapter(environment, "reach-v3",
        contact_force=lambda m, d, i, force: force.__setitem__(slice(None), np.arange(6))).capture(
            np.zeros(39), {"success": False}, wire.actions[0], np.clip(wire.actions[0].astype(np.float32), -1, 1))
    np.savez_compressed(directory / "physical_fixture.npz", **physical["arrays"])
    np.savez_compressed(directory / "wire_observation.npz", image=obs["observation/image"],
                        state=obs["observation/state"], prompt=np.array(obs["prompt"]))
    description = {"schema": "r17.p2.samples.v1", "synthetic_fixture": True,
        "checkpoint_loaded": False, "renderer_used": False, "quality_sr_claim": False,
        "model_spec": asdict(adapter.metadata()), "benchmark": FORMAL_SPEC.manifest(),
        "native_grids": [asdict(sampler_for_steps(k)) for k in (10, 2, 1)],
        "fixture_quantiles": {"state_q01": [-2]*4, "state_q99": [3]*4,
                              "action_q01": [-3, -4, -5, -1], "action_q99": [2, 3, 4, 1]},
        "decision": {"request": asdict(req), "actual_nfe": result.actual_nfe, "stage_calls": model.counts,
                     "normalized_shape": list(result.normalized_chunk.shape), "wire_shape": list(wire.actions.shape),
                     "wire_dtype": str(wire.actions.dtype), "sampler_identity": result.sampler.identity},
        "physical": {"metadata": physical["metadata"], "capabilities": physical["capabilities"],
                     "arrays": {k: {"dtype": str(a.dtype), "shape": list(a.shape)} for k, a in physical["arrays"].items()}},
        "paths": {"model_capture": "writer", "physical": "physical_fixture.npz", "wire_input": "wire_observation.npz"}}
    (directory / "sample_manifest.json").write_text(json.dumps(plain(description), sort_keys=True, indent=2) + "\n")
    (directory / "formal_500.json").write_text(json.dumps(MetaWorldBenchmarkAdapter().episode_specs(),
                                                         sort_keys=True, indent=2) + "\n")
    print(json.dumps({"samples": str(directory), "native_nfe": result.actual_nfe,
                      "model_capture_readback": "passed", "synthetic_fixture": True}))


if __name__ == "__main__":
    main()
