"""Bounded future fixed-observation GPU qualification; never invoked during P5 CPU work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observations", required=True, help="NPZ: images[N,480,480,3], states[N,4], prompts[N] (no pickle)")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--tokenizer-file", action="append", required=True)
    parser.add_argument("--code-file", action="append", required=True)
    parser.add_argument("--k", type=int, choices=(4, 2, 1), default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline-seconds", type=int, default=840)
    args = parser.parse_args()
    if not 1 <= args.deadline_seconds <= 840:
        parser.error("deadline must be at most 840 seconds (14 minutes)")
    import numpy as np
    import torch
    from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
    from exp.r17_integration.common.request_rng import RequestRNG
    from exp.r17_integration.common.serving import full_call
    from .loader import load_checkpoint
    from .provider import GrootMetaWorldAdapter
    from .wire_policy import GrootMetaWorldWirePolicy
    def measured(call):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(args.device))
        start = time.perf_counter()
        value = call()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(args.device))
        return value, (time.perf_counter() - start) * 1000

    begin = time.monotonic()
    report = {"schema": "r17.verification_receipt.v1", "device": args.device, "k": args.k,
              "passed": 0, "status": "partial", "checks": [],
              "limits": ["fixed observations only; live WS/renderer/quality gates require coordinator receipts"]}
    receipt = Path(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(report, indent=2) + "\n")  # external timeout still leaves a partial receipt
    try:
        backend, manifest = load_checkpoint(args.checkpoint, device=args.device, denoising_steps=args.k,
                                            tokenizer_files=args.tokenizer_file, code_files=args.code_file)
        report["model_manifest"] = manifest.to_dict()
        adapter = GrootMetaWorldAdapter(backend, manifest)
        with np.load(args.observations, allow_pickle=False) as fixture:
            images, states, prompts = (fixture[name].copy() for name in ("images", "states", "prompts"))
        if not len(images) == len(states) == len(prompts) or not len(images):
            raise ValueError("observation fixture columns differ or are empty")
        for index in range(min(2, len(images))):
            if time.monotonic() - begin >= args.deadline_seconds:
                raise TimeoutError("P5 qualification deadline")
            obs = {"image": images[index], "state": states[index], "prompt": str(prompts[index])}
            request = RequestContext("P5-G2-fixed", manifest.benchmark_protocol, "reach-v3", index, 0)
            prepared, prepare_ms = measured(lambda: adapter.prepare(obs, request))
            s1, key_ms = measured(lambda: adapter.run_stage1(prepared))
            s2, llm_ms = measured(lambda: adapter.run_stage2(s1, capture=True))
            result, sample_ms = measured(lambda: adapter.sample(s2, adapter.metadata().default_sampler,
                                                    RequestRNG(request), CaptureSpec(True, True)))
            wire, wire_ms = measured(lambda: adapter.to_wire(result.normalized_chunk, prepared))
            device_index = torch.device(args.device).index or 0
            devices = [device_index] if args.device.startswith("cuda") else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(RequestRNG(request).seed_for())
                upstream = GrootMetaWorldWirePolicy(backend).infer(obs)["actions"]
            # Real GPU arithmetic tolerance is recorded; CPU tests require exact equality.
            np.testing.assert_allclose(wire.actions, upstream, rtol=1e-4, atol=1e-4)
            plain, plain_wire = full_call(adapter, obs, request)
            if not torch.equal(result.normalized_chunk, plain.normalized_chunk):
                raise AssertionError("capture on/off normalized bytes differ")
            np.testing.assert_array_equal(wire.actions, plain_wire.actions)
            prepared = adapter.prepare(obs, request)
            s1 = adapter.run_stage1(prepared)
            fields = adapter.retrieval_fields(s1)
            s2 = adapter.run_stage2(s1)
            for snapshot in result.snapshots:
                resumed = adapter.resume(s2, snapshot, result.sampler, RequestRNG(request))
                if not torch.equal(resumed.normalized_chunk, result.normalized_chunk):
                    raise AssertionError("native snapshot suffix differs")
            report["checks"].append({"observation": index, "wire_shape": list(wire.actions.shape),
                "key_shapes": {name: list(value.shape) for name, value in fields.items()},
                "actual_nfe": result.actual_nfe, "native_grid": list(result.sampler.native_timesteps),
                "rtol": 1e-4, "atol": 1e-4, "debug_bytes_equal": True, "suffixes": len(result.snapshots),
                "timing_ms": {"prepare": prepare_ms, "stage1_and_key_transfer": key_ms,
                              "Ckey": prepare_ms + key_ms, "stage2_llm": llm_ms,
                              "stage3_native_k": sample_ms, "wire": wire_ms},
                "timing_scope": "isolated synchronized capture-on qualification; cold/preceding probes included"})
            report["passed"] += 1
        if time.monotonic() - begin >= args.deadline_seconds:
            raise TimeoutError("P5 qualification deadline exceeded")
        report["status"] = "passed_fixed_observations"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - begin
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
