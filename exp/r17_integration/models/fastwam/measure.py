"""Bounded, capture-off key/full phase measurement (no import-time model load)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time


def analytical_split(benchmark):
    """Reviewer parameter/token estimates, NOT measured runtime or FLOP counts.

    VAE/patchify FLOPs are unknown. Return coefficients so their unknown cost
    is never silently treated as zero in a claimed key/full fraction.
    """
    text = 2 * 4.6e9 * 128 / 1e12
    video_tokens = 98 if benchmark == "libero" else 49
    video = 2 * 5e9 * video_tokens / 1e12
    action = 2 * 1.02e9 * 32 * 10 / 1e12
    return {"scope": "analytical_parameter_times_token_estimate; not latency",
        "umt5_tflop_estimate": text, "video_prefill_tflop_estimate": video,
        "k10_action_tflop_estimate": action, "vae_patchify_tflop": None,
        "old_key_formula": f"VAE_patchify + {text:.4f} TFLOP",
        "new_key_formula": "VAE_patchify",
        "full_formula": f"VAE_patchify + {text+video+action:.4f} TFLOP",
        "key_text_calls_before_after": [1,0], "full_text_calls_before_after": [1,1],
        "measured_key_full_ratio": None}


def measure_adapter(adapter, observations, *, deadline_seconds=780, repeats=3):
    import torch
    from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
    from exp.r17_integration.common.request_rng import RequestRNG
    started = time.monotonic()
    rows = []
    receipt = {"schema": "r17.fastwam.cost_split.v1", "status": "partial",
        "model_identity": adapter.metadata().identity, "device": str(adapter.model.device),
        "capture_enabled": False, "capture_cp2_enabled": False, "batch_size": 1,
        "timing_scope": "synchronized_B1_model_path; formal_policy_cost_uses_P1_host_wall_ledger",
        "cost_definitions": {"key_ms": "prepare_ms + stage1_ms",
            "hit_path_ms": "prepare_ms + stage1_ms + wire_ms",
            "retrieval_search_ms": "excluded; not measured or assumed zero",
            "wire_ms": "full-output transform proxy; actual HIT transform is not timed here"},
        "rows": rows, "analytical": analytical_split(adapter.wire.benchmark),
        "limits": ["synchronized B1 profiling; excludes retrieval/search/transport/controller",
                   "CPU stubs establish call boundaries only, not real model pricing",
                   "published sidecar/raw-Wan loader parity unqualified"]}
    if not observations or repeats < 1:
        raise ValueError("nonempty observations and positive repeats required")
    def sync():
        if adapter.model.device.type == "cuda":
            torch.cuda.synchronize(adapter.model.device)
    def timed(operation):
        sync(); tick = time.perf_counter()
        result = operation()
        sync()
        return result, (time.perf_counter()-tick)*1000
    saved_observer = adapter.phase_observer
    try:
        # One capture-off warmup excluded from all measured rows.
        for index in range(repeats+1):
            if time.monotonic()-started >= deadline_seconds:
                break
            obs = observations[index % len(observations)]
            req = RequestContext("p3_cost_split", adapter.metadata().allowed_benchmark_protocols[0],
                "fixed_observation", index, 0, explicit_seed=42)
            phases = {}
            def observe(name, operation):
                result, ms = timed(operation)
                phases[name] = phases.get(name, 0.) + ms
                return result
            adapter.phase_observer = observe
            prepared, prepare_ms = timed(lambda: adapter.prepare(obs,req))
            s1, stage1_ms = timed(lambda: adapter.run_stage1(prepared))
            key_phases = dict(phases)
            s2, miss_ms = timed(lambda: adapter.run_stage2(s1))
            result, sample_ms = timed(lambda: adapter.sample(s2,adapter.sampler_for_steps(10),RequestRNG(req),CaptureSpec()))
            _, wire_ms = timed(lambda: adapter.to_wire(result.normalized_chunk,prepared))
            if index:
                key_ms = prepare_ms+stage1_ms
                hit_path_ms = key_ms+wire_ms
                full_ms = hit_path_ms+miss_ms+sample_ms
                rows.append({"index": index, "k": 10, "actual_nfe": result.actual_nfe,
                    "capture_enabled": False, "pricing_eligible": True,
                    "prepare_ms": prepare_ms, "stage1_ms": stage1_ms, "key_ms": key_ms,
                    "hit_path_ms": hit_path_ms, "retrieval_search_ms": None,
                    "miss_conditioning_prefill_ms": miss_ms,
                    "sample_ms": sample_ms, "wire_ms": wire_ms, "full_ms": full_ms,
                    "key_over_full": key_ms/full_ms, "hit_over_full": hit_path_ms/full_ms,
                    "key_phases_ms": key_phases, "full_phases_ms": phases})
        if len(rows) == repeats and time.monotonic()-started < deadline_seconds:
            receipt["status"] = "passed"
    finally:
        adapter.phase_observer = saved_observer
    receipt["elapsed_seconds"] = time.monotonic()-started
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--observations", required=True, help="real paired RGB frames/state/prompt NPZ; no pickle")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--deadline-seconds", type=float, default=780)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    import numpy as np
    from .loader import LoadSpec, load_adapter
    started = time.monotonic()
    try:
        spec = LoadSpec(**json.loads(Path(args.descriptor).read_text()))
        adapter = load_adapter(spec,device=args.device)
        with np.load(args.observations,allow_pickle=False) as data:
            observations = [{k:str(data[k][i]) if k=="prompt" else data[k][i].copy()
                for k in data.files} for i in range(min(3,len(data["state"])))]
        receipt = measure_adapter(adapter, observations, repeats=args.repeats,
            deadline_seconds=max(0,args.deadline_seconds-(time.monotonic()-started)))
    except Exception as exc:
        receipt = {"schema":"r17.fastwam.cost_split.v1", "status":"failed", "error":f"{type(exc).__name__}: {exc}"}
    receipt["elapsed_with_load_seconds"] = time.monotonic()-started
    Path(args.receipt).write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
