"""One-load, bounded Q-input diagnostic of whole-compiled versus R16 stages.

This is an explicit GPU command, never imported by serving. No GPU-slot lock,
server, renderer, training process or qualification asset is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import threading
import time


def action_difference(left, right):
    import numpy as np
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("action geometry/dtype mismatch")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite actions")
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"max_abs_diff": float(difference.max()), "mean_abs_diff": float(difference.mean()),
            "exact": left.tobytes() == right.tobytes(), "shape": list(left.shape),
            "dtype": str(left.dtype)}


def native_grid_witness(k):
    """Execute the actual stock and R16 Euler implementations with a CPU head.

    The compiled stock graph originates from this same sample_actions method;
    recording values inside its compiled graph would change that graph.
    """
    from types import SimpleNamespace as NS
    import torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Stage1Output, Stage2Output
    state = torch.zeros(1, 32)
    stage1 = Stage1Output(state, torch.zeros(1, 1, 1), torch.ones(1, 1, dtype=torch.bool),
                         torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, dtype=torch.long))
    stage2 = Stage2Output(stage1, ())
    class Head:
        _stage3_action_expert = PI0Pytorch._stage3_action_expert
        _stage3_with_intermediates = PI0Pytorch._stage3_with_intermediates
        run_stage3 = PI0Pytorch.run_stage3
        sample_actions = PI0Pytorch.sample_actions
        def __init__(self): self.times = []
        def _stage1_token_prep(self, obs):
            return (stage1.state, stage1.prefix_embs, stage1.prefix_pad_masks,
                    stage1.prefix_att_2d_masks_4d, stage1.prefix_position_ids)
        def _stage2_llm_backbone(self, *args): return ()
        def denoise_step(self, state, pads, kv, x, t):
            self.times.append(float(t[0]))
            return x * .125 + t[:, None, None] * .01
    noise = torch.arange(160, dtype=torch.float32).reshape(1, 5, 32) / 100
    head = Head()
    stock = head.sample_actions("cpu", NS(state=state), noise=noise, num_steps=k)
    stock_times = head.times[:]
    head.times.clear()
    staged = head.run_stage3(stage2, noise=noise, num_steps=k, return_intermediates=True).action_chunk
    return {"stock": stock_times, "r16_staged": head.times,
            "dt_float32": float(torch.tensor(-1 / k, dtype=torch.float32)),
            "cpu_native_loop_exact": torch.equal(stock, staged),
            "compiled_grid_evidence": "source plus execution of original CPU loop; no compiled graph instrumentation"}


def r16_reference(adapter, obs, request, k):
    """R16 concurrent B1 MISS: eager public stages, return_intermediates=True.

    Independent stock transforms/tensor preparation and original model calls;
    intentionally does not call adapter.prepare/sample/to_wire or full_call.
    """
    import numpy as np
    import torch
    from openpi.models.model import Observation
    from exp.r17_integration.common.request_rng import RequestRNG
    inputs = adapter.policy._input_transform({key: np.array(value, copy=True)
        if isinstance(value, np.ndarray) else value for key, value in obs.items()})
    tensors = {key: {name: torch.from_numpy(np.array(value)).to(adapter.policy._pytorch_device)[None]
                         for name, value in values.items()} if isinstance(values, dict)
               else torch.from_numpy(np.array(values)).to(adapter.policy._pytorch_device)[None]
               for key, values in inputs.items()}
    observation = Observation.from_dict(tensors)
    noise = torch.from_numpy(RequestRNG(request).normal((5, 32))).to(adapter.policy._pytorch_device)[None]
    with torch.no_grad():
        stage1 = adapter.model.run_stage1(observation)
        stage2 = adapter.model.run_stage2(stage1)
        result = adapter.model.run_stage3(stage2, noise=noise, num_steps=k, return_intermediates=True)
    normalized = np.array(result.action_chunk[0].detach().cpu(), copy=True)
    wire = adapter.policy._output_transform({"state": np.array(inputs["state"], copy=True),
                                             "actions": normalized.copy()})["actions"]
    return normalized, wire


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    job = json.loads(Path(args.job).read_text())
    report = {"status": "PARTIAL", "job": args.job, "model_loads": 0, "checks": [],
              "synthetic_input": True, "quality_claim": False, "pricing_claim": False,
              "gpu_lock_taken": False, "deadline_seconds": 870, "allocator_cap_mib": 12288,
              "max_process_gpu_mib": 0, "source_digests": {}}
    stop = threading.Event()
    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    def monitor():
        while not stop.wait(1):
            try:
                result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
                for line in result.stdout.splitlines():
                    pid, memory = line.split(",")
                    if int(pid.strip()) == os.getpid():
                        report["max_process_gpu_mib"] = max(report["max_process_gpu_mib"], int(memory.strip()))
            except (ValueError, subprocess.TimeoutExpired):
                pass
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    save()
    try:
        import numpy as np
        import torch
        from exp.r17_integration.qualification.model_runtime import load, wire_observations
        from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
        from exp.r17_integration.common.request_rng import RequestRNG
        from exp.r17_integration.common.serving import full_call
        from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, decision_seed
        from .adapter import sampler_for_steps
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(12288 * 1024 ** 2 / total, 0)
        monitor_thread.start()
        adapter, _ = load(job)
        report["model_loads"] = 1
        report["model_spec"] = __import__("dataclasses").asdict(adapter.metadata())
        for name in ("src/openpi/models_pytorch/pi0_pytorch.py", "src/openpi/policies/policy.py",
                     "src/openpi/cache/interceptor.py", "scripts/serve_policy.py",
                     "exp/offline_search/rounds/r16/replay.py", __file__):
            report["source_digests"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        print("MODEL_LOADED_ONCE", time.monotonic() - started, flush=True)
        observations = wire_observations(job)
        for k in (2, 1):
            grid = native_grid_witness(k)
            assert grid["stock"] == grid["r16_staged"] == list(sampler_for_steps(k).native_timesteps)
            for idx, obs in enumerate(observations):
                if time.monotonic() - started > 820:
                    raise TimeoutError("bounded diagnostic phase deadline")
                req = RequestContext("Q-pi-native", FORMAL_SPEC.identity, "reach-v3", 0, idx,
                    arm="qualification", attempt=1, bundle_generation="native-v1", task_id=43,
                    explicit_seed=decision_seed("reach-v3", 0, idx))
                adapter.policy._sample_kwargs = {"num_steps": k}
                adapter.policy._staged_inference = False
                print("STOCK_COMPILED", k, idx, flush=True)
                stock = adapter.policy.infer(obs, noise=RequestRNG(req).normal((5, 32)))["actions"]
                plain, wire = full_call(adapter, obs, req, sampler=sampler_for_steps(k))
                debug, debug_wire = full_call(adapter, obs, req, sampler=sampler_for_steps(k), capture=CaptureSpec(True, True))
                normalized, reference = r16_reference(adapter, obs, req, k)
                # Remove only the whole-call compiler wrapper, not any model parameter.
                compiled = adapter.policy._sample_actions
                eager = getattr(compiled, "_torchdynamo_orig_callable", None)
                if eager is None:
                    raise AssertionError("stock compiled callable has no original eager method")
                try:
                    adapter.policy._sample_actions = eager
                    stock_eager = adapter.policy.infer(obs, noise=RequestRNG(req).normal((5, 32)))["actions"]
                finally:
                    adapter.policy._sample_actions = compiled
                suffixes = []
                if debug.snapshots:
                    prepared = adapter.prepare(obs, req)
                    stage2 = adapter.run_stage2(adapter.run_stage1(prepared))
                    for snapshot in debug.snapshots:
                        suffix = adapter.resume(stage2, snapshot, debug.sampler, RequestRNG(req))
                        suffixes.append(action_difference(debug.normalized_chunk, suffix.normalized_chunk))
                    del prepared, stage2
                row = {"k": k, "observation": idx, "grid": grid,
                       "stock_compiled_vs_adapter": action_difference(stock, wire.actions),
                       "stock_eager_vs_adapter": action_difference(stock_eager, wire.actions),
                       "r16_vs_adapter": action_difference(reference, wire.actions),
                       "r16_normalized_vs_adapter": action_difference(normalized, plain.normalized_chunk),
                       "debug_vs_plain": action_difference(wire.actions, debug_wire.actions),
                       "suffixes": suffixes, "actual_nfe": plain.actual_nfe,
                       "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
                       "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2}
                report["checks"].append(row)
                save()
                print(json.dumps(row), flush=True)
                assert row["r16_vs_adapter"]["exact"] and row["r16_normalized_vs_adapter"]["exact"]
                assert row["debug_vs_plain"]["exact"] and row["stock_eager_vs_adapter"]["exact"]
                assert all(s["exact"] for s in suffixes)
                assert plain.actual_nfe == debug.actual_nfe == k
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        stop.set()
        if monitor_thread.is_alive(): monitor_thread.join(timeout=4)
        save()


if __name__ == "__main__":
    main()
