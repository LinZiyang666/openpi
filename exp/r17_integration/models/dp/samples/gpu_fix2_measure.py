"""One bounded, explicit GPU batch diagnostic using Q's original sealed job."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
from exp.r17_integration.models.dp.loader import load_adapter
from exp.r17_integration.models.dp.manifest import file_digest
from exp.r17_integration.models.dp.request_noise import RequestNoise
from exp.r17_integration.models.dp.sampler import SamplerProfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=270)
    args = parser.parse_args()
    start = time.monotonic()
    def deadline():
        if time.monotonic() - start > args.deadline_seconds:
            raise TimeoutError("GPU measurement deadline reached")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema="r17.p4.gpu_fix2.measurement.v1", status="running",
        gpu_executed=True, quality_claim=False, diagnostic_only=True,
        measured_release="GPU fix1 before production B1 gate; original Q pins preserved",
        job=str(args.job), job_sha256=file_digest(args.job), measurements=[])
    arrays = {}
    try:
        job = json.loads(args.job.read_text())
        assets = Path(job["assets"])
        load_path, obs_path = assets / "load.json", assets / "wire_observations.npz"
        document = json.loads(load_path.read_text())
        adapter = load_adapter(dict(document, device="cuda:0"))
        assert json.loads(json.dumps(asdict(adapter.metadata()))) == json.loads((assets / "model_spec.json").read_text())
        receipt.update(device=str(adapter.device), gpu=torch.cuda.get_device_name(0),
                       manifest=adapter.manifest(), load_sha256=file_digest(load_path),
                       observations_sha256=file_digest(obs_path), rows=[0, 1, 2, 0], seeds=[18, 37, 56, 75])
        with np.load(obs_path, allow_pickle=False) as data:
            observations = [{key: str(data[key][i]) if key == "prompt" else data[key][i].copy()
                             for key in data.files} for i in (0, 1, 2, 0)]
        context = RequestContext("p4_gpu_fix2", adapter.metadata().allowed_benchmark_protocols[0],
                                 "diagnostic_real_pairs", 0, 0, explicit_seed=18)
        prepared = [adapter.prepare(obs, context) for obs in observations]
        single_stages = [adapter.run_stage2(adapter.run_stage1(p)) for p in prepared]
        def metrics(one, two):
            delta = (one.double() - two.double()).abs()
            return dict(max_abs=float(delta.max()), max_rel=float((delta / two.double().abs().clamp_min(1e-12)).max()),
                        rmse=float(delta.square().mean().sqrt()), elements=delta.numel(),
                        old_tolerance_mismatches=int((delta > 1e-5 + 1e-4 * two.double().abs()).sum()))
        for k in (100, 2):
            deadline()
            profile = SamplerProfile(num_steps=k)
            baseline = [adapter.sample(s, profile, RequestNoise(seed, adapter.device), CaptureSpec())
                        for s, seed in zip(single_stages, receipt["seeds"])]
            arrays[f"k{k}_B1"] = torch.stack([r.normalized_chunk for r in baseline]).cpu().numpy()
            for b in (2, 4):
                deadline()
                combined = [adapter.run_stage2(s) for s in adapter.run_stage1_batch(prepared[:b])]
                for scope, stages in (("sampler_only", single_stages[:b]), ("end_to_end", combined)):
                    deadline()
                    results = adapter.sample_batch(stages, profile,
                        [RequestNoise(seed, adapter.device) for seed in receipt["seeds"][:b]], [CaptureSpec()] * b)
                    one = torch.stack([r.normalized_chunk for r in baseline[:b]])
                    two = torch.stack([r.normalized_chunk for r in results])
                    assert torch.isfinite(two).all()
                    assert all(r.actual_nfe == k for r in results)
                    rng_exact = all(a.rng_receipt == c.rng_receipt for a, c in zip(baseline, results))
                    assert rng_exact
                    condition_delta = max(float((a.data["conditioning"] - c.data["conditioning"]).abs().max())
                                          for a, c in zip(single_stages, stages))
                    row = dict(K=k, B=b, scope=scope, conditioning_max_abs=condition_delta,
                               rng_exact=rng_exact, **metrics(one, two))
                    receipt["measurements"].append(row)
                    arrays[f"k{k}_B{b}_{scope}"] = two.cpu().numpy()
                    print(json.dumps(row), flush=True)
        torch.cuda.synchronize()
        deadline()
        np.savez(args.output / "normalized_chunks.npz", **arrays)
        receipt.update(status="pass", exit_code=0)
    except BaseException as exc:
        receipt.update(status="error", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - start
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
