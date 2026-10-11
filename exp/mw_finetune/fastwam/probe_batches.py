"""Find the largest global128-compatible microbatch, including Adam allocation.

One model load; OOM attempts are recorded and released before the next candidate.
This only probes throughput. It does not save a training/evaluation checkpoint.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from .data import Dataset
from .loss import frozen_video_loss
from .model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["video_lora", "frozen_video"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--candidates", default="128,64,32,16,8,4")
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--profile-vae", action="store_true", help="time GPU-synchronized VAE work to bound cache benefit")
    a = p.parse_args()
    if os.environ.get("R17_GPU_LOCK_HELD") != "1":
        raise RuntimeError("Acquire the shared GPU flock before a batch sweep")
    torch.set_num_threads(8)
    begin = time.perf_counter()
    dataset = Dataset(a.data, allow_partial=True, mode=a.mode)
    policy = build_model(device="cuda:0", checkpointing=True, fp32_trainable=True, mode=a.mode)
    vae_durations = []
    if a.profile_vae:
        original_encode = policy.model._encode_video_latents

        def encode_profiled(*args, **kwargs):
            torch.cuda.synchronize()
            before = time.perf_counter()
            result = original_encode(*args, **kwargs)
            torch.cuda.synchronize()
            vae_durations.append(time.perf_counter() - before)
            return result

        policy.model._encode_video_latents = encode_profiled
    policy.model.text_encoder.eval()
    texts = list(dict.fromkeys(dataset.manifest["tasks"].values()))
    cached = {}
    for text in texts:
        context, mask = policy.model.encode_prompt(policy.config.prompt_template.format(task=text))
        cached[text] = (context[0].cpu(), mask[0].cpu())
    policy.model.text_encoder.to("cpu")
    policy.train()
    policy.model.vae.eval()
    params = [param for param in policy.parameters() if param.requires_grad]
    attempts = []
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for batch_size in map(int, a.candidates.split(",")):
        if 128 % batch_size:
            raise ValueError("Candidates must divide global batch128")
        accumulation = 128 // batch_size
        optimizer = torch.optim.AdamW(params, lr=1e-7, betas=(.9, .95), weight_decay=.01, foreach=False)
        policy.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        durations, batch, loss = [], None, None
        vae_durations.clear()
        try:
            for micro in range(accumulation * a.updates):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                indices = np.random.default_rng(42 + micro).integers(len(dataset), size=batch_size)
                batch = default_collate([dataset[int(i)] for i in indices])
                batch["context"] = torch.stack([cached[t][0] for t in batch["prompt"]])
                batch["context_mask"] = torch.stack([cached[t][1] for t in batch["prompt"]])
                batch.pop("prompt")
                batch = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = frozen_video_loss(policy, batch) if a.mode == "frozen_video" else policy(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite batch sweep loss")
                (loss / accumulation).backward()
                if (micro + 1) % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - t0)
            # Exclude the first optimizer update (kernel warmup + state allocation).
            steady = durations[accumulation:]
            seconds = float(np.mean(steady))
            row = {"status": "fits", "mode": a.mode, "micro_batch": batch_size, "accumulation": accumulation,
                "global_batch": 128, "microsteps": len(durations), "microstep_s": seconds,
                "microstep_it_s": 1/seconds, "optimizer_step_s": seconds*accumulation,
                "samples_s": batch_size/seconds, "projected_ten_epoch_hours": seconds*accumulation*16001/3600,
                "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
                "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9}
            if a.profile_vae:
                mean_vae = float(np.mean(vae_durations[accumulation:]))
                row["vae_mean_s"] = mean_vae
                row["ideal_zero_cost_vae_cache_ten_epoch_hours"] = max(0, seconds - mean_vae)*accumulation*16001/3600
            attempts.append(row)
            print(json.dumps(row), flush=True)
            break
        except torch.cuda.OutOfMemoryError as exc:
            row = {"status": "oom", "mode": a.mode, "micro_batch": batch_size,
                "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
                "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9, "error": str(exc)}
            attempts.append(row)
            print(json.dumps(row), flush=True)
        finally:
            batch = loss = None
            # The raw VAE can retain causal feature caches if encode raises OOM.
            policy.model.vae.model.clear_cache()
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
            output.write_text(json.dumps({"mode": a.mode, "attempts": attempts,
                "wall_s": time.perf_counter() - begin}, indent=2))
    else:
        raise RuntimeError("No requested microbatch fits")


if __name__ == "__main__":
    main()
