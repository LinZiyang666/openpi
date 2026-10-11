"""Single-GPU frozen-video or video-LoRA tuning, resumable at update boundaries.

Probe counts microsteps and uses the real accumulation/optimizer configuration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .common import BASE_REVISION, disk_gate, lr_at
from .data import Dataset
from .model import BASE, WAN, build_model, canonical_key, delta_state, save_delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/data/mw_finetune/fastwam/mt50_224")
    p.add_argument("--base", default=BASE)
    p.add_argument("--wan-root", default=WAN)
    p.add_argument("--output", default="/data/mw_finetune/fastwam/run")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", choices=["video_lora", "frozen_video"], default="frozen_video")
    p.add_argument("--steps", type=int, default=16001, help="ceil(204806*10/128); ten epoch-equivalent updates")
    p.add_argument("--micro-batch", type=int, default=None, help="default128 frozen or32 LoRA; override with measured batch")
    p.add_argument("--global-batch", type=int, default=128)
    p.add_argument("--accumulation", type=int, help="default global-batch/micro-batch")
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--milestone-every", type=int, default=16001, help="immutable serving snapshots; final always saved")
    p.add_argument("--probe", type=int, default=0, help="timed training microsteps, e.g. --probe 200")
    p.add_argument("--probe-warmup", type=int, default=10)
    p.add_argument("--max-seconds", type=float, default=840, help="probe-only time cap including model load")
    p.add_argument("--resume", nargs="?", const="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--bf16-optimizer", action="store_true", help="diagnostic only; default is FP32 action parameters/Adam moments")
    p.add_argument("--save-probe", action="store_true")
    a = p.parse_args()
    if a.micro_batch is None:
        a.micro_batch = 128 if a.mode == "frozen_video" else 32
    if a.accumulation is None:
        if a.global_batch % a.micro_batch:
            p.error("global batch must be divisible by micro batch")
        a.accumulation = a.global_batch // a.micro_batch
    if a.micro_batch * a.accumulation != a.global_batch:
        p.error("micro-batch * accumulation must equal global-batch")
    if min(a.micro_batch, a.accumulation, a.steps, a.checkpoint_every, a.milestone_every) < 1:
        p.error("batch, accumulation, steps, and checkpoint intervals must be positive")
    if a.probe and a.probe < a.accumulation:
        p.error("probe must include at least one optimizer update")
    if os.environ.get("R17_GPU_LOCK_HELD") != "1":
        raise RuntimeError("Use launch.sh so this process holds /tmp/r17_gpu.lock")
    import torch
    from torch.utils.data import default_collate

    start = time.perf_counter()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    resume_path = out / "latest.pt" if a.resume == "auto" else Path(a.resume) if a.resume else None
    # Initial reservation covers the first snapshot, or an atomic replacement
    # when resuming; each actual checkpoint rechecks its measured tensor sizes.
    if a.probe and not a.save_probe:
        reservation = 1
    elif resume_path and resume_path.exists():
        reservation = 13.5 if a.mode == "frozen_video" else 17.5
    else:
        reservation = 18 if a.mode == "frozen_video" else 23
    disk_gate(out, required_gb=reservation)
    torch.set_num_threads(8)
    torch.manual_seed(a.seed)
    dataset = Dataset(a.data, allow_partial=bool(a.probe), mode=a.mode)
    policy = build_model(a.base, a.device, a.wan_root, not a.no_checkpointing, not a.bf16_optimizer, a.mode)
    audit = policy._mt50_load_audit
    print(json.dumps({"load_audit": audit}), flush=True)
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01, foreach=False)
    # Fifty fixed training instructions; cached only for training, never at serving.
    tasks = sorted(set(dataset.manifest["tasks"].values()))
    policy.model.text_encoder.eval()
    context_cache = {}
    for offset in range(0, len(tasks), 4):
        prompts = tasks[offset:offset + 4]
        context, mask = policy.model.encode_prompt([policy.config.prompt_template.format(task=t) for t in prompts])
        for index, text in enumerate(prompts):
            context_cache[text] = (context[index].detach().cpu(), mask[index].detach().cpu())
    policy.model.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    metadata = {"format": "fastwam-mt50-delta-v1", "base": a.base, "base_revision": BASE_REVISION,
        "wan_root": a.wan_root, "stats": dataset.manifest["stats"],
        "data_revision": dataset.manifest["revision"], "data_index_sha256": dataset.manifest["index_sha256"],
        "data_partial": dataset.manifest["partial"], "mode": a.mode,
        "recipe": "official MT50 ten epochs, batch128, .05 warmup/.01 cosine floor; BF16 activations; FP32 action/proprio/LoRA parameters",
        "normalization": "min_max; no clipping; MT50 stats",
        "arguments": vars(a), "load_audit": audit, "global_batch": a.micro_batch * a.accumulation}
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    update = micro = 0
    if resume_path and resume_path.exists():
        state = torch.load(resume_path, weights_only=False, map_location="cpu")
        for key in ("base_revision", "data_index_sha256", "global_batch"):
            if state["metadata"][key] != metadata[key]:
                raise ValueError(f"Resume mismatch: {key}")
        for key in ("steps", "seed", "no_checkpointing", "bf16_optimizer", "mode", "micro_batch", "accumulation"):
            if state["metadata"]["arguments"][key] != vars(a)[key]:
                raise ValueError(f"Resume argument mismatch: {key}")
        expected = {canonical_key(k): p for k, p in policy.named_parameters() if p.requires_grad}
        if set(expected) != set(state["model"]):
            raise ValueError("Resume parameter set mismatch")
        with torch.no_grad():
            for k, param in expected.items():
                param.copy_(state["model"][k].to(device=param.device, dtype=param.dtype))
        optimizer.load_state_dict(state["optimizer"])
        update, micro = state["update"], state["micro"]
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
        del state
    elif resume_path and a.resume != "auto":
        raise FileNotFoundError(resume_path)

    def checkpoint():
        parameter_gb = sum(param.numel() * param.element_size() for param in params) / 1e9
        # Adam's two moments plus model delta; peak is an atomic resume replacement.
        headroom = 3 * parameter_gb + 1
        if not (out / "latest.pt").exists():
            headroom += parameter_gb
        disk_gate(out, required_gb=headroom)
        snapshot = {"model": delta_state(policy), "optimizer": optimizer.state_dict(),
            "metadata": metadata, "update": update, "micro": micro,
            "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()}
        torch.save(snapshot, out / "latest.pt.tmp")
        (out / "latest.pt.tmp").replace(out / "latest.pt")
        save_delta(policy, out / "policy_latest", {**metadata, "update": update})
        if update and (update % a.milestone_every == 0 or update == a.steps):
            save_delta(policy, out / f"policy_{update:06d}", {**metadata, "update": update})

    policy.train()
    # Unregistered frozen modules must remain in eval (T5 has dropout).
    policy.model.vae.eval()
    policy.model.text_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    durations, losses, grad_checked = [], [], False
    orders = {}
    completed = 0
    timed_start = time.perf_counter()
    stop_reason = "target"
    with (out / "train.jsonl").open("a", buffering=1) as log:
        while update < a.steps:
            if a.probe and (completed >= a.probe or time.perf_counter() - start >= a.max_seconds):
                stop_reason = "target" if completed >= a.probe else "time_cap"
                break
            t0 = time.perf_counter()
            # Independent index RNG; restart is exact at microstep boundaries.
            sample_offset = micro * a.micro_batch
            epoch = sample_offset // len(dataset)
            if epoch not in orders:
                orders = {epoch: np.random.default_rng(np.random.SeedSequence([a.seed, epoch])).permutation(len(dataset))}
            frame = sample_offset % len(dataset)
            indices = orders[epoch][frame:frame + a.micro_batch]
            if len(indices) < a.micro_batch:
                next_order = np.random.default_rng(np.random.SeedSequence([a.seed, epoch + 1])).permutation(len(dataset))
                orders[epoch + 1] = next_order
                indices = np.concatenate([indices, next_order[:a.micro_batch - len(indices)]])
            batch = default_collate([dataset[int(i)] for i in indices])
            batch["context"] = torch.stack([context_cache[t][0] for t in batch["prompt"]])
            batch["context_mask"] = torch.stack([context_cache[t][1] for t in batch["prompt"]])
            batch.pop("prompt")
            batch = {k: v.to(a.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            for group in optimizer.param_groups:
                group["lr"] = lr_at(update, total=a.steps, warmup=int(a.steps * 0.05), minimum=1e-6)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if a.mode == "frozen_video":
                    from .loss import frozen_video_loss
                    loss, metrics = frozen_video_loss(policy, batch)
                else:
                    loss, metrics = policy(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at micro {micro}")
            (loss / a.accumulation).backward()
            if not grad_checked:
                gradients = {"video_gradient_tensors": 0, "video_lora_gradient_tensors": 0, "action_gradient_tensors": 0, "proprio_gradient_norm": 0.0}
                for name, param in policy.named_parameters():
                    if param.grad is None:
                        continue
                    if "video" in name:
                        if "lora_A" in name or "lora_B" in name:
                            gradients["video_lora_gradient_tensors"] += 1
                        else:
                            gradients["video_gradient_tensors"] += 1
                    elif "proprio_encoder" in name:
                        gradients["proprio_gradient_norm"] += float(param.grad.float().norm())
                    else:
                        gradients["action_gradient_tensors"] += 1
                if gradients["video_gradient_tensors"] or not gradients["action_gradient_tensors"] or not gradients["proprio_gradient_norm"]:
                    raise RuntimeError(f"Invalid frozen-expert gradient audit: {gradients}")
                if a.mode == "video_lora" and not gradients["video_lora_gradient_tensors"]:
                    raise RuntimeError("Video LoRA received no gradients")
                print(json.dumps({"gradient_audit": gradients}), flush=True)
                metadata["gradient_audit"] = gradients
                grad_checked = True
            micro += 1
            completed += 1
            if micro % a.accumulation == 0:
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if not a.probe and update % a.checkpoint_every == 0:
                    checkpoint()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            if completed > a.probe_warmup:
                durations.append(seconds)
            losses.append(float(loss.detach()))
            if completed == 1 or completed % 10 == 0:
                row = {"micro": micro, "update": update, "loss": losses[-1], **metrics,
                    "microstep_s": seconds, "lr": optimizer.param_groups[0]["lr"],
                    "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
                    "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9}
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        summary = {"requested_microsteps": a.probe or micro, "completed_microsteps": completed,
            "optimizer_updates": update, "stop_reason": stop_reason,
            "microstep_it_s": len(durations)/sum(durations) if durations else None,
            "microstep_mean_s": float(np.mean(durations)) if durations else None,
            "microstep_p50_s": float(np.median(durations)) if durations else None,
            "estimated_optimizer_it_s": len(durations)/sum(durations)/a.accumulation if durations else None,
            "samples_s": len(durations)/sum(durations)*a.micro_batch if durations else None,
            "optimizer_step_s": float(np.mean(durations))*a.accumulation if durations else None,
            "projected_ten_epoch_hours": float(np.mean(durations))*a.accumulation*16001/3600 if durations else None,
            "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9,
            "wall_s": time.perf_counter()-start, "train_wall_s": time.perf_counter()-timed_start,
            "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
            "micro_batch": a.micro_batch, "accumulation": a.accumulation, "metadata": metadata}
        if not a.probe or a.save_probe:
            checkpoint_start = time.perf_counter()
            if micro % a.accumulation:
                # Resume saves only complete updates; probe can leave a partial accumulation.
                optimizer.zero_grad(set_to_none=True)
                micro -= micro % a.accumulation
            checkpoint()
            summary["checkpoint_s"] = time.perf_counter() - checkpoint_start
        summary["total_wall_s"] = time.perf_counter() - start
        (out / "probe.json" if a.probe else out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"summary": {k: v for k, v in summary.items() if k != "metadata"}}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import sys
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            argv = sys.argv
            output = Path(argv[argv.index("--output") + 1]) if "--output" in argv else Path("/data/mw_finetune/fastwam/run")
            output.mkdir(parents=True, exist_ok=True)
            failure = {"status": "oom", "error": str(exc), "argv": argv[1:],
                "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
                "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9}
            (output / "oom.json").write_text(json.dumps(failure, indent=2))
            print(json.dumps(failure), flush=True)
        raise
