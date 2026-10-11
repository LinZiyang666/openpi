"""Measured/resumable single-GPU entry point around NVIDIA's N1.5 finetune main.

The optimizer, loss, collator and model construction come from the pinned release.
Local additions: named embodiment, resource checks, measurements, and safe resume.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from .conventions import EMBODIMENT
from .embodiment import register_embodiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", choices=["4090", "h100"], required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/data/mw_finetune/datasets/metaworld_mt50_groot"))
    parser.add_argument("--base-model", type=Path, default=Path("/data/mw_finetune/models/groot_base"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe", type=int, metavar="STEPS", default=0)
    parser.add_argument("--smoke", action="store_true", help="two updates, save one checkpoint, inference; rerun to test exact resume")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-steps", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", choices=["auto", "never", "required"], default="auto")
    parser.add_argument("--tune-diffusion", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.smoke and args.probe:
        parser.error("choose smoke or probe")
    if args.probe < 0 or min(args.steps, args.accumulation, args.save_steps) < 1 or args.workers < 1:
        parser.error("steps, accumulation, save-steps and workers must be positive")
    if os.environ.get("R17_GPU_LOCK_HELD") != "1":
        raise RuntimeError("use launch_4090.sh/launch_h100.sh: GPU lock is required")
    free_mib = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    if free_mib < 12000:
        raise RuntimeError(f"only {free_mib} MiB free before starting")
    register_embodiment()
    import torch
    from transformers import TrainerCallback
    from gr00t import __file__ as gr00t_file
    from .convert_data import write_json

    vendor = Path(gr00t_file).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("r17_nvidia_gr00t_finetune", vendor / "scripts/gr00t_finetune.py")
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    mode = "smoke" if args.smoke else "probe" if args.probe else "train"
    args.output = args.output or Path(f"/data/mw_finetune/groot/{args.hardware}_{mode}")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.dataset / "conversion/report.json"
    report = json.loads(report_path.read_text())
    if not report.get("complete") or (mode == "train" and report.get("smoke_only")):
        raise ValueError("long training requires a complete full-MT50 conversion")
    verification_path = args.dataset / "conversion/verification.json"
    if not verification_path.exists():
        raise ValueError("run verify_data before GPU training")
    verification = json.loads(verification_path.read_text())
    if not verification.get("passed") or verification["frames"] != report["frames"]:
        raise ValueError("dataset verification failed or belongs to a different conversion")
    for name in ["info.json", "stats.json", "modality.json"]:
        actual_hash = hashlib.sha256((args.dataset / "meta" / name).read_bytes()).hexdigest()
        if verification.get("metadata_sha256", {}).get(name) != actual_hash:
            raise ValueError(f"dataset {name} changed since verification; rerun verify_data")
    args.batch_size = args.batch_size or (64 if args.hardware == "4090" else 128)
    tune_diffusion = args.tune_diffusion if args.tune_diffusion is not None else True
    max_updates = 2 if args.smoke else args.probe or args.steps
    if max_updates > args.steps:
        parser.error("probe cannot exceed --steps")
    run_config = {"hardware": args.hardware, "dataset": str(args.dataset.resolve()),
                  "base_model": str(args.base_model.resolve()), "batch_size": args.batch_size,
                  "gradient_accumulation_steps": args.accumulation, "max_steps": args.steps,
                  "learning_rate": args.lr, "tune_llm": False, "tune_visual": False,
                  "tune_projector": True, "tune_diffusion_model": tune_diffusion,
                  "embodiment": EMBODIMENT, "save_steps": args.save_steps, "mode": mode,
                  "stop_after_updates": max_updates}
    config_path = args.output / "r17_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError("output already belongs to another config; use a new output")
    from .checkpoints import checkpoint_status, preserve_incomplete, validate_resume_statistics
    checkpoints, incomplete = checkpoint_status(args.output)
    resume = bool(checkpoints) and args.resume != "never"
    if (checkpoints or incomplete) and args.resume == "never":
        raise FileExistsError("output has checkpoints; refusing to overwrite")
    if args.resume == "required" and not checkpoints:
        raise FileNotFoundError("--resume required but no checkpoint exists")
    if resume:
        validate_resume_statistics(checkpoints[-1], args.dataset)
    if resume and mode == "train":
        saved_step = json.loads((checkpoints[-1] / "trainer_state.json").read_text())["global_step"]
        if saved_step >= args.steps:
            print(f"Training already complete at {checkpoints[-1]}", flush=True)
            return
    if incomplete:
        print("Preserved incomplete snapshots: " + json.dumps(preserve_incomplete(incomplete)), flush=True)
    if not resume:
        write_json(config_path, run_config)
    write_json(args.output / "environment.json", {"host": socket.gethostname(), "torch": torch.__version__,
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
               "vendor_commit": subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip(),
               "free_mib_before": free_mib, "python": sys.version,
               "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")})

    class Measurement(TrainerCallback):
        def __init__(self):
            self.times = []
            self.start = None
            self.wall_start = time.perf_counter()
            self.parameter_counts = {}

        def on_train_begin(self, targs, state, control, model=None, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.previous_end = time.perf_counter()
            self.initial_step = state.global_step
            self.parameter_counts = {"total": sum(p.numel() for p in model.parameters()),
                                     "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            self.trainable_groups = {}
            for name, p in model.named_parameters():
                if p.requires_grad:
                    group = ".".join(name.split(".")[:2])
                    self.trainable_groups[group] = self.trainable_groups.get(group, 0) + p.numel()
            # Estimate full snapshots plus optimizer state, including all retained
            # checkpoints. No automatic checkpoint deletion under the shared rule.
            snapshot_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            optimizer_bytes = sum(p.numel() * 8 for p in model.parameters() if p.requires_grad)
            self.checkpoint_bytes = snapshot_bytes + optimizer_bytes
            remaining = max(0, args.steps - state.global_step)
            count = ((remaining + args.save_steps - 1) // args.save_steps) if mode == "train" else int(args.smoke)
            required = (snapshot_bytes + optimizer_bytes) * count + 2_000_000_000
            free = shutil.disk_usage(args.output).free
            if free - required < 30_000_000_000:
                raise RuntimeError(f"checkpoint budget needs {required / 1e9:.1f} GB plus 30 GB free floor")
            if args.hardware == "h100":
                # du counts existing allocated bytes; separate agents may grow
                # afterwards, so repeat the floor check before each save.
                used = int(subprocess.check_output(["du", "-s", "-B1", "/data/mw_finetune"], text=True).split()[0])
                if used + required > 120_000_000_000:
                    raise RuntimeError(f"shared H100 quota: used {used / 1e9:.1f} + projected {required / 1e9:.1f} > 120 GB")
            print(json.dumps({"parameter_counts": self.parameter_counts, "trainable_groups": self.trainable_groups,
                              "projected_checkpoint_gb": required / 1e9}), flush=True)

        def on_step_begin(self, targs, state, control, **kwargs):
            torch.cuda.synchronize()
            self.start = time.perf_counter()

        def on_step_end(self, targs, state, control, **kwargs):
            torch.cuda.synchronize()
            now = time.perf_counter()
            self.times.append(now - self.previous_end)
            self.previous_end = now
            completed = state.global_step - self.initial_step
            if completed >= max_updates or state.global_step >= args.steps:
                control.should_training_stop = True
                control.should_save = mode in {"train", "smoke"}
            elif mode == "probe":
                control.should_save = False
            if control.should_save and shutil.disk_usage(args.output).free < 50_000_000_000:
                raise RuntimeError("checkpoint save would threaten 30 GB disk floor")
            if control.should_save and args.hardware == "h100":
                used = int(subprocess.check_output(["du", "-s", "-B1", "/data/mw_finetune"], text=True).split()[0])
                if used + self.checkpoint_bytes + 500_000_000 > 120_000_000_000:
                    raise RuntimeError("shared H100 tree grew; checkpoint save would exceed the 120 GB quota")

        def on_train_end(self, targs, state, control, **kwargs):
            warmup = min(10, max(0, len(self.times) - 1))
            steady = self.times[warmup:]
            summary = {**run_config, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
                       "completed_updates": len(self.times), "final_global_step": state.global_step,
                       "warmup_updates_excluded": warmup,
                       "optimizer_it_per_s": len(steady) / sum(steady) if steady else None,
                       "microbatch_it_per_s": len(steady) * args.accumulation / sum(steady) if steady else None,
                       "samples_per_s": len(steady) * args.accumulation * args.batch_size / sum(steady) if steady else None,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                       "elapsed_seconds": time.perf_counter() - self.wall_start,
                       "parameters": self.parameter_counts, "trainable_groups": self.trainable_groups,
                       "step_seconds": self.times, "success": len(self.times) > 0}
            write_json(args.output / "measurements.json", summary)
            print("R17_MEASUREMENTS=" + json.dumps({k: v for k, v in summary.items() if k != "step_seconds"}), flush=True)

    measurement = Measurement()
    original_args = upstream.TrainingArguments

    def training_arguments(**kwargs):
        kwargs["save_total_limit"] = None  # No implicit deletion of old checkpoints.
        return original_args(**kwargs)

    upstream.TrainingArguments = training_arguments
    runner = upstream.TrainRunner

    class MeasuredRunner(runner):
        def create_trainer(self, **kwargs):
            trainer = super().create_trainer(**kwargs)
            trainer.add_callback(measurement)
            return trainer

        def train(self):
            self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
            if mode != "probe":
                self.trainer.save_state()
            # Final checkpoint already includes weights and experiment_cfg.
            # Avoid a second full model export in the parent output directory.

    upstream.TrainRunner = MeasuredRunner
    config = upstream.ArgsConfig(dataset_path=[str(args.dataset)], output_dir=str(args.output),
              data_config="exp.mw_finetune.groot.data_config:MetaWorldDataConfig", batch_size=args.batch_size,
              max_steps=args.steps, num_gpus=1, save_steps=args.save_steps, base_model_path=str(args.base_model),
              tune_llm=False, tune_visual=False, tune_projector=True, tune_diffusion_model=tune_diffusion,
              resume=resume, learning_rate=args.lr, dataloader_num_workers=args.workers,
              gradient_accumulation_steps=args.accumulation, dataloader_prefetch_factor=2,
              report_to="tensorboard", embodiment_tag=EMBODIMENT, video_backend="decord")
    print(json.dumps(dataclasses.asdict(config), indent=2), flush=True)
    upstream.main(config)
    if args.smoke:
        checkpoint = max(args.output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        # Free trainer references/optimizer before loading a second inference model.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        from .policy import load_policy
        from .data_config import MetaWorldDataConfig
        from gr00t.data.dataset import LeRobotSingleDataset
        from .conventions import CHANNELS
        cfg = MetaWorldDataConfig()
        ds = LeRobotSingleDataset(str(args.dataset), cfg.modality_config(), transforms=cfg.transform(),
                                 embodiment_tag=EMBODIMENT, video_backend="decord")
        raw = ds.get_step_data(int(ds.trajectory_ids[0]), 0)
        obs = {"image": raw["video.image"][0],
               "state": __import__("numpy").concatenate([raw[f"state.{name}"][0] for name in CHANNELS]),
               "prompt": raw["annotation.human.action.task_description"][0], "task_name": "smoke"}
        policy = load_policy(checkpoint, "cuda:0")
        policy.reset()
        actions = policy.infer(obs)["actions"]
        write_json(args.output / "inference_smoke.json", {"shape": list(actions.shape), "actions": actions.tolist(),
                   "checkpoint": str(checkpoint), "passed": True})
        # Actual optimizer/RNG reload occurs when the coordinator reruns this
        # command with --resume required. Checkpoint presence is checked here.
        print("R17_SMOKE_INFERENCE_OK", actions.shape, flush=True)


if __name__ == "__main__":
    main()
