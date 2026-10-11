"""Resumable stateless DP training; --probe counts optimizer updates."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch.utils.data import DataLoader

from .checkpoint import read_metadata
from .data import Batches, Dataset
from .model import DiffusionPolicy
from .policy import memory_cap
from .preprocess import LanguageEncoder, quantile_normalize


def learning_rate(step, warmup=500, peak=1e-4, decay=99500, end=1e-5):
    if step < warmup:
        return peak * (step + 1) / (warmup + 1)
    # Published Brandon openpi train_pytorch.py interprets decay_steps as the
    # total schedule endpoint, including warmup (Optax convention).
    progress = min(max((step - warmup) / max(1, decay - warmup), 0), 1)
    return end + (peak - end) * .5 * (1 + math.cos(math.pi * progress))


def fit_internal(model, manifest, dataset, embeddings):
    """DP limit affine on external-quantile vectors and per-width RGB pixels."""
    core = model._policy
    arrays = {"state": quantile_normalize(dataset.index["state"], manifest["external_stats"]["state"]),
              "action": quantile_normalize(dataset.index["action"], manifest["external_stats"]["actions"]),
              "lang_emb": embeddings.numpy()}
    summaries = {k: {s: getattr(np, s)(v, axis=0) for s in ("min", "max", "mean", "std")} for k, v in arrays.items()}
    summaries["base_0_rgb"] = manifest["image_stats"]
    summaries["left_wrist_0_rgb"] = manifest.get("wrist_image_stats", {
        "min": np.zeros(96), "max": np.ones(96), "mean": np.zeros(96), "std": np.zeros(96)})
    with torch.no_grad():
        for key, summary in summaries.items():
            lo, hi = np.asarray(summary["min"]), np.asarray(summary["max"])
            span = hi - lo
            # Stanford LinearNormalizer.fit(mode='limits'), constant dimensions centered at0.
            constant = span < 1e-4
            safe_span = np.where(constant, 2, span)
            scale = 2 / safe_span
            offset = -1 - scale * lo
            offset = np.where(constant, -np.asarray(summary["mean"]) * scale, offset)
            p = core.normalizer.params_dict[key]
            for name, values in [("scale", scale), ("offset", offset)]:
                p[name].copy_(torch.as_tensor(values, dtype=torch.float32))
            for name in ("min", "max", "mean", "std"):
                p["input_stats"][name].copy_(torch.as_tensor(summary[name], dtype=torch.float32))


def save_checkpoint(output, step, model, optimizer, metadata, recipe):
    final = output / f"checkpoint-{step:06d}"
    temp = output / f".checkpoint-{step:06d}.partial"
    if final.exists():
        raise FileExistsError(f"Refuse replacement of immutable {final}")
    if temp.exists():
        temp.rename(temp.with_name(temp.name + f".interrupted-{time.time_ns()}"))
    temp.mkdir()
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temp / "model.safetensors"))
    # CPU verification must not initialize a CUDA context on the occupied4090.
    on_cuda = next(model.parameters()).device.type == "cuda"
    state = {"optimizer": optimizer.state_dict(), "step": step, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if on_cuda else [], "recipe": recipe}
    torch.save(state, temp / "training.pt")
    metadata = dict(metadata, global_step=step)
    (temp / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (temp / "norm_stats.json").write_text(json.dumps({"norm_stats": recipe["external_stats"]}, indent=2))
    (temp / "complete.json").write_text(json.dumps({"step": step, "recipe_sha256": hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode()).hexdigest()}))
    for path in temp.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    def sync_directory(path):
        fd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    sync_directory(temp)
    temp.rename(final)
    sync_directory(output)
    link = output / ".latest.partial"
    if link.is_symlink():
        link.rename(output / f".latest.interrupted-{time.time_ns()}")
    link.symlink_to(final.name, target_is_directory=True)
    link.replace(output / "latest")
    sync_directory(output)
    return final


def newest_complete(output):
    """Recover a save completed just before a crash updating latest."""
    candidates = []
    for path in output.glob("checkpoint-*"):
        marker = path / "complete.json"
        if path.is_dir() and marker.exists():
            step = json.loads(marker.read_text())["step"]
            if path.name != f"checkpoint-{step:06d}":
                raise ValueError(f"Checkpoint name/step mismatch: {path}")
            if not all((path / name).is_file() for name in ("model.safetensors", "training.pt", "metadata.json", "norm_stats.json")):
                raise ValueError(f"Completed marker with absent files: {path}")
            candidates.append((step, path))
    return max(candidates)[1] if candidates else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("/data/mw_finetune/dp/checkpoints/metaworld"))
    ap.add_argument("--init", choices=("scratch", "pretrained"), default="scratch",
                    help="scratch reads architecture metadata ONLY; pretrained loads DP tensors")
    ap.add_argument("--domain", choices=("metaworld", "libero"), default="metaworld")
    ap.add_argument("--dataset", type=Path, default=Path("/data/mw_finetune/dp/mt50_96"))
    ap.add_argument("--text-encoder", default="/data/mw_finetune/dp/clip_text")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--accumulation", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--save-probe", action="store_true", help="Export a resumable checkpoint at probe end")
    ap.add_argument("--allow-partial-probe", action="store_true", help="Mechanics only; requires --probe; never a training run")
    ap.add_argument("--resume", choices=("auto", "required", "never"), default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--memory-limit-gib", type=float, default=8.0)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    a = ap.parse_args()
    if min(a.batch_size, a.accumulation, a.steps, a.save_every) < 1 or a.probe < 0:
        ap.error("batch/accumulation/steps/save-every must be positive; probe nonnegative")
    if a.allow_partial_probe and not a.probe:
        ap.error("Partial data is allowed only for bounded probes")
    a.output.mkdir(parents=True, exist_ok=True)
    # Prevent two launches from writing the same run, without taking a training GPU lock.
    import fcntl
    run_lock = (a.output / ".train.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    device = torch.device(a.device); memory_cap(device, a.memory_limit_gib)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    manifest = json.loads((a.dataset / "manifest.json").read_text())
    task_ids = sorted(map(int, manifest["tasks"]))
    if task_ids != list(range(len(task_ids))):
        raise ValueError("Expected contiguous source task indices")
    language_path = a.dataset / "language.safetensors"
    if language_path.exists():
        embeddings = load_file(str(language_path))["embeddings"]
    else:
        encoder = LanguageEncoder(a.text_encoder)
        embeddings = encoder([manifest["tasks"][str(i)] for i in task_ids]).clone()
        del encoder
    if embeddings.shape != (len(task_ids), 768):
        raise ValueError("Expected projected CLIP-L/14 instruction embeddings")
    dataset = Dataset(a.dataset, embeddings, allow_partial=a.allow_partial_probe)
    if (manifest["partial"] and not a.allow_partial_probe) or len(dataset) != manifest["frames"]:
        raise ValueError("Refuse incomplete training data")
    if a.domain == "metaworld" and (len(dataset) != 204806 or manifest["episodes"] != 2500 or len(task_ids) != 50):
        raise ValueError("Expected complete MT50 data")
    if a.domain == "libero" and (manifest.get("source_repo") != "physical-intelligence/libero" or len(task_ids) != 40):
        raise ValueError("Expected all four LIBERO suites /40 tasks")
    metadata = read_metadata(a.checkpoint)
    cfg = metadata["config"]["model"]
    if cfg["action_dim"] != (7 if a.domain == "libero" else 4):
        raise ValueError("Architecture metadata action dimension differs from domain")
    model = DiffusionPolicy(cfg)
    if a.init == "pretrained":
        model.load_state_dict(load_file(str(a.checkpoint / "model.safetensors")), strict=True)
    fit_internal(model, manifest, dataset, embeddings)
    model.to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=1e-4, betas=(.95, .999), eps=1e-8, weight_decay=1e-6)
    recipe = {"schema": "dp-train-v2", "domain": a.domain, "initialization": a.init,
        "seed": a.seed, "steps": a.steps, "batch_size": a.batch_size,
        "accumulation": a.accumulation, "precision": a.precision,
        "data_sha256": hashlib.sha256((a.dataset / "manifest.json").read_bytes()).hexdigest(),
        "index_sha256": manifest["index_sha256"], "external_stats": manifest["external_stats"],
        "init_sha256": hashlib.sha256((a.checkpoint / "model.safetensors").read_bytes()).hexdigest() if a.init == "pretrained" else None,
        "architecture_config": cfg,
        "architecture_sha256": hashlib.sha256((Path(__file__).with_name("model.py")).read_bytes()).hexdigest(),
        "lr": {"warmup": 500, "peak": 1e-4, "decay": 99500, "end": 1e-5, "decay_includes_warmup": True},
        "optimizer": {"betas": [.95, .999], "weight_decay": 1e-6, "eps": 1e-8, "clip": 10},
        "ema": None, "language_sha256": hashlib.sha256(embeddings.numpy().tobytes()).hexdigest()}
    metadata["config"].update(name="dp_" + a.domain, exp_name=a.output.name, batch_size=a.batch_size * a.accumulation,
                               num_train_steps=a.steps)
    metadata["config"]["data"]["repo_id"] = "physical-intelligence/libero" if a.domain == "libero" else "lerobot/metaworld_mt50"
    metadata["wrapper"] = {"feature_order": list(model._policy.obs_encoder.feature_order),
                            "language_kind": "projected", "second_camera": "zero",
                            "initialization": a.init}
    start = 0
    latest = newest_complete(a.output)
    if latest is not None and a.resume != "never":
        if not (latest / "complete.json").exists():
            raise ValueError("Latest checkpoint is incomplete")
        # Only our locally generated optimizer snapshot is unpickled; public files never are.
        state = torch.load(latest / "training.pt", map_location=device, weights_only=True)
        if state["recipe"] != recipe:
            raise ValueError("Recipe/data/init/seed mismatch on resume")
        model.load_state_dict(load_file(str(latest / "model.safetensors")), strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]; torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    elif a.resume == "required":
        raise FileNotFoundError("--resume required but no complete latest checkpoint")
    elif latest is not None:
        raise FileExistsError("--resume never refuses an existing run")
    stop = min(a.steps, start + a.probe) if a.probe else a.steps
    batches = Batches(len(dataset), a.seed, start, stop, a.batch_size, a.accumulation)
    # DataLoader iterator creation draws a base seed even with workers=0. Keep that
    # draw off the saved model RNG, or a restart shifts dropout/crop/noise streams.
    loader_rng = torch.Generator().manual_seed(a.seed)
    loader = iter(DataLoader(dataset, batch_sampler=batches, num_workers=a.workers,
                            pin_memory=device.type == "cuda", generator=loader_rng))
    elapsed, losses = [], []
    wall = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log = (a.output / "train.jsonl").open("a", buffering=1)
    completed = start
    for step in range(start, stop):
        begin = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0
        for _ in range(a.accumulation):
            batch = next(loader)
            obs = {k: v.to(device, non_blocking=True) for k, v in batch["obs"].items()}
            action = batch["action"].to(device, non_blocking=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and a.precision == "bf16" else contextlib.nullcontext()
            with amp:
                loss = model.loss(obs, action)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            (loss / a.accumulation).backward(); loss_sum += loss.detach().float().item() / a.accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        if not torch.isfinite(grad_norm):
            raise ValueError("Nonfinite gradient norm")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.monotonic() - begin
        elapsed.append(duration); losses.append(loss_sum); completed = step + 1
        record = {"step": completed, "loss": loss_sum, "lr": lr, "grad_norm": float(grad_norm),
                  "update_s": duration, "examples_per_s": a.batch_size * a.accumulation / duration}
        log.write(json.dumps(record) + "\n")
        if completed % 10 == 0 or completed == stop:
            print(json.dumps(record), flush=True)
        if (not a.probe or a.save_probe) and (completed % a.save_every == 0 or completed == stop):
            save_checkpoint(a.output, completed, model, optimizer, metadata, recipe)
    skip = min(5, max(0, len(elapsed) // 4))
    steady = elapsed[skip:]
    average = float(np.mean(steady)) if steady else None
    report = {"start_step": start, "completed_updates": completed - start, "end_step": completed,
        "domain": a.domain, "initialization": a.init,
        "partial_data": manifest["partial"],
        "timed_updates": len(steady), "warmup_excluded": skip, "optimizer_it_per_s": 1 / average if average else None,
        "seconds_per_update": average, "projection_hours": (a.steps - start) * average / 3600 if average else None,
        "wall_s": time.monotonic() - wall, "batch": a.batch_size, "accumulation": a.accumulation,
        "global_batch": a.batch_size * a.accumulation, "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None, "device": str(device), "memory_limit_gib": a.memory_limit_gib,
        "allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0,
        "reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0}
    (a.output / f"probe-{start:06d}-{completed:06d}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
