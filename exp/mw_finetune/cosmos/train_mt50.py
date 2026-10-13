"""Bounded single-GPU fine-tuning with the native Cosmos Policy loss/optimizer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
from unittest.mock import patch

import numpy as np
import torch

from .mt50_data import FrameSampler, MT50Dataset, PREPROCESSING


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="/data/mw_finetune/fastwam/mt50_224")
    p.add_argument("--assets", default="/data/mw_finetune/cosmos/mt50_assets")
    p.add_argument("--init", default="/data/mw_finetune/models/cosmos_libero/Cosmos-Policy-LIBERO-Predict2-2B.pt")
    p.add_argument("--output", required=True)
    p.add_argument("--vendor", default=os.environ.get("COSMOS_REPO", "/home/weiland/projects/openpi_ext/lines/cosmos-policy"))
    p.add_argument("--hf-home", default="/data/cosmos/hf_home")
    p.add_argument("--tokenizer", default="")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--export-every", type=int, default=5000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-hours", type=float, default=24)
    p.add_argument("--probe", type=int, default=0, help="N optimizer updates, including accumulation and Adam allocation")
    p.add_argument("--resume", default="auto", help="auto, none, or a training_state.pt file")
    p.add_argument("--print-config", action="store_true", help="Print configuration without loading a model or using CUDA")
    a = p.parse_args(argv)
    if min(a.batch, a.accum, a.steps, a.save_every, a.export_every) < 1 or a.max_hours <= 0 or a.probe < 0:
        p.error("batch, accum, steps, checkpoint intervals and max-hours must be positive")
    return a


def experiment_options(a, vae):
    return ["model.config.tokenizer.vae_pth=" + str(vae),
        "model.config.state_t=7", "model.config.min_num_conditional_frames=3",
        "model.config.max_num_conditional_frames=3", "model.config.tokenizer.chunk_duration=25",
        "model.config.net.sac_config.mode=block_wise", "model.config.net.sac_config.every_n_blocks=1",
        "model.config.mask_loss_for_action_future_state_prediction=True",
        "dataloader_train.dataset.use_wrist_images=False", "optimizer.lr=" + str(a.lr)]


def load_native(a):
    from exp.r17_integration.models.cosmos.loader import LoadSpec, activate_vendor, tokenizer_path
    from exp.r17_integration.models.cosmos.metaworld import offline_checkpoint_path
    spec = LoadSpec(a.init, vendor_root=a.vendor, hf_home=a.hf_home, tokenizer_path=a.tokenizer)
    activate_vendor(spec)
    # activate_vendor enables inference seeding. WanVAE would otherwise reset
    # all RNGs to 7 on every encode, corrupting training noise sampling.
    os.environ["DETERMINISTIC"] = "False"
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
    vae = str(tokenizer_path(spec).absolute())
    resolve = lambda uri: offline_checkpoint_path(uri, policy_path=a.init, vae_path=vae)
    # Training experiment retains sigma .01..200 and the official hybrid noise.
    with patch("cosmos_policy._src.imaginaire.utils.checkpoint_db.get_checkpoint_path", side_effect=resolve):
        model, config = load_model_from_checkpoint(
            experiment_name="cosmos_predict2_2b_480p_libero", s3_checkpoint_dir=a.init,
            config_file="cosmos_policy/config/config.py", load_ema_to_reg=False, to_device="cuda:0",
            experiment_opts=experiment_options(a, vae), seed=a.seed)
    model.train()
    # The Wan VAE interface is a plain wrapper; its torch module sits two levels down.
    vae_module = getattr(getattr(model.tokenizer, 'model', None), 'model', None)
    if isinstance(vae_module, torch.nn.Module):
        vae_module.eval().requires_grad_(False)
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    return model, optimizer, scheduler


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_cpu(v) for v in value)
    return value


def optimizer_restore(optimizer, saved, masters, device):
    optimizer.load_state_dict(saved)
    # Torch's generic restore casts moments to BF16 parameter dtype. Restore
    # native FP32 moments from their original bytes, avoiding that round trip.
    for group, old_group in zip(optimizer.param_groups, saved["param_groups"]):
        for param, old_id in zip(group["params"], old_group["params"]):
            for name in ("exp_avg", "exp_avg_sq"):
                if name in saved["state"].get(old_id, {}):
                    optimizer.state[param][name] = saved["state"][old_id][name].to(device=device)
    # Native FusedAdam keeps master parameters outside optimizer.state_dict().
    if masters is not None:
        optimizer.param_groups_master = [{"params": [p.to(device=device) if p is not None else None for p in group]}
                                         for group in masters]


def save_checkpoint(a, model, optimizer, scheduler, step, *, export, cuda=True):
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    master_groups = getattr(optimizer, "param_groups_master", None)
    state = dict(step=step, model=to_cpu(model.state_dict()), optimizer=to_cpu(optimizer.state_dict()),
        masters=to_cpu([g["params"] for g in master_groups]) if master_groups is not None else None,
        scheduler=to_cpu(scheduler.state_dict()), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if cuda else [], numpy_rng=np.random.get_state(),
        python_rng=random.getstate(), batch=a.batch, accum=a.accum, seed=a.seed,
        elapsed_at_save=time.time())
    temp = root / "training_state.pt.tmp"
    torch.save(state, temp)
    temp.replace(root / "training_state.pt")
    if export:
        dest = root / f"checkpoint-{step:06d}"
        dest.mkdir(exist_ok=True)
        torch.save(state["model"], dest / "consolidated.pt.tmp")
        (dest / "consolidated.pt.tmp").replace(dest / "consolidated.pt")
        for name in ("dataset_statistics.json", "t5_embeddings.pkl"):
            shutil.copyfile(Path(a.assets) / name, dest / name)
        cfg = dict(cosmos_mw=dict(**PREPROCESSING, protocol_source="R17 MT50 fine-tune; RECIPE.md",
                    trained_chunk=16, execution_steps=5, state_t=7, conditioning_frames=3),
                   training=dict(vars(a), completed_steps=step))
        (dest / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        shutil.copyfile(Path(__file__).with_name("RECIPE.md"), dest / "RECIPE.md")
        print(json.dumps(dict(checkpoint=str(dest), step=step)), flush=True)


def resume_checkpoint(a, model, optimizer, scheduler, *, device):
    path = Path(a.output) / "training_state.pt" if a.resume == "auto" else Path(a.resume)
    if a.resume == "none" or (a.resume == "auto" and not path.exists()):
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    # Same seed/global batch retains the shuffled sample sequence on either GPU.
    if a.batch * a.accum != state["batch"] * state["accum"] or a.seed != state["seed"]:
        raise ValueError("Resume requires the saved effective batch and seed")
    model.load_state_dict(state["model"])
    optimizer_restore(optimizer, state["optimizer"], state["masters"], device)
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if device != "cpu":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return int(state["step"])


def fit(a, model, optimizer, scheduler, dataset, *, device="cuda:0", started=None):
    """Optimizer-boundary resume; stop before starting work beyond the time limit."""
    started = time.monotonic() if started is None else started
    deadline = started + a.max_hours * 3600
    step = resume_checkpoint(a, model, optimizer, scheduler, device=device)
    first_step = step
    target = min(a.steps, step + a.probe) if a.probe else a.steps
    sampler = FrameSampler(len(dataset), a.seed, step * a.batch * a.accum,
                           max(0, target - step) * a.batch * a.accum)
    loader = torch.utils.data.DataLoader(dataset, batch_size=a.batch, sampler=sampler,
        num_workers=a.workers, pin_memory=device != "cpu", persistent_workers=a.workers > 0,
        generator=torch.Generator().manual_seed(a.seed))
    data = iter(loader)
    stopped = False
    def stop_requested(*_):
        nonlocal stopped
        stopped = True
    previous = {s: signal.signal(s, stop_requested) for s in (signal.SIGTERM, signal.SIGINT)}
    times, losses, action_losses = [], [], []
    cuda = device != "cpu"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    try:
        while step < target and not stopped and time.monotonic() < deadline:
            before = time.monotonic()
            total_loss, action_loss, policy_batches = 0., 0., 0
            complete = True
            for _ in range(a.accum):
                if stopped or time.monotonic() >= deadline:
                    complete = False
                    break
                batch = next(data)
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cuda):
                    output, loss = model.training_step(batch, step)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at update {step}")
                (loss / a.accum).backward()
                total_loss += float(loss.detach()) / a.accum
                l1 = output.get("demo_sample_action_l1_loss")
                if l1 is not None and torch.isfinite(l1):
                    action_loss += float(l1.detach())
                    policy_batches += 1
                # Release the preceding graph before evaluating the next batch.
                del output, loss, batch
            if not complete:
                optimizer.zero_grad(set_to_none=True)
                break
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if cuda:
                torch.cuda.synchronize()
            step += 1
            times.append(time.monotonic() - before)
            losses.append(total_loss)
            action_losses.append(action_loss / policy_batches if policy_batches else None)
            print(json.dumps(dict(step=step, step_seconds=times[-1], loss=total_loss,
                action_l1=action_losses[-1], lr=float(optimizer.param_groups[0]["lr"]))), flush=True)
            if not a.probe and step < target and (step % a.save_every == 0 or step % a.export_every == 0):
                save_checkpoint(a, model, optimizer, scheduler, step, export=step % a.export_every == 0, cuda=cuda)
        if step > first_step:
            save_checkpoint(a, model, optimizer, scheduler, step, export=True, cuda=cuda)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    measured = times[min(2, len(times) - 1):] if times else []
    report = dict(completed_updates=step - first_step, total_step=step,
        elapsed_seconds=time.monotonic() - started, effective_batch=a.batch * a.accum,
        step_seconds=times, median_step_seconds=float(np.median(measured)) if measured else None,
        loss=losses, action_l1=action_losses,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else 0,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30 if cuda else 0,
        stop_reason="target" if step == target else "signal" if stopped else "max_hours")
    if measured:
        report["projected_remaining_hours"] = (a.steps - step) * float(np.median(measured)) / 3600
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / ("probe.json" if a.probe else "train_report.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main(argv=None):
    a = parse_args(argv)
    if a.print_config:
        print(json.dumps(dict(vars(a), effective_batch=a.batch * a.accum,
            experiment="cosmos_predict2_2b_480p_libero", options=experiment_options(a, a.tokenizer or "LOCAL_WAN_VAE")), indent=2))
        return
    started = time.monotonic()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.set_device(0)
    model, optimizer, scheduler = load_native(a)
    dataset = MT50Dataset(a.prepared, a.assets, seed=a.seed)
    fit(a, model, optimizer, scheduler, dataset, started=started)


if __name__ == "__main__":
    main()
