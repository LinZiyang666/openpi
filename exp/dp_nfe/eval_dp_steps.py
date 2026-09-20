"""Evaluate an official Diffusion Policy checkpoint at a chosen number of denoising steps (DDPM or DDIM).

Mirrors ``eval.py`` of real-stanford/diffusion_policy (checkpoint -> workspace -> EMA policy -> task env_runner)
and only changes: ``policy.num_inference_steps`` (and, for ``--scheduler ddim``, swaps the DDPMScheduler for a
DDIMScheduler built from the same training config), the runner episode counts (``n_test`` seeded test episodes,
``n_train`` 0, no videos) and the dataset path used for env metadata (the lowdim hdf5 carries the same env_meta).
Also times ``predict_action`` at batch size 1 on a synthetic observation of the checkpoint's shape_meta.

Run from the diffusion_policy repo root inside its env::

    python eval_dp_steps.py --checkpoint ckpts/square_mh/epoch=...ckpt --output_dir out/square_mh_ddim4 \\
        --steps 4 --scheduler ddim --n_test 50 --dataset_path data/robomimic/datasets/square/mh/low_dim_abs.hdf5
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import click
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--steps", type=int, required=True, help="num_inference_steps")
@click.option("--scheduler", type=click.Choice(["ddpm", "ddim"]), default="ddpm")
@click.option("--n_test", type=int, default=50)
@click.option("--n_envs", type=int, default=0, help="parallel envs (0 = n_test capped at 25)")
@click.option("--dataset_path", default="", help="hdf5 for env metadata (robomimic tasks); empty = checkpoint's own path")
@click.option("--max_steps", type=int, default=0, help="override episode max steps (0 = checkpoint's)")
@click.option("--latency_only", is_flag=True, help="measure batch-1 latency, write latency.json and exit (no rollouts)")
def main(checkpoint, output_dir, device, steps, scheduler, n_test, n_envs, dataset_path, max_steps, latency_only):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # --- denoising steps / scheduler
    train_sched = policy.noise_scheduler
    train_cfg = dict(train_sched.config)
    if scheduler == "ddim":
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        keep = {k: v for k, v in train_cfg.items() if k in ("num_train_timesteps", "beta_start", "beta_end", "beta_schedule", "clip_sample", "prediction_type")}
        policy.noise_scheduler = DDIMScheduler(set_alpha_to_one=True, steps_offset=0, **keep)
    policy.num_inference_steps = steps
    print(f"DPSTEPS scheduler={scheduler} steps={steps} train_timesteps={train_cfg.get('num_train_timesteps')} pred={train_cfg.get('prediction_type')}", flush=True)

    # --- batch-1 latency on a synthetic observation of the policy's shape_meta
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    n_obs = cfg.n_obs_steps
    obs = {}
    for key, attr in shape_meta["obs"].items():
        obs[key] = torch.rand(1, n_obs, *attr["shape"], device=device)
    with torch.no_grad():
        for _ in range(3):
            policy.predict_action(obs)
        torch.cuda.synchronize()
        lat = []
        for _ in range(10):
            t0 = time.perf_counter()
            policy.predict_action(obs)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    lat_ms = float(np.mean(lat))
    print(f"DPSTEPS batch1 predict_action mean={lat_ms:.1f} ms (10 calls, {torch.cuda.get_device_name(device)})", flush=True)
    if latency_only:
        json.dump({"checkpoint": checkpoint, "scheduler": scheduler, "steps": steps, "batch1_latency_ms": round(lat_ms, 2),
                   "batch1_latency_ms_all": [round(x, 2) for x in lat], "device": torch.cuda.get_device_name(device)},
                  open(os.path.join(output_dir, "latency.json"), "w"), indent=2)
        print(f"DPSTEPS LATENCY DONE scheduler={scheduler} steps={steps} mean={lat_ms:.1f} ms", flush=True)
        return

    # --- env runner: seeded test episodes only, no videos
    rcfg = cfg.task.env_runner
    OmegaConf.set_struct(rcfg, False)
    rcfg.n_train = 0
    rcfg.n_train_vis = 0
    rcfg.n_test = n_test
    rcfg.n_test_vis = 0
    rcfg.n_envs = n_envs if n_envs > 0 else min(n_test, 25)
    if dataset_path:
        rcfg.dataset_path = dataset_path
    if max_steps > 0:
        rcfg.max_steps = max_steps
    # the lowdim hdf5's env_kwargs carry no camera setup; derive it from the checkpoint's rgb obs keys so the
    # robosuite env renders exactly the images the policy was trained on (84x84 agentview + eye-in-hand)
    # (PushT's single rgb key is plain ``image`` and its gym env needs no camera kwargs -> no injection there)
    cams = [k[: -len("_image")] for k in shape_meta["obs"] if k.endswith("_image")]
    if cams:
        import robomimic.utils.file_utils as FileUtils

        _orig_meta = FileUtils.get_env_metadata_from_dataset
        h = shape_meta["obs"][cams[0] + "_image"]["shape"][-2]
        w = shape_meta["obs"][cams[0] + "_image"]["shape"][-1]

        def _meta_with_cams(dataset_path, *a, **k):
            m = _orig_meta(dataset_path, *a, **k)
            m["env_kwargs"].update(camera_names=cams, camera_heights=h, camera_widths=w, camera_depths=False,
                                   use_camera_obs=True, has_offscreen_renderer=True)
            return m

        FileUtils.get_env_metadata_from_dataset = _meta_with_cams
        print(f"DPSTEPS env cameras={cams} {h}x{w}", flush=True)
    env_runner = hydra.utils.instantiate(rcfg, output_dir=output_dir)

    calls = {"n": 0, "ms": 0.0}
    orig = policy.predict_action

    def timed(obs_dict):
        t0 = time.perf_counter()
        out = orig(obs_dict)
        torch.cuda.synchronize()
        calls["n"] += 1
        calls["ms"] += (time.perf_counter() - t0) * 1000
        return out

    policy.predict_action = timed
    t0 = time.time()
    runner_log = env_runner.run(policy)
    wall = time.time() - t0
    json_log = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in runner_log.items()}
    summary = {
        "checkpoint": checkpoint,
        "task": cfg.task.name if "name" in cfg.task else str(cfg.task.get("task_name", "")),
        "scheduler": scheduler,
        "steps": steps,
        "n_test": n_test,
        "n_envs": rcfg.n_envs,
        "test_mean_score": json_log.get("test/mean_score"),
        "batch1_latency_ms": round(lat_ms, 2),
        "runner_predict_calls": calls["n"],
        "runner_predict_ms_mean_per_batch": round(calls["ms"] / max(calls["n"], 1), 2),
        "wall_s": round(wall, 1),
        "runner_log": json_log,
    }
    json.dump(summary, open(os.path.join(output_dir, "summary.json"), "w"), indent=2, sort_keys=True)
    print(f"DPSTEPS DONE task={summary['task']} scheduler={scheduler} steps={steps} test/mean_score={summary['test_mean_score']} wall={wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
