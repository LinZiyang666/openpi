"""Offline convention check on real source frames; not a success-rate estimate."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from .policy import load_policy
from .preprocess import image_tensor, quantile_normalize, quantile_unnormalize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=("metaworld", "libero"), required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); torch.set_num_threads(4)
    policy = load_policy(a.checkpoint, a.device, domain=a.domain)
    rows = pq.read_table(a.episode).to_pylist()
    tasks = {row["task_index"]: row["task"] for row in map(json.loads, a.tasks.read_text().splitlines())}
    frames = np.linspace(0, len(rows)-1, 8, dtype=int)
    base_key, wrist_key, state_key = ("image", "wrist_image", "state") if a.domain == "libero" else ("corner4.image", "gripperPOV.image", "observation.state")
    images = {}
    for key, source_key in (("base_0_rgb", base_key), ("left_wrist_0_rgb", wrist_key)):
        images[key] = torch.stack([image_tensor(np.asarray(Image.open(io.BytesIO(rows[i][source_key]["bytes"])).convert("RGB"))) for i in frames]).to(policy.device)
    state = np.asarray([rows[i][state_key] for i in frames], np.float32)
    actions = np.asarray([[rows[min(i+j, len(rows)-1)]["actions"] for j in range(16)] for i in frames], np.float32)
    lang = policy.language([tasks[rows[i]["task_index"]] for i in frames]).to(policy.device)
    gen = torch.Generator(device=policy.device).manual_seed(17)
    noise = torch.randn(actions.shape, device=policy.device, generator=gen)
    t = torch.tensor([10, 30, 50, 80] * 2, device=policy.device)
    results = []
    original_order = policy.model._policy.obs_encoder.feature_order
    with torch.inference_mode():
        for normalization in ("quantile", "zscore", "none"):
            def normalize(values, key):
                s = policy.stats[key]
                if normalization == "quantile": return quantile_normalize(values, s)
                if normalization == "zscore": return (values - np.asarray(s["mean"])) / (np.asarray(s["std"])+1e-6)
                return values
            obs = dict(images, state=torch.as_tensor(normalize(state, "state"), dtype=torch.float32, device=policy.device), lang_emb=lang)
            action = torch.as_tensor(normalize(actions, "actions"), dtype=torch.float32, device=policy.device)
            action = policy.model._policy.normalizer.normalize("action", action)
            noisy = policy.model.scheduler().add_noise(action, noise, t)
            for view in ("source_pair", "zero", "duplicate", "swapped"):
                condition_obs = dict(obs)
                if view == "zero": condition_obs["left_wrist_0_rgb"] = torch.zeros_like(images["left_wrist_0_rgb"])
                if view == "duplicate": condition_obs["left_wrist_0_rgb"] = images["base_0_rgb"]
                if view == "swapped": condition_obs.update(base_0_rgb=images["left_wrist_0_rgb"], left_wrist_0_rgb=images["base_0_rgb"])
                condition = policy.model.condition(condition_obs)
                pred = policy.model._policy.model(noisy, t, condition)
                results.append(dict(external_normalization=normalization, view=view, epsilon_mse=float(F.mse_loss(pred, noise))))
        # Fixed-noise physical16-action prediction on the native source pair.
        obs = dict(images, state=torch.as_tensor(quantile_normalize(state, policy.stats["state"]), device=policy.device), lang_emb=lang)
        prediction = policy.model.sample(obs, generator=torch.Generator(device=policy.device).manual_seed(0)).cpu().numpy()
        prediction = quantile_unnormalize(prediction, policy.stats["actions"])
        qerr = quantile_normalize(prediction, policy.stats["actions"]) - quantile_normalize(actions, policy.stats["actions"])
    report = dict(domain=a.domain, checkpoint=str(a.checkpoint), source_episode=str(a.episode), frames=frames.tolist(),
        native_pair_physical_actions_finite=bool(np.isfinite(prediction).all()), action_shape=list(prediction.shape),
        native_pair_quantile_action_mse=float(np.mean(qerr**2)), variants=results, policy_metadata=policy.get_metadata(),
        limitation="Eight source frames from one demonstration; teacher-forced noise check is not SR or proof of unpublished wrapper identity")
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
