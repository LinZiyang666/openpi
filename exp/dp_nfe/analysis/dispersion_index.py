"""Conditional action-distribution diagnostics for the x0-head experiment (plan §4.3).

Pure-numpy core (CPU-testable):
    dispersion(samples)          "conditional sampling dispersion": mean pairwise L2 between action chunks sampled for the
                                 same observation history, after per-dimension scaling by the training-pool action std
    mixture_diagnostic(samples)  PCA -> 2D, GMM(1) vs GMM(2) delta-BIC and a silhouette score of the 2-cluster split;
                                 a *fit diagnostic*, not a mode count (see plan §4.3); shipped with synthetic controls
    synthetic_controls()         unimodal high-variance / separated bimodal / skewed unimodal / bimodal off the first two
                                 PCs — the last two exist to show where the diagnostic misjudges
DP-env driver (``python -m exp.dp_nfe.analysis.dispersion_index sample ...``): draws ``n_samples`` executed action
slices per held-out observation history from a checkpoint with the trailing sampler, keyed initial noise shared across
arms, and writes ``diagnostics/<cell>_<sampler_k>.json`` with the raw samples' statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, Tuple

import numpy as np


# ----------------------------------------------------------------------------- core
def dispersion(samples: np.ndarray, action_std: np.ndarray) -> Dict[str, float]:
    """samples [N, T, D] chunks for one history; action_std [D] (zero-variance dims are dropped and counted)."""
    if samples.ndim != 3:
        raise ValueError("samples must be [N, T, D]")
    std = np.asarray(action_std, dtype=np.float64)
    keep = std > 1e-8
    x = samples[..., keep] / std[keep]
    flat = x.reshape(x.shape[0], -1)
    n = flat.shape[0]
    if n < 2:
        raise ValueError("need at least two samples")
    diff = flat[:, None, :] - flat[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(n, 1)
    return {"mean_pairwise_l2": float(d[iu].mean()), "n_samples": int(n), "n_dims_used": int(keep.sum()),
            "n_dims_dropped": int((~keep).sum()), "dim_scale": "per-dim training-pool action std"}


def pca2(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """First two principal components of ``x`` [N, F] and the fraction of variance they explain."""
    xc = x - x.mean(0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    z = xc @ vt[:2].T
    var = (s ** 2)
    return z, float(var[:2].sum() / max(var.sum(), 1e-12))


def _gmm_em(z: np.ndarray, k: int, iters: int = 200, seed: int = 0) -> float:
    """Diagonal-covariance GMM log-likelihood via EM (tiny, dependency-free)."""
    n, d = z.shape
    # deterministic init: k=1 -> overall mean; k=2 -> means of the two halves split at the median of the first axis
    if k == 1:
        mu = z.mean(0, keepdims=True).copy()
    else:
        order = np.argsort(z[:, 0])
        parts = np.array_split(order, k)
        mu = np.stack([z[pp].mean(0) for pp in parts])
    var = np.full((k, d), z.var(0) + 1e-6)
    pi = np.full(k, 1.0 / k)
    ll_old = -np.inf
    for _ in range(iters):
        logp = np.stack([np.log(pi[j] + 1e-12) - 0.5 * (np.log(2 * np.pi * var[j]).sum() + (((z - mu[j]) ** 2) / var[j]).sum(1))
                         for j in range(k)], 1)
        m = logp.max(1, keepdims=True)
        ll = float((m[:, 0] + np.log(np.exp(logp - m).sum(1))).sum())
        r = np.exp(logp - m); r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        pi = nk / n
        mu = (r.T @ z) / nk[:, None]
        var = np.stack([(r[:, j:j + 1] * (z - mu[j]) ** 2).sum(0) / nk[j] for j in range(k)]) + 1e-6
        if abs(ll - ll_old) < 1e-8:
            break
        ll_old = ll
    return ll


def bic(ll: float, n: int, k: int, d: int) -> float:
    """BIC of a k-component diagonal GMM in d dims with log-likelihood ``ll`` on n points."""
    p = k * (2 * d) + (k - 1)
    return -2 * ll + p * math.log(n)


def silhouette2(z: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette of a 2-label split of ``z`` [N, 2] (0.0 when a label is empty)."""
    n = len(z)
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    s = []
    for i in range(n):
        same = labels == labels[i]; same[i] = False
        other = labels != labels[i]
        if same.sum() == 0 or other.sum() == 0:
            return 0.0
        a = d[i, same].mean(); b = d[i, other].mean()
        s.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(s))


def mixture_diagnostic(samples: np.ndarray, action_std: np.ndarray, seed: int = 0) -> Dict[str, float]:
    """PCA(2) of the std-scaled chunks, GMM(2)-vs-GMM(1) delta-BIC (> 0 favours two components) and the silhouette of the
    median split along PC1 -- a fit diagnostic, not a mode count."""
    std = np.asarray(action_std, dtype=np.float64); keep = std > 1e-8
    if not keep.any():
        return {"delta_bic_2_vs_1": None, "silhouette_pc1_split": None,
                "pca_explained_2d": None, "n_samples": int(samples.shape[0]),
                "reason": "all action dimensions have zero training variance"}
    flat = (samples[..., keep] / std[keep]).reshape(samples.shape[0], -1)
    z, explained = pca2(flat)
    n, d = z.shape
    ll1 = _gmm_em(z, 1, seed=seed); ll2 = _gmm_em(z, 2, seed=seed)
    dbic = bic(ll1, n, 1, d) - bic(ll2, n, 2, d)  # > 0 favours two components
    # 2-cluster split along the first PC (deterministic) for the silhouette
    labels = (z[:, 0] > np.median(z[:, 0])).astype(int)
    return {"delta_bic_2_vs_1": float(dbic), "silhouette_pc1_split": silhouette2(z, labels),
            "pca_explained_2d": explained, "n_samples": int(n)}


def synthetic_controls(seed: int = 0, n: int = 128) -> Dict[str, np.ndarray]:
    """Four synthetic [n, 8, 2] sample clouds: unimodal high-variance, separated bimodal, skewed unimodal and bimodal
    only off the first two PCs (the last two show where the diagnostic misjudges)."""
    rng = np.random.default_rng(seed)
    T, D = 8, 2
    uni = rng.normal(size=(n, T, D)) * 1.5                                   # unimodal, high variance
    bi = np.concatenate([rng.normal(size=(n // 2, T, D)) * 0.2 + 2.0, rng.normal(size=(n // 2, T, D)) * 0.2 - 2.0])
    skew = np.exp(rng.normal(size=(n, T, D)) * 0.8)                          # skewed unimodal
    off = rng.normal(size=(n, T, D)) * 1.5; off[: n // 2, 0, 0] += 0.0       # bimodal only in a low-variance dim
    off[:, -1, -1] = np.where(np.arange(n) < n // 2, 0.3, -0.3) + rng.normal(size=n) * 0.02
    return {"unimodal_high_var": uni, "bimodal_separated": bi, "skewed_unimodal": skew, "bimodal_off_pc": off}


def check_controls(seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Mixture + dispersion diagnostics of :func:`synthetic_controls` (unit std)."""
    out = {}
    for name, s in synthetic_controls(seed).items():
        std = np.ones(s.shape[-1])
        out[name] = {**mixture_diagnostic(s, std, seed), **dispersion(s, std)}
    return out


# ----------------------------------------------------------------------------- DP-env driver
NORMALIZER_STD_KEY = "params_dict.action.input_stats.std"


def action_std_from_state_dict(sd) -> np.ndarray:
    """Per-dimension action std of a DP ``LinearNormalizer`` state_dict (``params_dict.action.input_stats.std``)."""
    if NORMALIZER_STD_KEY not in sd:
        raise KeyError(f"{NORMALIZER_STD_KEY} not in the normalizer state_dict (keys: {sorted(sd)[:5]}...)")
    return np.asarray(sd[NORMALIZER_STD_KEY].detach().cpu().numpy(), dtype=np.float64).reshape(-1)


def batch_obs(item: dict, n: int, device) -> dict:
    """Repeat one dataset item's observation ``n`` times: lowdim items carry ``obs`` as a tensor, image items as a dict
    of tensors (cameras + low-dim keys); ``action`` is dropped."""
    import torch

    def rep(v):
        return v.unsqueeze(0).repeat(n, *([1] * v.ndim)).to(device)
    obs = item["obs"]
    if isinstance(obs, dict):
        return {k: rep(v) for k, v in obs.items()}
    if not torch.is_tensor(obs):
        raise TypeError(f"unsupported obs type {type(obs)}")
    return {"obs": rep(obs)}


def executed_slice(policy) -> slice:
    """The executed dataset action slice, including the lowdim oa_step_convention offset."""
    start = int(policy.n_obs_steps) - int(getattr(policy, "oa_step_convention", True))
    return slice(start, start + int(policy.n_action_steps))


def observation_feature(item: dict, policy) -> np.ndarray:
    """Shared-normalizer history features; image channels use fixed 8x8 average pooling."""
    import torch
    obs = item["obs"]
    obs = obs if isinstance(obs, dict) else {"obs": obs}
    parts = []
    for key in sorted(obs):
        value = obs[key][:int(policy.n_obs_steps)].to(policy.device)
        value = policy.normalizer[key].normalize(value)
        if value.ndim == 4:
            value = torch.nn.functional.adaptive_avg_pool2d(value, (8, 8))
        parts.append(value.detach().cpu().numpy().reshape(-1))
    return np.concatenate(parts).astype(np.float64)


def paired_clip_samples(policy, obs: dict, ts, keyed, seed: int):
    """Run both complete denoising chains with the same history/sample noise keys and encoder RNG."""
    import torch
    keyed.on_reset()
    clip = ts.clip
    devices = [policy.device.index or 0] if policy.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            clipped = policy.predict_action(obs)["action"].cpu().numpy().astype(np.float64)
            keyed.decision = None
            ts.clip = False
            torch.manual_seed(seed)
            unclipped = policy.predict_action(obs)["action"].cpu().numpy().astype(np.float64)
    finally:
        ts.clip = clip
    return clipped, unclipped


def nearest_demo_distance(samples: np.ndarray, pool: np.ndarray, action_std: np.ndarray) -> Dict[str, float]:
    """Min L2 (per-dim std-scaled, zero-variance dims dropped) from every sampled chunk [N, T, D] to the training-pool
    executed chunks [M, T, D]; returns mean / median / max over samples."""
    std = np.asarray(action_std, dtype=np.float64); keep = std > 1e-8
    s = (samples[..., keep] / std[keep]).reshape(samples.shape[0], -1)
    p = (pool[..., keep] / std[keep]).reshape(pool.shape[0], -1)
    # blockwise to bound memory
    mins = np.empty(s.shape[0])
    for i in range(0, s.shape[0], 256):
        blk = s[i:i + 256]
        d2 = (blk ** 2).sum(1)[:, None] + (p ** 2).sum(1)[None, :] - 2 * blk @ p.T
        mins[i:i + 256] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return {"nearest_demo_mean": float(mins.mean()), "nearest_demo_median": float(np.median(mins)),
            "nearest_demo_max": float(mins.max())}


def clip_diagnostics(clipped: np.ndarray, unclipped: np.ndarray) -> Dict[str, float]:
    """Paired whole-chain clipping ablation in normalised action space: the fraction of coordinates of the *unclipped*
    samples outside the range, their mean exceedance and the mean |clipped - unclipped|."""
    out = np.abs(unclipped) > 1.0
    return {"frac_coords_out_of_range_unclipped": float(out.mean()),
            "mean_exceedance_unclipped": float(np.maximum(np.abs(unclipped) - 1.0, 0.0).mean()),
            "mean_abs_diff_clip_vs_unclipped": float(np.abs(clipped - unclipped).mean())}


def sample_from_checkpoint(dp_root, ckpt, heldout_cfg, out_json, n_histories=64, n_samples=64, sampler="ddim", k=100,
                           sampling_seed=0, device="cuda:0", normalizer_path=None, trainpool_cfg=None, eps_mode="recompute"):
    """Draw ``n_samples`` executed action chunks per held-out history (keyed initial noise shared across arms) and write
    per-history dispersion, mixture, paired clipping ablation and observation-local demo diagnostics to ``out_json``.

    * ``heldout_cfg`` / ``trainpool_cfg``: hydra dataset configs (dicts) of the held-out subset and of the training pool
      (the latter gives the nearest-demo reference; optional);
    * ``normalizer_path``: the frozen training-pool normaliser (``x0_normalizer.py``); its per-dim action std scales
      every distance. Without it the checkpoint's own normaliser std is used and recorded as ``std_source=checkpoint``;
    * works for lowdim and image policies (observation dict dispatch in :func:`batch_obs`)."""
    import sys, os
    sys.path.insert(0, str(dp_root)); os.chdir(str(dp_root))
    import hydra, torch
    from omegaconf import OmegaConf
    from exp.dp_nfe import dp_sampler
    from exp.dp_nfe.eval_dp_steps_v2 import load_workspace, KeyedNoise
    ws, cfg, policy, identity = load_workspace(pathlib.Path(dp_root), pathlib.Path(ckpt), pathlib.Path(out_json).parent)
    from exp.dp_nfe.x0_identity import sha256_path, verify_data_identity
    verify_data_identity(identity)
    for dataset_cfg, prefix in ((heldout_cfg, "heldout"), (trainpool_cfg, "trainpool")):
        if not dataset_cfg:
            continue
        if float(dataset_cfg.get("val_ratio", 0)) != 0:
            raise ValueError("diagnostic datasets require val_ratio=0")
        if prefix == "heldout" and identity.get("heldout_sha256"):
            paths = [dataset_cfg[k] for k in ("zarr_path", "dataset_dir", "dataset_path") if dataset_cfg.get(k)]
            if len(paths) != 1 or sha256_path(pathlib.Path(paths[0])) != identity["heldout_sha256"]:
                raise ValueError("diagnostic held-out data differs from checkpoint identity")
        if prefix == "trainpool" and identity.get("subset_manifest_path"):
            manifest = json.loads(pathlib.Path(identity["subset_manifest_path"]).read_text())
            hashes = manifest["image_exports" if identity["modality"] == "image" else "exports"]
            paths = [dataset_cfg[k] for k in ("zarr_path", "dataset_dir", "dataset_path") if dataset_cfg.get(k)]
            if len(paths) != 1 or sha256_path(pathlib.Path(paths[0])) != hashes["trainpool"]:
                raise ValueError("diagnostic reference differs from frozen training pool")
    policy.to(device); policy.eval()
    ds = hydra.utils.instantiate(OmegaConf.create(heldout_cfg))
    if n_histories < 1 or n_samples < 2 or len(ds) == 0:
        raise ValueError("diagnostics need a nonempty held-out dataset, histories > 0 and samples >= 2")
    rng = np.random.default_rng(sampling_seed)
    idx = np.sort(rng.choice(len(ds), size=min(n_histories, len(ds)), replace=False))
    ts = dp_sampler.install(policy, sampler, k, eps_mode=eps_mode)
    keyed = KeyedNoise(sampling_seed, start_seed=0, n_envs=n_samples)
    ts.noise_fn = lambda shape, dtype, dev, gen: keyed.initial_noise(shape, dtype, dev)
    ts.step_noise_fn = keyed.step_noise
    if normalizer_path:
        from exp.dp_nfe.x0_normalizer import normalizer_sha256
        sd = torch.load(str(normalizer_path), map_location="cpu")
        if identity.get("normalizer_sha256") and normalizer_sha256(sd) != identity["normalizer_sha256"]:
            raise ValueError("diagnostic normalizer differs from checkpoint identity")
        std = action_std_from_state_dict(sd); std_source = {"std_source": "frozen", "normalizer_path": str(normalizer_path),
                                                            "normalizer_sha256": normalizer_sha256(sd)}
    else:
        from exp.dp_nfe.x0_normalizer import normalizer_sha256
        sd = policy.normalizer.state_dict()
        if identity.get("normalizer_sha256") and normalizer_sha256(sd) != identity["normalizer_sha256"]:
            raise ValueError("checkpoint normalizer differs from frozen identity")
        std = action_std_from_state_dict(sd)
        std_source = {"std_source": "checkpoint", "normalizer_sha256": normalizer_sha256(sd)}
    sl = executed_slice(policy)
    pool = None
    if trainpool_cfg:
        tp = hydra.utils.instantiate(OmegaConf.create(trainpool_cfg))
        if not len(tp):
            raise ValueError("empty diagnostic training pool")
        ref_indices = np.sort(np.random.default_rng(sampling_seed).choice(len(tp), min(4096, len(tp)), replace=False))
        chunks, features = [], []
        for i in ref_indices:
            item = tp[int(i)]
            chunks.append(np.asarray(item["action"])[sl])
            features.append(observation_feature(item, policy))
        pool = np.stack(chunks).astype(np.float64)
        features = np.stack(features)
    results = []
    for hist_i, i in enumerate(idx):
        item = ds[int(i)]
        obs = batch_obs(item, n_samples, device)
        act, act_unclipped = paired_clip_samples(policy, obs, ts, keyed, sampling_seed + int(i))
        norm_c = policy.normalizer["action"].normalize(torch.as_tensor(act, dtype=torch.float32, device=device)).cpu().numpy()
        norm_u = policy.normalizer["action"].normalize(torch.as_tensor(act_unclipped, dtype=torch.float32, device=device)).cpu().numpy()
        rec = {"window": int(i), **dispersion(act, std), **mixture_diagnostic(act, std), **clip_diagnostics(norm_c, norm_u)}
        if pool is not None:
            distance = ((features - observation_feature(item, policy)) ** 2).mean(1)
            neighbors = np.argsort(distance, kind="stable")[:min(32, len(pool))]
            rec.update(nearest_demo_distance(act, pool[neighbors], std))
            rec["neighbor_windows"] = ref_indices[neighbors].tolist()
        results.append(rec)
    summary = {"checkpoint": str(ckpt), "cell": identity, "sampler": ts.describe(), "n_histories": len(results),
               "n_samples": n_samples, **std_source, "n_dims_dropped": int((std <= 1e-8).sum()),
               "trainpool_windows": None if pool is None else int(pool.shape[0]), "per_history": results,
               "reference_windows": None if pool is None else ref_indices.tolist(),
               "neighbor_rule": "32 nearest normalized observation histories in a fixed <=4096-window training-pool sample; images pooled to 8x8",
               "clip_comparison": "paired full denoising chains with clipping enabled/disabled, not a single-step pre/post comparison",
               "mean_dispersion": float(np.mean([r["mean_pairwise_l2"] for r in results])),
               "mean_delta_bic": (float(np.mean([r["delta_bic_2_vs_1"] for r in results])) if (std > 1e-8).any() else None),
               "mean_frac_out_of_range_unclipped": float(np.mean([r["frac_coords_out_of_range_unclipped"] for r in results])),
               "mean_nearest_demo": None if pool is None else float(np.mean([r["nearest_demo_mean"] for r in results])),
               "interpretation": "dispersion/BIC are fit diagnostics of the conditional sample cloud, not mode counts (plan §4.3)"}
    pathlib.Path(out_json).write_text(json.dumps(summary, indent=1, default=str))
    print("DISPERSION DONE", out_json, summary["mean_dispersion"], summary["mean_delta_bic"])


def main() -> None:
    """CLI: ``controls`` prints the synthetic-control diagnostics; ``sample`` runs :func:`sample_from_checkpoint`."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("controls"); c.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sample")
    s.add_argument("--dp-root", required=True); s.add_argument("-c", "--checkpoint", required=True)
    s.add_argument("--heldout-cfg", required=True, help="json of the hydra dataset config for the held-out subset")
    s.add_argument("--trainpool-cfg", default=None, help="json of the hydra dataset config for the training pool (nearest-demo)")
    s.add_argument("--normalizer", default=None, help="frozen training-pool normalizer (.pt) for the per-dim action std")
    s.add_argument("-o", "--out", required=True); s.add_argument("--n-histories", type=int, default=64)
    s.add_argument("--n-samples", type=int, default=64); s.add_argument("--sampler", default="ddim"); s.add_argument("--k", type=int, default=100)
    s.add_argument("--sampling-seed", type=int, default=0); s.add_argument("--device", default="cuda:0")
    s.add_argument("--eps-mode", default="recompute")
    a = ap.parse_args()
    if a.cmd == "controls":
        print(json.dumps(check_controls(a.seed), indent=1))
    else:
        sample_from_checkpoint(a.dp_root, a.checkpoint, json.loads(a.heldout_cfg), a.out, a.n_histories, a.n_samples,
                               a.sampler, a.k, a.sampling_seed, a.device, a.normalizer,
                               json.loads(a.trainpool_cfg) if a.trainpool_cfg else None, a.eps_mode)


if __name__ == "__main__":
    main()
