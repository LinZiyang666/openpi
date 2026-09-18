"""Step-ladder evaluation of a Diffusion Policy checkpoint (official or FixedStepWorkspace) with the trailing-grid
samplers of ``dp_sampler.py`` and per-episode scores in the aggregation schema. Runs inside the official DP env::

    PYTHONPATH=<openpi root>:<dp root> python -m exp.dp_nfe.eval_dp_steps_v2 --dp-root <dp root> -c <ckpt> -o <out dir> \
        --sampler ddim --k 1 --split test --n-test 100 --n-envs 25 \
        [--runner-override task.env_runner.dataset_path=/data/...] [--sampling-seed 0] [--attempt 1]

* input dispatch: lowdim policies get ``{'obs': [B, n_obs_steps, obs_dim]}`` from their runner unchanged; robomimic image
  runners need the camera kwargs injected into the lowdim hdf5 env meta (as the v1 evaluator did);
* the policy's ``conditional_sample`` is replaced by :class:`dp_sampler.TrailingSampler`; **every** random draw is keyed by
  ``(sampling_seed, episode_id, decision_idx, noise_kind, timestep)`` -- the initial noise (``kind=init``, timestep -1)
  and the DDPM posterior noise of every step (``kind=ddpm``, timestep t) -- where ``episode_id = start_seed + chunk *
  n_envs + row`` is the runner's env seed of that episode. The noise of an episode therefore does not depend on
  ``n_envs``, on the chunking, on other episodes terminating early or on a retry (G2 R1-B7);
* the cell identity is **bound to the checkpoint payload** (FixedStep checkpoints carry it; official checkpoints get a
  derived ``variant=official`` identity) -- there is no command-line label (G2 R1-B8);
* per-episode scores come from the runners' ``test/sim_max_reward_<seed>`` keys; the kitchen runner has none, so its
  ``env.call('get_attr', 'reward')`` results are recorded per chunk and turned into ``sum(reward)/7`` per episode;
* ``summary.json``: ``{"cell", "sampler", "split", "episode_ids_expected", "episodes", "complete", "error", "attempt",
  "manifest"}``; any exception during the rollout is written as ``complete=false`` + ``error`` and the process exits 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def noise_seed(sampling_seed: int, episode_id: int, decision_idx: int, kind: str, timestep: int) -> int:
    """64-bit seed of one random draw of the evaluation protocol, a pure function of
    ``(sampling_seed, episode_id, decision_idx, noise_kind, timestep)``."""
    key = f"{int(sampling_seed)}|{int(episode_id)}|{int(decision_idx)}|{kind}|{int(timestep)}".encode()
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def keyed_normal(shape, sampling_seed: int, episode_id: int, decision_idx: int, kind: str, timestep: int) -> torch.Tensor:
    """Standard-normal tensor of ``shape`` for one keyed draw (CPU, float32)."""
    g = torch.Generator().manual_seed(noise_seed(sampling_seed, episode_id, decision_idx, kind, timestep))
    return torch.randn(shape, generator=g, dtype=torch.float32)


class KeyedNoise:
    """Keyed randomness for the DP runners: ``policy.reset()`` (once per runner chunk) advances the chunk; episode row
    ``r`` of chunk ``c`` is episode ``start_seed + c * n_envs + r``; each policy call is one decision per row.

    ``initial_noise`` supplies the sampler's x_T per row (``kind=init``), ``step_noise`` the DDPM posterior noise of
    timestep ``t`` per row (``kind=ddpm``). Raises ``RuntimeError`` when used before the first reset."""

    def __init__(self, sampling_seed: int, start_seed: int, n_envs: int):
        self.sampling_seed = int(sampling_seed)
        self.start_seed = int(start_seed)
        self.n_envs = int(n_envs)
        self.chunk_idx = -1
        self.decision = None
        self._current = None  # decision indices of the rows of the call in flight (for step_noise)

    def on_reset(self):
        """Advance to the next runner chunk (called from the patched ``policy.reset``)."""
        self.chunk_idx += 1
        self.decision = None

    def episode_ids(self, n_rows: int):
        """Episode ids of the rows of the current chunk (``RuntimeError`` before the first reset)."""
        if self.chunk_idx < 0:
            raise RuntimeError("KeyedNoise used before policy.reset()")
        return [self.start_seed + self.chunk_idx * self.n_envs + r for r in range(n_rows)]

    def initial_noise(self, shape, dtype, device):
        """x_T of every row for this decision: keyed ``(seed, episode, decision, 'init', -1)``; advances the decision counters."""
        B = shape[0]
        if self.decision is None:
            self.decision = [0] * B
        ids = self.episode_ids(B)
        self._current = list(self.decision)
        rows = [keyed_normal(tuple(shape[1:]), self.sampling_seed, ids[r], self.decision[r], "init", -1) for r in range(B)]
        for r in range(B):
            self.decision[r] += 1
        return torch.stack(rows).to(dtype=dtype, device=device)

    def step_noise(self, t: int, like: torch.Tensor):
        """DDPM posterior noise of timestep ``t`` for every row of the call in flight: keyed ``(seed, episode, decision, 'ddpm', t)``."""
        B = like.shape[0]
        ids = self.episode_ids(B)
        dec = self._current if self._current is not None else [0] * B
        rows = [keyed_normal(tuple(like.shape[1:]), self.sampling_seed, ids[r], dec[r], "ddpm", int(t)) for r in range(B)]
        return torch.stack(rows).to(dtype=like.dtype, device=like.device)


def load_workspace(dp_root: pathlib.Path, ckpt: pathlib.Path, out_dir: pathlib.Path):
    """Instantiate the checkpoint's workspace class, load the weights and return
    ``(workspace, cfg, policy, identity)``. FixedStep checkpoints must be final (``global_step == budget``, else
    ``SystemExit``) and their pickled identity is returned verbatim; official checkpoints get
    ``{"task_name": cfg.task.name, "modality", "variant": "official", "head", "train_seed", "budget_id": "official"}``."""
    import dill
    import hydra
    payload = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    fixed = cfg._target_.endswith("FixedStepWorkspace")
    if fixed:
        gs = dill.loads(payload["pickles"]["global_step"])
        budget = int(cfg.x0.budget_steps)
        if gs != budget:
            raise SystemExit(f"refusing non-final checkpoint: global_step={gs} != budget={budget}")
        if "identity" not in payload["pickles"]:
            raise SystemExit("FixedStep checkpoint without identity")
        identity = dict(dill.loads(payload["pickles"]["identity"]))
        from exp.dp_nfe.x0_workspace import FixedStepWorkspace
        ws = FixedStepWorkspace(cfg, output_dir=str(out_dir))
        ws.load_payload(payload, exclude_keys=None, include_keys=None)
    else:
        cls = hydra.utils.get_class(cfg._target_)
        ws = cls(cfg, output_dir=str(out_dir))
        ws.load_payload(payload, exclude_keys=None, include_keys=None)
        has_img = "shape_meta" in cfg.task and any(k.endswith("_image") or k == "image" for k in cfg.task.shape_meta.obs)
        identity = {"task_name": str(cfg.task.name), "modality": "image" if has_img else "lowdim", "variant": "official",
                    "head": str(cfg.policy.noise_scheduler.prediction_type), "train_seed": int(cfg.training.seed),
                    "budget_id": "official", "cell_id": f"official_{cfg.task.name}"}
    policy = ws.ema_model if getattr(cfg.training, "use_ema", True) else ws.model
    return ws, cfg, policy, identity


def build_runner(cfg, out_dir, n_test, n_envs, start_seed, overrides):
    """Instantiate ``cfg.task.env_runner`` with ``n_train=0``, ``n_test`` test episodes from ``test_start_seed`` in
    chunks of ``n_envs`` (no videos); ``overrides`` are ``task.env_runner.<key>=<yaml scalar>`` strings (anything else
    is a ``SystemExit``). Returns ``(runner, resolved runner config)``."""
    import hydra
    from omegaconf import OmegaConf
    rcfg = cfg.task.env_runner
    OmegaConf.set_struct(rcfg, False)
    rcfg.n_train = 0; rcfg.n_train_vis = 0; rcfg.n_test = int(n_test); rcfg.n_test_vis = 0
    rcfg.n_envs = int(n_envs); rcfg.test_start_seed = int(start_seed)
    for ov in overrides:
        k, v = ov.split("=", 1)
        if not k.startswith("task.env_runner."):
            raise SystemExit(f"runner override must start with task.env_runner.: {ov}")
        OmegaConf.update(rcfg, k[len("task.env_runner."):], yaml_scalar(v), merge=True)
    return hydra.utils.instantiate(rcfg, output_dir=str(out_dir)), OmegaConf.to_container(rcfg, resolve=True)


def yaml_scalar(v: str):
    """Parse a runner-override value as a YAML scalar (``25`` -> int, ``/x`` -> str, ``true`` -> bool)."""
    import yaml
    return yaml.safe_load(v)


def inject_robomimic_cameras(cfg):
    """robomimic image policies: derive camera kwargs from the checkpoint's rgb obs keys (v1 behaviour) and patch
    ``FileUtils.get_env_metadata_from_dataset``; returns ``{"cameras", "h", "w"}`` or None for lowdim policies."""
    from omegaconf import OmegaConf
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True) if "shape_meta" in cfg.task else None
    if not shape_meta:
        return None
    cams = [k[: -len("_image")] for k in shape_meta["obs"] if k.endswith("_image")]
    if not cams:
        return None
    import robomimic.utils.file_utils as FileUtils
    h = shape_meta["obs"][cams[0] + "_image"]["shape"][-2]; w = shape_meta["obs"][cams[0] + "_image"]["shape"][-1]
    orig = FileUtils.get_env_metadata_from_dataset

    def patched(dataset_path, *a, **k):
        m = orig(dataset_path, *a, **k)
        m["env_kwargs"].update(camera_names=cams, camera_heights=h, camera_widths=w, camera_depths=False,
                               use_camera_obs=True, has_offscreen_renderer=True)
        return m
    FileUtils.get_env_metadata_from_dataset = patched
    return {"cameras": cams, "h": h, "w": w}


def collect_episodes(runner, log, start_seed, n_test, reward_chunks):
    """Per-episode scores keyed by env seed (= episode id): the runner keys ``test/sim_max_reward_<seed>``; fallback to
    the recorded kitchen reward lists (``sum/7`` per episode, chunk order). Returns ``(scores, source)``; ``SystemExit``
    when neither yields exactly ``n_test`` episodes."""
    eps = {}
    for k, v in log.items():
        if k.startswith("test/sim_max_reward_"):
            eps[k[len("test/sim_max_reward_"):]] = float(v)
    if eps:
        return eps, "runner_keys"
    if reward_chunks:
        n_envs = len(runner.env_fns)
        i = 0
        for chunk in reward_chunks:
            for r in chunk:
                if i >= n_test:
                    break
                eps[str(start_seed + i)] = float(np.sum(r) / 7.0)  # kitchen: completed sub-tasks / 7
                i += 1
        if len(eps) == n_test:
            return eps, "recorded_env_rewards/7"
    raise SystemExit(f"could not recover per-episode scores (got {len(eps)} of {n_test})")


def main() -> None:
    """CLI entry: load the checkpoint, install the trailing sampler with keyed noise, run the runner and write ``summary.json``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dp-root", required=True)
    ap.add_argument("-c", "--checkpoint", required=True)
    ap.add_argument("-o", "--output-dir", required=True)
    ap.add_argument("--sampler", choices=["ddim", "ddpm"], required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--split", choices=["screen", "test", "pilot"], default="test")
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--n-envs", type=int, default=25)
    ap.add_argument("--start-seed", type=int, default=None, help="default: screen 90000 / test 100000 / pilot 80000")
    ap.add_argument("--sampling-seed", type=int, default=0)
    ap.add_argument("--eps-mode", choices=["recompute", "raw"], default="recompute",
                    help="eps head direction term: recompute from clipped x0 (plan default) or the network's raw eps")
    ap.add_argument("--attempt", type=int, default=1, help="retry counter recorded by the queue (same inputs)")
    ap.add_argument("--runner-override", action="append", default=[])
    ap.add_argument("-d", "--device", default="cuda:0")
    a = ap.parse_args()
    dp_root = pathlib.Path(a.dp_root).resolve()
    sys.path.insert(0, str(dp_root)); os.chdir(str(dp_root))
    from exp.dp_nfe import dp_sampler
    out = pathlib.Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt = pathlib.Path(a.checkpoint)
    ws, cfg, policy, identity = load_workspace(dp_root, ckpt, out)
    device = torch.device(a.device)
    policy.to(device); policy.eval()
    cams = inject_robomimic_cameras(cfg)
    start_seed = a.start_seed if a.start_seed is not None else {"screen": 90000, "test": 100000, "pilot": 80000}[a.split]
    expected_ids = [str(start_seed + i) for i in range(a.n_test)]
    runner, rcfg = build_runner(cfg, out, a.n_test, a.n_envs, start_seed, a.runner_override)

    ts = dp_sampler.install(policy, a.sampler, a.k, eps_mode=a.eps_mode)
    keyed = KeyedNoise(a.sampling_seed, start_seed, a.n_envs)
    orig_reset = policy.reset
    def reset():
        keyed.on_reset(); return orig_reset()
    policy.reset = reset
    ts.noise_fn = lambda shape, dtype, device, generator: keyed.initial_noise(shape, dtype, device)
    ts.step_noise_fn = keyed.step_noise

    reward_chunks = []
    if hasattr(runner, "env") and hasattr(runner.env, "call"):
        orig_env_call = runner.env.call
        def rec_call(name, *args, **kw):
            res = orig_env_call(name, *args, **kw)
            if name == "get_attr" and args and args[0] == "reward":
                reward_chunks.append(list(res))
            return res
        runner.env.call = rec_call

    summary = {
        "cell": identity,
        "sampler": {**ts.describe(), "protocol_id": "trailing_v1", "sampling_seed": a.sampling_seed,
                    "noise_keying": "(sampling_seed, episode_id, decision_idx, kind, timestep)"},
        "split": a.split, "start_seed": start_seed, "n_test": a.n_test, "n_envs": a.n_envs,
        "episode_ids_expected": expected_ids, "episodes": {}, "complete": False, "error": None, "attempt": int(a.attempt),
        "manifest": {"checkpoint": str(ckpt), "checkpoint_sha256": _sha256(ckpt),
                     "fixed_step_workspace": bool(cfg._target_.endswith("FixedStepWorkspace")),
                     "dp_root": str(dp_root), "runner_cfg": rcfg, "cameras": cams,
                     "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
                     "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
    }
    t0 = time.time()
    try:
        log = runner.run(policy)
        wall = time.time() - t0
        episodes, source = collect_episodes(runner, log, start_seed, a.n_test, reward_chunks)
        if sorted(episodes) != sorted(expected_ids):
            raise RuntimeError(f"episode id set mismatch: got {len(episodes)} ids, expected {expected_ids[0]}..{expected_ids[-1]}")
        bad = [k for k, v in episodes.items() if not np.isfinite(v)]
        if bad:
            raise RuntimeError(f"non-finite episode scores: {bad[:5]}")
        summary.update({"episodes": episodes, "complete": True,
                        "mean_score": float(np.mean(list(episodes.values()))),
                        "runner_mean_score": float(log.get("test/mean_score", float("nan")))})
        summary["manifest"].update({"episode_score_source": source, "nfe_total": ts.nfe, "policy_calls": ts.calls,
                                    "wall_s": round(wall, 1)})
    except BaseException as e:  # noqa: BLE001 - any failure must be recorded as incomplete before re-raising
        import traceback
        summary["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
        summary["manifest"]["wall_s"] = round(time.time() - t0, 1)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        print(f"DPSTEPS2 FAIL cell={identity.get('cell_id')} {a.sampler}-{a.k} split={a.split}: {type(e).__name__}: {e}")
        sys.exit(1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"DPSTEPS2 DONE cell={identity.get('cell_id')} task={identity.get('task_name')} variant={identity.get('variant')} "
          f"head={identity.get('head')} seed={identity.get('train_seed')} {a.sampler}-{a.k} eps_mode={a.eps_mode} "
          f"split={a.split} mean={summary['mean_score']:.4f} n={a.n_test} nfe={ts.nfe} wall={wall:.0f}s")


if __name__ == "__main__":
    main()
