"""Minimal stand-ins for the Diffusion Policy modules ``exp.dp_nfe.x0_workspace`` imports, so its control flow (budget,
resume, identity, frozen normaliser, RNG boundary) is testable on CPU without the DP env. Installed into ``sys.modules``
by :func:`install` only when the real ``diffusion_policy`` is not importable; the env_dependent tests use the real one.

The stubs copy the *contracts* the workspace relies on:
* ``BaseWorkspace.save_checkpoint / load_payload`` -- ``{'cfg', 'state_dicts', 'pickles'}`` payloads, state_dict for every
  attribute with state_dict/load_state_dict, dill pickles of ``include_keys`` (transcribed from DP base_workspace.py);
* ``LinearNormalizer`` -- an ``nn.Module`` whose ``load_state_dict`` (re)creates its parameters from the keys
  (``params_dict.<field>.{scale,offset,input_stats.{min,max,mean,std}}``) like DP's DictOfTensorMixin, ``fit`` from a dict
  of arrays, ``normalize``/``unnormalize`` per field;
* ``hydra.utils.instantiate / get_class`` -- ``_target_`` import + call with the remaining keys;
* ``get_scheduler`` -- constant LR with linear warmup (only the *shape* of the schedule matters to the tests);
* a tiny policy (``TinyPolicy``: 2-layer MLP eps/x0 head with DP's ``compute_loss`` structure, ``set_normalizer``,
  ``noise_scheduler`` config) and a window dataset (``ArrayDataset``) that mimic the DP objects the workspace touches.
"""

from __future__ import annotations

import copy
import importlib
import pathlib
import sys
import threading
import types
from typing import Dict

import dill
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- normalizer
class LinearNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.params_dict = nn.ModuleDict()

    def _field(self, name):
        return self.params_dict[name]

    @torch.no_grad()
    def fit(self, data: Dict[str, np.ndarray], mode="limits", output_max=1.0, output_min=-1.0, range_eps=1e-4):
        for k, v in data.items():
            arr = torch.as_tensor(np.asarray(v), dtype=torch.float32).reshape(-1, np.asarray(v).shape[-1])
            mn, mx = arr.min(0).values, arr.max(0).values
            rng = (mx - mn).clamp_min(range_eps)
            scale = (output_max - output_min) / rng
            offset = output_min - scale * mn
            f = nn.ParameterDict({"scale": nn.Parameter(scale, requires_grad=False), "offset": nn.Parameter(offset, requires_grad=False),
                                  "input_stats": nn.ParameterDict({n: nn.Parameter(t, requires_grad=False) for n, t in
                                                                   (("min", mn), ("max", mx), ("mean", arr.mean(0)), ("std", arr.std(0)))})})
            self.params_dict[k] = f

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        params = nn.ModuleDict()
        for key, value in state_dict.items():
            if not key.startswith(prefix + "params_dict."):
                continue
            field, rest = key[len(prefix + "params_dict."):].split(".", 1)
            if field not in params:
                params[field] = nn.ParameterDict({"input_stats": nn.ParameterDict()})
            if rest.startswith("input_stats."):
                params[field]["input_stats"][rest.split(".", 1)[1]] = nn.Parameter(value.clone(), requires_grad=False)
            else:
                params[field][rest] = nn.Parameter(value.clone(), requires_grad=False)
        self.params_dict = params

    def __getitem__(self, key):
        return _Single(self._field(key))

    def normalize(self, x):
        if isinstance(x, dict):
            return {k: self[k].normalize(v) for k, v in x.items()}
        return self["action"].normalize(x)

    def unnormalize(self, x):
        if isinstance(x, dict):
            return {k: self[k].unnormalize(v) for k, v in x.items()}
        return self["action"].unnormalize(x)


class _Single:
    def __init__(self, f):
        self.f = f

    def normalize(self, x):
        return x * self.f["scale"] + self.f["offset"]

    def unnormalize(self, x):
        return (x - self.f["offset"]) / self.f["scale"]

    def get_output_stats(self):
        return {k: v for k, v in self.f["input_stats"].items()}


# ----------------------------------------------------------------------------- scheduler + policy + dataset
class TinyScheduler:
    """DDPM scheduler config surface: cosine alphas_cumprod, betas, add_noise."""

    def __init__(self, num_train_timesteps=100, prediction_type="epsilon", clip_sample=True, **_):
        import math
        T = num_train_timesteps
        def f(t):
            return math.cos((t / T + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = torch.tensor([min(1 - f(i + 1) / f(i), 0.999) for i in range(T)], dtype=torch.float32)
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(1 - betas, 0)
        self.config = types.SimpleNamespace(num_train_timesteps=T, prediction_type=prediction_type, clip_sample=clip_sample,
                                            variance_type="fixed_small")

    def add_noise(self, x0, noise, t):
        a = self.alphas_cumprod[t].reshape(-1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise


class TinyPolicy(nn.Module):
    """eps/x0-head policy on flattened (obs, action) windows with DP's compute_loss structure (normalise, sample t and
    noise from the global torch RNG, MSE against eps or x0)."""

    def __init__(self, noise_scheduler, obs_dim=3, action_dim=2, horizon=4, n_obs_steps=2, n_action_steps=2, hidden=16):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_dim, self.action_dim, self.horizon = obs_dim, action_dim, horizon
        self.n_obs_steps, self.n_action_steps = n_obs_steps, n_action_steps
        d = horizon * action_dim + n_obs_steps * obs_dim + 1
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Mish(), nn.Linear(hidden, horizon * action_dim))

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def model(self, traj, t, local_cond=None, global_cond=None):
        B = traj.shape[0]
        tt = torch.as_tensor(t, dtype=torch.float32, device=traj.device).reshape(-1)
        if tt.numel() == 1:
            tt = tt.repeat(B)
        x = torch.cat([traj.reshape(B, -1), global_cond.reshape(B, -1), tt[:, None] / 100.0], 1)
        return self.net(x).reshape(B, self.horizon, self.action_dim)

    def compute_loss(self, batch):
        nobs = self.normalizer["obs"].normalize(batch["obs"]); nact = self.normalizer["action"].normalize(batch["action"])
        cond = nobs[:, : self.n_obs_steps]
        B = nact.shape[0]
        noise = torch.randn_like(nact)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=nact.device)
        noisy = self.noise_scheduler.add_noise(nact, noise, t)
        pred = self.model(noisy, t, global_cond=cond)
        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else nact
        return torch.nn.functional.mse_loss(pred, target)

    def reset(self):
        pass

    def conditional_sample(self, *a, **k):
        raise NotImplementedError


class ArrayDataset(torch.utils.data.Dataset):
    """Sliding windows over a synthetic episode set stored as a .npz (obs [N, T, Do], action [N, T, Da])."""

    def __init__(self, path, horizon=4, **_):
        d = np.load(path)
        self.obs, self.action = d["obs"].astype(np.float32), d["action"].astype(np.float32)
        self.horizon = horizon
        self.index = [(e, s) for e in range(self.obs.shape[0]) for s in range(self.obs.shape[1] - horizon + 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, s = self.index[i]
        return {"obs": torch.from_numpy(self.obs[e, s:s + self.horizon]), "action": torch.from_numpy(self.action[e, s:s + self.horizon])}

    def get_normalizer(self):
        n = LinearNormalizer(); n.fit({"obs": self.obs, "action": self.action}); return n


class EMAModel:
    def __init__(self, model, power=0.75, **_):
        self.averaged_model = model; self.optimization_step = 0; self.decay = 0.0; self.power = power

    @torch.no_grad()
    def step(self, new_model):
        # DP's EMAModel: trainable params are averaged, non-trainable ones (the normaliser stats) copied verbatim
        self.optimization_step += 1
        self.decay = min(0.999, (1 + self.optimization_step) / (10 + self.optimization_step))
        for p_ema, p in zip(self.averaged_model.parameters(), new_model.parameters()):
            if not p.requires_grad:
                p_ema.copy_(p.detach())
            else:
                p_ema.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


# ----------------------------------------------------------------------------- DP + hydra module stubs
def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    if isinstance(x, dict):
        return {k: _copy_to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_copy_to_cpu(v) for v in x]
    return copy.deepcopy(x)


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg; self._output_dir = output_dir; self._saving_thread = None

    @property
    def output_dir(self):
        return self._output_dir

    def save_checkpoint(self, path=None, tag="latest", exclude_keys=None, include_keys=None, use_thread=True):
        path = pathlib.Path(path) if path else pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"
        exclude_keys = tuple(self.exclude_keys) if exclude_keys is None else exclude_keys
        include_keys = tuple(self.include_keys) + ("_output_dir",) if include_keys is None else include_keys
        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": {}, "pickles": {}}
        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    payload["state_dicts"][key] = _copy_to_cpu(value.state_dict()) if use_thread else value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()
        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs):
        path = pathlib.Path(path) if path else pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"
        payload = torch.load(path.open("rb"), pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload


def _instantiate(cfg, **kwargs):
    from omegaconf import OmegaConf
    c = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else dict(cfg)
    target = c.pop("_target_")
    mod, name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(mod), name)
    for k, v in list(c.items()):
        if isinstance(v, dict) and "_target_" in v:
            c[k] = _instantiate(v)
    return cls(**c, **kwargs)


def _get_class(path):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def _get_scheduler(name, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step + 1) / float(max(1, num_warmup_steps))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _dict_apply(x, fn):
    return {k: (_dict_apply(v, fn) if isinstance(v, dict) else fn(v)) for k, v in x.items()}


def _optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


_lock = threading.Lock()


def install() -> bool:
    """Install the stubs unless the real packages import; returns True when stubs are active."""
    with _lock:
        try:
            import diffusion_policy  # noqa: F401
            import hydra  # noqa: F401
            return False
        except ImportError:
            pass
        def mod(name, **attrs):
            m = types.ModuleType(name); m.__dict__.update(attrs); m.__path__ = []; sys.modules[name] = m; return m
        if "hydra" not in sys.modules:
            hydra = mod("hydra"); utils = mod("hydra.utils", instantiate=_instantiate, get_class=_get_class); hydra.utils = utils
        mod("diffusion_policy"); mod("diffusion_policy.common"); mod("diffusion_policy.model"); mod("diffusion_policy.model.common")
        mod("diffusion_policy.workspace")
        mod("diffusion_policy.common.pytorch_util", dict_apply=_dict_apply, optimizer_to=_optimizer_to)
        mod("diffusion_policy.model.common.lr_scheduler", get_scheduler=_get_scheduler)
        mod("diffusion_policy.model.common.normalizer", LinearNormalizer=LinearNormalizer)
        mod("diffusion_policy.workspace.base_workspace", BaseWorkspace=BaseWorkspace)
        return True
