"""Self-contained few-step samplers for Diffusion Policy checkpoints, shared by the eps and the x0 (``sample``) heads.

Why not ``diffusers.DDIMScheduler`` (0.11.1, the version pinned by the official repo): its ``set_timesteps`` builds the
*leading* grid (k=1 -> ``[0]``, k=2 -> ``[50, 0]``) so the first (or only) step feeds pure noise to a t=0 network, and its
``step`` uses the raw ``model_output`` in the direction term for ``prediction_type="sample"``, i.e. treats x0 as eps.
Both errors are invisible in the returned scores. This module fixes the grid (*trailing*: ``t_i = T-1 - i*T/k``, the
last step maps straight to x0) and writes the DDIM update once for both heads:

    x0  = (x_t - sqrt(1-a_t) * eps) / sqrt(a_t)       (eps head)   |   x0 = network output      (x0 head)
    x0  = clip(x0, -1, 1)                               (clip_sample, the DP default; actions are normalised to [-1, 1])
    eps = (x_t - sqrt(a_t) * x0) / sqrt(1-a_t)          (recomputed from the clipped x0 for BOTH heads)
    x_prev = sqrt(a_prev) * x0 + sqrt(1-a_prev) * eps   (eta = 0; a_prev = 1 after the last step -> x_prev = x0)

DDPM-100 (the auxiliary full-step anchor) re-implements the ``fixed_small`` posterior step of diffusers 0.11.1
``DDPMScheduler.step`` (whose ``sample`` branch is correct) so that the per-step noise can be keyed per environment row
and timestep; parity with the upstream step is covered by the tests.

``install(policy, sampler, k)`` swaps the policy's ``conditional_sample`` for one that runs this loop while keeping the
official ``predict_action`` (normaliser, conditioning, inpainting mask, action slicing) untouched. Works with both
``DiffusionUnetLowdimPolicy`` and ``DiffusionUnetHybridImagePolicy`` (same ``model(trajectory, t, local_cond, global_cond)``
call and the same ``conditional_sample`` signature). Python 3.9 / torch 1.12 compatible (the official DP env).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

ALLOWED_K = (1, 2, 4, 10, 100)
HEADS = ("epsilon", "sample")


def make_timesteps(T: int, k: int) -> List[int]:
    """Trailing grid ``[T-1, T-1-T/k, ..., T-1-(k-1)T/k]``; k must divide T and be one of ``ALLOWED_K`` for T=100."""
    if T <= 0 or k <= 0 or T % k != 0:
        raise ValueError(f"k={k} must divide T={T}")
    if T == 100 and k not in ALLOWED_K:
        raise ValueError(f"k={k} not in the pre-registered set {ALLOWED_K}")
    step = T // k
    return [T - 1 - i * step for i in range(k)]


EPS_MODES = ("recompute", "raw")


def to_x0_eps(x_t: torch.Tensor, pred: torch.Tensor, alpha_bar_t: float, head: str,
              clip: bool = True, eps_mode: str = "recompute") -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a network output into the (x0, eps) pair used by the update.

    ``eps_mode="recompute"`` (the plan's shared convention, = diffusers ``use_clipped_model_output=True``): eps is
    recomputed from the clipped x0 for both heads. ``eps_mode="raw"`` (diffusers' default for the eps head): the
    network's own eps is kept for the direction term while x0 is still clipped; for the x0 head "raw" is identical to
    "recompute" when clipping does not bind. The mode is recorded in the manifest.
    """
    if head not in HEADS:
        raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
    if eps_mode not in EPS_MODES:
        raise ValueError(f"eps_mode must be one of {EPS_MODES}, got {eps_mode!r}")
    a = float(alpha_bar_t)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha_bar_t must be in (0, 1), got {a}")
    sa, sb = math.sqrt(a), math.sqrt(1.0 - a)
    x0 = (x_t - sb * pred) / sa if head == "epsilon" else pred
    if clip:
        x0 = x0.clamp(-1.0, 1.0)
    if eps_mode == "raw" and head == "epsilon":
        eps = pred
    else:
        eps = (x_t - sa * x0) / sb
    return x0, eps


def ddim_step(x0: torch.Tensor, eps: torch.Tensor, alpha_bar_prev: float) -> torch.Tensor:
    """Deterministic (eta=0) DDIM update to the previous grid point; ``alpha_bar_prev=1`` returns x0 itself."""
    a = float(alpha_bar_prev)
    if not 0.0 < a <= 1.0:
        raise ValueError(f"alpha_bar_prev must be in (0, 1], got {a}")
    return math.sqrt(a) * x0 + math.sqrt(1.0 - a) * eps


def sample_ddim(model_fn: Callable[[torch.Tensor, int], torch.Tensor], x_T: torch.Tensor, timesteps: Sequence[int],
                alphas_cumprod: Sequence[float], head: str, clip: bool = True,
                inpaint: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                eps_mode: str = "recompute") -> Tuple[torch.Tensor, int]:
    """Run the trailing-grid DDIM loop. ``inpaint=(data, mask)`` re-imposes the conditioning before every network call
    and on the output, exactly as the DP policies do. Returns ``(x0_final, nfe)``."""
    x = x_T
    nfe = 0
    n = len(timesteps)
    for i, t in enumerate(timesteps):
        if inpaint is not None:
            data, mask = inpaint
            x = torch.where(mask, data, x)
        pred = model_fn(x, int(t))
        nfe += 1
        x0, eps = to_x0_eps(x, pred, float(alphas_cumprod[int(t)]), head, clip=clip, eps_mode=eps_mode)
        a_prev = float(alphas_cumprod[int(timesteps[i + 1])]) if i + 1 < n else 1.0
        x = ddim_step(x0, eps, a_prev)
    if inpaint is not None:
        data, mask = inpaint
        x = torch.where(mask, data, x)
    return x, nfe


def ddpm_step(x_t: torch.Tensor, pred: torch.Tensor, t: int, betas: Sequence[float], alphas_cumprod: Sequence[float],
              head: str, clip: bool, noise: Optional[torch.Tensor]) -> torch.Tensor:
    """One DDPM posterior step, the ``fixed_small`` variance of diffusers 0.11.1 ``DDPMScheduler.step`` written out so the
    per-step noise can be supplied per environment row (``noise`` is ignored at t=0)::

        x0     = clip(x0(pred))                                   (eps head: (x_t - sqrt(1-a_t) eps)/sqrt(a_t); x0 head: pred)
        mean   = sqrt(a_prev) b_t/(1-a_t) x0 + sqrt(1-b_t) (1-a_prev)/(1-a_t) x_t
        var    = max((1-a_prev)/(1-a_t) b_t, 1e-20);   x_prev = mean + sqrt(var) noise   (t > 0)
    """
    t = int(t)
    a_t = float(alphas_cumprod[t]); a_prev = float(alphas_cumprod[t - 1]) if t > 0 else 1.0; b_t = float(betas[t])
    x0, _ = to_x0_eps(x_t, pred, a_t, head, clip=clip, eps_mode="raw")
    coeff_x0 = math.sqrt(a_prev) * b_t / (1.0 - a_t)
    coeff_xt = math.sqrt(1.0 - b_t) * (1.0 - a_prev) / (1.0 - a_t)
    mean = coeff_x0 * x0 + coeff_xt * x_t
    if t == 0:
        return mean
    if noise is None:
        raise ValueError("ddpm_step needs noise for t > 0")
    var = max((1.0 - a_prev) / (1.0 - a_t) * b_t, 1e-20)
    return mean + math.sqrt(var) * noise


def sample_ddpm(model_fn: Callable[[torch.Tensor, int], torch.Tensor], x_T: torch.Tensor, betas: Sequence[float],
                alphas_cumprod: Sequence[float], head: str, clip: bool = True,
                inpaint: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                step_noise_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None) -> Tuple[torch.Tensor, int]:
    """Full-step DDPM (T network calls, t = T-1 .. 0). ``step_noise_fn(t, like)`` supplies the posterior noise (default
    ``torch.randn_like``); inpainting as in :func:`sample_ddim`. Returns ``(x0_final, nfe)``."""
    T = len(alphas_cumprod)
    x = x_T
    nfe = 0
    for t in range(T - 1, -1, -1):
        if inpaint is not None:
            data, mask = inpaint
            x = torch.where(mask, data, x)
        pred = model_fn(x, t)
        nfe += 1
        noise = None
        if t > 0:
            noise = step_noise_fn(t, x) if step_noise_fn is not None else torch.randn_like(x)
        x = ddpm_step(x, pred, t, betas, alphas_cumprod, head, clip, noise)
    if inpaint is not None:
        data, mask = inpaint
        x = torch.where(mask, data, x)
    return x, nfe


class TrailingSampler:
    """Replacement for ``policy.conditional_sample``.

    ``sampler='ddim'`` runs :func:`sample_ddim` on the trailing grid with ``k`` network calls; ``sampler='ddpm'`` runs
    :func:`sample_ddpm` for all T steps (the auxiliary anchor; ``k`` must equal T). The scheduler's beta schedule,
    ``prediction_type`` (head) and ``clip_sample`` are read from ``policy.noise_scheduler``. Keeps running ``nfe`` /
    ``calls`` counters, exposes the exact grid via :meth:`describe` and accepts keyed-noise hooks (``noise_fn`` for the
    initial noise, ``step_noise_fn`` for the DDPM posterior noise). Raises ``ValueError`` on an unknown sampler, an
    unregistered k, a k != T for DDPM, an unsupported variance type or an unknown eps_mode."""

    def __init__(self, policy, sampler: str, k: int, eps_mode: str = "recompute"):
        if sampler not in ("ddim", "ddpm"):
            raise ValueError(f"sampler must be 'ddim' or 'ddpm', got {sampler!r}")
        if eps_mode not in EPS_MODES:
            raise ValueError(f"eps_mode must be one of {EPS_MODES}")
        self.eps_mode = eps_mode
        sched = policy.noise_scheduler
        self.policy = policy
        self.sampler = sampler
        self.T = int(sched.config.num_train_timesteps)
        self.head = str(sched.config.prediction_type)
        self.clip = bool(sched.config.clip_sample)
        self.alphas_cumprod = [float(a) for a in sched.alphas_cumprod.tolist()]
        self.betas = [float(b) for b in sched.betas.tolist()]
        vt = str(getattr(sched.config, "variance_type", "fixed_small"))
        if sampler == "ddpm" and vt != "fixed_small":
            raise ValueError(f"only variance_type=fixed_small is implemented, scheduler has {vt!r}")
        if sampler == "ddpm":
            if k != self.T:
                raise ValueError(f"the DDPM anchor runs all T={self.T} steps; got k={k}")
            self.timesteps = list(range(self.T - 1, -1, -1))
        else:
            self.timesteps = make_timesteps(self.T, k)
        self.k = k
        self.nfe = 0
        self.calls = 0
        self._orig = policy.conditional_sample
        # hooks for keyed randomness (see eval_dp_steps_v2.KeyedNoise): initial noise per call, DDPM posterior noise per
        # (timestep, row) -- both default to torch.randn with the caller's generator
        self.noise_fn: Optional[Callable] = None       # (shape, dtype, device, generator) -> tensor
        self.step_noise_fn: Optional[Callable] = None  # (t, like_tensor) -> tensor

    def describe(self) -> Dict[str, object]:
        """Exact protocol of this sampler: sampler, k, T, head, clip, eps_mode, timesteps, alpha_bar / SNR at the first step, NFE per call."""
        return {"sampler": self.sampler, "k": self.k, "T": self.T, "head": self.head, "clip_sample": self.clip,
                "eps_mode": self.eps_mode,
                "timesteps": list(self.timesteps), "alpha_bar_first": self.alphas_cumprod[self.timesteps[0]],
                "snr_first": self.alphas_cumprod[self.timesteps[0]] / (1.0 - self.alphas_cumprod[self.timesteps[0]]),
                "nfe_per_call": len(self.timesteps)}

    def __call__(self, condition_data, condition_mask, local_cond=None, global_cond=None, generator=None, **kwargs):
        policy = self.policy
        model = policy.model
        if self.noise_fn is not None:
            x = self.noise_fn(tuple(condition_data.shape), condition_data.dtype, condition_data.device, generator)
        else:
            x = torch.randn(size=condition_data.shape, dtype=condition_data.dtype, device=condition_data.device,
                            generator=generator)
        self.calls += 1
        if self.sampler == "ddim":
            def model_fn(traj, t):
                return model(traj, t, local_cond=local_cond, global_cond=global_cond)
            out, nfe = sample_ddim(model_fn, x, self.timesteps, self.alphas_cumprod, self.head, clip=self.clip,
                                   inpaint=(condition_data, condition_mask), eps_mode=self.eps_mode)
            self.nfe += nfe
            return out
        def model_fn(traj, t):
            return model(traj, t, local_cond=local_cond, global_cond=global_cond)
        step_noise = self.step_noise_fn
        if step_noise is None:
            def step_noise(t, like):
                return torch.randn(like.shape, dtype=like.dtype, device=like.device, generator=generator)
        out, nfe = sample_ddpm(model_fn, x, self.betas, self.alphas_cumprod, self.head, clip=self.clip,
                               inpaint=(condition_data, condition_mask), step_noise_fn=step_noise)
        self.nfe += nfe
        return out

    def uninstall(self) -> None:
        """Restore the policy's original ``conditional_sample``."""
        self.policy.conditional_sample = self._orig


def install(policy, sampler: str, k: int, eps_mode: str = "recompute") -> TrailingSampler:
    """Swap ``policy.conditional_sample`` for a :class:`TrailingSampler` (``policy.predict_action`` and everything around
    the denoising loop stay official); returns the sampler, whose ``.uninstall()`` restores the original method."""
    ts = TrailingSampler(policy, sampler, k, eps_mode=eps_mode)
    policy.conditional_sample = ts
    return ts
