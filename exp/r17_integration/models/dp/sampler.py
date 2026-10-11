"""Native diffusers profiles. Each call constructs its own mutable scheduler."""
from dataclasses import dataclass

from .manifest import digest

_FIELDS = ("num_train_timesteps", "beta_schedule", "beta_start", "beta_end", "variance_type",
           "prediction_type", "clip_sample", "clip_sample_range")


@dataclass(frozen=True)
class SamplerProfile:
    kind: str = "auto"
    num_steps: int = 100
    eta: float = 0.0
    timestep_spacing: str | None = None

    def __post_init__(self):
        if not isinstance(self.num_steps, int) or isinstance(self.num_steps, bool) or not 1 <= self.num_steps <= 100:
            raise ValueError("DP K must be an integer in 1..100")
        if self.kind == "auto":
            object.__setattr__(self, "kind", "ddpm" if self.num_steps == 100 else "ddim")
        if self.kind not in ("ddpm", "ddim"):
            raise ValueError("DP sampler must be ddpm or explicit ddim")
        if self.kind == "ddpm" and self.num_steps != 100:
            raise ValueError("DDPM baseline requires K100; reduced K uses trailing DDIM")
        if not 0 <= self.eta <= 1 or (self.kind == "ddpm" and self.eta != 0):
            raise ValueError("eta is only applicable to explicit DDIM, range 0..1")
        spacing = "leading" if self.kind == "ddpm" else "trailing"
        if self.timestep_spacing is not None and self.timestep_spacing != spacing:
            raise ValueError(f"{self.kind} requires {spacing} timestep spacing")
        object.__setattr__(self, "timestep_spacing", spacing)

    def scheduler(self, cfg):
        from diffusers import DDPMScheduler, DDIMScheduler
        options = {k: cfg[k] for k in _FIELDS}
        if options["prediction_type"] != "epsilon" or options["variance_type"] != "fixed_small":
            raise ValueError("only audited epsilon/fixed_small DP scheduler is supported")
        if options["num_train_timesteps"] != 100:
            raise ValueError("audited DP training grid is 100 points")
        if self.kind == "ddim":
            options.pop("variance_type")
        options["timestep_spacing"] = self.timestep_spacing
        scheduler = (DDPMScheduler if self.kind == "ddpm" else DDIMScheduler)(**options)
        scheduler.set_timesteps(self.num_steps)
        if self.kind == "ddim":
            import numpy as np
            import torch
            # diffusers' floating arange(T,0,-T/K) can produce K+1 points
            # (e.g.K29). Construct exactly K trailing points by index.
            points = np.rint(np.arange(self.num_steps, 0, -1) * (100 / self.num_steps)).astype(np.int64) - 1
            scheduler.timesteps = torch.from_numpy(points)
        return scheduler

    def descriptor(self, cfg):
        import diffusers
        scheduler = self.scheduler(cfg)
        config = dict(scheduler.config)
        config.pop("_use_default_values", None)
        native = tuple(int(t) for t in scheduler.timesteps)
        descriptor = {"schema": "r17.dp.sampler.v2", "kind": self.kind, "num_steps": self.num_steps,
                      "eta": self.eta, "config": config, "native_grid": native,
                      "timestep_spacing": self.timestep_spacing,
                      "grid_rule": "round((K-i)*T/K)-1_exact_K" if self.kind == "ddim" else "upstream_full100",
                      "update_rule": "native_grid_predecessor_ddim_v1" if self.kind == "ddim" else "upstream_ddpm_v1",
                      "quality_status": "lossy" if self.num_steps == 1 else
                                        ("baseline" if self.kind == "ddpm" else "unqualified_reduced_steps"),
                      "quality_note": "K1 is a lossy research profile; no descent/quality qualification" if self.num_steps == 1 else
                                      "CPU math conformance is not a benchmark quality gate",
                      "point_kind": "native_integer_timestep", "diffusers_version": diffusers.__version__,
                      "betas": [float(b) for b in scheduler.betas]}
        return {**descriptor, "identity": digest(descriptor)}

    def step(self, scheduler, prediction, t, x, generator):
        if self.kind == "ddpm":
            return scheduler.step(prediction, t, x, generator=generator).prev_sample
        # DDIMScheduler.step uses t - T//K, which is NOT the next trailing
        # grid point when K does not divide T. Use the actual predecessor in
        # equations12/16 (same arithmetic as diffusers for divisor K).
        from diffusers.utils.torch_utils import randn_tensor
        native = tuple(int(point) for point in scheduler.timesteps)
        index = native.index(int(t))
        previous = native[index+1] if index+1 < len(native) else -1
        alpha = scheduler.alphas_cumprod[t]
        alpha_previous = scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.final_alpha_cumprod
        clean = (x - (1-alpha)**0.5 * prediction) / alpha**0.5
        if scheduler.config.clip_sample:
            clean = clean.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)
        std = self.eta * scheduler._get_variance(t, previous)**0.5
        result = alpha_previous**0.5 * clean + (1-alpha_previous-std**2)**0.5 * prediction
        if self.eta > 0:
            result = result + std * randn_tensor(prediction.shape, generator=generator,
                                                 device=prediction.device, dtype=prediction.dtype)
        return result

    def to_spec(self, cfg):
        from exp.r17_integration.common.contracts import SamplerSpec
        desc = self.descriptor(cfg)
        return SamplerSpec(sampler_id=f"r17_dp_{self.kind}_{self.timestep_spacing}_k{self.num_steps}_v2",
                           family=self.kind, num_steps=self.num_steps,
                           native_timesteps=desc["native_grid"],
                           config={k: v for k, v in desc.items() if k != "native_grid"},
                           point_kind="native_integer_timestep")


def checked_profile(spec, cfg):
    if isinstance(spec, SamplerProfile):
        return spec, spec.to_spec(cfg)
    profile = SamplerProfile(spec.family, spec.num_steps, spec.config.get("eta", 0.0),
                             spec.config.get("timestep_spacing"))
    if profile.to_spec(cfg).identity != spec.identity:
        pass
    return profile, spec
