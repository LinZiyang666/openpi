"""Oracle-epsilon test of DP reduced-K grids (diffusers, E's audited scheduler cfg).
Data x0 ~ N(mu, s^2) per channel; the exact MMSE eps-predictor for that data is used,
so any error is caused by the schedule, not by the network."""
import torch
from diffusers import DDPMScheduler, DDIMScheduler
cfg = dict(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", beta_start=1e-4, beta_end=0.02,
           prediction_type="epsilon", clip_sample=True, clip_sample_range=1.0)
mu, s = 0.4, 0.2
def oracle(sch):
    ab = sch.alphas_cumprod
    def eps(x, t):
        a = ab[int(t)]
        return (1 - a).sqrt() * (x - a.sqrt() * mu) / (a * s * s + 1 - a)
    return eps
def run(kind, k, spacing):
    extra = {} if kind == "ddim" else {"variance_type": "fixed_small"}
    sch = (DDPMScheduler if kind == "ddpm" else DDIMScheduler)(**cfg, **extra, timestep_spacing=spacing)
    sch.set_timesteps(k)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(20000, 16, 4, generator=g)
    e = oracle(sch)
    for t in sch.timesteps:
        x = sch.step(e(x, t), t, x, generator=g).prev_sample
    return [int(t) for t in sch.timesteps], float(x.mean()), float(x.std())
ab = DDPMScheduler(**cfg, variance_type="fixed_small").alphas_cumprod
print("alpha_bar[t] t=0,50,99:", [round(float(ab[t]), 5) for t in (0, 50, 99)])
print("target data: mean %.3f std %.3f" % (mu, s))
for kind in ("ddpm", "ddim"):
    for spacing in ("leading", "trailing"):
        for k in (100, 10, 2, 1):
            if kind == "ddim" and k == 100 and spacing == "trailing":
                continue
            grid, m, sd = run(kind, k, spacing)
            print(f"{kind:4s} {spacing:8s} K={k:3d} grid={grid if k<=10 else str(grid[:3])+'...'} -> mean {m:+.3f} std {sd:.3f}")
