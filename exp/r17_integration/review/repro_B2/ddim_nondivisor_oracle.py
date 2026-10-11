"""Exact-MMSE oracle on the fixed adapter DDIM step for divisor and nondivisor K."""
import sys
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src"]
import torch
from exp.r17_integration.models.dp.sampler import SamplerProfile
cfg = dict(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", beta_start=1e-4, beta_end=0.02,
           variance_type="fixed_small", prediction_type="epsilon", clip_sample=True, clip_sample_range=1.0)
mu, s = 0.4, 0.2
for k in (1, 2, 3, 7, 10, 13, 29, 99, 100):
    prof = SamplerProfile(num_steps=k); sch = prof.scheduler(cfg); ab = sch.alphas_cumprod
    g = torch.Generator().manual_seed(0); x = torch.randn(20000, 16, 4, generator=g)
    for t in sch.timesteps:
        a = ab[int(t)]; eps = (1 - a).sqrt() * (x - a.sqrt() * mu) / (a * s * s + 1 - a)
        x = prof.step(sch, eps, t, x, g)
    grid = [int(t) for t in sch.timesteps]
    print(f"K={k:3d} kind={prof.kind} n_points={len(grid)} first={grid[0]} last={grid[-1]} -> mean {float(x.mean()):+.3f} std {float(x.std()):.3f}")
