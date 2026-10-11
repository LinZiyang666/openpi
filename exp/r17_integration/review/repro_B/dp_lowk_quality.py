"""CPU-only: real LIBERO DP checkpoint, real training frames; compare K-step outputs
(E/adapter 'leading' grid vs 'trailing' grid) against the dataset action chunk and K100."""
import sys, json
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src"]
import numpy as np, torch
from safetensors.torch import load_file
from diffusers import DDPMScheduler
from exp.mw_finetune.dp.model import DiffusionPolicy
from exp.mw_finetune.dp.checkpoint import read_metadata
from exp.mw_finetune.dp.data import Dataset
torch.set_num_threads(4)
ck = "/data/mw_finetune/dp/checkpoints/libero"
cfg = read_metadata(ck)["config"]["model"]
m = DiffusionPolicy(cfg); r = m.load_state_dict(load_file(ck + "/model.safetensors"), strict=True); m.eval()
emb = load_file("/data/mw_finetune/dp/libero_96/language.safetensors")["embeddings"]
ds = Dataset("/data/mw_finetune/dp/libero_96", emb)
rng = np.random.default_rng(0); idx = rng.integers(len(ds), size=48)
items = [ds[int(i)] for i in idx]
obs = {k: torch.stack([it["obs"][k] for it in items]) for k in items[0]["obs"]}
gt_q = torch.stack([it["action"] for it in items])            # quantile-normalized dataset chunk
core = m._policy
def run(steps, spacing, seed=0):
    sch = DDPMScheduler(**{k: cfg[k] for k in ("num_train_timesteps","beta_schedule","beta_start","beta_end",
                        "variance_type","prediction_type","clip_sample","clip_sample_range")}, timestep_spacing=spacing)
    sch.set_timesteps(steps)
    g = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        cond = m.condition(obs)
        x = torch.randn((len(cond), 16, cfg["action_dim"]), generator=g)
        for t in sch.timesteps:
            x = sch.step(core.model(x, t, cond), t, x, generator=g).prev_sample
        clip_frac = float((x.abs() >= 0.999).float().mean())
        return core.normalizer.unnormalize("action", x), [int(t) for t in sch.timesteps], clip_frac
ref, _, _ = run(100, "leading")
out = {}
for spacing in ("leading", "trailing"):
    for k in (100, 10, 2, 1):
        if spacing == "trailing" and k == 100:
            continue
        a, grid, clip = run(k, spacing)
        out[f"{spacing}_K{k}"] = dict(grid=grid if k <= 10 else "99..0",
            mse_vs_dataset_first5=float(((a - gt_q)[:, :5] ** 2).mean()),
            mse_vs_K100_first5=float(((a - ref)[:, :5] ** 2).mean()),
            internal_clip_fraction=clip)
print(json.dumps({"strict": len(r.missing_keys) + len(r.unexpected_keys), "frames": len(items), **out}, indent=1))
