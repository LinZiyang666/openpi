"""Round-2 CPU check of the fixed DP reduced-K sampler on Review B's 48 real LIBERO frames.
K100 = E's untouched DDPM sampler (seed0, B48). K in {1,2,4,10}: the adapter's own SamplerProfile/step,
same seed0 B48 initial draw; cross-checked against diffusers DDIM(trailing) and DPAdapter.sample_batch."""
import sys, json
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src"]
import numpy as np, torch
from safetensors.torch import load_file
from diffusers import DDIMScheduler
from exp.mw_finetune.dp.model import DiffusionPolicy
from exp.mw_finetune.dp.checkpoint import read_metadata
from exp.mw_finetune.dp.data import Dataset
from exp.r17_integration.models.dp.sampler import SamplerProfile
from exp.r17_integration.models.dp.staged import DPAdapter
from exp.r17_integration.common.contracts import CaptureSpec, Stage2
from exp.r17_integration.models.dp.request_noise import RequestNoise
torch.set_num_threads(4)
ck = "/data/mw_finetune/dp/checkpoints/libero"
cfg = read_metadata(ck)["config"]["model"]
m = DiffusionPolicy(cfg); r = m.load_state_dict(load_file(ck + "/model.safetensors"), strict=True); m.eval()
emb = load_file("/data/mw_finetune/dp/libero_96/language.safetensors")["embeddings"]
ds = Dataset("/data/mw_finetune/dp/libero_96", emb)
idx = np.random.default_rng(0).integers(len(ds), size=48)
items = [ds[int(i)] for i in idx]
obs = {k: torch.stack([it["obs"][k] for it in items]) for k in items[0]["obs"]}
gt = torch.stack([it["action"] for it in items])
core = m._policy
with torch.inference_mode():
    cond = m.condition(obs)
    ref = m.sample(obs, generator=torch.Generator().manual_seed(0), steps=100)   # E DDPM100, internal-unnormalized
    x0 = torch.randn((48, 16, cfg["action_dim"]), generator=torch.Generator().manual_seed(0))
def loop(step_fn, timesteps):
    x = x0.clone()
    with torch.inference_mode():
        for t in timesteps:
            x = step_fn(core.model(x, t, cond), t, x)
    return x
stats = {"actions": {"q01": [-1.0]*7, "q99": [1.0]*7}, "state": {"q01": [-1.0]*8, "q99": [1.0]*8}}
adapter = DPAdapter(m, stats, lambda p: torch.zeros(1, 768), domain="libero", weight_identity="cpu-review")
out = {"strict_missing_unexpected": len(r.missing_keys) + len(r.unexpected_keys), "frames": 48}
for k in (1, 2, 4, 10):
    prof = SamplerProfile(num_steps=k)
    sch = prof.scheduler(cfg)
    xa = loop(lambda e, t, x: prof.step(sch, e, t, x, None), sch.timesteps)
    ref_sch = DDIMScheduler(**{n: cfg[n] for n in ("num_train_timesteps", "beta_schedule", "beta_start", "beta_end",
                              "prediction_type", "clip_sample", "clip_sample_range")}, timestep_spacing="trailing")
    ref_sch.set_timesteps(k)
    xd = loop(lambda e, t, x: ref_sch.step(e, t, x, eta=0.0).prev_sample, ref_sch.timesteps)
    with torch.inference_mode():
        stages = [Stage2({"conditioning": cond[i:i+1]}) for i in range(48)]
        res = adapter.sample_batch(stages, prof, [RequestNoise(0) for _ in range(48)], [CaptureSpec()] * 48,
                                   initial_noises=[x0[i:i+1] for i in range(48)])
    xs = torch.stack([q.normalized_chunk for q in res])
    a = core.normalizer.unnormalize("action", xa)
    out[f"K{k}"] = dict(profile_grid=[int(t) for t in sch.timesteps], diffusers_trailing_grid=[int(t) for t in ref_sch.timesteps],
        equal_to_diffusers_trailing=bool(torch.equal(xa, xd)), adapter_sample_batch_equal=bool(torch.equal(xa, xs)),
        quality_status=prof.descriptor(cfg)["quality_status"],
        mse_vs_K100_first5=float(((a - ref)[:, :5] ** 2).mean()), mse_vs_K100_all16=float(((a - ref) ** 2).mean()),
        mse_vs_dataset_first5=float(((a - gt)[:, :5] ** 2).mean()), internal_clip_fraction=float((xa.abs() >= .999).float().mean()))
out["K100_mse_vs_dataset_first5"] = float(((ref - gt)[:, :5] ** 2).mean())
print(json.dumps(out, indent=1))
