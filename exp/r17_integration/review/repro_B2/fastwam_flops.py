"""FLOP counts on the meta device (no weights): raw Wan2.2 VAE38 encode of one first frame, and the
UMT5-XXL encoder at 128 tokens. Used only to fill the unknown V term of the B4 key/full split."""
import sys, json
sys.path[:0] = ["/data/mw_finetune/vendor/FastWAM/src", "/data/mw_finetune/vendor/lerobot/src"]
import torch
from torch.utils.flop_counter import FlopCounterMode
from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
out = {}
with torch.device("meta"):
    vae = WanVideoVAE38()
for name, (h, w) in {"libero_224x448": (224, 448), "mw_224x224": (224, 224)}.items():
    x = torch.empty(1, 3, 1, h, w, device="meta")
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        z = vae.model.encode(x, vae.scale if hasattr(vae, "scale") else [vae.mean, vae.inv_std])
    out["vae_" + name] = {"tflop": fc.get_total_flops() / 1e12, "latent_shape": list(z.shape)}
try:
    from fastwam.models.wan22.wan_video_text_encoder import WanTextEncoder
    with torch.device("meta"):
        enc = WanTextEncoder()
    ids = torch.zeros(1, 128, dtype=torch.long, device="meta"); mask = torch.ones(1, 128, dtype=torch.long, device="meta")
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        enc(ids, mask)
    out["umt5_128tok_tflop"] = fc.get_total_flops() / 1e12
    out["umt5_params_b"] = sum(p.numel() for p in enc.parameters()) / 1e9
except Exception as e:
    out["umt5_error"] = f"{type(e).__name__}: {e}"
print(json.dumps(out, indent=1))
