"""Opt-in loader shim: official raw Wan weights, unchanged LeRobot MoT.

No modification to either upstream checkout; never downloads Wan components.
"""
from pathlib import Path


def install_raw_loaders(wan_root="/data/xwam/wan22_5b"):
    import torch
    from fastwam.models.wan22.wan_video_text_encoder import WanTextEncoder
    from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
    from fastwam.models.wan22.helpers.state_dict_converters import wan_video_vae_state_dict_converter
    import lerobot.policies.fastwam.modeling_fastwam as implementation

    root = Path(wan_root)
    for name in ("Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/tokenizer_config.json"):
        if not (root / name).exists():
            raise FileNotFoundError(root / name)

    class BatchedVAE(WanVideoVAE38):
        def encode(self, videos, device=None, tiled=False, **kwargs):
            if tiled:
                raise NotImplementedError("The MT50 loader uses fixed-size batched VAE encoding")
            if isinstance(videos, (list, tuple)):
                videos = torch.stack(videos)
            parameter = next(self.parameters())
            return self.model.encode(videos.to(device=parameter.device, dtype=parameter.dtype), self.scale)

    def load_vae(*, torch_dtype, device):
        # Real buffers (latent mean/std) are nonpersistent; instantiate on CPU.
        vae = BatchedVAE()
        state = torch.load(root / "Wan2.2_VAE.pth", map_location="cpu", weights_only=True, mmap=True)
        vae.load_state_dict(wan_video_vae_state_dict_converter(state), strict=True, assign=True)
        return vae.to(device=device, dtype=torch_dtype).eval().requires_grad_(False)

    def load_text(*, torch_dtype, device, **kwargs):
        with torch.device("meta"):
            encoder = WanTextEncoder()
        state = torch.load(root / "models_t5_umt5-xxl-enc-bf16.pth", map_location="cpu", weights_only=True, mmap=True)
        encoder.load_state_dict(state, strict=True, assign=True)
        return encoder.to(device=device, dtype=torch_dtype).eval().requires_grad_(False)

    implementation.load_pretrained_wan_vae = load_vae
    implementation.load_pretrained_wan_text_encoder = load_text
