"""CONTRACT.md adapter; 100 DDPM steps, one observation, physical actions."""
from pathlib import Path
import hashlib
import numpy as np
from safetensors.torch import load_file
import torch

from .checkpoint import read_metadata, read_stats
from .model import DiffusionPolicy
from .preprocess import LanguageEncoder, image_tensor, quantile_normalize, quantile_unnormalize


def memory_cap(device, gib=8.0):
    device = torch.device(device)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(gib * 2**30 / total, device)


class Policy:
    def __init__(self, checkpoint, device, *, domain="metaworld", denoising_steps=100,
                 text_encoder="/data/mw_finetune/dp/clip_text", language_kind="projected",
                 feature_order=None, second_camera="zero", memory_limit_gib=8.0,
                 cuda_graph=False):
        self.checkpoint = str(Path(checkpoint).resolve())
        with (Path(checkpoint) / "model.safetensors").open("rb") as stream:
            self.weight_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.device = torch.device(device)
        self.cuda_graph = cuda_graph
        if cuda_graph and self.device.type != "cuda":
            raise ValueError("cuda_graph=True requires a CUDA device")
        memory_cap(device, memory_limit_gib)
        self.metadata = read_metadata(checkpoint)
        cfg = self.metadata["config"]["model"]
        expected = 7 if domain == "libero" else 4
        if domain not in ("libero", "metaworld") or cfg["action_dim"] != expected:
            raise ValueError("checkpoint action dimension disagrees with requested domain")
        options = self.metadata.get("wrapper", {})
        self.order = tuple(feature_order or options.get("feature_order", ["base_0_rgb", "left_wrist_0_rgb", "state", "lang_emb"]))
        self.language_kind = options.get("language_kind", language_kind)
        self.second_camera = options.get("second_camera", second_camera)
        if self.second_camera not in ("zero", "duplicate"):
            raise ValueError("second_camera must be zero or duplicate")
        if not 1 <= denoising_steps <= 100:
            raise ValueError("DDPM step count must be 1..100")
        self.steps, self.domain = denoising_steps, domain
        self.model = DiffusionPolicy(cfg, self.order)
        result = self.model.load_state_dict(load_file(str(Path(checkpoint) / "model.safetensors")), strict=True)
        self.strict_load = {"missing": result.missing_keys, "unexpected": result.unexpected_keys,
                            "keys": len(self.model.state_dict())}
        self.model.to(self.device).eval()
        self.stats = read_stats(checkpoint)
        self.language = LanguageEncoder(text_encoder, self.language_kind)
        if cuda_graph:
            from .cuda_graph import DenoiserGraph, check_observations, require_bit_exact
            self._denoiser_graph = DenoiserGraph(self.model._policy.model)
            # Finish capture and verification before the server binds its port.
            # All draws use fresh local generators, including the self-check.
            self.cuda_graph_self_check = require_bit_exact(
                check_observations(self.domain, self.stats),
                lambda obs: self._infer(obs, cuda_graph=False),
                lambda obs: self._infer(obs, cuda_graph=True))

    def reset(self):
        pass

    @torch.inference_mode()
    def infer(self, obs):
        return self._infer(obs, cuda_graph=self.cuda_graph)

    @torch.inference_mode()
    def _infer(self, obs, *, cuda_graph):
        if self.domain == "libero":
            image, state = obs["observation/image"], obs["observation/state"]
            second = image_tensor(obs["observation/wrist_image"])
        else:
            image, state = obs["image"], obs["state"]
            second = image_tensor(image) if self.second_camera == "duplicate" else torch.zeros(3, 96, 96)
        state = np.asarray(state, np.float32)
        if state.shape != (self.model.cfg["lowdims"][0]["dim"],) or not np.isfinite(state).all():
            raise ValueError("Invalid state")
        inputs = {"base_0_rgb": image_tensor(image).unsqueeze(0), "left_wrist_0_rgb": second.unsqueeze(0),
                  "state": torch.from_numpy(quantile_normalize(state, self.stats["state"])).unsqueeze(0),
                  "lang_emb": self.language([obs["prompt"]])}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Legacy LIBERO wire has no seed. A fixed fresh generator also makes
        # those calls independent of any advancing global/cross-client RNG.
        generator = torch.Generator(device=self.device).manual_seed(int(obs.get("seed", 0)))
        if cuda_graph:
            action = self._denoiser_graph.sample(self.model, inputs, generator=generator, steps=self.steps)[0].float().cpu().numpy()
        else:
            action = self.model.sample(inputs, generator=generator, steps=self.steps)[0].float().cpu().numpy()
        action = quantile_unnormalize(action, self.stats["actions"]).astype(np.float32)
        if not np.isfinite(action).all():
            raise ValueError("Nonfinite DP action")
        return {"actions": action}

    def get_metadata(self):
        metadata = {"architecture": "dp-transformer-film-resnet18", "n_obs_steps": 1, "horizon": 16,
                "denoising_steps": self.steps, "scheduler": "DDPM/squaredcos_cap_v2/epsilon",
                "domain": self.domain, "language_kind": self.language_kind, "feature_order": self.order,
                "second_camera": "wire wrist" if self.domain == "libero" else self.second_camera,
                "external_normalization": "quantile", "internal_normalization": "checkpoint affine",
                "strict_load": self.strict_load, "policy_rng": "fresh private generator from obs.seed; missing seed defaults to0",
                "weight_sha256": self.weight_sha256,
                "architecture_sha256": hashlib.sha256(Path(__file__).with_name("model.py").read_bytes()).hexdigest(),
                "reconstruction": "checkpoint-exact tensors; wrapper conventions reconstructed (see RECIPE)"}
        # Preserve the exact legacy metadata when the default-off path is used.
        if self.cuda_graph:
            metadata.update(cuda_graph=True, cuda_graph_self_check=self.cuda_graph_self_check)
        return metadata


def load_policy(checkpoint, device, **kw):
    return Policy(checkpoint, device, **kw)
