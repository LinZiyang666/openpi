"""Separate published LIBERO processors and C's MetaWorld wire semantics."""
from __future__ import annotations

import numpy as np

from exp.r17_integration.common.contracts import PreparedObs, StateView, WireChunk


class FastWAMWirePolicy:
    def __init__(self, benchmark, stats, *, prompt_template="{task}", execution_protocol="system"):
        if benchmark not in {"libero", "metaworld"}:
            raise ValueError("unsupported Fast-WAM embodiment")
        self.benchmark, self.prompt_template = benchmark, prompt_template
        from .protocols import execution_protocol as get_protocol
        self.protocol = get_protocol(benchmark, execution_protocol)
        self.action_dim, self.state_dim = (7, 8) if benchmark == "libero" else (4, 4)
        self.stats = {}
        for name, dim in (("action", self.action_dim), ("observation.state", self.state_dim)):
            bounds = {k: np.asarray(stats[name][k], dtype=np.float32) for k in ("min", "max")}
            if any(v.shape != (dim,) or not np.isfinite(v).all() for v in bounds.values()):
                raise ValueError(f"invalid {name} normalization dimensions")
            if (bounds["max"] < bounds["min"]).any():
                raise ValueError("inverted normalization bounds")
            self.stats[name] = bounds

    def _denominator(self, name):
        d = self.stats[name]["max"] - self.stats[name]["min"]
        if self.benchmark == "libero":
            return np.where(d == 0, np.float32(1e-8), d)
        return np.maximum(d, np.float32(1e-6))

    def extract_state_only(self, obs):
        state = np.asarray(self._field(obs, "state", "observation/state"), dtype=np.float32)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError(f"expected finite state ({self.state_dim},)")
        return StateView(np.ascontiguousarray(state.copy()))

    @staticmethod
    def _field(obs, name, canonical):
        if canonical in obs:
            if name in obs and not np.array_equal(obs[name], obs[canonical]):
                raise ValueError(f"conflicting {name}/{canonical} observation aliases")
            return obs[canonical]
        if name not in obs:
            raise ValueError(f"missing required real observation {canonical}")
        return obs[name]

    def prepare(self, obs, request):
        import torch
        from PIL import Image
        state = self.extract_state_only(obs).state
        if self.protocol.fixed_seed is not None and request.explicit_seed != self.protocol.fixed_seed:
            raise ValueError("released LIBERO protocol requires explicit_seed=42 on every request")
        prompt = obs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("nonempty prompt required")
        names = ("image", "image2") if self.benchmark == "libero" else ("image",)
        images = []
        for name in names:
            canonical = "observation/image" if name == "image" else "observation/wrist_image"
            image = np.asarray(self._field(obs, name, canonical))
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("camera must be HWC RGB uint8")
            if self.benchmark == "metaworld" and image.shape != (480, 480, 3):
                raise ValueError("MetaWorld expects original 480x480 RGB frame")
            if self.benchmark == "libero" and image.shape != (256, 256, 3):
                raise ValueError("LIBERO expects original 256x256 RGB frame per camera; no client resize")
            if self.benchmark == "metaworld":
                # Match C exactly: PIL bilinear 480→224, then float conversion.
                image = np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR))
            pixels = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
            if self.benchmark == "libero" and pixels.shape[-2:] != (224, 224):
                # Same as vendor _resize_frames, per camera before width concatenation.
                pixels = torch.nn.functional.interpolate(pixels, size=(224, 224), mode="bilinear",
                    align_corners=False, antialias=True)
            images.append(pixels)
        normalized = 2 * (state - self.stats["observation.state"]["min"]) / self._denominator("observation.state") - 1
        return PreparedObs({"input_image": torch.cat(images, dim=-1),
            "proprio": torch.from_numpy(normalized.copy()).unsqueeze(0),
            "prompt": self.prompt_template.format(task=prompt)}, request, state.copy())

    def to_wire(self, normalized, prepared=None):
        if hasattr(normalized, "detach"):
            normalized = normalized.detach().float().cpu().numpy()
        x = np.asarray(normalized, dtype=np.float32)
        if x.shape != (32, self.action_dim) or not np.isfinite(x).all():
            raise ValueError("Fast-WAM normalized chunk shape or finiteness invalid")
        # Keep operation order identical to each original normalization path.
        if self.benchmark == "libero":
            out = (x + 1) / 2 * self._denominator("action") + self.stats["action"]["min"]
            out[..., -1] = np.sign(-(out[..., -1] * 2 - 1))
        else:
            out = (x + 1) * self._denominator("action") / 2 + self.stats["action"]["min"]
        if not np.isfinite(out).all():
            raise ValueError("Fast-WAM emitted nonfinite wire actions")
        return WireChunk(np.ascontiguousarray(out, dtype=np.float32))
