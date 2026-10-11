"""Semantic request randomness independent of concurrency and retries."""
from __future__ import annotations

from .contracts import RequestContext, digest


class RequestRNG:
    def __init__(self, request: RequestContext):
        self.request = request
        self.identity = digest(request.noise_identity)

    def seed_for(self, stream: str = "initial", index: int = 0) -> int:
        if index < 0:
            raise ValueError("noise index must be nonnegative")
        if stream == "initial" and index == 0 and self.request.explicit_seed is not None:
            return self.request.explicit_seed
        return int(digest((self.identity, stream, index))[:16], 16) & ((1 << 63) - 1)

    def generator(self, stream: str = "initial", index: int = 0):
        import numpy as np
        return np.random.default_rng(self.seed_for(stream, index))

    def normal(self, shape, *, stream: str = "initial", index: int = 0, dtype="float32"):
        return self.generator(stream, index).standard_normal(shape).astype(dtype)

    def torch_generator(self, *, stream: str = "initial", index: int = 0, device="cpu"):
        import torch
        return torch.Generator(device=device).manual_seed(self.seed_for(stream, index))

    def posterior_noise(self, shape, loop_index: int):
        return self.normal(shape, stream="posterior", index=loop_index)

    def receipt(self):
        return {"schema": "r17.request_rng.v1", "identity": self.identity,
                "initial_seed": self.seed_for(), "posterior_rule": "semantic_seed_per_loop_index"}
