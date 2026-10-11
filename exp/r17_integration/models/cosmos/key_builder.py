"""Genuine policy-VAE camera features; no external encoder or invented view."""
import numpy as np


class CosmosKeyBuilder:
    def __init__(self, feature_spec):
        self.spec, self._fields = feature_spec, None

    def fields_from_latent(self, latent, state):
        import torch
        if tuple(latent.shape) != (1, 16, 9, 28, 28):
            raise ValueError("official Cosmos LIBERO latent geometry must be (1,16,9,28,28)")
        fields = {"vision_0": latent[:, :, 3], "vision_1": latent[:, :, 2]}
        result = {name: torch.nn.functional.adaptive_avg_pool2d(value.float(), (4, 4)).flatten()
                  .detach().cpu().numpy().copy() for name, value in fields.items()}
        result["robot_state"] = np.ascontiguousarray(state, dtype=np.float32)
        if any(value.shape != (self.spec.fields[name],) or not np.isfinite(value).all()
               for name, value in result.items()):
            raise ValueError("Cosmos native key geometry/finiteness mismatch")
        return result

    def collect(self, checkpoint, **kwargs):
        stage1 = kwargs.get("stage1")
        if stage1 is None:
            raise ValueError("Cosmos keys require the actual current VAE stage")
        self._fields = {**kwargs.get("retrieval_fields", stage1.fields), "state": stage1.state}

    @property
    def cached_data(self):
        return self._fields or {}

    def build(self, checkpoint_id=None):
        import torch
        if self._fields is None:
            raise ValueError("Cosmos stage1 was not collected")
        return {name: torch.from_numpy(self._fields[name].copy()) for name in self.spec.fields}

    def clear(self):
        self._fields = None
