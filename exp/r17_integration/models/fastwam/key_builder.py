"""Policy-owned fused pre-DiT keys; no external visual model or fake cameras."""
from __future__ import annotations

import numpy as np


class FastWAMKeyBuilder:
    def __init__(self, feature_spec):
        self.spec, self._fields = feature_spec, None

    def fields_from_tokens(self, video_pre, state):
        import torch
        tokens = video_pre["tokens"]
        grid = video_pre["meta"]["grid_sizes"][0].detach().cpu().tolist()
        f, h, w = map(int, grid)
        if tuple(grid) != self.spec.token_grid or f != 1 or tokens.shape != (1, f*h*w, self.spec.hidden_width):
            raise ValueError("native Fast-WAM token geometry differs from FeatureSpec")
        if int(video_pre["meta"]["tokens_per_frame"]) != h*w:
            raise ValueError("invalid native tokens_per_frame")
        spatial = tokens.float().reshape(1, h, w, -1).permute(0, 3, 1, 2)
        pooled = torch.nn.functional.adaptive_avg_pool2d(spatial, (4, 4))
        # Row-major bins, then hidden width, matching explicit metadata.
        vector = pooled.permute(0, 2, 3, 1).reshape(-1).detach().cpu().numpy().copy()
        rs = np.asarray(state, dtype=np.float32)[list(self.spec.active_state_indices)]
        fields = {"vision_0": np.ascontiguousarray(vector, dtype=np.float32),
                  "robot_state": np.ascontiguousarray(rs, dtype=np.float32)}
        if any(v.shape != (self.spec.fields[k],) or not np.isfinite(v).all() for k,v in fields.items()):
            raise ValueError("invalid Fast-WAM retrieval fields")
        return fields

    def collect(self, checkpoint, **kwargs):
        stage1 = kwargs.get("stage1")
        if stage1 is None:
            raise ValueError("Fast-WAM key collection requires actual stage1")
        self._fields = {**kwargs.get("retrieval_fields", stage1.fields), "state": stage1.state}

    @property
    def cached_data(self):
        return self._fields or {}

    def build(self, checkpoint_id=None):
        import torch
        if self._fields is None:
            raise ValueError("no Fast-WAM stage1 collected")
        return {k: torch.from_numpy(np.ascontiguousarray(self._fields[k], dtype=np.float32).copy()) for k in self.spec.fields}

    def clear(self):
        self._fields = None
