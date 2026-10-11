"""Validate both sets of statistics and decode the scheduler space once."""
import numpy as np
import torch

from .manifest import digest, tensor_digest


class Normalization:
    def __init__(self, model, stats):
        from exp.mw_finetune.dp.preprocess import quantile_normalize, quantile_unnormalize
        self._normalize, self._unnormalize = quantile_normalize, quantile_unnormalize
        self.core = model._policy.normalizer
        self.stats = stats
        dims = {"state": model.cfg["lowdims"][0]["dim"], "actions": model.cfg["action_dim"]}
        for name, dim in dims.items():
            for field in ("q01", "q99"):
                value = np.asarray(stats[name][field], np.float32)
                if value.shape != (dim,) or not np.isfinite(value).all():
                    raise ValueError(f"invalid external {name}.{field}")
            if np.any(np.asarray(stats[name]["q99"]) < np.asarray(stats[name]["q01"])):
                raise ValueError(f"reversed quantiles for {name}")
        internal = {}
        for name, values in self.core.params_dict.items():
            for field in ("scale", "offset"):
                value = values[field]
                if not torch.isfinite(value).all() or (field == "scale" and torch.any(value == 0)):
                    raise ValueError(f"invalid internal {name}.{field}")
            internal[name] = {k: tensor_digest(v) for k, v in values.items() if k != "input_stats"}
        self.identity = digest({"external_quantiles": stats, "internal_affine": internal})

    def state(self, state):
        return self._normalize(state, self.stats["state"])

    def to_wire(self, trajectory):
        action = self.core.unnormalize("action", trajectory).float().detach().cpu().numpy()
        action = self._unnormalize(action, self.stats["actions"]).astype(np.float32)
        if not np.isfinite(action).all():
            raise ValueError("nonfinite decoded DP actions")
        return np.ascontiguousarray(action)
