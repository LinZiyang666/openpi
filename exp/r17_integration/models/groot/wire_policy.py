"""Reuse B's native MW wire conversion around the stock GR00T interceptor."""
from __future__ import annotations

import numpy as np

from exp.mw_finetune.groot.conventions import from_groot_actions, to_groot_observation as _to_groot_observation


def observation_field(obs, name):
    """Accept both B's canonical keys and the actual MetaWorld worker wire."""
    alias = "observation/" + name
    if name in obs:
        if alias in obs and not np.array_equal(np.asarray(obs[name]), np.asarray(obs[alias])):
            raise ValueError(f"conflicting {name} and {alias} observations")
        return obs[name]
    if alias in obs:
        return obs[alias]
    raise KeyError(f"missing {name} or {alias}")


def to_groot_observation(obs):
    # Only translate field names. B still owns image/state validation and native transforms.
    return _to_groot_observation({"image": observation_field(obs, "image"),
                                 "state": observation_field(obs, "state"), "prompt": obs["prompt"]})


class GrootMetaWorldWirePolicy:
    def __init__(self, inner, manifest=None, *, lock=None):
        self.inner = inner
        self.manifest = manifest
        self.lock = lock

    def infer(self, obs):
        if self.manifest is not None:
            runner = getattr(self.inner, "_runner", None)
            model = runner._model if runner is not None else getattr(self.inner, "model", None)
            if model is not None and model.action_head.num_inference_timesteps != self.manifest.denoising_steps:
                raise RuntimeError("GR00T head runs a different K from the immutable endpoint")
        native = to_groot_observation(obs)
        if self.lock is None:
            result = self.inner.get_action(native)
        else:
            with self.lock:
                result = self.inner.get_action(native)
        reply = {"actions": from_groot_actions(result)}
        if "__hit_meta__" in result:
            reply["__hit_meta__"] = result["__hit_meta__"]
        return reply

    def get_metadata(self):
        return self.manifest.to_dict() if self.manifest is not None else self.inner.get_metadata()

    def __getattr__(self, name):
        return getattr(self.inner, name)
