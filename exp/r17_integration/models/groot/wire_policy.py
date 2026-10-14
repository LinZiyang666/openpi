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


def to_groot_observation(obs, benchmark="metaworld"):
    if benchmark == "libero":
        from exp.libero_groot.policy_adapter import build_groot_observation
        return build_groot_observation(obs)
    # Only translate field names. B still owns image/state validation and native transforms.
    return _to_groot_observation({"image": observation_field(obs, "image"),
                                 "state": observation_field(obs, "state"), "prompt": obs["prompt"]})


def wire_actions(result, benchmark="metaworld"):
    if benchmark == "libero":
        from exp.libero_groot import libero_keys
        from exp.libero_groot.policy_adapter import chunk_to_libero_actions, validate_action_chunk
        # Vendor inverse retains B1; historical wire adapter consumes T x 1.
        actions = {key:(np.asarray(result[key])[0] if np.asarray(result[key]).ndim == 3 else np.asarray(result[key]))
                   for key in libero_keys.ACTION_KEYS}
        return chunk_to_libero_actions(validate_action_chunk(actions))
    return from_groot_actions(result)


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
        benchmark = self.manifest.benchmark if self.manifest is not None else "metaworld"
        native = to_groot_observation(obs, benchmark)
        inner = self.inner
        if isinstance(inner, dict):
            suite = obs["__request__"]["task_name"].split("/", 1)[0]
            inner = inner[suite]
        if self.lock is None:
            result = inner.get_action(native)
        else:
            with self.lock:
                result = inner.get_action(native)
        reply = {"actions": wire_actions(result, benchmark)}
        if "__hit_meta__" in result:
            reply["__hit_meta__"] = result["__hit_meta__"]
        return reply

    def get_metadata(self):
        return self.manifest.to_dict() if self.manifest is not None else self.inner.get_metadata()

    def __getattr__(self, name):
        return getattr(self.inner, name)
