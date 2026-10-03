"""Numeric pre-decision snapshots; adapted from P3 snapshots/rng_snapshot."""
import json
import random

import numpy as np

from .adapter import unwrap


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"skipped_type": type(value).__name__}


def capture(env, selection_p=1.):
    inner = unwrap(env)
    data = inner.sim.data
    result = {key: np.array(getattr(data, key), copy=True) for key in
              ("time", "qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "userdata")
              if getattr(data, key, None) is not None}
    if callable(getattr(env, "get_sim_state", None)):
        result["sim_state"] = np.array(env.get_sim_state(), copy=True)
    result["env_timestep"] = np.asarray(getattr(inner, "timestep", -1))
    result["env_cur_time"] = np.asarray(getattr(inner, "cur_time", np.nan))
    skipped = []
    for i, robot in enumerate(getattr(inner, "robots", [])):
        controller = getattr(robot, "controller", None)
        for name, value in vars(controller).items() if hasattr(controller, "__dict__") else []:
            key = "controller_{}_{}".format(i, name)
            if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
                result[key] = value.copy()
            elif isinstance(value, (int, float, bool, np.number)):
                result[key] = np.asarray(value)
            else:
                skipped.append(key + ":" + type(value).__name__)
    rng = dict(python_global=clean(random.getstate()), numpy_global=clean(np.random.get_state()))
    for owner, obj in (("outer", env), ("inner", inner)):
        for attr in ("np_random", "rng", "_rng"):
            value = getattr(obj, attr, None)
            if hasattr(value, "bit_generator"):
                rng[owner + "." + attr] = clean(value.bit_generator.state)
            elif hasattr(value, "get_state"):
                rng[owner + "." + attr] = clean(value.get_state())
    result.update(rng_json=np.asarray(json.dumps(rng, sort_keys=True)),
                  controller_skipped_json=np.asarray(json.dumps(skipped)),
                  restore_certified=np.asarray(False), selection_p=np.asarray(selection_p, dtype=np.float64))
    return result
