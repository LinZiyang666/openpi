"""Cheap pre-inference simulator snapshots, no simulator imports at module load.

Physics snapshots are available today; exact paired branch restart additionally
needs controller and server history validation (see BRANCHES.md).
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


def atomic_npz(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **values)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def find_marker(obj):
    if isinstance(obj, dict):
        if "p3_anchor" in obj:
            return obj
        for val in obj.values():
            found = find_marker(val)
            if found is not None:
                return found
    return None


def capture(env):
    inner = getattr(env, "env", env)
    state = {"sim_state": np.array(env.get_sim_state(), copy=True),
             "env_timestep": np.asarray(inner.timestep), "env_cur_time": np.asarray(inner.cur_time)}
    sim = getattr(inner, "sim", None)
    for key in ("ctrl", "act", "qacc_warmstart", "mocap_pos", "mocap_quat", "userdata"):
        value = getattr(getattr(sim, "data", None), key, None)
        if value is not None:
            state["sim_data_" + key] = np.array(value, copy=True)
    # Controller numerical fields are captured for audit, not blindly restored.
    skipped = []
    for i, robot in enumerate(getattr(inner, "robots", [])):
        controller = getattr(robot, "controller", None)
        for name, value in vars(controller).items() if controller is not None else []:
            key = f"controller_{i}_{name}"
            if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
                state[key] = value.copy()
            elif isinstance(value, (int, float, bool)):
                state[key] = np.asarray(value)
            else:
                skipped.append(f"{key}:{type(value).__name__}")
    state["controller_skipped_json"] = np.asarray(json.dumps(skipped))
    return state


class Client:
    def __init__(self, inner, directory, every=1):
        if every < 1:
            raise ValueError("snapshot interval must be positive")
        self.inner, self.directory, self.every = inner, Path(directory), every
        self.identity, self.env = {}, None
        self.anchors = 0
        self.failure = None

    def __getattr__(self, key):
        return getattr(self.inner, key)

    def episode_start(self, **kw):
        self.identity = copy.deepcopy(kw.get("extra_metadata") or {})
        self.identity["task"] = kw.get("task")
        self.anchors = 0
        self.failure = None
        return self.inner.episode_start(**kw)

    def infer(self, obs):
        # State capture is read-only and happens before any served action.
        t = time.perf_counter()
        try:
            values = capture(self.env)
            result = self.inner.infer(obs)
            marker = find_marker(result)
            if marker is None:
                raise ValueError("snapshot worker requires an enabled P3 server")
            if marker["p3_anchor"]:
                index = self.anchors
                self.anchors += 1
                if index % self.every == 0:
                    uid = self.identity["task_uid"]
                    attempt = int(self.identity.get("attempt", 1))
                    step = int(marker["p3_step"])
                    key = hashlib.sha256(uid.encode()).hexdigest()[:24]
                    path = self.directory / f"{key}_a{attempt}" / f"step_{step:06d}.npz"
                    for name in ("observation/image", "observation/wrist_image", "observation/state"):
                        values[name.replace("/", "_")] = np.array(obs[name], copy=True)
                    values["served_wire_chunk"] = np.array(result["actions"], copy=True)
                    values["metadata_json"] = np.asarray(json.dumps(dict(
                        schema="r6p3.snapshot.v1", **self.identity, decision_step=step,
                        anchor_index=index, prompt=obs["prompt"], restore_certified=False,
                        max_env_timestep=getattr(self, "max_env_timestep", None),
                        replan_steps=getattr(self, "replan_steps", None),
                        capture_plus_infer_ms=1000 * (time.perf_counter() - t))))
                    atomic_npz(path, values)
            return result
        except Exception as exc:
            self.failure = exc
            raise


def run_episode(env, client, *args, **kw):
    from examples.libero.main import _run_episode
    client.env = env
    client.max_env_timestep = int(args[3]) + int(args[2].num_steps_wait)
    client.replan_steps = int(args[2].replan_steps)
    result = _run_episode(env, client, *args, **kw)
    # Stock rollout catches inference exceptions. Raise outside that catch so
    # an incomplete snapshot episode cannot become an accepted ordinary failure.
    if client.failure is not None:
        raise RuntimeError("P3 snapshot collection failed") from client.failure
    return result
