"""V2 LIBERO client adapter; no LIBERO imports until coordinator execution.

Preserves the stock episode loop and all its action transforms. The tap observes
env.step and (when writable) sim.step; it never calls an extra simulator step.
Privileged observations are written locally and never sent to the server.
"""
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from .snapshots import capture, atomic_npz, find_marker


def clean(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return {"unavailable_type": type(x).__name__}


def physical(env, obs=None):
    inner = getattr(env, "env", env)
    sim = getattr(inner, "sim", None)
    data = getattr(sim, "data", None)
    result = dict(privileged=True, timestep=getattr(inner, "timestep", None),
                  cur_time=getattr(inner, "cur_time", None), sim_state=np.array(env.get_sim_state(), copy=True))
    for key in ("qpos", "qvel", "ctrl", "act", "body_xpos", "body_xquat", "cvel", "sensordata", "efc_force", "cfrc_ext"):
        value = getattr(data, key, None)
        if value is not None:
            result[key] = np.array(value, copy=True)
    result["contacts"] = []
    if data is not None and hasattr(data, "contact"):
        for c in data.contact[:data.ncon]:
            result["contacts"].append({k: clean(getattr(c, k)) for k in ("geom1", "geom2", "dist", "pos", "frame", "efc_address") if hasattr(c, k)})
    result["observation_numeric"] = {k: np.array(v, copy=True) for k, v in (obs or {}).items()
        if isinstance(v, np.ndarray) and v.dtype.kind in "biuf" and v.ndim <= 1}
    # LIBERO's goal predicate evaluator is read-only; absent APIs are explicit.
    goals = getattr(inner, "goal_state", None)
    if goals is None:
        goals = getattr(inner, "parsed_problem", {}).get("goal_state")
    evaluate = getattr(inner, "_eval_predicate", None)
    if goals is not None and callable(evaluate):
        values = {}
        for pred in goals:
            try:
                values[json.dumps(clean(pred))] = bool(evaluate(pred))
            except Exception as exc:
                values[json.dumps(clean(pred))] = {"unavailable": type(exc).__name__ + ":" + str(exc)}
        result["predicates"] = values
    else:
        result["predicates"] = {"unavailable": "env goal_state/_eval_predicate not exposed"}
    result["event_labels"] = {"grasp_slip_release": "unannotated; contacts/poses/finger states are measurements, not intent labels"}
    model = getattr(sim, "model", None)
    result["entity_ids"] = {name: clean(getattr(model, name)) for name in ("body_names", "geom_names", "actuator_names") if hasattr(model, name)}
    return result


def rng_snapshot(env):
    inner = getattr(env, "env", env)
    result = dict(numpy_global=clean(np.random.get_state()), python_global=clean(random.getstate()))
    for owner, obj in (("outer", env), ("inner", inner)):
        for attr in ("np_random", "rng", "_rng"):
            rng = getattr(obj, attr, None)
            if hasattr(rng, "bit_generator"):
                result[owner + "." + attr] = clean(rng.bit_generator.state)
            elif hasattr(rng, "get_state"):
                result[owner + "." + attr] = clean(rng.get_state())
    return result


class EnvTap:
    def __init__(self, env, client):
        self.inner, self.client = env, client
        self.obs = None

    def __getattr__(self, key):
        return getattr(self.inner, key)

    def set_init_state(self, state):
        obs = self.inner.set_init_state(state)
        self.obs = obs
        self.client.emit(dict(ev="reset", init_state_sha256=hashlib.sha256(np.asarray(state).tobytes()).hexdigest(),
                             initial=physical(self.inner, obs), rng=rng_snapshot(self.inner)))
        return obs

    def step(self, action):
        client = self.client
        t = time.perf_counter()
        before = physical(self.inner, self.obs)
        inner = getattr(self.inner, "env", self.inner)
        sim = getattr(inner, "sim", None)
        commands, missing = [], None
        original = getattr(sim, "step", None)
        tapped = False
        if callable(original):
            def physics_step(*args, **kw):
                commands.append(np.array(sim.data.ctrl, copy=True))
                return original(*args, **kw)
            try:
                sim.step = physics_step
                tapped = True
            except (AttributeError, TypeError) as exc:
                missing = "sim.step not writable: " + str(exc)
        else:
            missing = "sim.step unavailable"
        try:
            result = self.inner.step(action)
        except Exception as exc:
            client.emit(dict(ev="control_error", control=client.controls, decision_step=client.decision,
                             error=type(exc).__name__ + ":" + str(exc)))
            raise
        finally:
            if tapped:
                sim.step = original
        obs, reward, done, info = result
        self.obs = obs
        after = physical(self.inner, obs)
        row = dict(ev="control", control=client.controls, decision_step=client.decision,
            chunk_offset=client.offset, parent_anchor=client.parent_anchor, source=client.source,
            wait_phase=client.decision is None, action_issued=np.array(action, copy=True),
            actuator_ctrl_each_physics_step=commands, actuator_trace_missing=missing,
            before=before, after=after, reward=clean(reward), done=bool(done), info=clean(info),
            env_step_wall_ms=1000 * (time.perf_counter() - t), wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())
        client.emit(row)
        client.controls += 1
        if client.decision is not None:
            client.offset += 1
        return result


class Client:
    def __init__(self, inner, directory, every=1, snapshot_p=1.):
        if every < 1 or not 0 <= snapshot_p <= 1:
            raise ValueError("invalid snapshot sample")
        self.inner, self.directory, self.every, self.snapshot_p = inner, Path(directory), every, snapshot_p
        self.env = None
        self.identity = {}
        self.file = None
        self.failure = None
        self.controls = self.offset = 0
        self.decision = self.parent_anchor = None
        self.source = "settle"

    def __getattr__(self, key):
        return getattr(self.inner, key)

    def emit(self, row):
        if self.file is None:
            raise RuntimeError("telemetry episode identity is missing")
        self.file.write(json.dumps(clean(dict(schema="r6p3.client.v2", **self.identity, **row)), separators=(",", ":")) + "\n")
        self.file.flush()  # retain unaccepted/error prefixes for missingness audit

    def episode_start(self, **kw):
        if self.file is not None:
            self.file.close()
        self.identity = copy.deepcopy(kw.get("extra_metadata") or {})
        self.identity["task"] = kw.get("task")
        self.identity["environment_seed"] = int(os.environ.get("P3_ENV_SEED", "7"))
        uid, attempt = self.identity["task_uid"], int(self.identity.get("attempt", 1))
        key = hashlib.sha256(uid.encode()).hexdigest()[:24]
        self.path = self.directory / f"{key}_a{attempt}"
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = (self.path / "controls.jsonl").open("x")
        self.controls = self.offset = self.anchors = 0
        self.decision = self.parent_anchor = self.failure = None
        self.source = "settle"
        self.emit(dict(ev="attempt_start", source_identity=kw, restore_certified=False,
            physical_schema=dict(adapter="r6p3.telemetry.v2", quantities="MuJoCo model-native coordinates/units; observation keys retain evaluator names",
                body_xpos="world position", body_xquat="world orientation quaternion",
                action_issued="post-model output transform, before env clipping/scaling",
                actuator_ctrl_each_physics_step="actual sim.data.ctrl before each exposed sim.step call",
                contacts="geom IDs, gap, world contact position/frame and constraint address where exposed")))
        return self.inner.episode_start(**kw)

    def infer(self, obs):
        try:
            values = capture(self.env)
            values["rng_json"] = np.asarray(json.dumps(rng_snapshot(self.env)))
            t = time.perf_counter()
            result = self.inner.infer(obs)
            # Nested v1 hit metadata precedes the explicit v2 marker in some
            # interceptors' dictionaries. The top-level marker is authoritative.
            marker = result.get("__p3__") or find_marker(result)
            if marker is None or marker.get("p3_version") != 2:
                raise ValueError("V2 client requires V2 server marker")
            self.decision, self.offset = int(marker["p3_step"]), 0
            self.parent_anchor, self.source = marker["parent_anchor"], marker["source"]
            row = dict(ev="decision", decision_step=self.decision, control=self.controls,
                marker=marker, wire_chunk=np.array(result["actions"], copy=True),
                infer_wall_ms=(time.perf_counter() - t) * 1000, wall_ns=time.time_ns(),
                observation_state=np.array(obs["observation/state"], copy=True),
                requested_controls=5, observation_sha256=hashlib.sha256(
                    np.asarray(obs["observation/image"]).tobytes() + np.asarray(obs["observation/wrist_image"]).tobytes()
                    + np.asarray(obs["observation/state"]).tobytes()).hexdigest())
            if marker["p3_anchor"]:
                index = self.anchors
                self.anchors += 1
                # Stable pre-outcome sample, includes ordinary/high confidence anchors.
                u = int.from_bytes(hashlib.sha256(f"snapshot:{self.identity['task_uid']}:{self.decision}".encode()).digest()[:8], "big") / 2**64
                selected = index % self.every == 0 and u < self.snapshot_p
                row["snapshot"] = dict(selected=selected, uniform=u, probability=self.snapshot_p if index % self.every == 0 else 0.,
                                       periodic_every=self.every, restore_certified=False)
                if selected:
                    for name in ("observation/image", "observation/wrist_image", "observation/state"):
                        values[name.replace("/", "_")] = np.array(obs[name], copy=True)
                    values["served_wire_chunk"] = np.array(result["actions"], copy=True)
                    values["metadata_json"] = np.asarray(json.dumps(clean(dict(schema="r6p3.snapshot.v2", **self.identity,
                        decision_step=self.decision, control=self.controls, marker=marker, prompt=obs["prompt"],
                        restore_certified=False, remaining_env_controls=self.max_env_timestep - int(values["env_timestep"]),
                        boundary="pre-infer simulator/RNG; server pre-action record linked by uid/attempt/step",
                        snapshot_probability=row["snapshot"]["probability"]))))
                    start = time.perf_counter()
                    path = self.path / f"step_{self.decision:06d}.npz"
                    atomic_npz(path, values)
                    row["snapshot"].update(path=path.name, bytes=path.stat().st_size, write_ms=(time.perf_counter() - start) * 1000)
            self.emit(row)
            return result
        except Exception as exc:
            self.failure = exc
            self.emit(dict(ev="profile_error", error=type(exc).__name__ + ":" + str(exc)))
            raise


def run_episode(env, client, *args, **kw):
    from examples.libero.main import _run_episode
    client.env = env
    client.max_env_timestep = int(args[3]) + int(args[2].num_steps_wait)
    timing = kw.get("client_timing")
    if timing is None:
        timing = {}
        kw["client_timing"] = timing
    tapped = EnvTap(env, client)
    try:
        result = _run_episode(tapped, client, *args, **kw)
        client.emit(dict(ev="rollout_end", success=bool(result[0]), controls=client.controls,
            final_decision=client.decision, final_chunk_completed=client.offset,
            final_env_timestep=result[-1], timing=timing, remaining_env_controls=client.max_env_timestep - result[-1]))
        if client.failure is not None:
            raise RuntimeError("P3 telemetry collection failed") from client.failure
        return result
    except Exception as exc:
        client.emit(dict(ev="rollout_error", controls=client.controls, error=type(exc).__name__ + ":" + str(exc)))
        raise
