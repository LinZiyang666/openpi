"""Passive stock-runner injection. Capture failures never escape into actions."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from exp.offline_search.debug.schema import CONTROL_BLOCK_SIZE, SCHEMA_VERSION, decision_id, episode_key, snapshot_sampled, status
from .adapter import MujocoAdapter
from .snapshots import capture as snapshot, clean


class FileSink:
    def __init__(self, directory, uid, attempt):
        self.directory = Path(directory) / episode_key(uid, attempt)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events = (self.directory / "events.jsonl").open("x")

    def write(self, text):
        self.events.write(text)
        self.events.flush()

    def write_file(self, name, data):
        from exp.offline_search.debug.transport.protocol import filename
        filename(name)
        path = self.directory / name
        if path.exists():
            raise ValueError("duplicate immutable file")
        partial = path.with_name(name + ".part")
        with partial.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(partial), str(path))
        fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def close(self):
        self.events.flush()
        os.fsync(self.events.fileno())
        self.events.close()


def npz_bytes(arrays):
    buffer = io.BytesIO()
    fixed = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("object array forbidden: " + key)
        if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and not np.little_endian):
            array = array.astype(array.dtype.newbyteorder("<"))
        fixed[key] = array
    np.savez_compressed(buffer, **fixed)
    return buffer.getvalue()


class Client:
    def __init__(self, inner, directory, campaign=None, arm=None, env_seed=7, oracle=False,
                 oracle_radius=0.10, oracle_lift=0.03, adapter_factory=MujocoAdapter, sink_factory=None):
        self.inner, self.directory = inner, Path(directory)
        self.campaign = campaign or os.environ.get("OSDEBUG_CAMPAIGN", self.directory.parent.parent.name)
        self.arm = arm or os.environ.get("OSDEBUG_ARM", self.directory.parent.name)
        self.env_seed, self.oracle_enabled = int(env_seed), bool(oracle)
        self.oracle_radius, self.oracle_lift = float(oracle_radius), float(oracle_lift)
        self.adapter_factory, self.sink_factory = adapter_factory, sink_factory
        self.env = self.adapter = self.sink = None
        self.ended = True
        self.last_obs = None

    def __getattr__(self, key):
        return getattr(self.inner, key)

    def _safe(self, label, fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except Exception as exc:
            row = dict(status="error", operation=label, reason=type(exc).__name__ + ":" + str(exc))
            self.errors.append(row)
            self.episode["capture_errors"] = self.errors
            return None

    def emit(self, row):
        self.sink.write(json.dumps(clean(dict(schema=SCHEMA_VERSION, **self.identity, **row)),
                                   separators=(",", ":"), allow_nan=False) + "\n")

    def episode_start(self, **kw):
        meta = kw.get("extra_metadata") or {}
        self.errors = []
        self.controls = self.offset = self.seq = self.block = 0
        self.decision = self.pending = None
        self.rows = []
        self.ended = False
        self.sink = self.adapter = None
        self.last_obs = None
        identity_error = None
        try:
            uid, attempt = str(meta["task_uid"]), int(meta.get("attempt", 1))
            gen = int(meta.get("dispatch_gen", attempt))
            if not uid or attempt < 1 or gen < 0:
                raise ValueError("invalid capture identity")
        except Exception as exc:
            uid, attempt, gen = "capture_error:" + str(self.arm), 1, 0
            identity_error = dict(status="error", operation="identity", reason=type(exc).__name__ + ":" + str(exc))
        self.identity = dict(task_uid=uid, attempt=attempt, dispatch_gen=gen, episode_key=episode_key(uid, attempt))
        self.episode = dict(schema=SCHEMA_VERSION, **self.identity, arm=self.arm, campaign=self.campaign,
                            suite=kw.get("experiment"), task_id=meta.get("task_id"), task_text=kw.get("task"),
                            init=meta.get("subset_init_state_idx", meta.get("init", kw.get("episode_id"))),
                            orig_init_state_idx=meta.get("orig_init_state_idx"), env_seed=self.env_seed,
                            t_start=time.time(), oracle=dict(enabled=self.oracle_enabled, privileged=self.oracle_enabled),
                            capture_errors=self.errors, capabilities={}, server_join="unverified")
        if identity_error:
            self.errors.append(identity_error)
        def open_sink():
            if self.sink_factory:
                self.sink = self.sink_factory(self.directory, uid, attempt)
            elif os.environ.get("OSDEBUG_STREAM"):
                from exp.offline_search.debug.transport.sink import StreamSink
                self.sink = StreamSink.from_environment(self.directory, uid, attempt)
            else:
                self.sink = FileSink(self.directory, uid, attempt)
        self._safe("episode_open", open_sink)
        return self.inner.episode_start(**kw)

    def configure(self, env, args, max_steps):
        self.env = env
        self.settle_controls = int(args.num_steps_wait)
        self.episode.update(max_steps=int(max_steps), settle_controls=self.settle_controls,
                            replan_steps=int(args.replan_steps))
        self.env_seed = int(getattr(args, "seed", self.env_seed))
        self.episode["env_seed"] = self.env_seed

    def reset_capture(self, initial_state, obs):
        self.last_obs = obs
        self.adapter = self.adapter_factory(self.env)
        reset_state = np.asarray(self.env.get_sim_state(), dtype=np.float64)
        self.episode.update(entities=self.adapter.catalog, capabilities=self.adapter.capabilities,
                            control_dt=float(getattr(self.adapter.inner, "control_timestep", np.nan)),
                            initial_state_sha256=hashlib.sha256(np.asarray(initial_state).tobytes()).hexdigest(),
                            reset_state_sha256=hashlib.sha256(reset_state.tobytes()).hexdigest(),
                            reset_state_sha256_scope="environment get_sim_state flat float64, including simulator time",
                            camera_names=[], image_shapes={}, action_dim=None, state_dim=None)
        self.episode["versions"] = dict(python=sys.version, numpy=np.__version__, **{
            name: getattr(sys.modules.get(name), "__version__", "not_exposed") for name in ("mujoco", "mujoco_py", "robosuite", "libero")})
        self.emit(dict(ev="reset", env_seed=self.env_seed, snapshot_file="snap_reset.npz"))
        self.save_snapshot("snap_reset.npz", 1.)

    def save_snapshot(self, name, probability):
        self.sink.write_file(name, npz_bytes(snapshot(self.env, probability)))

    def _end_decision(self):
        if self.pending is not None:
            self.pending["n_applied"] = self.offset
            row, self.pending = self.pending, None
            self.emit(row)

    def infer(self, obs):
        self._safe("decision_event", self._end_decision)
        seq = self.seq
        self.seq += 1
        self.decision, self.offset = seq, 0
        selected = snapshot_sampled(self.campaign, self.identity["task_uid"], seq)
        name = "snap_{:06d}.npz".format(seq) if selected else None
        if seq == 0 and self.adapter is not None:
            self._safe("settle_reference", self.adapter.set_reference)
            # A diagnostic failure must not enter capture_errors or admission.
            try:
                self.episode["reset_selfcheck"] = self.adapter.resting_selfcheck()
            except Exception as exc:
                self.episode["reset_selfcheck"] = dict(status="unsupported", diagnostic_only=True,
                                                       reason=type(exc).__name__ + ":" + str(exc))
            self._safe("settle_event", self.emit, dict(ev="settle", n_controls=self.controls,
                                                     reference="pre-first-decision (post-settle)", selfcheck=self.episode["reset_selfcheck"]))
            self.episode["post_settle_obj_pos"] = self.adapter.reference.tolist() if self.adapter.reference is not None else None
        if selected:
            self._safe("snapshot", self.save_snapshot, name, 1. if seq == 0 else 1. / 16)
        did = decision_id(self.identity["episode_key"], self.identity["dispatch_gen"], seq)
        outgoing = dict(obs)
        sent = time.time()
        outgoing["__debug__"] = dict(v=1, **self.identity, decision_id=did, decision_seq=seq, t_client_send=sent)
        if self.oracle_enabled:
            oracle = self._safe("oracle", self.adapter.oracle, self.last_obs, self.oracle_radius, self.oracle_lift) if self.adapter else None
            outgoing["__oracle__"] = oracle or dict(v=1, privileged=True, status="error", reason="oracle capture failed", in_window=False)
        def wire_manifest():
            cameras = sorted(k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3)
            self.episode.update(camera_names=cameras, image_shapes={k: list(obs[k].shape) for k in cameras},
                                state_dim=int(np.asarray(obs["observation/state"]).size))
        self._safe("wire_manifest", wire_manifest)
        t0 = time.perf_counter()
        # This is the only inference call. Diagnostic errors cannot skip/retry it.
        result = self.inner.infer(outgoing)
        received = time.time()
        returned = dict(result)
        echo = returned.pop("__debug__", None)
        self.pending = dict(ev="decision", decision_seq=seq, decision_id=did, control_idx_start=self.controls,
                            n_applied=0, server_echo=echo.get("decision_id") if isinstance(echo, dict) else None,
                            server_meta=echo, t_send=sent, t_recv=received, infer_ms=1000 * (time.perf_counter() - t0),
                            snapshot_file=name, snapshot_status=status("available" if name else "not_sampled", "hashed pre-decision sample"))
        verified = isinstance(echo, dict) and echo.get("decision_id") == did and echo.get("status") == "available"
        self.pending["server_join"] = "verified" if verified else "unverified"
        self.episode["server_join"] = "verified" if verified and (seq == 0 or self.episode["server_join"] == "verified") else "unverified"
        if self.oracle_enabled:
            self.pending["oracle"] = outgoing["__oracle__"]
        return returned

    def control_capture(self, action, result, commands, missing):
        obs, reward, done, info = result
        self.last_obs = obs
        row = self.adapter.physical(obs)
        row.update(control_idx=self.controls, decision_seq=-1 if self.decision is None else self.decision,
                   chunk_offset=-1 if self.decision is None else self.offset, is_settle=self.decision is None,
                   action=np.array(action, dtype=np.float64, copy=True), reward=float(reward), done=bool(done))
        nu = int(self.adapter.model.nu)
        row["act_substep"] = np.asarray(commands, dtype=np.float32).reshape((-1, nu)) if commands else np.empty((0, nu), np.float32)
        self.adapter.capabilities["actuator_substeps"] = status("unsupported" if missing else "available",
                                                              missing or "actual ctrl before each exposed sim.step")
        self.episode["action_dim"] = int(row["action"].size)
        self.rows.append(row)
        if len(self.rows) >= CONTROL_BLOCK_SIZE:
            self.flush_controls()

    def flush_controls(self):
        if not self.rows:
            return
        rows = self.rows
        arrays = {}
        flat = {"contact_geom", "contact_dist", "contact_pos", "contact_frame", "contact_force"}
        ints = {"control_idx", "decision_seq", "chunk_offset"}
        for key in rows[0]:
            if key in flat:
                arrays[key] = np.concatenate([r[key] for r in rows], axis=0)
            elif key == "act_substep":
                # Varying physics substep counts remain lossless with NaN padding
                # and explicit counts; no object arrays or fabricated commands.
                steps = np.asarray([len(r[key]) for r in rows], dtype=np.int32)
                arrays["act_substep_count"] = steps
                arrays[key] = np.full((len(rows), int(steps.max()), int(self.adapter.model.nu)), np.nan, np.float32)
                for i, r in enumerate(rows):
                    arrays[key][i, :len(r[key])] = r[key]
            else:
                arrays[key] = np.asarray([r[key] for r in rows], dtype=np.int32 if key in ints else bool if key in ("done", "is_settle") else np.float64)
        arrays["contact_off"] = np.r_[0, np.cumsum([len(r["contact_geom"]) for r in rows])].astype(np.int32)
        self.sink.write_file("controls_{:04d}.npz".format(self.block), npz_bytes(arrays))
        self.rows = []
        self.block += 1

    def finish(self, success, termination_reason, error=None):
        if self.ended:
            return
        self._safe("decision_event", self._end_decision)
        self._safe("control_block", self.flush_controls)
        self._safe("snapshot_final", self.save_snapshot, "snap_final.npz", 1.)
        self.episode.update(t_end=time.time(), n_controls=self.controls, n_decisions=self.seq, success=bool(success),
                            termination_reason=termination_reason, error=error, capture_errors=self.errors)
        if self.adapter:
            self.episode["capabilities"] = self.adapter.capabilities
        self._safe("done_event", self.emit, dict(ev="done", success=bool(success), termination_reason=termination_reason,
                                                n_controls=self.controls, n_decisions=self.seq, capture_errors=self.errors))
        self._safe("episode_metadata", self.sink.write_file, "episode.json",
                   (json.dumps(clean(self.episode), sort_keys=True, allow_nan=False) + "\n").encode()) if self.sink else None
        if self.sink:
            self._safe("transport_close", self.sink.close)
        self.ended = True


class EnvTap:
    def __init__(self, env, client):
        self.inner, self.client = env, client

    def __getattr__(self, key):
        return getattr(self.inner, key)

    def set_init_state(self, state):
        obs = self.inner.set_init_state(state)
        self.client._safe("reset", self.client.reset_capture, state, obs)
        return obs

    def step(self, action):
        client = self.client
        issued = client._safe("action_copy", np.array, action, dtype=np.float64, copy=True)
        commands = []
        sim = client.adapter.sim if client.adapter else None
        original = getattr(sim, "step", None)
        instance_dict = getattr(sim, "__dict__", None)
        had_instance_step = instance_dict is None or "step" in instance_dict
        tapped, missing = False, "sim.step unavailable"
        if callable(original):
            def physics_step(*args, **kw):
                client._safe("actuator_substep", lambda: commands.append(np.array(sim.data.ctrl, copy=True)))
                return original(*args, **kw)
            try:
                sim.step = physics_step
                tapped, missing = True, None
            except (AttributeError, TypeError) as exc:
                missing = "sim.step not writable: " + str(exc)
        try:
            result = self.inner.step(action)
        finally:
            if tapped:
                if had_instance_step:
                    client._safe("restore_sim_step", setattr, sim, "step", original)
                else:
                    client._safe("restore_sim_step", delattr, sim, "step")
        client._safe("control", client.control_capture, issued, result, commands, missing)
        client.controls += 1
        if client.decision is not None:
            client.offset += 1
        return result


def run_episode(env, client, *args, **kw):
    from examples.libero.main import _run_episode
    client.configure(env, args[2], args[3])
    timing = kw.setdefault("client_timing", {})
    try:
        result = _run_episode(EnvTap(env, client), client, *args, **kw)
    except Exception as exc:
        client._safe("lifecycle_error", client.emit, dict(ev="error", error=type(exc).__name__ + ":" + str(exc)))
        client.finish(False, "exception", type(exc).__name__ + ":" + str(exc))
        raise
    client.finish(result[0], timing.get("termination_reason", "success" if result[0] else "step_cap"))
    return result
