"""Observe native MuJoCo model/data; never consult robosuite ``.sim``.

The schema is numeric and copied at the call boundary for P6's existing writers.
Missing semantics are explicit; raw body state does not certify task predicates.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _status(state, reason):
    return {"status": state, "reason": reason}


def _copy(obj, name, dtype=np.float64):
    value = getattr(obj, name, None)
    return None if value is None else np.array(value, dtype=dtype, copy=True)


class NativeMetaWorldPhysicalAdapter:
    def __init__(self, env, task_name=None, contact_force=None):
        self.env = env
        self.model, self.data = env.model, env.data
        self.task_name = task_name
        self.force_fn = contact_force
        if self.force_fn is None and type(self.model).__module__.startswith("mujoco"):
            import mujoco
            self.force_fn = mujoco.mj_contactForce
        self.catalog = self._catalog()

    def _catalog(self):
        m = self.model
        result = {}
        for kind, count in (("body", "nbody"), ("geom", "ngeom"), ("joint", "njnt"),
                            ("site", "nsite"), ("actuator", "nu")):
            names = []
            for i in range(int(getattr(m, count, 0))):
                item = getattr(m, kind)(i)
                names.append(getattr(item, "name", None) or None)
            result[kind + "_names"] = names
        for key in ("geom_bodyid", "body_parentid", "jnt_bodyid", "jnt_type", "jnt_qposadr", "jnt_dofadr"):
            value = _copy(m, key, np.int64)
            result[key] = None if value is None else value.tolist()
        # MuJoCo free joint == 0, qpos is xyz+wxyz; all joints remain catalogued.
        result["free_joint_body_ids"] = [int(b) for b, t in zip(result.get("jnt_bodyid") or [],
                                                               result.get("jnt_type") or []) if t == 0]
        result["entity_semantics"] = "native IDs; task object/robot roles require explicit mapping"
        result["units"] = "native MuJoCo; world-frame xpos, wxyz xquat, cvel angular then linear"
        return result

    def capture(self, obs=None, info=None, requested_action=None, applied_action=None):
        d = self.data
        arrays = {}
        capabilities = {}
        for key in ("qpos", "qvel", "act", "ctrl", "xpos", "xquat", "cvel", "site_xpos", "sensordata"):
            value = _copy(d, key)
            capabilities[key] = _status("available" if value is not None else "unsupported", "native data." + key)
            if value is not None:
                arrays[key] = value
        if obs is not None:
            arrays["raw_observation"] = np.array(obs, dtype=np.float64, copy=True)
        for key, value in (("requested_action", requested_action), ("applied_action", applied_action)):
            if value is not None:
                arrays[key] = np.array(value, dtype=np.float64, copy=True)
        n = int(getattr(d, "ncon", 0))
        contacts = list(getattr(d, "contact", [])[:n])
        arrays["contact_geom_ids"] = np.array([[c.geom1, c.geom2] for c in contacts], dtype=np.int64).reshape(-1, 2)
        arrays["contact_dist"] = np.array([c.dist for c in contacts], dtype=np.float64)
        arrays["contact_pos"] = np.array([c.pos for c in contacts], dtype=np.float64).reshape(-1, 3)
        arrays["contact_frame"] = np.array([c.frame for c in contacts], dtype=np.float64).reshape(-1, 9)
        forces = np.full((len(contacts), 6), np.nan, dtype=np.float64)
        force_errors = []
        if self.force_fn:
            for i in range(len(contacts)):
                try:
                    self.force_fn(self.model, d, i, forces[i])
                except Exception as exc:
                    force_errors.append(str(exc))
        arrays["contact_force"] = forces
        capabilities["contacts"] = _status("available" if hasattr(d, "contact") else "unsupported", "native data.contact")
        capabilities["contact_force"] = _status("error" if force_errors else "available" if self.force_fn else "unsupported",
                                                   "; ".join(force_errors) or "native mj_contactForce")
        target = getattr(self.env, "_target_pos", None)
        mapped = self.task_name in ("reach-v3", "reach-wall-v3") and target is not None
        if mapped:
            arrays["goal_position"] = np.array(target, dtype=np.float64, copy=True)
        capabilities["goal"] = _status("available" if mapped else "unsupported", "explicit reach _target_pos map; real-env certification pending" if mapped else "no certified task entity map")
        capabilities["predicates"] = _status("unsupported", "MetaWorld info.success is authoritative; no LIBERO predicate evaluator")
        capabilities["actuator_substeps"] = _status("unsupported", "control boundary only")
        capabilities["robot_joint_map"] = _status("unsupported", "raw joint/body IDs retained; robot role map unqualified")
        capabilities["eef_gripper_map"] = _status("unsupported", "raw site/joint state retained; task implementation mapping pending")
        capabilities["restore"] = _status("unsupported", "no task roundtrip certificate")
        task_data = getattr(self.env, "_last_rand_vec", None)
        rng = getattr(self.env, "np_random", None)
        rng_state = getattr(getattr(rng, "bit_generator", None), "state", None)
        if task_data is not None:
            arrays["last_rand_vec"] = np.array(task_data, dtype=np.float64, copy=True)
        return {"schema": "r17.physical.v1", "arrays": arrays, "capabilities": capabilities,
                "metadata": {"task_name": self.task_name, "catalog": self.catalog,
                             "time": float(getattr(d, "time", 0.0)), "info": dict(info or {}),
                             "rng_state": rng_state, "restore_certified": False,
                             "model_assets_digest": None,
                             "model_assets_status": "unsupported: asset bytes not provided"}}


def task_data_digest(task_data: bytes) -> str:
    return "sha256:" + hashlib.sha256(task_data).hexdigest()
