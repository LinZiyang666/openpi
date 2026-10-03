"""Read-only MuJoCo/LIBERO adapter. Model IDs, never compact name lists.

P3 telemetry.py supplied the env/sim and predicate access pattern. This adapter
copies only state, resolves the model's full ID space and obtains contact forces.
Other benchmarks can supply an adapter with the same catalog/physical methods.
"""
import json

import numpy as np

from exp.offline_search.debug.schema import status


def unwrap(env):
    """Return the environment that owns the simulator.

    Outer render wrappers may forward ``sim``; prefer the level that also exposes the task's goal evaluator
    (LIBERO's problem env, ``env.env`` as in P3), else the first level with a simulator.
    """
    seen = set()
    first = None
    while id(env) not in seen:
        seen.add(id(env))
        if getattr(env, "sim", None) is not None:
            if first is None:
                first = env
            if callable(getattr(env, "_eval_predicate", None)) or getattr(env, "parsed_problem", None) is not None:
                return env
        child = getattr(env, "env", None)
        if child is None:
            break
        env = child
    return first if first is not None else env


def names_by_id(model, kind, count):
    """Unnamed model entries retain their slot (the defect in P3's lists)."""
    method = getattr(model, kind + "_id2name", None)
    result = []
    for i in range(count):
        if callable(method):
            name = method(i)
        else:
            try:
                name = getattr(model, kind)(i).name  # native mujoco named access
            except (AttributeError, TypeError):
                import mujoco
                enum = getattr(mujoco.mjtObj, "mjOBJ_" + kind.upper())
                name = mujoco.mj_id2name(model, enum, i)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        result.append(name if name else None)
    return result


def _array(data, key, shape, dtype=np.float64):
    value = getattr(data, key, None)
    return np.full(shape, np.nan, dtype=dtype) if value is None else np.array(value, dtype=dtype, copy=True)


class MujocoAdapter:
    def __init__(self, env, contact_force=None):
        self.env = env
        self.inner = unwrap(env)
        self.sim = self.inner.sim
        self.model, self.data = self.sim.model, self.sim.data
        self.force_fn = contact_force
        if self.force_fn is None:
            self.force_fn = getattr(self.sim, "contact_force", None)
        if self.force_fn is None:
            try:
                if type(self.model).__module__.startswith("mujoco_py"):
                    from mujoco_py import functions
                    self.force_fn = functions.mj_contactForce
                else:
                    import mujoco
                    self.force_fn = mujoco.mj_contactForce
            except ImportError:
                pass
        self.catalog = self._catalog()
        self.movable_ids = np.asarray([r["body_id"] for r in self.catalog["movable"]], dtype=np.int32)
        self.goals = self.catalog["goal_state"]
        self.reference = None
        self.capabilities = {
            "contacts": status("available" if hasattr(self.data, "contact") and self.force_fn else "unsupported",
                               "model ID maps; mj_contactForce" if self.force_fn else "mj_contactForce unavailable"),
            "predicates": status("available" if callable(getattr(self.inner, "_eval_predicate", None)) else "unsupported",
                                 "read-only goal evaluation"),
            "actuator_substeps": status("unsupported", "sim.step tap not yet exercised"),
            "snapshots": status("available", "numeric copy; restore uncertified"),
            "physics": status("available", "MuJoCo model-native float64 state"),
        }

    def _catalog(self):
        model = self.model
        counts = {"body": int(model.nbody), "geom": int(model.ngeom), "joint": int(model.njnt),
                  "site": int(model.nsite), "actuator": int(model.nu)}
        result = {kind + "_names": names_by_id(model, kind, n) for kind, n in counts.items()}
        # MuJoCo calls its joint namespace 'joint', but its structural arrays 'jnt'.
        result.update(geom_bodyid=np.asarray(model.geom_bodyid).astype(int).tolist(),
                      body_parentid=np.asarray(model.body_parentid).astype(int).tolist(),
                      joint_bodyid=np.asarray(model.jnt_bodyid).astype(int).tolist(),
                      joint_type=np.asarray(model.jnt_type).astype(int).tolist())
        roots = set(int(i) for i in model.jnt_bodyid)
        robot_roots = set()
        for robot in getattr(self.inner, "robots", []):
            ids = getattr(robot, "_ref_joint_indexes", [])
            robot_roots.update(int(model.jnt_bodyid[i]) for i in ids)
        # LIBERO robot naming is a benchmark adapter convention, not a task rule.
        robot_roots.update(i for i, name in enumerate(result["body_names"]) if name and name.startswith("robot"))
        moving = set()
        for body in range(1, counts["body"]):
            ancestor = body
            has_motion, robot_body = False, False
            while ancestor > 0:
                if ancestor in robot_roots:
                    robot_body = True
                    break
                if ancestor in roots:
                    has_motion = True
                ancestor = int(model.body_parentid[ancestor])
            if has_motion and not robot_body:
                moving.add(body)
        object_roots = {}
        # obj_body_id is the authoritative LIBERO mapping, including fixtures.
        for key, body in getattr(self.inner, "obj_body_id", {}).items():
            if np.isscalar(body):
                object_roots[str(key)] = int(body)
        descriptors = []
        for attribute in ("objects", "objects_dict", "fixtures_dict"):
            value = getattr(self.inner, attribute, [])
            descriptors.extend(value.values() if isinstance(value, dict) else value)
        for obj in descriptors:
            name, root = getattr(obj, "name", None), getattr(obj, "root_body", None)
            if name and root in result["body_names"]:
                object_roots[str(name)] = result["body_names"].index(root)
        goals = getattr(self.inner, "goal_state", None)
        if goals is None:
            goals = getattr(self.inner, "parsed_problem", {}).get("goal_state", [])
        goals = [list(p) for p in (goals or [])]
        goal_objects = {str(p[1]) for p in goals if len(p) >= 3}
        result["goal_state"] = goals
        result["predicates"] = [json.dumps(p, separators=(",", ":")) for p in goals]
        result["object_body_ids"] = object_roots
        result["movable"] = []
        for body in sorted(moving):
            aliases = [k for k, v in object_roots.items() if v == body]
            result["movable"].append(dict(body_id=body, name=result["body_names"][body], aliases=aliases,
                                           role="goal_object" if goal_objects.intersection(aliases) else "movable"))
        result["table_geom_ids"] = [i for i, name in enumerate(result["geom_names"])
                                    if name and "table" in name.lower() and "visual" not in name.lower()]
        result["units"] = "MuJoCo model-native (LIBERO metres, seconds); body/eef poses world frame"
        result["quaternion_conventions"] = dict(eef="xyzw (LIBERO observation)", objects="wxyz (MuJoCo body_xquat)")
        result["obj_vel_convention"] = "native MuJoCo cvel: angular then linear velocity"
        return result

    def predicates(self):
        values = np.full(len(self.goals), np.nan, dtype=np.float64)
        evaluate = getattr(self.inner, "_eval_predicate", None)
        if not callable(evaluate):
            return values
        errors = []
        for i, goal in enumerate(self.goals):
            try:
                values[i] = float(bool(evaluate(goal)))
            except Exception as exc:
                errors.append(type(exc).__name__ + ":" + str(exc))
        if errors:
            self.capabilities["predicates"] = status("error", "; ".join(errors))
        return values

    def physical(self, obs=None):
        data, model = self.data, self.model
        obs = obs or {}
        ids = self.movable_ids
        def proprio(key, fallback_shape):
            val = obs.get(key)
            if val is None and key in ("robot0_gripper_qpos", "robot0_gripper_qvel"):
                robots = getattr(self.inner, "robots", [])
                if robots:
                    velocity = key.endswith("qvel")
                    idx = getattr(robots[0], "_ref_gripper_joint_vel_indexes" if velocity else "_ref_gripper_joint_pos_indexes", None)
                    if idx is not None:
                        val = getattr(data, "qvel" if velocity else "qpos")[idx]
            if val is None:
                self.capabilities[key] = status("unsupported", "observation and simulator robot index adapter unavailable")
            return np.array(val, dtype=np.float64, copy=True) if val is not None else np.full(fallback_shape, np.nan)
        pos = getattr(data, "body_xpos", getattr(data, "xpos", None))
        quat = getattr(data, "body_xquat", getattr(data, "xquat", None))
        n = int(getattr(data, "ncon", 0))
        contacts = list(data.contact[:n]) if hasattr(data, "contact") else []
        force = np.full((n, 6), np.nan, dtype=np.float32)
        for i in range(n):
            if self.force_fn:
                try:
                    native_force = np.empty(6, dtype=np.float64)
                    # robosuite binding_utils wraps the native MuJoCo structs; the C API needs the raw ones.
                    self.force_fn(getattr(model, "_model", model), getattr(data, "_data", data), i, native_force)
                    force[i] = native_force
                except Exception as exc:
                    self.capabilities["contacts"] = status("error", "mj_contactForce: " + str(exc))
        def contact_field(key, shape, dtype):
            return np.asarray([getattr(c, key) for c in contacts], dtype=dtype).reshape((n,) + shape)
        return dict(qpos=_array(data, "qpos", (int(model.nq),)), qvel=_array(data, "qvel", (int(model.nv),)),
                    eef_pos=proprio("robot0_eef_pos", (3,)), eef_quat=proprio("robot0_eef_quat", (4,)),
                    gripper_qpos=proprio("robot0_gripper_qpos", (2,)), gripper_qvel=proprio("robot0_gripper_qvel", (2,)),
                    obj_pos=np.array(pos[ids], dtype=np.float64, copy=True),
                    obj_quat=np.array(quat[ids], dtype=np.float64, copy=True),
                    obj_vel=_array(data, "cvel", (int(model.nbody), 6))[ids],
                    contact_geom=np.asarray([(c.geom1, c.geom2) for c in contacts], dtype=np.int32).reshape(n, 2),
                    contact_dist=contact_field("dist", (), np.float32),
                    contact_pos=contact_field("pos", (3,), np.float32),
                    contact_frame=contact_field("frame", (9,), np.float32), contact_force=force,
                    predicates=self.predicates())

    def set_reference(self):
        """Post-settle baseline; never use the initial above-table spawn height."""
        pos = getattr(self.data, "body_xpos", getattr(self.data, "xpos", None))
        self.reference = np.array(pos[self.movable_ids], dtype=np.float64, copy=True)

    def resting_selfcheck(self):
        """Diagnostic support connectivity, including stacks and fixtures.

        A free body's descendants share its contact node. Any contact with a
        non-free body anchors that node; support then propagates through other
        free objects. An isolated free-object cycle is not an anchor.
        """
        gb = self.catalog["geom_bodyid"]
        parents = self.catalog["body_parentid"]
        free_roots = {int(b) for b, t in zip(self.model.jnt_bodyid, self.model.jnt_type) if int(t) == 0}
        free_roots.intersection_update(set(self.movable_ids.tolist()))
        def free_owner(body):
            while body > 0:
                if body in free_roots:
                    return body
                body = parents[body]
            return None
        edges = {body: set() for body in free_roots}
        supported, direct_table = set(), set()
        table = set(self.catalog["table_geom_ids"])
        for c in self.data.contact[:self.data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            left, right = free_owner(gb[g1]), free_owner(gb[g2])
            for obj, other, other_geom in ((left, right, g2), (right, left, g1)):
                if obj is not None:
                    if other is None:
                        supported.add(obj)
                    elif other != obj:
                        edges[obj].add(other)
                    if other_geom in table:
                        direct_table.add(obj)
        pending = list(supported)
        while pending:
            body = pending.pop()
            for neighbor in edges[body] - supported:
                supported.add(neighbor)
                pending.append(neighbor)
        missing = sorted(free_roots - supported)
        return dict(status="available", diagnostic_only=True, passed=not missing,
                    reason="contact chain to any non-free body; diagnostic only",
                    resting_body_ids=sorted(free_roots), supported_body_ids=sorted(supported),
                    missing_support_contacts=missing,
                    missing_table_contacts=sorted(free_roots - direct_table),
                    geom_count=int(self.model.ngeom), naming="model id2name for every ID, unnamed slots retained")

    def oracle(self, obs, radius=0.10, lift=0.03):
        """E3 window: unlifted AND unsatisfied relational goal, distance to eef.

        Thresholds belong to this explicit diagnostic adapter/config only.
        The ordinary collector has no geometric thresholds or task lookup.
        """
        if self.reference is None:
            raise ValueError("oracle requires post-settle reference")
        truth = self.physical(obs)
        objects = []
        unresolved = []
        mapping = self.catalog["object_body_ids"]
        for name in sorted({str(p[1]) for p in self.goals if len(p) >= 3}):
            body = mapping.get(name)
            if body is None or body not in self.movable_ids:
                entry = dict(object_id=name, body_id=body, status="unsupported", in_window=False,
                             reason="goal subject has no body mapping" if body is None else
                             "goal subject is not a movable body", predicate_known=False)
                objects.append(entry)
                unresolved.append(entry)
                continue
            k = self.movable_ids.tolist().index(body)
            pred_ids = [i for i, p in enumerate(self.goals) if len(p) >= 3 and p[1] == name]
            satisfied = bool(np.all(truth["predicates"][pred_ids] == 1))
            known = bool(np.all(np.isfinite(truth["predicates"][pred_ids])))
            lifted = bool(truth["obj_pos"][k, 2] - self.reference[k, 2] > lift)
            distance = float(np.linalg.norm(truth["eef_pos"] - truth["obj_pos"][k]))
            finite = bool(np.isfinite(distance) and np.isfinite(self.reference[k]).all())
            entry = dict(object_id=name, body_id=int(body), distance=distance if finite else None, lifted=lifted,
                         satisfied=satisfied, predicate_known=known, status="available" if known and finite else "unsupported",
                         reason="" if known and finite else "goal predicate status unresolved" if not known else
                         "end-effector/object/post-settle position unavailable",
                         in_window=known and finite and not satisfied and not lifted and distance < radius)
            objects.append(entry)
            if entry["status"] != "available":
                unresolved.append(entry)
        resolved = [r for r in objects if r["status"] == "available"]
        eligible = [r for r in resolved if not r["lifted"] and not r["satisfied"]]
        nearest = min(eligible, key=lambda r: r["distance"]) if eligible else None
        return dict(v=1, privileged=True, diagnostic=True,
                    status=("partial" if resolved else "unsupported") if unresolved else "available",
                    reason="unresolved goal objects: " + ",".join(r["object_id"] for r in unresolved) if unresolved else "",
                    unresolved_objects=unresolved,
                    in_window=any(r["in_window"] for r in objects),
                    grasp_window=any(r["in_window"] for r in objects), distance=nearest["distance"] if nearest else None,
                    object_id=nearest["object_id"] if nearest else None, objects=objects, radius=radius, lift_threshold=lift)
