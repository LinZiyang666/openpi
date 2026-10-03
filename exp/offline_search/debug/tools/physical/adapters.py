"""Environment conventions, goal semantics and frames, separate from analysis.

LIBERO defaults reproduce E3's published heuristic units. Other environments
must declare these scales and conventions in physical_adapter or a JSON config.
No task text/name rules are used. RoboCasa/MetaWorld implement the same Goal
contract through a captured goal_objects list and success sub-predicates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

import numpy as np

from .common import Unavailable, fingerprint, present


@dataclass(frozen=True)
class Goal:
    predicate: int
    relation: str
    object: str = ""
    destination: str = ""


def entries(catalog, kind):
    singular = {"bodies": "body", "geoms": "geom"}.get(kind, kind.rstrip("s"))
    value = catalog.get(kind, catalog.get(singular, []))
    if not len(value) and kind in ("bodies", "geoms"):
        namespace = "body" if kind == "bodies" else "geom"
        names = catalog.get(namespace + "_names", [])
        value = [
            {"name": name or "unnamed_%s_%d" % (namespace, i), "id": i}
            for i, name in enumerate(names)
        ]
        if kind == "geoms":
            body_ids = catalog.get("geom_bodyid", [])
            for i, row in enumerate(value):
                if i < len(body_ids):
                    row["body_id"] = body_ids[i]
    if isinstance(value, dict):
        return [
            dict(v, name=k) if isinstance(v, dict) else {"name": k, "id": v}
            for k, v in value.items()
        ]
    return [
        dict(x) if isinstance(x, dict) else {"name": str(x), "id": i}
        for i, x in enumerate(value)
    ]


def predicate_tokens(value):
    if isinstance(value, dict):
        args = value.get("args", value.get("objects", []))
        return [
            str(
                value.get(
                    "predicate", value.get("name", value.get("relation", "unknown"))
                )
            )
        ] + list(args)
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (list, dict)):
                return predicate_tokens(parsed)
        except ValueError:
            pass
        return value.strip("() ").split()
    return []


class EnvironmentAdapter:
    """Override goals/destination for an environment with richer recorded semantics."""

    def __init__(self, episode, config=None):
        self.episode = episode
        self.entities = episode.meta.get("entities", {}) or {}
        if isinstance(self.entities, str):
            self.entities = json.loads(self.entities)
        declared = dict(episode.manifest.get("physical_adapter", {}) or {})
        declared.update(episode.meta.get("physical_adapter", {}) or {})
        self.config = dict(declared, **(config or {}))
        environment = str(
            self.config.get(
                "environment",
                episode.meta.get("environment", episode.meta.get("suite", "")),
            )
        ).lower()
        self.libero = "libero" in environment or environment in ("l10", "sp", "spatial")
        defaults = {
            "gripper_dim": 6,
            "close_sign": 1,
            "gripper_threshold": 0.0,
            "obj_quat_order": "xyzw",
            "eef_quat_order": "xyzw",
            "lift_height": 0.03,
            "near_radius": 0.10,
            "place_radius": 0.10,
            "disturbance_radius": 0.05,
            "tail_controls": 100,
            "stall_path": 0.03,
            "oscillation_path": 0.15,
            "oscillation_ratio": 0.15,
            "lift_window": 40,
            "units": "meters",
            "defaults_source": "E3 published LIBERO heuristics",
        }
        defaults.update(
            calibration_move_delta=0.002, near_padding=0.02, near_bounds=[0.08, 0.15]
        )
        self.settings = dict(defaults if self.libero else {}, **self.config)
        key = json.dumps(
            [episode.meta.get("suite"), episode.meta.get("task_id")],
            separators=(",", ":"),
        )
        calibrated = self.config.get("near_radius_by_task", {}).get(key)
        if calibrated is not None and "near_radius" not in self.config:
            self.settings["near_radius"] = calibrated["radius"]
            self.settings["near_radius_source"] = calibrated
        if (
            self.libero
            and "geom_bodyid" in self.entities
            and "obj_quat_order" not in self.config
        ):
            self.settings["obj_quat_order"] = (
                "wxyz"  # MuJoCo body_xquat; proprio is xyzw.
            )
        self.names = [x["name"] for x in entries(self.entities, "movable")]
        self.name_to_index = {name: i for i, name in enumerate(self.names)}
        if len(self.names) != len(self.name_to_index):
            raise Unavailable("duplicate movable entity names")
        for i, entity in enumerate(entries(self.entities, "movable")):
            for alias in entity.get("aliases", []):
                if alias in self.name_to_index and self.name_to_index[alias] != i:
                    raise Unavailable("ambiguous movable entity alias")
                self.name_to_index[alias] = i
        self.predicates = self.entities.get("predicates", [])
        if isinstance(self.predicates, dict):
            self.predicates = list(self.predicates)

    @property
    def fingerprint(self):
        return fingerprint(self.settings)

    def setting(self, name):
        if name not in self.settings:
            raise Unavailable("adapter must declare " + name)
        return self.settings[name]

    def closed(self):
        self.episode.require("action")
        index = int(self.setting("gripper_dim"))
        if not 0 <= index < self.episode.controls["action"].shape[1]:
            raise Unavailable("gripper_dim outside action geometry")
        return (
            self.episode.controls["action"][:, index]
            - float(self.setting("gripper_threshold"))
        ) * float(self.setting("close_sign")) > 0

    def goals(self):
        explicit = self.settings.get(
            "goal_objects",
            self.entities.get("goal_objects", self.episode.meta.get("goal_objects")),
        )
        if explicit is not None:
            return [
                Goal(
                    int(g["predicate"]),
                    g.get("relation", "placement"),
                    g.get("object", ""),
                    g.get("destination", ""),
                )
                for g in explicit
            ]
        goals = []
        for i, p in enumerate(self.predicates):
            if isinstance(p, dict) and p.get("object"):
                goals.append(
                    Goal(
                        i,
                        p.get("relation", p.get("name", "placement")),
                        p["object"],
                        p.get("destination", ""),
                    )
                )
                continue
            tokens = predicate_tokens(p)
            if not tokens:
                goals.append(Goal(i, "unknown"))
                continue
            relation = tokens[0].lower()
            if self.libero and relation in ("in", "on") and len(tokens) >= 3:
                goals.append(Goal(i, relation, tokens[1], tokens[2]))
            else:
                goals.append(Goal(i, relation))
        return goals

    def reference(self):
        ep = self.episode
        ep.require("is_settle", "obj_pos")
        if ep.controls["obj_pos"].shape[1] != len(self.names):
            raise Unavailable("obj_pos not aligned with entities.movable")
        settle = np.asarray(ep.controls["is_settle"], bool)
        active = np.flatnonzero(~settle)
        if not len(active):
            raise Unavailable("no post-settle active control")
        first = int(active[0])
        if settle[first:].any():
            raise Unavailable("settling/discontinuity inside active episode")
        if first:
            return ep.controls["obj_pos"][first - 1], first - 1, "last_settle_after"
        reset = ep.snapshots.get("reset", ep.snapshots.get("snap_reset", {}))
        if "obj_pos" in reset:
            return np.asarray(reset["obj_pos"], float), -1, "reset_before_control_zero"
        if "post_settle_obj_pos" in ep.meta:
            return (
                np.asarray(ep.meta["post_settle_obj_pos"], float),
                -1,
                "declared_post_settle",
            )
        raise Unavailable(
            "no post-settle object reference; control zero is not a reset state"
        )

    def destination(self, goal):
        name = goal.destination
        if name in self.name_to_index:
            return self.episode.controls["obj_pos"][:, self.name_to_index[name]]
        for entity in entries(self.entities, "movable"):
            aliases = entity.get("regions", entity.get("aliases", []))
            if name in aliases:
                return self.episode.controls["obj_pos"][
                    :, self.name_to_index[entity["name"]]
                ]
        # LIBERO region identifiers encode a body prefix; explicit aliases win.
        if self.libero:
            match = re.match(r"(.+?_\d+)(?:_|$)", name)
            base = match.group(1) if match else name
            if base in self.name_to_index:
                return self.episode.controls["obj_pos"][:, self.name_to_index[base]]
        destinations = self.settings.get(
            "destinations", self.entities.get("destinations", {})
        )
        if isinstance(destinations, dict) and name in destinations:
            value = destinations[name]
            pos = value.get("pos") if isinstance(value, dict) else value
            if pos is not None:
                return np.broadcast_to(np.asarray(pos, float), (self.episode.n, 3))
        if "destination_pos" in self.episode.controls:
            names = self.entities.get("destination_names", [])
            if self.libero and name not in names:
                match = re.match(r"(.+?_\d+)(?:_|$)", name)
                base = match.group(1) if match else name
                matches = [x for x in names if x.startswith(base)]
                name = matches[0] if len(matches) == 1 else name
            if name in names:
                return self.episode.controls["destination_pos"][:, names.index(name)]
        return None

    def quat_matrix(self, quat, kind="obj"):
        order = self.setting(kind + "_quat_order")
        return quat_matrix(quat, order)

    def width(self, index):
        q = self.episode.controls.get("gripper_qpos")
        if q is None or not np.isfinite(q[index]).all():
            return None
        if self.libero and np.size(q[index]) == 2:
            return float(abs(q[index, 0] - q[index, 1]))
        width = self.settings.get("width_weights")
        return float(np.asarray(q[index]) @ width) if width is not None else None

    def contact_map(self):
        """Authoritative numeric IDs only; never enumerate a filtered name list."""
        status = self.entities.get(
            "geom_mapping_status", self.episode.meta.get("geom_mapping_status")
        )
        if isinstance(status, dict):
            status = status.get("status")
        if status in ("unsupported", "error", "unavailable") or self.episode.meta.get(
            "p3_contact_names_invalid"
        ):
            raise Unavailable("geom mapping is known invalid (P3 v2)")
        bodies = {
            int(x["id"]): x
            for x in entries(self.entities, "bodies")
            if present(x.get("id"))
        }
        geoms = entries(self.entities, "geoms")
        if not geoms:
            raise Unavailable("no authoritative geom id-to-body catalog")
        result = {}
        for g in geoms:
            if not present(g.get("id")) or not present(g.get("body_id")):
                raise Unavailable("geom catalog lacks numeric id/body_id")
            bid = int(g["body_id"])
            if bid not in bodies:
                raise Unavailable("geom refers to unknown body id")
            body = bodies[bid]
            role = g.get("role", body.get("role", ""))
            name = g.get("entity", body.get("entity", body["name"]))
            for entity in entries(self.entities, "movable"):
                if entity.get("body_id") == bid:
                    name = entity["name"]
                    role = role or entity.get("role", "")
                    break
            result[int(g["id"])] = {
                "name": g["name"],
                "body": body["name"],
                "body_id": bid,
                "entity": name,
                "role": role,
            }
        if len(result) != len(geoms):
            raise Unavailable("duplicate geom IDs")
        return result

    def contacts(self, control):
        mapping = self.contact_map()
        c = self.episode.controls
        if "contact_off" not in c or "contact_geom" not in c:
            raise Unavailable("contacts not captured")
        off = np.asarray(c["contact_off"])
        geom = np.asarray(c["contact_geom"])
        if (
            off.shape != (self.episode.n + 1,)
            or off[0] != 0
            or (np.diff(off) < 0).any()
            or off[-1] != len(geom)
        ):
            raise Unavailable("invalid contact offsets")
        result = []
        for i in range(int(off[control]), int(off[control + 1])):
            left, right = [mapping.get(int(x)) for x in geom[i]]
            if left is None or right is None:
                raise Unavailable("contact geom missing from catalog")
            force = c.get("contact_force")
            normal = (
                float(force[i, 0])
                if force is not None and np.isfinite(force[i]).all()
                else None
            )
            result.append(
                {
                    "left": left,
                    "right": right,
                    "normal_force": normal,
                    "force_status": "available"
                    if normal is not None
                    else "unavailable",
                }
            )
        return result


def quat_matrix(quat, order="xyzw"):
    q = np.asarray(quat, float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or (norm <= np.finfo(float).eps).any():
        raise Unavailable("invalid orientation quaternion")
    q = q / norm
    if order == "wxyz":
        q = q[..., [1, 2, 3, 0]]
    elif order != "xyzw":
        raise Unavailable("unknown quaternion order: " + str(order))
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def relative_pose(adapter, eef_pos, eef_quat, obj_pos, obj_quat):
    ro = adapter.quat_matrix(obj_quat, "obj")
    re = adapter.quat_matrix(eef_quat, "eef")
    xyz = ro.T @ (eef_pos - obj_pos)
    rotation = ro.T @ re
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return xyz, yaw


def calibrate(episodes, config=None):
    """E3 p95+padding carry radius on successful discovery physics only."""
    config = dict(config or {})
    if "near_radius" in config or "near_radius_by_task" in config:
        return config
    groups = {}
    for ep in episodes:
        if not ep.outcome["success"] or not 0 <= int(ep.meta.get("init", 50)) < 30:
            continue
        try:
            ep.require("eef_pos", "obj_pos", "is_settle")
            adapter = EnvironmentAdapter(ep, config)
            reference, _, _ = adapter.reference()
            closed = adapter.closed()
            for goal in adapter.goals():
                if goal.object not in adapter.name_to_index:
                    continue
                k = adapter.name_to_index[goal.object]
                obj = ep.controls["obj_pos"][:, k]
                lifted = obj[:, 2] - reference[k, 2] > float(
                    adapter.setting("lift_height")
                )
                moving = np.r_[
                    False,
                    np.linalg.norm(np.diff(obj, axis=0), axis=1)
                    > float(adapter.setting("calibration_move_delta")),
                ]
                select = (
                    lifted & moving & closed & ~ep.controls["is_settle"].astype(bool)
                )
                if select.any():
                    key = json.dumps(
                        [ep.meta.get("suite"), ep.meta.get("task_id")],
                        separators=(",", ":"),
                    )
                    values, keys, settings = groups.setdefault(
                        key, ([], set(), adapter.settings)
                    )
                    values.extend(
                        np.linalg.norm(ep.controls["eef_pos"] - obj, axis=1)[
                            select
                        ].tolist()
                    )
                    keys.add(ep.key)
        except (Unavailable, ValueError, KeyError):
            continue
    result = {}
    for key, (values, keys, settings) in groups.items():
        result[key] = {
            "radius": float(
                np.clip(
                    np.quantile(values, 0.95) + settings["near_padding"],
                    *settings["near_bounds"],
                )
            ),
            "controls": len(values),
            "training_keys": sorted(keys),
            "source": "E3 successful discovery moving/lifted/closed p95+adapter padding, clipped to adapter bounds",
        }
    config["near_radius_by_task"] = result
    return config
