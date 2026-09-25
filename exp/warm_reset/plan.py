"""Freeze arm YAMLs and task identities for the warm reset conductor entry.

Experiment choices live here; serving and conductor retain their generic APIs.
The environment and arm vocabulary comes from the existing step_diag protocol.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import tempfile
import uuid
from pathlib import Path

import yaml

from exp.step_diag import envs as E
from openpi.cache.config import effective_denoise_schedule, load_cache_config
from openpi.cache.warm_reset.types import WarmResetSpec, resolve_plan


def sha(text: str) -> str:
    """Hash the exact UTF-8 YAML sent over the control channel."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_of(text: str):
    """Use the production loader without requiring server-local artifact paths."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as fh:
        fh.write(text)
        fh.flush()
        return load_cache_config(fh.name, check_files=False)


def arm_block(
    env_id: str, arm: str, *, evidence_dir: str, namespace: str
) -> dict | None:
    """Translate the established arm vocabulary into a production YAML block."""
    env = E.ENVS[env_id]
    if not re.fullmatch(r"[a-z0-9]+_t[0-9.]+(?:_n[0-9]+)?", arm):
        raise ValueError(f"invalid arm id: {arm!r}")
    mode = E.warm_mode_of(arm)
    n = E.warm_steps_of(arm)
    if mode == "warm":
        if n is not None:
            raise ValueError("exact warm references cannot override N")
        env.schedule.remaining_steps(E.warm_t_of(arm))
        return None
    source = "self" if mode.startswith("self") else "cache"
    base = mode[4:] if source == "self" else mode
    modes = dict(E.WARM_VARIANT_MODES)
    if env.policy == "groot":
        modes.update(E.GROOT_SHOOT_MODES)
    if base not in modes or (env.policy == "pi05" and mode == "selfwarmshoot"):
        raise ValueError(f"unsupported {env.policy} arm: {arm}")
    variant = modes[base]
    shoot = variant in E.SHOOT_ENTRY_T
    level = (
        E.SHOOT_ENTRY_T[variant]
        if shoot
        else E.MID_ENTRY_T_BY_VARIANT[variant][env.policy]
        if variant in E.MID_ENTRY_T_BY_VARIANT
        else 1.0
    )
    block = {
        "start": {
            "source": source,
            "point": "final" if "final" in base else "snapshot",
        },
        "grid": {
            "kind": "shoot" if shoot else "reset",
            "step_budget" if shoot else "entry_t": level,
        },
        "num_steps": n if n is not None else "remaining",
        "evidence_dir": evidence_dir,
    }
    if source == "self":
        block["self_seed"] = {
            "namespace": namespace,
            "identity_keys": ["experiment", "task", "orig_init_state_idx", "attempt"],
        }
    return block


def default_arms(env_id: str) -> list[str]:
    """Use the existing self-family roster, including its exact warm references."""
    env = E.ENVS[env_id]
    arms = (
        (*E.MACRO13_ARMS_BY_POLICY[env.policy], *E.SELF13_ARMS_BY_POLICY[env.policy])
        if env.benchmark == "robocasa365"
        else E.LIBERO_SELF_ARMS_BY_POLICY[env.policy]
    )
    return list(dict.fromkeys(a for a in arms if "_t" in a))


def validate_tasks(tasks: list[dict]) -> None:
    """Require a unique task roster with explicit original init-state identities."""
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a nonempty list")
    ids, names = set(), set()
    for task in tasks:
        tid, name, indices = task["task_id"], task["name"], task["init_indices"]
        if type(tid) is not int or tid < 0 or tid in ids:
            raise ValueError("task_id must be a unique nonnegative integer")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("task name must be nonempty and unique")
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(i) is not int or i < 0 for i in indices)
        ):
            raise ValueError("init_indices must contain nonnegative integers")
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate original init state")
        ids.add(tid)
        names.add(name)


def prepare(
    *,
    out: Path,
    env_id: str,
    base_yaml: Path,
    arms: list[str],
    tasks: list[dict],
    servers: list[str],
    evidence_root: str,
    namespace: str,
    rollout: dict,
) -> dict:
    """Create a fresh immutable run plan and reviewable YAML files; launch nothing."""
    from openpi.conductor.task import ServerEndpoint

    validate_tasks(tasks)
    env = E.ENVS[env_id]
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    if not namespace.strip() or not Path(evidence_root).is_absolute():
        raise ValueError(
            "namespace must be nonempty; server evidence root must be absolute"
        )
    if not arms or len(arms) != len(set(arms)):
        raise ValueError("arms must be nonempty and unique")
    endpoints = []
    for address in servers:
        host, port = address.rsplit(":", 1)
        if not host or not 0 < int(port) < 65536:
            raise ValueError(f"invalid server address: {address}")
        endpoints.append(ServerEndpoint(host, int(port)))
    if not endpoints or len(servers) != len(set(servers)):
        raise ValueError("servers must be nonempty and unique")
    if (
        type(rollout.get("replan_steps")) is not int
        or not 0 < rollout["replan_steps"] <= env.action_horizon
    ):
        raise ValueError("replan_steps must be within the action horizon")
    if env.benchmark == "robocasa365":
        for key in ("base_seed", "layout", "style"):
            if type(rollout.get(key)) is not int or rollout[key] < 0:
                raise ValueError(f"RoboCasa requires a nonnegative {key}")
    elif type(rollout.get("seed")) is not int or rollout["seed"] < 0:
        raise ValueError("LIBERO requires a nonnegative worker seed")
    cfg = config_of(base_yaml.read_text())
    base = dataclasses.asdict(cfg)
    if "${" in json.dumps(base):
        raise ValueError("base YAML contains unresolved environment variables")
    if (
        not cfg.enabled
        or effective_denoise_schedule(cfg).schedule_id != env.schedule_id
    ):
        raise ValueError(
            "base YAML enabled/schedule disagrees with the selected environment"
        )
    cp1 = cfg.checkpoints.get("cp1")
    if cp1 is None or not cp1.enabled or cp1.gate.type != "always_search":
        raise ValueError("base YAML requires enabled cp1 with always_search gate")
    if cfg.write_policy.type != "never":
        raise ValueError("base YAML must use write_policy: never (frozen library)")
    if cfg.backend.type != "in_memory" or not cfg.backend.in_memory.preload_path:
        raise ValueError("base YAML requires a frozen in_memory preload_path")
    if not Path(cfg.backend.in_memory.preload_path).is_absolute():
        raise ValueError("preload_path must be absolute on every server host")
    token = uuid.uuid4().hex[:12]
    evidence_dir = str(Path(evidence_root) / token)
    records = []
    for arm in arms:
        raw = copy.deepcopy(base)
        start_t = E.warm_t_of(arm)
        raw["checkpoints"]["cp1"]["judge"] = {
            "type": "always_warm_start",
            "start_t": start_t,
        }
        raw["warm_reset"] = arm_block(
            env_id, arm, evidence_dir=evidence_dir, namespace=namespace
        )
        text = yaml.safe_dump(raw, sort_keys=False)
        arm_cfg = config_of(text)
        spec = (
            WarmResetSpec.from_config(arm_cfg.warm_reset)
            if arm_cfg.warm_reset is not None
            else None
        )
        if spec is not None:
            resolve_plan(spec, env.schedule, start_t)
        records.append(
            {
                "arm": arm,
                "yaml_id": f"wr_{token}_{arm}",
                "yaml": text,
                "yaml_sha256": sha(text),
                "start_t": start_t,
                "spec_digest": spec.digest() if spec is not None else None,
            }
        )
    plan = {
        "schema": "warm_reset_run_v1",
        "token": token,
        "env_id": env_id,
        "schedule_id": env.schedule_id,
        "k": env.k_full,
        "tasks": tasks,
        "servers": [{"host": e.host, "port": e.port} for e in endpoints],
        "evidence_dir": evidence_dir,
        "namespace": namespace,
        "rollout": rollout,
        "arms": records,
    }
    out.mkdir(parents=True)
    (out / "yamls").mkdir()
    for arm in records:
        (out / "yamls" / f"{arm['yaml_id']}.yaml").write_text(
            arm["yaml"], encoding="utf-8"
        )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def read_plan(root: Path) -> dict:
    """Verify frozen YAML copies before scheduling or interpreting evidence."""
    plan = json.loads((root / "plan.json").read_text())
    if plan.get("schema") != "warm_reset_run_v1":
        raise ValueError("unsupported plan schema")
    validate_tasks(plan["tasks"])
    if (
        not plan["arms"]
        or len({a["yaml_id"] for a in plan["arms"]}) != len(plan["arms"])
        or len({a["arm"] for a in plan["arms"]}) != len(plan["arms"])
    ):
        raise ValueError("plan arms must be nonempty and unique")
    env = E.ENVS[plan["env_id"]]
    if (plan["schedule_id"], plan["k"]) != (env.schedule_id, env.k_full):
        raise ValueError("plan schedule changed")
    for arm in plan["arms"]:
        actual = (root / "yamls" / f"{arm['yaml_id']}.yaml").read_text()
        if actual != arm["yaml"] or sha(actual) != arm["yaml_sha256"]:
            raise ValueError(f"YAML changed after prepare: {arm['yaml_id']}")
        cfg = config_of(actual)
        if (
            effective_denoise_schedule(cfg).schedule_id != plan["schedule_id"]
            or cfg.checkpoints["cp1"].judge.start_t != arm["start_t"]
            or E.warm_t_of(arm["arm"]) != arm["start_t"]
        ):
            raise ValueError("arm schedule/start_t changed")
        spec = (
            WarmResetSpec.from_config(cfg.warm_reset)
            if cfg.warm_reset is not None
            else None
        )
        if (spec.digest() if spec else None) != arm["spec_digest"]:
            raise ValueError("spec digest changed")
    return plan
