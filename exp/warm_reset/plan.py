"""Freeze arm YAMLs and task identities for the warm reset conductor entry.

Experiment choices live here; serving and conductor retain their generic APIs.
The arm vocabulary comes from the existing step_diag protocol; environments
come from the entry's registry (``exp.warm_reset.envs``).

Arm kinds (``arm_kind``):

* ``warm`` -- a warm-family arm (``warm_t*`` exact resume, cache / self warm
  reset / shoot) derived from ``--base-yaml`` (retrieval + frozen library);
* ``self_only`` -- a self warm-reset arm under ``self_trigger="always"``: a
  library-free ``warm_reset`` block with ``trigger: always`` and its own
  ``start_t``, no checkpoint, no library;
* ``miss`` -- ``full`` / ``plain_k<k>``: a library-free ``miss`` block with
  ``num_steps`` (Pi0.5 per bundle; GR00T on an endpoint started with that
  step count, ``k_servers``).

New plan keys are written only when used (``kind`` / ``steps`` / ``endpoints``
on an arm, ``server_steps``, ``experiment_id``, ``pin_id`` / ``pins``,
``init_pool_sha256``), and every reader defaults them, so plans prepared
before these options existed read, replan and admit unchanged.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import pathlib
import re
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import yaml

from exp.step_diag import envs as E
from exp.warm_reset.envs import EntryEnv, get_env
from openpi.cache.config import effective_denoise_schedule, load_cache_config
from openpi.cache.warm_reset.types import MissSpec, WarmResetSpec, resolve_plan

KIND_WARM = "warm"
KIND_SELF_ONLY = "self_only"
KIND_MISS = "miss"
_PLAIN = re.compile(r"plain_k([1-9][0-9]*)")


def sha(text: str) -> str:
    """Hash the exact UTF-8 YAML sent over the control channel."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_of(text: str):
    """Use the production loader without requiring server-local artifact paths."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as fh:
        fh.write(text)
        fh.flush()
        return load_cache_config(fh.name, check_files=False)


# ------------------------------------------------------------------
# Arm vocabulary
# ------------------------------------------------------------------


def miss_steps_of(env: EntryEnv, arm: str) -> Optional[int]:
    """MISS Euler steps of ``full`` / ``plain_k<k>``, or ``None`` for any other arm id."""
    if arm == "full":
        return env.k_full
    match = _PLAIN.fullmatch(arm)
    return int(match.group(1)) if match else None


def arm_kind(env: EntryEnv, arm: str, self_trigger: str = "verdict") -> str:
    """``miss`` / ``self_only`` / ``warm`` of an arm id (``ValueError`` if unknown)."""
    if miss_steps_of(env, arm) is not None:
        return KIND_MISS
    if not re.fullmatch(r"[a-z0-9]+_t[0-9.]+(?:_n[0-9]+)?", arm):
        raise ValueError(f"invalid arm id: {arm!r}")
    if self_trigger not in ("verdict", "always"):
        raise ValueError(f"self trigger must be 'verdict' or 'always', got {self_trigger!r}")
    if E.is_self_mode(E.warm_mode_of(arm)) and self_trigger == "always":
        return KIND_SELF_ONLY
    return KIND_WARM


def required_steps(env: EntryEnv, arm: str) -> int:
    """The live head step count an arm needs on GR00T (its MISS steps, else K)."""
    steps = miss_steps_of(env, arm)
    return env.k_full if steps is None else steps


def arm_block(
    env_id: str, arm: str, *, evidence_dir: str, namespace: str
) -> dict | None:
    """Translate the established warm-family arm vocabulary into a ``warm_reset`` block."""
    env = get_env(env_id)
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


def miss_block(env: EntryEnv, arm: str, *, evidence_dir: str) -> dict:
    """The ``miss`` block of ``full`` / ``plain_k<k>``."""
    steps = miss_steps_of(env, arm)
    if steps is None:
        raise ValueError(f"{arm!r} is not a full / plain_k<k> arm")
    return {"num_steps": steps, "evidence_dir": evidence_dir}


def library_free_base(env: EntryEnv) -> dict:
    """A config that retrieves nothing: every checkpoint disabled, no library.

    The key builder / keys only satisfy the validator (nothing is ever
    collected or searched); ``denoise_schedule`` names the environment's loop,
    which a GR00T self start is checked against.
    """
    return {
        "enabled": True,
        "timer": {"enabled": False},
        "keys": {
            **{f: {"enabled": False, "weight": 0.0} for f in ("vision_0", "vision_1", "vision_2", "prompt_emb")},
            "robot_state": {"enabled": True, "weight": 1.0},
        },
        "key_builder": {"type": "placeholder"},
        "checkpoints": {
            "cp1": {"enabled": False, "search_strategy": {"type": "weighted_rrf_knn", "top_k": 1}},
        },
        "backend": {"type": "in_memory", "vector_dims": {"robot_state": 32}},
        "write_policy": {"type": "never"},
        "denoise_schedule": env.schedule_id,
    }


def default_arms(env_id: str) -> list[str]:
    """The environment's ``--arms all`` roster (step_diag: its self-family warm arms)."""
    return list(get_env(env_id).default_arms)


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


# ------------------------------------------------------------------
# Prepare
# ------------------------------------------------------------------


def _endpoints(addresses: list[str]):
    from openpi.conductor.task import ServerEndpoint

    out = []
    for address in addresses:
        host, port = address.rsplit(":", 1)
        if not host or not 0 < int(port) < 65536:
            raise ValueError(f"invalid server address: {address}")
        out.append(ServerEndpoint(host, int(port)))
    return out


def _base_config(base_yaml: Path, env: EntryEnv) -> dict:
    """The retrieval base of warm arms, as a dataclass dict (checked as before)."""
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
    if base.get("miss") is None:
        # Written only when present: the arm YAML stays the pre-``miss`` text.
        base.pop("miss", None)
    return base


def _pool_digest(rollout: dict, init_pool_sha256: Optional[str]) -> Optional[str]:
    """``sha256_tree`` of a locally readable LIBERO pool, or the operator's digest."""
    pool = rollout.get("init_states_dir") or ""
    local = E.sha256_tree(pool) if pool and pathlib.Path(pool).is_dir() else None
    if local and init_pool_sha256 and local != init_pool_sha256:
        raise ValueError("--init-pool-sha256 disagrees with the pool readable at prepare")
    return local or init_pool_sha256 or None


def prepare(
    *,
    out: Path,
    env_id: str,
    base_yaml: Optional[Path],
    arms: list[str],
    tasks: list[dict],
    servers: list[str],
    evidence_root: str,
    namespace: str,
    rollout: dict,
    self_trigger: str = "verdict",
    k_servers: Optional[dict[int, list[str]]] = None,
    experiment_id: Optional[str] = None,
    pinned_objects: Optional[str] = None,
    init_pool_sha256: Optional[str] = None,
) -> dict:
    """Create a fresh immutable run plan and reviewable YAML files; launch nothing.

    ``base_yaml`` is required only by ``warm`` arms. ``k_servers`` (GR00T)
    maps a live step count to the endpoints started with it; every arm is then
    pinned to the endpoints of the count it needs (``required_steps``).
    ``experiment_id`` (RoboCasa) is the stable episode experiment of the run,
    ``pinned_objects`` (RoboCasa PnP) the pin manifest whose per-task slot maps
    are frozen into the plan.
    """
    validate_tasks(tasks)
    env = get_env(env_id)
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    if not namespace.strip() or not Path(evidence_root).is_absolute():
        raise ValueError(
            "namespace must be nonempty; server evidence root must be absolute"
        )
    if not arms or len(arms) != len(set(arms)):
        raise ValueError("arms must be nonempty and unique")
    kinds = {arm: arm_kind(env, arm, self_trigger) for arm in arms}
    endpoints = _endpoints(servers)
    if not endpoints or len(servers) != len(set(servers)):
        raise ValueError("servers must be nonempty and unique")
    if (
        type(rollout.get("replan_steps")) is not int
        or not 0 < rollout["replan_steps"] <= env.action_horizon
    ):
        raise ValueError("replan_steps must be within the action horizon")
    env.adapter.validate_rollout(env, rollout)
    extra_keys: dict = {}
    if experiment_id is not None:
        if not env.adapter.supports_experiment_id:
            raise ValueError(f"--experiment-id is not supported by the {env.adapter.name} adapter")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", experiment_id):
            raise ValueError("experiment id must be a nonempty [A-Za-z0-9._-] string")
        extra_keys["experiment_id"] = experiment_id
    if pinned_objects:
        if not env.adapter.supports_pinned_objects:
            raise ValueError(f"--pinned-objects is not supported by the {env.adapter.name} adapter")
        from exp.robocasa365.pinned_objects import (
            load_pin_manifest,
            resolve_manifest_path,
        )

        pin_id, table = load_pin_manifest(resolve_manifest_path(pinned_objects))
        missing = [t["name"] for t in tasks if t["name"] not in table]
        if missing:
            raise ValueError(f"pin manifest has no slot map for tasks {missing}")
        extra_keys.update(pin_id=pin_id, pins={t["name"]: dict(table[t["name"]]) for t in tasks})
    if rollout.get("init_states_dir") or init_pool_sha256:
        extra_keys["init_pool_sha256"] = _pool_digest(rollout, init_pool_sha256)

    # Per-step-count endpoints (GR00T's MISS count is process-level).
    server_steps: dict[str, int] = {}
    if k_servers:
        if env.policy != "groot":
            raise ValueError("per-step-count endpoints are a GR00T option; Pi0.5 sets MISS steps per bundle")
        server_steps = dict.fromkeys(servers, env.k_full)
        for k, addresses in sorted(k_servers.items()):
            if type(k) is not int or k < 1 or k == env.k_full or not addresses:
                raise ValueError(f"k-servers need a step count other than K={env.k_full} and endpoints")
            for address in addresses:
                if address in server_steps:
                    raise ValueError(f"endpoint {address} is listed for two step counts")
                server_steps[address] = k
        endpoints += _endpoints([a for a in server_steps if a not in servers])
    if env.policy == "groot":
        for arm in arms:
            need = required_steps(env, arm)
            if need != env.k_full and need not in server_steps.values():
                raise ValueError(
                    f"GR00T arm {arm} needs a server started with --denoising-steps {need} "
                    "(prepare --k-servers)"
                )

    base = None
    if any(kind == KIND_WARM for kind in kinds.values()):
        if base_yaml is None:
            raise ValueError("warm-family (library) arms require --base-yaml")
        base = _base_config(base_yaml, env)
    token = uuid.uuid4().hex[:12]
    evidence_dir = str(Path(evidence_root) / token)
    records = []
    for arm in arms:
        kind = kinds[arm]
        record: dict = {"arm": arm, "yaml_id": f"wr_{token}_{arm}"}
        if kind == KIND_WARM:
            raw = copy.deepcopy(base)
            start_t = E.warm_t_of(arm)
            raw["checkpoints"]["cp1"]["judge"] = {
                "type": "always_warm_start",
                "start_t": start_t,
            }
            raw["warm_reset"] = arm_block(
                env_id, arm, evidence_dir=evidence_dir, namespace=namespace
            )
        elif kind == KIND_SELF_ONLY:
            raw = library_free_base(env)
            start_t = E.warm_t_of(arm)
            raw["warm_reset"] = {
                **arm_block(env_id, arm, evidence_dir=evidence_dir, namespace=namespace),
                "trigger": "always",
                "start_t": start_t,
            }
        else:
            raw = library_free_base(env)
            start_t = None
            raw["miss"] = miss_block(env, arm, evidence_dir=evidence_dir)
        text = yaml.safe_dump(raw, sort_keys=False)
        arm_cfg = config_of(text)
        if kind == KIND_MISS:
            spec = MissSpec.from_config(arm_cfg.miss)
        else:
            spec = (
                WarmResetSpec.from_config(arm_cfg.warm_reset)
                if arm_cfg.warm_reset is not None
                else None
            )
            if spec is not None:
                resolve_plan(spec, env.schedule, start_t)
        record.update(
            yaml=text,
            yaml_sha256=sha(text),
            start_t=start_t,
            spec_digest=spec.digest() if spec is not None else None,
        )
        if kind != KIND_WARM:
            record["kind"] = kind
        if kind == KIND_MISS:
            record["steps"] = spec.num_steps
        if server_steps:
            need = required_steps(env, arm)
            record["endpoints"] = [a for a, k in server_steps.items() if k == need]
        records.append(record)
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
        **extra_keys,
    }
    if server_steps:
        plan["server_steps"] = server_steps
    out.mkdir(parents=True)
    (out / "yamls").mkdir()
    for arm in records:
        (out / "yamls" / f"{arm['yaml_id']}.yaml").write_text(
            arm["yaml"], encoding="utf-8"
        )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


# ------------------------------------------------------------------
# Read
# ------------------------------------------------------------------


def arm_kind_of(record: dict) -> str:
    """The frozen kind of a plan's arm record (pre-registry plans: ``warm``)."""
    return record.get("kind", KIND_WARM)


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
    env = get_env(plan["env_id"])
    if (plan["schedule_id"], plan["k"]) != (env.schedule_id, env.k_full):
        raise ValueError("plan schedule changed")
    addresses = {f"{s['host']}:{s['port']}" for s in plan["servers"]}
    for arm in plan["arms"]:
        actual = (root / "yamls" / f"{arm['yaml_id']}.yaml").read_text()
        if actual != arm["yaml"] or sha(actual) != arm["yaml_sha256"]:
            raise ValueError(f"YAML changed after prepare: {arm['yaml_id']}")
        cfg = config_of(actual)
        kind = arm_kind_of(arm)
        if not set(arm.get("endpoints", addresses)) <= addresses:
            raise ValueError("arm endpoints are not plan servers")
        if kind == KIND_MISS:
            steps = miss_steps_of(env, arm["arm"])
            if (
                cfg.miss is None
                or cfg.warm_reset is not None
                or arm["start_t"] is not None
                or steps is None
                or arm.get("steps") != steps
                or cfg.miss.num_steps != steps
                or MissSpec.from_config(cfg.miss).digest() != arm["spec_digest"]
            ):
                raise ValueError(f"MISS arm changed: {arm['arm']}")
            continue
        if (
            effective_denoise_schedule(cfg).schedule_id != plan["schedule_id"]
            or E.warm_t_of(arm["arm"]) != arm["start_t"]
        ):
            raise ValueError("arm schedule/start_t changed")
        if kind == KIND_SELF_ONLY:
            spec = WarmResetSpec.from_config(cfg.warm_reset) if cfg.warm_reset is not None else None
            if spec is None or not spec.always or spec.start_t != arm["start_t"] or not spec.self_start:
                raise ValueError(f"library-free self arm changed: {arm['arm']}")
        else:
            if cfg.checkpoints["cp1"].judge.start_t != arm["start_t"]:
                raise ValueError("arm schedule/start_t changed")
            spec = (
                WarmResetSpec.from_config(cfg.warm_reset)
                if cfg.warm_reset is not None
                else None
            )
        if (spec.digest() if spec else None) != arm["spec_digest"]:
            raise ValueError("spec digest changed")
    return plan
