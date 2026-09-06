"""Pin-table primitives shared by the drivers, the worker, and the auditor.

Every consumer of a pin table has to agree on three things or the identity
checks are theatre: how a mjcf path is normalized, how the canonical JSON is
spelled, and what gets hashed. They live here so there is exactly one spelling
of each.

Vocabulary (frozen by the plan, §2 D5):

``pinned_objects``  ``{task_name: {slot_name: "objects/<...>/model.xml"}}``.
                    Paths are asset-relative because the three machines keep
                    their asset trees at different prefixes.
``pin_id``          the whole table's identity: what an artifact records and a
                    config expects.
``pin_task_id``     one task's identity. A worker only ever receives its own
                    task's slice, so this is what it can verify unaided.

Both are sha256 over canonical JSON, each under its own domain separator: a
one-task table would otherwise hash identically under both, and the two
identities are only worth having if they are independent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# The marker that separates an asset tree's root from the part that is stable
# across machines. robocasa's own rebasing door (env_utils.create_obj) splits on
# exactly this, so we normalize the same way or the two disagree.
_OBJECTS_MARKER = "/objects/"


def normalize_mjcf_path(mjcf_path: str) -> str:
    """Reduce an absolute or relative mjcf path to its machine-independent tail.

    Raises:
        ValueError: if the path has no ``/objects/`` segment. Such a path cannot
            be rebased by robocasa either, so accepting it here would only move
            the failure to a place with less context.
    """
    path = mjcf_path.replace("\\", "/")
    if _OBJECTS_MARKER not in f"/{path}":
        raise ValueError(f"mjcf path has no {_OBJECTS_MARKER!r} segment: {mjcf_path!r}")
    tail = f"/{path}".split(_OBJECTS_MARKER)[-1]
    return f"objects/{tail}"


def canonical_json(payload: Any) -> str:
    """The one spelling of canonical JSON used for every pin hash.

    Matches ``run_collect.compute_plan_hash`` so a reader comparing the two
    hashes is not also comparing two serializers.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# Domain separators. Without them a single-task table hashes to the same string
# under both functions, and a worker that verified only its own slice would
# satisfy the global check by coincidence -- which defeats the purpose of having
# two independent identities.
_PIN_TABLE_DOMAIN = "robocasa365/pin_table/v1"
_PIN_TASK_DOMAIN = "robocasa365/pin_task/v1"


def compute_pin_id(pinned_objects: dict[str, dict[str, str]]) -> str:
    payload = {"domain": _PIN_TABLE_DOMAIN, "pinned_objects": pinned_objects}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def compute_pin_task_id(task_name: str, slot_map: dict[str, str]) -> str:
    payload = {"domain": _PIN_TASK_DOMAIN, "task": task_name, "slots": slot_map}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def load_pin_manifest(path: str | Path) -> tuple[str, dict[str, dict[str, str]]]:
    """Load a pin table and prove it is self-consistent.

    The file carries its own ``pin_id``; we recompute it rather than trust it,
    so a hand-edited table cannot travel under the identity of the table it was
    edited from.

    Returns:
        ``(pin_id, pinned_objects)``.

    Raises:
        ValueError: on a missing key or a declared identity that does not match
            the contents.
    """
    data = json.loads(Path(path).read_text())
    for key in ("pin_id", "pinned_objects"):
        if key not in data:
            raise ValueError(f"pin manifest {path} is missing {key!r}")
    pinned_objects = data["pinned_objects"]
    recomputed = compute_pin_id(pinned_objects)
    if recomputed != data["pin_id"]:
        raise ValueError(
            f"pin manifest {path} declares pin_id {data['pin_id']} but its "
            f"contents hash to {recomputed}"
        )
    return recomputed, pinned_objects


# The authoritative evaluation roster for the pinned PickPlace line (plan D1/D15).
# Ordered: the position is the task_id in every run-plan built from it, so a
# reordering is a different experiment, not a cosmetic change.
PNP_ROSTER: tuple[str, ...] = (
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
)

# Frozen arm shapes. The totals are derived, but stating them makes an arithmetic
# slip in either factor visible instead of silently rescaling the experiment.
PNP_CACHE_ARM = {"cells": 132, "episodes_per_task": 8, "total": 5_280}
PNP_TEACHER_ARM = {"cells": 1, "episodes_per_task": 50, "total": 250}


def resolve_manifest_path(path: str | Path) -> str:
    """Absolute, existing path to a pin manifest.

    Drivers forward this string to workers that run with a DIFFERENT cwd (the
    external RoboCasa checkout, not the openpi repo), so a relative path that
    opens fine on the driver silently points at nothing in the child. Resolving
    once, on the driver, is the only place that knows the right base.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"pin manifest {path!r} does not resolve to a file ({resolved})")
    return str(resolved)


def assert_pnp_eval_identity(
    tasks: list[tuple[str, int]],
    *,
    cells: int,
    arm: dict[str, int],
    label: str,
) -> None:
    """Refuse to dispatch an evaluation whose shape is not the frozen one.

    Both evaluation drivers default to the 13-task ``DEFAULT_EVAL_TASKS``, and
    membership checks alone let an explicit-but-wrong subset through: five tasks
    in the wrong order relabels every ``task_id``, and seven episodes per task
    quietly reshapes the budget. The gate is on the DESIGN (roster order, cell
    count, trials per task) rather than on the dispatched subset, so resuming a
    partial run with ``--only`` stays possible.

    Raises:
        ValueError: on any deviation, naming the specific one.
    """
    names = [name for name, _ in tasks]
    if tuple(names) != PNP_ROSTER:
        raise ValueError(
            f"{label}: task roster {names} is not the frozen ordered roster "
            f"{list(PNP_ROSTER)}"
        )
    counts = {n for _, n in tasks}
    if counts != {arm["episodes_per_task"]}:
        raise ValueError(
            f"{label}: episodes per task {sorted(counts)} != frozen "
            f"{arm['episodes_per_task']}"
        )
    if cells != arm["cells"]:
        raise ValueError(f"{label}: cell count {cells} != frozen {arm['cells']}")
    total = cells * sum(n for _, n in tasks)
    if total != arm["total"]:
        raise ValueError(f"{label}: budget {total} episodes != frozen {arm['total']}")


def assert_pnp_run_plan_identity(
    run_plans: list[dict[str, Any]],
    *,
    arm: dict[str, int],
    pin_id: str,
    label: str,
) -> None:
    """Validate the serialized graphs that a formal pinned eval will dispatch.

    ``assert_pnp_eval_identity`` protects parsed CLI inputs, including on an
    agent-only process that never constructs a graph.  This second gate is
    intentionally downstream of ``build_run_plan``: it proves that the actual
    serialized episode identities still have the frozen roster, ranges and
    aggregate budget before a driver starts serving work.
    """
    if len(run_plans) != arm["cells"]:
        raise ValueError(
            f"{label}: run-plan count {len(run_plans)} != frozen cell count {arm['cells']}"
        )

    expected_tasks = [
        {
            "task_name": name,
            "task_id": task_id,
            "episode_lo": 0,
            "episode_hi": arm["episodes_per_task"] - 1,
        }
        for task_id, name in enumerate(PNP_ROSTER)
    ]
    expected_suffixes = [
        ("eval", task_id, episode_idx)
        for task_id in range(len(PNP_ROSTER))
        for episode_idx in range(arm["episodes_per_task"])
    ]

    all_uids: list[str] = []
    plan_hashes: list[str] = []
    for index, plan in enumerate(run_plans):
        params = plan.get("params", {})
        if params.get("pin_id") != pin_id:
            raise ValueError(
                f"{label}: run-plan {index} pin_id {params.get('pin_id')!r} "
                f"!= manifest {pin_id!r}"
            )
        if params.get("tasks") != expected_tasks:
            raise ValueError(
                f"{label}: run-plan {index} task ranges differ from the frozen "
                f"roster/range {expected_tasks}"
            )

        uids = plan.get("uids")
        if not isinstance(uids, list):
            raise ValueError(f"{label}: run-plan {index} has no uid list")
        suffixes: list[tuple[str, int, int]] = []
        for uid in uids:
            try:
                yaml_id, phase, task_id_text, episode_idx_text = str(uid).rsplit(":", 3)
                task_id = int(task_id_text)
                episode_idx = int(episode_idx_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label}: malformed task uid {uid!r}") from exc
            if task_id not in range(len(PNP_ROSTER)):
                raise ValueError(f"{label}: task uid {uid!r} has out-of-range task_id")
            task_name = yaml_id.rsplit("__", 1)[-1]
            if task_name != PNP_ROSTER[task_id]:
                raise ValueError(
                    f"{label}: task uid {uid!r} names {task_name!r} at task_id "
                    f"{task_id}, expected {PNP_ROSTER[task_id]!r}"
                )
            suffixes.append((phase, task_id, episode_idx))
        if suffixes != expected_suffixes:
            raise ValueError(
                f"{label}: run-plan {index} serialized episode order/ranges "
                "differ from the frozen dispatch graph"
            )
        if set(plan.get("prefixes", {})) != set(uids):
            raise ValueError(f"{label}: run-plan {index} prefixes do not cover exactly its uids")
        all_uids.extend(uids)
        plan_hashes.append(str(plan.get("plan_hash", "")))

    if len(all_uids) != arm["total"]:
        raise ValueError(
            f"{label}: serialized budget {len(all_uids)} episodes != frozen {arm['total']}"
        )
    if len(set(all_uids)) != len(all_uids):
        raise ValueError(f"{label}: task uids are not unique across run-plans")
    if not all(plan_hashes) or len(set(plan_hashes)) != len(plan_hashes):
        raise ValueError(f"{label}: run-plan hashes are missing or not unique across cells")


def realized_objects_of(env: Any) -> dict[str, str]:
    """Read back which mesh instance each slot ACTUALLY got, for this episode.

    Call after ``env.reset()`` — ``_create_objects`` has run by then and has
    written the sampled ``info`` back onto every cfg, the implicit container
    included.

    ``gym.make`` hands back a wrapper, and attribute forwarding through
    Gymnasium wrappers is not a stable contract, so we unwrap first — the same
    thing ``build_bucket_variants.object_class_of`` does for the same reason.

    Returns:
        ``{slot_name: "objects/<...>/model.xml"}``.
    """
    inner = getattr(env, "unwrapped", env)
    cfgs = getattr(inner, "object_cfgs", None)
    if cfgs is None:
        raise ValueError(
            "env exposes no object_cfgs; realized provenance cannot be read "
            "(was this called before reset, or on a non-kitchen env?)"
        )
    realized: dict[str, str] = {}
    for obj_num, cfg in enumerate(cfgs):
        name = cfg.get("name", f"obj_{obj_num + 1}")
        info = cfg.get("info") or {}
        mjcf_path = info.get("mjcf_path")
        if mjcf_path is None:
            raise ValueError(f"slot {name!r} has no info.mjcf_path after reset")
        realized[name] = normalize_mjcf_path(mjcf_path)
    return realized
