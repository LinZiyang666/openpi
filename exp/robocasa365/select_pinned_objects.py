"""Pick one exact mesh instance per object slot, for the PickPlace pin experiment.

Why this is not a hand-written table: pinning a slot by exact ``.xml`` path takes
``sample_kitchen_object_helper``'s deterministic branch
(``kitchen_object_utils.py:361-382``), and that branch skips every filter the
random branch applies — ``exclude_obj_groups``, the seven capability flags, and
the ``obj_instance_split`` slice. Nothing downstream notices. So a pin that a
human eyeballed can quietly seat a pretrain-split instance, or a non-graspable
object in a slot the task expects to grasp.

This script therefore REPLAYS the random branch's constraints
(``kitchen_object_utils.py:383-473``) and only ever picks from the set that
branch could legally have produced. The constraints are read off the task's own
cfgs rather than transcribed, so a slot cannot be forgotten and the implicit
``obj_container`` is covered like any other.

Run it on a machine that has the asset tree (the three island-A hosts); the
local weiland-wsl checkout has no ``robocasa/models/assets/objects``.

Usage::

    python -m exp.robocasa365.select_pinned_objects \\
        --layout 1 --style 1 \\
        --out exp/robocasa365/config/pnp_pinned_objects.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from exp.robocasa365.pinned_objects import (
    compute_pin_id,
    normalize_mjcf_path,
)

# The exact-5 roster (plan D1/D15). Order is the authority: it is the task_id
# order in every run-plan built from it.
DEFAULT_TASKS = (
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
)

# The capability flags the random branch checks, in the order it checks them.
CAPABILITY_FLAGS = (
    "graspable",
    "washable",
    "microwavable",
    "cookable",
    "fridgable",
    "freezable",
    "dishwashable",
)


def legal_categories(
    groups: Any,
    exclude_groups: Any,
    flags: dict[str, Any],
    obj_registries: tuple[str, ...],
) -> list[str]:
    """Replay ``kitchen_object_utils.py:383-438``'s category filter.

    Note the capability rule is ANY-registry-fails, not any-registry-passes: a
    category is dropped if it fails the flag in *any* registry that carries it.
    """
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS

    if not isinstance(groups, (tuple, list)):
        groups = [groups]
    if exclude_groups is None:
        exclude_groups = []
    if not isinstance(exclude_groups, (tuple, list)):
        exclude_groups = [exclude_groups]

    invalid_categories = [cat for g in exclude_groups for cat in OBJ_GROUPS[g]]

    valid: list[str] = []
    for g in groups:
        for cat in OBJ_GROUPS[g]:
            if cat in valid or cat in invalid_categories:
                continue
            if not any(reg in OBJ_CATEGORIES[cat] for reg in obj_registries):
                continue
            bad = False
            for reg in obj_registries:
                if reg not in OBJ_CATEGORIES[cat]:
                    continue
                meta = OBJ_CATEGORIES[cat][reg]
                for flag in CAPABILITY_FLAGS:
                    if flags.get(flag) is True and getattr(meta, flag) is not True:
                        bad = True
            if bad:
                continue
            valid.append(cat)
    return valid


def split_slice(mjcf_paths: list[str], split: str | None) -> list[str]:
    """Replay the ``obj_instance_split`` slice (``:450-461``)."""
    if split is None:
        return list(mjcf_paths)
    split_th = max(len(mjcf_paths) - 5, int(math.ceil(len(mjcf_paths) / 2)))
    if split == "pretrain":
        return list(mjcf_paths[:split_th])
    if split == "target":
        return list(mjcf_paths[split_th:])
    raise ValueError(f"unknown split {split!r}")


def object_extent(mjcf_path: str, scale: float) -> list[float]:
    """The object's bounding extent, exactly as ``sample_kitchen_object`` computes it."""
    from robosuite.utils.mjcf_utils import find_elements, string_to_array

    root = ET.parse(mjcf_path).getroot()
    bbox = find_elements(root=root, tags="geom", attribs={"name": "reg_bbox"})
    if bbox is None:
        raise ValueError(f"{mjcf_path} has no reg_bbox geom; max_size is uncheckable")
    half = string_to_array(bbox.get("size"))
    return list((half * 2) * scale)


def candidates_for_slot(
    cfg: dict[str, Any],
    obj_registries: tuple[str, ...],
    split: str,
) -> list[tuple[str, str, str]]:
    """Every instance the random branch could legally have produced for this slot.

    Returns:
        ``[(category, registry, absolute mjcf path), ...]``, deterministically
        ordered.
    """
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES

    # A missing obj_groups key means "all" — env_utils.create_obj:1429 supplies
    # that default, so a slot can be unconstrained without saying so.
    groups = cfg.get("obj_groups", "all")
    flags = {flag: cfg.get(flag) for flag in CAPABILITY_FLAGS}
    rotate_upright = bool(cfg.get("rotate_upright", False))

    out: list[tuple[str, str, str]] = []
    for cat in legal_categories(groups, cfg.get("exclude_obj_groups"), flags, obj_registries):
        for reg in obj_registries:
            if reg not in OBJ_CATEGORIES[cat]:
                continue
            for path in split_slice(OBJ_CATEGORIES[cat][reg].mjcf_paths, split):
                if rotate_upright:
                    path = path.replace("model.xml", "model_upright.xml")
                out.append((cat, reg, path))
    return sorted(out, key=lambda t: normalize_mjcf_path(t[2]))


def ranked_choices(
    cfg: dict[str, Any],
    obj_registries: tuple[str, ...],
    split: str,
) -> list[dict[str, Any]]:
    """Legal instances for this slot, best-first.

    Ordered by how close the object's volume is to the median of the legal set.
    Size matters twice, in opposite directions. Too large and the placement
    sampler cannot fit it in its region -- and because a pin makes every one of
    robocasa's 50 ``_load_model`` retries draw the SAME object, what is normally
    a resample-and-move-on becomes an unrecoverable "could not initialize task".
    Too small and the target becomes needlessly hard to grasp, depressing the
    teacher for a reason unrelated to retrieval. The median is the least
    surprising point between those.

    Ties break on the normalized path so the ordering is reproducible.
    """
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES

    max_size = cfg.get("max_size") or (None, None, None)
    out: list[dict[str, Any]] = []
    for cat, reg, path in candidates_for_slot(cfg, obj_registries, split):
        if not Path(path).exists():
            # rotate_upright slots point at model_upright.xml, which only two
            # categories ship; a missing file means the replay produced a path
            # robocasa itself could not have loaded.
            continue
        scale = OBJ_CATEGORIES[cat][reg].get_mjcf_kwargs()["scale"]
        try:
            extent = object_extent(path, scale)
        except ValueError:
            continue
        if any(m is not None and extent[i] > m for i, m in enumerate(max_size)):
            continue
        out.append(
            {
                "name": cfg["name"],
                "mjcf_path": normalize_mjcf_path(path),
                "category": cat,
                "registry": reg,
                "scale": scale,
                "extent": extent,
                "volume": extent[0] * extent[1] * extent[2],
            }
        )
    if not out:
        raise ValueError(
            f"slot {cfg['name']!r}: no instance survives the replayed constraints"
        )
    volumes = sorted(c["volume"] for c in out)
    median = volumes[len(volumes) // 2]
    out.sort(key=lambda c: (abs(c["volume"] - median), c["mjcf_path"]))
    for rank, choice in enumerate(out):
        choice["rank"] = rank
        choice["candidates"] = len(out)
    return out


def slots_of(task_name: str, layout: int, style: int, seed: int) -> list[dict[str, Any]]:
    """Build the task once, unpinned, and read back its full slot list.

    Reading the cfgs instead of transcribing them is what makes the implicit
    ``obj_container`` (created inside ``_create_objects``, invisible in the task
    source) impossible to miss, and it carries each slot's own constraints along
    with it.
    """
    env = build_probe(task_name, layout, style, None)
    try:
        env.reset(seed=seed)
        inner = getattr(env, "unwrapped", env)
        return [dict(cfg) for cfg in inner.object_cfgs]
    finally:
        env.close()


def build_probe(task_name: str, layout: int, style: int, pinned: dict[str, str] | None):
    """Construct the task, optionally pinned. Caller owns closing it."""
    import gymnasium as gym
    import robocasa  # noqa: F401 - registers the robocasa/ namespace

    pin_kw = {} if pinned is None else {"pinned_objects": pinned}
    return gym.make(
        f"robocasa/{task_name}",
        split=None,
        obj_instance_split="target",
        layout_and_style_ids=[(layout, style)],
        **pin_kw,
    )


def task_builds(task_name: str, layout: int, style: int, pinned: dict[str, str], seeds: list[int]) -> str | None:
    """Return None if the pinned task initializes on every seed, else the reason.

    This is not belt-and-braces on top of the constraint replay -- it tests
    something the replay cannot see. Whether an object FITS its placement region
    is decided by the sampler at load time, and a pin turns robocasa's 50
    retries into 50 identical attempts, so an object that is merely a bit too
    large stops being "resample and move on" and becomes a task that can never
    start.
    """
    try:
        env = build_probe(task_name, layout, style, pinned)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return f"gym.make failed: {exc}"
    try:
        for seed in seeds:
            env.reset(seed=seed)
    except Exception as exc:  # noqa: BLE001
        return f"reset(seed={seed}) failed: {type(exc).__name__}: {exc}"
    finally:
        env.close()
    return None


def choose_for_task(
    task_name: str,
    cfgs: list[dict[str, Any]],
    registries: tuple[str, ...],
    split: str,
    layout: int,
    style: int,
    seeds: list[int],
    max_attempts: int,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Pick one instance per slot and prove the combination actually loads."""
    ranked = {cfg["name"]: ranked_choices(cfg, registries, split) for cfg in cfgs}
    cursors = {name: 0 for name in ranked}

    def materialize() -> list[dict[str, Any]]:
        # Category distinctness is enforced here rather than inside the ranking
        # because it is a property of the SET: the instruction names a category,
        # so two slots sharing one makes the prompt ambiguous about which object
        # in the scene it means.
        used: set[str] = set()
        picks: list[dict[str, Any]] = []
        for cfg in cfgs:
            name = cfg["name"]
            options = ranked[name]
            idx = cursors[name]
            while idx < len(options) and options[idx]["category"] in used:
                idx += 1
            if idx >= len(options):
                raise ValueError(
                    f"slot {name!r}: no remaining candidate with an unused category"
                )
            cursors[name] = idx
            used.add(options[idx]["category"])
            picks.append(options[idx])
        return picks

    for attempt in range(max_attempts):
        picks = materialize()
        slot_map = {pick["name"]: pick["mjcf_path"] for pick in picks}
        reason = task_builds(task_name, layout, style, slot_map, seeds)
        if reason is None:
            for pick in picks:
                pick["attempts"] = attempt + 1
            return slot_map, picks
        # Blame the largest object: placement failures scale with size, and
        # advancing the biggest slot is the move most likely to make room.
        biggest = max(picks, key=lambda pick: pick["volume"])
        print(
            f"  {task_name} attempt {attempt + 1}: {reason.splitlines()[0][:90]}"
            f" -> advancing {biggest['name']!r} past {biggest['mjcf_path']}",
            file=sys.stderr,
        )
        cursors[biggest["name"]] += 1
    raise ValueError(f"{task_name}: no pinned combination loaded in {max_attempts} attempts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--split", default="target", choices=("target", "pretrain"))
    ap.add_argument("--registries", default="objaverse,lightwheel")
    ap.add_argument("--probe-seed", type=int, default=0, help="seed for the one unpinned probe build per task")
    ap.add_argument(
        "--verify-seeds", default="1000000,1000001",
        help="seeds the pinned combination must initialize on before it is accepted",
    )
    ap.add_argument("--max-attempts", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = tuple(t for t in args.tasks.split(",") if t)
    registries = tuple(r for r in args.registries.split(",") if r)
    verify_seeds = [int(s) for s in args.verify_seeds.split(",") if s]

    pinned: dict[str, dict[str, str]] = {}
    provenance: dict[str, Any] = {}
    for task_name in tasks:
        cfgs = slots_of(task_name, args.layout, args.style, args.probe_seed)
        slot_map, picks = choose_for_task(
            task_name, cfgs, registries, args.split,
            args.layout, args.style, verify_seeds, args.max_attempts,
        )
        pinned[task_name] = slot_map
        provenance[task_name] = picks
        print(f"{task_name}: {len(slot_map)} slots -> {sorted(slot_map)}", file=sys.stderr)

    payload = {
        "pin_id": compute_pin_id(pinned),
        "pinned_objects": pinned,
        "provenance": {
            "split": args.split,
            "obj_registries": list(registries),
            "layout": args.layout,
            "style": args.style,
            "verify_seeds": verify_seeds,
            "slots": provenance,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    print(f"pin_id={payload['pin_id']}  ->  {out}")


if __name__ == "__main__":
    main()
