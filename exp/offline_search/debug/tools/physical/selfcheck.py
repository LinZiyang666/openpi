"""T9: catalog integrity, post-settle support, seed and successor redundancy."""

from __future__ import annotations

import numpy as np

from .adapters import EnvironmentAdapter, entries
from .common import (
    load_config,
    Unavailable,
    load_episodes,
    parser,
    parallel_map,
    report,
    safe_rows,
)


def supported_objects(adapter, mapping, pairs):
    """Ground a graph of free roots at every non-free contact body.

    Collision children belong to their nearest free ancestor. A floating cycle
    of free objects has no ground and therefore cannot support itself.
    """
    catalog = adapter.entities
    movable = entries(catalog, "movable")
    free_bodies = {
        int(body)
        for body, kind in zip(
            catalog.get("joint_bodyid", []), catalog.get("joint_type", [])
        )
        if int(kind) == 0
    }
    if "joint_type" not in catalog:
        free_bodies = {
            int(x["body_id"])
            for x in movable
            if x.get("body_id") is not None
            and x.get("role", "object") not in ("articulation", "fixture", "robot")
        }
    parents = catalog.get("body_parentid", [])

    def free_root(body):
        visited = set()
        while body not in visited:
            if body in free_bodies:
                return body
            visited.add(body)
            if not parents or body == 0:
                return None
            if not 0 <= body < len(parents):
                raise Unavailable("body parent catalog is incomplete")
            parent = int(parents[body])
            if parent == body:
                return None
            body = parent
        raise Unavailable("cyclic body parent catalog")

    graph = {body: set() for body in free_bodies}
    supported = set()
    for left, right in pairs:
        a = free_root(mapping[int(left)]["body_id"])
        b = free_root(mapping[int(right)]["body_id"])
        if a is None and b is not None:
            supported.add(b)
        elif b is None and a is not None:
            supported.add(a)
        elif a is not None and b is not None and a != b:
            graph[a].add(b)
            graph[b].add(a)
    pending = list(supported)
    while pending:
        body = pending.pop()
        for other in graph[body] - supported:
            supported.add(other)
            pending.append(other)
    resting = [
        x
        for x in movable
        if x.get("body_id") in free_bodies and x.get("rests_on_table", True)
    ]
    missing = sorted(x["name"] for x in resting if int(x["body_id"]) not in supported)
    return resting, supported, missing


def check(episode, config=None):
    rows = []

    def record(name, status, **values):
        rows.append(dict(episode.identity, check=name, status=status, **values))

    try:
        episode.require("is_settle")
        record("control_contiguity", "available", passed=True, n_controls=episode.n)
    except Unavailable as exc:
        record("control_contiguity", "unavailable", passed=False, reason=str(exc))
        return rows
    adapter = EnvironmentAdapter(episode, config)
    seed = episode.meta.get("env_seed")
    record(
        "environment_seed",
        "available" if seed is not None else "unavailable",
        passed=seed is not None,
        reason="" if seed is not None else "environment seed not logged",
    )
    try:
        ref, index, source = adapter.reference()
        record(
            "post_settle_reference",
            "available",
            passed=True,
            reference_control=index,
            source=source,
        )
    except Unavailable as exc:
        record("post_settle_reference", "unavailable", passed=False, reason=str(exc))
        index = None
    try:
        mapping = adapter.contact_map()
        c = episode.controls
        if "contact_off" not in c or "contact_geom" not in c:
            raise Unavailable("contact arrays not captured")
        off, geom = np.asarray(c["contact_off"]), np.asarray(c["contact_geom"])
        if (
            off.shape != (episode.n + 1,)
            or off[0] != 0
            or (np.diff(off) < 0).any()
            or off[-1] != len(geom)
        ):
            raise Unavailable("invalid flat contact offsets")
        if geom.ndim != 2 or geom.shape[1] != 2:
            raise Unavailable("invalid contact geom pair shape")
        missing = sorted(set(int(x) for x in geom.flat) - set(mapping))
        if missing:
            raise Unavailable("contact geom IDs missing from catalog: " + str(missing))
        record(
            "geom_naming",
            "available",
            passed=True,
            geoms=len(mapping),
            contact_pairs=len(geom),
        )
        if index is None or index < 0:
            raise Unavailable(
                "settled contacts unavailable; no after-settle reference sample"
            )
        resting, supported, unsupported = supported_objects(
            adapter, mapping, geom[int(off[index]) : int(off[index + 1])]
        )
        record(
            "resting_objects_contact_table",
            "available",
            passed=not unsupported,
            reference_control=index,
            denominator=len(resting),
            missing_objects=unsupported,
            supported_free_body_ids=sorted(supported),
            support_rule="contact chain to any non-free body",
        )
    except Unavailable as exc:
        record("geom_naming_or_support", "unavailable", passed=False, reason=str(exc))
    if "before_qpos" in episode.controls and "qpos" in episode.controls:
        before, after = episode.controls["before_qpos"], episode.controls["qpos"]
        equal = np.all(before[1:] == after[:-1], axis=1)
        record(
            "before_after_redundancy",
            "available",
            equal_controls=int(equal.sum()),
            denominator=len(equal),
            passed=bool(equal.all()),
        )
    else:
        record(
            "before_after_redundancy",
            "not_applicable",
            reason="schema stores each after-state once; before arrays absent",
        )
    return rows


def main(argv=None):
    args = parser("selfcheck").parse_args(argv)
    episodes = load_episodes(args.run_root, args.arms, args.p3v2, args.limit)
    config = load_config(args, episodes)
    rows = [
        row
        for group in parallel_map(
            lambda ep: safe_rows(check, ep, config), episodes, args.procs
        )
        for row in group
    ]
    report(
        args.out,
        "selfcheck",
        {"selfcheck": rows},
        {
            "accepted_episodes": len(episodes),
            "failed_checks": sum(r.get("passed") is False for r in rows),
        },
        episodes=episodes,
    )


if __name__ == "__main__":
    main()
