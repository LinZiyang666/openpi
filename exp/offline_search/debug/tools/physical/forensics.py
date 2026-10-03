"""T1 failure taxonomy and T2 descriptive simulator-truth timelines.

Adapted from E3 forensics.py; labels remain explicit kinematic heuristics,
not validated causal diagnoses. All positions refer to after-control samples.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .adapters import EnvironmentAdapter, predicate_tokens
from .common import (
    load_config,
    RULE_VERSION,
    Unavailable,
    load_episodes,
    parser,
    parallel_map,
    report,
)


def first(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def segments(mask):
    mask = np.asarray(mask, bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def physical(episode, config=None):
    episode.require("eef_pos", "obj_pos", "predicates", "is_settle", "decision_seq")
    adapter = EnvironmentAdapter(episode, config)
    goals = adapter.goals()
    if not goals:
        raise Unavailable("no captured success sub-predicates")
    predicates = np.asarray(episode.controls["predicates"], float)
    if predicates.ndim != 2 or predicates.shape[1] != len(adapter.predicates):
        raise Unavailable("predicate vector does not match catalog")
    ref, reference_index, reference_status = adapter.reference()
    closed = adapter.closed()
    n = episode.n
    active = ~episode.controls["is_settle"].astype(bool)
    stage = np.full(n, "unknown", dtype="<U24")
    stage[~active] = "settle"
    details = []
    relation_data = []
    for goal in goals:
        if not 0 <= goal.predicate < predicates.shape[1]:
            raise Unavailable("goal refers to unknown predicate index")
        ps = (predicates[:, goal.predicate] > 0.5) & active
        info = {
            "predicate": goal.predicate,
            "relation": goal.relation,
            "object": goal.object,
            "destination": goal.destination,
            "final": bool(ps[-1]),
            "ever": bool(ps.any()),
            "first_true": first(ps),
            "label": "ok" if ps[-1] else "fixture_not_done",
            "onset_control": None,
        }
        if not goal.object:
            if not ps[-1] and ps.any():
                info.update(
                    label="fixture_undone",
                    onset_control=int(np.flatnonzero(ps[:-1] & ~ps[1:])[-1] + 1),
                )
            details.append(info)
            continue
        if goal.object not in adapter.name_to_index:
            info.update(label="unavailable", reason="goal object pose not captured")
            details.append(info)
            continue
        k = adapter.name_to_index[goal.object]
        obj = episode.controls["obj_pos"][:, k]
        distance = np.linalg.norm(episode.controls["eef_pos"] - obj, axis=1)
        near = (distance < float(adapter.setting("near_radius"))) & active
        lifted = (
            obj[:, 2] - ref[k, 2] > float(adapter.setting("lift_height"))
        ) & active
        carried = lifted & near & closed
        carry_segments = segments(carried)
        destination = adapter.destination(goal)
        dest_distance = (
            np.linalg.norm(obj[:, :2] - destination[:, :2], axis=1)
            if destination is not None
            else None
        )
        info.update(
            first_near=first(near),
            first_lift=first(carried),
            carry_segments=carry_segments,
            dmin=float(distance[active].min()),
            carried_controls=int(carried.sum()),
            close_near=int((closed & ~np.r_[False, closed[:-1]] & near).sum()),
        )
        if ps[-1]:
            pass
        elif ps.any():
            info.update(
                label="undone",
                onset_control=int(np.flatnonzero(ps[:-1] & ~ps[1:])[-1] + 1),
            )
        elif not near.any():
            info.update(label="never_reached")
        elif not carried.any():
            info.update(label="grasp_miss", onset_control=first(near))
        elif carry_segments[-1][1] == n:
            info.update(label="held_not_placed", onset_control=carry_segments[-1][0])
        else:
            release = carry_segments[-1][1]
            info["dest_xy_release"] = (
                float(dest_distance[release]) if dest_distance is not None else None
            )
            if dest_distance is None:
                info.update(
                    label="unknown_release",
                    onset_control=release,
                    reason="destination frame unavailable; drop/misplacement ambiguous",
                )
            elif dest_distance[release] < float(adapter.setting("place_radius")):
                info.update(label="misplace", onset_control=release)
            else:
                info.update(
                    label="drop_regrasp_fail" if len(carry_segments) > 1 else "drop",
                    onset_control=carry_segments[0][1],
                )
        details.append(info)
        relation_data.append(
            {
                "goal": goal,
                "k": k,
                "near": near,
                "lifted": lifted,
                "carried": carried,
                "distance": distance,
                "predicate": ps,
                "dest_distance": dest_distance,
            }
        )
    for t in np.flatnonzero(active):
        unsatisfied = [x for x in relation_data if not x["predicate"][t]]
        if unsatisfied:
            x = min(unsatisfied, key=lambda x: (not x["carried"][t], x["distance"][t]))
            released = t > 0 and x["carried"][t - 1] and not x["carried"][t]
            if released:
                stage[t] = "release"
            elif x["carried"][t]:
                stage[t] = (
                    "place"
                    if x["dest_distance"] is not None
                    and x["dest_distance"][t] < float(adapter.setting("place_radius"))
                    else "carry"
                )
            elif x["near"][t] and not x["lifted"][t]:
                stage[t] = "grasp_window"
            else:
                stage[t] = "approach"
        elif (predicates[t] > 0.5).all():
            stage[t] = "release" if closed[t] else "retreat"
        else:
            stage[t] = "fixture"
    return adapter, {
        "stage": stage,
        "details": details,
        "relations": relation_data,
        "reference_index": reference_index,
        "reference_status": reference_status,
        "reference": ref,
        "active": active,
        "closed": closed,
    }


def analyse(episode, config=None):
    adapter, data = physical(episode, config)
    failing = [x for x in data["details"] if x["label"] != "ok"]
    failing.sort(
        key=lambda x: (
            x.get("first_near") is None,
            x.get("first_near") or 0,
            x["predicate"],
        )
    )
    success = episode.outcome["success"]
    if success is None:
        raise Unavailable("journal outcome unavailable")
    label = (
        "success"
        if success
        else failing[0]["label"]
        if failing
        else "all_preds_true_but_fail"
    )
    onset = failing[0]["onset_control"] if failing and not success else None
    wrong, disturbed = [], []
    excluded = {g.object for g in adapter.goals()} | {
        g.destination for g in adapter.goals()
    }
    excluded |= {token for p in adapter.predicates for token in predicate_tokens(p)[1:]}
    excluded |= {
        adapter.names[adapter.name_to_index[x]]
        for x in list(excluded)
        if x in adapter.name_to_index
    }
    for relation in data["relations"]:
        dp = adapter.destination(relation["goal"])
        if dp is not None:
            for k, name in enumerate(adapter.names):
                if np.array_equal(dp, episode.controls["obj_pos"][:, k]):
                    excluded.add(name)
    for k, name in enumerate(adapter.names):
        if name in excluded:
            continue
        obj = episode.controls["obj_pos"][:, k]
        lift = obj[:, 2] - data["reference"][k, 2] > float(
            adapter.setting("lift_height")
        )
        near = np.linalg.norm(obj - episode.controls["eef_pos"], axis=1) < float(
            adapter.setting("near_radius")
        )
        t = first(lift & near & data["active"])
        if t is not None:
            wrong.append({"object": name, "control": t})
        if np.linalg.norm(obj[-1] - data["reference"][k]) > float(
            adapter.setting("disturbance_radius")
        ):
            disturbed.append(name)
    if label == "never_reached" and wrong:
        label, onset = "wrong_object", min(x["control"] for x in wrong)
    tail = episode.controls["eef_pos"][data["active"]][
        -int(adapter.setting("tail_controls")) :
    ]
    path = float(np.linalg.norm(np.diff(tail, axis=0), axis=1).sum())
    net = float(np.linalg.norm(tail[-1] - tail[0]))
    motion = (
        "stall"
        if path < float(adapter.setting("stall_path"))
        else "oscillation"
        if path > float(adapter.setting("oscillation_path"))
        and net / path < float(adapter.setting("oscillation_ratio"))
        else "moving"
    )
    decision = episode.decision_at(onset) if onset is not None else {}
    row = dict(
        episode.identity,
        status="available",
        label=label,
        onset_control=onset,
        onset_decision=decision.get("decision_seq"),
        onset_decision_id=decision.get("decision_id"),
        onset_decision_status="available"
        if onset is not None and decision.get("decision_id")
        else "not_applicable"
        if onset is None
        else "unavailable",
        onset_resolution="after_control",
        onset_interval=[onset - 1, onset] if onset is not None else None,
        onset_confidence="heuristic",
        truth_validated=False,
        rule_version=RULE_VERSION,
        adapter_hash=adapter.fingerprint,
        adapter_settings=adapter.settings,
        n_controls=episode.n,
        active_controls=int(data["active"].sum()),
        reference_control=data["reference_index"],
        reference_status=data["reference_status"],
        subgoals_done=sum(x["final"] for x in data["details"]),
        subgoals=len(data["details"]),
        predicate_detail=data["details"],
        wrong_objects=wrong,
        disturbed=disturbed,
        tail_motion=motion,
        tail_path=path,
        tail_net=net,
    )
    return row, data


def timeline_rows(episode, config=None):
    row, data = analyse(episode, config)
    controls = []
    for i, stage in enumerate(data["stage"]):
        decision = episode.decision_at(i)
        controls.append(
            dict(
                episode.identity,
                control_idx=i,
                decision_seq=decision.get("decision_seq"),
                decision_id=decision.get("decision_id"),
                truth_stage=str(stage),
                src=decision.get("src", decision.get("source")),
                src_status="available"
                if decision.get("src", decision.get("source")) is not None
                else "not_applicable"
                if str(stage) == "settle"
                else "unavailable",
                blind_age_controls=decision.get("blind_age_controls"),
                status="available",
                rule_version=RULE_VERSION,
                evidence_time="after_control",
            )
        )
    decisions = []
    for decision in episode.decisions:
        seq = int(decision["decision_seq"])
        hits = np.flatnonzero(episode.controls["decision_seq"] == seq)
        pre = episode.before_index(decision)
        counts = Counter(str(x) for x in data["stage"][hits])
        decisions.append(
            dict(
                episode.identity,
                decision_id=decision.get("decision_id"),
                decision_seq=seq,
                truth_stage=counts.most_common(1)[0][0] if counts else "unavailable",
                truth_stage_pre=str(data["stage"][pre])
                if 0 <= pre < episode.n
                else "unavailable",
                pre_control_idx=pre,
                applied_controls=len(hits),
                stage_controls=dict(counts),
                src=decision.get("src", decision.get("source")),
                status="available",
                rule_version=RULE_VERSION,
                truth_validated=False,
            )
        )
    return row, controls, decisions


def build(episodes, config=None, procs=4):
    def one(ep):
        try:
            return timeline_rows(ep, config)
        except (Unavailable, ValueError, KeyError, IndexError) as exc:
            return dict(ep.identity, status="unavailable", reason=str(exc)), [], []

    output = parallel_map(one, episodes, procs)
    return {
        "episodes_forensics": [x[0] for x in output],
        "controls_truth": [r for x in output for r in x[1]],
        "decisions_truth": [r for x in output for r in x[2]],
    }


def main(argv=None):
    args = parser("forensics").parse_args(argv)
    episodes = load_episodes(args.run_root, args.arms, args.p3v2, args.limit)
    config = load_config(args, episodes)
    tables = build(episodes, config, args.procs)
    labels = Counter(
        r.get("label", "unavailable") for r in tables["episodes_forensics"]
    )
    report(
        args.out,
        "forensics",
        tables,
        {
            "accepted_episodes": len(episodes),
            "labels": dict(labels),
            "labels_validated": False,
            "onsets_file": "episodes_forensics.csv",
        },
        episodes=episodes,
    )


if __name__ == "__main__":
    main()
