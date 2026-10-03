"""Object-frame grasp alignment and recurrent carry-loss evidence (E3 T3/T4)."""

from __future__ import annotations

import numpy as np

from .adapters import relative_pose
from .common import Unavailable, auroc, fingerprint
from .forensics import physical, segments


def provenance(episode, index):
    d = episode.decision_at(index)
    return {
        k: d.get(k)
        for k in (
            "decision_id",
            "decision_seq",
            "src",
            "blind_age_controls",
            "rows",
            "weights",
            "anchor_decision_id",
            "chunk_offset",
        )
    }


def grasps(episode, config=None):
    adapter, data = physical(episode, config)
    episode.require("obj_quat", "eef_quat")
    closed = data["closed"]
    rows = []
    audited_objects = set()
    for relation in data["relations"]:
        k, goal = relation["k"], relation["goal"]
        if k in audited_objects:
            continue
        audited_objects.add(k)
        for t in np.flatnonzero(closed & ~np.r_[False, closed[:-1]] & data["active"]):
            pre = int(t) - 1
            if pre < 0:
                continue  # no invented reset pose
            delta = (
                episode.controls["eef_pos"][pre] - episode.controls["obj_pos"][pre, k]
            )
            if np.linalg.norm(delta) >= float(adapter.setting("near_radius")):
                continue
            xyz, yaw = relative_pose(
                adapter,
                episode.controls["eef_pos"][pre],
                episode.controls["eef_quat"][pre],
                episode.controls["obj_pos"][pre, k],
                episode.controls["obj_quat"][pre, k],
            )
            horizon = int(adapter.setting("lift_window"))
            lifted = bool(relation["carried"][t : min(episode.n, t + horizon)].any())
            complete = int(t) + horizon <= episode.n or bool(episode.outcome["success"])
            rows.append(
                dict(
                    episode.identity,
                    status="available",
                    object=goal.object,
                    predicate=goal.predicate,
                    close_control=int(t),
                    pre_control=pre,
                    object_frame_xyz=xyz.tolist(),
                    object_frame_yaw=yaw,
                    xy_offset=float(np.linalg.norm(xyz[:2])),
                    distance=float(np.linalg.norm(xyz)),
                    finger_width=adapter.width(pre),
                    lift_within_window=lifted if lifted or complete else None,
                    window_controls=horizon,
                    observed_window_controls=min(horizon, episode.n - int(t)),
                    outcome_status="available" if lifted or complete else "unavailable",
                    outcome_reason=""
                    if lifted or complete
                    else "terminally truncated lift window",
                    adapter_hash=adapter.fingerprint,
                    reset_state_sha256=episode.meta.get("reset_state_sha256"),
                    **provenance(episode, int(t)),
                )
            )
    return rows


def grasp_references(rows, reference_arms):
    """Reference successful policy grasps on the exact task/init and object."""
    reference = {}
    for row in rows:
        if (
            row.get("arm") in reference_arms
            and row.get("success")
            and row.get("lift_within_window")
        ):
            key = (row.get("suite"), row["task_id"], row["init"], row["object"])
            reference.setdefault(key, []).append(row)
    for row in rows:
        if row.get("status") != "available":
            continue
        key = (row.get("suite"), row["task_id"], row["init"], row["object"])
        refs = [
            x
            for x in reference.get(key, [])
            if x["episode_key"] != row["episode_key"]
            and x.get("env_seed") is not None
            and x.get("env_seed") == row.get("env_seed")
            and x.get("reset_state_sha256") is not None
            and x.get("reset_state_sha256") == row.get("reset_state_sha256")
        ]
        row["reference_count"] = len(refs)
        row["reference_status"] = "available" if refs else "unavailable"
        row["reference_reason"] = (
            ""
            if refs
            else "no successful reference-policy grasp for this task/init/object"
        )
        row["reference_xyz_error"] = min(
            (
                float(
                    np.linalg.norm(
                        np.asarray(row["object_frame_xyz"]) - x["object_frame_xyz"]
                    )
                )
                for x in refs
            ),
            default=None,
        )
        row["reference_yaw_error"] = min(
            (
                float(
                    abs(
                        np.arctan2(
                            np.sin(row["object_frame_yaw"] - x["object_frame_yaw"]),
                            np.cos(row["object_frame_yaw"] - x["object_frame_yaw"]),
                        )
                    )
                )
                for x in refs
            ),
            default=None,
        )
    # Descriptive two-cluster atlas: fit discovery successful reference grasps only.
    groups = {}
    for row in rows:
        if (
            row.get("arm") in reference_arms
            and row.get("lift_within_window")
            and row.get("success")
            and int(row["init"]) < 30
        ):
            groups.setdefault(
                (row.get("suite"), row["task_id"], row["object"]), []
            ).append(row)
    atlas = []
    for key, group in groups.items():
        if len(group) < 4:
            continue
        x = np.array(
            [
                r["object_frame_xyz"]
                + [np.sin(r["object_frame_yaw"]), np.cos(r["object_frame_yaw"])]
                for r in group
            ]
        )
        centers = x[[0, np.argmax(np.linalg.norm(x - x[0], axis=1))]].copy()
        for _ in range(50):
            labels = np.argmin(
                np.linalg.norm(x[:, None] - centers[None], axis=2), axis=1
            )
            new = np.array(
                [
                    x[labels == i].mean(0) if (labels == i).any() else centers[i]
                    for i in range(2)
                ]
            )
            if np.allclose(new, centers):
                break
            centers = new
        fit_hash = fingerprint(
            {"centers": centers, "episodes": sorted(r["episode_key"] for r in group)}
        )
        for i in range(2):
            atlas.append(
                {
                    "suite": key[0],
                    "task_id": key[1],
                    "object": key[2],
                    "cluster": i,
                    "center_xyz_sin_cos": centers[i].tolist(),
                    "support": int((labels == i).sum()),
                    "fit_hash": fit_hash,
                    "feature_tier": "privileged",
                    "status": "available",
                }
            )
    valid = [
        r
        for r in rows
        if r.get("lift_within_window") is not None and r.get("status") == "available"
    ]
    summary = {
        "attempts_with_observed_outcome": len(valid),
        "failed_attempts": sum(not r["lift_within_window"] for r in valid),
        "offset_auroc": auroc(
            [r["xy_offset"] for r in valid],
            [not r["lift_within_window"] for r in valid],
        )
        if valid
        else None,
        "auroc_denominator": len(valid),
        "reference_arms": reference_arms,
    }
    return atlas, summary


def drops(episode, config=None):
    adapter, data = physical(episode, config)
    rows = []
    for relation in data["relations"]:
        k, goal = relation["k"], relation["goal"]
        for start, end in segments(relation["carried"]):
            if end == episode.n or relation["predicate"][end]:
                continue
            distance = relation["dest_distance"]
            if distance is not None and distance[end] < float(
                adapter.setting("place_radius")
            ):
                continue
            if distance is None:
                status, mechanism = "unavailable", "unknown_release"
            else:
                status = "available"
                mechanism = (
                    "premature_open_command"
                    if not data["closed"][end]
                    else "closed_command_slip_candidate"
                )
            dt = episode.meta.get("control_dt", episode.manifest.get("control_dt"))
            accel = None
            if dt is not None and float(dt) > 0 and end >= 2:
                pos = episode.controls["eef_pos"][end - 2 : end + 1]
                accel = float(
                    np.linalg.norm(pos[2] - 2 * pos[1] + pos[0]) / float(dt) ** 2
                )
            contacts, contacts_status, contacts_reason = [], "available", ""
            try:
                contacts = adapter.contacts(end)
            except Unavailable as exc:
                contacts_status, contacts_reason = "unavailable", str(exc)
            rows.append(
                dict(
                    episode.identity,
                    status=status,
                    reason=""
                    if status == "available"
                    else "destination frame unavailable",
                    object=goal.object,
                    carry_start=start,
                    loss_control=end,
                    mechanism=mechanism,
                    gripper_command=float(
                        episode.controls["action"][
                            end, int(adapter.setting("gripper_dim"))
                        ]
                    ),
                    close_command=bool(data["closed"][end]),
                    finger_width=adapter.width(end - 1),
                    lift_height=float(
                        episode.controls["obj_pos"][end - 1, k, 2]
                        - data["reference"][k, 2]
                    ),
                    eef_acceleration=accel,
                    acceleration_status="available"
                    if accel is not None
                    else "unavailable",
                    dest_xy_at_loss=float(distance[end])
                    if distance is not None
                    else None,
                    contacts_status=contacts_status,
                    contacts_reason=contacts_reason,
                    named_contacts_at_loss=contacts,
                    mechanism_confidence="heuristic; command alone cannot certify a slip",
                    **provenance(episode, end),
                )
            )
    return rows
