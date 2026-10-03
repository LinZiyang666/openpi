"""Strict physically matched pairs and separately event-aligned divergence."""

from __future__ import annotations

import numpy as np

from .adapters import EnvironmentAdapter
from .common import Unavailable, cluster_interval, auroc
from .forensics import analyse, first


def model(episode):
    value = episode.meta.get("model", episode.manifest.get("model", ""))
    if isinstance(value, dict):
        value = value.get("name", value.get("model", value.get("type", "")))
    return str(value).lower()


def paired_episodes(episodes, reference_arm):
    groups = {}
    for ep in episodes:
        key = (
            model(ep),
            ep.meta.get("suite"),
            ep.meta.get("task_id"),
            ep.meta.get("init"),
        )
        groups.setdefault(key, []).append(ep)
    result = []
    for values in groups.values():
        refs = [ep for ep in values if ep.meta.get("arm") == reference_arm]
        others = [ep for ep in values if ep.meta.get("arm") != reference_arm]
        if not others:
            result += [
                (ep, None, "no comparison arm captured for this task/init")
                for ep in refs
            ]
            continue
        if len(refs) != 1:
            for ep in others:
                result.append(
                    (
                        ep,
                        None,
                        "expected exactly one accepted reference attempt for this task/init",
                    )
                )
        else:
            result += [(ep, refs[0], "") for ep in others]
    return result


def validate_pair(a, b):
    for ep in (a, b):
        ep.require("action", "qpos", "qvel", "eef_pos", "is_settle")
    for name in ("env_seed", "reset_state_sha256"):
        if a.meta.get(name) is None or b.meta.get(name) is None:
            raise Unavailable("physical pairing requires " + name)
        if a.meta[name] != b.meta[name]:
            raise Unavailable(name + " differs; physically comparable pairing rejected")
    sa, sb = a.controls["is_settle"].astype(bool), b.controls["is_settle"].astype(bool)
    na, nb = int(sa.sum()), int(sb.sum())
    if na != nb or not np.array_equal(sa[:na], sb[:nb]):
        raise Unavailable("settling schedule differs")
    for name in ("qpos", "qvel", "action"):
        if a.controls[name].shape[1:] != b.controls[name].shape[1:]:
            raise Unavailable(name + " dimensions differ")
        if not bit_equal(a.controls[name][:na], b.controls[name][:nb]):
            raise Unavailable(
                "settling " + name + " differs before issued active controls"
            )


def bit_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def first_bit_difference(a, b):
    n = min(len(a), len(b))
    if a.shape[1:] != b.shape[1:] or a.dtype != b.dtype:
        raise Unavailable("bit comparison has incompatible arrays")
    x = np.ascontiguousarray(a[:n]).view(np.uint8).reshape(n, -1)
    y = np.ascontiguousarray(b[:n]).view(np.uint8).reshape(n, -1)
    return first(np.any(x != y, axis=1))


def relation_events(data):
    result = {}
    for x in data["relations"]:
        for name, mask in (
            ("near", x["near"]),
            ("lift", x["carried"]),
            ("goal", x["predicate"]),
        ):
            index = first(mask)
            if index is not None:
                result[(x["goal"].object, name)] = index
    return result


def compare(a, b, config=None, envelope=None, twin=False, scales=None):
    validate_pair(a, b)
    if twin and (model(a) != "groot" or model(b) != "groot"):
        raise Unavailable(
            "bit-identical twin analysis admitted only for recorded GR00T model"
        )
    n = min(a.n, b.n)
    action_first = first_bit_difference(a.controls["action"], b.controls["action"])
    state_first = first_bit_difference(a.controls["qpos"], b.controls["qpos"])
    velocity_first = first_bit_difference(a.controls["qvel"], b.controls["qvel"])
    perturbed = action_first
    vision_first = None
    for i in range(n):
        da, db = a.decision_at(i), b.decision_at(i)
        if (
            da.get("vision") is not None
            and db.get("vision") is not None
            and bool(da["vision"]) != bool(db["vision"])
        ):
            vision_first = i
            break
    if vision_first is not None:
        perturbed = (
            min(perturbed, vision_first) if perturbed is not None else vision_first
        )
    if twin:
        end = perturbed if perturbed is not None else n
        if not bit_equal(
            a.controls["qpos"][:end], b.controls["qpos"][:end]
        ) or not bit_equal(a.controls["qvel"][:end], b.controls["qvel"][:end]):
            raise Unavailable(
                "GR00T pair is not bit-identical before the first lever/action perturbation"
            )
        if (
            "obj_pos" in a.controls
            and "obj_pos" in b.controls
            and not bit_equal(a.controls["obj_pos"][:end], b.controls["obj_pos"][:end])
        ):
            raise Unavailable("object histories differ before perturbation")
    adapter_a, adapter_b = EnvironmentAdapter(a, config), EnvironmentAdapter(b, config)
    common_objects = sorted(set(adapter_a.names) & set(adapter_b.names))
    eef_gap = np.linalg.norm(
        a.controls["eef_pos"][:n] - b.controls["eef_pos"][:n], axis=1
    )
    object_gap = None
    rotation_gap = None
    if common_objects and "obj_pos" in a.controls and "obj_pos" in b.controls:
        xa = a.controls["obj_pos"][
            :n, [adapter_a.name_to_index[x] for x in common_objects]
        ]
        xb = b.controls["obj_pos"][
            :n, [adapter_b.name_to_index[x] for x in common_objects]
        ]
        object_gap = np.linalg.norm(xa - xb, axis=2).max(axis=1)
        if "obj_quat" in a.controls and "obj_quat" in b.controls:
            qa = a.controls["obj_quat"][
                :n, [adapter_a.name_to_index[x] for x in common_objects]
            ]
            qb = b.controls["obj_quat"][
                :n, [adapter_b.name_to_index[x] for x in common_objects]
            ]
            ra, rb = adapter_a.quat_matrix(qa), adapter_b.quat_matrix(qb)
            relative = np.einsum("nkij,nkjl->nkil", np.swapaxes(ra, -1, -2), rb)
            rotation_gap = np.arccos(
                np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
            ).max(axis=1)
    envelope = envelope or {}
    eef_threshold, object_threshold = (
        envelope.get("eef_gap_threshold"),
        envelope.get("object_gap_threshold"),
    )
    object_onset = (
        first(object_gap > object_threshold)
        if object_gap is not None and object_threshold is not None
        else None
    )
    onset_status = (
        "available"
        if object_gap is not None and object_threshold is not None
        else "unavailable"
    )
    truth_a, truth_b = None, None
    try:
        label_a, truth_a = analyse(a, config)
        label_b, truth_b = analyse(b, config)
    except Unavailable:
        label_a, label_b = {}, {}
    identity = dict(
        a.identity, arm_reference=b.meta.get("arm"), episode_key_reference=b.key
    )
    rec = dict(
        identity,
        status="available",
        twin=twin,
        shared_controls=n,
        first_action_difference=action_first,
        first_state_difference=state_first,
        first_velocity_difference=velocity_first,
        first_vision_difference=vision_first,
        perturbation_control=perturbed,
        perturbation_decision=a.decision_at(perturbed).get("decision_seq")
        if perturbed is not None
        else None,
        first_eef_envelope_exit=first(eef_gap > eef_threshold)
        if eef_threshold is not None
        else None,
        eef_envelope_status="available" if eef_threshold is not None else "unavailable",
        object_divergence_onset=object_onset,
        object_onset_status=onset_status,
        object_onset_reason=""
        if onset_status == "available"
        else "requires object poses and a frozen calibrated envelope",
        first_object_bit_difference=first(object_gap > 0)
        if object_gap is not None
        else None,
        stage_at_object_onset=str(truth_a["stage"][object_onset])
        if truth_a is not None and object_onset is not None
        else None,
        outcome_flip=bool(a.outcome["success"] != b.outcome["success"]),
        success_reference=b.outcome["success"],
        label=label_a.get("label"),
        label_reference=label_b.get("label"),
        envelope=envelope,
    )
    rec["sigma_diagnostics_status"] = "unavailable"
    if perturbed is not None:
        delta = a.controls["action"][perturbed] - b.controls["action"][perturbed]
        rec["action_gap_at_perturbation"] = float(np.sqrt(np.mean(delta**2)))
        rec["lever_blind"] = a.decision_at(perturbed).get("vision") is False
        if scales is not None and a.reader_arm is not None and b.reader_arm is not None:
            try:
                da, db = a.decision_at(perturbed), b.decision_at(perturbed)
                ha = a.reader_arm.decision_arrays(
                    ["served_chunk"], [da["decision_id"]]
                )["served_chunk"][0]
                hb = b.reader_arm.decision_arrays(
                    ["served_chunk"], [db["decision_id"]]
                )["served_chunk"][0]
                dims = scales["action_dims"]
                length = min(5, len(ha), len(hb))
                gap = (ha[:length, dims] - hb[:length, dims]) / scales["action_sigma"][
                    dims
                ]
                if not np.isfinite(gap).all():
                    raise Unavailable("served normalized chunks unavailable")
                rec.update(
                    action_gap_sigma_rms=float(np.sqrt(np.mean(gap**2))),
                    sigma_diagnostics_status="available",
                )
                grip = scales["gripper_dim"]
                rec["gripper_mode_difference"] = bool(
                    np.sign(ha[:length, grip].mean())
                    != np.sign(hb[:length, grip].mean())
                )
                by_seq_a = {int(d["decision_seq"]): d for d in a.decisions}
                by_seq_b = {int(d["decision_seq"]): d for d in b.decisions}
                for lag in (1, 2, 4, 8, 16, 32):
                    seq = int(da["decision_seq"]) + lag
                    if seq not in by_seq_a or seq not in by_seq_b:
                        continue
                    try:
                        sa = a.reader_arm.decision_arrays(
                            ["state_norm"], [by_seq_a[seq]["decision_id"]]
                        )["state_norm"][0]
                        sb = b.reader_arm.decision_arrays(
                            ["state_norm"], [by_seq_b[seq]["decision_id"]]
                        )["state_norm"][0]
                        width = min(len(sa), len(sb), len(scales["state_sigma"]))
                        active = scales["state_active"][:width]
                        delta = (sa[:width] - sb[:width]) / scales["state_sigma"][
                            :width
                        ]
                        if active.any() and np.isfinite(delta[active]).all():
                            rec["state_gap_sigma_lag%d" % lag] = float(
                                np.sqrt(np.mean(delta[active] ** 2))
                            )
                    except KeyError:
                        continue
            except (KeyError, Unavailable, ValueError, IndexError) as exc:
                rec["sigma_diagnostics_reason"] = str(exc)
    curves, event_curves = [], []
    for i in range(n):
        delta = a.controls["qpos"][i] - b.controls["qpos"][i]
        curves.append(
            dict(
                identity,
                status="available",
                control_idx=i,
                controls_since_perturbation=i - perturbed
                if perturbed is not None
                else None,
                eef_gap=float(eef_gap[i]),
                qpos_rms=float(np.sqrt(np.mean(delta**2))),
                object_gap=float(object_gap[i]) if object_gap is not None else None,
                object_rotation_gap=float(rotation_gap[i])
                if rotation_gap is not None
                else None,
                object_gap_status="available"
                if object_gap is not None
                else "unavailable",
            )
        )
    if truth_a is not None and truth_b is not None:
        ea, eb = relation_events(truth_a), relation_events(truth_b)
        rec["first_event_difference"] = next(
            (
                "%s:%s" % key
                for key in sorted(
                    set(ea) | set(eb), key=lambda x: min(ea.get(x, a.n), eb.get(x, b.n))
                )
                if ea.get(key) != eb.get(key)
            ),
            None,
        )
        for key in sorted(set(ea) & set(eb)):
            ta, tb = ea[key], eb[key]
            for offset in range(-20, 41):
                ia, ib = ta + offset, tb + offset
                if not (0 <= ia < a.n and 0 <= ib < b.n):
                    continue
                gap = float(
                    np.linalg.norm(
                        a.controls["eef_pos"][ia] - b.controls["eef_pos"][ib]
                    )
                )
                pose_gap = None
                if (
                    key[0] in adapter_a.name_to_index
                    and key[0] in adapter_b.name_to_index
                ):
                    pose_gap = float(
                        np.linalg.norm(
                            a.controls["obj_pos"][ia, adapter_a.name_to_index[key[0]]]
                            - b.controls["obj_pos"][ib, adapter_b.name_to_index[key[0]]]
                        )
                    )
                event_curves.append(
                    dict(
                        identity,
                        status="available",
                        object=key[0],
                        event=key[1],
                        event_offset=offset,
                        control_a=ia,
                        control_reference=ib,
                        eef_gap=gap,
                        object_gap=pose_gap,
                        truth_stage=str(truth_a["stage"][ia]),
                        truth_stage_reference=str(truth_b["stage"][ib]),
                    )
                )
    return rec, curves, event_curves


def build(episodes, reference_arm, config=None, envelope=None, twin=False, scales=None):
    rows, curves, event_curves = [], [], []
    comparisons = paired_episodes(episodes, reference_arm)
    for a, b, error in comparisons:
        try:
            if b is None:
                raise Unavailable(error)
            rec, curve, events = compare(a, b, config, envelope, twin, scales)
            rows.append(rec)
            curves.extend(curve)
            event_curves.extend(events)
        except (Unavailable, KeyError, ValueError, IndexError) as exc:
            rows.append(
                dict(
                    a.identity,
                    arm_reference=reference_arm,
                    status="unavailable",
                    reason=str(exc),
                )
            )
    valid = [r for r in rows if r["status"] == "available"]
    transition = {}
    for r in valid:
        key = "%s->%s:%s" % (
            bool(r["success_reference"]),
            bool(r["success"]),
            r.get("stage_at_object_onset") or "unknown",
        )
        transition[key] = transition.get(key, 0) + 1
    summary = {
        "accepted_episodes": len(episodes),
        "matched_pairs": sum(b is not None for _, b, _ in comparisons),
        "comparison_rows": len(rows),
        "admitted_pairs": len(valid),
        "transitions_by_stage": transition,
        "outcome_flip_interval": cluster_interval(valid, "outcome_flip"),
    }
    for field in ["action_gap_sigma_rms"] + [
        "state_gap_sigma_lag%d" % lag for lag in (1, 2, 4, 8, 16, 32)
    ]:
        supported = [r for r in valid if field in r]
        if supported:
            summary[field] = {
                "denominator": len(supported),
                "p50": float(np.median([r[field] for r in supported])),
                "p90": float(np.quantile([r[field] for r in supported], 0.9)),
                "auroc_outcome_flip": auroc(
                    [r[field] for r in supported],
                    [r["outcome_flip"] for r in supported],
                ),
            }
    return {
        "pairs": rows,
        "divergence_curves": curves,
        "event_aligned": event_curves,
    }, summary


def library_scales(root):
    """E4 successful-library action/state sigma; constants excluded."""
    import json
    from pathlib import Path

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    success = np.load(root / "success.npy", mmap_mode="r", allow_pickle=False).astype(
        bool
    )
    if not success.any():
        raise Unavailable("no successful library rows for sigma diagnostics")
    action = np.load(root / "action.npy", mmap_mode="r", allow_pickle=False)
    rs = np.load(root / "rs.npy", mmap_mode="r", allow_pickle=False)
    valid = int(manifest["act_valid_dims"])
    grip = int(manifest["gripper_dim"])
    action_sigma = (
        action[success, : int(manifest["exec_steps"]), :]
        .reshape(-1, action.shape[-1])
        .std(0)
    )
    dims = [
        i for i in range(valid) if i != grip and action_sigma[i] > np.finfo(float).eps
    ]
    state_sigma = rs[success, : int(manifest["rs_valid_dims"])].std(0)
    state_active = state_sigma > np.finfo(float).eps
    if not dims or not state_active.any():
        raise Unavailable("no varying valid library action/state coordinates")
    state_sigma = np.where(state_active, state_sigma, 1.0)
    return {
        "action_sigma": action_sigma,
        "action_dims": dims,
        "gripper_dim": grip,
        "state_sigma": state_sigma,
        "state_active": state_active,
    }
