"""Geometry-driven cards and report artifacts without guessed simulator truth."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from exp.r17_integration.search.store import atomic_json


def decision_profile(binding, arrays):
    layout = binding["model"]["layout"]
    actions = np.asarray(arrays["normalized_actions"])
    active = layout.get("valid_action_indices") or list(range(layout["valid_action_dim"]))
    if actions.ndim != 2 or actions.shape[0] != layout["horizon"] or max(active) >= actions.shape[1]:
        raise ValueError("profile action/layout mismatch")
    values = actions[:, active]
    gripper = layout.get("gripper_channel")
    return dict(real_cameras=binding["model"]["feature_spec"]["real_cameras"],
                feature_widths=binding["model"]["feature_spec"]["fields"],
                horizon=layout["horizon"], active_channels=active,
                action_rms=float(np.sqrt(np.mean(values.astype(np.float64)**2))),
                gripper_channel=gripper, gripper_closed_convention=layout.get("gripper_closed_convention", "unspecified"),
                gripper_values=actions[:, gripper].tolist() if gripper is not None else None,
                units=layout.get("units", "unknown"), frame=layout.get("frame", "unknown"),
                native_grid=arrays["native_grid"].tolist() if "native_grid" in arrays else None,
                native_grid_status="available" if "native_grid" in arrays else "unsupported")


def physical_profile(record, arrays):
    caps = record.get("capabilities", {})
    result = dict(schema="r17.physical_profile.v1", capabilities=caps, raw_arrays={
        name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in arrays.items()}, tools={})
    for tool, capability in (("forensics", "contact_force"), ("grasp", "predicates"),
                             ("drop", "predicates"), ("goal", "goal"), ("restore", "restore")):
        if capability == "predicates" and "task_predicates" in caps:
            capability = "task_predicates"
        cap = caps.get(capability, dict(status="unsupported", reason="native semantics not provided"))
        result["tools"][tool] = dict(status=cap["status"], reason=cap.get("reason", ""),
                                     analysis_status="raw_only" if tool == "forensics" else "pending_semantic_analyzer")
    result["tools"]["cards"] = dict(status="available", reason="numeric physical/control card only")
    # These are observed magnitudes, not grasp/drop predicates.
    if "contact_force" in arrays:
        force = np.asarray(arrays["contact_force"], np.float64)
        finite = force[np.isfinite(force)]
        result["observed_max_contact_force"] = float(np.max(np.abs(finite))) if finite.size else None
    result["success"] = record.get("physical_metadata", {}).get("success")
    return result


def write_report(report, directory, *, decision_cards=(), physical_cards=()):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    value = dict(report, decision_cards=list(decision_cards), physical_cards=list(physical_cards))
    atomic_json(directory / "report.json", value)
    rows = [dict(task=task, **record) for task, record in report["per_task"].items()]
    with (directory / "tasks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "completed", "successes", "sr", "difficulty"])
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(figsize=(8, 4))
    available = [row for row in rows if row["sr"] is not None]
    axes.bar(range(len(available)), [row["sr"] for row in available])
    axes.set_ylim(0, 1)
    axes.set_xticks(range(len(available)), [row["task"] for row in available], rotation=45, ha="right")
    axes.set_ylabel("Accepted episode success rate")
    fig.tight_layout()
    fig.savefig(directory / "success.png", dpi=120)
    plt.close(fig)
    text = html.escape(json.dumps(value, indent=2, allow_nan=False))
    (directory / "report.html").write_text("<!doctype html><meta charset=utf-8><title>R17 analysis</title>"
                                           "<h1>R17 accepted episode analysis</h1><img src=success.png alt='Per-task success rate'>"
                                           "<pre>" + text + "</pre>")
    return dict(json=str(directory / "report.json"), csv=str(directory / "tasks.csv"),
                html=str(directory / "report.html"), png=str(directory / "success.png"))
