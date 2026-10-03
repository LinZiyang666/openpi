"""Frozen discovery fits, retrospective boundaries, and prefix-only scores.

G0 reproduces R7's median-head/two-means/three-row-majority rule. The causal
version confirms that majority one row late. Stop/waypoint/change-point signals
use only portable robot measurements, never physical goal labels or outcomes.
"""

from __future__ import annotations

import numpy as np

from .adapters import EnvironmentAdapter
from .common import Unavailable, fingerprint


def two_means(values):
    """R7 stages.two_means, kept dependency-free (deterministic quartile init)."""
    x = np.asarray(values, float)
    centers = np.quantile(x, [0.25, 0.75])
    if centers[0] == centers[1]:
        centers = np.array([x.min(), x.max()])
    for _ in range(100):
        modes = x > centers.mean()
        new = np.array(
            [
                x[~modes].mean() if (~modes).any() else centers[0],
                x[modes].mean() if modes.any() else centers[1],
            ]
        )
        if np.allclose(centers, new):
            break
        centers = new
    return centers


def features(ep, config=None):
    ep.require("eef_pos", "eef_quat", "action", "is_settle")
    adapter = EnvironmentAdapter(ep, config)
    rot = adapter.quat_matrix(ep.controls["eef_quat"], "eef")
    translation = np.r_[
        0.0, np.linalg.norm(np.diff(ep.controls["eef_pos"], axis=0), axis=1)
    ]
    relative = np.einsum("nij,njk->nik", np.swapaxes(rot[:-1], 1, 2), rot[1:])
    angular = np.r_[
        0.0,
        np.arccos(
            np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        ),
    ]
    index = int(adapter.setting("gripper_dim"))
    command = np.asarray(ep.controls["action"][:, index], float)
    width = np.array(
        [
            adapter.width(i) if adapter.width(i) is not None else np.nan
            for i in range(ep.n)
        ]
    )
    return {
        "position": np.asarray(ep.controls["eef_pos"], float),
        "rotation": rot,
        "translation": translation,
        "angular": angular,
        "command": command,
        "width": width,
        "active": ~ep.controls["is_settle"].astype(bool),
    }


def fit(episodes, config=None, library_root=None):
    discovery = [
        ep
        for ep in episodes
        if ep.outcome["success"] and int(ep.meta.get("init", 50)) < 30
    ]
    valid = []
    for ep in discovery:
        try:
            valid.append((ep, features(ep, config)))
        except Unavailable:
            continue
    if not valid:
        raise Unavailable(
            "no successful discovery episodes (init 0-29) with portable features"
        )
    command, translation, angular = [], [], []
    for ep, f in valid:
        # Median actual head per decision, terminal partial heads remain marked.
        for seq in sorted(
            set(int(x) for x in ep.controls["decision_seq"] if int(x) >= 0)
        ):
            hits = ep.controls["decision_seq"] == seq
            command.append(float(np.median(f["command"][hits])))
        select = f["active"] & np.r_[False, f["active"][:-1]]
        translation.extend(f["translation"][select].tolist())
        angular.extend(f["angular"][select].tolist())
    source = "successful discovery physics; diagnostic demonstrations, init 0-29"
    command_space = "wire"
    gripper_dim = int(EnvironmentAdapter(valid[0][0], config).setting("gripper_dim"))
    if library_root:
        from pathlib import Path

        root = Path(library_root)
        action = np.load(root / "action.npy", mmap_mode="r", allow_pickle=False)
        success = np.load(
            root / "success.npy", mmap_mode="r", allow_pickle=False
        ).astype(bool)
        manifest = __import__("json").loads((root / "manifest.json").read_text())
        command = np.median(
            action[
                success, : int(manifest["exec_steps"]), int(manifest["gripper_dim"])
            ],
            axis=1,
        ).tolist()
        source = "G0 from immutable successful library action heads; geometry from successful discovery physics"
        command_space = "library_normalized"
        gripper_dim = int(manifest["gripper_dim"])
    if not command or not translation:
        raise Unavailable(
            "discovery/library has no usable command heads or displacements"
        )
    matrix = np.array([translation, angular]).T
    mean = matrix.mean(0)
    std = matrix.std(0)
    std[std <= np.finfo(float).eps] = 1.0
    centers = two_means(command)
    positive_translation = np.array(translation)[
        np.array(translation) > np.finfo(float).eps
    ]
    positive_angular = np.array(angular)[np.array(angular) > np.finfo(float).eps]
    result = {
        "schema": "r8.segment_fit.v1",
        "source": source,
        "training_keys": sorted(ep.key for ep, _ in valid),
        "training_clusters": sorted(
            {
                (ep.meta.get("suite"), int(ep.meta["task_id"]), int(ep.meta["init"]))
                for ep, _ in valid
            }
        ),
        "holdout": "within-task init positions 30-49; never fit",
        "gripper_centers": centers.tolist(),
        "gripper_threshold": float(centers.mean()),
        "gripper_space": command_space,
        "gripper_dim": gripper_dim,
        "command_threshold_supported": bool(centers[0] != centers[1]),
        "stop_quantiles": {
            str(q): float(np.quantile(translation, q)) for q in (0.10, 0.25)
        },
        "translation_epsilon": float(np.median(positive_translation))
        if len(positive_translation)
        else None,
        "angular_epsilon": float(np.median(positive_angular))
        if len(positive_angular)
        else None,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "portable_features": [
            "eef_displacement",
            "SO3_angular_increment",
            "issued_gripper_history",
        ],
        "training_median_active_controls": float(
            np.median([int(f["active"].sum()) for _, f in valid])
        ),
        "forbidden_features": [
            "object_pose",
            "contacts",
            "predicates",
            "future_outcome",
            "episode_progress",
        ],
    }
    # A fixed causal mean-change threshold from discovery prefixes, no holdout.
    excursions = []
    for _, f in valid:
        x = (np.column_stack([f["translation"], f["angular"]]) - mean) / std
        for t in range(10, len(x)):
            if f["active"][t - 10 : t].all():
                excursions.append(
                    float(
                        np.linalg.norm(x[t - 5 : t].mean(0) - x[t - 10 : t - 5].mean(0))
                    )
                )
    result["change_threshold"] = (
        float(np.quantile(excursions, 0.95)) if excursions else None
    )
    result["fit_hash"] = fingerprint(result)
    return result


def baseline_commands(ep, f, fitted):
    """Compare library centers only to normalized served heads, never wire units."""
    if fitted.get("gripper_space") != "library_normalized":
        return f
    if ep.reader_arm is None:
        raise Unavailable("exact library G0 needs captured normalized served chunks")
    ids = [d["decision_id"] for d in ep.decisions]
    try:
        chunks = ep.reader_arm.decision_arrays(["served_chunk"], ids)["served_chunk"]
    except KeyError as exc:
        raise Unavailable(
            "normalized served heads unavailable for library G0: " + str(exc)
        )
    commands = f["command"].copy()
    dim = fitted["gripper_dim"]
    for d, chunk in zip(ep.decisions, chunks):
        head = chunk[:5, dim]
        if not np.isfinite(head).all():
            raise Unavailable("normalized served head is nonfinite")
        commands[ep.controls["decision_seq"] == int(d["decision_seq"])] = float(
            np.median(head)
        )
    return dict(f, command=commands)


def r7_boundaries(ep, f, fitted):
    rows = []
    for seq in sorted(set(int(x) for x in ep.controls["decision_seq"] if int(x) >= 0)):
        hit = np.flatnonzero(ep.controls["decision_seq"] == seq)
        rows.append(
            (
                int(hit[0]),
                float(np.median(f["command"][hit])) > fitted["gripper_threshold"],
            )
        )
    if not rows:
        return []
    modes = np.array([x[1] for x in rows], bool)
    if len(modes) > 2:
        modes = np.r_[
            modes[0], (modes[:-2].astype(int) + modes[1:-1] + modes[2:]) >= 2, modes[-1]
        ]
    return [rows[i][0] for i in np.flatnonzero(modes[1:] != modes[:-1]) + 1]


def rdp(points, epsilon):
    """Geometric reconstruction knots; retrospective, explicitly future-using."""
    if len(points) <= 2:
        return list(range(len(points)))
    keep, work = {0, len(points) - 1}, [(0, len(points) - 1)]
    while work:
        start, end = work.pop()
        if end - start < 2:
            continue
        frac = np.linspace(0.0, 1.0, end - start + 1)[:, None]
        chord = points[start] + frac * (points[end] - points[start])
        error = np.linalg.norm(points[start : end + 1] - chord, axis=1)
        i = int(np.argmax(error))
        if error[i] > epsilon:
            middle = start + i
            keep.add(middle)
            work.extend([(start, middle), (middle, end)])
    return sorted(keep)


def geometry_points(f, fitted, factor=1.0):
    translation = fitted.get("translation_epsilon")
    if translation is None:
        raise Unavailable("no varying discovery translation for waypoint scale")
    result = f["position"] / (translation * factor)
    angular = fitted.get("angular_epsilon")
    if angular is not None:
        # Rotation-matrix chord distance / sqrt(2) approaches SO(3) angle
        # locally. Reconstruction is independently scored by exact SO(3) angle.
        result = np.column_stack(
            [
                result,
                f["rotation"].reshape(len(result), 9)
                / (np.sqrt(2.0) * angular * factor),
            ]
        )
    return result


def reconstruction(f, points, start, end):
    from scipy.spatial.transform import Rotation, Slerp

    index = np.arange(start, end)
    position = np.array(
        [np.interp(index, points, f["position"][points, axis]) for axis in range(3)]
    ).T
    translation = np.linalg.norm(position - f["position"][start:end], axis=1)
    if len(points) >= 2:
        rotations = Slerp(
            np.asarray(points, float), Rotation.from_matrix(f["rotation"][points])
        )(index).as_matrix()
        relative = np.einsum(
            "nij,njk->nik", np.swapaxes(rotations, 1, 2), f["rotation"][start:end]
        )
        angular = np.arccos(
            np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        )
    else:
        angular = np.zeros(len(index))
    return {
        "translation_max": float(translation.max()),
        "translation_rms": float(np.sqrt(np.mean(translation**2))),
        "SO3_rotation_max": float(angular.max()),
        "SO3_rotation_rms": float(np.sqrt(np.mean(angular**2))),
    }


def mean_change_points(features, penalty, min_length=5):
    """Exact penalized mean-shift DP (PELT objective, no unsafe pruning)."""
    x = np.asarray(features, float)
    n, dims = x.shape
    sums = np.vstack([np.zeros(dims), np.cumsum(x, axis=0)])
    squares = np.r_[0.0, np.cumsum((x * x).sum(axis=1))]
    cost, previous = np.full(n + 1, np.inf), np.full(n + 1, -1, int)
    cost[0] = -penalty
    for end in range(min_length, n + 1):
        starts = np.arange(0, end - min_length + 1)
        length = end - starts
        sse = (
            squares[end]
            - squares[starts]
            - ((sums[end] - sums[starts]) ** 2).sum(axis=1) / length
        )
        candidate = cost[starts] + np.maximum(sse, 0.0) + penalty
        arg = int(np.argmin(candidate))
        cost[end], previous[end] = candidate[arg], starts[arg]
    boundaries = []
    end = n
    while end > 0 and previous[end] >= 0:
        end = int(previous[end])
        if end:
            boundaries.append(end)
    return sorted(boundaries)


def candidates(fitted):
    result = ["gripper_r7", "elapsed_fixed"]
    result += ["stop_q%s_d%d" % (q, dwell) for q in (10, 25) for dwell in (1, 5, 10)]
    if fitted.get("translation_epsilon") is not None:
        result += ["waypoint_x%s" % x for x in (0.5, 1.0, 2.0)]
    if fitted.get("change_threshold") is not None:
        result += ["change_x%s" % x for x in (1.0, 2.0, 4.0)]
    return result


def causal_scores(f, fitted, candidate, stride=5):
    """Output at t uses samples <t. Fits are fixed, independent of the suffix."""
    n = len(f["command"])
    output = np.zeros(n)
    first_active = int(np.flatnonzero(f["active"])[0]) if f["active"].any() else n
    modes = []
    anchor = first_active
    mean, std = np.asarray(fitted["feature_mean"]), np.asarray(fitted["feature_std"])
    x = (np.column_stack([f["translation"], f["angular"]]) - mean) / std
    geometric = (
        geometry_points(f, fitted, float(candidate.split("x")[1]))
        if candidate.startswith("waypoint_")
        else None
    )
    for t in range(first_active, n):
        if candidate == "gripper_r7":
            if t > first_active and (t - first_active) % stride == 0:
                modes.append(
                    bool(
                        np.median(f["command"][t - stride : t])
                        > fitted["gripper_threshold"]
                    )
                )
            if len(modes) == 3:
                output[t] = float(modes[0] != (sum(modes) >= 2))
            elif len(modes) >= 4:
                previous, current = sum(modes[-4:-1]) >= 2, sum(modes[-3:]) >= 2
                output[t] = float(previous != current)
        elif candidate.startswith("stop_"):
            q, dwell = candidate.split("_")[1:]
            dwell = int(dwell[1:])
            threshold = fitted["stop_quantiles"][str(int(q[1:]) / 100.0)]
            if t - dwell >= first_active:
                speed = float(np.max(f["translation"][t - dwell : t]))
                output[t] = float(speed <= threshold) + 1.0 / (
                    1.0
                    + speed
                    / max(fitted.get("translation_epsilon") or 1.0, np.finfo(float).eps)
                )
        elif candidate.startswith("waypoint_"):
            if t - anchor >= 2:
                points = geometric[anchor:t]
                chord = points[0] + np.linspace(0.0, 1.0, len(points))[:, None] * (
                    points[-1] - points[0]
                )
                error = float(np.linalg.norm(points - chord, axis=1).max())
                output[t] = error
                if error > 1.0:
                    anchor = t - 1
        elif candidate.startswith("change_"):
            if t - first_active >= 10:
                statistic = float(
                    np.linalg.norm(x[t - 5 : t].mean(0) - x[t - 10 : t - 5].mean(0))
                )
                threshold = fitted["change_threshold"] * float(candidate.split("x")[1])
                output[t] = statistic / max(threshold, np.finfo(float).eps)
        elif candidate == "elapsed_fixed":
            output[t] = (t - first_active) / max(
                fitted["training_median_active_controls"], 1.0
            )
    return output


def retrospective(ep, f, fitted, candidate, resolution=1):
    active = np.flatnonzero(f["active"])
    start, end = int(active[0]), int(active[-1]) + 1
    baseline = r7_boundaries(ep, f, fitted)
    if candidate == "gripper_r7":
        return baseline
    if candidate == "elapsed_fixed":
        spacing = max(1, int(fitted["training_median_active_controls"] / 5.0))
        return list(range(start + spacing, end, spacing))
    if candidate.startswith("stop_"):
        score = causal_scores(f, fitted, candidate)
        stopped = score >= 1.0
        return sorted(
            set(baseline + (np.flatnonzero(stopped[1:] & ~stopped[:-1]) + 1).tolist())
        )
    if candidate.startswith("waypoint_"):
        points_scaled = geometry_points(f, fitted, float(candidate.split("x")[1]))
        index = list(range(start, end, resolution))
        if index[-1] != end - 1:
            index.append(end - 1)
        points = points_scaled[index]
        knots = [index[i] for i in rdp(points, 1.0)]
        return sorted(set(baseline + knots[1:-1]))
    factor = float(candidate.split("x")[1])
    x = (
        np.column_stack([f["translation"][start:end], f["angular"][start:end]])
        - fitted["feature_mean"]
    ) / fitted["feature_std"]
    boundaries = mean_change_points(
        x, factor * x.shape[1] * np.log(max(len(x), 2)), min_length=5
    )
    return sorted(set(baseline + [start + x for x in boundaries]))


def boundary_quality(predicted, truth, tolerance):
    """One-to-one boundary matching; dense ticks cannot multiply true hits."""
    predicted, truth = sorted(predicted), sorted(truth)
    i, j, errors = 0, 0, []
    while i < len(predicted) and j < len(truth):
        if abs(predicted[i] - truth[j]) <= tolerance:
            errors.append(abs(predicted[i] - truth[j]))
            i += 1
            j += 1
        elif predicted[i] < truth[j]:
            i += 1
        else:
            j += 1
    precision = (
        len(errors) / len(predicted) if predicted else (1.0 if not truth else 0.0)
    )
    recall = len(errors) / len(truth) if truth else 1.0
    return {
        "predicted_boundaries": len(predicted),
        "truth_boundaries": len(truth),
        "matched_boundaries": len(errors),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "boundary_mae": float(np.mean(errors)) if errors else None,
        "tolerance_controls": tolerance,
    }


def prefix_audit(f, fitted, candidate):
    full = causal_scores(f, fitted, candidate)
    cutoffs = sorted(set([len(full) // 3, len(full) // 2, max(1, len(full) - 1)]))
    for cut in cutoffs:
        prefix = {k: v[:cut] for k, v in f.items()}
        if not np.array_equal(causal_scores(prefix, fitted, candidate), full[:cut]):
            return False
    return True
