"""Same-observation proposal metrics, restricted to declared action channels."""
import numpy as np

from .common import dimensions


def action_metrics(arm, left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    horizon, width = min(len(left), len(right)), min(left.shape[-1], right.shape[-1])
    dims, scale, grip, threshold, units = dimensions(arm, width)
    a, b = left[:horizon, :width], right[:horizon, :width]
    motion = [dim for dim in dims if dim != grip]
    output = []
    for offset in range(horizon):
        if not np.isfinite(a[offset, dims]).all() or not np.isfinite(b[offset, dims]).all():
            continue
        delta = (a[offset] - b[offset]) / scale
        value = {"future_offset": offset, "mse": float(np.mean(delta[dims] ** 2)),
                 "rms": float(np.sqrt(np.mean(delta[dims] ** 2))), "units": units,
                 "motion_rms": float(np.sqrt(np.mean(delta[motion] ** 2))) if motion else None,
                 "gripper_abs_error": float(abs(a[offset, grip] - b[offset, grip])) if grip is not None else None,
                 "gripper_flip": bool((a[offset, grip] > threshold) != (b[offset, grip] > threshold))
                    if grip is not None and threshold is not None else None}
        x, y = a[offset, motion], b[offset, motion]
        norm = np.linalg.norm(x) * np.linalg.norm(y)
        value["direction_cosine"] = float(x @ y / norm) if norm else None
        output.append(value)
    return output


def noise_floor(arm, draws):
    draws = np.asarray(draws, float)
    pairs = []
    for left in range(len(draws)):
        for right in range(left + 1, len(draws)):
            pairs.extend(action_metrics(arm, draws[left], draws[right]))
    result = {}
    for row in pairs:
        result.setdefault(row["future_offset"], []).append(row["mse"])
    return {offset: float(np.mean(values)) for offset, values in result.items()}
