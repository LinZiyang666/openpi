"""Weighted catalogue provenance, preserving zero-weight structural members."""
from collections import defaultdict

import numpy as np

from .common import field, number, Unavailable, catalog


def catalog_map(arm):
    table = catalog(arm)
    if table is None or not len(table):
        raise Unavailable("library row catalog unavailable")
    if "row" not in table:
        raise Unavailable("catalog row identifiers unavailable")
    has_lib = "lib" in table
    result = {}
    for row in table.to_dict("records"):
        key = (str(row["lib"]), int(row["row"])) if has_lib else int(row["row"])
        if key in result:
            raise ValueError("ambiguous library catalog row")
        result[key] = row
    return result, has_lib


def kernel(rows, weights, catalog=None, lib=None):
    rows = np.asarray(rows, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    if rows.ndim != 1 or weights.shape != rows.shape or not len(rows):
        raise Unavailable("rows/weights absent or shape mismatch")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise Unavailable("invalid retrieval weights")
    weights = weights / weights.sum()
    out = {"members": len(rows), "positive_members": int((weights > 0).sum()), "row_effective_count": float(1 / (weights @ weights))}
    if catalog is None:
        return out
    mapping, has_lib = catalog
    stages, modes, demos = defaultdict(float), defaultdict(float), defaultdict(float)
    unknown = event = tail = unknown_stage = unknown_catalog = 0.
    unknown_members = tail_members = 0
    progresses = []
    for rid, weight in zip(rows, weights):
        item = mapping.get((str(lib), int(rid)) if has_lib else int(rid))
        if item is None:
            unknown += weight
            unknown_stage += weight
            unknown_catalog += weight
            unknown_members += 1
            continue
        mode = field(item, "mode")
        event_near = field(item, "event_near")
        stage = field(item, "stage", "stage_run")
        if mode is None or number(mode) == -1:
            unknown += weight
            unknown_members += 1
        else:
            modes[str(mode)] += weight
        if stage is not None and number(stage) != -1:
            stages[str(stage)] += weight
        else:
            unknown_stage += weight
        if event_near is not None and bool(event_near):
            event += weight
        demo = field(item, "episode")
        if demo is not None:
            demos[(str(item.get("task_id")), str(demo))] += weight
        if number(item.get("next")) == -1:
            tail += weight
            tail_members += 1
        progress = number(item.get("progress"))
        if progress is not None:
            progresses.append((progress, weight))
    demo_known_mass = sum(demos.values())
    progress_known_mass = sum(w for _, w in progresses)
    out.update(mode_mass=dict(modes), stage_mass=dict(stages), unknown_mass=float(unknown),
               unknown_stage_mass=float(unknown_stage), unknown_catalog_mass=float(unknown_catalog),
               unknown_members=unknown_members, event_mass=float(event), known_demo_mass=float(demo_known_mass),
               effective_demos=float(demo_known_mass ** 2 / sum(v * v for v in demos.values())) if demo_known_mass else None,
               cross_stage_kernel=bool(sum(value > 0 for value in stages.values()) > 1),
               cross_mode_kernel=bool(sum(value > 0 for value in modes.values()) > 1),
               structural_cross_stage_kernel=len(stages) > 1, structural_cross_mode_kernel=len(modes) > 1,
               unsupported_next_mass=float(tail), unsupported_next_members=tail_members,
               progress_known_mass=float(progress_known_mass),
               progress=sum(p * w for p, w in progresses) if np.isclose(progress_known_mass, 1.) else None,
               top_demo=list(max(demos, key=demos.get)) if demos else None)
    return out


def overlap(rows_a, weights_a, rows_b, weights_b):
    """Histogram intersection over row IDs, duplicate members merged."""
    masses = []
    for rows, weights in ((rows_a, weights_a), (rows_b, weights_b)):
        values = np.asarray(weights, dtype=float)
        if values.sum() <= 0 or not np.isfinite(values).all() or (values < 0).any():
            return None
        histogram = defaultdict(float)
        for row, weight in zip(rows, values / values.sum()):
            histogram[int(row)] += float(weight)
        masses.append(histogram)
    return sum(min(value, masses[1].get(row, 0.)) for row, value in masses[0].items())
