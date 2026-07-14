"""Authoritative l10 D+ trajectory -> (task_id, orig_init_state_idx) resolver (plan §3-B2).

The libero_10 init map lacks trajectory_id/episode_number, the D+ h5 carry no seed/
init_idx/full sim state (only robot_state + vision features), and episode_NNNN repeats
across tasks with retry ordering -- so collection order is NOT authoritative and
`robot_state@step0` alone is INSUFFICIENT (it is proprioception; LIBERO inits vary by
scene/object state, which the vision features encode).

Two paths (plan §3-B2):
  * CANONICAL DEFAULT = re-collect l10 D+ with the init identity RECORDED in each h5's
    attrs. ``resolve_from_recorded_attrs`` reads that attr; no matching, no ambiguity.
  * ALTERNATIVE (only if it passes the rejection fixture) = scene-observing replay
    fingerprint. ``match_by_vision_fingerprint`` matches each query's initial ``vision_0``
    (mean-pooled, L2-normalized) to per-init reference fingerprints produced by resetting
    each held-out ``.init`` through the same CP1 encoder, with BOTH a frozen absolute
    cosine-distance threshold and a nearest-vs-runner-up margin; any miss/tie -> reject
    (ambiguity fallback = recollect).
"""

from __future__ import annotations

import numpy as np

# Frozen fingerprint thresholds (plan §3-B2 / Post-G1 polish).
FP_ABS_MAX = 0.02  # max cosine distance for a match
FP_MARGIN_MIN = 0.05  # min (runner_up - best) cosine-distance margin


# ------------------------------------------------------------------
# Canonical default: recorded init identity
# ------------------------------------------------------------------
def resolve_from_recorded_attrs(h5_attr_rows: list[dict]) -> dict:
    """Map trajectory_id -> (task_id, orig_init_state_idx) from recorded h5 attrs.

    ``h5_attr_rows`` is one dict per D+ h5 with keys
    ``{trajectory_id, task_id, orig_init_state_idx}`` (written by the recollection run).
    Fails loud on a missing field or a duplicate (task, init).
    """
    out: dict = {}
    seen: set = set()
    for r in h5_attr_rows:
        for k in ("trajectory_id", "task_id", "orig_init_state_idx"):
            if k not in r:
                raise ValueError(f"recollected h5 attr missing {k!r}: {r}")
        tid = r["trajectory_id"]
        if tid in out:  # a trajectory key may never silently overwrite an identity
            raise ValueError(f"duplicate trajectory_id {tid!r} in recollected attrs")
        ident = (int(r["task_id"]), int(r["orig_init_state_idx"]))
        if ident in seen:
            raise ValueError(f"duplicate identity {ident} in recollected attrs")
        seen.add(ident)
        out[tid] = ident
    return out


# ------------------------------------------------------------------
# Alternative: scene-observing vision fingerprint
# ------------------------------------------------------------------
def _fingerprint(vision_0: np.ndarray) -> np.ndarray:
    """Mean-pool a [tokens, dim] initial vision_0 to one L2-normalized vector."""
    v = np.asarray(vision_0, dtype=np.float64).mean(axis=0)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def match_by_vision_fingerprint(
    query_fp: np.ndarray,
    reference_fps: dict,
    *,
    abs_max: float = FP_ABS_MAX,
    margin_min: float = FP_MARGIN_MIN,
) -> int:
    """Return the matched ``orig_init_state_idx`` or raise on ambiguity/miss.

    ``reference_fps`` maps orig_init_state_idx -> reference fingerprint (from sim replay).
    Cosine distance ``1 - cos``. A match requires best <= ``abs_max`` AND
    (runner_up - best) >= ``margin_min``; otherwise reject (caller falls back to recollect).
    """
    q = _fingerprint(query_fp) if np.asarray(query_fp).ndim > 1 else _asfp(query_fp)
    inits = sorted(reference_fps)
    dists = np.array([1.0 - float(q @ _asfp(reference_fps[i])) for i in inits])
    order = np.argsort(dists)
    best_i, best_d = inits[order[0]], dists[order[0]]
    runner_d = dists[order[1]] if len(order) > 1 else np.inf
    if best_d > abs_max:
        raise ValueError(f"no init within abs threshold (best d={best_d:.4f} > {abs_max})")
    if (runner_d - best_d) < margin_min:
        raise ValueError(f"ambiguous match (margin {runner_d - best_d:.4f} < {margin_min})")
    return best_i


def _asfp(v: np.ndarray) -> np.ndarray:
    """Normalize a stored reference fingerprint (already pooled) to unit length."""
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)
