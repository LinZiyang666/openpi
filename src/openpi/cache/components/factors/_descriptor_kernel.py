"""Single-window descriptor kernel: jerk / direction / dispersion / path_length.

Pure functions. The 8 online factors (``factors/online.py``) and the 8
offline factors (``factors/offline.py``) both call this kernel on a
single z-scored, active-DOF-restricted window so the four shipped
descriptors are computed once and used by both factor families.
Algorithms are defined in ``docs/cache/verdict_factor_judge.md`` §3.2.

Descriptor naming: the kernel accepts both the refactored names
(``jerk`` / ``direction`` / ``dispersion`` / ``path_length``) and the
legacy aliases (``jerk`` / ``dir`` / ``curv_radius`` / ``cum_disp``);
they dispatch to the same per-descriptor helper functions below.

Public surface:

- `compute_descriptors(descriptors, pts, v, j) -> dict[str, float]`
  Returns one float per descriptor name. NaN propagates per the
  verdict-factor missing-signal contract (§2.8.8 of the B0 plan):
  zero-length / single-point windows or zero-norm denominators yield
  NaN for the affected descriptor; all-NaN windows are short-circuited
  to MISS by `CompositeJudge`.

The kernel does NOT do z-score, active masking, splice / chain assembly,
or library_stats lookup. Those happen in the calling factor (F1a or F1b)
before invoking this kernel — the kernel only sees a tensor `pts` of
shape `[W, D_act]` already restricted to the active subspace, plus the
matching first/second differences `v` (`[W-1, D_act]`) and `j`
(`[W-2, D_act]`).

Coupling map:
  DEPENDS ON:  nothing internal (stdlib + torch)
  CONSUMED BY: components/factors/online.py  (online descriptor extractors)
               components/factors/offline.py (offline descriptor writers)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Symbolic key constants. The orientation table below is the single
# source of truth for the descriptor name set.
#
# Refactor B2: each formula now has both a legacy name (used by the B1-B6
# legacy factor classes that stay alive for old-yaml compatibility) and a
# refactored name. The new names ship with the 17-factor flat layout in
# online.py / offline.py / topk.py:
#     dir         -> direction
#     curv_radius -> dispersion
#     cum_disp    -> path_length
#     jerk        -> jerk          (unchanged)
# The dispatch table maps every accepted alias to one helper; the formula
# is identical regardless of which alias the caller used.
_JERK = "jerk"
_DIR = "dir"                # legacy alias
_DIRECTION = "direction"
_CURV_RADIUS = "curv_radius"  # legacy alias
_DISPERSION = "dispersion"
_CUM_DISP = "cum_disp"      # legacy alias
_PATH_LENGTH = "path_length"

_LEGACY_NAMES: frozenset[str] = frozenset({_JERK, _DIR, _CURV_RADIUS, _CUM_DISP})
_NEW_NAMES: frozenset[str] = frozenset({_JERK, _DIRECTION, _DISPERSION, _PATH_LENGTH})
_KNOWN: frozenset[str] = _LEGACY_NAMES | _NEW_NAMES

# Public orientation table — single source of truth across factors layer
# (B2 refactor migrated this here from the per-factor modules to break the
# lazy-import cycle the online/offline extractors used to need). Both the
# legacy 4 names and the refactored 4 names are listed; the legacy entries
# stay alive for the B1-B6 backward-compat path.
_DESCRIPTOR_ORIENTATIONS: dict[str, str] = {
    # Refactored names (used by the 17-factor flat layout)
    _JERK:        "risky",
    _DIRECTION:   "safe",
    _DISPERSION:  "non_monotonic",
    _PATH_LENGTH: "non_monotonic",
    # Legacy aliases (kept until B7 cut-over)
    _DIR:         "safe",
    _CURV_RADIUS: "non_monotonic",
    _CUM_DISP:    "non_monotonic",
}

# ----------------------------------------------------------------------
# Public entrypoint
# ----------------------------------------------------------------------


def compute_descriptors(
    descriptors: list[str],
    pts: torch.Tensor,
    v: torch.Tensor,
    j: torch.Tensor,
) -> dict[str, float]:
    """Compute the requested descriptors over one window.

    pts: [W,   D_act] — z-scored points in active subspace
    v:   [W-1, D_act] — first differences (already on z-scored space)
    j:   [W-2, D_act] — second differences

    A descriptor with insufficient samples (e.g. dir on a 1-step window)
    or a degenerate denominator (e.g. all-zero velocities) is returned
    as NaN.
    """
    out: dict[str, float] = {}
    for d in descriptors:
        if d not in _KNOWN:
            # Not implemented yet — extension descriptors (dirvar, path,
            # freq, autocorr) are reserved per docs §3.2 but their
            # algorithms ship later. Fail loud at the kernel boundary
            # so config / __init__ caught the intent first.
            raise NotImplementedError(
                f"descriptor {d!r} is not implemented in B1+B2 "
                f"(known: {sorted(_KNOWN)})"
            )
        if d == _JERK:
            out[d] = _jerk(j)
        elif d in (_DIR, _DIRECTION):
            out[d] = _dir(v)
        elif d in (_CURV_RADIUS, _DISPERSION):
            out[d] = _curv_radius(pts)
        elif d in (_CUM_DISP, _PATH_LENGTH):
            out[d] = _cum_disp(pts)
    return out


# ----------------------------------------------------------------------
# Individual descriptors
# ----------------------------------------------------------------------


def _jerk(j: torch.Tensor) -> float:
    # `j` shape: [W-2, D_act]. Median over time of |Δ²a|, then mean over
    # active DOFs — median absorbs gripper bistable single-frame spikes
    # (per docs §3.2.1).
    if j.numel() == 0 or j.shape[0] == 0:
        return float("nan")
    per_dof = j.abs().median(dim=0).values        # [D_act]
    return float(per_dof.mean())


def _dir(v: torch.Tensor) -> float:
    # `v` shape: [W-1, D_act]. Mean cosine between consecutive velocity
    # vectors in the active subspace. Need at least 2 velocity samples
    # (W >= 3). PyTorch's F.cosine_similarity adds an eps to the
    # denominator and returns 0.0 (not NaN) when both inputs are zero
    # vectors — that quietly classifies a stationary window as
    # "neutral direction" instead of "no signal", which we don't want.
    # Detect zero-norm pairs explicitly and surface them as NaN per the
    # missing-signal contract.
    if v.shape[0] < 2:
        return float("nan")
    v1 = v[:-1]                                   # [W-2, D_act]
    v2 = v[1:]                                    # [W-2, D_act]
    n1 = v1.norm(dim=-1)                          # [W-2]
    n2 = v2.norm(dim=-1)                          # [W-2]
    valid_mask = (n1 > 0) & (n2 > 0)              # [W-2] bool
    if not valid_mask.any():
        return float("nan")
    cos = F.cosine_similarity(v1, v2, dim=-1)     # [W-2]
    return float(cos[valid_mask].mean())


def _curv_radius(pts: torch.Tensor) -> float:
    # `pts` shape: [W, D_act]. Mean Euclidean distance from window
    # points to their centroid — geometric dispersion in active subspace
    # (per docs §3.2.3).
    if pts.shape[0] < 2:
        return float("nan")
    centroid = pts.mean(dim=0, keepdim=True)      # [1, D_act]
    return float((pts - centroid).norm(dim=-1).mean())


def _cum_disp(pts: torch.Tensor) -> float:
    # `pts` shape: [W, D_act]. Sum of consecutive Euclidean step lengths
    # — accumulated path length in active subspace (per docs §3.2.4).
    if pts.shape[0] < 2:
        return float("nan")
    return float((pts[1:] - pts[:-1]).norm(dim=-1).sum())


# ----------------------------------------------------------------------
# Active-mask short-circuit helper
# ----------------------------------------------------------------------


def all_nan_for(descriptors: list[str]) -> dict[str, float]:
    """Return {d: NaN} for each requested descriptor.

    Used by F1a / F1b to bail out without computation when the active
    mask sums to zero or input shape is degenerate. Centralizing the
    sentinel here keeps the NaN-propagation rule consistent with what
    `compute_descriptors` would have produced.
    """
    nan = float("nan")
    return {d: nan for d in descriptors}


def is_all_nan(values: dict[str, float]) -> bool:
    """True iff every value in the dict is NaN. Mirrors CompositeJudge's
    cold-start sentinel detection so callers can check parity locally."""
    if not values:
        return False
    return all(math.isnan(v) for v in values.values())
