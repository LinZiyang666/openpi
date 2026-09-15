"""Online risk-indexed threshold (``judge.type: "online_rit"``).

The offline RIT rule fits one risk curve per reuse tier on a shadow cohort and
freezes its cuts. Deployment then walks a state distribution the cohort never
visited (reuse itself moves the robot), so the frozen curves are calibrated
for the wrong distribution. This judge keeps the same monotone-cut rule class
but learns the curves online from a signal every warm start already exposes:
the *continuation disagreement* between the first denoising update the frozen
policy makes under the current observation and the update the library stored
for the same intermediate state.

Signal (plan logs/online_rit_groot_plan.log.md §3.1)
----------------------------------------------------
For a library snapshot ``x_t`` at loop index ``i`` (ascending GR00T schedule,
``t = i / N``):

* stored update      ``u_e = (x_{t_{i+1}} - x_t) * N`` -- consecutive snapshots
  differ by exactly ``dt * pred`` of the original loop, and the library's
  ``action_chunk`` is the loop end ``x_1``, so no new storage is needed;
* current update     ``u_now = (denoise_step(x_t | current stage 2) - x_t) * N``;
* disagreement       ``d = mean_h || W_i * m * (u_now - u_e)_h ||_2`` over the
  first ``h_exec`` action steps, ``W_i`` an inverse-sigma scale computed from
  the library's own updates and ``m`` the executed-dimension mask.

Estimator (§3.3)
----------------
Every (tier, knot) keeps the most recent ``window`` kernel-weighted samples
of ``d``; a sample at score ``s`` in ``[u_k, u_{k+1}]`` is written to both
knots with linear weights. The knot value is the weighted empirical
``1 - alpha`` quantile, projected to be non-increasing in ``s`` with a
weighted pool-adjacent-violators pass over the *valid* knots (weight sum at
least ``n_min``). Validity is a separate mask: the value arrays never carry
infinities, and a query is labelled by where it lands relative to the valid
knots (``supported`` / ``gap_interpolated`` / ``tail_extrapolated`` /
``unavailable``).

Dispatch (§3.4)
---------------
Tiers are walked riskiest first (fewest remaining denoising steps first); the
first tier whose cut the score clears is executed, else MISS. There is no
FULL_HIT: the cheapest tier resumes at the last snapshot and runs one current-
conditioned step (owner ruling 2026-09-14). A MISS with a candidate keeps the
winner id so the interceptor can side-evaluate every tier (feedback mode
``fm1``) from the backbone it pays for anyway.

Public interface
----------------
``reference_update``, ``continuation_disagreement``, ``weighted_quantile``,
``pav_nonincreasing``, ``cut_at_pl``, ``TierSpec``, ``ContinuationFeedback``,
``DecisionSnapshot``, ``OnlineRiskCurves``, ``ContinuationSpec``,
``OnlineRitJudge``.

Key dependencies: ``openpi.cache.types.DenoiseSchedule`` (snapshot geometry),
``openpi.cache.storage_types.CachePayload`` (snapshots), ``JudgeResult`` /
``HitType`` (verdict contract), ``openpi.cache.online_state.CurveRegistry``
(shared state across connections).
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
from typing import Any, Optional

import numpy as np
import torch

from openpi.cache.components.judge import HitType, JudgeResult
from openpi.cache.storage_types import CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID, DenoiseSchedule

SUPPORTED = "supported"
GAP_INTERPOLATED = "gap_interpolated"
TAIL_EXTRAPOLATED = "tail_extrapolated"
UNAVAILABLE = "unavailable"
SUPPORT_KINDS = (SUPPORTED, GAP_INTERPOLATED, TAIL_EXTRAPOLATED, UNAVAILABLE)

SOURCE_EXECUTED = "executed"
SOURCE_SHADOW = "shadow"

FEEDBACK_MODES = ("fm0", "fm1")

# Scores are a convex combination of per-field ``0.5 * (1 + tanh(.))`` terms,
# so the domain is [0, 1]; anything further out than this is a data error.
SCORE_DOMAIN = (0.0, 1.0)
SCORE_TOL = 1e-6


# ------------------------------------------------------------------
# Signal
# ------------------------------------------------------------------


def reference_update(
    payload: CachePayload, t: float, schedule: DenoiseSchedule
) -> torch.Tensor:
    """The library's own update out of snapshot ``t``: ``(x_next - x_t) * N``.

    ``x_next`` is the next stored snapshot, or ``action_chunk`` (the loop end)
    when ``t`` is the last snapshot. Raises ``ValueError`` when either tensor is
    absent so a library that cannot supply the reference fails before any
    online curve is touched.
    """
    if not payload.intermediates:
        raise ValueError("payload has no intermediates")
    t_key = round(float(t), 4)
    index = schedule.snapshot_index(t_key)
    x_t = _lookup_snapshot(payload, t_key)
    if index + 1 < schedule.num_steps:
        t_next = schedule.snapshot_t(index + 1)
        x_next = _lookup_snapshot(payload, t_next)
    else:
        if payload.action_chunk is None:
            raise ValueError("payload has no action_chunk for the last-step reference")
        x_next = torch.as_tensor(payload.action_chunk)
    x_t = torch.as_tensor(x_t, dtype=torch.float32)
    x_next = torch.as_tensor(x_next, dtype=torch.float32)
    if x_t.shape != x_next.shape:
        raise ValueError(
            f"snapshot shape {tuple(x_t.shape)} != next shape {tuple(x_next.shape)}"
        )
    return (x_next - x_t) * float(schedule.num_steps)


def _lookup_snapshot(payload: CachePayload, t: float) -> torch.Tensor:
    inter = payload.intermediates or {}
    if t in inter:
        return inter[t]
    for key, value in inter.items():
        if abs(float(key) - t) < 1e-6:
            return value
    raise ValueError(f"payload has no snapshot at t={t:.4f}")


def continuation_disagreement(
    u_now: torch.Tensor,
    u_ref: torch.Tensor,
    scale: torch.Tensor,
    mask: torch.Tensor,
    h_exec: int,
) -> float:
    """Mean over the executed window of the scaled, masked L2 update gap.

    ``scale`` is applied only where ``mask`` is true (its other entries are
    ignored, not merely expected to be zero). A non-finite result raises: the
    caller counts it as a rejected observation rather than feeding it to the
    curves.
    """
    if h_exec < 1:
        raise ValueError("h_exec must be >= 1")
    a = torch.as_tensor(u_now, dtype=torch.float32)
    b = torch.as_tensor(u_ref, dtype=torch.float32)
    if a.dim() == 3:
        a = a[0]
    if b.dim() == 3:
        b = b[0]
    if a.shape != b.shape:
        raise ValueError(f"update shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
    m = torch.as_tensor(mask, dtype=torch.bool)
    if m.dim() != 1 or m.shape[0] != a.shape[-1] or not bool(m.any()):
        raise ValueError("mask must be a 1-D bool vector over action dims with >= 1 true")
    w = torch.as_tensor(scale, dtype=torch.float32)
    if w.shape != m.shape:
        raise ValueError("scale must have one entry per action dim")
    diff = (a[:h_exec] - b[:h_exec])[..., m] * w[m]
    per_step = torch.linalg.vector_norm(diff, dim=-1)  # [h_exec]
    value = float(per_step.mean())
    if not math.isfinite(value):
        raise ValueError("continuation disagreement is not finite")
    return value


# ------------------------------------------------------------------
# Estimator primitives
# ------------------------------------------------------------------


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Smallest value whose cumulative weight reaches ``level`` of the total."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0 or w.size != v.size:
        raise ValueError("values and weights must be non-empty and equal length")
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative with positive total")
    order = np.argsort(v, kind="stable")
    cum = np.cumsum(w[order])
    target = float(level) * float(cum[-1])
    idx = int(np.searchsorted(cum, target - 1e-12 * max(1.0, cum[-1]), side="left"))
    idx = min(idx, v.size - 1)
    return float(v[order][idx])


def pav_nonincreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators projection onto non-increasing sequences."""
    v = list(np.asarray(values, dtype=np.float64))
    w = list(np.asarray(weights, dtype=np.float64))
    if len(v) != len(w):
        raise ValueError("values and weights must be equal length")
    blocks: list[list[float]] = []  # [value, weight, count]
    for value, weight in zip(v, w):
        blocks.append([value, weight, 1])
        while len(blocks) >= 2 and blocks[-2][0] < blocks[-1][0]:
            b1, b0 = blocks.pop(), blocks.pop()
            total = b0[1] + b1[1]
            merged = (b0[0] * b0[1] + b1[0] * b1[1]) / total if total > 0 else max(b0[0], b1[0])
            blocks.append([merged, total, b0[2] + b1[2]])
    out: list[float] = []
    for value, _, count in blocks:
        out.extend([value] * count)
    return np.asarray(out, dtype=np.float64)


def cut_at_pl(knots_valid: np.ndarray, q_valid: np.ndarray, delta: float) -> float:
    """Smallest ``s`` on the valid piecewise-linear curve with ``q(s) <= delta``.

    Same crossing semantics as ``exp.rit_pareto.rit_k.cut_at`` on all-finite
    input (first knot if the curve starts at or below ``delta``, ``+inf`` if it
    never gets there, else the linear crossing inside the first segment whose
    right knot is admissible). ``knots_valid`` / ``q_valid`` are the *valid*
    subset, so no ``inf / inf`` can arise.
    """
    knots = np.asarray(knots_valid, dtype=np.float64)
    q = np.asarray(q_valid, dtype=np.float64)
    if knots.size == 0 or knots.shape != q.shape:
        raise ValueError("knots_valid and q_valid must be equal, non-empty")
    if not np.isfinite(q).all() or not np.isfinite(knots).all():
        raise ValueError("cut_at_pl needs finite knots and values")
    if not math.isfinite(delta):
        raise ValueError("delta must be finite")
    if q[0] <= delta:
        return float(knots[0])
    if q[-1] > delta:
        return math.inf
    idx = int(np.where(q <= delta)[0][0])
    k = idx - 1
    drop = q[k] - q[idx]
    if drop <= 0:
        return float(knots[idx])
    return float(knots[k] + (q[k] - delta) / drop * (knots[idx] - knots[k]))


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TierSpec:
    """One reuse tier: resume timestep, its loop index and remaining steps."""

    start_t: float
    index: int
    remaining_steps: int

    @property
    def name(self) -> str:
        return f"warm{int(round(self.start_t * 1000)):03d}"


def tier_specs(start_ts: list[float], schedule: DenoiseSchedule) -> tuple[TierSpec, ...]:
    """Riskiest-first tier specs; ``start_ts`` must be strictly decreasing snapshots."""
    specs = []
    prev = None
    for t in start_ts:
        t4 = round(float(t), 4)
        if t4 not in schedule.timestep_set:
            raise ValueError(
                f"start_t={t} is not a snapshot of {schedule.schedule_id}: {list(schedule.timesteps)}"
            )
        if prev is not None and t4 >= prev:
            raise ValueError("tiers must be strictly decreasing in start_t (riskiest first)")
        prev = t4
        specs.append(TierSpec(t4, schedule.snapshot_index(t4), schedule.remaining_steps(t4)))
    if not specs:
        raise ValueError("at least one tier is required")
    return tuple(specs)


@dataclasses.dataclass(frozen=True)
class ContinuationFeedback:
    """One disagreement observation for one tier of one decision."""

    tier_index: int
    d: float
    source: str

    def __post_init__(self) -> None:
        if self.source not in (SOURCE_EXECUTED, SOURCE_SHADOW):
            raise ValueError(f"unknown feedback source {self.source!r}")
        if not math.isfinite(self.d):
            raise ValueError("feedback d must be finite")


@dataclasses.dataclass(frozen=True)
class DecisionSnapshot:
    """What the judge saw at decision time, captured atomically under the lock."""

    decision_id: tuple
    decision_revision: int
    score: float
    q_pre: dict[int, Optional[float]]
    cuts: dict[int, Optional[float]]
    support_kind: dict[int, str]
    cut_available: dict[int, bool]
    shadowed: tuple[int, ...]

    def to_json(self) -> dict:
        return {
            "decision_id": list(self.decision_id),
            "decision_revision": self.decision_revision,
            "score": self.score,
            "q_pre": {str(k): v for k, v in self.q_pre.items()},
            "cuts": {str(k): v for k, v in self.cuts.items()},
            "support_kind": {str(k): v for k, v in self.support_kind.items()},
            "cut_available": {str(k): v for k, v in self.cut_available.items()},
            "shadowed": list(self.shadowed),
        }


# ------------------------------------------------------------------
# Estimator
# ------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


class OnlineRiskCurves:
    """Per-tier score-conditioned upper-quantile curves learned online.

    Not thread-safe by itself: ``CurveRegistry`` serialises access.
    """

    def __init__(
        self,
        *,
        knots: list[float],
        tier_indices: list[int],
        alpha: float,
        window: int,
        n_min: int,
        update_enabled: bool = True,
        fixed_params: Optional[dict] = None,
    ) -> None:
        k = np.asarray(knots, dtype=np.float64)
        if k.ndim != 1 or k.size < 3 or not (np.diff(k) > 0).all():
            raise ValueError("knots must be strictly increasing with at least three values")
        if not (0.0 < float(alpha) <= 0.5):
            raise ValueError("alpha must lie in (0, 0.5]")
        if int(window) < 1 or int(n_min) < 1 or int(window) < int(n_min):
            raise ValueError("window >= n_min >= 1 is required")
        if not tier_indices or len(set(tier_indices)) != len(tier_indices):
            raise ValueError("tier_indices must be non-empty and unique")
        self.knots = k
        self.tier_indices = tuple(int(i) for i in tier_indices)
        self.alpha = float(alpha)
        self.window = int(window)
        self.n_min = int(n_min)
        self.update_enabled = bool(update_enabled)
        self.fixed_params = dict(fixed_params or {})
        # tier -> knot -> deque of (d, weight, sample_id)
        self._windows: dict[int, list[collections.deque]] = {
            i: [collections.deque(maxlen=self.window) for _ in range(k.size)]
            for i in self.tier_indices
        }
        self._q: dict[int, np.ndarray] = {}
        self._valid: dict[int, np.ndarray] = {}
        self._weight_sum: dict[int, np.ndarray] = {}
        self.revision = 0
        self.n_updates = 0
        self.n_feedback = 0
        self.n_observed = 0  # feedback seen while frozen (not learned)
        self.n_rejected = 0
        self._seen_decisions: set[tuple] = set()
        self.source_revision: Optional[int] = None
        self.source_sha256: Optional[str] = None
        for i in self.tier_indices:
            self._recompute(i)

    # -- queries -----------------------------------------------------------

    def weight_sum(self, tier: int) -> np.ndarray:
        return self._weight_sum[tier].copy()

    def valid(self, tier: int) -> np.ndarray:
        return self._valid[tier].copy()

    def q_values(self, tier: int) -> np.ndarray:
        """Knot values after PAV; entries at invalid knots are NaN placeholders."""
        return self._q[tier].copy()

    def _valid_arrays(self, tier: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid = self._valid[tier]
        idx = np.where(valid)[0]
        return idx, self.knots[idx], self._q[tier][idx]

    def query(self, tier: int, score: float) -> tuple[Optional[float], str]:
        """``(q(score), support kind)``; ``None`` when unavailable."""
        s = _check_score(score)
        idx, knots_v, q_v = self._valid_arrays(tier)
        if idx.size == 0 or s < knots_v[0]:
            return None, UNAVAILABLE
        if s >= knots_v[-1]:
            if s == knots_v[-1]:
                return float(q_v[-1]), SUPPORTED
            return float(q_v[-1]), TAIL_EXTRAPOLATED
        seg = int(np.searchsorted(knots_v, s, side="right") - 1)
        left, right = idx[seg], idx[seg + 1]
        kind = SUPPORTED if right == left + 1 else GAP_INTERPOLATED
        if s == knots_v[seg]:
            return float(q_v[seg]), SUPPORTED
        w = (s - knots_v[seg]) / (knots_v[seg + 1] - knots_v[seg])
        return float((1.0 - w) * q_v[seg] + w * q_v[seg + 1]), kind

    def cut(self, tier: int, delta: float) -> tuple[float, bool]:
        """``(theta, available)``: ``theta = inf`` when no valid knot admits ``delta``."""
        _, knots_v, q_v = self._valid_arrays(tier)
        if knots_v.size == 0:
            return math.inf, False
        theta = cut_at_pl(knots_v, q_v, float(delta))
        return theta, math.isfinite(theta)

    def cuts(self, delta: float) -> dict[int, float]:
        return {i: self.cut(i, delta)[0] for i in self.tier_indices}

    def shadowed(self, delta: float) -> tuple[int, ...]:
        """Tiers that can never fire: a cheaper (earlier) tier has a cut no higher."""
        cuts = self.cuts(delta)
        out = []
        best = math.inf
        for i in self.tier_indices:
            theta = cuts[i]
            if theta >= best:
                out.append(i)
            best = min(best, theta)
        return tuple(out)

    # -- learning ----------------------------------------------------------

    def update_batch(
        self, score: float, feedback: list[ContinuationFeedback], decision_id: tuple
    ) -> dict:
        """Commit every observation of one decision atomically.

        Returns a diagnostic dict; when the state is frozen the observations
        are counted but nothing else moves.
        """
        key = tuple(decision_id)
        if key in self._seen_decisions:
            raise ValueError(f"decision {key} was already committed")
        s = _check_score(score)
        accepted = [fb for fb in feedback if fb.tier_index in self._windows]
        if len(accepted) != len(feedback):
            raise ValueError("feedback names a tier this estimator does not track")
        self._seen_decisions.add(key)
        rev_before = self.revision
        if not self.update_enabled:
            self.n_observed += len(accepted)
            return {
                "learned": False,
                "revision_before": rev_before,
                "revision_after": self.revision,
                "n_feedback": len(accepted),
            }
        seg = int(np.clip(np.searchsorted(self.knots, s, side="right") - 1, 0, self.knots.size - 2))
        left, right = self.knots[seg], self.knots[seg + 1]
        w_right = (s - left) / (right - left)
        touched: set[int] = set()
        for fb in accepted:
            windows = self._windows[fb.tier_index]
            sample_id = (key, fb.tier_index, fb.source)
            if w_right < 1.0:
                windows[seg].append((float(fb.d), float(1.0 - w_right), sample_id))
            if w_right > 0.0:
                windows[seg + 1].append((float(fb.d), float(w_right), sample_id))
            touched.add(fb.tier_index)
            self.n_feedback += 1
        for tier in touched:
            self._recompute(tier)
        if accepted:
            self.revision += 1
            self.n_updates += 1
        return {
            "learned": bool(accepted),
            "revision_before": rev_before,
            "revision_after": self.revision,
            "n_feedback": len(accepted),
        }

    def _recompute(self, tier: int) -> None:
        windows = self._windows[tier]
        n_knots = self.knots.size
        q_raw = np.full(n_knots, np.nan)
        wsum = np.zeros(n_knots)
        for k, win in enumerate(windows):
            if not win:
                continue
            vals = np.array([d for d, _, _ in win])
            wts = np.array([w for _, w, _ in win])
            wsum[k] = float(wts.sum())
            q_raw[k] = weighted_quantile(vals, wts, 1.0 - self.alpha)
        valid = wsum >= self.n_min
        q = np.full(n_knots, np.nan)
        if valid.any():
            idx = np.where(valid)[0]
            q[idx] = pav_nonincreasing(q_raw[idx], wsum[idx])
        self._q[tier] = q
        self._valid[tier] = valid
        self._weight_sum[tier] = wsum

    # -- persistence -------------------------------------------------------

    def learning_state(self) -> dict:
        """Everything that determines the next update (and only that)."""
        return {
            "knots": [float(x) for x in self.knots],
            "tier_indices": list(self.tier_indices),
            "alpha": self.alpha,
            "window": self.window,
            "n_min": self.n_min,
            "fixed_params": self.fixed_params,
            "revision": self.revision,
            "n_updates": self.n_updates,
            "n_feedback": self.n_feedback,
            "windows": {
                str(i): [
                    [[d, w, [list(sid[0]), sid[1], sid[2]]] for d, w, sid in win]
                    for win in self._windows[i]
                ]
                for i in self.tier_indices
            },
        }

    def learning_state_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.learning_state()).encode("utf-8")).hexdigest()

    def snapshot(self) -> dict:
        """Full state: learning state plus derived curves and run counters."""
        state = self.learning_state()
        state["learning_state_sha256"] = self.learning_state_sha256()
        state["update_enabled"] = self.update_enabled
        state["n_observed"] = self.n_observed
        state["n_rejected"] = self.n_rejected
        state["source_revision"] = self.source_revision
        state["source_sha256"] = self.source_sha256
        state["curves"] = {
            str(i): {
                "q": [None if not v else float(x) for x, v in zip(self._q[i], self._valid[i])],
                "valid": [bool(v) for v in self._valid[i]],
                "weight_sum": [float(x) for x in self._weight_sum[i]],
            }
            for i in self.tier_indices
        }
        state["seen_decisions"] = [list(k) for k in sorted(self._seen_decisions, key=repr)]
        return seal_state(state)

    @classmethod
    def from_snapshot(
        cls, state: dict, *, update_enabled: bool, reset_counters: bool = True
    ) -> "OnlineRiskCurves":
        """Rebuild the exact learning state after verifying both content hashes.

        ``learning_state_sha256`` (windows / order / params / learning counters)
        is always recomputed from the loaded windows and compared; a snapshot
        whose content was edited after it was written is refused whatever
        ``reset_counters`` says. ``state_sha256`` covers the whole serialised
        document and is verified when present. Only after that do the current
        stream's counters start from zero (``reset_counters``), with the source
        revision / hash recorded for provenance.
        """
        verify_state_sha(state)
        obj = cls(
            knots=state["knots"],
            tier_indices=state["tier_indices"],
            alpha=state["alpha"],
            window=state["window"],
            n_min=state["n_min"],
            update_enabled=update_enabled,
            fixed_params=state.get("fixed_params"),
        )
        for i in obj.tier_indices:
            wins = state["windows"][str(i)]
            if len(wins) != obj.knots.size:
                raise ValueError("snapshot window layout does not match the knots")
            for k, win in enumerate(wins):
                for d, w, sid in win:
                    obj._windows[i][k].append((float(d), float(w), (tuple(sid[0]), int(sid[1]), str(sid[2]))))
            obj._recompute(i)
        # Restore the learning counters first so the hash is computed over the
        # state exactly as it was written, then decide whether to keep them.
        obj.revision = int(state.get("revision", 0))
        obj.n_updates = int(state.get("n_updates", 0))
        obj.n_feedback = int(state.get("n_feedback", 0))
        obj._seen_decisions = {tuple(_as_hashable(x) for x in k) for k in state.get("seen_decisions", [])}
        expected = state.get("learning_state_sha256")
        if expected is None:
            raise ValueError("snapshot carries no learning_state_sha256")
        if obj.learning_state_sha256() != expected:
            raise ValueError("snapshot learning_state_sha256 does not match its content")
        if reset_counters:
            obj.source_revision = obj.revision
            obj.source_sha256 = expected
            obj.revision = 0
            obj.n_updates = 0
            obj.n_feedback = 0
            obj._seen_decisions = set()
        return obj


def seal_state(state: dict) -> dict:
    """(Re)compute ``state_sha256`` over every other key of a serialisable state dict."""
    body = {k: v for k, v in state.items() if k != "state_sha256"}
    state["state_sha256"] = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    return state


def verify_state_sha(state: dict) -> None:
    """Refuse a state document whose ``state_sha256`` no longer matches its content."""
    expected = state.get("state_sha256")
    if expected is None:
        return
    body = {k: v for k, v in state.items() if k != "state_sha256"}
    actual = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    if actual != expected:
        raise ValueError("state_sha256 does not match the document content")


def _as_hashable(x):
    return tuple(x) if isinstance(x, list) else x


def _check_score(score: float) -> float:
    s = float(score)
    if not math.isfinite(s):
        raise ValueError("score must be finite")
    lo, hi = SCORE_DOMAIN
    if s < lo - SCORE_TOL or s > hi + SCORE_TOL:
        raise ValueError(f"score {s} outside the fused-score domain {SCORE_DOMAIN}")
    return min(max(s, lo), hi)


# ------------------------------------------------------------------
# Judge
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ContinuationSpec:
    """What the interceptor needs to turn a warm start into feedback."""

    tiers: tuple[TierSpec, ...]
    scales: dict[int, torch.Tensor]  # tier index -> [D] inverse-sigma (0 off-mask)
    masks: dict[int, torch.Tensor]  # tier index -> [D] bool executed-dim mask
    h_exec: int
    feedback_mode: str
    schedule: DenoiseSchedule

    def tier_by_start_t(self, start_t: float) -> TierSpec:
        t4 = round(float(start_t), 4)
        for tier in self.tiers:
            if tier.start_t == t4:
                return tier
        raise KeyError(f"no tier at start_t={start_t}")


class OnlineRitJudge:
    """Three-way verdict (warm@tier or MISS) from shared online risk curves.

    The curves live in a ``CurveRegistry`` entry shared by every connection
    of the same bundle; this object holds the per-connection pending decision
    and the episode identity used for logging.
    """

    def __init__(
        self,
        *,
        registry,
        registry_key,
        spec: ContinuationSpec,
        delta: float,
        yaml_id: str,
    ) -> None:
        if not math.isfinite(delta) or delta < 0:
            raise ValueError("delta must be finite and >= 0")
        if spec.feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"feedback_mode must be one of {FEEDBACK_MODES}")
        self._registry = registry
        self._key = registry_key
        self._spec = spec
        self._delta = float(delta)
        self._yaml_id = yaml_id
        self._decision_idx = 0
        self._episode: dict[str, Any] = {}
        self._pending: Optional[DecisionSnapshot] = None
        self._pending_verdict: Optional[dict] = None

    # -- protocol surface --------------------------------------------------

    @property
    def continuation_spec(self) -> ContinuationSpec:
        return self._spec

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def yaml_id(self) -> str:
        return self._yaml_id

    def on_episode_start(
        self, task_key: str = "", extra_metadata=None, provisional: bool = False, **_
    ) -> None:
        if provisional:
            return
        extra = dict(extra_metadata or {})
        self._episode = {
            "task_key": task_key,
            "task_id": extra.get("task_id"),
            "orig_init_state_idx": extra.get("orig_init_state_idx"),
            "task_uid": extra.get("task_uid"),
            "attempt": extra.get("attempt"),
        }
        self._decision_idx = 0
        self._pending = None
        self._pending_verdict = None

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """No-op: the judge never reads the action history."""

    def on_task_end(self) -> None:
        """Connection close: write a full snapshot so the terminal state is on disk."""
        self._registry.flush(self._key, "task_end")

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        **kwargs,
    ) -> JudgeResult:
        decision_idx = self._decision_idx
        self._decision_idx += 1
        self._pending = None
        self._pending_verdict = None
        if checkpoint_id is not CheckpointID.CP1:
            return JudgeResult(HitType.MISS)
        if not results:
            self._pending_verdict = {"decision_idx": decision_idx, "candidate": False}
            return JudgeResult(HitType.MISS)
        top = results[0]
        decision_id = self.decision_id(decision_idx)
        snapshot = self._registry.decision_snapshot(
            self._key, decision_id, float(top.score), self._delta
        )
        self._pending = snapshot
        verdict = HitType.MISS
        start_t: Optional[float] = None
        for tier in self._spec.tiers:
            theta = snapshot.cuts.get(tier.index)
            if theta is not None and snapshot.score >= theta:
                verdict = HitType.WARM_START
                start_t = tier.start_t
                break
        self._pending_verdict = {
            "decision_idx": decision_idx,
            "candidate": True,
            "verdict": verdict.name,
            "start_t": start_t,
            "winner_id": top.id,
        }
        if verdict is HitType.WARM_START:
            return JudgeResult(HitType.WARM_START, top.id, start_t=start_t)
        return JudgeResult(HitType.MISS, winner_id=top.id)

    # -- feedback ----------------------------------------------------------

    def decision_id(self, decision_idx: int) -> tuple:
        ep = self._episode
        return (
            self._yaml_id,
            ep.get("task_uid") or ep.get("task_key") or "",
            ep.get("attempt"),
            int(decision_idx),
        )

    @property
    def pending_snapshot(self) -> Optional[DecisionSnapshot]:
        return self._pending

    def record_continuation(
        self,
        checkpoint_id: CheckpointID,
        snapshot: Optional[DecisionSnapshot],
        feedback: list[ContinuationFeedback],
        *,
        n_rejected: int = 0,
        fb_batch_size: int = 0,
        invalid_reasons: Optional[list[str]] = None,
    ) -> dict:
        """Commit this decision's feedback and return the hit_meta payload.

        Any rejected observation (non-finite disagreement, missing capture)
        marks the whole stream invalid in the registry: the decision is still
        logged and the action still served, but the arm's terminal state and
        its formal aggregate refuse the stream (plan §3.7 / D15).
        """
        reasons = list(invalid_reasons or [])
        if n_rejected and not reasons:
            reasons.append(f"{n_rejected} rejected observation(s)")
        if reasons:
            self._registry.mark_invalid(self._key, reasons, decision=self._pending_verdict)
        base = dict(self._pending_verdict or {})
        base.update(
            {
                "yaml_id": self._yaml_id,
                "task_uid": self._episode.get("task_uid"),
                "attempt": self._episode.get("attempt"),
                "fb": [
                    {"tier": fb.tier_index, "d": fb.d, "source": fb.source} for fb in feedback
                ],
                "fb_batch_size": int(fb_batch_size),
                "rejected": int(n_rejected),
                "feedback_mode": self._spec.feedback_mode,
                "update_enabled": self._registry.curves(self._key).update_enabled,
                "server_instance_id": self._registry.server_instance_id,
                "delta": self._delta,
            }
        )
        base["flow_invalid"] = self._registry.is_invalid(self._key)
        base["invalid_reasons"] = reasons
        if snapshot is None:
            base.update({"learned": False, "q_pre": None, "cuts": None})
            self._pending = None
            self._pending_verdict = None
            return base
        diag = self._registry.record_batch(
            self._key, snapshot, feedback, n_rejected=n_rejected, episode=self._episode
        )
        base.update(diag)
        base.update(
            {
                "decision_revision": snapshot.decision_revision,
                "q_pre": {str(k): v for k, v in snapshot.q_pre.items()},
                "cuts": {str(k): (None if v is None or not math.isfinite(v) else v) for k, v in snapshot.cuts.items()},
                "cut_available": {str(k): v for k, v in snapshot.cut_available.items()},
                "support_kind": {str(k): v for k, v in snapshot.support_kind.items()},
                "shadowed": list(snapshot.shadowed),
            }
        )
        self._pending = None
        self._pending_verdict = None
        return base


# ------------------------------------------------------------------
# Feedback from first-step updates (shared by the interceptor and the bench)
# ------------------------------------------------------------------


def feedback_from_updates(
    spec: ContinuationSpec,
    payload: CachePayload,
    schedule: DenoiseSchedule,
    *,
    executed: Optional[tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]],
    side: list[tuple[float, torch.Tensor, torch.Tensor]],
) -> tuple[list[ContinuationFeedback], list[str]]:
    """Turn ``(t, x_in_used, x_out)`` pairs into feedback rows.

    ``executed`` is the warm start that ran (``None`` on a MISS); ``side`` are
    the side-evaluated tiers with the input tensor *as the head consumed it*
    (already cast to the head dtype), so an FP32 snapshot stored in the library
    but consumed in BF16 does not turn its rounding into disagreement. Every
    failure to form a finite observation is returned as a reason; the caller
    decides what an invalid observation does to the stream (it is never
    silently dropped).
    """
    n = float(schedule.num_steps)
    out: list[ContinuationFeedback] = []
    reasons: list[str] = []

    def observe(tier: TierSpec, x_in, x_out, source: str) -> None:
        if x_in is None or x_out is None:
            reasons.append(f"{source}:{tier.name}:missing_capture")
            return
        try:
            u_now = (x_out.detach().float().cpu() - x_in.detach().float().cpu()) * n
            u_ref = reference_update(payload, tier.start_t, schedule)
            d = continuation_disagreement(u_now, u_ref, spec.scales[tier.index], spec.masks[tier.index], spec.h_exec)
        except ValueError as exc:
            reasons.append(f"{source}:{tier.name}:{exc}")
            return
        out.append(ContinuationFeedback(tier.index, d, source))

    if executed is not None:
        start_t, x_in, x_out = executed
        observe(spec.tier_by_start_t(start_t), x_in, x_out, SOURCE_EXECUTED)
    for t, x_in, x_out in side:
        observe(spec.tier_by_start_t(t), x_in, x_out, SOURCE_SHADOW)
    return out, reasons


# ------------------------------------------------------------------
# Construction from config
# ------------------------------------------------------------------


def load_update_scales(path: str, tier_indices: list[int]) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict, str]:
    """Read the NPZ written by ``exp/online_rit/library_prep.py``.

    Returns ``(scales, masks, meta, sha256)`` with one ``scale_d_<i>`` /
    ``mask_d_<i>`` pair per tier index; a tier the file does not carry is an
    error, as is a mask without a single executed dimension.
    """
    import io
    import pathlib

    raw = pathlib.Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()

    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta_json"])) if "meta_json" in npz else {}
        scales: dict[int, torch.Tensor] = {}
        masks: dict[int, torch.Tensor] = {}
        for i in tier_indices:
            sk, mk = f"scale_d_{i}", f"mask_d_{i}"
            if sk not in npz or mk not in npz:
                raise ValueError(f"{path}: no scale/mask for tier index {i}")
            mask = torch.as_tensor(np.asarray(npz[mk], dtype=bool))
            scale = torch.as_tensor(np.asarray(npz[sk], dtype=np.float32))
            if mask.dim() != 1 or scale.shape != mask.shape or not bool(mask.any()):
                raise ValueError(f"{path}: malformed scale/mask for tier index {i}")
            if not torch.isfinite(scale[mask]).all():
                raise ValueError(f"{path}: non-finite scale for tier index {i}")
            scales[i], masks[i] = scale, mask
    return scales, masks, meta, sha


def config_fingerprint(cfg, *, schedule: DenoiseSchedule, scales_sha: str, init_sha: Optional[str]) -> str:
    """Digest of every field that must agree between connections sharing one entry."""
    payload = {
        "tiers": [round(float(t), 4) for t in cfg.tiers],
        "alpha": float(cfg.alpha),
        "delta": float(cfg.delta),
        "knots": [float(k) for k in cfg.knots],
        "feedback_mode": cfg.feedback_mode,
        "update_enabled": bool(cfg.update_enabled),
        "window": int(cfg.window),
        "n_min": int(cfg.n_min),
        "h_exec": int(cfg.h_exec),
        "schedule_id": schedule.schedule_id,
        "scales_sha256": scales_sha,
        "init_state_sha256": init_sha,
        "source_library_sha256": cfg.source_library_sha256,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_online_rit_judge(cfg, *, schedule: DenoiseSchedule, registry, yaml_id: str, library_sha256: str) -> OnlineRitJudge:
    """Assemble the judge from a validated ``JudgeConfig`` (called by the config factory)."""
    import pathlib

    specs = tier_specs([float(t) for t in cfg.tiers], schedule)
    tier_indices = [t.index for t in specs]
    scales, masks, meta, scales_sha = load_update_scales(cfg.update_scales_path, tier_indices)
    # The scales were computed on one library; the served library must be that
    # one, or the explicitly declared source of an S3 -> S3b swap.
    scales_lib = meta.get("library_sha256")
    if not scales_lib:
        raise ValueError(f"{cfg.update_scales_path}: meta carries no library_sha256")
    allowed = {scales_lib}
    if cfg.source_library_sha256:
        if cfg.source_library_sha256 != scales_lib:
            raise ValueError("source_library_sha256 does not match the library the scales were computed on")
        allowed.add(library_sha256)
    if library_sha256 not in allowed or (library_sha256 != scales_lib and not cfg.source_library_sha256):
        raise ValueError(
            f"served library {library_sha256[:12]} is not the scales' library {scales_lib[:12]} "
            "and no source_library_sha256 mapping is declared"
        )
    if meta.get("schedule_id") not in (None, schedule.schedule_id):
        raise ValueError("update_scales schedule differs from the served schedule")
    fixed = {"scales_sha256": scales_sha, "schedule_id": schedule.schedule_id, "h_exec": int(cfg.h_exec)}
    init_state = None
    init_sha = None
    if cfg.init_state_path:
        raw = pathlib.Path(cfg.init_state_path).read_bytes()
        init_sha = hashlib.sha256(raw).hexdigest()
        init_state = json.loads(raw.decode("utf-8"))
        verify_state_sha(init_state)
        if [float(k) for k in init_state["knots"]] != [float(k) for k in cfg.knots]:
            raise ValueError("init_state knots differ from the config knots")
        if list(init_state["tier_indices"]) != tier_indices:
            raise ValueError("init_state tier indices differ from the config tiers")
        if float(init_state["alpha"]) != float(cfg.alpha) or int(init_state["window"]) != int(cfg.window) or int(init_state["n_min"]) != int(cfg.n_min):
            raise ValueError("init_state alpha/window/n_min differ from the config")
        got = dict(init_state.get("fixed_params") or {})
        for k, v in fixed.items():
            if got.get(k) != v:
                raise ValueError(f"init_state fixed_params[{k!r}]={got.get(k)!r} differs from the served {v!r}")
    fingerprint = config_fingerprint(cfg, schedule=schedule, scales_sha=scales_sha, init_sha=init_sha)

    def factory() -> OnlineRiskCurves:
        if init_state is not None:
            # from_snapshot verifies both content hashes before any counter reset.
            return OnlineRiskCurves.from_snapshot(init_state, update_enabled=bool(cfg.update_enabled))
        return OnlineRiskCurves(
            knots=[float(k) for k in cfg.knots],
            tier_indices=tier_indices,
            alpha=float(cfg.alpha),
            window=int(cfg.window),
            n_min=int(cfg.n_min),
            update_enabled=bool(cfg.update_enabled),
            fixed_params=fixed,
        )

    key = registry.attach(
        yaml_id=yaml_id,
        library_sha256=library_sha256,
        fingerprint=fingerprint,
        factory=factory,
        snapshot_every=int(cfg.snapshot_every or 200),
        log_dir=cfg.state_log_dir,
    )
    spec = ContinuationSpec(
        tiers=specs,
        scales=scales,
        masks=masks,
        h_exec=int(cfg.h_exec),
        feedback_mode=cfg.feedback_mode,
        schedule=schedule,
    )
    return OnlineRitJudge(registry=registry, registry_key=key, spec=spec, delta=float(cfg.delta), yaml_id=yaml_id)
