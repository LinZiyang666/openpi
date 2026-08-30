"""The single H1 adjudication implementation (confirmation plan 3.4 / 3.6).

``evaluate_h1_verdict`` is called by the development amendment analyzer, by
the full-adjudication power Monte Carlo and by the confirmation analyzer.
There is exactly one implementation of the composite decision

    pass  <=>  left_support_ok  and  joint_miss <= max_joint_miss  and  q05 > 0

so that power is estimated for the very rule the confirmation applies.

Inputs are paired per-cell sufficient statistics (cost sum, decisions,
success) for every roster arm over the same cell grid, plus a task-
stratified paired bootstrap index (R x n_cells). Each replicate aggregates
``(T, D, S, E)`` per arm and evaluates the budget-mixture AUC difference.
The numeric audit policy (plan 3.1-4) is applied to the full sample and to
the caller-supplied subset of replicates; a mismatch fails closed.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Mapping, Sequence

import numpy as np

from exp.dispatch_surface.analysis import budget_mixture as bm
from exp.dispatch_surface.analysis.estimator_version import budget_mixture_digest

Z_095 = 1.6449
MAX_JOINT_MISS = 0.01


@dataclasses.dataclass(frozen=True)
class FrozenDesign:
    """Everything the verdict depends on besides the data."""
    family_a: str
    family_b: str
    roster: Mapping[str, Sequence[str]]   # family -> arms in frozen roster order
    B_L: float
    B_H: float
    R: int
    max_joint_miss: float = MAX_JOINT_MISS
    estimator_digest: str = dataclasses.field(default_factory=budget_mixture_digest)

    def __post_init__(self):
        if self.estimator_digest != budget_mixture_digest():
            raise SystemExit("FrozenDesign estimator digest != budget_mixture_v1")
        if not (math.isfinite(self.B_L) and math.isfinite(self.B_H) and self.B_H > self.B_L):
            raise SystemExit("FrozenDesign has an invalid budget interval")
        for fam in (self.family_a, self.family_b):
            if fam not in self.roster or not self.roster[fam]:
                raise SystemExit(f"FrozenDesign roster lacks family {fam!r}")
        if int(self.R) <= 0:
            raise SystemExit("FrozenDesign R must be positive")


@dataclasses.dataclass
class CellArrays:
    """Per-arm arrays aligned to a sorted cell list."""
    cells: list
    cost: dict[str, np.ndarray]
    dec: dict[str, np.ndarray]
    succ: dict[str, np.ndarray]


def cell_arrays(cells_by_arm: Mapping[str, Mapping], arms: Sequence[str]) -> CellArrays:
    """cells_by_arm[arm][cell] = (cost_sum, n_decisions, success)."""
    grid = None
    for a in arms:
        keys = set(cells_by_arm[a])
        if grid is None:
            grid = keys
        elif keys != grid:
            raise SystemExit(f"arm {a} does not cover the same paired grid")
    cells = sorted(grid)
    cost, dec, succ = {}, {}, {}
    for a in arms:
        cost[a] = np.array([float(cells_by_arm[a][c][0]) for c in cells], dtype=np.float64)
        dec[a] = np.array([float(cells_by_arm[a][c][1]) for c in cells], dtype=np.float64)
        succ[a] = np.array([float(cells_by_arm[a][c][2]) for c in cells], dtype=np.float64)
        if (dec[a] <= 0).any():
            raise SystemExit(f"arm {a} has a cell with no decisions")
        if ((succ[a] != 0.0) & (succ[a] != 1.0)).any():
            raise SystemExit(f"arm {a} success must be 0/1 per cell")
    return CellArrays(cells, cost, dec, succ)


def stats_for_index(arrays: CellArrays, arms: Sequence[str], idx: np.ndarray) -> dict[str, bm.ArmStats]:
    """Aggregate (T, D, S, E) over the resampled cells ``idx`` (1-D int array)."""
    E = float(len(idx))
    return {a: bm.ArmStats(float(arrays.cost[a][idx].sum()), float(arrays.dec[a][idx].sum()),
                           float(arrays.succ[a][idx].sum()), E) for a in arms}


def full_sample_stats(arrays: CellArrays, arms: Sequence[str]) -> dict[str, bm.ArmStats]:
    return stats_for_index(arrays, arms, np.arange(len(arrays.cells)))


@dataclasses.dataclass
class Verdict:
    passed: bool
    reason: str
    effect: float | None          # full-sample plug-in AUC difference (None if infeasible)
    mean: float
    q05: float
    q95: float
    sd: float
    joint_miss: float
    left_support_ok: bool
    step_effect: float | None
    step_q05: float
    step_q95: float
    replicate_values: np.ndarray
    active_bitsets: dict[str, list[bytes]]
    active_freq: dict[str, dict[str, float]]
    audit: dict


def evaluate_hypothesis(cells_by_arm: Mapping[str, Mapping], design: FrozenDesign,
                        index: np.ndarray, *, audit_replicates: Sequence[int] = ()) -> Verdict:
    arms_a = list(design.roster[design.family_a])
    arms_b = list(design.roster[design.family_b])
    arrays = cell_arrays(cells_by_arm, arms_a + arms_b)
    index = np.asarray(index)
    if index.ndim != 2 or index.shape[0] != design.R or index.shape[1] != len(arrays.cells):
        raise SystemExit(f"bootstrap index must be ({design.R}, {len(arrays.cells)}), got {index.shape}")
    B_L, B_H = design.B_L, design.B_H
    # --- full sample ---
    full = full_sample_stats(arrays, arms_a + arms_b)
    left_ok = bm.feasible(arms_a, full, B_L) and bm.feasible(arms_b, full, B_L)
    effect = None
    step_effect = None
    audit = {"full_sample": None, "replicates": []}
    if left_ok:
        effect = bm.auc_norm(arms_a, full, B_L, B_H) - bm.auc_norm(arms_b, full, B_L, B_H)
        step_effect = bm.step_auc_norm(arms_a, full, B_L, B_H) - bm.step_auc_norm(arms_b, full, B_L, B_H)
        audit["full_sample"] = {design.family_a: bm.audit_family(arms_a, full, B_L, B_H),
                                design.family_b: bm.audit_family(arms_b, full, B_L, B_H)}
    # --- replicates ---
    R = design.R
    vals = np.empty(R)
    step_vals = np.empty(R)
    bits_a: list[bytes] = []
    bits_b: list[bytes] = []
    cnt_a = {a: 0 for a in arms_a}
    cnt_b = {a: 0 for a in arms_b}
    audit_set = set(int(r) for r in audit_replicates)
    misses = 0
    for r in range(R):
        st = stats_for_index(arrays, arms_a + arms_b, index[r])
        auc_a, ua = bm.family_replicate(arms_a, st, B_L, B_H)
        auc_b, ub = bm.family_replicate(arms_b, st, B_L, B_H)
        miss = auc_a is None or auc_b is None
        v = bm.SUPPORT_MISS_AUC if miss else auc_a - auc_b
        sv, _ = bm.step_auc_with_support(arms_a, st, arms_b, st, B_L, B_H)
        vals[r] = v
        step_vals[r] = sv
        if miss:
            misses += 1
            bits_a.append(bm.bitset_bytes(arms_a, []))
            bits_b.append(bm.bitset_bytes(arms_b, []))
            continue
        for a in ua:
            cnt_a[a] += 1
        for a in ub:
            cnt_b[a] += 1
        bits_a.append(bm.bitset_bytes(arms_a, ua))
        bits_b.append(bm.bitset_bytes(arms_b, ub))
        if r in audit_set:
            audit["replicates"].append({"r": r, design.family_a: bm.audit_family(arms_a, st, B_L, B_H),
                                        design.family_b: bm.audit_family(arms_b, st, B_L, B_H)})
    joint_miss = misses / R
    q05 = float(np.quantile(vals, 0.05))
    q95 = float(np.quantile(vals, 0.95))
    sd = float(np.std(vals, ddof=1)) if R > 1 else float("nan")
    support_ok = left_ok and joint_miss <= design.max_joint_miss
    passed = bool(support_ok and q05 > 0.0)
    if not left_ok:
        reason = "left_support_fail"
    elif joint_miss > design.max_joint_miss:
        reason = "joint_miss_exceeds"
    elif q05 > 0.0:
        reason = "q05_positive"
    else:
        reason = "q05_not_positive"
    return Verdict(
        passed=passed, reason=reason, effect=effect, mean=float(vals.mean()), q05=q05, q95=q95, sd=sd,
        joint_miss=joint_miss, left_support_ok=left_ok, step_effect=step_effect,
        step_q05=float(np.quantile(step_vals, 0.05)), step_q95=float(np.quantile(step_vals, 0.95)),
        replicate_values=vals,
        active_bitsets={design.family_a: bits_a, design.family_b: bits_b},
        active_freq={design.family_a: {a: cnt_a[a] / R for a in arms_a},
                     design.family_b: {a: cnt_b[a] / R for a in arms_b}},
        audit=audit,
    )


def evaluate_h1_verdict(cells_by_arm: Mapping[str, Mapping], design: FrozenDesign,
                        index: np.ndarray, *, audit_replicates: Sequence[int] = ()) -> Verdict:
    """H1 = family_a (SV) vs family_b (threshold) under the frozen composite rule."""
    if design.family_a != "sv" or design.family_b != "threshold":
        raise SystemExit("H1 is SV vs threshold; use evaluate_hypothesis for descriptive pairs")
    return evaluate_hypothesis(cells_by_arm, design, index, audit_replicates=audit_replicates)


def verdict_summary(v: Verdict) -> dict:
    """JSON-serialisable summary (no arrays)."""
    return {
        "passed": v.passed, "reason": v.reason, "effect_plugin": v.effect, "bootstrap_mean": v.mean,
        "bootstrap_q05": v.q05, "bootstrap_q95": v.q95, "sd": v.sd, "joint_miss": v.joint_miss,
        "left_support_ok": v.left_support_ok,
        "step_envelope": {"effect_plugin": v.step_effect, "q05": v.step_q05, "q95": v.step_q95},
        "active_freq": v.active_freq,
        "active_bitset_rollup_sha256": {fam: bm.bitset_rollup_sha256(b) for fam, b in v.active_bitsets.items()},
        "audit": v.audit,
    }


def audit_replicate_indices(R: int, *, estimator_digest: str, input_digest: str, max_count: int = 100) -> list[int]:
    """Frozen audit subset (plan 3.1-4b): up to ``max_count`` replicates drawn
    without replacement from a stream seeded by sha256(estimator|input)."""
    import hashlib
    seed = int.from_bytes(hashlib.sha256(f"{estimator_digest}|{input_digest}".encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    k = min(max_count, R)
    return sorted(int(x) for x in rng.choice(R, size=k, replace=False))
