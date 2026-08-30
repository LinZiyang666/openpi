"""Budget-mixture value function (confirmation plan section 3.1, estimator
``budget_mixture_v1``).

An operating family is a roster of measured arms. Each arm ``i`` carries the
sufficient statistics of its episodes: total analytic compute ``T``, total
decisions ``D``, total successes ``S`` and the episode count ``E`` (the same
``E`` for every arm of a paired replicate). An episode-level randomized
mixture with weights ``p`` has decision-weighted ratio-of-sums cost
``C(p) = sum p_i T_i / sum p_i D_i`` and success rate ``sum p_i S_i / E``.
The family's value under a compute budget ``B`` is

    V_F(B) = max_p  sum p_i s_i   s.t.  sum p_i (t_i - B d_i) <= 0, sum p = 1, p >= 0

with per-episode means ``t = T/E``, ``d = D/E``, ``s = S/E``. The single
budget constraint makes the optimal basis a single arm or a tight two-arm
mixture; a standalone-dominated arm can still be optimal (its decisions
subsidise an expensive partner), so NOTHING is pruned. Left of the cheapest
arm the family is infeasible (support miss, never extrapolated); once every
arm is feasible the value is the best single SR (budget semantics are
implicit in the ``<=`` constraint, no extra plateau rule).

Pure functions only: no I/O, no randomness. Every analytic evaluation
performs cheap internal invariant checks; a deterministic adaptive Simpson
integral is provided for the frozen audit policy (plan 3.1-4) and is never
substituted for the analytic value.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

VALUE_TOL = 1e-12
BREAKPOINT_TOL_MS = 1e-9
DENOM_TOL = 1e-12
CONTINUITY_TOL = 1e-10
AUDIT_TOL = 1e-8
SIMPSON_ABS_TOL = 1e-10
SIMPSON_MAX_DEPTH = 60
SUPPORT_MISS_AUC = -1.0


class NumericMismatch(RuntimeError):
    """An internal invariant or the frozen numeric audit failed: fail closed."""


@dataclasses.dataclass(frozen=True)
class ArmStats:
    T: float  # total analytic compute (ms) over the arm's episodes
    D: float  # total decisions
    S: float  # total successes
    E: float  # number of episodes

    def __post_init__(self):
        for name in ("T", "D", "S", "E"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
                raise ValueError(f"ArmStats.{name} must be a finite number, got {v!r}")
        if self.T < 0 or self.D <= 0 or self.E <= 0 or self.S < 0 or self.S > self.E:
            raise ValueError(f"ArmStats out of domain: T>=0, D>0, E>0, 0<=S<=E required, got {self}")

    @property
    def t(self) -> float:
        return float(self.T) / float(self.E)

    @property
    def d(self) -> float:
        return float(self.D) / float(self.E)

    @property
    def s(self) -> float:
        return float(self.S) / float(self.E)

    @property
    def c(self) -> float:
        """Standalone ratio-of-sums cost T/D."""
        return float(self.T) / float(self.D)


Basis = tuple[int, ...]


def _arrays(arms: Sequence[str], stats: Mapping[str, ArmStats]):
    if not arms:
        raise ValueError("empty roster")
    if len(set(arms)) != len(arms):
        raise ValueError("roster has duplicate arm ids")
    t, d, s = [], [], []
    for a in arms:
        st = stats[a]
        if not isinstance(st, ArmStats):
            raise ValueError(f"{a}: stats must be ArmStats")
        t.append(st.t)
        d.append(st.d)
        s.append(st.s)
    return t, d, s


def _better(value: float, basis: Basis, best_value, best_basis: Basis | None) -> bool:
    """Canonical tie-break: higher value; within VALUE_TOL fewer arms, then the
    lexicographically smallest tuple of roster indices."""
    if best_value is None:
        return True
    if value > best_value + VALUE_TOL:
        return True
    if value < best_value - VALUE_TOL:
        return False
    if len(basis) != len(best_basis):
        return len(basis) < len(best_basis)
    return basis < best_basis


def _np_arrays(arms: Sequence[str], stats: Mapping[str, ArmStats]):
    t, d, s = _arrays(arms, stats)
    return np.asarray(t, dtype=np.float64), np.asarray(d, dtype=np.float64), np.asarray(s, dtype=np.float64)


def value_at(arms: Sequence[str], stats: Mapping[str, ArmStats], B: float) -> tuple[float | None, Basis]:
    """(V_F(B), canonical basis) or (None, ()) when the family is infeasible at B.

    Vectorised enumeration of every feasible single arm and every tight pair
    (feasible i, infeasible j); ties within VALUE_TOL go to fewer arms, then
    to the lexicographically smallest tuple of roster indices."""
    if not math.isfinite(B):
        raise ValueError("budget must be finite")
    t, d, s = _np_arrays(arms, stats)
    g = t - B * d
    feas = g <= 0.0
    if not feas.any():
        return None, ()
    single_vals = np.where(feas, s, -np.inf)
    best_single = float(single_vals.max())
    i_single = int(np.argmax(single_vals))          # first (smallest index) among exact maxima
    # canonical among singles within tolerance: smallest index
    cand = np.where(feas & (single_vals >= best_single - VALUE_TOL))[0]
    i_single = int(cand.min())
    best_single = float(s[i_single])
    infeas = ~feas
    if not infeas.any():
        return best_single, (i_single,)
    gi = g[feas][:, None]
    gj = g[infeas][None, :]
    p_j = (-gi) / (gj - gi)
    valid = (p_j >= 0.0) & (p_j <= 1.0)
    vals = (1.0 - p_j) * s[feas][:, None] + p_j * s[infeas][None, :]
    vals = np.where(valid, vals, -np.inf)
    if not np.isfinite(vals).any():
        return best_single, (i_single,)
    best_pair = float(vals.max())
    if best_pair <= best_single + VALUE_TOL:
        return best_single, (i_single,)
    fi = np.where(feas)[0]
    fj = np.where(infeas)[0]
    ii, jj = np.where(vals >= best_pair - VALUE_TOL)
    pairs = sorted((min(fi[a], fj[b]), max(fi[a], fj[b])) for a, b in zip(ii, jj))
    return best_pair, tuple(int(x) for x in pairs[0])


def hull_at_zero(arms: Sequence[str], stats: Mapping[str, ArmStats], B: float) -> float | None:
    """Equivalent geometric form (test oracle): the upper concave hull of the
    points (x_i = t_i - B d_i, s_i) maximised over x <= 0."""
    t, d, s = _arrays(arms, stats)
    pts = sorted((t[i] - B * d[i], s[i]) for i in range(len(arms)))
    if pts[0][0] > 0.0:
        return None
    best_single = max(sv for x, sv in pts if x <= 0.0)
    # upper hull (monotone chain) over all points, then value at x = 0
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    v_at_zero = None
    if hull[0][0] <= 0.0 <= hull[-1][0]:
        for (x0, y0), (x1, y1) in zip(hull, hull[1:]):
            if x0 <= 0.0 <= x1:
                v_at_zero = y0 if x1 == x0 else y0 + (y1 - y0) * (0.0 - x0) / (x1 - x0)
                break
        if v_at_zero is None:
            v_at_zero = hull[-1][1]
    return max(best_single, v_at_zero if v_at_zero is not None else best_single)


def feasible(arms: Sequence[str], stats: Mapping[str, ArmStats], B: float) -> bool:
    return min(stats[a].c for a in arms) <= B


def _collinearity_roots(t, d, s, i_idx, j_idx, k_idx) -> np.ndarray:
    """Roots B at which points i, j, k (broadcastable index arrays) are collinear."""
    den = (d[j_idx] - d[i_idx]) * (s[k_idx] - s[i_idx]) - (d[k_idx] - d[i_idx]) * (s[j_idx] - s[i_idx])
    num = (t[j_idx] - t[i_idx]) * (s[k_idx] - s[i_idx]) - (t[k_idx] - t[i_idx]) * (s[j_idx] - s[i_idx])
    with np.errstate(divide="ignore", invalid="ignore"):
        roots = np.where(np.abs(den) < DENOM_TOL, np.nan, num / den)
    return roots


def _merge(values, B_L: float, B_H: float) -> list[float]:
    inside = sorted(float(b) for b in values if math.isfinite(b) and B_L < b < B_H)
    merged: list[float] = []
    for b in inside:
        if merged and abs(b - merged[-1]) <= BREAKPOINT_TOL_MS:
            continue
        if abs(b - B_L) <= BREAKPOINT_TOL_MS or abs(b - B_H) <= BREAKPOINT_TOL_MS:
            continue
        merged.append(b)
    return merged


def breakpoints(arms: Sequence[str], stats: Mapping[str, ArmStats], B_L: float, B_H: float) -> list[float]:
    """FULL candidate breakpoint set in (B_L, B_H): every single-arm cost and
    every three-point collinearity root (O(K^3), vectorised). Used as the
    reference enumeration in tests and by the Simpson audit; the production
    piece sweep below only needs the events that can change the active basis."""
    if not (math.isfinite(B_L) and math.isfinite(B_H) and B_H > B_L):
        raise ValueError("invalid budget interval")
    t, d, s = _np_arrays(arms, stats)
    n = len(arms)
    cand = list(t / d)
    if n >= 3:
        i, j, k = np.array([(a, b, c) for a in range(n) for b in range(a + 1, n) for c in range(b + 1, n)]).T
        cand.extend(_collinearity_roots(t, d, s, i, j, k).tolist())
    return _merge(cand, B_L, B_H)


def _next_event(t, d, s, basis: Basis, cur: float, B_H: float) -> float:
    """Smallest budget > cur at which the active basis can change: a single
    arm crossing x = 0 (its cost) or a collinearity involving a basis arm."""
    n = len(t)
    events = [float(x) for x in (t / d) if cur + BREAKPOINT_TOL_MS < x < B_H]
    if n >= 3:
        others = np.arange(n)
        for b in basis:
            rest = others[others != b]
            jj, kk = np.array([(x, y) for x in rest for y in rest if x < y]).T if len(rest) >= 2 else (np.array([], int), np.array([], int))
            if len(jj):
                roots = _collinearity_roots(t, d, s, np.full(len(jj), b), jj, kk)
                events.extend(float(x) for x in roots if math.isfinite(x) and cur + BREAKPOINT_TOL_MS < x < B_H)
    return min(events) if events else B_H


@dataclasses.dataclass(frozen=True)
class Piece:
    lo: float
    hi: float
    basis: Basis      # roster indices, sorted
    cheap: int        # index of the arm feasible on the piece (pair pieces); equals basis[0] for singles
    dear: int | None  # the infeasible partner (pair pieces) or None


def _piece_value(piece: Piece, t, d, s, B: float) -> float:
    if piece.dear is None:
        return s[piece.cheap]
    i, j = piece.cheap, piece.dear
    gamma = d[i] - d[j]
    delta = t[j] - t[i]
    den = gamma * B + delta
    if abs(den) < DENOM_TOL:
        raise NumericMismatch("pair piece denominator vanished")
    p_j = (B * d[i] - t[i]) / den
    return s[i] + (s[j] - s[i]) * p_j


def _make_piece(lo: float, hi: float, basis: Basis, t, d) -> Piece:
    if len(basis) == 1:
        return Piece(lo, hi, basis, basis[0], None)
    i, j = basis
    ci, cj = t[i] / d[i], t[j] / d[j]
    cheap, dear = (i, j) if ci <= cj else (j, i)
    return Piece(lo, hi, basis, cheap, dear)


def pieces(arms: Sequence[str], stats: Mapping[str, ArmStats], B_L: float, B_H: float,
           *, full_enumeration: bool = False) -> list[Piece]:
    """Partition [B_L, B_H] into open pieces of constant canonical basis.

    Production path: an event sweep — from the current piece start the next
    basis change can only occur at a single-arm cost crossing or at a
    collinearity involving an arm of the current basis, so only those roots
    are solved (O(K^2) per piece). ``full_enumeration=True`` uses every
    O(K^3) root (reference for tests). Both are verified by the same
    invariants: the piece formula reproduces V at both ends, pieces are
    contiguous and cover the interval. Raises NumericMismatch on failure."""
    if not feasible(arms, stats, B_L):
        raise NumericMismatch("family infeasible at B_L; pieces undefined")
    t, d, s = _arrays(arms, stats)
    tn, dn, sn = _np_arrays(arms, stats)
    if full_enumeration:
        bps = [B_L] + breakpoints(arms, stats, B_L, B_H) + [B_H]
    else:
        bps = [B_L]
        cur = B_L
        guard = 0
        while cur < B_H:
            guard += 1
            if guard > 100000:
                raise NumericMismatch("piece sweep did not terminate")
            probe = min(B_H, cur + max(BREAKPOINT_TOL_MS * 10.0, (B_H - cur) * 1e-9))
            _v, basis = value_at(arms, stats, probe)
            if not basis:
                raise NumericMismatch("family infeasible inside the interval")
            nxt = _next_event(tn, dn, sn, basis, cur, B_H)
            if nxt <= cur:
                nxt = B_H
            bps.append(nxt)
            cur = nxt
    out: list[Piece] = []
    for lo, hi in zip(bps, bps[1:]):
        if not (hi > lo):
            raise NumericMismatch("non-positive piece length")
        mid = 0.5 * (lo + hi)
        v, basis = value_at(arms, stats, mid)
        if v is None:
            raise NumericMismatch("family infeasible inside the interval")
        piece = _make_piece(lo, hi, basis, t, d)
        # invariants: the piece formula must reproduce V at both ends and at
        # two interior probes (a missed event would break at least one of them)
        for x in (lo, lo + 0.25 * (hi - lo), mid + 0.25 * (hi - lo), hi):
            vx, _ = value_at(arms, stats, x)
            fx = _piece_value(piece, t, d, s, x)
            if vx is None or abs(vx - fx) > CONTINUITY_TOL:
                raise NumericMismatch(f"piece [{lo}, {hi}] does not match V at {x}")
        out.append(piece)
    if abs(out[0].lo - B_L) > 0 or abs(out[-1].hi - B_H) > 0:
        raise NumericMismatch("pieces do not cover the interval")
    for a, b in zip(out, out[1:]):
        if a.hi != b.lo:
            raise NumericMismatch("pieces are not contiguous")
    return out


def _integrate_piece(piece: Piece, t, d, s) -> float:
    lo, hi = piece.lo, piece.hi
    if piece.dear is None:
        return s[piece.cheap] * (hi - lo)
    i, j = piece.cheap, piece.dear
    alpha, beta = d[i], -t[i]
    gamma, delta = d[i] - d[j], t[j] - t[i]
    for x in (lo, hi):
        if abs(gamma * x + delta) < DENOM_TOL:
            raise NumericMismatch("pair piece denominator vanished at an endpoint")
    if abs(gamma) < DENOM_TOL:
        frac = (alpha * (hi * hi - lo * lo) / 2.0 + beta * (hi - lo)) / delta
    else:
        c1 = alpha / gamma
        c2 = (beta - alpha * delta / gamma) / gamma
        frac = c1 * (hi - lo) + c2 * (math.log(abs(gamma * hi + delta)) - math.log(abs(gamma * lo + delta)))
    return s[i] * (hi - lo) + (s[j] - s[i]) * frac


def auc_norm(arms: Sequence[str], stats: Mapping[str, ArmStats], B_L: float, B_H: float) -> float:
    """Exact normalised integral of V_F over [B_L, B_H]; NumericMismatch if infeasible."""
    t, d, s = _arrays(arms, stats)
    total = sum(_integrate_piece(p, t, d, s) for p in pieces(arms, stats, B_L, B_H))
    return total / (B_H - B_L)


def auc_with_support(arms_a: Sequence[str], stats_a: Mapping[str, ArmStats],
                     arms_b: Sequence[str], stats_b: Mapping[str, ArmStats],
                     B_L: float, B_H: float) -> tuple[float, bool]:
    """AUC_norm(A) - AUC_norm(B), or (SUPPORT_MISS_AUC, True) when either family
    is infeasible at B_L. The miss value is kept in the distribution."""
    if not (math.isfinite(B_L) and math.isfinite(B_H) and B_H > B_L):
        raise ValueError("invalid budget interval")
    if not feasible(arms_a, stats_a, B_L) or not feasible(arms_b, stats_b, B_L):
        return SUPPORT_MISS_AUC, True
    return auc_norm(arms_a, stats_a, B_L, B_H) - auc_norm(arms_b, stats_b, B_L, B_H), False


def active_basis_union(arms: Sequence[str], stats: Mapping[str, ArmStats], B_L: float, B_H: float) -> dict:
    """Union of the canonical bases of every positive-length piece (roster
    order) plus the endpoint bases, kept separately."""
    ps = pieces(arms, stats, B_L, B_H)
    active: set[int] = set()
    for p in ps:
        active.update(p.basis)
    _, b_lo = value_at(arms, stats, B_L)
    _, b_hi = value_at(arms, stats, B_H)
    return {
        "active": [arms[i] for i in sorted(active)],
        "endpoint_bases": {"B_L": [arms[i] for i in b_lo], "B_H": [arms[i] for i in b_hi]},
        "pieces": [{"lo": p.lo, "hi": p.hi, "basis": [arms[i] for i in p.basis]} for p in ps],
    }


def bitset_bytes(arms: Sequence[str], active: Sequence[str]) -> bytes:
    """Fixed-length big-endian bitset in roster order (bit 0 = first arm = MSB of byte 0)."""
    n = len(arms)
    nbytes = (n + 7) // 8
    idx = {a: k for k, a in enumerate(arms)}
    val = 0
    for a in active:
        val |= 1 << (8 * nbytes - 1 - idx[a])
    return val.to_bytes(nbytes, "big")


def bitset_rollup_sha256(bitsets: Sequence[bytes]) -> str:
    h = hashlib.sha256()
    for b in bitsets:
        h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- extrema

def _piece_derivative_params(piece: Piece, t, d, s):
    """Return (k, gamma, delta) so that f'(B) = k / (gamma B + delta)^2 (k = 0 for singles)."""
    if piece.dear is None:
        return 0.0, 0.0, 1.0
    i, j = piece.cheap, piece.dear
    alpha, beta = d[i], -t[i]
    gamma, delta = d[i] - d[j], t[j] - t[i]
    return (s[j] - s[i]) * (alpha * delta - beta * gamma), gamma, delta


def difference_extrema(arms_a, stats_a, arms_b, stats_b, B_L: float, B_H: float) -> dict:
    """Exact extrema of V_A(B) - V_B(B) on [B_L, B_H]: candidates are the union
    of both families' breakpoints and the closed-form stationary points of the
    difference of two linear-fractional pieces."""
    ta, da, sa = _arrays(arms_a, stats_a)
    tb, db, sb = _arrays(arms_b, stats_b)
    pa, pb = pieces(arms_a, stats_a, B_L, B_H), pieces(arms_b, stats_b, B_L, B_H)
    cuts = sorted({B_L, B_H} | {p.lo for p in pa} | {p.hi for p in pa} | {p.lo for p in pb} | {p.hi for p in pb})
    cand: list[float] = list(cuts)
    stationary: list[float] = []

    def find(pcs, x):
        for p in pcs:
            if p.lo <= x <= p.hi:
                return p
        raise NumericMismatch("point outside pieces")

    for lo, hi in zip(cuts, cuts[1:]):
        mid = 0.5 * (lo + hi)
        qa, qb = find(pa, mid), find(pb, mid)
        k1, g1, e1 = _piece_derivative_params(qa, ta, da, sa)
        k2, g2, e2 = _piece_derivative_params(qb, tb, db, sb)
        if k1 == 0.0 and k2 == 0.0:
            continue
        # k1/(g1 B + e1)^2 = k2/(g2 B + e2)^2  ->  sqrt(k1) (g2 B + e2) = +- sqrt(k2) (g1 B + e1)
        if k1 * k2 <= 0.0:
            continue  # one side has zero/opposite slope: difference monotone on the piece
        r1, r2 = math.sqrt(abs(k1)), math.sqrt(abs(k2))
        for sign in (1.0, -1.0):
            den = r1 * g2 - sign * r2 * g1
            if abs(den) < DENOM_TOL:
                continue
            x = (sign * r2 * e1 - r1 * e2) / den
            if lo < x < hi:
                stationary.append(x)
                cand.append(x)
    vals = []
    for x in sorted(set(cand)):
        va, _ = value_at(arms_a, stats_a, x)
        vb, _ = value_at(arms_b, stats_b, x)
        vals.append((x, va - vb))
    i_min = min(range(len(vals)), key=lambda k: vals[k][1])
    i_max = max(range(len(vals)), key=lambda k: vals[k][1])
    return {"breakpoints": [x for x, _ in vals], "stationary_points": sorted(stationary),
            "diffs": [v for _, v in vals],
            "min": vals[i_min][1], "argmin_budget": vals[i_min][0],
            "max": vals[i_max][1], "argmax_budget": vals[i_max][0],
            "a_dominated": bool(vals[i_max][1] <= 0.0), "b_dominated": bool(vals[i_min][1] >= 0.0)}


# ------------------------------------------------------ step envelope (sensitivity)

def step_value(arms: Sequence[str], stats: Mapping[str, ArmStats], B: float) -> float | None:
    """Measured-policy-only envelope max{s_i : T_i/D_i <= B} (no mixing)."""
    vals = [stats[a].s for a in arms if stats[a].c <= B]
    return max(vals) if vals else None


def step_auc_norm(arms: Sequence[str], stats: Mapping[str, ArmStats], B_L: float, B_H: float) -> float:
    if not feasible(arms, stats, B_L):
        raise NumericMismatch("family infeasible at B_L")
    cuts = sorted({B_L, B_H} | {stats[a].c for a in arms if B_L < stats[a].c < B_H})
    total = 0.0
    for lo, hi in zip(cuts, cuts[1:]):
        v = step_value(arms, stats, 0.5 * (lo + hi))
        total += v * (hi - lo)
    return total / (B_H - B_L)


def step_auc_with_support(arms_a, stats_a, arms_b, stats_b, B_L: float, B_H: float) -> tuple[float, bool]:
    if not feasible(arms_a, stats_a, B_L) or not feasible(arms_b, stats_b, B_L):
        return SUPPORT_MISS_AUC, True
    return step_auc_norm(arms_a, stats_a, B_L, B_H) - step_auc_norm(arms_b, stats_b, B_L, B_H), False


# ------------------------------------------------------------- numeric audit

def simpson_auc_norm(arms: Sequence[str], stats: Mapping[str, ArmStats], B_L: float, B_H: float,
                     *, abs_tol: float = SIMPSON_ABS_TOL, max_depth: int = SIMPSON_MAX_DEPTH) -> float:
    """Deterministic adaptive Simpson over V_F (audit only; never the estimand)."""
    if not feasible(arms, stats, B_L):
        raise NumericMismatch("family infeasible at B_L")

    def f(x: float) -> float:
        v, _ = value_at(arms, stats, x)
        if v is None:
            raise NumericMismatch("infeasible inside interval during audit")
        return v

    def simpson(a, fa, b, fb, m, fm):
        return (b - a) / 6.0 * (fa + 4.0 * fm + fb)

    def rec(a, fa, b, fb, m, fm, whole, tol, depth):
        lm, rm = 0.5 * (a + m), 0.5 * (m + b)
        flm, frm = f(lm), f(rm)
        left = simpson(a, fa, m, fm, lm, flm)
        right = simpson(m, fm, b, fb, rm, frm)
        if depth >= max_depth or abs(left + right - whole) <= 15.0 * tol:
            return left + right + (left + right - whole) / 15.0
        return rec(a, fa, m, fm, lm, flm, left, tol / 2.0, depth + 1) + rec(m, fm, b, fb, rm, frm, right, tol / 2.0, depth + 1)

    # integrate piecewise between the analytic breakpoints so the kinks are on nodes
    cuts = [B_L] + breakpoints(arms, stats, B_L, B_H) + [B_H]
    total = 0.0
    for a, b in zip(cuts, cuts[1:]):
        m = 0.5 * (a + b)
        fa, fb, fm = f(a), f(b), f(m)
        total += rec(a, fa, b, fb, m, fm, simpson(a, fa, b, fb, m, fm), abs_tol, 0)
    return total / (B_H - B_L)


def audit_family(arms: Sequence[str], stats: Mapping[str, ArmStats], B_L: float, B_H: float) -> dict:
    """Compare the analytic integral with Simpson; NumericMismatch beyond AUDIT_TOL."""
    analytic = auc_norm(arms, stats, B_L, B_H)
    numeric = simpson_auc_norm(arms, stats, B_L, B_H)
    if abs(analytic - numeric) > AUDIT_TOL:
        raise NumericMismatch(f"analytic {analytic!r} vs Simpson {numeric!r} differ by more than {AUDIT_TOL}")
    return {"analytic": analytic, "simpson": numeric, "abs_diff": abs(analytic - numeric)}


def standalone_dominance(arms: Sequence[str], stats: Mapping[str, ArmStats]) -> dict[str, bool]:
    """Descriptive only: is the arm weakly dominated on (cost, SR) by another arm?
    Never used for pruning."""
    out = {}
    for a in arms:
        ca, sa = stats[a].c, stats[a].s
        out[a] = any((stats[b].c <= ca and stats[b].s >= sa) and (stats[b].c < ca or stats[b].s > sa)
                     for b in arms if b != a)
    return out


def family_replicate(arms: Sequence[str], stats: Mapping[str, ArmStats], B_L: float, B_H: float) -> tuple[float | None, list[str]]:
    """One pieces() pass per replicate: (AUC_norm, active arms) or (None, []) if infeasible."""
    if not feasible(arms, stats, B_L):
        return None, []
    t, d, s = _arrays(arms, stats)
    ps = pieces(arms, stats, B_L, B_H)
    total = sum(_integrate_piece(p, t, d, s) for p in ps)
    active: set[int] = set()
    for p in ps:
        active.update(p.basis)
    return total / (B_H - B_L), [arms[i] for i in sorted(active)]
