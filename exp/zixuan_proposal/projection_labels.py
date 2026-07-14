"""Compatibility-label + mask + fold core for TRACER Phase-6 projection training.

Plan §A (frozen labels) and §C (folds). This module holds the pure, testable core —
the c^A / c^X / c_ab compatibility construction, the symmetric P(a)/N(a) masks, and the
deterministic per-task folds — separate from the env-coupled pkl loading and identity
resolution (which live in ``build_projection_trainset.py`` / ``build_l10_dplus_resolver.py``).

Key frozen decisions (plan §A/§C):
- Action whitening: only the 7 active DOFs (``library_stats.action_active_mask``),
  divided by ``library_stats.action_sigma``; flatten 7 x H; read H from the array.
- c^A (Eq 10) = exp(-||.||^2 / sigma_A^2); c^X (Eq 11) = max over matched tau of
  exp(-||.||^2 / sigma_X^2); c_ab = eta*c^A + (1-eta)*c^X, eta in {1.0, 0.5}.
- sigma_A^2/sigma_X^2 = median of squared cross-init distances over train+val folds
  ONLY (degeneracy guard: fall back to 1.0 iff median <= 1e-8).
- P(a) = {c>=rho_+ AND y_a=+1 AND y_b=+1 AND id_a!=id_b} (SUCCESS-SUCCESS -> symmetric).
  N(a) = {c<=rho_- AND id_a!=id_b} (outcome-agnostic -> breaks the batch shortcut).
  Gray-zone (rho_-<c<rho_+) is in neither. rho_+/rho_- = 90th/40th pct of the
  cross-init success-success c on train+val, with a >=50%-valid-anchor guard.
- Folds by resolved (task_id, orig_init_state_idx): per task sort even-init D+ ascending;
  n>=3 -> [n-1]=test, [n-2]=val, rest=train; n<3 -> train-only (represented = n>=3 tasks).
"""

from __future__ import annotations

import dataclasses

import numpy as np

SIGMA_DEGEN_EPS = 1e-8
SIGMA_FALLBACK = 1.0
RHO_PLUS_PCT = 90.0
RHO_MINUS_PCT = 40.0
RHO_PLUS_FALLBACK = 0.6
RHO_MINUS_FALLBACK = 0.3
MIN_VALID_ANCHOR_FRAC = 0.5


@dataclasses.dataclass(frozen=True)
class Entry:
    """A resolved cache entry used for label construction.

    ident:        resolved identity (task_id, orig_init_state_idx) -- LOEO/mask key.
    outcome:      +1 (D+) or -1 (D-).
    action_flat:  whitened flattened action chunk [7*H].
    snap_flat:    dict tau -> whitened flattened denoise snapshot [7*H] (may be empty).
    """

    ident: tuple
    outcome: int
    action_flat: np.ndarray
    snap_flat: dict


# ------------------------------------------------------------------
# Whitening
# ------------------------------------------------------------------
def whiten_flatten(chunk: np.ndarray, active_mask: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Whiten a [H, action_dim] chunk on active DOFs and flatten to [n_active*H].

    Divides active dims by ``sigma`` (guarded), drops inactive dims. H is read from the
    array, never assumed.
    """
    chunk = np.asarray(chunk, dtype=np.float64)  # [H, action_dim]
    active = np.asarray(active_mask, dtype=bool)
    sig = np.asarray(sigma, dtype=np.float64)
    w = chunk[:, active] / (sig[active] + 1e-8)  # [H, n_active]
    return w.reshape(-1)


# ------------------------------------------------------------------
# Bandwidths + compatibility (plan §A)
# ------------------------------------------------------------------
def _median_sq_dist(vectors: list[np.ndarray], idents: list[tuple], rng: np.random.Generator, n_pairs: int) -> float:
    """Median squared L2 over cross-init pairs (sampled). Degeneracy -> fallback."""
    m = len(vectors)
    if m < 2:
        return SIGMA_FALLBACK
    dists = []
    tries = 0
    max_tries = n_pairs * 5
    while len(dists) < n_pairs and tries < max_tries:
        i, j = rng.integers(0, m, size=2)
        tries += 1
        if i == j or idents[i] == idents[j]:
            continue
        d = vectors[i] - vectors[j]
        dists.append(float(d @ d))
    if not dists:
        return SIGMA_FALLBACK
    med = float(np.median(dists))
    return med if med > SIGMA_DEGEN_EPS else SIGMA_FALLBACK


def fit_bandwidths(entries: list[Entry], *, eta: float, seed: int = 7, n_pairs: int = 200_000) -> dict:
    """Fit sigma_A^2 (and sigma_X^2 if eta<1) from D+ entries (train+val callers only)."""
    rng = np.random.default_rng(seed)
    dplus = [e for e in entries if e.outcome == 1]
    sig_a = _median_sq_dist([e.action_flat for e in dplus], [e.ident for e in dplus], rng, n_pairs)
    out = {"sigma_A_sq": sig_a, "eta": eta}
    if eta < 1.0:
        # pool all matched-tau snapshot pairs into one scalar (single pooled sigma_X)
        snap_vecs, snap_idents = [], []
        for e in dplus:
            for v in e.snap_flat.values():
                snap_vecs.append(v)
                snap_idents.append(e.ident)
        out["sigma_X_sq"] = _median_sq_dist(snap_vecs, snap_idents, rng, n_pairs)
    return out


def compat_matrix(entries: list[Entry], bw: dict) -> np.ndarray:
    """Dense c_ab matrix (Eq 10-12). c^X uses the matched-tau intersection (max over tau)."""
    n = len(entries)
    a = np.stack([e.action_flat for e in entries])  # [n, D]
    # c^A: exp(-||a_i - a_j||^2 / sigma_A^2)
    sq = np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=-1)  # [n, n]
    c_a = np.exp(-sq / bw["sigma_A_sq"])
    eta = bw["eta"]
    if eta >= 1.0:
        return c_a
    c_x = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            taus = set(entries[i].snap_flat) & set(entries[j].snap_flat)
            if not taus:
                c_x[i, j] = 0.0
                continue
            best = max(
                np.exp(-float(np.sum((entries[i].snap_flat[t] - entries[j].snap_flat[t]) ** 2)) / bw["sigma_X_sq"])
                for t in taus
            )
            c_x[i, j] = best
    return eta * c_a + (1.0 - eta) * c_x


# ------------------------------------------------------------------
# Thresholds + symmetric P/N masks (plan §A)
# ------------------------------------------------------------------
def fit_thresholds(entries: list[Entry], c: np.ndarray) -> tuple[float, float]:
    """rho_+/rho_- = 90/40 pct of cross-init success-success c; guarded fallback."""
    vals = []
    for i, ei in enumerate(entries):
        if ei.outcome != 1:
            continue
        for j, ej in enumerate(entries):
            if j <= i or ej.outcome != 1 or ei.ident == ej.ident:
                continue
            vals.append(c[i, j])
    if len(vals) < 2:
        return RHO_PLUS_FALLBACK, RHO_MINUS_FALLBACK
    rp = float(np.percentile(vals, RHO_PLUS_PCT))
    rm = float(np.percentile(vals, RHO_MINUS_PCT))
    if not (rp > rm):
        return RHO_PLUS_FALLBACK, RHO_MINUS_FALLBACK
    return rp, rm


def build_masks(entries: list[Entry], c: np.ndarray, rho_plus: float, rho_minus: float):
    """Symmetric P(a)/N(a) masks (plan §A).

    P = success-success, c>=rho_+, different identity (symmetric because both endpoints
    are successes). N = c<=rho_-, different identity, outcome-agnostic (symmetric).
    Gray-zone pairs are in neither. Returns (pos_mask, neg_mask) bool [n, n].
    """
    n = len(entries)
    y_pos = np.array([e.outcome == 1 for e in entries])
    idents = [e.ident for e in entries]
    same_id = np.array([[idents[i] == idents[j] for j in range(n)] for i in range(n)])
    diff = ~same_id & ~np.eye(n, dtype=bool)
    both_pos = np.outer(y_pos, y_pos)
    pos = (c >= rho_plus) & both_pos & diff
    neg = (c <= rho_minus) & diff
    neg = neg & ~pos  # disjointness: a pair cannot be both
    # Symmetrize defensively (float pct thresholds keep it symmetric already).
    pos = pos & pos.T
    neg = neg & neg.T
    return pos, neg


def valid_anchor_fraction(pos: np.ndarray, neg: np.ndarray) -> float:
    """Fraction of anchors with >=1 positive AND >=1 negative."""
    n = pos.shape[0]
    ok = (pos.any(axis=1) & neg.any(axis=1)).sum()
    return float(ok) / n if n else 0.0


# ------------------------------------------------------------------
# Folds (plan §C)
# ------------------------------------------------------------------
def assign_folds(idents_even_dplus: list[tuple]) -> dict:
    """Per-task fold assignment over even-init D+ identities (plan §C exact rule).

    Returns {ident: "train"|"val"|"test"}. Per task sorted by init: n>=3 -> last=test,
    second-last=val, rest=train; n<3 -> all train. Represented tasks are those with n>=3.
    """
    by_task: dict = {}
    for tid, init in idents_even_dplus:
        by_task.setdefault(tid, []).append(init)
    out: dict = {}
    for tid, inits in by_task.items():
        s = sorted(set(inits))
        if len(s) >= 3:
            out[(tid, s[-1])] = "test"
            out[(tid, s[-2])] = "val"
            for k in s[:-2]:
                out[(tid, k)] = "train"
        else:
            for k in s:
                out[(tid, k)] = "train"
    return out


def represented_tasks(fold_of: dict) -> set:
    """Tasks that contributed a test identity (n>=3)."""
    return {tid for (tid, _), f in fold_of.items() if f == "test"}
