"""Shared helpers of family f2_state (R1): M2 state_window and M3 row_filter_hmm.

Everything here respects the valid-dimension rules of harness/dims.py:
  * robot_state: only rs[..., :8] (dims.valid_state); z-scaled per dim by the std of the SEARCHED library;
  * actions: only [:5, :7] enters distances / dispersions / continuity (dims.valid_action), sigma-normalized by
    ctx.action_sigma (the err metric's sigma_d).
All fitted quantities come from the library only (T1, seconds).
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.harness import dims

BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}   # the "10x" library per model (FINDINGS: coverage)
RS_N = 8                                           # valid robot_state dims (both models)
SEG = dims.EXEC_STEPS * dims.ACT_DIMS              # 35 = executed valid block


def open_search_library(lib, ctx, which: str):
    """-> (LibraryView, stored library name usable as Result.library)."""
    if which == "current":
        return lib, "current"
    if which == "big":
        name = BIG[ctx.model]
        return ctx.open_library(name), name
    raise ValueError(f"lib must be 'big' or 'current', got {which!r}")


def rs_scale(L, model: str) -> np.ndarray:
    """Per-dim std (ddof 0) of the library's valid robot_state dims (+1e-6), float32 [8]."""
    rs = np.asarray(dims.valid_state(L.rs, model), np.float64)
    return (rs.std(0) + 1e-6).astype(np.float32)


def lib_windows(Z: np.ndarray, prev: np.ndarray, m: int) -> np.ndarray:
    """Library state windows [L, 8m]: [z(rs_r), z(rs_prev(r)), z(rs_prev^2(r)), ...]; when the history runs out the
    oldest available row is repeated (a row with prev = -1 repeats itself)."""
    L = Z.shape[0]
    out = np.empty((L, RS_N * m), np.float32)
    cur = np.arange(L)
    for j in range(m):
        out[:, RS_N * j:RS_N * (j + 1)] = Z[cur]
        p = prev[cur]
        cur = np.where(p >= 0, p, cur)
    return out


def query_window(q, inv_sd: np.ndarray, m: int) -> np.ndarray:
    """Query state window [8m] from online inputs only: current rs and the m-1 previous decisions' rs of this
    episode (q.hist_rs), z-scaled; the oldest available state is repeated (step 0 -> m copies of q.rs)."""
    x = np.empty(RS_N * m, np.float32)
    x[:RS_N] = np.asarray(q.rs[:RS_N], np.float32) * inv_sd
    if m > 1:
        h = q.hist_rs[-(m - 1):] if q.step > 0 else None
        n = 0 if h is None else h.shape[0]
        last = x[:RS_N]
        for j in range(1, m):
            if j <= n:
                last = np.asarray(h[n - j, :RS_N], np.float32) * inv_sd
            x[RS_N * j:RS_N * (j + 1)] = last
    return x


def seg_norm(action, sigma: np.ndarray) -> np.ndarray:
    """sigma-normalized executed valid block, flattened: [L, 35] float32 (action[:, :5, :7] / sigma_d)."""
    a = np.asarray(dims.valid_action(action), np.float32) / sigma.astype(np.float32)
    return np.ascontiguousarray(a.reshape(a.shape[0], SEG))


def tail_norm(action, sigma: np.ndarray) -> np.ndarray:
    """sigma-normalized unexecuted tail action[:, 5:10, :7] / sigma_d, flattened [L, 35] float32."""
    a = np.asarray(action[:, dims.EXEC_STEPS:2 * dims.EXEC_STEPS, dims.ACT_VALID], np.float32) / sigma.astype(np.float32)
    return np.ascontiguousarray(a.reshape(a.shape[0], SEG))


def l2_rows(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Euclidean distance of every row of X to x (difference form: exact, no cancellation), float32 [C]."""
    d = X - x
    return np.sqrt(np.einsum("ij,ij->i", d, d))


def topk_smallest(d: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest values, ascending by (value, index): deterministic tie-breaking."""
    C = d.shape[0]
    k = min(k, C)
    idx = np.argpartition(d, k - 1)[:k] if k < C else np.arange(C)
    return idx[np.lexsort((idx, d[idx]))]


def topk_largest(b: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values, descending by value, ties -> lower index."""
    C = b.shape[0]
    k = min(k, C)
    idx = np.argpartition(-b, k - 1)[:k] if k < C else np.arange(C)
    return idx[np.lexsort((idx, -b[idx]))]


def dispersion(Sk: np.ndarray) -> float:
    """Top-k action dispersion: RMS over (k, 5, 7) of the sigma-normalized deviation from the top-k mean.
    Sk: [k, 35] sigma-normalized executed valid blocks."""
    if Sk.shape[0] < 2:
        return 0.0
    dev = Sk - Sk.mean(0, keepdims=True)
    return float(np.sqrt(np.mean(dev * dev)))


def synth_mean(A: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """(Weighted) mean of full library chunks A [k, H, 32] -> (H, 32) float32."""
    if w is None:
        return A.mean(0).astype(np.float32)
    w = np.asarray(w, np.float64)
    w = w / w.sum()
    return np.tensordot(w, A.astype(np.float64), axes=1).astype(np.float32)


def synth_med(A: np.ndarray) -> np.ndarray:
    """B's median synthesis: per-step median over the k chunks on dims 0..5, gripper (dim 6) = majority sign
    (+1 / -1, ties -> +1), everything else (dims 7..31, never scored) from the top-1 chunk."""
    out = np.array(A[0], np.float32, copy=True)
    out[:, :dims.GRIPPER_DIM] = np.median(A[:, :, :dims.GRIPPER_DIM], axis=0)
    g = np.sign(A[:, :, dims.GRIPPER_DIM]).sum(0)
    out[:, dims.GRIPPER_DIM] = np.where(g >= 0, 1.0, -1.0)
    return out


# ------------------------------------------------------------------------------------ fit-side scales
def _other_episode_knn(Q, X, q_ep, x_ep, kmax: int, chunk: int = 1024):
    """Within one task: for every row of Q the kmax nearest rows of X from ANOTHER episode (squared Euclidean via
    GEMM, float32). Returns (d2 sorted [n, kmax] float32, idx [n, kmax] int64)."""
    n, C = Q.shape[0], X.shape[0]
    kmax = min(kmax, C)
    x2 = np.einsum("ij,ij->i", X, X)
    d_out = np.empty((n, kmax), np.float32)
    i_out = np.empty((n, kmax), np.int64)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        q = Q[lo:hi]
        D = np.einsum("ij,ij->i", q, q)[:, None] + x2[None, :] - 2.0 * (q @ X.T)
        np.maximum(D, 0.0, out=D)
        D[q_ep[lo:hi, None] == x_ep[None, :]] = np.inf
        part = np.argpartition(D, kmax - 1, axis=1)[:, :kmax]
        dp = np.take_along_axis(D, part, 1)
        o = np.argsort(dp, axis=1, kind="stable")
        d_out[lo:hi] = np.take_along_axis(dp, o, 1)
        i_out[lo:hi] = np.take_along_axis(part, o, 1)
    return d_out, i_out


def window_scales(blocks: dict, k: int | None = None):
    """tau_w = median over library rows of the window distance to the nearest row of another episode (same task);
    tau_a (if k) = median over library rows of the top-k (other-episode, window distance) action dispersion."""
    d1s, disps = [], []
    for b in blocks.values():
        kk = max(1, k or 1)
        d2, idx = _other_episode_knn(b.W, b.W, b.ep, b.ep, kk)
        d1s.append(np.sqrt(d2[:, 0]))
        if k:
            Sk = b.S[idx]                                     # [C, k, 35]
            dev = Sk - Sk.mean(1, keepdims=True)
            disps.append(np.sqrt(np.mean(dev * dev, axis=(1, 2))))
    d1 = np.concatenate(d1s)
    d1 = d1[np.isfinite(d1)]
    tau_w = float(np.median(d1))
    tau_a = float(np.median(np.concatenate(disps))) if k else float("nan")
    return max(tau_w, 1e-6), (max(tau_a, 1e-6) if k else tau_a)


def continuity_scale(blocks: dict) -> float:
    """tau_c (B-P1's s_c): median over library rows of RMS_sigma(tail(r)[5:10, :7] - nearest other-episode head
    [0:5, :7]) within the task (the library's own tail -> head continuity)."""
    cs = []
    for b in blocks.values():
        d2, _ = _other_episode_knn(b.T, b.S, b.ep, b.ep, 1)
        cs.append(np.sqrt(d2[:, 0] / SEG))
    c = np.concatenate(cs)
    c = c[np.isfinite(c)]
    return max(float(np.median(c)), 1e-6)


def regime_code(q) -> int:
    """-1: step 0 / undecidable; 0: previous decision was a MISS (fresh tail, policy chunk); 1: HIT (stale tail)."""
    ph = q.prev_hit
    return -1 if ph is None else (1 if ph else 0)
