"""R2 family G1 -- AWM: `awm_kernel` (ideation A-P1), `awm_vision_only` (A-P2) and the closed-loop insurance switches
(A-P3), one Method class with kwargs (the name encodes every non-default kwarg).

Representation (per library row / query):
  x = [PCA-64 of key_v0, PCA-64 of key_v1, rs[:8]]   features="joint" (136-d)
  x = [PCA-64 of key_v0, PCA-64 of key_v1]           features="vision" (128-d, no robot state anywhere)
  PCA basis follows the FIT data: fit on big -> the r01 big-library bases (derived/r01/f4_vision/pca/<m>_<s>/<big>/
  {v0,v1}: centred randomized SVD k=128 of the 500-episode library, leading 64 columns); fit on current -> the same
  randomized SVD recipe on the current library's pooled keys (<= 2,645 rows): computed in fit (5-60 s depending on
  memory-bandwidth contention) or loaded from the optional cache <derived>/r02/g1_awm/pca_current (pca_current.py,
  same function, single-threaded BLAS; used only when its meta matches the library's row count + ids.json sha1).
Extras (scalars, per decision): d1, d1_rel (d1 / median d), disp5, dst, regime (0 step0 / 1 fresh / 2 stale), lib_ep /
  lib_step (top-1 row's library episode / step), w_eff (effective kernel size 1 / sum wn^2), c0 (fresh only),
  still (step >= 1: centred-key cosine to the previous decision, both cameras summed; diagnostic, unused),
  nscale (norm_cap: mean rescale factor over the executed steps), gvote / gflip (hyst: t=0 vote, served-sign flip).
Metric (T1, closed form, per task, on the FIT library's rows of the task):
  z-score x (mean / std over the fit rows); heads h = action[:5,:7] / sigma_d (35-d); for every fit row its nn = 3
  nearest rows by head L2 among OTHER episodes of the same task; S_w = mean (xn_i - xn_j)(xn_i - xn_j)^T over these
  action-similar pairs; M = (S_w + lam * tr(S_w)/d * I)^-1; W = chol(M) (codes=0) or the top-r generalized-eigen
  directions of the total covariance in the whitened space (codes=r, "LDA-like"). code(x) = ((x - mean)/std) @ W,
  distance = Euclidean between codes. state_scale s multiplies the z-scored state block by s AFTER the fit (ideation
  A's st_x3; applied to the early fit too).
  Early fit (early=True): the same fit on the fit rows with step <= 2 (own mean / std / W0); used at step 0 only;
  candidates are still all rows. step0_joint=True (vision only): the step-0 branch uses the joint 136-d early fit
  (the only use of state in the vision arm).
Candidates: every row of the task in the CANDIDATE library: lib="big" (pi0.5 bpool_cs / GR00T bpool_all, 500 ep) or
  lib="current" (deployed ~50 ep). fit_data="same" = the candidate library; fit_data="big" with lib="current" =
  BORROWED big-library information (basis + metric fitted on the 500 episodes, candidates = the 50-episode library).
Regimes (online-legal inputs q.step, q.prev_hit):
  step 0            : f = -d0 (early metric; the main metric when early=False)
  fresh (prev MISS) : f = -d / median(d) - lam_c * c / s_c,  c = RMS_sigma(head_r - prev_a_exec[5:10, :7])
  stale (prev HIT, or prev_hit None at step >= 1): f = -d            (continuity is never used after a HIT)
Synthesis: top-k (k=16) by f (ties by row); d~ = -f; w_i = exp(-((d~_i - d~_1) / (d~_kref - d~_1))^2); the returned
  action is sum_i w_i chunk_i / sum_i w_i over the full (H, 32) library chunks (library actions only).
Confidence (scales from library pseudo-queries: every candidate-library row as a query against the OTHER episodes of
  its task, main metric, same kernel):
  stale : z(-d1) + z(-disp5) + z(-dst)          (vision: z(-d1) + z(-disp5))
  step 0: z(-disp5) * std(pseudo-query z-sum)   (the stale z-sum's spread, so both regimes share one scale)
  fresh : -c0 / s_c - disp5 / s_a
  d1 = nearest code distance; disp5 = RMS_sigma of the top-5 heads around the synthesized head; dst = state 1-NN L2
  / s_d; s_d (median library state 1-NN across episodes) and s_c (median 1-NN library tail -> other-episode head, as
  M1) per task; s_a = median pseudo-query disp5.
Insurance (A-P3; offline-neutral by design, meant for the pure-cache closed loop):
  norm_cap > 0 : per step t, a[t,:6] *= min(norm_cap, (sum_i w_i |chunk_i[t,:6]|_s / sum w) / |a[t,:6]|_s), |.|_s =
                 norm in sigma_d units; the ratio is >= 1 (triangle inequality): it only undoes the mean's shrinkage.
  hyst > 0     : gripper a[t,6] = +-1 by the sign of v_t = sum_i w_i sign(chunk_i[t,6]) / sum w, but the previously
                 served sign is kept unless |v_t| > hyst (sequential over the chunk's steps; the reference for t = 0
                 is the sign this method served at the last executed step (4) of its previous decision). No
                 hysteresis at episode step 0. Per-episode state = that last served sign (cleared in reset()).
Step-0 storage: with full-rank codes an entry's early code is an exact affine map of its main code (and of rs for
  step0_joint): Z0_r = Z_r A + rs_r A_s + b with A = W^-1 diag(std/std0) W0, so only |Z0_r - b|^2 (one float) is kept
  per entry and the step-0 distance is one d x d matvec + the usual GEMV. Rank-r codes keep Z0 explicitly (r floats).
bytes_per_entry (f32): code (136 | 128 | r) + rs[:8] (joint: dst confidence; vision + step0_joint: early code) +
  step-0 extra (1 float | r floats). Fixed per suite, not per entry: PCA mean + 64-column basis for both cameras
  2 x (32768 + 32768 x 64) x 4 B = 17.0 MB, per-task mean / std / W (/ W0 / A) d x d f32 (~0.15 MB per task).
Fitted object: plain in-memory ndarrays (picklable for --os-fit-artifact; read-only, shared across connections /
forked workers); query() writes only the per-episode gripper sign.
G3 base contract (g3_recovery/README.md; V6 StuckRecovery / V7 wrap AWM): os_library / os_fit_library (candidate /
  fit library), synth_k = k, synth_T = 1, os_score_all(q) -> (all task rows, S, aux) with S = -((d~ - d~_1) /
  (d~_kref - d~_1))^2 (strictly monotone in -d~, so the stable top-k is AWM's own, and the wrapper's weights
  exp(-(S_0 - S_i)/1) are AWM's kernel bit for bit), aux = {vis_v0, vis_v1: task-centred PCA-64 cosine per camera,
  awm_d, awm_c}; os_synth(q, rows, w) = AWM's kernel mean (+ V3 insurance, same per-episode gripper state);
  os_confidence(...) = AWM's confidence for the served set. Passthrough(AWM) == AWM.query() bit for bit
  (tools/contract_check.py PASS). The vis arrays cost 2 x 64 f32 per entry (512 B) only when a wrapper uses them.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from exp.offline_search.harness import api, dims

BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
PCA_BIG = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/pca")
PCA_CUR = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r02/g1_awm/pca_current")  # optional cache
PCA_RECIPE = "rsvd_k128_o32_p3_seed0_keep64"
PDIM = 64                                   # PCA dims per camera
RSV = 8                                     # valid robot-state dims (dims.RS_VALID)
NH = dims.EXEC_STEPS * dims.ACT_DIMS        # 35: executed head in sigma units
EARLY_MAX_STEP = 2
PQ_CHUNK = 1024


def _fmt(x) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


# ------------------------------------------------------------------------------------------------ fitting helpers
def pca_fit(X, keep=PDIM, K=128, over=32, power=3, seed=0, chunk=2048):
    """Centred randomized SVD of the rows of X (memmap [n, D]) with the r01 m7_pca_fit recipe (k=128, oversampling
    32, 3 power iterations, seed 0); returns mean f32 [D], basis f32 [D, keep] (leading PCs), proj f32 [n, keep]."""
    n, D = X.shape
    cuts = [(lo, min(n, lo + chunk)) for lo in range(0, n, chunk)]
    mu = np.zeros(D, np.float64)
    for lo, hi in cuts:
        mu += np.asarray(X[lo:hi], np.float64).sum(0)
    mu = (mu / n).astype(np.float32)

    def xmul(M):
        out = np.empty((n, M.shape[1]), np.float32)
        muM = mu @ M
        for lo, hi in cuts:
            out[lo:hi] = np.asarray(X[lo:hi], np.float32) @ M - muM
        return out

    def xtmul(Y):
        out = np.zeros((D, Y.shape[1]), np.float32)
        for lo, hi in cuts:
            out += np.asarray(X[lo:hi], np.float32).T @ Y[lo:hi]
        out -= np.outer(mu, Y.sum(0))
        return out

    r = min(K + over, n)
    Om = np.random.default_rng(seed).standard_normal((D, r)).astype(np.float32)
    Q, _ = np.linalg.qr(xmul(Om))
    for _ in range(power):
        Z, _ = np.linalg.qr(xtmul(Q))
        Q, _ = np.linalg.qr(xmul(Z))
    _, _, Vt = np.linalg.svd(xtmul(Q).T, full_matrices=False)
    V = np.ascontiguousarray(Vt[:keep].T.astype(np.float32))
    return mu, V, xmul(V)


def lib_fingerprint(L) -> dict:
    """Identity of a stored library for the PCA cache: row count + sha1 of ids.json + the PCA recipe."""
    import hashlib

    return {"n": int(L.L), "ids_sha1": hashlib.sha1((L.dir / "ids.json").read_bytes()).hexdigest(), "recipe": PCA_RECIPE}


def pca_current(L, lib_key, field):
    """PCA-64 of the current library's pooled keys: loaded (fully, in memory) from the optional cache written by
    pca_current.py when its meta matches this library, else computed here with pca_fit (same result)."""
    d = PCA_CUR / lib_key / field
    try:
        meta = json.loads((d / "meta.json").read_text())
        if all(meta.get(k) == v for k, v in lib_fingerprint(L).items()):
            return tuple(np.array(np.load(d / f"{n}.npy"), dtype=np.float32, copy=True) for n in ("mean", "basis", "proj"))
    except (OSError, ValueError):
        pass
    return pca_fit(getattr(L, f"key_{field}"))


def project(X, mu, B, chunk=2048):
    """(X - mu) @ B for a key memmap X [n, D] (f32, chunked)."""
    out = np.empty((X.shape[0], B.shape[1]), np.float32)
    muB = mu @ B
    for lo in range(0, X.shape[0], chunk):
        hi = min(X.shape[0], lo + chunk)
        out[lo:hi] = np.asarray(X[lo:hi], np.float32) @ B - muB
    return out


def fit_metric(X, H, ep, nn=3, lam=0.1, rank=0):
    """Ideation A's fit_transform (p3_white.py), float64. X [n, d] features, H [n, 35] sigma-scaled heads, ep [n]
    episode ids. Returns (mean, std, W) with code = ((x - mean) / std) @ W."""
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    Xn = (X - mean) / std
    h2 = (H * H).sum(1)
    DH = h2[:, None] - 2 * H @ H.T + h2[None, :]
    DH[ep[:, None] == ep[None, :]] = np.inf
    k = min(nn, DH.shape[1] - 1)
    nn_idx = np.argpartition(DH, k, axis=1)[:, :k]
    okm = np.isfinite(np.take_along_axis(DH, nn_idx, 1))
    diffs = (Xn[:, None, :] - Xn[nn_idx])[okm]
    Sw = diffs.T @ diffs / max(len(diffs), 1)
    d = Sw.shape[0]
    Minv = np.linalg.inv(Sw + lam * np.trace(Sw) / d * np.eye(d))
    A = np.linalg.cholesky(Minv)
    if not rank:
        return mean, std, A
    St = np.cov(Xn.T) + 1e-6 * np.eye(d)
    Ms = A.T @ St @ A
    ev, V = np.linalg.eigh((Ms + Ms.T) / 2)
    return mean, std, A @ V[:, np.argsort(-ev)[:rank]]


def _masked_min(Q, R, ep_q, ep_r, rms_dim=None):
    """Per query row: min over R rows of another episode of the L2 (or RMS over rms_dim dims) distance (chunked)."""
    r2 = (R * R).sum(1)
    out = np.empty(Q.shape[0], np.float64)
    for lo in range(0, Q.shape[0], PQ_CHUNK):
        hi = min(Q.shape[0], lo + PQ_CHUNK)
        q = Q[lo:hi]
        D = (q * q).sum(1)[:, None] - 2 * q @ R.T + r2[None]
        np.maximum(D, 0, out=D)
        D[ep_q[lo:hi, None] == ep_r[None, :]] = np.inf
        out[lo:hi] = D.min(1)
    out = out / rms_dim if rms_dim else out
    return np.sqrt(out)


def _kernel_w(rel, kref):
    """rel [.., k] = d~ - d~_1 (sorted ascending); w = exp(-(rel / rel[kref-1])^2)."""
    k = rel.shape[-1]
    ref = np.maximum(rel[..., min(kref, k) - 1:min(kref, k)], 1e-6)
    return np.exp(-(rel / ref) ** 2)


class _Task:
    """Per-task fitted arrays (plain ndarrays; read-only after fit)."""


# --------------------------------------------------------------------------------------------------------- method
class AWM(api.Method):
    """kwargs: features joint|vision, lib big|current, fit_data same|big, kref (8), k (16), codes (0 = full, r = rank-r),
    lam (0.1), state_scale (1), early (True), step0_joint (False, vision only), lam_c (0.5), norm_cap (0 = off, 1.15),
    hyst (0 = off, 0.4), insure (True = norm_cap 1.15 + hyst 0.4), nn (3)."""

    tier = "T1"
    family = "g1_awm"

    def __init__(self, features="joint", lib="big", fit_data="same", kref=8, k=16, codes=0, lam=0.1, state_scale=1.0,
                 early=True, step0_joint=False, lam_c=0.5, norm_cap=0.0, hyst=0.0, insure=False, nn=3):
        if features not in ("joint", "vision"):
            raise ValueError(f"features must be joint | vision, got {features!r}")
        if lib not in ("big", "current"):
            raise ValueError(f"lib must be big | current, got {lib!r}")
        if fit_data not in ("same", "big"):
            raise ValueError(f"fit_data must be same | big, got {fit_data!r}")
        if state_scale != 1.0 and features != "joint":
            raise ValueError("state_scale needs features=joint")
        if step0_joint and (features != "vision" or not early):
            raise ValueError("step0_joint is the vision arm's joint early-fit step-0 branch (features=vision, early)")
        if insure:
            norm_cap = norm_cap or 1.15
            hyst = hyst or 0.4
        if norm_cap and norm_cap < 1.0:
            raise ValueError("norm_cap must be 0 (off) or >= 1")
        self.features, self.lib, self.fit_data = features, lib, fit_data
        self.kref, self.k, self.codes, self.lam = int(kref), int(k), int(codes), float(lam)
        self.state_scale, self.early, self.step0_joint = float(state_scale), bool(early), bool(step0_joint)
        self.lam_c, self.norm_cap, self.hyst, self.nn = float(lam_c), float(norm_cap), float(hyst), int(nn)
        self.fit_src = "big" if (lib == "big" or fit_data == "big") else "current"
        self.feat0 = "joint" if (features == "joint" or step0_joint) else "vision"
        parts = ["AWM", "joint" if features == "joint" else "vis", "big" if lib == "big" else "cur",
                 "fbig" if self.fit_src == "big" else "fcur"]
        if self.kref != 8:
            parts.append(f"kr{self.kref}")
        if self.k != 16:
            parts.append(f"k{self.k}")
        if self.codes:
            parts.append(f"r{self.codes}")
        if self.lam != 0.1:
            parts.append(f"lam{_fmt(self.lam)}")
        if self.state_scale != 1.0:
            parts.append(f"st{_fmt(self.state_scale)}")
        if not self.early:
            parts.append("noearly")
        if self.step0_joint:
            parts.append("s0joint")
        if self.lam_c != 0.5:
            parts.append(f"lc{_fmt(self.lam_c)}")
        if self.nn != 3:
            parts.append(f"nn{self.nn}")
        if self.norm_cap == 1.15 and self.hyst == 0.4:
            parts.append("ins")
        else:
            if self.norm_cap:
                parts.append(f"nc{_fmt(self.norm_cap)}")
            if self.hyst:
                parts.append(f"hy{_fmt(self.hyst)}")
        self.name = "_".join(parts)
        self._last_g = None

    # ------------------------------------------------------------------------------------------------------ fit
    def _feats(self, P0, P1, rs, which):
        parts = [P0, P1] + ([rs] if which == "joint" else [])
        return np.concatenate(parts, 1).astype(np.float64)

    def fit(self, lib, ctx):
        self.model = ctx.model
        big = BIG[ctx.model]
        self.cand_name = big if self.lib == "big" else "current"
        fit_name = big if self.fit_src == "big" else "current"
        Lc = lib if self.cand_name == "current" else ctx.open_library(self.cand_name)
        Lf = Lc if fit_name == self.cand_name else ctx.open_library(fit_name)
        sig = np.asarray(ctx.action_sigma, np.float64)
        self.sig = sig.astype(np.float32)
        self.H = int(Lc.H)
        # ---- PCA basis (follows the fit data) + projections of fit / candidate rows
        with self.prof.section("pca"):
            Pf, Pc, mus, Bs = {}, {}, {}, {}
            for f in ("v0", "v1"):
                if self.fit_src == "big":
                    d = PCA_BIG / ctx.lib_key / big / f
                    meta = json.loads((d / "meta.json").read_text())
                    if meta["n"] != Lf.L:
                        raise api.ContractError(f"PCA cache {d} was fitted on {meta['n']} rows, {big} has {Lf.L}")
                    mus[f] = np.array(np.load(d / "mean.npy"), dtype=np.float32, copy=True)
                    Bs[f] = np.ascontiguousarray(np.load(d / "basis.npy", mmap_mode="r")[:, :PDIM], np.float32)
                    Pf[f] = np.ascontiguousarray(np.load(d / "proj.npy", mmap_mode="r")[:, :PDIM], np.float32)
                    Pc[f] = Pf[f] if Lc is Lf else project(getattr(Lc, f"key_{f}"), mus[f], Bs[f])
                else:
                    mus[f], Bs[f], Pf[f] = pca_current(Lf, ctx.lib_key, f)
                    Pc[f] = Pf[f]
            self.mu0, self.mu1 = mus["v0"], mus["v1"]
            self.B0, self.B1 = Bs["v0"], Bs["v1"]
            self.muB0, self.muB1 = (self.mu0 @ self.B0).astype(np.float32), (self.mu1 @ self.B1).astype(np.float32)
            # query-side projection in the row-major (64, 32768) layout: one contiguous GEMV per camera (~20 % faster)
            self.B0T, self.B1T = np.ascontiguousarray(self.B0.T), np.ascontiguousarray(self.B1.T)
            del self.B0, self.B1
        # ---- per-row library arrays
        def rows_arrays(L, P):
            act7 = np.asarray(L.action[:, :, dims.ACT_VALID], np.float64)
            heads = (act7[:, :dims.EXEC_STEPS] / sig).reshape(len(act7), NH)
            tails = (act7[:, 5:10] / sig).reshape(len(act7), NH)
            rs = np.asarray(dims.valid_state(np.asarray(L.rs, np.float32), ctx.model), np.float64)
            return dict(P0=np.asarray(P["v0"], np.float64), P1=np.asarray(P["v1"], np.float64), rs=rs, heads=heads,
                        tails=tails, ep=np.asarray(L.episode, np.int64), st=np.asarray(L.step, np.int64))
        C = rows_arrays(Lc, Pc)
        F = C if Lf is Lc else rows_arrays(Lf, Pf)
        self.act = np.array(Lc.action, dtype=np.float32, order="C", copy=True)   # (L, H, 32) candidate chunks (in memory)
        self.lib_ep = np.array(Lc.episode, dtype=np.int32, copy=True)
        self.lib_step = np.array(Lc.step, dtype=np.int32, copy=True)
        feat = self.features
        dfe = 2 * PDIM + (RSV if feat == "joint" else 0)
        pq = {"d1": [], "disp": [], "dst": []}
        self.tasks = {}
        with self.prof.section("metric"):
            for t in Lc.tasks():
                rc = np.asarray(Lc.rows_of_task(t), np.int64)
                rf = np.asarray(Lf.rows_of_task(t), np.int64)
                if len(rf) < 2:
                    raise api.ContractError(f"task {t}: {len(rf)} fit rows in {fit_name}")
                T = _Task()
                T.rows = rc
                Xf = self._feats(F["P0"][rf], F["P1"][rf], F["rs"][rf], feat)
                mean, std, W = fit_metric(Xf, F["heads"][rf], F["ep"][rf], nn=self.nn, lam=self.lam, rank=self.codes)
                if self.state_scale != 1.0:
                    std = std.copy()
                    std[-RSV:] /= self.state_scale
                Xc = self._feats(C["P0"][rc], C["P1"][rc], C["rs"][rc], feat)
                Z64 = ((Xc - mean) / std) @ W
                T.Z = np.ascontiguousarray(Z64, np.float32)
                Zr = T.Z.astype(np.float64)
                T.z2 = (Zr * Zr).sum(1).astype(np.float32)
                T.Wf = np.ascontiguousarray(W / std[:, None], np.float32)
                T.shift = ((mean / std) @ W).astype(np.float32)
                T.HD = np.ascontiguousarray(C["heads"][rc], np.float32)
                T.h2 = (C["heads"][rc] ** 2).sum(1).astype(np.float32)
                for fi, key in ((0, "P0"), (1, "P1")):                      # G3 aux: task-centred unit PCA-64
                    Pm = C[key][rc].mean(0)
                    V = C[key][rc] - Pm
                    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
                    setattr(T, f"Vm{fi}", Pm.astype(np.float32))
                    setattr(T, f"V{fi}", np.ascontiguousarray(V, np.float32))
                T.RS = np.ascontiguousarray(C["rs"][rc], np.float32)
                T.rs2 = (C["rs"][rc] ** 2).sum(1).astype(np.float32)
                ep_c = C["ep"][rc]
                # ---- early fit (step-0 branch)
                T.Z0 = T.A0 = T.As0 = T.n20 = T.W0f = T.c0 = None
                if self.early:
                    f0 = self.feat0
                    Xf0 = self._feats(F["P0"][rf], F["P1"][rf], F["rs"][rf], f0)
                    m0 = F["st"][rf] <= EARLY_MAX_STEP
                    if m0.sum() < 2 or len(np.unique(F["ep"][rf][m0])) < 2:
                        raise api.ContractError(f"task {t}: too few early rows ({int(m0.sum())}) in {fit_name}")
                    mn0, sd0, W0 = fit_metric(Xf0[m0], F["heads"][rf][m0], F["ep"][rf][m0], nn=self.nn, lam=self.lam,
                                              rank=self.codes)
                    if f0 == "joint" and self.state_scale != 1.0:
                        sd0 = sd0.copy()
                        sd0[-RSV:] /= self.state_scale
                    Xc0 = self._feats(C["P0"][rc], C["P1"][rc], C["rs"][rc], f0)
                    Z0 = ((Xc0 - mn0) / sd0) @ W0                               # explicit early codes (f64)
                    T.W0f = np.ascontiguousarray(W0 / sd0[:, None], np.float32)
                    shift0 = (mn0 / sd0) @ W0
                    if self.codes:                                              # rank-r: keep Z0 explicitly
                        T.Z0 = np.ascontiguousarray(Z0, np.float32)
                        b = np.zeros(W0.shape[1])
                        Y = T.Z0.astype(np.float64)
                    else:                                                       # affine map of the main code
                        Winv = np.linalg.inv(W)
                        dv = dfe
                        A = Winv @ (np.diag(std / sd0[:dv]) @ W0[:dv])
                        b = ((mean - mn0[:dv]) / sd0[:dv]) @ W0[:dv]
                        Y = Zr @ A
                        if f0 != feat:                                          # vision main code -> joint early
                            As = W0[dv:] / sd0[dv:, None]
                            b = b - (mn0[dv:] / sd0[dv:]) @ W0[dv:]
                            T.As0 = np.ascontiguousarray(As, np.float32)
                            Y = Y + T.RS.astype(np.float64) @ As
                        T.A0 = np.ascontiguousarray(A, np.float32)
                        dev = np.abs(Y + b - Z0).max() / max(np.abs(Z0).max(), 1e-9)
                        if dev > 1e-3:
                            raise api.ContractError(f"task {t}: early-code affine map deviates by {dev:.2e} (rel)")
                    T.n20 = (Y * Y).sum(1).astype(np.float32)
                    T.c0 = (shift0 + b).astype(np.float32)
                # ---- per-task scales on the candidate library (other episodes of the task)
                T.s_d = float(np.median(_masked_min(C["rs"][rc], C["rs"][rc], ep_c, ep_c))) + 1e-6
                T.s_c = float(np.median(_masked_min(C["tails"][rc], C["heads"][rc], ep_c, ep_c, rms_dim=NH))) + 1e-6
                # ---- library pseudo-queries (main metric, same kernel): d1, disp5, dst
                n = len(rc)
                same_max = int(np.bincount(np.unique(ep_c, return_inverse=True)[1]).max())
                kk = max(1, min(self.k, n - same_max))
                for lo in range(0, n, PQ_CHUNK):
                    hi = min(n, lo + PQ_CHUNK)
                    q = Zr[lo:hi]
                    D2 = (q * q).sum(1)[:, None] - 2 * q @ Zr.T + (Zr * Zr).sum(1)[None]
                    np.maximum(D2, 0, out=D2)
                    D = np.sqrt(D2)
                    D[ep_c[lo:hi, None] == ep_c[None, :]] = np.inf
                    idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
                    Dk = np.take_along_axis(D, idx, 1)
                    o = np.argsort(Dk, axis=1, kind="stable")
                    idx = np.take_along_axis(idx, o, 1)
                    Dk = np.take_along_axis(Dk, o, 1)
                    w = _kernel_w(Dk - Dk[:, :1], self.kref)
                    Hk = C["heads"][rc][idx]                                    # (m, kk, 35)
                    head = (Hk * w[:, :, None]).sum(1) / w.sum(1)[:, None]
                    disp = np.sqrt(np.mean((Hk[:, :5] - head[:, None]) ** 2, axis=(1, 2)))
                    pq["d1"].append(Dk[:, 0])
                    pq["disp"].append(disp)
                pq["dst"].append(_masked_min(C["rs"][rc], C["rs"][rc], ep_c, ep_c) / T.s_d)
                self.tasks[int(t)] = T
        with self.prof.section("scales"):
            v = {k2: np.concatenate(a) for k2, a in pq.items()}
            self.s_a = float(np.median(v["disp"])) + 1e-6
            self.zmu = {k2: float(np.mean(-a)) for k2, a in v.items()}
            self.zsd = {k2: float(np.std(-a)) + 1e-9 for k2, a in v.items()}
            zs = (-v["d1"] - self.zmu["d1"]) / self.zsd["d1"] + (-v["disp"] - self.zmu["disp"]) / self.zsd["disp"]
            if feat == "joint":
                zs = zs + (-v["dst"] - self.zmu["dst"]) / self.zsd["dst"]
            self.zs_sd = float(np.std(zs)) + 1e-9
        self.os_library, self.os_fit_library = self.cand_name, fit_name     # G3 base contract
        self.fixed_bytes = int(4 * (2 * (self.mu0.size + self.B0T.size)
                                    + sum(T.Wf.size + T.shift.size + (T.W0f.size if T.W0f is not None else 0)
                                          + (T.A0.size if T.A0 is not None else 0) for T in self.tasks.values())))
        self.n_cand = {t: int(len(T.rows)) for t, T in self.tasks.items()}
        # fitted arrays are read-only (shared across forked workers / plugin connections)
        for a in [self.sig, self.mu0, self.mu1, self.B0T, self.B1T, self.muB0, self.muB1, self.act, self.lib_ep,
                  self.lib_step] + [x for T in self.tasks.values() for x in vars(T).values() if isinstance(x, np.ndarray)]:
            a.flags.writeable = False

    # ---------------------------------------------------------------------------------------------------- query
    def reset(self, episode):
        self._last_g = None

    def query(self, q):
        T, step, regime, k0, k1, xv, rs8, d, med, c, dt = self._dist(q)
        with self.prof.section("synth"):
            n = dt.shape[0]
            k = min(self.k, n)
            idx = np.argpartition(dt, k - 1)[:k] if k < n else np.arange(n)
            idx = idx[np.lexsort((idx, dt[idx]))]
            dk = dt[idx].astype(np.float64)
            w = _kernel_w(dk - dk[0], self.kref)
            rows = T.rows[idx]
            wn, Ck, a = self._mix(rows, w)
            disp5 = self._disp5(T, a, idx[:5])
        d1 = float(d.min())
        dst = self._dst(T, rs8)
        c0 = float(c[idx[0]]) if c is not None else float("nan")
        conf = self._conf(T, regime, d1, disp5, dst, c0)
        # extras: scalars only, no NaN (keys that do not apply to the regime are omitted; the harness pads them with
        # NaN, and the closed-loop selftest compares present keys bit for bit)
        ex = {"d1": d1, "d1_rel": d1 / med, "disp5": disp5, "dst": dst, "regime": float(regime),
              "lib_ep": float(self.lib_ep[rows[0]]), "lib_step": float(self.lib_step[rows[0]]),
              "w_eff": float(1.0 / (wn.astype(np.float64) ** 2).sum())}
        if regime == 1:
            ex["c0"] = c0
        if step > 0:
            h0, h1 = q.hist_key_v0[-1], q.hist_key_v1[-1]
            s = 0.0
            for cur, prv, mu in ((k0, h0, self.mu0), (k1, h1, self.mu1)):
                u = cur - mu
                v = np.asarray(prv, np.float32) - mu
                s += float(u @ v) / max(float(np.sqrt((u @ u) * (v @ v))), 1e-12)
            ex["still"] = s
        self._insure(a, Ck, wn, step, ex)
        return api.Result(topk=rows.astype(np.int64), scores=-dt[idx].astype(np.float64), confidence=float(conf),
                          action=a, library=self.cand_name, extras=ex)

    # ------------------------------------------------------------------------------------ shared query pieces
    def _dist(self, q):
        """Per-candidate distances of one decision (query() and os_score_all() share this arithmetic)."""
        T = self.tasks[int(q.task_id)]
        step = int(q.step)
        with self.prof.section("project"):
            k0 = np.asarray(q.key_v0, np.float32)
            k1 = np.asarray(q.key_v1, np.float32)
            xv = np.concatenate([self.B0T @ k0 - self.muB0, self.B1T @ k1 - self.muB1])
            rs8 = np.asarray(dims.valid_state(q.rs, self.model), np.float32)
        ph = q.prev_hit if step > 0 else None
        regime = 0 if step == 0 else (1 if ph is False else 2)
        with self.prof.section("distance"):
            if regime == 0 and self.early:
                x0 = np.concatenate([xv, rs8]) if self.feat0 == "joint" else xv
                y = x0 @ T.W0f - T.c0
                if T.Z0 is not None:
                    cross = T.Z0 @ y
                else:
                    cross = T.Z @ (T.A0 @ y)
                    if T.As0 is not None:
                        cross = cross + T.RS @ (T.As0 @ y)
                d2 = T.n20 - 2.0 * cross + float(y @ y)
            else:
                x = np.concatenate([xv, rs8]) if self.features == "joint" else xv
                z = x @ T.Wf - T.shift
                d2 = T.z2 - 2.0 * (T.Z @ z) + float(z @ z)
            d = np.sqrt(np.maximum(d2, 0.0))
            med = float(np.median(d)) + 1e-12
            c = None
            if regime == 1:
                tail = (np.asarray(q.prev_a_exec[5:10, dims.ACT_VALID], np.float32) / self.sig).ravel()
                c = np.sqrt(np.maximum(T.h2 - 2.0 * (T.HD @ tail) + float(tail @ tail), 0.0) / NH)
                dt = d / med + self.lam_c * c / T.s_c
            else:
                dt = d
        return T, step, regime, k0, k1, xv, rs8, d, med, c, dt

    def _mix(self, rows, w):
        """Kernel mean of the full (H, 32) chunks of `rows` with unnormalized weights w (f64) -> (wn, chunks, a)."""
        wn = (w / w.sum()).astype(np.float32)
        Ck = self.act[rows]                                                      # (k, H, 32)
        a = np.tensordot(wn, Ck, 1)                                              # (H, 32) f32
        return wn, Ck, a

    def _disp5(self, T, a, pos5):
        head = (a[:dims.EXEC_STEPS, dims.ACT_VALID] / self.sig).ravel()
        return float(np.sqrt(np.mean((T.HD[pos5] - head) ** 2)))

    @staticmethod
    def _dst(T, rs8):
        ds2 = T.rs2 - 2.0 * (T.RS @ rs8) + float(rs8 @ rs8)
        return float(np.sqrt(max(float(ds2.min()), 0.0))) / T.s_d

    def _conf(self, T, regime, d1, disp5, dst, c0):
        zd = (-disp5 - self.zmu["disp"]) / self.zsd["disp"]
        if regime == 2:
            conf = (-d1 - self.zmu["d1"]) / self.zsd["d1"] + zd
            if self.features == "joint":
                conf += (-dst - self.zmu["dst"]) / self.zsd["dst"]
        elif regime == 0:
            conf = self.zs_sd * zd
        else:
            conf = -c0 / T.s_c - disp5 / self.s_a
        return conf

    def _insure(self, a, Ck, wn, step, ex):
        """Closed-loop synthesis switches (A-P3), in place on a; updates the per-episode gripper sign."""
        if self.norm_cap > 0:
            s6 = self.sig[:6]
            cn = np.linalg.norm(Ck[:, :, :6] / s6, axis=2)                      # (k, H)
            an = np.linalg.norm(a[:, :6] / s6, axis=1)                          # (H,)
            fac = np.where(an > 1e-8, np.minimum(self.norm_cap, (wn @ cn) / np.maximum(an, 1e-12)), 1.0)
            fac = np.maximum(fac, 1.0).astype(np.float32)
            a[:, :6] *= fac[:, None]
            ex["nscale"] = float(fac[:dims.EXEC_STEPS].mean())
        if self.hyst > 0:
            g = np.where(Ck[:, :, dims.GRIPPER_DIM] >= 0, np.float32(1.0), np.float32(-1.0))
            vote = wn @ g                                                        # (H,) in [-1, 1]
            served = np.where(vote >= 0, 1.0, -1.0).astype(np.float32)
            prev = self._last_g if step > 0 else None
            if prev is not None:
                for t in range(served.shape[0]):
                    if abs(float(vote[t])) <= self.hyst:
                        served[t] = prev
                    prev = float(served[t])
            ex["gvote"] = float(vote[0])
            ex["gflip"] = float(self._last_g is not None and step > 0 and float(served[0]) != self._last_g)
            a[:, dims.GRIPPER_DIM] = served
            self._last_g = float(served[dims.EXEC_STEPS - 1])

    # ------------------------------------------------------------------------ G3 base contract (V6 / V7 wrappers)
    # g3_recovery/README.md "base contract". The wrapper redoes top-k + kernel weights itself with
    # w_i = exp(-(S_0 - S_i) / synth_T); AWM's kernel exp(-((d~_i - d~_1) / (d~_kref - d~_1))^2) is reproduced exactly
    # by the score S_i = -((d~_i - d~_1) / (d~_kref - d~_1))^2 with synth_T = 1 (a strictly monotone transform of -d~,
    # so argsort(-S, stable)[:synth_k] is AWM's own top-k, ties by row). Under V6's recovery (excluded candidates) the
    # wrapper keeps these global-reference weights (not re-centred on the restricted set); the blend's forced member
    # gets the top weight 1 = AWM's w_1.
    synth_T = 1.0

    @property
    def synth_k(self):
        return self.k

    def os_score_all(self, q):
        """(rows, S, aux) over ALL task candidates; stateless. aux: vis_v0 / vis_v1 = task-centred PCA-64 cosine per
        camera (the base's own visual representation), awm_d (whitened distance, f32), awm_c (fresh continuity)."""
        T, step, regime, k0, k1, xv, rs8, d, med, c, dt = self._dist(q)
        dt64 = dt.astype(np.float64)
        n = dt64.shape[0]
        kr = min(self.kref, min(self.k, n))
        d1t = dt64.min()
        ref = max(float(np.partition(dt64, kr - 1)[kr - 1] - d1t), 1e-6)
        S = -((dt64 - d1t) / ref) ** 2
        aux = {"awm_d": d}
        if c is not None:
            aux["awm_c"] = c
        for f, p in ((0, xv[:PDIM]), (1, xv[PDIM:])):
            pc = p - getattr(T, f"Vm{f}")
            aux[f"vis_v{f}"] = getattr(T, f"V{f}") @ (pc / max(float(np.sqrt(pc @ pc)), 1e-12))
        return T.rows, S, aux

    def os_synth(self, q, rows, w):
        """AWM's synthesis of the given ordered rows / unnormalized weights (+ V3 insurance when on; updates the
        per-episode gripper sign exactly as query() does)."""
        wn, Ck, a = self._mix(np.asarray(rows, np.int64), np.asarray(w, np.float64))
        self._insure(a, Ck, wn, int(q.step), {})
        return a

    def os_confidence(self, q, rows, scores, aux, served_rows, w):
        """AWM's confidence for a served set: d1 over all candidates, disp5 of the served top-5 around their kernel
        mean (before insurance, as query()), dst, fresh continuity of the served top-1."""
        T = self.tasks[int(q.task_id)]
        step = int(q.step)
        ph = q.prev_hit if step > 0 else None
        regime = 0 if step == 0 else (1 if ph is False else 2)
        served_rows = np.asarray(served_rows, np.int64)
        pos = np.searchsorted(T.rows, served_rows)
        _, _, a = self._mix(served_rows, np.asarray(w, np.float64))
        disp5 = self._disp5(T, a, pos[:5])
        rs8 = np.asarray(dims.valid_state(q.rs, self.model), np.float32)
        c0 = float(aux["awm_c"][pos[0]]) if regime == 1 else float("nan")
        return self._conf(T, regime, float(np.asarray(aux["awm_d"]).min()), disp5, self._dst(T, rs8), c0)

    def bytes_per_entry(self):
        dcode = self.codes or (2 * PDIM + (RSV if self.features == "joint" else 0))
        extra_rs = RSV if (self.features == "joint" or self.step0_joint) else 0
        early = (self.codes or 1) if self.early else 0
        return float(4 * (dcode + extra_rs + early))
