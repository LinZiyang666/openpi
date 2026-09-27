"""R3 family H1 -- AWM3: AWM (r02/g1_awm/awm.py) plus independent trap-prevention switches (SELECTION.md row H1;
ideation A P1/P2/P3, ideation C proposals 1/2). Subclass of AWM: retrieval, kernel synthesis, confidence, PCA,
extras and the G3 base contract are AWM's; with every switch at its default AWM3(lib="current", kref=5) reproduces
AWM(lib="current", kref=5) bit for bit (tools/equiv_check.py), and AWM3(prior_alpha=1) reproduces
AWM(lib="current", fit_data="big", kref=5) bit for bit.

Switches (every non-default value is encoded in `name`):
  prior_alpha a in [0, 1]  -- BORROWED BIG-LIBRARY INFORMATION (A-P1 / C-P1). Candidates stay the deployed `current`
      library; the per-task metric is fitted in the big library's PCA-64 basis (r01 bases, bpool_cs / bpool_all) with
      the big rows' per-task mean / std, and S = (1 - a) * S_w(current rows) + a * S_w(big rows), both covariances
      of action-similar pairs (AWM's nn = 3 head neighbours among other episodes of the task) expressed in that SAME
      basis and SAME standardization, then AWM's trace-scaled ridge lam * tr(S) / d and Cholesky whitening. The early
      (step-0) metric is built the same way on the step <= 2 rows of both libraries. a = 1: the big fit alone (= AWM
      fit_data="big"); a = 0: the current rows' covariance in the borrowed basis / standardization (basis-only
      borrowing). Confidence scales are AWM's (candidate-library pseudo-queries, the blended metric).
  ridge_main r             -- non-borrowing twin (C-P2): the MAIN metric's ridge factor (AWM: lam = 0.1); the early
      metric keeps lam. Current-only fit.
  shared_beta b            -- offline-only ablation (C, measured worse on traces): S_task <- (1 - b) * S_task + b *
      mean of the OTHER tasks' S_w (each in its own per-task standardized coordinates, as C's metric_cv.py);
      main metric only.
  grip_commit              -- A-P2, the stateful part. v = sum_i wn_i * sign(g_i[step 0]) over the kernel members
      (wn normalized). Served sign = sign(v) if |v| >= grip_thr (0.8) else the previously EXECUTED sign; a change vs
      the executed sign requires |v| >= thr on this AND the previous decision (2-decision confirmation; grip_confirm
      "mag" = magnitudes only [default, the brief's wording], "sign" = the previous vote must also point to the new
      sign) and no executed sign change within the last grip_dwell (3) decisions. The previously executed sign and
      the change history are read from q.hist_a_exec (steps 0..4, dim 6 of every executed chunk; in mixed mode the
      chunk after a MISS is the policy's), never from this method's own output; only the previous decision's vote
      (a proposal, not an execution) is per-episode instance state (reset()). Step 0: the top-1 row's step-0 sign.
      The served sign (+-1: LIBERO applies sign(), the executed chunk then carries the sign exactly) replaces dim 6
      of executed steps 0..4; translation / rotation stay AWM's kernel mean over all members (grip_class_mean=True:
      mean over the served class only when it has >= 3 members -- a variant, C measured mode-only averaging worse).
  term_guard               -- A-P3. Candidate rows with lib_step >= ep_len - term_rows (2; variant 1) are dropped
      unless q.step >= term_late (0.9) * median library episode length of the task AND the executed gripper has
      already re-opened after having been closed in this episode (from q.hist_a_exec; open sign = the library's
      step-0 gripper sign, -1 pi0.5 / +1 GR00T). term_gate="late" (variant, name "tgp"): the step condition alone
      opens the gate (spatial library episodes end with the gripper still closed, so the re-open clause rarely
      holds there). If the mask would empty the task's candidates it is not applied.
      The mask is applied in os_score_all too (V6 / V7 / MixedJudge wrappers see the same candidate set).
Extras (<= 8 new keys): gvote (always), w_term (always: kernel weight on the last-2 rows of library episodes),
  gheld / gflip / gdwell (grip_commit), term_masked / term_open (term_guard), prior_alpha (prior variants).
G3 base contract: os_score_all (masked rows / S / aux), os_synth (AWM mean + insurance + grip_commit), os_confidence
  (masked aux); Passthrough(AWM3) == AWM3.query() bit for bit (r02/g3_recovery/tools/contract_check.py).
"""
from __future__ import annotations

import json

import numpy as np

from exp.offline_search.harness import api, dims
from exp.offline_search.rounds.r02.g1_awm.awm import (AWM, BIG, EARLY_MAX_STEP, NH, PCA_BIG, PDIM, PQ_CHUNK, RSV,
                                                      _fmt, _kernel_w, _masked_min, _Task, pca_current, project)

GD = dims.GRIPPER_DIM
ES = dims.EXEC_STEPS


# ------------------------------------------------------------------------------------------ metric pieces
# fit_metric (awm.py) split into its statistics and its whitening so the covariance can be blended / pooled in
# between; the arithmetic of every line is awm.fit_metric's (bit-identical W when nothing is blended).
def _standardize(X):
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    return mean, std


def _sw(Xn, H, ep, nn=3):
    """S_w of AWM's action-similar pairs: for every row its nn nearest heads among OTHER episodes; mean outer product
    of the standardized feature differences. Xn [n, d] standardized features, H [n, 35] sigma heads, ep [n]."""
    h2 = (H * H).sum(1)
    DH = h2[:, None] - 2 * H @ H.T + h2[None, :]
    DH[ep[:, None] == ep[None, :]] = np.inf
    k = min(nn, DH.shape[1] - 1)
    nn_idx = np.argpartition(DH, k, axis=1)[:, :k]
    okm = np.isfinite(np.take_along_axis(DH, nn_idx, 1))
    diffs = (Xn[:, None, :] - Xn[nn_idx])[okm]
    return diffs.T @ diffs / max(len(diffs), 1)


def _whiten(Sw, lam, rank, Xn):
    d = Sw.shape[0]
    Minv = np.linalg.inv(Sw + lam * np.trace(Sw) / d * np.eye(d))
    A = np.linalg.cholesky(Minv)
    if not rank:
        return A
    St = np.cov(Xn.T) + 1e-6 * np.eye(d)
    Ms = A.T @ St @ A
    ev, V = np.linalg.eigh((Ms + Ms.T) / 2)
    return A @ V[:, np.argsort(-ev)[:rank]]


def _stats(Xf, Hf, epf, Xc=None, Hc=None, epc=None, alpha=None, nn=3):
    """(mean, std, S, Xn_f): standardization from the FIT rows Xf; S = S_w(fit rows) when alpha is None, else the
    blend (1 - alpha) * S_w(current rows, same standardization) + alpha * S_w(fit rows)."""
    mean, std = _standardize(Xf)
    Xn = (Xf - mean) / std
    if alpha is None or alpha == 1.0:
        S = _sw(Xn, Hf, epf, nn)
    elif alpha == 0.0:
        S = _sw((Xc - mean) / std, Hc, epc, nn)
    else:
        S = (1.0 - alpha) * _sw((Xc - mean) / std, Hc, epc, nn) + alpha * _sw(Xn, Hf, epf, nn)
    return mean, std, S, Xn


# --------------------------------------------------------------------------------------------------------- method
class AWM3(AWM):
    """kwargs (on top of AWM's: features, lib, fit_data, kref (5 here), k, codes, lam, state_scale, early,
    step0_joint, lam_c, norm_cap, nn): prior_alpha (None | 0..1), ridge_main (None = lam), shared_beta (0),
    grip_commit (False), grip_thr (0.8), grip_dwell (3), grip_confirm (mag | sign), grip_class_mean (False),
    term_guard (False), term_rows (2), term_late (0.9), term_gate (both | late)."""

    family = "h1_trap"

    def __init__(self, lib="current", fit_data="same", kref=5, prior_alpha=None, ridge_main=None, shared_beta=0.0,
                 grip_commit=False, grip_thr=0.8, grip_dwell=3, grip_confirm="mag", grip_class_mean=False,
                 term_guard=False, term_rows=2, term_late=0.9, term_gate="both", **kw):
        if term_gate not in ("both", "late"):
            raise ValueError(f"term_gate must be both | late, got {term_gate!r}")
        if prior_alpha is not None:
            prior_alpha = float(prior_alpha)
            if lib != "current":
                raise ValueError("prior_alpha: candidates stay the deployed library (lib=current)")
            if fit_data != "same":
                raise ValueError("prior_alpha selects the fit source itself (big basis + standardization); leave "
                                 "fit_data=same")
            if not 0.0 <= prior_alpha <= 1.0:
                raise ValueError(f"prior_alpha must be in [0, 1], got {prior_alpha}")
            if shared_beta:
                raise ValueError("shared_beta is the non-borrowing twin's ablation; not combinable with prior_alpha")
            fit_data = "big"
        if grip_commit and (kw.get("hyst") or kw.get("insure")):
            raise ValueError("grip_commit replaces AWM's gripper hysteresis (hyst / insure); do not combine")
        if grip_confirm not in ("mag", "sign"):
            raise ValueError(f"grip_confirm must be mag | sign, got {grip_confirm!r}")
        if not 0.0 <= float(shared_beta) <= 1.0:
            raise ValueError(f"shared_beta must be in [0, 1], got {shared_beta}")
        if int(term_rows) < 1:
            raise ValueError("term_rows must be >= 1")
        super().__init__(lib=lib, fit_data=fit_data, kref=kref, **kw)
        self.prior_alpha = prior_alpha
        self.ridge_main = self.lam if ridge_main is None else float(ridge_main)
        self.shared_beta = float(shared_beta)
        self.grip_commit, self.grip_thr, self.grip_dwell = bool(grip_commit), float(grip_thr), int(grip_dwell)
        self.grip_confirm, self.grip_class_mean = grip_confirm, bool(grip_class_mean)
        self.term_guard, self.term_rows, self.term_late = bool(term_guard), int(term_rows), float(term_late)
        self.term_gate = term_gate
        parts = ["AWM3" + self.name[3:]]
        if self.prior_alpha is not None and self.prior_alpha != 1.0:
            parts.append(f"pa{_fmt(self.prior_alpha)}")
        if self.ridge_main != self.lam:
            parts.append(f"rm{_fmt(self.ridge_main)}")
        if self.shared_beta > 0:
            parts.append(f"sb{_fmt(self.shared_beta)}")
        if self.grip_commit:
            g = "gc"
            if self.grip_thr != 0.8:
                g += f"t{_fmt(self.grip_thr)}"
            if self.grip_dwell != 3:
                g += f"d{self.grip_dwell}"
            if self.grip_confirm == "sign":
                g += "s"
            if self.grip_class_mean:
                g += "m"
            parts.append(g)
        if self.term_guard:
            t = "tg" + ("p" if self.term_gate == "late" else "") + ("" if self.term_rows == 2 else str(self.term_rows))
            if self.term_late != 0.9:
                t += f"l{_fmt(self.term_late)}"
            parts.append(t)
        self.name = "_".join(parts)
        self._pv = None                      # previous decision's gripper vote (per episode; reset())

    # ------------------------------------------------------------------------------------------------------ fit
    def fit(self, lib, ctx):
        """AWM.fit with the metric fit factored out (blend / ridge / pooling) and the per-row terminal flags,
        per-task median episode length and the library's open-gripper sign added. Everything else is AWM's."""
        self.model = ctx.model
        big = BIG[ctx.model]
        self.cand_name = big if self.lib == "big" else "current"
        fit_name = big if self.fit_src == "big" else "current"
        Lc = lib if self.cand_name == "current" else ctx.open_library(self.cand_name)
        Lf = Lc if fit_name == self.cand_name else ctx.open_library(fit_name)
        sig = np.asarray(ctx.action_sigma, np.float64)
        self.sig = sig.astype(np.float32)
        self.H = int(Lc.H)
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
            self.B0T, self.B1T = np.ascontiguousarray(self.B0.T), np.ascontiguousarray(self.B1.T)
            del self.B0, self.B1

        def rows_arrays(L, P):
            act7 = np.asarray(L.action[:, :, dims.ACT_VALID], np.float64)
            heads = (act7[:, :dims.EXEC_STEPS] / sig).reshape(len(act7), NH)
            tails = (act7[:, 5:10] / sig).reshape(len(act7), NH)
            rs = np.asarray(dims.valid_state(np.asarray(L.rs, np.float32), ctx.model), np.float64)
            return dict(P0=np.asarray(P["v0"], np.float64), P1=np.asarray(P["v1"], np.float64), rs=rs, heads=heads,
                        tails=tails, ep=np.asarray(L.episode, np.int64), st=np.asarray(L.step, np.int64))
        C = rows_arrays(Lc, Pc)
        F = C if Lf is Lc else rows_arrays(Lf, Pf)
        self.act = np.array(Lc.action, dtype=np.float32, order="C", copy=True)
        self.lib_ep = np.array(Lc.episode, dtype=np.int32, copy=True)
        self.lib_step = np.array(Lc.step, dtype=np.int32, copy=True)
        # ---- H1 additions: terminal flags (candidate library), open-gripper sign of the library
        eplen = np.asarray(Lc.ep_len, np.int64)
        self.term_flag = (C["st"] >= eplen - self.term_rows)                       # bool [L]: masked by term_guard
        self.term2 = (C["st"] >= eplen - 2).astype(np.float32)                    # KPI: last-2 rows (fixed)
        g0 = np.asarray(Lc.action[:, 0, GD], np.float64)[C["st"] == 0]
        self.open_sign = 1.0 if float(g0.mean()) >= 0 else -1.0
        feat = self.features
        dfe = 2 * PDIM + (RSV if feat == "joint" else 0)
        alpha = self.prior_alpha
        # ---- pass 1: main-metric statistics per task (so shared_beta can pool across tasks before whitening)
        tasks = [int(t) for t in Lc.tasks()]
        rows_c, rows_f, stats = {}, {}, {}
        with self.prof.section("metric"):
            for t in tasks:
                rc = np.asarray(Lc.rows_of_task(t), np.int64)
                rf = np.asarray(Lf.rows_of_task(t), np.int64)
                if len(rf) < 2:
                    raise api.ContractError(f"task {t}: {len(rf)} fit rows in {fit_name}")
                rows_c[t], rows_f[t] = rc, rf
                Xf = self._feats(F["P0"][rf], F["P1"][rf], F["rs"][rf], feat)
                if alpha is None:
                    stats[t] = _stats(Xf, F["heads"][rf], F["ep"][rf], nn=self.nn)
                else:
                    Xc = self._feats(C["P0"][rc], C["P1"][rc], C["rs"][rc], feat)
                    stats[t] = _stats(Xf, F["heads"][rf], F["ep"][rf], Xc, C["heads"][rc], C["ep"][rc], alpha, self.nn)
            if self.shared_beta > 0 and len(tasks) > 1:
                b = self.shared_beta
                pooled = {t: np.mean([stats[u][2] for u in tasks if u != t], axis=0) for t in tasks}
                stats = {t: (m, s, (1.0 - b) * S + b * pooled[t], Xn) for t, (m, s, S, Xn) in stats.items()}
            pq = {"d1": [], "disp": [], "dst": []}
            self.tasks = {}
            for t in tasks:
                rc, rf = rows_c[t], rows_f[t]
                T = _Task()
                T.rows = rc
                mean, std, Sw, Xn_f = stats[t]
                W = _whiten(Sw, self.ridge_main, self.codes, Xn_f)
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
                for fi, key in ((0, "P0"), (1, "P1")):
                    Pm = C[key][rc].mean(0)
                    V = C[key][rc] - Pm
                    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
                    setattr(T, f"Vm{fi}", Pm.astype(np.float32))
                    setattr(T, f"V{fi}", np.ascontiguousarray(V, np.float32))
                T.RS = np.ascontiguousarray(C["rs"][rc], np.float32)
                T.rs2 = (C["rs"][rc] ** 2).sum(1).astype(np.float32)
                ep_c = C["ep"][rc]
                # H1: terminal mask of the task's candidates, median library episode length (decisions)
                T.term = np.ascontiguousarray(self.term_flag[rc])
                T.med_len = float(np.median([eplen[rc][ep_c == e][0] for e in np.unique(ep_c)]))
                # ---- early fit (step-0 branch)
                T.Z0 = T.A0 = T.As0 = T.n20 = T.W0f = T.c0 = None
                if self.early:
                    f0 = self.feat0
                    Xf0 = self._feats(F["P0"][rf], F["P1"][rf], F["rs"][rf], f0)
                    m0 = F["st"][rf] <= EARLY_MAX_STEP
                    if m0.sum() < 2 or len(np.unique(F["ep"][rf][m0])) < 2:
                        raise api.ContractError(f"task {t}: too few early rows ({int(m0.sum())}) in {fit_name}")
                    if alpha is None:
                        mn0, sd0, S0, Xn0 = _stats(Xf0[m0], F["heads"][rf][m0], F["ep"][rf][m0], nn=self.nn)
                    else:
                        Xc0 = self._feats(C["P0"][rc], C["P1"][rc], C["rs"][rc], f0)
                        m0c = C["st"][rc] <= EARLY_MAX_STEP
                        if m0c.sum() < 2 or len(np.unique(C["ep"][rc][m0c])) < 2:
                            raise api.ContractError(f"task {t}: too few early rows ({int(m0c.sum())}) in current")
                        mn0, sd0, S0, Xn0 = _stats(Xf0[m0], F["heads"][rf][m0], F["ep"][rf][m0], Xc0[m0c],
                                                   C["heads"][rc][m0c], C["ep"][rc][m0c], alpha, self.nn)
                    W0 = _whiten(S0, self.lam, self.codes, Xn0)
                    if f0 == "joint" and self.state_scale != 1.0:
                        sd0 = sd0.copy()
                        sd0[-RSV:] /= self.state_scale
                    Xc0 = self._feats(C["P0"][rc], C["P1"][rc], C["rs"][rc], f0)
                    Z0 = ((Xc0 - mn0) / sd0) @ W0
                    T.W0f = np.ascontiguousarray(W0 / sd0[:, None], np.float32)
                    shift0 = (mn0 / sd0) @ W0
                    if self.codes:
                        T.Z0 = np.ascontiguousarray(Z0, np.float32)
                        b = np.zeros(W0.shape[1])
                        Y = T.Z0.astype(np.float64)
                    else:
                        Winv = np.linalg.inv(W)
                        dv = dfe
                        A = Winv @ (np.diag(std / sd0[:dv]) @ W0[:dv])
                        b = ((mean - mn0[:dv]) / sd0[:dv]) @ W0[:dv]
                        Y = Zr @ A
                        if f0 != feat:
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
                    Hk = C["heads"][rc][idx]
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
        self.os_library, self.os_fit_library = self.cand_name, fit_name
        self.fixed_bytes = int(4 * (2 * (self.mu0.size + self.B0T.size)
                                    + sum(T.Wf.size + T.shift.size + (T.W0f.size if T.W0f is not None else 0)
                                          + (T.A0.size if T.A0 is not None else 0) for T in self.tasks.values())))
        self.n_cand = {t: int(len(T.rows)) for t, T in self.tasks.items()}
        for a in [self.sig, self.mu0, self.mu1, self.B0T, self.B1T, self.muB0, self.muB1, self.act, self.lib_ep,
                  self.lib_step, self.term_flag, self.term2] + [x for T in self.tasks.values() for x in vars(T).values()
                                                                 if isinstance(x, np.ndarray)]:
            a.flags.writeable = False

    # ---------------------------------------------------------------------------------------------------- query
    def reset(self, episode):
        super().reset(episode)
        self._pv = None

    @staticmethod
    def _hist_gsign(q):
        """Executed gripper signs of this episode so far, flattened in time (step x executed steps), int8 +-1."""
        h = np.asarray(q.hist_a_exec)[:, :ES, GD]
        return np.where(h >= 0, 1, -1).astype(np.int8).ravel()

    def _term_mask(self, q, T, gs=None):
        """-> (valid bool mask over T.rows or None, n_masked, gate_open). None = no mask (switch off, gate open, or
        the mask would empty the candidate set)."""
        if not self.term_guard:
            return None, 0.0, 0.0
        step = int(q.step)
        gate = 0.0
        if step > 0 and step >= self.term_late * T.med_len:
            if self.term_gate == "late":
                gate = 1.0
            else:
                s = self._hist_gsign(q) if gs is None else gs
                closed = np.flatnonzero(s == -self.open_sign)
                if closed.size and bool((s[closed[0]:] == self.open_sign).any()):
                    gate = 1.0
        if gate:
            return None, 0.0, gate
        nm = int(T.term.sum())
        if nm == 0 or nm == T.term.size:
            return None, 0.0, gate
        return ~T.term, float(nm), gate

    def _grip(self, a, Ck, wn, q, ex, gs=None):
        """gvote always; grip_commit's served sign on executed steps 0..4 of a (in place); updates self._pv."""
        sg = np.where(Ck[:, 0, GD] >= 0, 1.0, -1.0)
        v = float(np.dot(wn.astype(np.float64), sg))
        ex["gvote"] = v
        if not self.grip_commit:
            return
        step = int(q.step)
        held = flip = 0.0
        if step == 0 or q.prev_a_exec is None:
            served = 1.0 if float(Ck[0, 0, GD]) >= 0 else -1.0
            dwell = 0
        else:
            s = self._hist_gsign(q) if gs is None else gs
            prev_sign = float(s[-1])
            chg = np.flatnonzero(s[1:] != s[:-1])
            dwell = step - int((chg[-1] + 1) // ES) if chg.size else step
            vs = 1.0 if v >= 0 else -1.0
            if vs == prev_sign:
                served = prev_sign
            else:
                pv = self._pv
                confirmed = pv is not None and abs(pv) >= self.grip_thr and (
                    self.grip_confirm == "mag" or (pv >= 0) == (v >= 0))
                if abs(v) >= self.grip_thr and confirmed and dwell >= self.grip_dwell:
                    served, flip = vs, 1.0
                else:
                    served, held = prev_sign, 1.0
        self._pv = v
        a[:ES, GD] = np.float32(served)
        if self.grip_class_mean:
            m = sg == served
            nm = int(m.sum())
            if 3 <= nm < m.size:
                wm = wn[m].astype(np.float64)
                wm = (wm / wm.sum()).astype(np.float32)
                a[:, :GD] = np.tensordot(wm, Ck[m][:, :, :GD], 1)
        ex["gheld"], ex["gflip"], ex["gdwell"] = held, flip, float(dwell)

    def query(self, q):
        T, step, regime, k0, k1, xv, rs8, d, med, c, dt = self._dist(q)
        gs = self._hist_gsign(q) if (step > 0 and (self.term_guard or self.grip_commit)) else None
        valid, n_masked, gate = self._term_mask(q, T, gs)
        with self.prof.section("synth"):
            dsel = dt if valid is None else np.where(valid, dt, np.float32(np.inf))
            n = dsel.shape[0]
            k = min(self.k, n if valid is None else int(valid.sum()))
            # top-k by distance, exact ties by lower row (= the G3 wrappers' topk_pos convention; AWM's plain
            # argpartition picks an arbitrary tied row -- duplicate step-0 keys exist in the 10x library)
            if k < n:
                part = np.argpartition(dsel, k - 1)[:k]
                cand = np.flatnonzero(dsel <= dsel[part].max())
                idx = cand[np.lexsort((cand, dsel[cand]))][:k]
            else:
                idx = np.arange(n)
                idx = idx[np.lexsort((idx, dsel[idx]))]
            dk = dsel[idx].astype(np.float64)
            w = _kernel_w(dk - dk[0], self.kref)
            rows = T.rows[idx]
            wn, Ck, a = self._mix(rows, w)
            disp5 = self._disp5(T, a, idx[:5])
        d1 = float(d.min()) if valid is None else float(d[valid].min())
        dst = self._dst(T, rs8)
        c0 = float(c[idx[0]]) if c is not None else float("nan")
        conf = self._conf(T, regime, d1, disp5, dst, c0)
        ex = {"d1": d1, "d1_rel": d1 / med, "disp5": disp5, "dst": dst, "regime": float(regime),
              "lib_ep": float(self.lib_ep[rows[0]]), "lib_step": float(self.lib_step[rows[0]]),
              "w_eff": float(1.0 / (wn.astype(np.float64) ** 2).sum()),
              "w_term": float(np.dot(wn.astype(np.float64), self.term2[rows].astype(np.float64)))}
        if regime == 1:
            ex["c0"] = c0
        if step > 0:
            h0, h1 = q.hist_key_v0[-1], q.hist_key_v1[-1]
            s = 0.0
            for cur, prv, mu in ((k0, h0, self.mu0), (k1, h1, self.mu1)):
                u = cur - mu
                vv = np.asarray(prv, np.float32) - mu
                s += float(u @ vv) / max(float(np.sqrt((u @ u) * (vv @ vv))), 1e-12)
            ex["still"] = s
        if self.term_guard:
            ex["term_masked"], ex["term_open"] = n_masked, gate
        if self.prior_alpha is not None:
            ex["prior_alpha"] = self.prior_alpha
        self._insure(a, Ck, wn, step, ex)
        self._grip(a, Ck, wn, q, ex, gs)
        return api.Result(topk=rows.astype(np.int64), scores=-dt[idx].astype(np.float64), confidence=float(conf),
                          action=a, library=self.cand_name, extras=ex)

    # ------------------------------------------------------------------------ G3 base contract (V6 / V7 wrappers)
    def os_score_all(self, q):
        """(rows, S, aux) over the ADMISSIBLE task candidates (term_guard's mask applied); stateless."""
        T, step, regime, k0, k1, xv, rs8, d, med, c, dt = self._dist(q)
        valid, _, _ = self._term_mask(q, T)
        sel = slice(None) if valid is None else valid
        dt64 = dt[sel].astype(np.float64)
        n = dt64.shape[0]
        kr = min(self.kref, min(self.k, n))
        d1t = dt64.min()
        ref = max(float(np.partition(dt64, kr - 1)[kr - 1] - d1t), 1e-6)
        S = -((dt64 - d1t) / ref) ** 2
        aux = {"awm_d": d[sel]}
        if c is not None:
            aux["awm_c"] = c[sel]
        for f, p in ((0, xv[:PDIM]), (1, xv[PDIM:])):
            pc = p - getattr(T, f"Vm{f}")
            aux[f"vis_v{f}"] = (getattr(T, f"V{f}") @ (pc / max(float(np.sqrt(pc @ pc)), 1e-12)))[sel]
        return T.rows[sel], S, aux

    def os_synth(self, q, rows, w):
        """AWM's kernel mean of the given rows / weights + insurance + grip_commit (same per-episode state as
        query(): the served gripper sign of a wrapper's selection is this method's)."""
        wn, Ck, a = self._mix(np.asarray(rows, np.int64), np.asarray(w, np.float64))
        ex = {}
        self._insure(a, Ck, wn, int(q.step), ex)
        self._grip(a, Ck, wn, q, ex)
        return a

    def os_confidence(self, q, rows, scores, aux, served_rows, w):
        """AWM's confidence for a served set; aux arrays are indexed by the ADMISSIBLE rows (masked), T.HD by the
        task's rows."""
        T = self.tasks[int(q.task_id)]
        step = int(q.step)
        ph = q.prev_hit if step > 0 else None
        regime = 0 if step == 0 else (1 if ph is False else 2)
        served_rows = np.asarray(served_rows, np.int64)
        pos = np.searchsorted(T.rows, served_rows)
        _, _, a = self._mix(served_rows, np.asarray(w, np.float64))
        disp5 = self._disp5(T, a, pos[:5])
        rs8 = np.asarray(dims.valid_state(q.rs, self.model), np.float32)
        c0 = float("nan")
        if regime == 1:
            pm = np.searchsorted(np.asarray(rows, np.int64), served_rows[:1])
            c0 = float(aux["awm_c"][pm[0]])
        return self._conf(T, regime, float(np.asarray(aux["awm_d"]).min()), disp5, self._dst(T, rs8), c0)

    def bytes_per_entry(self):
        return float(super().bytes_per_entry() + (1.0 if self.term_guard else 0.0))
