"""R1 family F1 -- action-chunk continuity (M1 `chunk_cont`) and the reference R_tail (`tail_passthrough`).

M1 = merge of ideation A-P1 (tail_continuity) and B-P1 (chunk_continuity_search), see rounds/r01/NOTES_ideation_*.md.

Per task, library index: heads  H_r = action[r, :5, :7] / sigma   (35-d, sigma = err-metric sigma_d)
                          state  S_r = rs[r, :8]                    (valid robot_state dims only)
                          window W_r = [S_r, S_prev(r), S_prev2(r)] / sd   (24-d, stale="window" only; sd = std of the
                                                                    library's rs[:8]; missing prev -> repeat oldest)
Query regimes (online-legal: q.step, q.prev_hit = HIT/MISS flag of the previous decision):
  0 step 0         : f = d / s_d                                   (state only)
  1 fresh (MISS)   : f = c / s_c + alpha * d / s_d [+ beta * c2 / s_c2]
                     c  = RMS_sigma(H_r - prev_a_exec[5:10, :7])   (previous chunk's unexecuted tail)
                     c2 = RMS_sigma(H_r - hist_a_exec[-2][10:15, :7])   (GR00T only, step >= 2)
  2 stale (HIT)    : stale="cont"   -> same score as fresh (B-P1 as proposed)
                     stale="window" -> f = w / s_w, w = L2 of the 3-decision z-scaled state window (A-P2)
  (step >= 1 with prev_hit None -- undecidable -- is treated as stale: the conservative choice for the confidence.)
The beta term is part of the continuity score: it is added whenever that score is used (fresh; stale only under
stale="cont") and step >= 2.

Synthesis: "top1" (no synthesized action; the harness scores topk[0]) | "kmean5" kernel-weighted mean of the top-5 full
chunks, w_i = exp(-(f_i / f_0)^2) (A-P1) | "med3" per-step median of the top-3 for dims 0-5, dim 6 = majority sign
(ties -> +1), everything else (dims 7..31) from the top-1 chunk (B-P1). Only library actions are used.

Confidence (regime-aware, fixed library-fitted scales; never the continuity after a HIT, where it is trivially small):
  fresh      : -c0 / s_c - disp_k / s_a
  step 0     : -d0 / s_d - disp_k / s_a
  stale      : -d0 / s_d - disp_k / s_a      (stale="cont")   |   -w0 / s_w - disp_k / s_a   (stale="window")
  disp_k = mean pairwise RMS_sigma among the heads of the top-k (k = 5 for kmean5, else 3); c0 / d0 / w0 of the top-1.

Library-fitted scales (T1, seconds; library rows as pseudo-queries against rows of OTHER episodes of the same task):
  s_d  median 1-NN rs L2                         s_c  median 1-NN RMS_sigma(tail_r[5:10] -> head)
  s_c2 median 1-NN RMS_sigma(chunk_r[10:15] -> head)   (GR00T, beta > 0)
  s_w  median 1-NN window L2 (stale="window")    s_a  median disp_k of the top-k under the fresh score (this alpha)

Library: lib="current" | "big" (pi05 -> bpool_cs, GR00T -> bpool_all).
"""
from __future__ import annotations

import json

import numpy as np

from exp.offline_search.harness import api, dims

BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
TOPK = 10
NH = dims.EXEC_STEPS * dims.ACT_DIMS          # 35
WIN = 3                                       # decisions in the state window


def _fmt(x) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def _pair_rms(Q, R, ndim):
    """RMS distance matrix (n, m) between rows of Q (n, D) and R (m, D) (float32 GEMM expansion)."""
    q2 = np.einsum("ij,ij->i", Q, Q)[:, None]
    r2 = np.einsum("ij,ij->i", R, R)[None, :]
    e = q2 - 2.0 * (Q @ R.T) + r2
    np.maximum(e, 0.0, out=e)
    e /= ndim
    return np.sqrt(e, out=e)


def _pair_l2(Q, R):
    return _pair_rms(Q, R, 1.0)


_IU = {k: np.triu_indices(k, 1) for k in range(2, 11)}


def _disp(Hk):
    """Mean pairwise RMS among k head rows (k, 35) -> float; 0 for k < 2."""
    k = Hk.shape[0]
    if k < 2:
        return 0.0
    iu = _IU[k]
    D = Hk[iu[0]] - Hk[iu[1]]
    return float(np.sqrt(np.einsum("ij,ij->i", D, D) / D.shape[1]).mean())


def _median_rows(ch):
    """Per-element median over axis 0 of a (k, ...) stack; exact min/max network for k = 3."""
    if ch.shape[0] == 3:
        a, b, c = ch[0], ch[1], ch[2]
        return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))
    return np.median(ch, axis=0)


def _disp_batch(Hk):
    """Hk (n, k, 35) -> (n,) mean pairwise RMS."""
    k = Hk.shape[1]
    if k < 2:
        return np.zeros(Hk.shape[0], np.float32)
    iu = np.triu_indices(k, 1)
    D = Hk[:, iu[0]] - Hk[:, iu[1]]
    return np.sqrt(np.mean(D * D, axis=2)).mean(axis=1)


def _windows(Z, prev, m=WIN):
    """Z (L, 8) scaled states, prev (L,) previous-row pointer (-1 none) -> (L, 8*m); missing -> repeat the oldest."""
    L = Z.shape[0]
    out = np.empty((L, Z.shape[1] * m), np.float32)
    cur = np.arange(L)
    for j in range(m):
        out[:, j * Z.shape[1]:(j + 1) * Z.shape[1]] = Z[cur]
        nxt = prev[cur]
        cur = np.where(nxt >= 0, nxt, cur)
    return out


class _Task:
    __slots__ = ("rows", "HD", "h2", "RS", "W", "ep", "st")


class ChunkCont(api.Method):
    """M1 chunk_cont. kwargs: lib in {current, big}, alpha >= 0, synth in {top1, kmean5, med3},
    stale in {window, cont}, beta >= 0 (GR00T only; pi05 cells are skipped when beta > 0)."""

    tier = "T1"
    family = "f1_continuity"

    def __init__(self, lib="big", alpha=0.25, synth="med3", stale="window", beta=0):
        if lib not in ("current", "big"):
            raise ValueError(f"lib must be 'current' or 'big', got {lib!r}")
        if synth not in ("top1", "kmean5", "med3"):
            raise ValueError(f"synth must be top1 | kmean5 | med3, got {synth!r}")
        if stale not in ("window", "cont"):
            raise ValueError(f"stale must be window | cont, got {stale!r}")
        self.lib, self.alpha, self.synth, self.stale, self.beta = lib, float(alpha), synth, stale, float(beta)
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be >= 0")
        self.kd = 5 if synth == "kmean5" else 3          # dispersion / synthesis neighbourhood
        st = {"window": "Win", "cont": "Cont"}[stale]
        self.name = f"M1_{lib}_a{_fmt(alpha)}_{synth}_st{st}_b{_fmt(beta)}"

    # ------------------------------------------------------------------------------------------ fit
    def fit(self, lib, ctx):
        if self.beta > 0 and ctx.model != "groot":
            raise api.SkipCell("beta > 0 needs a 16-step chunk (GR00T); pi05 H=10 has no steps 10..14")
        self.model = ctx.model
        self.H = ctx.horizon
        self.libname = "current" if self.lib == "current" else BIG[ctx.model]
        L = lib if self.libname == "current" else ctx.open_library(self.libname)
        self.act = L.action                                   # read-only memmap, shared by the forked workers
        sig = np.asarray(ctx.action_sigma, np.float32)
        self.sig = sig
        act7 = np.asarray(L.action[:, :, dims.ACT_VALID], np.float32)
        heads = (act7[:, :dims.EXEC_STEPS] / sig).reshape(len(act7), NH)
        tails = (act7[:, 5:10] / sig).reshape(len(act7), NH)
        tails2 = (act7[:, 10:15] / sig).reshape(len(act7), NH) if self.beta > 0 else None
        rs = np.ascontiguousarray(dims.valid_state(np.asarray(L.rs, np.float32), ctx.model))
        ep = np.asarray(L.episode, np.int64)
        stp = np.asarray(L.step, np.int64)
        use_win = self.stale == "window"
        if use_win:
            self.sd = (rs.std(0) + 1e-6).astype(np.float32)
            Wall = _windows(rs / self.sd, np.asarray(L.prev, np.int64))
        self.tasks = {}
        mins = {"d": [], "c": [], "c2": [], "w": []}
        with self.prof.section("fit_index"):
            for t in L.tasks():
                r = np.asarray(L.rows_of_task(t), np.int64)
                T = _Task()
                T.rows = r
                T.HD = np.ascontiguousarray(heads[r])
                T.h2 = np.einsum("ij,ij->i", T.HD, T.HD)
                T.RS = np.ascontiguousarray(rs[r])
                T.W = np.ascontiguousarray(Wall[r]) if use_win else None
                T.ep = ep[r]
                T.st = stp[r]
                self.tasks[int(t)] = T
                same = T.ep[:, None] == T.ep[None, :]
                D = _pair_l2(T.RS, T.RS); D[same] = np.inf; mins["d"].append(D.min(1))
                Cm = _pair_rms(tails[r], T.HD, NH); Cm[same] = np.inf; mins["c"].append(Cm.min(1))
                if tails2 is not None:
                    C2 = _pair_rms(tails2[r], T.HD, NH); C2[same] = np.inf; mins["c2"].append(C2.min(1))
                if use_win:
                    Wd = _pair_l2(T.W, T.W); Wd[same] = np.inf; mins["w"].append(Wd.min(1))
        med = {}
        for k, v in mins.items():
            x = np.concatenate(v) if v else np.zeros(0)
            x = x[np.isfinite(x)]
            med[k] = float(np.median(x)) if x.size else float("nan")
        self.s_d, self.s_c = med["d"] + 1e-9, med["c"] + 1e-9
        self.s_c2 = med["c2"] + 1e-9 if tails2 is not None else float("nan")
        self.s_w = med["w"] + 1e-9 if use_win else float("nan")
        # s_a: library rows as fresh-regime pseudo-queries (score of this alpha), dispersion of their top-kd heads
        disps = []
        with self.prof.section("fit_scales"):
            for t, T in self.tasks.items():
                r = T.rows
                same = T.ep[:, None] == T.ep[None, :]
                F = _pair_rms(tails[r], T.HD, NH) / self.s_c
                if self.alpha > 0:
                    F += self.alpha * _pair_l2(T.RS, T.RS) / self.s_d
                F[same] = np.inf
                k = min(self.kd, max(1, len(r) - 1))
                o = np.argpartition(F, k - 1, axis=1)[:, :k]
                ok = np.isfinite(np.take_along_axis(F, o, 1)).all(1)
                disps.append(_disp_batch(T.HD[o[ok]]))
        self.s_a = float(np.median(np.concatenate(disps))) + 1e-9
        self.scales = {"s_d": self.s_d, "s_c": self.s_c, "s_c2": self.s_c2, "s_w": self.s_w, "s_a": self.s_a}
        try:
            (ctx.scratch / f"{self.name}_scales.json").write_text(json.dumps(
                {"library": self.libname, "L": int(L.L), **self.scales}, indent=1))
        except Exception:
            pass

    def bytes_per_entry(self):
        # 35-d head index + 8-d state (+ 24-d state window when the stale regime uses it), float32
        return float(4 * (NH + 8 + (8 * WIN if self.stale == "window" else 0)))

    # ---------------------------------------------------------------------------------------- query
    def _qwindow(self, q):
        rs = dims.valid_state(np.asarray(q.rs, np.float32), self.model)
        parts = [rs]
        h = q.hist_rs
        n = h.shape[0]
        for j in range(1, WIN):
            parts.append(dims.valid_state(np.asarray(h[n - j], np.float32), self.model) if n - j >= 0 else parts[-1])
        return np.concatenate(parts) / np.tile(self.sd, WIN)

    def _core(self, q):
        """Primary computation shared with R_tail. Returns (order, f, conf, action_or_None, extras)."""
        T = self.tasks.get(int(q.task_id))
        if T is None or T.rows.size == 0:
            raise api.ContractError(f"task {q.task_id} has no rows in library {self.libname}")
        step = int(q.step)
        with self.prof.section("score"):
            rs = dims.valid_state(np.asarray(q.rs, np.float32), self.model)
            dv = T.RS - rs
            d = np.sqrt(np.einsum("ij,ij->i", dv, dv))
            c = None
            w = None
            if step == 0:
                regime = 0
                f = d / self.s_d
            else:
                ph = q.prev_hit
                regime = 1 if ph is False else 2
                tail = (np.asarray(q.prev_a_exec[5:10, dims.ACT_VALID], np.float32) / self.sig).reshape(NH)
                c = np.sqrt(np.maximum(T.h2 - 2.0 * (T.HD @ tail) + float(tail @ tail), 0.0) / NH)
                if regime == 1 or self.stale == "cont":
                    f = c / self.s_c
                    if self.alpha > 0:
                        f = f + self.alpha * (d / self.s_d)
                    if self.beta > 0 and step >= 2:
                        t2 = (np.asarray(q.hist_a_exec[-2][10:15, dims.ACT_VALID], np.float32) / self.sig).reshape(NH)
                        c2 = np.sqrt(np.maximum(T.h2 - 2.0 * (T.HD @ t2) + float(t2 @ t2), 0.0) / NH)
                        f = f + self.beta * (c2 / self.s_c2)
                else:
                    x = self._qwindow(q)
                    wv = T.W - x
                    w = np.sqrt(np.einsum("ij,ij->i", wv, wv))
                    f = w / self.s_w
        with self.prof.section("select"):
            C = f.shape[0]
            K = min(TOPK, C)
            o = np.argpartition(f, K - 1)[:K] if C > K else np.arange(C)
            o = o[np.lexsort((o, f[o]))]
        with self.prof.section("synth"):
            kd = min(self.kd, K)
            disp = _disp(T.HD[o[:kd]])
            action = None
            if self.synth == "med3":
                ch = np.asarray(self.act[T.rows[o[:min(3, K)]]], np.float32)
                action = ch[0].copy()
                action[:, :6] = _median_rows(ch[:, :, :6])
                g = np.sign(ch[:, :, dims.GRIPPER_DIM]).sum(0)
                action[:, dims.GRIPPER_DIM] = np.where(g >= 0, 1.0, -1.0)
            elif self.synth == "kmean5":
                k5 = min(5, K)
                ch = np.asarray(self.act[T.rows[o[:k5]]], np.float32)
                fk = f[o[:k5]].astype(np.float64)
                wk = np.exp(-(fk / (fk[0] + 1e-6)) ** 2)
                wk /= wk.sum()
                action = np.tensordot(wk, ch, axes=(0, 0)).astype(np.float32)
        i0 = o[0]
        c0 = float(c[i0]) if c is not None else float("nan")
        d0 = float(d[i0])
        w0 = float(w[i0]) if w is not None else float("nan")
        if regime == 1:
            conf = -c0 / self.s_c - disp / self.s_a
        elif regime == 2 and self.stale == "window":
            conf = -w0 / self.s_w - disp / self.s_a
        else:
            conf = -d0 / self.s_d - disp / self.s_a
        extras = {"regime": regime, "c0": c0, "d0": d0, "w0": w0, "disp": disp, "f0": float(f[i0]),
                  "lib_ep": int(T.ep[i0]), "lib_step": int(T.st[i0])}
        return o, f, float(conf), action, extras

    def query(self, q):
        o, f, conf, action, extras = self._core(q)
        T = self.tasks[int(q.task_id)]
        return api.Result(topk=T.rows[o], scores=-f[o].astype(np.float64), confidence=conf, action=action,
                          library=self.libname, extras=extras)


class TailPassthrough(ChunkCont):
    """Reference R_tail (NOT a library action): execute the previous chunk's unexecuted tail, i.e. replan-10 for
    this decision. action[:H-5] = prev_a_exec[5:H], action[H-5:] = prev_a_exec[H-1] (pi05: tail 5..9 then pad;
    GR00T: 5..15 then pad). Step 0 (no previous chunk): the M1 primary's step-0 action. topk / scores / confidence /
    extras are those of the M1 primary (big library, alpha 0.25, med3, stale window, beta 0) for its own pick.
    In cache arms the previous chunk was a library row, so the tail there is a library chunk's tail."""

    uses_nonlibrary_action = True
    family = "f1_continuity"

    def __init__(self):
        super().__init__(lib="big", alpha=0.25, synth="med3", stale="window", beta=0)
        self.name = "Rtail_passthrough"

    def query(self, q):
        o, f, conf, action, extras = self._core(q)
        T = self.tasks[int(q.task_id)]
        p = q.prev_a_exec
        if p is not None:
            H = self.H
            action = np.empty((H, dims.ACT_FULL_DIMS), np.float32)
            action[:H - dims.EXEC_STEPS] = p[dims.EXEC_STEPS:H]
            action[H - dims.EXEC_STEPS:] = p[H - 1]
            extras["tail"] = 1
        else:
            extras["tail"] = 0
        return api.Result(topk=T.rows[o], scores=-f[o].astype(np.float64), confidence=conf, action=action,
                          library=self.libname, extras=extras)
