"""R1 family F4 -- M7 `cascade_biglib` (ideation B-P4): state/continuity pre-filter on the BIG library to M=16
candidates per query, then a vision re-rank of those 16.

Library: pi0.5 -> bpool_cs (500 episodes/suite, Aug batch), GR00T -> bpool_all (500 episodes/suite). Task filter as
the online system. Per task: heads H_r = action[r, :5, :7] / sigma (35-d), states S_r = rs[r, :8] (valid dims only).

Regimes (online-legal: q.step, q.prev_hit):
  step 0            : f_pre = d / s_d                         (state only)
  fresh (prev MISS) : f_pre = c / s_c + alpha * d / s_d        c = RMS_sigma(H_r - prev_a_exec[5:10, :7]), alpha 0.25
  stale (prev HIT)  : f_pre = d / s_d                         (continuity to a library tail is trivially ~0 after a
                                                               HIT; prev_hit None at step >= 1 is treated as stale)
Prefilter: the M = 16 smallest f_pre. Re-rank the 16 by (higher = better)
  fresh             : f = vis + 0.5 * (-c / s_c)
  step 0 / stale    : f = vis                                 ("none": f = -d / s_d, i.e. the prefilter order)
with the vision term `vis` (rerank x keys):
  b0fused (raw keys): B0's online formula on the 16: sum_i w_i * 0.5 * (tanh((x_i - mu_i) / sigma_i) + 1) over
                      (cos v0, cos v1, -L2 rs), weights / mu / sigma = trace_dual yaml (ctx.current_params()).
  both              : (cos_v0 - mu_v0) / sigma_v0 + (cos_v1 - mu_v1) / sigma_v1 (added after the M9 probe: both cameras
                      z-summed is the best training-free early-step representation), keys raw or pcaK (per-field PCA).
  v1only            : linear z of the wrist cosine, (cos_v1 - mu_v) / sigma_v, with keys "raw" (pooled 32768-d) or
                      "pcaK" (cosine of the centered keys in the leading K PCs of a task-agnostic PCA fitted on the
                      big library, m7_pca_fit.py; K in {32, 64, 128}).
                      mu_v / sigma_v are FIXED library-fitted constants (choice documented here, instead of the
                      "z within the 16" alternative): library rows as pseudo-queries, their 16 state-prefiltered
                      candidates from OTHER episodes of the same task, mean / std of all those cosines. So sigma_v is
                      the typical spread of the cosine inside a prefiltered 16-set, the vision z has the same unit
                      scale as a within-16 z, and it keeps an absolute meaning for the confidence.
  none              : no vision term (control).
Synthesis "med3": per-step median of the top-3 (by f) chunks on dims 0..5, dim 6 = majority sign (ties -> +1), all
other entries (dims 7..31, steps 5..) from the top-1 chunk. Library actions only.
Confidence (regime-aware, library-fitted scales):
  fresh             : -c0 / s_c - disp3 / s_a
  step 0 / stale    : -d0 / s_d - disp3 / s_a + g,   g = (vis0 - mu_g) / sigma_g   (0 for "none")
  disp3 = mean pairwise RMS_sigma among the top-3 heads; c0 / d0 / vis0 of the top-1; mu_g / sigma_g = mean / std of
  the picked candidate's vis over the library pseudo-queries (as "the library-internal top-1 score spread", B-P3).
Scales as B-P1 (library rows vs rows of OTHER episodes of the same task): s_d median 1-NN rs L2, s_c median 1-NN
RMS_sigma(tail_r[5:10] -> head), s_a median disp3 of the top-3 under the fresh prefilter score.

Tier: T0 (raw keys) / T1 (PCA, library-only fit). Bytes/entry: head 35 + state 8 floats, plus the vision
representation (b0fused / both raw: two 32768-d keys; v1only raw: one; pcaK: K floats per field).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from exp.offline_search.harness import api, dims

BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
NH = dims.EXEC_STEPS * dims.ACT_DIMS        # 35
PCA_DIR = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/pca")
_EPS = np.float32(1e-8)


def _fmt(x) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def _pair_rms(Q, R, ndim):
    q2 = np.einsum("ij,ij->i", Q, Q)[:, None]
    r2 = np.einsum("ij,ij->i", R, R)[None, :]
    e = q2 - 2.0 * (Q @ R.T) + r2
    np.maximum(e, 0.0, out=e)
    e /= ndim
    return np.sqrt(e, out=e)


def _disp(Hk):
    k = Hk.shape[0]
    if k < 2:
        return 0.0
    iu = np.triu_indices(k, 1)
    D = Hk[iu[0]] - Hk[iu[1]]
    return float(np.sqrt(np.einsum("ij,ij->i", D, D) / D.shape[1]).mean())


def _disp_batch(Hk):
    k = Hk.shape[1]
    if k < 2:
        return np.zeros(Hk.shape[0], np.float32)
    iu = np.triu_indices(k, 1)
    D = Hk[:, iu[0]] - Hk[:, iu[1]]
    return np.sqrt(np.mean(D * D, axis=2)).mean(axis=1)


def _median3(ch):
    a, b, c = ch[0], ch[1], ch[2]
    return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))


def _zt(x, mu, sigma):
    sigma = sigma if sigma > 1e-12 else 1.0
    return np.float32(0.5) * (np.tanh((x - np.float32(mu)) / np.float32(sigma)) + np.float32(1.0))


def _cos_rows(Kmm, rows, q):
    """Cosine of q against library key rows (gathered from the read-only memmap, sorted access)."""
    o = np.argsort(rows)
    V = np.asarray(Kmm[rows[o]], np.float32)
    q = np.asarray(q, np.float32)
    c = (V @ q) / np.maximum(np.linalg.norm(V, axis=1) * np.float32(np.sqrt(np.dot(q, q))), _EPS)
    out = np.empty_like(c)
    out[o] = c
    return out


class _Task:
    __slots__ = ("rows", "HD", "h2", "RS", "ep", "st", "P0", "P1")


class CascadeBigLib(api.Method):
    """M7 cascade_biglib. kwargs: rerank in {b0fused, v1only, both, none}, keys in {raw, pca32, pca64, pca128},
    M (prefilter size, 16), alpha (state weight in the fresh prefilter, 0.25), lam_c (continuity weight in the fresh
    re-rank, 0.5)."""

    tier = "T0"
    family = "f4_vision"

    def __init__(self, rerank="b0fused", keys="raw", M=16, alpha=0.25, lam_c=0.5):
        if rerank not in ("b0fused", "v1only", "both", "none"):
            raise ValueError(f"rerank must be b0fused | v1only | both | none, got {rerank!r}")
        if keys not in ("raw", "pca32", "pca64", "pca128"):
            raise ValueError(f"keys must be raw | pca32 | pca64 | pca128, got {keys!r}")
        if rerank == "b0fused" and keys != "raw":
            raise ValueError("b0fused uses B0's global mu/sigma, defined for the raw pooled keys only")
        self.rerank, self.keys, self.M, self.alpha, self.lam_c = rerank, keys, int(M), float(alpha), float(lam_c)
        self.pk = int(keys[3:]) if keys.startswith("pca") else 0
        self.tier = "T1" if self.pk else "T0"
        self.fields = {"b0fused": ("v0", "v1"), "both": ("v0", "v1"), "v1only": ("v1",), "none": ()}[rerank]
        kname = "none" if rerank == "none" else keys
        extra = "" if (self.M == 16 and self.alpha == 0.25 and self.lam_c == 0.5) else \
            f"_M{self.M}_a{_fmt(alpha)}_lc{_fmt(lam_c)}"
        self.name = f"M7_cascade_{kname}_{rerank}_med3{extra}"

    # ------------------------------------------------------------------------------------------ fit
    def fit(self, lib, ctx):
        self.model = ctx.model
        self.libname = BIG[ctx.model]
        L = ctx.open_library(self.libname)
        self.act = L.action
        sig = np.asarray(ctx.action_sigma, np.float32)
        self.sig = sig
        act7 = np.asarray(L.action[:, :, dims.ACT_VALID], np.float32)
        heads = (act7[:, :dims.EXEC_STEPS] / sig).reshape(len(act7), NH)
        tails = (act7[:, 5:10] / sig).reshape(len(act7), NH)
        rs = np.ascontiguousarray(dims.valid_state(np.asarray(L.rs, np.float32), ctx.model))
        ep = np.asarray(L.episode, np.int64)
        stp = np.asarray(L.step, np.int64)
        p = ctx.current_params()
        self.w, self.mu, self.sg = p["weights"], p["mu"], p["sigma"]
        self.Kmm = {f: getattr(L, f"key_{f}") for f in self.fields} if not self.pk else {}   # read-only memmaps
        P = {}
        self.pca, self.pmeta = {}, {}
        for f in (self.fields if self.pk else ()):
            d = PCA_DIR / ctx.lib_key / self.libname / f
            meta = json.loads((d / "meta.json").read_text())
            if meta["n"] != L.L:
                raise api.ContractError(f"PCA cache {d} was fitted on {meta['n']} rows, library has {L.L}")
            self.pca[f] = (np.load(d / "mean.npy").astype(np.float32),
                           np.ascontiguousarray(np.load(d / "basis.npy", mmap_mode="r")[:, :self.pk], np.float32))
            X = np.load(d / "proj.npy")[:, :self.pk].astype(np.float32)
            X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), _EPS)
            P[f] = X
            self.pmeta[f] = {"explained": meta["explained"].get(str(self.pk)), "n": meta["n"]}
        self.tasks = {}
        mins = {"d": [], "c": []}
        with self.prof.section("fit_index"):
            for t in L.tasks():
                r = np.asarray(L.rows_of_task(t), np.int64)
                T = _Task()
                T.rows = r
                T.HD = np.ascontiguousarray(heads[r])
                T.h2 = np.einsum("ij,ij->i", T.HD, T.HD)
                T.RS = np.ascontiguousarray(rs[r])
                T.ep = ep[r]
                T.st = stp[r]
                T.P0 = np.ascontiguousarray(P["v0"][r]) if "v0" in P else None
                T.P1 = np.ascontiguousarray(P["v1"][r]) if "v1" in P else None
                self.tasks[int(t)] = T
                same = T.ep[:, None] == T.ep[None, :]
                D = _pair_rms(T.RS, T.RS, 1.0); D[same] = np.inf; mins["d"].append(D.min(1))
                C = _pair_rms(tails[r], T.HD, NH); C[same] = np.inf; mins["c"].append(C.min(1))
        med = {k: float(np.median(np.concatenate(v)[np.isfinite(np.concatenate(v))])) for k, v in mins.items()}
        self.s_d, self.s_c = med["d"] + 1e-9, med["c"] + 1e-9
        # s_a (fresh-prefilter top-3 dispersion) + vision calibration (library pseudo-queries, state prefilter)
        rng = np.random.default_rng(0)
        disps = []
        cal = {f: [] for f in self.fields}
        cal_sets = []
        self.mu_v = {f: 0.0 for f in self.fields}
        self.sg_v = {f: 1.0 for f in self.fields}
        with self.prof.section("fit_scales"):
            for t, T in self.tasks.items():
                r = T.rows
                same = T.ep[:, None] == T.ep[None, :]
                F = _pair_rms(tails[r], T.HD, NH) / self.s_c + self.alpha * _pair_rms(T.RS, T.RS, 1.0) / self.s_d
                F[same] = np.inf
                k = min(3, max(1, len(r) - 1))
                o = np.argpartition(F, k - 1, axis=1)[:, :k]
                ok = np.isfinite(np.take_along_axis(F, o, 1)).all(1)
                disps.append(_disp_batch(T.HD[o[ok]]))
                if self.rerank == "none":
                    continue
                sel = rng.choice(len(r), size=min(40, len(r)), replace=False)
                Dq = _pair_rms(T.RS[sel], T.RS, 1.0)
                Dq[T.ep[sel][:, None] == T.ep[None, :]] = np.inf
                M = min(self.M, len(r))
                cand = np.argsort(Dq, axis=1, kind="stable")[:, :M]
                for j, i in enumerate(sel):
                    cj = cand[j][np.isfinite(Dq[j, cand[j]])]
                    if cj.size == 0:
                        continue
                    cs = self._cosines(T, cj, {f: self._qvec(f, T, i) for f in self.fields})
                    for f in self.fields:
                        cal[f].append(cs[f])
                    cal_sets.append((cs, Dq[j, cj]))
        self.s_a = float(np.median(np.concatenate(disps))) + 1e-9
        self.mu_g = self.sg_g = float("nan")
        if self.rerank != "none":
            for f in self.fields:
                ca = np.concatenate(cal[f])
                self.mu_v[f], self.sg_v[f] = float(ca.mean()), float(ca.std() + 1e-9)
            picks = [float(np.max(self._vis(cs, dd))) for cs, dd in cal_sets]
            self.mu_g, self.sg_g = float(np.mean(picks)), float(np.std(picks) + 1e-9)
        self.scales = {"s_d": self.s_d, "s_c": self.s_c, "s_a": self.s_a, "mu_v": self.mu_v, "sg_v": self.sg_v,
                       "mu_g": self.mu_g, "sg_g": self.sg_g}
        try:
            (ctx.scratch / f"{self.name}_scales.json").write_text(json.dumps(
                {"library": self.libname, "L": int(L.L), **self.scales,
                 **({"pca": self.pmeta} if self.pk else {})}, indent=1))
        except Exception:
            pass

    def _qvec(self, f, T, i):
        """Library row T.rows[i] as a pseudo-query: its raw key (raw) or its normalized projection (pca)."""
        if self.pk:
            return (T.P0 if f == "v0" else T.P1)[i]
        return self.Kmm[f][T.rows[i]]

    def _cosines(self, T, cand, qv):
        """{field: cosine of the query vector vs the candidates}; qv: raw keys (raw) or unit projections (pca)."""
        out = {}
        for f in self.fields:
            if self.pk:
                out[f] = (T.P0 if f == "v0" else T.P1)[cand] @ qv[f]
            else:
                out[f] = _cos_rows(self.Kmm[f], T.rows[cand], qv[f])
        return out

    def _vis(self, cs, dsel):
        """Vision term (higher = better) from the field cosines and the candidates' state distances."""
        if self.rerank == "b0fused":
            return (self.w[0] * _zt(cs["v0"], self.mu[0], self.sg[0]) + self.w[1] * _zt(cs["v1"], self.mu[1], self.sg[1])
                    + self.w[2] * _zt(-np.asarray(dsel, np.float32), self.mu[2], self.sg[2]))
        return sum((cs[f] - self.mu_v[f]) / self.sg_v[f] for f in self.fields)

    def bytes_per_entry(self):
        per = (self.pk if self.pk else dims.KEY_DIM) * 4
        return float(4 * (NH + 8) + len(self.fields) * per)

    # ---------------------------------------------------------------------------------------- query
    def query(self, q):
        T = self.tasks.get(int(q.task_id))
        if T is None or T.rows.size == 0:
            raise api.ContractError(f"task {q.task_id} has no rows in library {self.libname}")
        step = int(q.step)
        with self.prof.section("prefilter"):
            rs = dims.valid_state(np.asarray(q.rs, np.float32), self.model)
            dv = T.RS - rs
            d = np.sqrt(np.einsum("ij,ij->i", dv, dv))
            c = None
            if step == 0:
                regime = 0
                fpre = d / self.s_d
            else:
                regime = 1 if q.prev_hit is False else 2
                if regime == 1:
                    tail = (np.asarray(q.prev_a_exec[5:10, dims.ACT_VALID], np.float32) / self.sig).reshape(NH)
                    c = np.sqrt(np.maximum(T.h2 - 2.0 * (T.HD @ tail) + float(tail @ tail), 0.0) / NH)
                    fpre = c / self.s_c + self.alpha * (d / self.s_d)
                else:
                    fpre = d / self.s_d
            C = fpre.shape[0]
            M = min(self.M, C)
            cand = np.argpartition(fpre, M - 1)[:M] if C > M else np.arange(C)
            cand = cand[np.lexsort((cand, fpre[cand]))]           # prefilter order (ties -> library order)
        with self.prof.section("vision"):
            cs = {}
            if self.rerank == "none":
                vis = np.zeros(M, np.float32)
            else:
                qv = {}
                for f in self.fields:
                    x = q.key_v0 if f == "v0" else q.key_v1
                    if self.pk:
                        mu, B = self.pca[f]
                        x = (np.asarray(x, np.float32) - mu) @ B
                        x /= max(float(np.linalg.norm(x)), 1e-8)
                    qv[f] = x
                cs = self._cosines(T, cand, qv)
                vis = self._vis(cs, d[cand])
        with self.prof.section("rerank"):
            if regime == 1:
                f = vis + self.lam_c * (-c[cand] / self.s_c)
            elif self.rerank == "none":
                f = -d[cand] / self.s_d
            else:
                f = vis
            f = np.asarray(f, np.float64)
            o = np.lexsort((np.arange(M), -f))                       # stable: ties keep the prefilter order
            rk = cand[o]
        with self.prof.section("synth"):
            k3 = min(3, M)
            disp = _disp(T.HD[rk[:k3]])
            ch = np.asarray(self.act[T.rows[rk[:k3]]], np.float32)
            action = ch[0].copy()
            if k3 == 3:
                action[:, :6] = _median3(ch[:, :, :6])
            g = np.sign(ch[:, :, dims.GRIPPER_DIM]).sum(0)
            action[:, dims.GRIPPER_DIM] = np.where(g >= 0, 1.0, -1.0)
        i0 = rk[0]
        c0 = float(c[i0]) if c is not None else float("nan")
        d0 = float(d[i0])
        vis0 = float(vis[o[0]])
        if regime == 1:
            conf = -c0 / self.s_c - disp / self.s_a
            gterm = 0.0
        else:
            gterm = 0.0 if self.rerank == "none" else (vis0 - self.mu_g) / self.sg_g
            conf = -d0 / self.s_d - disp / self.s_a + gterm
        extras = {"regime": regime, "c0": c0, "d0": d0, "disp": disp, "vis0": vis0,
                  "cos1_0": float(cs["v1"][o[0]]) if "v1" in cs else float("nan"),
                  "cos0_0": float(cs["v0"][o[0]]) if "v0" in cs else float("nan"), "g": float(gterm),
                  "rank_pre": int(o[0]), "lib_ep": int(T.ep[i0]), "lib_step": int(T.st[i0])}
        return api.Result(topk=T.rows[rk], scores=f[o], confidence=float(conf), action=action,
                          library=self.libname, extras=extras)
