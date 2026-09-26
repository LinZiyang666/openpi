"""R1 family F4 -- M9(c) `vision_zsum`: the best training-free visual representation of the M9 probe, wrapped as a
plain retrieval method over ALL candidates of the task (no step alignment, no continuity).

Representation (m9_tables.md): the pooled 4x4 keys of BOTH cameras, task-centred -- the task mean key over the
library rows of the task is subtracted from the query and the candidates before the cosine ("res_pool_tm") -- and
the two fields fused by a per-query z-score over the task candidates (equal weights, no tanh):
    S = z_q(cos(q0 - m0_t, c0 - m0_t)) + z_q(cos(q1 - m1_t, c1 - m1_t)) [+ st * z_q(-||rs_q - rs_c||)]
z_q = standardize over this query's task candidates. center="none" uses the raw cosines (the probe's "pool_both").
keys="pca128": cosine in the leading 128 PCs of the task-agnostic big-library PCA (m7_pca_fit.py), still task-centred
(big library only). Everything is fitted on the library only (task means, norms): T0 (raw) / T1 (PCA).

Selection: synth="top1" (harness scores topk[0]) or "med3" (per-step median of the top-3 on dims 0..5, gripper
majority, rest from the top-1). Confidence: S of the top-1 / n_terms - disp3 / s_a (s_a = median top-3 head
dispersion of library self-queries under the same score; small constant role, observation-only method).
Library: lib="current" (B0's library, per-task copies as B0) | "big" (pi0.5 bpool_cs / GR00T bpool_all; task rows are
contiguous there, so raw keys are zero-copy memmap views shared by the forked workers -- B0-like cost x10).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from exp.offline_search.harness import api, dims

BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
NH = dims.EXEC_STEPS * dims.ACT_DIMS
PCA_DIR = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/pca")
_EPS = 1e-8


def _median3(ch):
    a, b, c = ch[0], ch[1], ch[2]
    return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))


def _disp(Hk):
    k = Hk.shape[0]
    if k < 2:
        return 0.0
    iu = np.triu_indices(k, 1)
    D = Hk[iu[0]] - Hk[iu[1]]
    return float(np.sqrt(np.einsum("ij,ij->i", D, D) / D.shape[1]).mean())


def _z(x):
    x = np.asarray(x, np.float64)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


class _F:
    """One field of one task: candidate matrix V (view or copy), task mean m, per-row c.m and |c - m|."""
    __slots__ = ("V", "m", "cm", "nc", "mm")


class _Task:
    __slots__ = ("rows", "RS", "HD", "f")


class VisionZSum(api.Method):
    tier = "T0"
    family = "f4_vision"

    def __init__(self, lib="big", keys="raw", center="tm", fields="both", st=0.0, synth="top1"):
        if lib not in ("current", "big") or keys not in ("raw", "pca128") or center not in ("tm", "none") \
                or fields not in ("both", "v1") or synth not in ("top1", "med3"):
            raise ValueError(f"bad kwargs lib={lib} keys={keys} center={center} fields={fields} synth={synth}")
        if keys == "pca128" and lib != "big":
            raise ValueError("pca128 keys exist for the big library only")
        self.lib, self.keys, self.center, self.fields, self.st, self.synth = lib, keys, center, fields, float(st), synth
        self.tier = "T1" if keys == "pca128" else "T0"
        stn = f"_st{self.st:g}".replace(".", "p") if self.st else ""
        self.name = f"M9c_vzsum_{lib}_{keys}_{center}_{fields}{stn}_{synth}"

    def fit(self, lib, ctx):
        self.model = ctx.model
        self.libname = "current" if self.lib == "current" else BIG[ctx.model]
        L = lib if self.libname == "current" else ctx.open_library(self.libname)
        self.act = L.action
        sig = np.asarray(ctx.action_sigma, np.float32)
        heads = (np.asarray(L.action[:, :dims.EXEC_STEPS, dims.ACT_VALID], np.float32) / sig).reshape(L.L, NH)
        rs = np.ascontiguousarray(dims.valid_state(np.asarray(L.rs, np.float32), ctx.model))
        fl = ("v0", "v1") if self.fields == "both" else ("v1",)
        src = {}
        if self.keys == "pca128":
            self.pca = {}
            for f in fl:
                d = PCA_DIR / ctx.lib_key / self.libname / f
                meta = json.loads((d / "meta.json").read_text())
                if meta["n"] != L.L:
                    raise api.ContractError(f"PCA cache {d}: {meta['n']} rows != library {L.L}")
                self.pca[f] = (np.load(d / "mean.npy").astype(np.float32),
                               np.ascontiguousarray(np.load(d / "basis.npy"), np.float32))
                src[f] = np.load(d / "proj.npy").astype(np.float32)
        else:
            src = {f: getattr(L, f"key_{f}") for f in fl}
        self.fl = fl
        self.tasks = {}
        with self.prof.section("fit_index"):
            for t in L.tasks():
                r = np.asarray(L.rows_of_task(t), np.int64)
                T = _Task()
                T.rows, T.RS, T.HD = r, np.ascontiguousarray(rs[r]), np.ascontiguousarray(heads[r])
                T.f = {}
                contig = r.size > 0 and np.all(np.diff(r) == 1)
                for f in fl:
                    F = _F()
                    F.V = src[f][r[0]:r[-1] + 1] if contig else np.ascontiguousarray(src[f][r], np.float32)
                    # task mean (float64 accumulate, chunked) and per-row statistics
                    m = np.zeros(F.V.shape[1], np.float64)
                    cm = np.empty(r.size, np.float64)
                    nn = np.empty(r.size, np.float64)
                    for lo in range(0, r.size, 1024):
                        m += np.asarray(F.V[lo:lo + 1024], np.float64).sum(0)
                    m /= max(r.size, 1)
                    if self.center == "none":
                        m[:] = 0.0
                    F.m = m.astype(np.float32)
                    F.mm = float(np.dot(F.m.astype(np.float64), F.m.astype(np.float64)))
                    for lo in range(0, r.size, 1024):
                        X = np.asarray(F.V[lo:lo + 1024], np.float32)
                        cm[lo:lo + X.shape[0]] = X.astype(np.float64) @ F.m.astype(np.float64)
                        nn[lo:lo + X.shape[0]] = np.einsum("ij,ij->i", X, X)
                    F.cm = cm
                    F.nc = np.sqrt(np.maximum(nn - 2 * cm + F.mm, 0.0))
                    T.f[f] = F
                self.tasks[int(t)] = T
        # s_a: library self-queries (40 per task, other episodes), dispersion of the top-3 under the same score
        ep = np.asarray(L.episode, np.int64)
        rng = np.random.default_rng(0)
        disps = []
        with self.prof.section("fit_scales"):
            for t, T in self.tasks.items():
                sel = rng.choice(T.rows.size, size=min(40, T.rows.size), replace=False)
                for i in sel:
                    S = self._score(T, {f: np.asarray(T.f[f].V[i], np.float32) for f in fl}, T.RS[i],
                                    projected=True)
                    S[ep[T.rows] == ep[T.rows[i]]] = -np.inf
                    o = np.argsort(-S, kind="stable")[:3]
                    disps.append(_disp(T.HD[o]))
        self.s_a = float(np.median(disps)) + 1e-9
        try:
            (ctx.scratch / f"{self.name}_scales.json").write_text(json.dumps({"s_a": self.s_a, "L": int(L.L)}))
        except Exception:
            pass

    def _cos(self, F, q):
        """centred cosine of query vector q (already in the stored space) against all task candidates of F."""
        qc = F.V @ q                                   # [C]  (zero-copy view GEMV for the big library)
        qm = float(np.dot(q.astype(np.float64), F.m.astype(np.float64)))
        qq = float(np.dot(q.astype(np.float64), q.astype(np.float64)))
        num = qc.astype(np.float64) - qm - F.cm + F.mm
        nq = np.sqrt(max(qq - 2 * qm + F.mm, 0.0))
        return num / np.maximum(nq * F.nc, _EPS)

    def _score(self, T, qv, rs, projected=False):
        S = np.zeros(T.rows.size, np.float64)
        for f in self.fl:
            q = qv[f]
            if self.keys == "pca128" and not projected:
                mu, B = self.pca[f]
                q = (np.asarray(q, np.float32) - mu) @ B
            with self.prof.section(f"sim_{f}"):
                S += _z(self._cos(T.f[f], np.asarray(q, np.float32)))
        if self.st:
            dv = T.RS - rs
            S += self.st * _z(-np.sqrt(np.einsum("ij,ij->i", dv, dv)))
        return S

    def bytes_per_entry(self):
        per = 128 * 4 if self.keys == "pca128" else dims.KEY_DIM * 4
        return float(len(self.fl) * per + (8 * 4 if self.st else 0))

    def query(self, q):
        T = self.tasks.get(int(q.task_id))
        if T is None or T.rows.size == 0:
            raise api.ContractError(f"task {q.task_id} has no rows in library {self.libname}")
        qv = {"v0": q.key_v0, "v1": q.key_v1}
        rs = dims.valid_state(np.asarray(q.rs, np.float32), self.model)
        S = self._score(T, qv, rs)
        with self.prof.section("select"):
            K = min(10, S.size)
            o = np.argpartition(-S, K - 1)[:K] if S.size > K else np.arange(S.size)
            o = o[np.lexsort((o, -S[o]))]
        disp = _disp(T.HD[o[:3]])
        action = None
        if self.synth == "med3" and o.size >= 3:
            ch = np.asarray(self.act[T.rows[o[:3]]], np.float32)
            action = ch[0].copy()
            action[:, :6] = _median3(ch[:, :, :6])
            g = np.sign(ch[:, :, dims.GRIPPER_DIM]).sum(0)
            action[:, dims.GRIPPER_DIM] = np.where(g >= 0, 1.0, -1.0)
        nterm = len(self.fl) + (1 if self.st else 0)
        conf = float(S[o[0]]) / nterm - disp / self.s_a
        extras = {"s0": float(S[o[0]]), "disp": disp, "n_cand": int(S.size)}
        return api.Result(topk=T.rows[o], scores=S[o], confidence=conf, action=action, library=self.libname,
                          extras=extras)
