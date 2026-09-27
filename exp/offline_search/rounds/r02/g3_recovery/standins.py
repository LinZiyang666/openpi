"""Stand-in BASE selectors for developing / smoking the G3 wrappers until G1 (AWM) and G2 (PCA cosine) land.
They are the r01 methods with the G3 base hook (README.md "base contract") added in a subclass -- the r01 files are
not modified.

VZSKernel   r01 f4 M9c VisionZSum (two-camera task-centred cosine z-sum + st * z(-|rs - rs_c|) over ALL task
            candidates; current library: raw 32768-d keys, big library: the r01 big-library PCA-128 codes) with a
            P1-style synthesis: kernel mean of the top-k (w = exp(-(S_0 - S_i)/T)), optional step-0 alignment (at
            step 0 only library rows with step 0 are candidates). Confidence: VisionZSum's
            S_top1 / n_terms - disp3 / s_a over the served set. aux = the per-camera task-centred cosines.
            ~ G2's P1 (PCA-32 == raw within .008, R1) -- a stand-in, not a proposal.
B0KernelHook r01 f3 M4 B0TopkConsensus (B0's fused score, top-k synthesis mean | kernel, current library) with the
            hook (aux = B0's per-camera cosines). Its query() is untouched, so Passthrough over it checks that the
            wrapper's own top-k + synthesis reproduce an independent implementation.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from exp.offline_search.harness import api, dims

try:
    from . import g3_core as core
except ImportError:
    import g3_core as core

_VZS = core.load_class("exp/offline_search/rounds/r01/f4_vision/m9_vision_zsum.py:VisionZSum")
_B0C = core.load_class("exp/offline_search/rounds/r01/f3_b0plus/method.py:B0TopkConsensus")
_m9 = sys.modules[_VZS.__module__]
_z = _m9._z
_disp = _m9._disp


def _tag(x) -> str:
    return f"{x:g}".replace(".", "p")


class VZSKernel(_VZS):
    family = "g3_recovery"

    def __init__(self, lib="current", st=1.0, k=8, T=0.5, align0=True):
        keys = "raw" if lib == "current" else "pca128"
        super().__init__(lib=lib, keys=keys, center="tm", fields="both", st=st, synth="top1")
        self.synth_k, self.synth_T, self.align0 = int(k), float(T), bool(align0)
        self.os_uses_prev = False                 # os_score_all ignores prev_hit / prev_a_exec / hist_*
        self.name = (f"G3sb_vzs_{lib}_{keys}_st{_tag(self.st)}_k{self.synth_k}_T{_tag(self.synth_T)}"
                     f"{'_al0' if self.align0 else ''}")

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        self.os_library = self.libname
        self.os_fit_library = self.libname        # task means / s_a (raw) or PCA basis (pca128) come from it
        L = lib if self.libname == "current" else ctx.open_library(self.libname)
        lstep = np.asarray(L.step, np.int64)
        self.step0 = {t: np.flatnonzero(lstep[T.rows] == 0) for t, T in self.tasks.items()}
        self.nterm = len(self.fl) + (1 if self.st else 0)

    def _score_parts(self, T, q):
        S = np.zeros(T.rows.size, np.float64)
        cos = {}
        for f in self.fl:
            x = np.asarray(getattr(q, f"key_{f}"), np.float32)
            if self.keys == "pca128":
                mu, B = self.pca[f]
                x = (x - mu) @ B
            c = self._cos(T.f[f], np.asarray(x, np.float32))
            cos[f] = c
            S += _z(c)
        if self.st:
            rs = dims.valid_state(np.asarray(q.rs, np.float32), self.model)
            dv = T.RS - rs
            S += self.st * _z(-np.sqrt(np.einsum("ij,ij->i", dv, dv)))
        return S, cos

    def os_score_all(self, q):
        t = int(q.task_id)
        T = self.tasks.get(t)
        if T is None or T.rows.size == 0:
            raise api.ContractError(f"task {t} has no rows in library {self.libname}")
        S, cos = self._score_parts(T, q)
        rows = T.rows
        if self.align0 and q.step == 0:
            m = self.step0[t]
            if m.size >= self.synth_k:
                rows, S, cos = rows[m], S[m], {f: c[m] for f, c in cos.items()}
        return rows, S, {"vis_v0": cos.get("v0"), "vis_v1": cos.get("v1")}

    def os_confidence(self, q, rows, S, aux, srows, w):
        """VisionZSum's confidence over the served set: S(top-1) / n_terms - disp(top-3 heads) / s_a."""
        T = self.tasks[int(q.task_id)]
        pos = np.searchsorted(T.rows, np.asarray(srows[:3], np.int64))
        return float(S[np.flatnonzero(rows == srows[0])[0]]) / self.nterm - _disp(T.HD[pos]) / self.s_a

    def query(self, q):
        rows, S, aux = self.os_score_all(q)
        sel = core.topk_pos(S, self.synth_k)
        w = core.kernel_weights(S[sel], self.synth_T)
        srows = rows[sel]
        action = core.synth(self.act, srows, w, math.isinf(self.synth_T))
        conf = self.os_confidence(q, rows, S, aux, srows, w)
        wn = w / w.sum()
        return api.Result(topk=srows, scores=S[sel], confidence=conf, action=action, library=self.libname,
                          extras={"s0": float(S[sel[0]]), "n_cand": float(rows.size),
                                  "k_eff": float(1.0 / np.sum(wn * wn))})


class B0KernelHook(_B0C):
    family = "g3_recovery"

    def __init__(self, k=5, synth="kernel"):
        super().__init__(k=k, synth=synth)
        self.name = f"G3sb_b0cons_k{self.k}_{synth}"
        self.need_libstats = synth == "kernel"
        self.os_uses_prev = False

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        self.os_library = "current"
        self.os_fit_library = "current"
        self.synth_k = self.k
        self.synth_T = float(self.stats["sd_S"]) if self.how == "kernel" else math.inf

    def os_score_all(self, q):
        rows, f, parts = self.score_all(q)
        return rows, np.asarray(f, np.float64), {"vis_v0": parts[0], "vis_v1": parts[1]}

    def os_confidence(self, q, rows, S, aux, srows, w):
        return float(S.max())
