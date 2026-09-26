"""F3 (B0-plus) shared core: B0's ranking reused verbatim, new selection / synthesis / confidence on top.

Everything here keeps B0's scoring (harness.baselines.FusedKNN.score_all: per-field cosine on the pooled keys v0/v1,
robot_state -L2, zscore+tanh with the trace_dual yaml mu/sigma, weighted sum in float32) and only changes what happens
after the fused score vector is known:

  * selection   B0 order (stable argsort of -fused, first 10), optionally re-ordered by action continuity (M5);
  * synthesis   top1 | mean | kernel | med over the first k of the order (full (H, 32) chunks; only [:5, :7] scored);
  * confidence  B0 fused score (M4/M5/M8) or the consistency head of M6.

Valid dims only (harness/dims.py): continuity / dispersion / gripper statistics use actions[:5, :7] (sigma-scaled);
the previous chunk's unexecuted tail is prev_a_exec[5:10, :7]. Regime flag = q.prev_hit (None at step 0).

Library-fitted scales (LOEO: every current-library row is a query against the rows of the same task from the OTHER
episodes, ranked by the exact B0 formula; its "previous chunk" is the library row prev(r), i.e. a teacher chunk):
  mu_S, sd_S   mean / std of the B0 top-1 fused score           (M4 kernel temperature = sd_S; M6 g = (S - mu_S)/sd_S)
  s_c[base]    median continuity of the base selector's action    (M6 cont term, rows with prev(r) >= 0)
  s_a[base]    median dispersion of the base selector's set       (M6 dispersion term)
B-P1's own definitions (nearest other-episode head continuity / 3-nearest-head dispersion) are computed too and only
recorded in ctx.scratch/f3_libstats.json for reference.
"""
from __future__ import annotations

import json
import math

import numpy as np

from exp.offline_search.harness import api, dims
from exp.offline_search.harness.baselines import _EPS, FusedKNN, _zt

NSHORT = api.TOPK_SAVE          # B0 shortlist length (10)
BIG_LIB = {"pi05": "bpool_cs", "groot": "bpool_all"}
REGIME_STEP0, REGIME_MISS, REGIME_HIT, REGIME_UNK = 0.0, 1.0, 2.0, 3.0
NAN = float("nan")


# ------------------------------------------------------------------------------------------ small helpers
def regime_of(q) -> float:
    """0 step 0 | 1 after a MISS (prev chunk from the policy: fresh tail) | 2 after a HIT (library tail) | 3 unknown."""
    if q.step == 0:
        return REGIME_STEP0
    h = q.prev_hit
    return REGIME_UNK if h is None else (REGIME_HIT if h else REGIME_MISS)


def rms_rows(X, y):
    """RMS over the flattened (5,7) block of every row of X (k,5,7) against y (5,7) -> (k,)."""
    d = X - y
    return np.sqrt(np.mean(d * d, axis=(1, 2)))


_IU: dict = {}


def dispersion(X):
    """Mean pairwise RMS (sigma units) over the distinct pairs of the k heads X (k,5,7); NaN for k < 2."""
    k = X.shape[0]
    if k < 2:
        return NAN
    F = X.reshape(k, -1)
    iu = _IU.get(k)
    if iu is None:
        iu = _IU[k] = np.triu_indices(k, 1)
    D = F[iu[0]] - F[iu[1]]
    return float(np.sqrt(np.einsum("ij,ij->i", D, D) / F.shape[1]).mean())


def gripper_sign(x):
    """B's convention: sign with 0 -> +1."""
    s = np.sign(x)
    return np.where(s == 0, 1.0, s)


def synthesize(chunks, how, w=None):
    """chunks (k,H,32) float32 library actions (first = best) -> (H,32) float32.
    mean: plain mean of the full chunks; kernel: weighted mean (w sums to 1); med: per-step median of dims 0..5,
    dim 6 = majority gripper sign (B's rule: sum of signs >= 0 -> +1), dims 7.. from the first chunk."""
    c = np.asarray(chunks, np.float64)
    if how == "top1" or c.shape[0] == 1:
        return np.asarray(chunks[0], np.float32).copy()
    if how == "mean":
        a = c.mean(axis=0)
    elif how == "kernel":
        a = np.tensordot(np.asarray(w, np.float64), c, axes=1)
    elif how == "med":
        a = c[0].copy()
        s = np.sort(c[:, :, :dims.GRIPPER_DIM], axis=0)          # == np.median(axis=0), without its overhead
        k = s.shape[0]
        a[:, :dims.GRIPPER_DIM] = s[k // 2] if k % 2 else (s[k // 2 - 1] + s[k // 2]) / 2.0
        g = np.sign(c[:, :, dims.GRIPPER_DIM]).sum(axis=0)
        a[:, dims.GRIPPER_DIM] = np.where(g >= 0, 1.0, -1.0)
    else:
        raise ValueError(f"unknown synthesis {how!r}")
    return a.astype(np.float32)


def borda_order(cont10, lam):
    """Positions 0..n-1 of a B0-ordered shortlist re-ordered by rank_B0 + lam * rank_cont (lam = inf: pure continuity
    order). Ties -> better B0 rank (stable sort over the B0-ordered arrays)."""
    n = cont10.shape[0]
    rc = np.empty(n, np.float64)
    rc[np.argsort(cont10, kind="stable")] = np.arange(n)
    key = rc if math.isinf(lam) else np.arange(n, dtype=np.float64) + lam * rc
    return np.argsort(key, kind="stable"), key


def lam_tag(lam) -> str:
    return "inf" if math.isinf(float(lam)) else f"{float(lam):g}".replace(".", "p")


def parse_lam(lam) -> float:
    if isinstance(lam, str):
        return math.inf if lam.lower() in ("inf", "infinity", "∞") else float(lam)
    return float(lam)


# ----------------------------------------------------------------------------------------------- base class
class _Block:
    __slots__ = ("rows", "V0", "n0", "V1", "n1", "RS")


class B0Plus(FusedKNN):
    """FusedKNN (B0's exact scoring) + the F3 selection / synthesis machinery. Subclasses set the policy."""

    tier = "T0"
    family = "f3_b0plus"
    need_libstats = False

    def __init__(self, name):
        super().__init__("current", name, extras=False)

    # -- fit --------------------------------------------------------------------------------------------
    def fit(self, lib, ctx):
        super().fit(lib, ctx)                           # self.w / mu / sg / blocks / rs_dim on library current
        self._fit_payload(lib, ctx, "current")
        if self.need_libstats:
            with ctx.prof.section("libstats"):
                self.stats = self._libstats(lib, ctx)

    def _fit_payload(self, lib, ctx, name):
        """Arrays of the library the Result refers to: actions (memmap), scaled heads, episode / step."""
        self.lib_name = name
        self.sigma = np.asarray(ctx.action_sigma, np.float64)
        self.act = np.asarray(lib.action)                             # (L,H,32) view of the memmap (no copy)
        self.G5 = gripper_sign(np.asarray(lib.action[:, :dims.EXEC_STEPS, dims.GRIPPER_DIM], np.float64))  # (L,5)
        self.HD = np.ascontiguousarray(np.asarray(dims.valid_action(lib.action), np.float64) / self.sigma)
        self.lep = np.asarray(lib.episode, np.int64)
        self.lstep = np.asarray(lib.step, np.int64)

    # -- per query ---------------------------------------------------------------------------------------
    def _prev(self, q):
        """(scaled tail (5,7) or None, last executed gripper sign or None)."""
        pa = q.prev_a_exec
        if pa is None:
            return None, None
        pa = np.asarray(pa, np.float64)
        tail = pa[dims.EXEC_STEPS:2 * dims.EXEC_STEPS, dims.ACT_VALID] / self.sigma
        return tail, float(gripper_sign(pa[dims.EXEC_STEPS - 1, dims.GRIPPER_DIM]))

    def run_base(self, q, k, how, lam=None, kernel_T=None):
        """B0 score -> B0 shortlist (10) -> optional continuity re-order (lam, only after a MISS at step >= 1) ->
        synthesis over the first k. Returns a dict with everything the policies / extras need."""
        rows, f, parts = self.score_all(q)
        with self.prof.section("rank"):
            o = np.argsort(-f, kind="stable")[:NSHORT]           # B0 order (positions in the task block)
        regime = regime_of(q)
        tail, gprev = self._prev(q)
        lrows = rows[o]                                          # library rows of the shortlist, B0 order
        H10 = self.HD[lrows]                                     # (n,5,7) scaled heads
        with self.prof.section("cont"):
            cont10 = rms_rows(H10, tail) if tail is not None else None
        perm = np.arange(o.size)
        key = None
        if lam is not None and regime == REGIME_MISS and cont10 is not None:
            with self.prof.section("rerank"):
                perm, key = borda_order(cont10, lam)
        order = o[perm]                                          # final order (positions in the task block)
        kk = min(k, order.size)
        sel = rows[order[:kk]]
        w = None
        with self.prof.section("synth"):
            if how == "top1" or kk == 1:
                action = None
            else:
                chunks = np.asarray(self.act[sel], np.float32)
                if how == "kernel":
                    fs = f[order[:kk]].astype(np.float64)
                    w = np.exp(-(fs[0] - fs) / kernel_T)
                    w /= w.sum()
                action = synthesize(chunks, how, w)
        return dict(rows=rows, f=f, parts=parts, o=o, perm=perm, key=key, order=order, sel=sel, kk=kk, action=action,
                    w=w, regime=regime, tail=tail, gprev=gprev, H10=H10, cont10=cont10)

    def base_extras(self, r):
        """Per-decision diagnostics shared by every F3 method."""
        f, o, order, perm = r["f"], r["o"], r["order"], r["perm"]
        c0, c1, d, _, _, _ = r["parts"]
        j = order[0]                                             # pick (position in the task block)
        pick_row = int(r["rows"][j])
        tail = r["tail"]
        H10 = r["H10"]
        Hsel = H10[perm[:r["kk"]]]
        if tail is not None:
            cont_b0 = float(r["cont10"][0])
            cont0 = float(r["cont10"][perm[0]])
            if r["action"] is None:
                cont_act = cont0
            else:
                ah = np.asarray(dims.valid_action(r["action"]), np.float64) / self.sigma
                cont_act = float(np.sqrt(np.mean((ah - tail) ** 2)))
            g10 = self.G5[r["rows"][o]]                                                        # (n,5)
            flip10 = float((g10 != r["gprev"]).any(axis=1).mean())
        else:
            cont_b0 = cont0 = cont_act = flip10 = NAN
        ex = {
            "regime": r["regime"],
            "b0_top1": float(f[o[0]]), "b0_pick": float(f[j]), "rank_b0": float(perm[0]),
            "margin": float(f[o[0]] - f[o[1]]) if o.size > 1 else NAN,
            "cos_v0": float(c0[j]), "cos_v1": float(c1[j]), "dist_rs": float(d[j]),
            "cont0": cont0, "cont_b0": cont_b0, "cont_act": cont_act,
            "disp_set": dispersion(Hsel) if r["kk"] > 1 else NAN,
            "disp5_b0": dispersion(H10[:5]),
            "flip10": flip10,
            "pick_ep": float(self.lep[pick_row]), "pick_step": float(self.lstep[pick_row]),
            "k": float(r["kk"]),
        }
        if r["w"] is not None:
            ex["w_top1"] = float(r["w"][0])
            ex["k_eff"] = float(1.0 / np.sum(r["w"] ** 2))
        return ex

    def result(self, r, conf, extras):
        order = r["order"]
        if r["key"] is None:
            scores = r["f"][order]
        else:
            scores = -r["key"][r["perm"]]                        # the method's own (re-ordering) scale
        return api.Result(topk=r["rows"][order], scores=np.asarray(scores, np.float64), confidence=float(conf),
                          action=r["action"], library=self.lib_name, extras=extras)

    # -- library-fitted scales (LOEO over the current library) --------------------------------------------
    def _libstats(self, lib, ctx):
        """See module doc. Uses the task blocks of library current built by FusedKNN.fit (float32 copies)."""
        prev = np.asarray(lib.prev, np.int64)
        TL = np.asarray(lib.action[:, dims.EXEC_STEPS:2 * dims.EXEC_STEPS, dims.ACT_VALID], np.float64) / self.sigma
        top1, conts, disps = [], {"b0": [], "m4k5med": [], "m5l2k3": []}, {"b0": [], "m4k5med": [], "m5l2k3": []}
        bp1_c, bp1_a = [], []
        for t, b in self.blocks.items():
            rows = b.rows
            C = rows.size
            ep = self.lep[rows]
            other = ep[:, None] != ep[None, :]
            if C < 4 or not other.any():
                continue
            c0 = (b.V0 @ b.V0.T) / np.maximum(b.n0[:, None] * b.n0[None, :], _EPS)
            c1 = (b.V1 @ b.V1.T) / np.maximum(b.n1[:, None] * b.n1[None, :], _EPS)
            r2 = np.einsum("ij,ij->i", b.RS, b.RS)
            dd = np.sqrt(np.maximum(r2[:, None] - 2.0 * (b.RS @ b.RS.T) + r2[None, :], 0.0)).astype(np.float32)
            F = np.zeros((C, C), np.float32)
            F += self.w[0] * _zt(c0, self.mu[0], self.sg[0])
            F += self.w[1] * _zt(c1, self.mu[1], self.sg[1])
            F += self.w[2] * _zt(-dd, self.mu[2], self.sg[2])
            F[~other] = -np.inf
            O = np.argsort(-F, axis=1, kind="stable")[:, :NSHORT]
            HDt = self.HD[rows]                                   # (C,5,7)
            # B-P1 reference scales (35-d, other episodes of the task)
            Ff = HDt.reshape(C, -1)
            a2 = np.einsum("ij,ij->i", Ff, Ff)
            DH = np.sqrt(np.maximum((a2[:, None] - 2.0 * Ff @ Ff.T + a2[None, :]) / Ff.shape[1], 0.0))
            DH[~other] = np.inf
            for i in range(C):
                oi = O[i][np.isfinite(F[i, O[i]])]
                if oi.size < 5:
                    continue
                top1.append(float(F[i, oi[0]]))
                H10 = HDt[oi]
                disps["b0"].append(dispersion(H10[:5]))
                disps["m4k5med"].append(disps["b0"][-1])
                p = prev[rows[i]]
                nn3 = np.argsort(DH[i], kind="stable")[:3]
                if np.all(np.isfinite(DH[i, nn3])):
                    bp1_a.append(dispersion(HDt[nn3]))
                if p < 0:
                    continue
                tail = TL[p]
                cont10 = rms_rows(H10, tail)
                conts["b0"].append(float(cont10[0]))
                med5 = synthesize(np.asarray(self.act[rows[oi[:5]]], np.float32), "med")
                conts["m4k5med"].append(float(np.sqrt(np.mean(
                    (np.asarray(dims.valid_action(med5), np.float64) / self.sigma - tail) ** 2))))
                perm, _ = borda_order(cont10, 2.0)
                med3 = synthesize(np.asarray(self.act[rows[oi[perm[:3]]]], np.float32), "med")
                conts["m5l2k3"].append(float(np.sqrt(np.mean(
                    (np.asarray(dims.valid_action(med3), np.float64) / self.sigma - tail) ** 2))))
                disps["m5l2k3"].append(dispersion(H10[perm[:3]]))
                # B-P1 s_c: library tail -> nearest other-episode head
                dc = np.sqrt(np.mean((HDt - tail) ** 2, axis=(1, 2)))
                dc[~other[i]] = np.inf
                bp1_c.append(float(dc.min()))
        top1 = np.asarray(top1, np.float64)
        st = {"n_top1": int(top1.size), "mu_S": float(top1.mean()), "sd_S": float(top1.std()),
              "s_c": {k: float(np.median(v)) for k, v in conts.items()},
              "s_a": {k: float(np.median(v)) for k, v in disps.items()},
              "n_cont": int(len(conts["b0"])),
              "bp1_s_c": float(np.median(bp1_c)), "bp1_s_a": float(np.median(bp1_a)),
              "q_top1": [float(x) for x in np.percentile(top1, [5, 25, 50, 75, 95])]}
        try:
            (ctx.scratch / "f3_libstats.json").write_text(json.dumps(st, indent=1))
        except Exception:
            pass
        return st


# ------------------------------------------------------------------------------------------ big library blocks
def row_norms_chunked(V, chunk=2048):
    """np.linalg.norm(V, axis=1) in float32, row chunks (bit-identical per row, no full-size temporary)."""
    out = np.empty(V.shape[0], np.float32)
    for lo in range(0, V.shape[0], chunk):
        out[lo:lo + chunk] = np.linalg.norm(V[lo:lo + chunk], axis=1).astype(np.float32)
    return out


def big_blocks(B):
    """Per-task B0 blocks over a big library WITHOUT copying the 32768-d keys: the big libraries are stored task-sorted,
    so each task is a contiguous row range and the block is a view into the (tmpfs-backed, shared) memmap."""
    K0 = np.asarray(B.key_v0)
    K1 = np.asarray(B.key_v1)
    blocks = {}
    for t in B.tasks():
        rows = np.asarray(B.rows_of_task(t), np.int64)
        b = _Block()
        b.rows = rows
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        if hi - lo == rows.size and np.array_equal(rows, np.arange(lo, hi)):
            b.V0 = K0[lo:hi]
            b.V1 = K1[lo:hi]
        else:                                             # not task-contiguous: fall back to a copy (as B0)
            b.V0 = np.ascontiguousarray(K0[rows], dtype=np.float32)
            b.V1 = np.ascontiguousarray(K1[rows], dtype=np.float32)
        b.n0 = row_norms_chunked(b.V0)
        b.n1 = row_norms_chunked(b.V1)
        b.RS = np.ascontiguousarray(B.rs[rows], dtype=np.float32)
        blocks[t] = b
    return blocks
