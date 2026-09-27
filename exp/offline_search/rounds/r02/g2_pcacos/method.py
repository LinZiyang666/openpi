"""R2 family G2 (cosine school): V4 `vis_pca_allcand_kmean` (ideation B-P1) and V5 fresh fusion (B-P4 + A's additive
fusion). Exact algorithms / measured numbers: rounds/r02/NOTES_ideation_B.md (F1, F2, F4, F7), diag_B/diag.py, diag3.py.

Representation (per camera f in {v0, v1}): PCA-k code of the pooled 32768-d key, x_f = key_f @ B_f - mu_f @ B_f,
B_f / mu_f fitted on the `fit` library (precompute.py; fit "big" = r01 F4 bases of the 10x library, fit "cur" = exact
PCA of the current library). Library codes are precomputed in the same basis (derived/r02/g2_pcacos/proj).

V4 `PcaCosV4` -- every regime (step 0, after a HIT, after a MISS):
  per task t: Z_f = unit(P_f[rows_t] - m_{f,t})  (m_{f,t} = task mean of the codes), RS = rs[:8] of the candidates
  query:      c_f = Z_f @ unit(x_f - m_{f,t})      (task-centred cosine against ALL task candidates)
              S   = z(c_v0) + z(c_v1) + st * z(-||rs_q[:8] - RS||)      (z = standardize over the task's candidates)
              step 0 (align0): candidates with library step > 0 get S = -1e9 (z computed over all candidates first,
              as diag.py) -> only library step-0 rows can be picked
              top-kk by S (kk = 8), kernel weights w_i = exp((S_i - S_max) / T) (T = 0.5) | plain mean
              action = sum_i w_i * action[row_i]  (full (H, 32) library chunks; only [:5, :7] is scored)
  confidence  (t_dnn + t_disp + t_over + t_cos) / sd_sum, t_* = library-fitted z of
              -dnn (min rs distance over the eligible candidates), -disp (mean pairwise RMS_sigma of the synthesized
              set's heads [:5, :7]), -overtime (step / median library ep_len of the task, rows-weighted as diag.py),
              +(max_c c_v0 + max_c c_v1). mu / sd of each feature and sd_sum from library LOEO pseudo-queries (every
              task: <= 256 evenly spaced rows scored against the OTHER episodes of the task with the same rule).
  No continuity term anywhere (after a HIT the previous chunk is a library chunk).
V5 `PcaCosV5Fresh` -- V4 at step 0 and after a HIT (prev_hit True or unknown); after a MISS (prev_hit False):
              F = -c / s_c + lam * (z(c_v0) + z(c_v1)) / 2 + alpha * z(-d)   over ALL task candidates
              c = RMS_sigma(head[:5, :7] - prev_a_exec[5:10, :7]); s_c = library median 1-NN continuity (row tail
              [5:10] -> nearest other-episode head of the task); alpha = .25; kernel-8 T = .5 on F (diag_v5: kernel-8
              T .5 beats mean-5 by .003-.009 on 10x and .015-.024 on current, both lam).
              confidence (t_c + t_disp + t_vis) / sd_sum: z of -c0/s_c (top-1), -disp, +vis0 ((z_v0 + z_v1)/2 of the
              top-1); library LOEO scales (pseudo-query tail = the previous library row's tail).

Effect decomposition (protocol §9): lib in {"current", "big"} (big = bpool_cs pi0.5 / bpool_all GR00T) x fit in
{"same", "big"} (PCA basis source; "same" = the candidate library). lib=current fit=big is labelled "borrowed" (tag
LcurFbig). Confidence scales / s_c / thresholds are always fitted on the candidate library itself.

Extras (per decision): regime (0 step0 / 1 after MISS / 2 after HIT / 3 unknown), branch (0 V4 / 1 V5 fresh), ncand,
s0, s5, w_top1, k_eff, cos0_0, cos1_0 (top-1's cosines), cosmax0, cosmax1, d0, dnn, d5, disp, overtime, lag5 (step -
mean library step of the top-5), libprog5, libmotion5, terminal0 (top-1 has no next row), pick_ep, pick_step, motion
(||rs_t - rs_t-1||), stuck_n (trailing decisions with motion < library 10th pct), vself (min camera cosine of
consecutive query codes), win_d (3-decision state window 1-NN distance), t_dnn, t_disp, t_over, t_cos, and on the
fresh branch c0, c_act, vis0, f0, t_c, t_vis. No NaN (the closed-loop selftest compares extras with ==): sentinels
motion -1 (step 0), vself -2 (no previous decision), win_d -1 (step < 2), disp 0 (single candidate).
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from exp.offline_search.harness import api, dims

DERIVED = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r02/g2_pcacos")
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
ST_PER_MODEL = {"pi05": 0.5, "groot": 3.0}      # ideation B F1 (diag3): best state weight per model
RSV = 8                                          # valid robot_state dims (dims.RS_VALID)
NH = dims.EXEC_STEPS * dims.ACT_DIMS             # 35
NSAVE = api.TOPK_SAVE                            # rows returned (10)
NEG = -1e9
NQ_LOEO = 256                                    # pseudo-queries per task for the confidence scales
EPS = 1e-8
NAN = float("nan")
REGIME_STEP0, REGIME_MISS, REGIME_HIT, REGIME_UNK = 0, 1, 2, 3


def _tag(x) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def _z(x):
    """standardize a 1-D float64 vector (std ddof 0; std < 1e-12 -> 1)."""
    m = x.mean()
    s = x.std()
    return (x - m) / (s if s > 1e-12 else 1.0)


def _zmask(X, M):
    """row-wise standardize X [n, C] over the columns where M [n, C] is True (others -> 0, never used)."""
    cnt = np.maximum(M.sum(1, keepdims=True), 1)
    m = np.where(M, X, 0.0).sum(1, keepdims=True) / cnt
    v = np.where(M, (X - m) ** 2, 0.0).sum(1, keepdims=True) / cnt
    s = np.sqrt(v)
    s[s < 1e-12] = 1.0
    return np.where(M, (X - m) / s, 0.0)


def _unit_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


_IU: dict = {}


def _disp(H):
    """mean pairwise RMS over the rows of H [k, NH] (sigma units); NaN for k < 2."""
    k = H.shape[0]
    if k < 2:
        return NAN
    iu = _IU.get(k)
    if iu is None:
        iu = _IU[k] = np.triu_indices(k, 1)
    D = H[iu[0]] - H[iu[1]]
    return float(np.sqrt(np.einsum("ij,ij->i", D, D) / H.shape[1]).mean())


def _disp_batch(H):
    """[n, k, NH] -> [n] mean pairwise RMS."""
    k = H.shape[1]
    iu = np.triu_indices(k, 1)
    D = H[:, iu[0]] - H[:, iu[1]]
    return np.sqrt((D * D).mean(2)).mean(1)


def _topk(S, K):
    """first K positions of S in descending order (ties -> lower position), deterministic."""
    C = S.shape[0]
    K = min(K, C)
    idx = np.argpartition(-S, K - 1)[:K] if C > K else np.arange(C)
    return idx[np.lexsort((idx, -S[idx]))]


def _pair_l2(A, B):
    a2 = np.einsum("ij,ij->i", A, A)[:, None]
    b2 = np.einsum("ij,ij->i", B, B)[None, :]
    return np.sqrt(np.maximum(a2 - 2.0 * (A @ B.T) + b2, 0.0))


def _pair_rms(A, B):
    return _pair_l2(A, B) / math.sqrt(A.shape[1])


def regime_of(q) -> int:
    if q.step == 0:
        return REGIME_STEP0
    h = q.prev_hit
    return REGIME_UNK if h is None else (REGIME_HIT if h else REGIME_MISS)


class _Task:
    __slots__ = ("rows", "Z0", "Z1", "m0", "m1", "RS", "step", "s0mask", "has0", "HD", "h2", "ep", "nxt", "prog",
                 "motion", "W", "wmask", "C")


# ----------------------------------------------------------------------------------------------- artifacts
def load_basis(key: str, fitsrc: str, f: str, k: int):
    d = DERIVED / "pca" / key / f"fit_{fitsrc}" / f
    mu = np.load(d / "mean.npy").astype(np.float32)
    B = np.load(d / "basis.npy", mmap_mode="r")[:, :k]
    B = np.ascontiguousarray(B, np.float32)
    return mu, B


def load_codes(key: str, fitsrc: str, libname: str, f: str, k: int, lib, mu, B):
    """library codes [L, k] in the fit basis: the precomputed file, else computed (and written atomically)."""
    p = DERIVED / "proj" / key / f"fit_{fitsrc}" / f"{libname}_{f}.npy"
    if p.exists():
        P = np.load(p, mmap_mode="r")
        if P.shape[0] != lib.L:
            raise api.ContractError(f"{p}: {P.shape[0]} rows != library {libname} L={lib.L}")
        return np.ascontiguousarray(P[:, :k], np.float32)
    X = getattr(lib, f"key_{f}")
    out = np.empty((lib.L, B.shape[1]), np.float32)
    muB = mu @ B
    for lo in range(0, lib.L, 2048):
        out[lo:lo + 2048] = np.asarray(X[lo:lo + 2048], np.float32) @ B - muB
    return out


# ------------------------------------------------------------------------------------------------- V4
class PcaCosV4(api.Method):
    """V4 vis_pca_allcand_kmean (see module doc)."""

    tier = "T1"
    family = "g2_pcacos"

    def __init__(self, lib="big", fit="same", k=32, st=1.0, synth="kmean", kk=8, T=0.5, align0=True):
        if lib not in ("current", "big"):
            raise ValueError(f"lib must be current | big, got {lib!r}")
        if fit not in ("same", "big"):
            raise ValueError(f"fit must be same | big, got {fit!r}")
        if synth not in ("kmean", "mean"):
            raise ValueError(f"synth must be kmean | mean, got {synth!r}")
        if not (st == "pm" or isinstance(st, (int, float))):
            raise ValueError(f"st must be a number or 'pm' (per model), got {st!r}")
        self.lib_sel, self.fit_sel = lib, fit
        self.k, self.st_arg, self.synth, self.kk, self.T, self.align0 = int(k), st, synth, int(kk), float(T), bool(align0)
        self.name = self._name()
        self._prev_x = None                  # per-episode state (reset() clears it): previous query codes, for vself
        self._prev_step = -1

    def _name(self, prefix="V4pc"):
        L = "cur" if self.lib_sel == "current" else "big"
        F = "big" if (self.fit_sel == "big" or self.lib_sel == "big") else "cur"
        st = "pm" if self.st_arg == "pm" else _tag(self.st_arg)
        syn = f"km{self.kk}T{_tag(self.T)}" if self.synth == "kmean" else f"mean{self.kk}"
        return f"{prefix}_L{L}F{F}_p{self.k}_st{st}_{syn}" + ("" if self.align0 else "_noal")

    # ------------------------------------------------------------------------------------------ fit
    def fit(self, lib, ctx):
        self.model = ctx.model
        self.key = ctx.lib_key
        self.st = ST_PER_MODEL[ctx.model] if self.st_arg == "pm" else float(self.st_arg)
        self.libname = "current" if self.lib_sel == "current" else BIG[ctx.model]
        self.fitsrc = "big" if (self.fit_sel == "big" or self.lib_sel == "big") else "cur"
        L = lib if self.libname == "current" else ctx.open_library(self.libname)
        self.sig = np.asarray(ctx.action_sigma, np.float64)
        with self.prof.section("fit_basis"):
            self.Bt, self.muB, codes = {}, {}, {}
            for f in ("v0", "v1"):
                mu, B = load_basis(self.key, self.fitsrc, f, self.k)
                self.Bt[f] = np.ascontiguousarray(B.T)                 # [k, D] (query GEMV)
                self.muB[f] = (mu @ B).astype(np.float32)
                codes[f] = load_codes(self.key, self.fitsrc, self.libname, f, self.k, L, mu, B)
        self._act_path = str(pathlib.Path(ctx.root) / "library" / self.key / self.libname / "action.npy")
        self._act = np.load(self._act_path, mmap_mode="r")
        rs = np.ascontiguousarray(np.asarray(L.rs, np.float32)[:, :RSV])
        act7 = np.asarray(L.action[:, :2 * dims.EXEC_STEPS, dims.ACT_VALID], np.float64)
        HD = (act7[:, :dims.EXEC_STEPS] / self.sig).reshape(L.L, NH)
        TL = (act7[:, dims.EXEC_STEPS:2 * dims.EXEC_STEPS] / self.sig).reshape(L.L, NH)
        step = np.asarray(L.step, np.int64)
        ep = np.asarray(L.episode, np.int64)
        nxt = np.asarray(L.next, np.int64)
        prv = np.asarray(L.prev, np.int64)
        prog = np.asarray(L.progress, np.float64)
        eplen = np.asarray(L.ep_len, np.float64)
        motion = np.zeros(L.L, np.float64)
        ok = nxt >= 0
        motion[ok] = np.linalg.norm(rs[nxt[ok]] - rs[ok], axis=1)
        self.motion_thr = float(np.percentile(motion[ok], 10))
        has3 = prv >= 0
        has3[has3] = prv[prv[has3]] >= 0
        Wall = np.zeros((L.L, 3 * RSV), np.float32)
        Wall[has3] = np.concatenate([rs[has3], rs[prv[has3]], rs[prv[prv[has3]]]], 1)
        self.medlen, self.tasks = {}, {}
        with self.prof.section("fit_tasks"):
            for t in L.tasks():
                r = np.asarray(L.rows_of_task(t), np.int64)
                T = _Task()
                T.rows, T.C = r, r.size
                for f, (zn, mn) in (("v0", ("Z0", "m0")), ("v1", ("Z1", "m1"))):
                    Pc = codes[f][r]
                    m = Pc.mean(0).astype(np.float32)
                    setattr(T, mn, m)
                    setattr(T, zn, np.ascontiguousarray(_unit_rows(Pc - m), np.float32))
                T.RS = np.ascontiguousarray(rs[r])
                T.step = step[r]
                T.s0mask = T.step == 0
                T.has0 = bool(T.s0mask.any())
                T.HD = np.ascontiguousarray(HD[r], np.float32)
                T.h2 = np.einsum("ij,ij->i", T.HD, T.HD).astype(np.float64)
                T.ep, T.nxt, T.prog, T.motion = ep[r], nxt[r], prog[r], motion[r]
                T.wmask = has3[r]
                T.W = np.ascontiguousarray(Wall[r][T.wmask])
                self.tasks[int(t)] = T
                self.medlen[int(t)] = float(np.median(eplen[r]))
        with self.prof.section("fit_scales"):
            self._fit_scales(TL, prv, codes)
        info = {"name": self.name, "library": self.libname, "fit_basis": self.fitsrc, "k": self.k, "st": self.st,
                "L": int(L.L), "motion_thr": self.motion_thr, "scales": self.scales}
        try:
            (pathlib.Path(ctx.scratch) / f"{self.name}_fit.json").write_text(json.dumps(info, indent=1))
        except Exception:  # noqa: BLE001
            pass

    # library LOEO pseudo-queries -> feature means / sds of the confidence terms
    def _loeo_v4(self, T, qi, codes_q, rs_q, step_q, medlen):
        """V4 features of pseudo-queries qi (positions in T) against the other episodes of the task."""
        other = T.ep[qi][:, None] != T.ep[None, :]
        Cs = []
        S = np.zeros((qi.size, T.C))
        for Z, m, P in ((T.Z0, T.m0, codes_q[0]), (T.Z1, T.m1, codes_q[1])):
            Cm = (_unit_rows(P - m) @ Z.T).astype(np.float64)
            Cs.append(Cm)
            S += _zmask(Cm, other)
        D = _pair_l2(rs_q, T.RS).astype(np.float64)
        S += self.st * _zmask(-D, other)
        elig = other.copy()
        if self.align0:
            s0 = (step_q == 0)[:, None] & T.s0mask[None, :] & other
            use = (step_q == 0) & s0.any(1)
            elig[use] = s0[use]
        S = np.where(elig, S, -np.inf)
        o = np.argsort(-S, axis=1, kind="stable")[:, :self.kk]
        valid = np.isfinite(np.take_along_axis(S, o, 1))
        H = T.HD[o].astype(np.float64)                          # [n, kk, NH]
        disp = np.zeros(qi.size)
        if self.kk > 1:
            full = valid.all(1)
            disp[full] = _disp_batch(H[full])
            for i in np.nonzero(~full)[0]:                      # step-0 pseudo-queries with < kk eligible rows
                v = H[i][valid[i]]
                disp[i] = _disp(v) if v.shape[0] > 1 else 0.0
        dnn = np.where(elig, D, np.inf).min(1)
        cmax = sum(np.where(elig, Cm, -np.inf).max(1) for Cm in Cs)
        over = step_q / medlen
        return dict(dnn=dnn, disp=disp, over=over, cmax=cmax)

    def _fit_scales(self, TL, prv, codes):
        feats = {"dnn": [], "disp": [], "over": [], "cmax": []}
        cs = []
        for t, T in self.tasks.items():
            if T.C < 3 or np.unique(T.ep).size < 2:
                continue
            n = min(NQ_LOEO, T.C)
            qi = np.unique(np.linspace(0, T.C - 1, n).round().astype(np.int64))
            cq = [codes["v0"][T.rows[qi]], codes["v1"][T.rows[qi]]]
            r = self._loeo_v4(T, qi, cq, T.RS[qi], T.step[qi].astype(np.float64), self.medlen[t])
            for k_ in feats:
                feats[k_].append(r[k_])
            # continuity scale s_c: every row's own tail -> nearest other-episode head of the task
            Cc = _pair_rms(TL[T.rows].astype(np.float64), T.HD.astype(np.float64))
            Cc[T.ep[:, None] == T.ep[None, :]] = np.inf
            cs.append(Cc.min(1))
        self.s_c = float(np.median(np.concatenate(cs)[np.isfinite(np.concatenate(cs))])) + 1e-9
        sc = {}
        for k_, v in feats.items():
            x = np.concatenate(v).astype(np.float64)
            x = x[np.isfinite(x)]
            sc[k_] = (float(x.mean()), float(x.std()) if x.std() > 1e-12 else 1.0)
        # sd of the summed confidence over the pseudo-queries (one scale for every branch)
        tot = []
        for i in range(len(feats["dnn"])):
            tot.append(self._conf_terms_v4(feats["dnn"][i], feats["disp"][i], feats["over"][i], feats["cmax"][i],
                                           sc)[0])
        tot = np.concatenate(tot)
        sc["sum_sd"] = float(np.std(tot[np.isfinite(tot)])) or 1.0
        self.scales = {"v4": sc, "s_c": self.s_c}
        self._fit_scales_fresh(TL, prv, codes)

    def _fit_scales_fresh(self, TL, prv, codes):
        """hook for V5 (fresh-branch confidence scales)."""

    @staticmethod
    def _conf_terms_v4(dnn, disp, over, cmax, sc):
        t_dnn = -(dnn - sc["dnn"][0]) / sc["dnn"][1]
        t_disp = -(np.nan_to_num(disp, nan=sc["disp"][0]) - sc["disp"][0]) / sc["disp"][1]
        t_over = -(over - sc["over"][0]) / sc["over"][1]
        t_cos = (cmax - sc["cmax"][0]) / sc["cmax"][1]
        return t_dnn + t_disp + t_over + t_cos, (t_dnn, t_disp, t_over, t_cos)

    # --------------------------------------------------------------------------------------- query
    def reset(self, episode):
        self._prev_x = None
        self._prev_step = -1

    def bytes_per_entry(self):
        return float(2 * 4 * self.k + 4 * RSV + 4)          # two PCA-k f32 codes + rs[:8] f32 + int32 step

    def _project(self, q):
        with self.prof.section("project"):
            x0 = self.Bt["v0"] @ np.asarray(q.key_v0, np.float32) - self.muB["v0"]
            x1 = self.Bt["v1"] @ np.asarray(q.key_v1, np.float32) - self.muB["v1"]
        return x0, x1

    def _vision(self, T, x0, x1):
        with self.prof.section("cos"):
            u0 = x0 - T.m0
            u0 /= max(float(np.linalg.norm(u0)), EPS)
            u1 = x1 - T.m1
            u1 /= max(float(np.linalg.norm(u1)), EPS)
            c0 = (T.Z0 @ u0).astype(np.float64)
            c1 = (T.Z1 @ u1).astype(np.float64)
            z0, z1 = _z(c0), _z(c1)
        return c0, c1, z0, z1

    def _state(self, T, rs):
        dv = T.RS - rs
        return np.sqrt(np.einsum("ij,ij->i", dv, dv)).astype(np.float64)

    def _synth(self, T, S, o):
        kk = min(self.kk, o.size)
        sel = o[:kk]
        ok = S[sel] > NEG / 2                                   # step-0 alignment: never mix in masked rows
        if not ok.all():
            sel = sel[ok] if ok.any() else sel[:1]
        if self.synth == "kmean":
            w = np.exp((S[sel] - S[sel[0]]) / self.T)
        else:
            w = np.ones(kk)
        w /= w.sum()
        ch = np.asarray(self._act[T.rows[sel]], np.float64)
        action = np.tensordot(w, ch, axes=1).astype(np.float32)
        return sel, w, action

    def _history_extras(self, q, T, rs, x0, x1):
        step = int(q.step)
        ex = {}
        if step >= 1:
            h = np.asarray(q.hist_rs, np.float32)[:, :RSV]
            seq = np.vstack([h, rs[None, :]])
            mv = np.linalg.norm(np.diff(seq, axis=0), axis=1)
            below = mv < self.motion_thr
            nb = int(np.argmin(below[::-1])) if not below.all() else int(below.size)
            ex["motion"] = float(mv[-1])
            ex["stuck_n"] = float(nb)
        else:
            ex["motion"] = -1.0                                  # sentinel: no previous decision (step 0)
            ex["stuck_n"] = 0.0
        if self._prev_x is not None and self._prev_step == step - 1:
            p0, p1 = self._prev_x
            v0 = float(x0 @ p0 / max(np.linalg.norm(x0) * np.linalg.norm(p0), EPS))
            v1 = float(x1 @ p1 / max(np.linalg.norm(x1) * np.linalg.norm(p1), EPS))
            ex["vself"] = min(v0, v1)
        else:
            ex["vself"] = -2.0                                   # sentinel: no previous decision
        if step >= 2 and T.W.shape[0]:
            h = np.asarray(q.hist_rs[step - 2:step], np.float32)[:, :RSV]
            wq = np.concatenate([rs, h[1], h[0]])
            dw = T.W - wq
            ex["win_d"] = float(np.sqrt(np.einsum("ij,ij->i", dw, dw)).min())
        else:
            ex["win_d"] = -1.0                                   # sentinel: step < 2
        self._prev_x = (x0.copy(), x1.copy())
        self._prev_step = step
        return ex

    def _common_extras(self, q, T, o, sel, w, S, c0, c1, d, elig, regime, branch):
        j = o[0]
        top5 = o[:5]
        ex = {"regime": float(regime), "branch": float(branch), "ncand": float(elig.sum() if elig is not None else T.C),
              "s0": float(S[j]), "s5": float(S[top5].mean()), "w_top1": float(w[0]), "k_eff": float(1.0 / np.sum(w * w)),
              "cos0_0": float(c0[j]), "cos1_0": float(c1[j]), "d0": float(d[j]), "d5": float(d[top5].mean()),
              "disp": _disp(T.HD[sel].astype(np.float64)) if sel.size > 1 else 0.0,
              "overtime": float(q.step / self.medlen[int(q.task_id)]),
              "lag5": float(q.step - T.step[top5].mean()), "libprog5": float(T.prog[top5].mean()),
              "libmotion5": float(T.motion[top5].mean()), "terminal0": float(T.nxt[j] < 0),
              "pick_ep": float(T.ep[j]), "pick_step": float(T.step[j])}
        if elig is None:
            ex["dnn"] = float(d.min())
            ex["cosmax0"], ex["cosmax1"] = float(c0.max()), float(c1.max())
        else:
            ex["dnn"] = float(d[elig].min())
            ex["cosmax0"], ex["cosmax1"] = float(c0[elig].max()), float(c1[elig].max())
        return ex

    def _v4(self, q, T, x0, x1, rs, regime):
        c0, c1, z0, z1 = self._vision(T, x0, x1)
        with self.prof.section("score"):
            d = self._state(T, rs)
            S = z0 + z1 + self.st * _z(-d)
            elig = None
            if q.step == 0 and self.align0 and T.has0:
                elig = T.s0mask
                S = np.where(elig, S, NEG)
            o = _topk(S, max(NSAVE, self.kk))
        with self.prof.section("synth"):
            sel, w, action = self._synth(T, S, o)
        ex = self._common_extras(q, T, o, sel, w, S, c0, c1, d, elig, regime, 0)
        sc = self.scales["v4"]
        tot, (t_dnn, t_disp, t_over, t_cos) = self._conf_terms_v4(
            ex["dnn"], ex["disp"] if sel.size > 1 else 0.0, ex["overtime"], ex["cosmax0"] + ex["cosmax1"], sc)
        conf = float(tot / sc["sum_sd"])
        ex.update(t_dnn=float(t_dnn), t_disp=float(t_disp), t_over=float(t_over), t_cos=float(t_cos))
        return o, S, action, conf, ex

    def query(self, q):
        T = self.tasks.get(int(q.task_id))
        if T is None:
            raise api.ContractError(f"task {q.task_id} has no rows in library {self.libname}")
        regime = regime_of(q)
        x0, x1 = self._project(q)
        rs = np.asarray(q.rs, np.float32)[:RSV]
        o, S, action, conf, ex = self._v4(q, T, x0, x1, rs, regime)
        ex.update(self._history_extras(q, T, rs, x0, x1))
        top = o[:NSAVE]
        return api.Result(topk=T.rows[top], scores=S[top].astype(np.float64), confidence=conf, action=action,
                          library=self.libname, extras=ex)

    # pickling for --os-fit-artifact: drop the action memmap, reopen it from the store path
    def __getstate__(self):
        d = dict(self.__dict__)
        d["_act"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if getattr(self, "_act_path", None):
            self._act = np.load(self._act_path, mmap_mode="r")


# ------------------------------------------------------------------------------------------------- V5
class PcaCosV5Fresh(PcaCosV4):
    """V5: V4 at step 0 / after a HIT; after a MISS the continuity + vision + state fusion over ALL candidates."""

    def __init__(self, lam=1.0, alpha=0.25, **kw):
        super().__init__(**kw)
        self.lam, self.alpha = float(lam), float(alpha)
        self.name = self._name("V5fr") + f"_lam{_tag(self.lam)}" + ("" if self.alpha == 0.25 else f"_a{_tag(self.alpha)}")

    def _fresh_scores(self, c, z0, z1, d):
        return -c / self.s_c + self.lam * (z0 + z1) / 2.0 + self.alpha * _z(-d)

    def _fit_scales_fresh(self, TL, prv, codes):
        feats = {"c0": [], "disp": [], "vis0": []}
        for t, T in self.tasks.items():
            pr = prv[T.rows]
            cand = np.nonzero(pr >= 0)[0]
            if cand.size < 3 or np.unique(T.ep).size < 2:
                continue
            n = min(NQ_LOEO, cand.size)
            qi = cand[np.unique(np.linspace(0, cand.size - 1, n).round().astype(np.int64))]
            other = T.ep[qi][:, None] != T.ep[None, :]
            V = np.zeros((qi.size, T.C))
            for Z, m, f in ((T.Z0, T.m0, "v0"), (T.Z1, T.m1, "v1")):
                P = codes[f][T.rows[qi]]
                V += _zmask((_unit_rows(P - m) @ Z.T).astype(np.float64), other)
            D = _pair_l2(T.RS[qi], T.RS).astype(np.float64)
            tail = TL[pr[qi]]                                           # previous library row's tail
            Cc = _pair_rms(tail.astype(np.float64), T.HD.astype(np.float64))
            F = -Cc / self.s_c + self.lam * V / 2.0 + self.alpha * _zmask(-D, other)
            F = np.where(other, F, -np.inf)
            o = np.argsort(-F, axis=1, kind="stable")[:, :self.kk]
            feats["c0"].append(np.take_along_axis(Cc, o[:, :1], 1)[:, 0] / self.s_c)
            feats["disp"].append(_disp_batch(T.HD[o].astype(np.float64)) if self.kk > 1 else np.zeros(qi.size))
            feats["vis0"].append(np.take_along_axis(V, o[:, :1], 1)[:, 0] / 2.0)
        sc = {}
        for k_, v in feats.items():
            x = np.concatenate(v).astype(np.float64)
            sc[k_] = (float(x.mean()), float(x.std()) if x.std() > 1e-12 else 1.0)
        tot = np.concatenate([self._conf_terms_fresh(a, b, c, sc)[0]
                              for a, b, c in zip(feats["c0"], feats["disp"], feats["vis0"])])
        sc["sum_sd"] = float(np.std(tot)) or 1.0
        self.scales["fresh"] = sc

    @staticmethod
    def _conf_terms_fresh(c0n, disp, vis0, sc):
        t_c = -(c0n - sc["c0"][0]) / sc["c0"][1]
        t_disp = -(disp - sc["disp"][0]) / sc["disp"][1]
        t_vis = (vis0 - sc["vis0"][0]) / sc["vis0"][1]
        return t_c + t_disp + t_vis, (t_c, t_disp, t_vis)

    def query(self, q):
        regime = regime_of(q)
        if regime != REGIME_MISS:
            return super().query(q)
        T = self.tasks.get(int(q.task_id))
        if T is None:
            raise api.ContractError(f"task {q.task_id} has no rows in library {self.libname}")
        x0, x1 = self._project(q)
        rs = np.asarray(q.rs, np.float32)[:RSV]
        c0, c1, z0, z1 = self._vision(T, x0, x1)
        with self.prof.section("score"):
            d = self._state(T, rs)
            tail = (np.asarray(q.prev_a_exec[dims.EXEC_STEPS:2 * dims.EXEC_STEPS, dims.ACT_VALID], np.float64)
                    / self.sig).reshape(NH)
            c = np.sqrt(np.maximum(T.h2 - 2.0 * (T.HD @ tail.astype(np.float32)).astype(np.float64)
                                   + float(tail @ tail), 0.0) / NH)
            F = self._fresh_scores(c, z0, z1, d)
            o = _topk(F, max(NSAVE, self.kk))
        with self.prof.section("synth"):
            sel, w, action = self._synth(T, F, o)
        ex = self._common_extras(q, T, o, sel, w, F, c0, c1, d, None, regime, 1)
        j = o[0]
        vis0 = float((z0[j] + z1[j]) / 2.0)
        ah = np.asarray(action[:dims.EXEC_STEPS, dims.ACT_VALID], np.float64) / self.sig
        c_act = float(np.sqrt(np.mean((ah.reshape(NH) - tail) ** 2)))
        sc = self.scales["fresh"]
        tot, (t_c, t_disp, t_vis) = self._conf_terms_fresh(float(c[j]) / self.s_c,
                                                           ex["disp"] if sel.size > 1 else 0.0, vis0, sc)
        conf = float(tot / sc["sum_sd"])
        ex.update(c0=float(c[j]), c_act=c_act, vis0=vis0, f0=float(F[j]), t_c=float(t_c), t_disp=float(t_disp),
                  t_vis=float(t_vis))
        ex.update(self._history_extras(q, T, rs, x0, x1))
        top = o[:NSAVE]
        return api.Result(topk=T.rows[top], scores=F[top].astype(np.float64), confidence=conf, action=action,
                          library=self.libname, extras=ex)
