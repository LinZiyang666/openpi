"""M3 `row_filter_hmm` (R1, ideation A-P4): forward (HMM) filter over the task's library rows.

Hidden state = a library row x of the task (C rows of the searched library). Per episode a belief b over the C rows
is carried forward (within the episode only; reset() clears it; deterministic).

Transition  T(x'|x) = alpha [x' = next(x)] + beta [x' = next^2(x)] + gamma [x' = x] + rest / C  (uniform), where a
            missing next / next^2 sends its mass to the uniform term. Implemented in O(C): the next and next^2 maps
            are injective, so the scatter-add is a plain fancy-index add; plus gamma*b; plus a constant.
Emission    log e(x) = -d_w(x)^2 / (2 tau_w^2), d_w = state-window distance (em = 1: the current rs only, em = 3: the
            3-decision window of M2), tau_w = median within-task 1-NN window distance to another episode;
            in the FRESH regime (q.prev_hit is False, step >= 1: the previous chunk came from the policy) and with
            cont=True additionally  -c(x)^2 / (2 tau_c^2),  c(x) = RMS_sigma(head_x[:5, :7] - prev_a_exec[5:10, :7])
            (35-d), tau_c = median library tail -> nearest other-episode head continuity (B-P1's s_c).
            After a HIT (stale tail = a library chunk) the continuity emission is NOT used (it would be trivially ~0
            along the library trajectory the previous pick came from -> spurious evidence).
Update      b <- e * (T^T b), normalized; b_0 ~ e_0 at step 0.
Temperature (variant): temp multiplies both tau_w and tau_c (temp = 1: the spec's library-fitted widths).
Output      top-5 rows of b; action = posterior-weighted mean of their full chunks; topk = top-10 rows of b;
            scores = b[topk]; confidence = sum of b over the top-5 ("mass", the spec's default), -entropy(b) ("ent"),
            or ("lev", added after the smoke showed mass/ent to be anti-informative after a HIT) the log predictive
            evidence of the current observation, log sum_x e(x) (T^T b)(x) with e unnormalized. Caveat for "lev":
            its scale differs between regimes (the fresh regime adds the continuity term), so one closed-loop
            threshold would favour after-HIT decisions; offline each cell is a single regime.

Offline caveat: the harness feeds the TRACE's executed history, not this method's own picks; the filter only uses
online-legal inputs (rs / hist_rs, prev_a_exec, prev_hit), never its previous output as an observation.

Extras: regime (-1 step 0 / 0 after MISS / 1 after HIT), fresh (continuity emission used), mass5, ent, neff
(1 / sum b^2), dw1 / c1 (state distance / continuity of the MAP row; c1 NaN when not fresh), dwmin / cmin (best over
all rows, i.e. what a plain kNN would see), ep1 / st1 (library episode / step of the MAP row), trk (MAP row vs the
previous MAP row: 1 = its next, 2 = its next^2, 3 = same row, 0 = elsewhere, -1 = step 0), reinit (belief restarted
because the step sequence was broken; 0 normally), lev (log predictive evidence of the current observation,
log sum_x e(x) (T^T b)(x), e unnormalized: a post-hoc "am I lost" confidence candidate).
"""
from __future__ import annotations

import json

import numpy as np

from exp.offline_search.harness import api, dims

try:                                   # dotted-module form (package import)
    from . import f2_common as C
except ImportError:                    # file-path form: the file's directory is on sys.path
    import f2_common as C

TOP_OUT = 5


class _Blk:
    __slots__ = ("rows", "W", "S", "T", "ep", "lep", "lst", "n1", "n2", "n1_src", "n1_dst", "n2_src", "n2_dst", "u")


class RowFilterHMM(api.Method):
    tier = "T1"
    family = "f2_state"

    def __init__(self, lib: str = "big", alpha: float = 0.6, beta: float = 0.15, gamma: float = 0.1, em: int = 1,
                 cont: bool = True, conf: str = "mass", temp: float = 1.0):
        if lib not in ("big", "current"):
            raise ValueError(f"lib must be big|current, got {lib!r}")
        if conf not in ("mass", "ent", "lev"):
            raise ValueError(f"conf must be mass|ent|lev, got {conf!r}")
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        if min(self.alpha, self.beta, self.gamma) < 0 or self.alpha + self.beta + self.gamma >= 1.0:
            raise ValueError("need alpha, beta, gamma >= 0 and alpha + beta + gamma < 1 (uniform term > 0)")
        self.lib, self.em, self.cont, self.conf = lib, int(em), bool(cont), conf
        self.temp = float(temp)
        if not self.temp > 0:
            raise ValueError("temp must be > 0")
        if self.em < 1:
            raise ValueError("em must be >= 1")
        p = lambda v: f"{int(round(100 * v)):02d}"  # noqa: E731
        self.name = (f"M3hmm_{lib}_a{p(self.alpha)}b{p(self.beta)}g{p(self.gamma)}_em{self.em}_"
                     f"{'cont' if self.cont else 'nocont'}_{conf}" + (f"_t{self.temp:g}" if self.temp != 1.0 else ""))
        self.bel = None
        self.last_step = -1
        self.prev_map = -1

    # ------------------------------------------------------------------------------------------ fit
    def fit(self, lib, ctx):
        L, self.libname = C.open_search_library(lib, ctx, self.lib)
        sigma = np.asarray(ctx.action_sigma, np.float32)
        self.sigma = sigma
        sd = C.rs_scale(L, ctx.model)
        self.inv_sd = (1.0 / sd).astype(np.float32)
        Z = np.asarray(dims.valid_state(L.rs, ctx.model), np.float32) * self.inv_sd
        prev = np.asarray(L.prev, np.int64)
        nxt = np.asarray(L.next, np.int64)
        W = C.lib_windows(Z, prev, self.em)
        S = C.seg_norm(L.action, sigma)
        Tl = C.tail_norm(L.action, sigma)
        ep = np.asarray(L.episode, np.int64)
        st = np.asarray(L.step, np.int64)
        loc = np.full(L.L, -1, np.int64)
        self.blocks = {}
        for t in L.tasks():
            b = _Blk()
            b.rows = np.asarray(L.rows_of_task(t), np.int64)
            C_ = b.rows.size
            loc[b.rows] = np.arange(C_)
            b.W = np.ascontiguousarray(W[b.rows])
            b.S = np.ascontiguousarray(S[b.rows])
            b.T = np.ascontiguousarray(Tl[b.rows])
            b.ep = ep[b.rows]
            b.lep = b.ep.astype(np.float32)
            b.lst = st[b.rows].astype(np.float32)
            g1 = nxt[b.rows]
            b.n1 = np.where(g1 >= 0, loc[np.maximum(g1, 0)], -1)
            if np.any((g1 >= 0) & (b.n1 < 0)):
                raise ValueError(f"task {t}: next pointer leaves the task")
            b.n2 = np.where(b.n1 >= 0, b.n1[np.maximum(b.n1, 0)], -1)
            b.n1_src = np.flatnonzero(b.n1 >= 0)
            b.n1_dst = b.n1[b.n1_src]
            b.n2_src = np.flatnonzero(b.n2 >= 0)
            b.n2_dst = b.n2[b.n2_src]
            for dst in (b.n1_dst, b.n2_dst):      # injective maps -> fancy-index += is an exact scatter-add
                if np.unique(dst).size != dst.size:
                    raise ValueError(f"task {t}: next map is not injective")
            b.u = (1.0 - self.gamma - self.alpha * (b.n1 >= 0) - self.beta * (b.n2 >= 0)).astype(np.float64)
            self.blocks[int(t)] = b
        self.action = L.action
        self.tau_w, _ = C.window_scales(self.blocks, None)
        self.tau_c = C.continuity_scale(self.blocks) if self.cont else float("nan")
        self.inv_2tw2 = 1.0 / (2.0 * (self.temp * self.tau_w) ** 2)
        self.inv_2tc2 = 1.0 / (2.0 * (self.temp * self.tau_c) ** 2) if self.cont else 0.0
        self.fit_info = {"library": self.libname, "L": int(L.L), "em": self.em, "alpha": self.alpha,
                         "beta": self.beta, "gamma": self.gamma, "cont": self.cont, "conf": self.conf, "temp": self.temp,
                         "tau_w": self.tau_w, "tau_c": self.tau_c, "rs_sd": sd.tolist(),
                         "cands_per_task": {str(t): int(b.rows.size) for t, b in self.blocks.items()}}
        try:
            ctx.scratch.mkdir(parents=True, exist_ok=True)
            (ctx.scratch / f"{self.name}_fit.json").write_text(json.dumps(self.fit_info, indent=1))
        except Exception:
            pass

    def reset(self, episode):
        self.bel = None
        self.last_step = -1
        self.prev_map = -1

    # ---------------------------------------------------------------------------------------- query
    def _predict(self, b: _Blk, bel: np.ndarray) -> np.ndarray:
        out = self.gamma * bel
        out[b.n1_dst] += self.alpha * bel[b.n1_src]
        out[b.n2_dst] += self.beta * bel[b.n2_src]
        out += float(bel @ b.u) / bel.size
        return out

    def query(self, q):
        b = self.blocks[q.task_id]
        reg = C.regime_code(q)
        fresh = self.cont and q.step >= 1 and reg == 0
        with self.prof.section("emission"):
            x = C.query_window(q, self.inv_sd, self.em)
            dw = C.l2_rows(b.W, x)
            loge = -(dw.astype(np.float64) ** 2) * self.inv_2tw2
            if fresh:
                tail = (np.asarray(q.prev_a_exec[dims.EXEC_STEPS:2 * dims.EXEC_STEPS, dims.ACT_VALID], np.float32)
                        / self.sigma).reshape(C.SEG)
                dc = b.S - tail
                c = np.sqrt(np.einsum("ij,ij->i", dc, dc) / C.SEG)
                loge -= (c.astype(np.float64) ** 2) * self.inv_2tc2
            else:
                c = None
            lmax = float(loge.max())
            e = np.exp(loge - lmax)
        reinit = 0
        with self.prof.section("filter"):
            if self.bel is None or q.step != self.last_step + 1 or self.bel.size != b.rows.size:
                reinit = int(self.bel is not None or q.step != 0)
                post = e.copy()
                prev_map = -1
                lev = lmax + float(np.log(post.mean()))          # log sum_x e(x) / C (uniform prior)
            else:
                post = e * self._predict(b, self.bel)
                prev_map = self.prev_map
                lev = lmax + float(np.log(post.sum()))           # log sum_x e(x) (T^T b)(x)
            post /= post.sum()
            self.bel = post
            self.last_step = q.step
        with self.prof.section("output"):
            o = C.topk_largest(post, api.TOPK_SAVE)
            o5 = o[:TOP_OUT]
            w = post[o5]
            A = np.asarray(self.action[b.rows[o5]], np.float32)
            act = C.synth_mean(A, w)
            mass5 = float(w.sum())
            pz = post[post > 0]
            ent = float(-(pz * np.log(pz)).sum())
        conf = mass5 if self.conf == "mass" else (-ent if self.conf == "ent" else lev)
        j = int(o[0])
        if prev_map < 0:
            trk = -1
        elif j == b.n1[prev_map]:
            trk = 1
        elif j == b.n2[prev_map]:
            trk = 2
        elif j == prev_map:
            trk = 3
        else:
            trk = 0
        self.prev_map = j
        ex = {"regime": reg, "fresh": int(fresh), "mass5": mass5, "ent": ent, "neff": float(1.0 / np.dot(post, post)),
              "dw1": float(dw[j]), "dwmin": float(dw.min()),
              "c1": float(c[j]) if c is not None else float("nan"),
              "cmin": float(c.min()) if c is not None else float("nan"),
              "ep1": float(b.lep[j]), "st1": float(b.lst[j]), "trk": trk, "reinit": reinit, "lev": lev}
        return api.Result(topk=b.rows[o], scores=post[o].astype(np.float64), confidence=float(conf), action=act,
                          library=self.libname, extras=ex)

    def bytes_per_entry(self):
        # z-scaled state window (em x 8 f32) + next pointer (int32) + 35-d head block when continuity is used
        return float(4 * C.RS_N * self.em + 4 + (4 * C.SEG if self.cont else 0))
