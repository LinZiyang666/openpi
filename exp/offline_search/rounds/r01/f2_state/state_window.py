"""M2 `state_window` (R1, ideation A-P2): vision-free observation-side retrieval.

Query  = the last m robot states (rs[:8] of the current decision and of the m-1 previous decisions of this episode,
         online history q.hist_rs), z-scaled per dim by the searched library's std over the valid dims; the oldest
         available state is repeated when the history is shorter (step 0 -> m copies of q.rs).
Library windows are built the same way along the library `prev` pointers.
Search = Euclidean kNN within the task. Synthesis over the top-k:
    "mean": plain mean of the k full chunks (A's default);
    "med":  per-step median of dims 0..5, gripper majority sign (+-1, ties +1), dims 7.. from the top-1 (B's rule).
Confidence = -d_0/tau_w - disp_k/tau_a, with (library-fitted, T1)
    tau_w = median within-task 1-NN window distance to a row of another episode,
    tau_a = median top-k (other-episode) action dispersion on library rows,
    disp_k = RMS over (k, 5, 7) of the sigma-normalized deviation of the top-k executed blocks from their mean.
Library: "big" (bpool_cs for pi0.5 / bpool_all for GR00T, ~10x) or "current".

This method has no continuity term, so its confidence is regime-independent (no spurious after-HIT confidence);
the regime (q.prev_hit) is only reported in the extras (x_regime: -1 step 0, 0 after MISS, 1 after HIT).

Extras: d0 (window distance of the top-1), dk (of the k-th), disp (top-k dispersion), cw = -d0/tau_w,
ca = -disp/tau_a (the two confidence terms), regime, ep1 / st1 (library episode / step of the top-1),
gdis (fraction of the top-k whose step-0 gripper sign disagrees with the majority).
"""
from __future__ import annotations

import json

import numpy as np

from exp.offline_search.harness import api, dims

try:                                   # dotted-module form (package import)
    from . import f2_common as C
except ImportError:                    # file-path form: the file's directory is on sys.path
    import f2_common as C


class _Blk:
    __slots__ = ("rows", "W", "S", "ep", "lep", "lst")


class StateWindow(api.Method):
    tier = "T1"
    family = "f2_state"

    def __init__(self, lib: str = "big", m: int = 3, k: int = 5, synth: str = "mean"):
        if lib not in ("big", "current"):
            raise ValueError(f"lib must be big|current, got {lib!r}")
        if synth not in ("mean", "med"):
            raise ValueError(f"synth must be mean|med, got {synth!r}")
        self.lib, self.m, self.k, self.synth = lib, int(m), int(k), synth
        if self.m < 1 or self.k < 1:
            raise ValueError("m and k must be >= 1")
        self.name = f"M2sw_{lib}_m{self.m}_k{self.k}_{synth}"

    # ------------------------------------------------------------------------------------------ fit
    def fit(self, lib, ctx):
        L, self.libname = C.open_search_library(lib, ctx, self.lib)
        sigma = np.asarray(ctx.action_sigma, np.float32)
        sd = C.rs_scale(L, ctx.model)
        self.inv_sd = (1.0 / sd).astype(np.float32)
        Z = np.asarray(dims.valid_state(L.rs, ctx.model), np.float32) * self.inv_sd
        prev = np.asarray(L.prev, np.int64)
        W = C.lib_windows(Z, prev, self.m)
        S = C.seg_norm(L.action, sigma)
        ep = np.asarray(L.episode, np.int64)
        st = np.asarray(L.step, np.int64)
        self.blocks = {}
        for t in L.tasks():
            b = _Blk()
            b.rows = np.asarray(L.rows_of_task(t), np.int64)
            b.W = np.ascontiguousarray(W[b.rows])
            b.S = np.ascontiguousarray(S[b.rows])
            b.ep = ep[b.rows]
            b.lep = b.ep.astype(np.float32)
            b.lst = st[b.rows].astype(np.float32)
            self.blocks[int(t)] = b
        self.action = L.action              # read-only memmap, shared by the forked workers
        self.tau_w, self.tau_a = C.window_scales(self.blocks, self.k)
        self.fit_info = {"library": self.libname, "L": int(L.L), "m": self.m, "k": self.k, "synth": self.synth,
                         "tau_w": self.tau_w, "tau_a": self.tau_a, "rs_sd": sd.tolist(),
                         "cands_per_task": {str(t): int(b.rows.size) for t, b in self.blocks.items()}}
        try:
            ctx.scratch.mkdir(parents=True, exist_ok=True)
            (ctx.scratch / f"{self.name}_fit.json").write_text(json.dumps(self.fit_info, indent=1))
        except Exception:
            pass

    def reset(self, episode):
        pass                                # no per-episode state (history comes from q.hist_rs)

    # ---------------------------------------------------------------------------------------- query
    def query(self, q):
        b = self.blocks[q.task_id]
        with self.prof.section("window"):
            x = C.query_window(q, self.inv_sd, self.m)
        with self.prof.section("knn"):
            d = C.l2_rows(b.W, x)
            o = C.topk_smallest(d, max(self.k, api.TOPK_SAVE))
        ok = o[: self.k]
        with self.prof.section("synth"):
            A = np.asarray(self.action[b.rows[ok]], np.float32)          # [k, H, 32]
            act = C.synth_mean(A) if self.synth == "mean" else C.synth_med(A)
            disp = C.dispersion(b.S[ok])
        d0 = float(d[o[0]])
        cw = -d0 / self.tau_w
        ca = -disp / self.tau_a
        g0 = A[:, 0, dims.GRIPPER_DIM] >= 0
        maj = g0.sum() * 2 >= g0.size
        ex = {"d0": d0, "dk": float(d[ok[-1]]), "disp": disp, "cw": cw, "ca": ca, "regime": C.regime_code(q),
              "ep1": float(b.lep[o[0]]), "st1": float(b.lst[o[0]]), "gdis": float(np.mean(g0 != maj))}
        return api.Result(topk=b.rows[o], scores=(-d[o]).astype(np.float64), confidence=float(cw + ca),
                          action=act, library=self.libname, extras=ex)

    def bytes_per_entry(self):
        return float(4 * C.RS_N * self.m)    # the materialized z-scaled window (m x 8 float32)
