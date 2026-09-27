"""Exact fast M8 (closed-loop control CL0 / CL1-on-10x): r01 f3_b0plus B0BigLibCons with a pruned exact scorer.

Why: B0BigLibCons scores B0's fused formula over the 10x library with two float32 GEMVs over [C, 32768] per query
(C = 1.1k-3.0k task candidates): 280-770 MB streamed per query, ~40 / 77 ms single thread (spatial / l10; ~9.7 GB/s,
memory-bandwidth bound). Prenormalizing the keys does not help (same bytes).

How (bit-identical picks): per task t and camera f an UNCENTRED PCA basis V (D x m, m = 128, fitted on the task's
big-library keys, precomputed) with, per library row, a_v = V^T v (float64) and the exact residual norm
||v - V a_v||. For a query q (b = V^T q):
    v.q = (v - V a_v).(q - V b) + 2 a_v.b - a_v^T G b          (exact identity, G = V^T V, any b)
so  |v.q - a_v.(2b - G b)| <= ||r_v|| ||r_q||,  ||r_q||^2 = ||q||^2 - 2 b.b + b^T G b.
The cosine bounds (+-1e-4 slack, >> float32 GEMV rounding) map monotonically through B0's zscore-tanh
(weights >= 0) to bounds on the fused score f (+-1e-5); every candidate whose upper bound is below the 10th largest
lower bound cannot be in B0's top-10 (survivors S1). The camera with the wider score interval is then computed
exactly on S1 (B0's float32 arithmetic), the bounds are tightened with it (survivors S2 subset of S1) and the other
camera is computed exactly on S2. Exact cosines come from a row-gather GEMV that reproduces the per-row rounding of
B0's full-block GEMV (see _gather_dot) or, when more than half the block is needed, from the full-block GEMV itself;
so the top-10 order / scores / extras equal B0BigLibCons bit for bit (_m8x_verify.py).
Like B0BigLibCons it holds views of the multi-GB big-library key memmaps: do not use --os-fit-artifact with it (a
pickle would copy the keys); its fit is seconds.

Precompute (one process, BLAS threads = the CPU range):
  OMP_NUM_THREADS=18 OPENBLAS_NUM_THREADS=18 taskset -c 9-17,53-61 .venv/bin/python \
      exp/offline_search/rounds/r02/g2_pcacos/m8fast.py precompute [keys]
writes DERIVED/m8fast/<key>/<big>/t<task>_<f>_{Vt.npy f32[m, D], A.npy f64[C, m], rv.npy f64[C], G.npy f64[m, m]}.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

if __name__ == "__main__":                                  # script mode: make the repo importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

DERIVED = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r02/g2_pcacos")
ROOT = pathlib.Path("/dev/shm/offline_search_store")
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
M_DEFAULT = 128
F3_METHOD = pathlib.Path(__file__).resolve().parents[2] / "r01" / "f3_b0plus" / "method.py"


def _load_f3():
    """r01 f3_b0plus/method.py under the harness's module name for that file (shared if already loaded).
    harness.run is NOT imported (it hides the GPU at import time; this module is also loaded by the plugin)."""
    p = F3_METHOD.resolve()
    name = "osm_" + hashlib.md5(str(p).encode()).hexdigest()[:12]
    if name in sys.modules:
        return sys.modules[name]
    if str(p.parent) not in sys.path:
        sys.path.append(str(p.parent))                    # for its `import f3_core` (appended: no shadowing)
    sp = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(sp)
    sys.modules[name] = mod
    sp.loader.exec_module(mod)
    return mod


_f3 = _load_f3()
_core = _f3.core
from exp.offline_search.harness import api  # noqa: E402
from exp.offline_search.harness.baselines import _EPS, _l2, _zt  # noqa: E402

COS_SLACK = 1e-4
F_SLACK = 1e-5
GATHER = 32


def art_dir(key: str, big: str) -> pathlib.Path:
    return DERIVED / "m8fast" / key / big


class _Bound:
    __slots__ = ("Vt", "A", "rv", "G")


class B0BigLibConsFast(_f3.B0BigLibCons):
    """B0BigLibCons (M8) with the exact pruned scorer. Same kwargs (k, synth) + m (basis rank of the bound)."""

    family = "g2_pcacos"

    def __init__(self, k=1, synth="top1", m=M_DEFAULT, check=False, refine=True):
        super().__init__(k=k, synth=synth)
        self.m = int(m)
        self.refine = bool(refine)          # two-stage (camera-by-camera) exact re-scoring; False: both cameras on S1
        self.check = bool(check)            # debug: also run the full exact path and assert equality
        self.name = self.name.replace("M8_", "M8x_", 1)
        self.n_pruned = 0                   # diagnostics (per instance): pruned / full-path queries, summed survivors
        self.n_full = 0                     # queries whose first survivor set needed a full-block GEMV (1 camera)
        self.sub_total = 0
        self.sub2_total = 0

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        big = _core.BIG_LIB[ctx.model]
        d = art_dir(ctx.lib_key, big)
        self.fast_ok = all(x >= 0 for x in self.w) and d.exists()
        self.bounds = {}
        if not self.fast_ok:
            return
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("L") != int(ctx.open_library(big).L):
            raise api.ContractError(f"{d}: m8fast artifacts were built for L={meta.get('L')}")
        with ctx.prof.section("bounds_load"):
            for t, b in self.blocks.items():
                per = {}
                for f in ("v0", "v1"):
                    B = _Bound()
                    B.Vt = np.ascontiguousarray(np.load(d / f"t{t}_{f}_Vt.npy")[:self.m], np.float32)
                    B.A = np.ascontiguousarray(np.load(d / f"t{t}_{f}_A.npy")[:, :self.m], np.float64)
                    B.rv = np.load(d / f"t{t}_{f}_rv{self.m}.npy").astype(np.float64)
                    B.G = np.load(d / f"t{t}_{f}_G.npy")[:self.m, :self.m].astype(np.float64)
                    if B.A.shape[0] != b.rows.size:
                        raise api.ContractError(f"{d}: task {t} has {B.A.shape[0]} rows, block has {b.rows.size}")
                    per[f] = B
                self.bounds[int(t)] = per

    # -- exact float32 dots of a row subset, per row bitwise equal to B0's full-block GEMV `V @ q`.
    # OpenBLAS sgemv_t (this build: HASWELL kernels) computes rows in groups of 4 with one arithmetic; the last C % 4
    # rows of a call go through remainder kernels whose rounding differs (2 / 3 rows), and numpy sends a 1-row product
    # to sdot (differs as well). So: survivors in the grouped region are gathered into calls whose size is a multiple
    # of 4 (padded with a duplicate row); survivors among the last C % 4 rows are taken from a call on the block's last
    # 4 + C % 4 rows, which reproduces the full call's tail structure. Verified by _m8x_verify.py (bit for bit).
    @staticmethod
    def _gather_dot(V, sub, q):
        C = V.shape[0]
        tail0 = C - C % 4
        out = np.empty(sub.size, np.float32)
        main = sub < tail0
        mi = np.nonzero(main)[0]
        for lo in range(0, mi.size, GATHER):
            pos = sub[mi[lo:lo + GATHER]]
            n = pos.size
            pad = (-n) % 4
            if pad:
                pos = np.concatenate([pos, np.repeat(pos[-1:], pad)])
            out[mi[lo:lo + GATHER]] = (V[pos] @ q)[:n]
        if not main.all():
            ti = np.nonzero(~main)[0]
            start = max(tail0 - 4, 0)
            y = V[start:C] @ q
            out[ti] = y[sub[ti] - start]
        return out

    def _cos_bounds(self, B, n, q32, qn):
        b = (B.Vt @ q32).astype(np.float64)
        q64 = q32.astype(np.float64)
        q2 = float(q64 @ q64)
        Gb = B.G @ b
        rq2 = q2 - 2.0 * float(b @ b) + float(b @ Gb)
        rq = np.sqrt(max(rq2, 0.0) + 1e-6 * q2)
        est = B.A @ (2.0 * b - Gb)
        rad = B.rv * rq
        den = np.maximum(n * qn, _EPS).astype(np.float64)       # the float32 denominator B0 divides by
        return (est - rad) / den - COS_SLACK, (est + rad) / den + COS_SLACK

    def _zt64(self, x, i):
        sg = self.sg[i] if self.sg[i] > 1e-12 else 1.0
        return 0.5 * (np.tanh((x - self.mu[i]) / sg) + 1.0)

    def _exact_cos(self, V, n, sub, q, qn, C):
        """B0's exact float32 cosine for positions `sub` (full-block GEMV when more than half the block is needed)."""
        if sub.size * 2 > C:
            dots = (V @ q)[sub]
        else:
            dots = self._gather_dot(V, sub, q)
        return dots / np.maximum(n[sub] * qn, _EPS)

    def score_all(self, q):
        b = self.blocks[q.task_id]
        per = self.bounds.get(int(q.task_id)) if self.fast_ok else None
        C = b.rows.size
        if per is None or C <= 4 * _core.NSHORT:
            return super().score_all(q)
        K = _core.NSHORT
        q0 = np.asarray(q.key_v0, np.float32)
        q1 = np.asarray(q.key_v1, np.float32)
        with self.prof.section("sim_rs"):
            d = _l2(b.RS, q.rs)
            s2 = _zt(-d, self.mu[2], self.sg[2])
        with self.prof.section("bound"):
            qn = (np.float32(np.sqrt(np.dot(q0, q0))), np.float32(np.sqrt(np.dot(q1, q1))))
            lo0, hi0 = self._cos_bounds(per["v0"], b.n0, q0, qn[0])
            lo1, hi1 = self._cos_bounds(per["v1"], b.n1, q1, qn[1])
            z = ((self._zt64(lo0, 0), self._zt64(hi0, 0)), (self._zt64(lo1, 1), self._zt64(hi1, 1)))
            s2d = self.w[2] * s2.astype(np.float64)
            flo = self.w[0] * z[0][0] + self.w[1] * z[1][0] + s2d - F_SLACK
            fhi = self.w[0] * z[0][1] + self.w[1] * z[1][1] + s2d + F_SLACK
            thr = np.partition(flo, C - K)[C - K]
            S1 = np.nonzero(fhi >= thr)[0]
        # refine the camera with the wider score interval first (exact on S1), re-bound with it, then the other
        # camera exact on the (smaller) survivor set S2. Any row outside S2 has f <= its upper bound < a threshold
        # that at least K rows of S2 reach, so B0's top-K lies in S2 (see module doc).
        wid = [self.w[0] * float((z[0][1][S1] - z[0][0][S1]).sum()), self.w[1] * float((z[1][1][S1] - z[1][0][S1]).sum())]
        a = 0 if wid[0] >= wid[1] else 1
        o_ = 1 - a
        if not self.refine:
            if S1.size * 2 > C:
                self.n_full += 1
                return super().score_all(q)
            with self.prof.section("exact"):
                c0 = self._gather_dot(b.V0, S1, q0) / np.maximum(b.n0[S1] * qn[0], _EPS)
                c1 = self._gather_dot(b.V1, S1, q1) / np.maximum(b.n1[S1] * qn[1], _EPS)
                s0 = _zt(c0, self.mu[0], self.sg[0])
                s1 = _zt(c1, self.mu[1], self.sg[1])
                s2s = s2[S1]
                f = np.zeros(S1.size, np.float32)
                f += self.w[0] * s0
                f += self.w[1] * s1
                f += self.w[2] * s2s
            self.n_pruned += 1
            self.sub_total += int(S1.size)
            self.sub2_total += int(S1.size)
            return b.rows[S1], f, (c0, c1, d[S1], s0, s1, s2s)
        VV, NN, QQ = (b.V0, b.V1), (b.n0, b.n1), (q0, q1)
        with self.prof.section("exact"):
            cA = self._exact_cos(VV[a], NN[a], S1, QQ[a], qn[a], C)
            sA = _zt(cA, self.mu[a], self.sg[a])
            flo2 = self.w[a] * sA.astype(np.float64) + self.w[o_] * z[o_][0][S1] + s2d[S1] - F_SLACK
            fhi2 = self.w[a] * sA.astype(np.float64) + self.w[o_] * z[o_][1][S1] + s2d[S1] + F_SLACK
            k2 = min(K, S1.size)
            thr2 = np.partition(flo2, S1.size - k2)[S1.size - k2]
            keep = fhi2 >= thr2
            sub = S1[keep]
            cB = self._exact_cos(VV[o_], NN[o_], sub, QQ[o_], qn[o_], C)
            sB = _zt(cB, self.mu[o_], self.sg[o_])
            cA, sA = cA[keep], sA[keep]
            c0, s0, c1, s1 = (cA, sA, cB, sB) if a == 0 else (cB, sB, cA, sA)
            s2s = s2[sub]
            f = np.zeros(sub.size, np.float32)
            f += self.w[0] * s0
            f += self.w[1] * s1
            f += self.w[2] * s2s
        self.n_pruned += 1
        self.n_full += int(S1.size * 2 > C)
        self.sub_total += int(S1.size)
        self.sub2_total += int(sub.size)
        if self.check:
            rows_f, f_f, _ = super().score_all(q)
            o_full = np.argsort(-f_f, kind="stable")[:K]
            o_sub = np.argsort(-f, kind="stable")[:K]
            assert np.array_equal(rows_f[o_full], b.rows[sub][o_sub]) and np.array_equal(f_f[o_full], f[o_sub]), \
                "m8fast: pruned top-10 differs from the full exact scorer"
        return b.rows[sub], f, (c0, c1, d[sub], s0, s1, s2s)


# ------------------------------------------------------------------------------------------------ precompute
def precompute(keys, mmax=M_DEFAULT):
    for key in keys:
        model = key.split("_")[0]
        big = BIG[model]
        lib = ROOT / "library" / key / big
        task = np.load(lib / "task_id.npy").astype(np.int64)
        d = art_dir(key, big)
        d.mkdir(parents=True, exist_ok=True)
        for f in ("v0", "v1"):
            K = np.load(lib / f"key_{f}.npy", mmap_mode="r")
            for t in np.unique(task):
                t0 = time.time()
                rows = np.nonzero(task == t)[0]
                lo, hi = int(rows[0]), int(rows[-1]) + 1
                assert hi - lo == rows.size, "big library not task-contiguous"
                X = np.asarray(K[lo:hi], np.float64)
                G_ = X @ X.T
                ev, U = np.linalg.eigh(G_)
                idx = np.argsort(ev)[::-1][:mmax]
                V = (X.T @ U[:, idx]) / np.sqrt(np.maximum(ev[idx], 1e-30))
                V32 = np.ascontiguousarray(V.astype(np.float32))            # [D, m] as stored
                V64 = V32.astype(np.float64)
                A = X @ V64                                                   # [C, m], consistent with V32
                np.save(d / f"t{t}_{f}_Vt.npy", np.ascontiguousarray(V32.T))
                np.save(d / f"t{t}_{f}_A.npy", A)
                np.save(d / f"t{t}_{f}_G.npy", V64.T @ V64)
                for m in sorted({32, 64, 96, 128, mmax}):
                    if m > mmax:
                        continue
                    R = X - A[:, :m] @ V64[:, :m].T
                    np.save(d / f"t{t}_{f}_rv{m}.npy", np.sqrt(np.einsum("ij,ij->i", R, R)))
                print(f"[{key}] {f} task {t}: C={rows.size} ({time.time() - t0:.1f}s) resid@{mmax} "
                      f"{1 - ev[idx].sum() / ev.sum():.2e}", flush=True)
        L = int(task.size)
        (d / "meta.json").write_text(json.dumps({"key": key, "lib": big, "L": L, "mmax": mmax,
                                                 "src_size": (lib / "key_v0.npy").stat().st_size}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "precompute":
        ks = sys.argv[2].split(",") if len(sys.argv) > 2 else ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]
        precompute(ks)
    else:
        print(__doc__)
