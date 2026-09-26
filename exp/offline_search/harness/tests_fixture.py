"""Tiny synthetic store in the exact §5.1 layout, for developing / testing the harness without the real data.

    python -m exp.offline_search.harness.tests_fixture --out /tmp/os_fixture [--key-dim 512] [--seed 0]

Per model x suite: 3 tasks; library/current = 3 successful demos per task, bpool_all = current + 2 more per task
(one failed); queries per arm = 4 episodes per task (inits 0, 1, 10, 2; the tok subsample = inits 0 and 10).
Keys are built so that cosines land in the same range as the real ones (so the zscore+tanh normalizers of the real
configs are not saturated) and the recorded online search (rec_top1 / rec_score / rec_perfield) is computed by an
independent float64 reference of the online formula -- B0 must reproduce it. Floor files are written for pi05 only
(GR00T exercises the "no floor -> NaN" path). Action/state padding follows dims.py.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

from . import dims
from .store import ARMS, MODELS, SUITES, current_params

N_TASKS = 3
LIB_EPS = 3
EXTRA_EPS = 2
Q_INITS = (0, 1, 10, 2)
TOK_INITS = (0, 10, 20, 30, 40)
N_BASIS = 12
TOK_SHAPE = (16, 32)
IMG = (8, 8, 3)


class _Task:
    def __init__(self, rng, D, model, suite, t):
        self.t = t
        self.U0 = rng.standard_normal((D, N_BASIS))
        self.U1 = rng.standard_normal((D, N_BASIS))
        self.rs_amp = rng.uniform(0.8, 1.6, dims.RS_VALID[model])
        self.rs_frq = rng.uniform(0.3, 1.2, dims.RS_VALID[model])
        self.rs_phi = rng.uniform(0, 2 * math.pi, dims.RS_VALID[model])
        self.a_amp = rng.uniform(0.2, 0.6, 6)
        self.a_frq = rng.uniform(0.5, 2.0, 6)
        self.a_phi = rng.uniform(0, 2 * math.pi, 6)
        self.g_on, self.g_off = sorted(rng.uniform(0.2, 0.8, 2))


def _basis(p):
    c = np.linspace(0, 1, N_BASIS)
    return np.exp(-0.5 * ((np.asarray(p)[..., None] - c) / 0.08) ** 2)


def _key(rng, T, which, p, c, alpha, beta, noise):
    U = T.U0 if which == 0 else T.U1
    g = U @ _basis(p) + noise * rng.standard_normal(U.shape[0])
    g = g / np.linalg.norm(g) * beta
    return (alpha * c + g).astype(np.float32)


def _rs(rng, T, model, p, noise=0.05):
    v = T.rs_amp * np.sin(2 * math.pi * T.rs_frq * p + T.rs_phi) + noise * rng.standard_normal(T.rs_amp.size)
    out = np.zeros(dims.RS_DIM[model], np.float32)
    out[: v.size] = v
    return out


def _action(rng, T, model, p, dp, noise=0.03):
    H = dims.HORIZON[model]
    a = np.zeros((H, dims.ACT_FULL_DIMS), np.float32)
    for h in range(H):
        ph = p + h * dp
        a[h, :6] = T.a_amp * np.sin(2 * math.pi * T.a_frq * ph + T.a_phi) + noise * rng.standard_normal(6)
        a[h, 6] = 1.0 if T.g_on <= ph <= T.g_off else -1.0
        a[h, 6] += 0.01 * rng.standard_normal()
    if model == "pi05":
        a[:, 7:] = 0.001
    else:
        a[:, 7:] = 0.8 * rng.standard_normal((H, dims.ACT_FULL_DIMS - 7))
    return a


def _ref_scores(model, suite, q0, q1, qr, L0, L1, LR):
    """Independent float64 reference of the online fused score (audit_values.py _cos/_norm)."""
    p = current_params(model, suite)

    def cos(q, M):
        q = q.astype(np.float64)
        M = M.astype(np.float64)
        return (M @ q) / np.maximum(np.linalg.norm(M, axis=1) * np.linalg.norm(q), 1e-8)

    def nz(x, i):
        return 0.5 * (np.tanh((x - p["mu"][i]) / p["sigma"][i]) + 1.0)

    d = np.linalg.norm(LR.astype(np.float64) - qr.astype(np.float64), axis=1)
    s = np.stack([nz(cos(q0, L0), 0), nz(cos(q1, L1), 1), nz(-d, 2)], 1)
    return s @ np.asarray(p["weights"]), s


def _save(d: pathlib.Path, **arrs):
    d.mkdir(parents=True, exist_ok=True)
    for k, v in arrs.items():
        np.save(d / f"{k}.npy", v)


def build(out, key_dim=512, seed=0):
    out = pathlib.Path(out)
    rng = np.random.default_rng(seed)
    for m in MODELS:
        H = dims.HORIZON[m]
        for s in SUITES:
            key = f"{m}_{s}"
            p = current_params(m, s)
            # cos(q, l) ~ alpha^2 / (alpha^2 + beta^2) for unrelated phases, -> 1 for the same phase
            lo = [p["mu"][i] - 3 * p["sigma"][i] for i in (0, 1)]
            alpha = 1.0
            betas = [alpha * math.sqrt(1 / lo_i - 1) for lo_i in lo]
            c = [rng.standard_normal(key_dim) for _ in (0, 1)]
            c = [v / np.linalg.norm(v) for v in c]
            tasks = [_Task(rng, key_dim, m, s, t) for t in range(N_TASKS)]
            task_str = {t: f"synthetic task {t} of {key}" for t in range(N_TASKS)}

            def make_ep(T, n, jitter=0.0, noise=0.25):
                rows = []
                for k in range(n):
                    ph = k / max(n - 1, 1) + jitter * rng.standard_normal()
                    ph = float(np.clip(ph, 0, 1))
                    rows.append(dict(
                        v0=_key(rng, T, 0, ph, c[0], alpha, betas[0], noise),
                        v1=_key(rng, T, 1, ph, c[1], alpha, betas[1], noise),
                        rs=_rs(rng, T, m, ph), a=_action(rng, T, m, ph, 1.0 / max(n - 1, 1) / 2),
                        t0=rng.standard_normal(TOK_SHAPE).astype(np.float16),
                        t1=rng.standard_normal(TOK_SHAPE).astype(np.float16),
                        i0=rng.integers(0, 255, IMG, dtype=np.uint8), i1=rng.integers(0, 255, IMG, dtype=np.uint8)))
                return rows

            # ---------------- libraries
            lib_eps = []
            for T in tasks:
                for e in range(LIB_EPS + EXTRA_EPS):
                    n = int(rng.integers(12, 19))
                    lib_eps.append(dict(task_id=T.t, success=e != LIB_EPS + EXTRA_EPS - 1, cur=e < LIB_EPS,
                                        rows=make_ep(T, n, jitter=0.02), n=n, name=f"{key}_lib_t{T.t}_e{e}"))
            for lname, sel in (("current", [e for e in lib_eps if e["cur"]]), ("bpool_all", lib_eps)):
                R = [(ei, k, r) for ei, e in enumerate(sel) for k, r in enumerate(e["rows"])]
                Lr = len(R)
                starts, cum = [], 0
                for e in sel:
                    starts.append(cum)
                    cum += e["n"]
                prev = np.full(Lr, -1, np.int32)
                nxt = np.full(Lr, -1, np.int32)
                for ei, e in enumerate(sel):
                    s0 = starts[ei]
                    for k in range(e["n"]):
                        if k > 0:
                            prev[s0 + k] = s0 + k - 1
                        if k < e["n"] - 1:
                            nxt[s0 + k] = s0 + k + 1
                d = out / "library" / key / lname
                _save(d,
                      key_v0=np.stack([r["v0"] for _, _, r in R]), key_v1=np.stack([r["v1"] for _, _, r in R]),
                      rs=np.stack([r["rs"] for _, _, r in R]), action=np.stack([r["a"] for _, _, r in R]),
                      task_id=np.array([sel[ei]["task_id"] for ei, _, _ in R], np.int16),
                      episode=np.array([ei for ei, _, _ in R], np.int32),
                      step=np.array([k for _, k, _ in R], np.int16),
                      ep_len=np.array([sel[ei]["n"] for ei, _, _ in R], np.int16),
                      progress=np.array([k / max(sel[ei]["n"] - 1, 1) for ei, k, _ in R], np.float32),
                      success=np.array([sel[ei]["success"] for ei, _, _ in R], bool), prev=prev, next=nxt)
                _save(d / "tok", v0=np.stack([r["t0"] for _, _, r in R]), v1=np.stack([r["t1"] for _, _, r in R]),
                      **({"img0": np.stack([r["i0"] for _, _, r in R]), "img1": np.stack([r["i1"] for _, _, r in R])}
                         if m == "pi05" else {}))
                (d / "ids.json").write_text(json.dumps([f"{sel[ei]['name']}:{k}" for ei, k, _ in R]))
                (d / "episodes.json").write_text(json.dumps(
                    [dict(name=e["name"], task_id=e["task_id"], task=task_str[e["task_id"]], success=e["success"],
                          start=starts[i], end=starts[i] + e["n"]) for i, e in enumerate(sel)]))
                (d / "meta.json").write_text(json.dumps({"tasks": {task_str[t]: t for t in task_str}, "synthetic": True}))
                if lname == "current":
                    cur = dict(v0=np.stack([r["v0"] for _, _, r in R]), v1=np.stack([r["v1"] for _, _, r in R]),
                               rs=np.stack([r["rs"] for _, _, r in R]), a=np.stack([r["a"] for _, _, r in R]),
                               tid=np.array([sel[ei]["task_id"] for ei, _, _ in R]))
            # ---------------- queries
            for a in ARMS:
                eps, rows_all, r0 = [], [], 0
                for T in tasks:
                    for init in Q_INITS:
                        n = int(rng.integers(10, 21))
                        rows = make_ep(T, n, jitter=0.04, noise=0.35)
                        eps.append(dict(uid=f"fx:{key}_{a}:{T.t}:{init}", file=f"/synthetic/{key}_{a}_{T.t}_{init}.h5",
                                        task=task_str[T.t], task_id=T.t, init=init, success=bool(rng.random() < 0.8),
                                        num_steps=n, start=r0, end=r0 + n))
                        rows_all += rows
                        r0 += n
                N = r0
                ep = np.concatenate([np.full(e["num_steps"], i, np.int32) for i, e in enumerate(eps)])
                step = np.concatenate([np.arange(e["num_steps"], dtype=np.int16) for e in eps])
                K0 = np.stack([r["v0"] for r in rows_all])
                K1 = np.stack([r["v1"] for r in rows_all])
                RS = np.stack([r["rs"] for r in rows_all])
                A_inf = np.stack([r["a"] for r in rows_all])
                rec_top1 = np.zeros(N, np.int32)
                rec_score = np.zeros(N, np.float32)
                rec_pf = np.zeros((N, 3), np.float32)
                for i in range(N):
                    tid = eps[ep[i]]["task_id"]
                    cand = np.flatnonzero(cur["tid"] == tid)
                    f, sp = _ref_scores(m, s, K0[i], K1[i], RS[i], cur["v0"][cand], cur["v1"][cand], cur["rs"][cand])
                    j = int(np.argmax(f))
                    rec_top1[i] = cand[j]
                    rec_score[i] = f[j]
                    rec_pf[i] = sp[j]
                A_hit = cur["a"][rec_top1]
                A_exec = A_inf if a == "inf" else A_hit
                tok_rows = np.concatenate([np.arange(e["start"], e["end"]) for e in eps if e["init"] in TOK_INITS])
                d = out / "queries" / f"{key}_{a}"
                raw = np.concatenate([RS[:, :7], np.zeros((N, 1), np.float32)], 1) + 0.01
                _save(d, ep=ep, step=step, key_v0=K0, key_v1=K1, rs=RS, raw_state=raw.astype(np.float32),
                      a_inf=A_inf, a_hit=A_hit.astype(np.float32), a_exec=A_exec.astype(np.float32),
                      rec_top1=rec_top1, rec_score=rec_score, rec_perfield=rec_pf)
                _save(d / "tok", rows=tok_rows.astype(np.int64),
                      v0=np.stack([rows_all[i]["t0"] for i in tok_rows]), v1=np.stack([rows_all[i]["t1"] for i in tok_rows]),
                      img0=np.stack([rows_all[i]["i0"] for i in tok_rows]), img1=np.stack([rows_all[i]["i1"] for i in tok_rows]))
                (d / "episodes.json").write_text(json.dumps(eps, indent=1))
                (d / "manifest.json").write_text(json.dumps({"synthetic": True, "N": N, "M": int(tok_rows.size)}))
            # ---------------- floor (pi05 only)
            if m == "pi05":
                fd = out / "floor" / key
                fd.mkdir(parents=True, exist_ok=True)
                T = tasks[0]
                P = 30
                A = np.stack([_action(rng, T, m, 0.0, 0.05) for _ in range(P)])
                B = np.stack([_action(rng, T, m, 0.0, 0.05) for _ in range(P)])
                np.savez(fd / "step0_pairs.npz", a_inf_arm=A, a_cache_arm=B, task_id=np.zeros(P, np.int32),
                         init=np.arange(P, dtype=np.int32))
                S, K = 60, 3
                ph = rng.uniform(0, 1, S)
                rec = np.stack([_action(rng, T, m, x, 0.05) for x in ph])
                fr = np.stack([np.stack([_action(rng, T, m, x, 0.05) for _ in range(K)]) for x in ph])
                np.savez(fd / "resample.npz", a_recorded=rec, a_fresh=fr, third=np.minimum(2, (ph * 3).astype(np.int32)),
                         arm=np.array(["inf", "cache"] * (S // 2)))
    (out / "queries" / "READY").write_text("synthetic fixture\n")
    (out / "library" / "READY").write_text("synthetic fixture\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-dim", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    p = build(a.out, a.key_dim, a.seed)
    print(f"fixture written to {p}")


if __name__ == "__main__":
    main()
