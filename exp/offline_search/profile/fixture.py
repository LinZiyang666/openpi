"""fixture -- tiny synthetic store + runner-format run dirs for testing the profile tools.

Matches the documented layouts (protocol §5.1 / R0-A extract_queries.py / R0-D floor scripts /
runner per-decision npz). Key dims are small (default 48) -- the tools never assume 32768.
Padding dims follow the real data: pi05 action dims 7..31 ~0.001, groot 7..31 noise std 0.8,
pi05 rs dims 8..31 exactly 0. Gripper (dim 6) is +-1 and flips at ~55% of each episode.

  python -m exp.offline_search.profile.fixture <dir>      # writes <dir>/store and <dir>/run (real harness runner)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import common as C

H = {"pi05": 10, "groot": 16}
DRS = {"pi05": 32, "groot": 8}


def _traj_actions(rng, task, n_steps, model, jitter):
    """Smooth per-episode action chunks [n, H, 32] in a normalized-looking space."""
    h = H[model]
    t = np.arange(n_steps)[:, None] + np.arange(h)[None, :] / 5.0          # [n, H] in decision units
    p = t / n_steps
    base = np.stack([np.sin(2 * np.pi * (p + 0.1 * d) + task) * (0.5 + 0.1 * d) for d in range(6)], -1)
    base = base + jitter * rng.standard_normal(base.shape)
    grip = np.where(p < 0.55 + 0.03 * rng.standard_normal(), -1.0, 1.0)[..., None]
    a = np.zeros((n_steps, h, 32), np.float32)
    a[..., :6] = base
    a[..., 6:7] = grip
    if model == "pi05":
        a[..., 7:] = 0.001 * rng.standard_normal(a[..., 7:].shape)
    else:
        a[..., 7:] = 0.8 * rng.standard_normal(a[..., 7:].shape)
    return a


def _keys(rng, task, prog, key_dim, noise):
    """Pooled-key-like vectors: shared positive offset (saturated cos) + task + progress structure."""
    n = prog.size
    g = np.random.default_rng(1000 + task)
    tdir = g.standard_normal(key_dim)
    pdir = g.standard_normal((3, key_dim))
    feats = np.stack([np.sin(np.pi * prog), np.cos(np.pi * prog), prog], -1)
    x = 5.0 + 0.3 * tdir[None] + feats @ pdir * 0.4 + noise * rng.standard_normal((n, key_dim))
    return x.astype(np.float32)


def _rs(rng, prog, model):
    n = prog.size
    r = np.zeros((n, DRS[model]), np.float32)
    r[:, :8] = np.stack([prog * (1 + 0.1 * d) for d in range(8)], -1) + 0.02 * rng.standard_normal((n, 8))
    return r


def build_store(root, n_tasks=3, eps_per_task=4, lib_eps_per_task=6, key_dim=48, seed=0, resample_ms=("pi05_spatial",)):
    root = Path(root)
    rng = np.random.default_rng(seed)
    libs = {}
    for ms in C.MODEL_SUITES:
        m, s = ms.split("_")
        rows = {k: [] for k in ("key_v0", "key_v1", "rs", "action", "task_id", "episode", "step", "ep_len", "progress",
                                "success", "traj")}
        ep_id = 0
        for t in range(n_tasks):
            for e in range(lib_eps_per_task):
                n = int(rng.integers(8, 14))
                prog = np.arange(n) / n
                rows["action"].append(_traj_actions(rng, t, n, m, 0.05))
                rows["key_v0"].append(_keys(rng, t, prog, key_dim, 0.05))
                rows["key_v1"].append(_keys(rng, t + 7, prog, key_dim, 0.05))
                rows["rs"].append(_rs(rng, prog, m))
                rows["task_id"].append(np.full(n, t, np.int32))
                rows["episode"].append(np.full(n, ep_id, np.int32))
                rows["traj"].append(np.full(n, ep_id, np.int32))
                rows["step"].append(np.arange(n, dtype=np.int32))
                rows["ep_len"].append(np.full(n, n, np.int32))
                rows["progress"].append(prog.astype(np.float32))
                rows["success"].append(np.full(n, e % 5 != 4))
                ep_id += 1
        full = {k: np.concatenate(v) for k, v in rows.items()}
        N = full["task_id"].size
        nxt = np.arange(1, N + 1, dtype=np.int32)
        prv = np.arange(-1, N - 1, dtype=np.int32)
        last = np.r_[full["episode"][1:] != full["episode"][:-1], True]
        first = np.r_[True, full["episode"][1:] != full["episode"][:-1]]
        nxt[last] = -1
        prv[first] = -1
        full["next"], full["prev"] = nxt, prv
        keep_ep = np.unique(full["episode"])[::2]          # current = every other bpool episode
        variants = {"bpool_all": np.arange(N), "current": np.flatnonzero(np.isin(full["episode"], keep_ep))}
        for name, idx in variants.items():
            d = root / "library" / ms / name
            d.mkdir(parents=True, exist_ok=True)
            remap = np.full(N, -1, np.int64)
            remap[idx] = np.arange(idx.size)
            for k, v in full.items():
                v = v[idx]
                if k in ("next", "prev"):
                    v = np.where(v >= 0, remap[np.maximum(v, 0)], -1).astype(np.int32)
                np.save(d / f"{k}.npy", v)
            (d / "meta.json").write_text(json.dumps({"tasks": {str(t): f"task {t}" for t in range(n_tasks)},
                                                     "source": "synthetic fixture", "rows": int(idx.size)}))
        libs[ms] = {"full": full, "cur_idx": variants["current"]}
        (root / "library" / ms).mkdir(parents=True, exist_ok=True)
    (root / "library" / "READY").write_text("fixture\n")

    for cell in C.CELLS:
        m, s, a = C.parse_cell(cell)
        ms = f"{m}_{s}"
        lib = libs[ms]["full"]
        cur = libs[ms]["cur_idx"]
        rng_c = np.random.default_rng([seed, C.CELLS.index(cell)])
        d = root / "queries" / cell
        d.mkdir(parents=True, exist_ok=True)
        eps, cols, r = [], {k: [] for k in ("ep", "step", "key_v0", "key_v1", "rs", "raw_state", "a_inf", "a_hit",
                                            "a_exec", "rec_top1", "rec_score", "rec_perfield")}, 0
        for t in range(n_tasks):
            for e in range(eps_per_task):
                init = 10 * e
                g0 = np.random.default_rng([seed, 77, C.MODEL_SUITES.index(ms), t, init])  # same init -> same start
                n = int(g0.integers(8, 14)) if a == "inf" else int(np.random.default_rng([seed, 78, t, init]).integers(8, 16))
                prog = np.arange(n) / n
                a_inf = _traj_actions(rng_c, t, n, m, 0.08)
                k0 = _keys(rng_c, t, prog, key_dim, 0.08)
                k1 = _keys(rng_c, t + 7, prog, key_dim, 0.08)
                rs = _rs(rng_c, prog, m)
                ci = cur[lib["task_id"][cur] == t]
                L1 = lib["key_v1"][ci]
                cos = (k1 @ L1.T) / (np.linalg.norm(k1, axis=1)[:, None] * np.linalg.norm(L1, axis=1)[None])
                top = ci[np.argmax(cos, 1)]
                rec = np.searchsorted(cur, top).astype(np.int32)    # row index into library current
                a_hit = lib["action"][top]
                cols["ep"].append(np.full(n, len(eps), np.int32))
                cols["step"].append(np.arange(n, dtype=np.int16))
                cols["key_v0"].append(k0); cols["key_v1"].append(k1); cols["rs"].append(rs)
                cols["raw_state"].append(rs[:, :8].copy())
                cols["a_inf"].append(a_inf); cols["a_hit"].append(a_hit)
                cols["a_exec"].append(a_inf if a == "inf" else a_hit)
                cols["rec_top1"].append(rec); cols["rec_score"].append(cos.max(1).astype(np.float32))
                cols["rec_perfield"].append(np.full((n, 3), 0.5, np.float32))
                eps.append(dict(uid=f"{cell}-t{t}-i{init}", file=f"/synthetic/{cell}/{t}_{init}.h5", task=f"task {t}",
                                task_id=t, init=init, success=bool((t + e) % 3 != 0), num_steps=n, start=r, end=r + n))
                r += n
        for k, v in cols.items():
            np.save(d / f"{k}.npy", np.concatenate(v))
        (d / "episodes.json").write_text(json.dumps(eps, indent=1))
        (d / "manifest.json").write_text(json.dumps({"schema": "fixture", "rows": r}))
    (root / "queries" / "READY").write_text("fixture\n")

    for ms in C.MODEL_SUITES:
        cur_lib = C.Library(root / "library" / ms / "current")
        sigma = C.lib_sigma(cur_lib["action"])
        qi, qc = C.QueryCell(root, ms + "_inf"), C.QueryCell(root, ms + "_cache")
        A = qi["a_inf"][qi.ep_start]
        B = qc["a_inf"][qc.ep_start]
        fd = root / "floor" / ms
        fd.mkdir(parents=True, exist_ok=True)
        e = C.seg_err(A, B, sigma)
        np.savez(fd / "step0_pairs.npz", task_id=qi.ep_task, init=np.zeros_like(qi.ep_task), a_inf_arm=A, a_cache_arm=B,
                 err=e, sigma=sigma)
        if ms in resample_ms:
            S, K = 12, 3
            rows = rng.choice(qi.n, S, replace=False)
            rec = qi["a_inf"][rows]
            fresh = rec[:, None] + 0.05 * rng.standard_normal((S, K) + rec.shape[1:]).astype(np.float32)
            np.savez(fd / "resample.npz", a_recorded=rec, a_fresh=fresh, a_replay_recorded_noise=rec,
                     third=qi.third[rows].astype(np.int32), arm=np.array(["inf"] * S), step=qi.step[rows], sigma=sigma)
    return root


try:  # test method for the real runner (mixed libraries + extras); needs the harness
    from exp.offline_search.harness import api as _api

    class MixedLib(_api.Method):
        """Cosine kNN on key_v1; odd steps search bpool_all, even steps current -> runner writes library='mixed'."""

        name = "fx_mixed"
        tier = "T0"
        family = "fixture"

        def fit(self, lib, ctx):
            self.libs = {"current": lib, "bpool_all": ctx.open_library("bpool_all")}
            self.V = {}
            for n, L in self.libs.items():
                for t in L.tasks():
                    r = L.rows_of_task(t)
                    V = np.asarray(L.key_v1[r], np.float64)
                    self.V[n, t] = (r, V / np.linalg.norm(V, axis=1, keepdims=True))

        def query(self, q):
            use = "bpool_all" if q.step % 2 else "current"
            rows, V = self.V[use, q.task_id]
            x = np.asarray(q.key_v1, np.float64)
            with self.prof.section("sim"):
                s = V @ (x / np.linalg.norm(x))
            o = np.argsort(-s, kind="stable")[:10]
            gap = float(s[o[0]] - s[o[1]]) if o.size > 1 else 0.0
            return _api.Result(topk=rows[o], scores=s[o], confidence=gap, library=use, extras={"gap": gap, "vec": s[o[:3]]})

        def bytes_per_entry(self):
            return 8.0
except Exception:  # noqa: BLE001  pragma: no cover
    MixedLib = None


RUN_METHODS = ("exp.offline_search.harness.baselines:B0Current", "exp.offline_search.harness.baselines:B2Random",
               "exp.offline_search.profile.fixture:MixedLib")


def build_run(store_root, run_root, methods=RUN_METHODS, workers=2, seed=0):
    """Real runner output (harness/run.py) of a few methods on the fixture store:
    <run>/<method>/<cell>.npz + <cell>.json + progress.jsonl + DONE."""
    from exp.offline_search.harness import run as hrun
    out = {}
    for spec in methods:
        r = hrun.run(spec, {}, "all", root=str(store_root), out=str(run_root), workers=workers, seed=seed, profile=True,
                     scoreboard="none", verbose=False)
        if r["errors"]:
            raise RuntimeError(f"fixture run of {spec} failed: {list(r['errors'].values())[0][-2000:]}")
        out[r["method"]] = Path(r["dir"])
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    base = Path(argv[0])
    build_store(base / "store")
    build_run(base / "store", base / "run")
    print(f"fixture written under {base}")


if __name__ == "__main__":
    main()
