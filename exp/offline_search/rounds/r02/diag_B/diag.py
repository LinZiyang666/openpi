"""R2 ideation-B diagnostics (read-only on the store; writes only to the scratch dir).

Per cell: project query keys into the big-library PCA basis (v0, v1), project the current library with the same
basis, then evaluate:
  A. PCA-k two-camera task-centred cosine z-sum (+ st * z(-state L2)) over ALL task candidates, mean-k synthesis,
     on the big and the current library, per regime (step0 / fresh / stale).  Optional step alignment at step 0.
  B. stale-regime confidence features (drift / stuck detectors) -> AURC.
  C. gripper-transition decisions: flip-aware synthesis variants (err + grip_mis).
  D. fresh regime: continuity fused with the visual z-sum over all candidates.
  E. single-thread timing of the PCA-32 path.
Usage: taskset -c ... python diag.py <cell> [--k 32] [--out DIR]
"""
import argparse, json, os, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path("/dev/shm/offline_search_store")
PCA_DIR = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r01/f4_vision/pca")
OUT = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r02_ideation_B")
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
RSV = 8
EPS = 1e-8


def log(*a):
    print(*a, flush=True)


def aurc(conf, err):
    """mean over c of risk(c), conf descending (stable, ties by row order)."""
    o = np.lexsort((np.arange(conf.size), -conf))
    e = err[o]
    cm = np.cumsum(e) / np.arange(1, e.size + 1)
    return float(cm.mean())


def risk_at(conf, err, c):
    o = np.lexsort((np.arange(conf.size), -conf))
    n = max(1, int(round(c * conf.size)))
    return float(err[o[:n]].mean())


def zrows(S):
    m = S.mean(1, keepdims=True)
    s = S.std(1, keepdims=True)
    s[s < 1e-12] = 1.0
    return (S - m) / s


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


def project_keys(Kmm, mu, B, chunk=1024):
    out = np.empty((Kmm.shape[0], B.shape[1]), np.float32)
    muB = mu @ B
    for lo in range(0, Kmm.shape[0], chunk):
        out[lo:lo + chunk] = np.asarray(Kmm[lo:lo + chunk], np.float32) @ B - muB
    return out


class Lib:
    pass


def fit_pca_current(key, kmax, cache_dir):
    """task-agnostic PCA fitted on the CURRENT library keys (centered SVD); returns {f: (mu, B[:, :kmax])}."""
    out = {}
    for f in ("v0", "v1"):
        cb = cache_dir / f"pcacur_{key}_{f}_basis.npy"; cm = cache_dir / f"pcacur_{key}_{f}_mean.npy"
        if cb.exists():
            out[f] = (np.load(cm), np.load(cb)[:, :kmax]); continue
        X = np.asarray(np.load(ROOT / "library" / key / "current" / f"key_{f}.npy", mmap_mode="r"), np.float32)
        mu = X.mean(0)
        Xc = (X - mu).astype(np.float32)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        B = np.ascontiguousarray(Vt[:128].T.astype(np.float32))
        np.save(cb, B); np.save(cm, mu.astype(np.float32))
        out[f] = (mu.astype(np.float32), B[:, :kmax])
    return out


def load_lib(key, name, pca, kdim, cache_dir, pca_cur=None):
    d = ROOT / "library" / key / name
    L = Lib()
    L.name = name
    L.act = np.load(d / "action.npy", mmap_mode="r")
    L.rs = np.asarray(np.load(d / "rs.npy", mmap_mode="r")[:, :RSV], np.float32)
    L.task = np.asarray(np.load(d / "task_id.npy"), np.int64)
    L.ep = np.asarray(np.load(d / "episode.npy"), np.int64)
    L.step = np.asarray(np.load(d / "step.npy"), np.int64)
    L.ep_len = np.asarray(np.load(d / "ep_len.npy"), np.int64)
    L.progress = np.asarray(np.load(d / "progress.npy"), np.float32)
    L.success = np.asarray(np.load(d / "success.npy"), bool)
    L.prev = np.asarray(np.load(d / "prev.npy"), np.int64)
    L.next = np.asarray(np.load(d / "next.npy"), np.int64)
    L.L = L.task.size
    L.P = {}
    for f in ("v0", "v1"):
        cp = cache_dir / f"libproj_{key}_{name}_{f}.npy"
        if cp.exists():
            P = np.load(cp)
        elif name == BIG[key.split("_")[0]]:
            P = np.load(PCA_DIR / key / name / f / "proj.npy").astype(np.float32)
            np.save(cp, P)
        else:
            mu, B = pca[f]
            P = project_keys(np.load(d / f"key_{f}.npy", mmap_mode="r"), mu, B)
            np.save(cp, P)
        L.P[f] = P[:, :kdim]
    L.Pc = {}
    if pca_cur is not None:
        for f in ("v0", "v1"):
            cp = cache_dir / f"libprojcur_{key}_{name}_{f}.npy"
            if cp.exists():
                P = np.load(cp)
            else:
                mu, B = pca_cur[f]
                P = project_keys(np.load(d / f"key_{f}.npy", mmap_mode="r"), mu, B)
                np.save(cp, P)
            L.Pc[f] = P[:, :kdim]
    # heads (5,7) and per-row scalar library statistics
    a = np.asarray(L.act[:, :, :7], np.float32)
    L.head = a[:, :5]
    L.tail = a[:, 5:10]
    # motion of the library between consecutive decisions (state delta to the next row)
    nxt = L.next
    ok = nxt >= 0
    dm = np.zeros(L.L, np.float32)
    dm[ok] = np.linalg.norm(L.rs[nxt[ok]] - L.rs[ok], axis=1)
    L.motion_next = dm
    L.has_next = ok
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--kdims", default="32,64")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cell = args.cell
    model, suite, arm = cell.split("_")
    key = f"{model}_{suite}"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"
    cache_dir.mkdir(exist_ok=True)
    big = BIG[model]
    kdims = [int(x) for x in args.kdims.split(",")]
    KMAX = max(kdims)
    t0 = time.time()

    # ---------------------------------------------------------------- PCA bases
    pca = {}
    for f in ("v0", "v1"):
        d = PCA_DIR / key / big / f
        pca[f] = (np.load(d / "mean.npy").astype(np.float32),
                  np.ascontiguousarray(np.load(d / "basis.npy", mmap_mode="r")[:, :KMAX], np.float32))

    # ---------------------------------------------------------------- queries
    qd = ROOT / "queries" / cell
    eps_json = json.load(open(qd / "episodes.json"))
    ep = np.asarray(np.load(qd / "ep.npy"), np.int64)
    step = np.asarray(np.load(qd / "step.npy"), np.int64)
    N = ep.size
    task = np.asarray([e["task_id"] for e in eps_json], np.int64)[ep]
    ep_success = np.asarray([bool(e["success"]) for e in eps_json])[ep]
    ep_nsteps = np.asarray([e["num_steps"] for e in eps_json], np.int64)[ep]
    rs = np.asarray(np.load(qd / "rs.npy", mmap_mode="r")[:, :RSV], np.float32)
    raw = np.asarray(np.load(qd / "raw_state.npy"), np.float32)
    a_inf = np.asarray(np.load(qd / "a_inf.npy", mmap_mode="r")[:, :5, :7], np.float32)
    a_exec_full = np.load(qd / "a_exec.npy", mmap_mode="r")
    a_exec = np.asarray(a_exec_full[:, :10, :7], np.float32)
    pca_cur = fit_pca_current(key, KMAX, cache_dir)
    QPc = {}
    for f in ("v0", "v1"):
        cp = cache_dir / f"qprojcur_{cell}_{f}.npy"
        if cp.exists():
            P = np.load(cp)
        else:
            mu, B = pca_cur[f]
            P = project_keys(np.load(qd / f"key_{f}.npy", mmap_mode="r"), mu, B)
            np.save(cp, P)
        QPc[f] = P
    QP = {}
    for f in ("v0", "v1"):
        cp = cache_dir / f"qproj_{cell}_{f}.npy"
        if cp.exists():
            P = np.load(cp)
        else:
            mu, B = pca[f]
            P = project_keys(np.load(qd / f"key_{f}.npy", mmap_mode="r"), mu, B)
            np.save(cp, P)
        QP[f] = P
    log(f"[{cell}] queries N={N} projected in {time.time() - t0:.1f}s")

    # previous decision index within the episode (-1 at step 0)
    prev_idx = np.full(N, -1, np.int64)
    same = np.r_[False, (ep[1:] == ep[:-1]) & (step[1:] == step[:-1] + 1)]
    prev_idx[same] = np.arange(N)[same] - 1
    assert np.all((step == 0) == (prev_idx < 0)), "episode order assumption broken"
    regime = np.where(step == 0, 0, 1 if arm == "inf" else 2)   # 0 step0, 1 fresh, 2 stale

    # ---------------------------------------------------------------- libraries + sigma
    Lcur = load_lib(key, "current", pca, KMAX, cache_dir, pca_cur)
    Lbig = load_lib(key, big, pca, KMAX, cache_dir, pca_cur)
    sigma = np.asarray(Lcur.act[:, :5, :7], np.float64).reshape(-1, 7).std(0)  # per dim over rows x steps
    sigma = np.asarray(Lcur.act[:, :5, :7], np.float64).std(axis=(0, 1))
    sig = sigma.astype(np.float32)
    log(f"[{cell}] sigma={np.round(sigma, 3)} libs: current L={Lcur.L} big L={Lbig.L}")

    def err_of(ahat):        # ahat [N,5,7]
        d = (ahat.astype(np.float64) - a_inf) / sigma
        return np.sqrt((d * d).mean(axis=(1, 2)))

    def gmis_of(ahat):
        return ((ahat[:, :, 6] >= 0) != (a_inf[:, :, 6] >= 0)).mean(1)

    # tail of the previous executed chunk (sigma units), gripper of the previous executed step
    tail = np.zeros((N, 5, 7), np.float32)
    tail[prev_idx >= 0] = a_exec[prev_idx[prev_idx >= 0], 5:10]
    gprev = np.ones(N, np.float32)
    gprev[prev_idx >= 0] = np.where(a_exec[prev_idx[prev_idx >= 0], 4, 6] >= 0, 1.0, -1.0)
    # gripper transition of the truth: any sign in a_inf[:5,6] != gprev (step>=1) or sign change within
    gi = np.where(a_inf[:, :, 6] >= 0, 1.0, -1.0)
    trans = (gi != gprev[:, None]).any(1)
    trans_within = (gi != gi[:, :1]).any(1)
    trans[step == 0] = trans_within[step == 0]

    results = {"cell": cell, "N": int(N), "n_regime": {str(r): int((regime == r).sum()) for r in (0, 1, 2)},
               "sigma": sigma.tolist(), "A": {}, "B": {}, "C": {}, "D": {}, "E": {}}

    def summarize(err, gm=None, name=None, conf=None):
        o = {}
        for r, rn in ((0, "step0"), (1, "fresh"), (2, "stale")):
            m = regime == r
            if m.sum() == 0:
                continue
            o[rn] = {"err": float(err[m].mean()), "p50": float(np.median(err[m])), "n": int(m.sum())}
            if gm is not None:
                o[rn]["gmis"] = float(gm[m].mean())
            if conf is not None:
                o[rn]["aurc"] = aurc(conf[m], err[m])
            if r == 2:
                o[rn]["err_succ"] = float(err[m & ep_success].mean()) if (m & ep_success).any() else None
                o[rn]["err_fail"] = float(err[m & ~ep_success].mean()) if (m & ~ep_success).any() else None
        return o

    # ---------------------------------------------------------------- scoring engine
    def score_task(L, tq, kdim, st, fields=("v0", "v1"), align=None, basis="big", mask=None, alignw=None):
        """returns (qidx, cand_rows, S [nq, C]) for task tq"""
        qi = np.nonzero(task == tq)[0]
        ci = np.nonzero(L.task == tq)[0]
        if mask is not None:
            ci2 = ci[mask[ci]]
            if ci2.size >= 5:
                ci = ci2
        S = np.zeros((qi.size, ci.size), np.float64)
        parts = {}
        LP = L.P if basis == "big" else L.Pc
        QPb = QP if basis == "big" else QPc
        for f in fields:
            Pc = LP[f][ci, :kdim]
            m = Pc.mean(0)
            Zc = unit(Pc - m)
            Zq = unit(QPb[f][qi, :kdim] - m)
            C = Zq @ Zc.T
            parts[f] = C
            S += zrows(C)
        Dq = None
        if st:
            q2 = np.einsum("ij,ij->i", rs[qi], rs[qi])[:, None]
            c2 = np.einsum("ij,ij->i", L.rs[ci], L.rs[ci])[None, :]
            Dq = np.sqrt(np.maximum(q2 - 2.0 * (rs[qi] @ L.rs[ci].T) + c2, 0.0))
            S += st * zrows(-Dq)
        if align is not None:
            # step-0 alignment: at query step 0 only library rows with step <= align are allowed
            s0 = step[qi] == 0
            bad = (L.step[ci] > align)[None, :] & s0[:, None]
            S = np.where(bad, -1e9, S)
        if alignw is not None:
            # step window at ALL steps: |lib step - query step| <= alignw (soft: -1e9 outside, fallback if none)
            bad = np.abs(L.step[ci][None, :] - step[qi][:, None]) > alignw
            bad[bad.all(1)] = False
            S = np.where(bad, -1e9, S)
        return qi, ci, S, parts, Dq

    def topk_rows(S, k):
        k = min(k, S.shape[1])
        o = np.argpartition(-S, k - 1, axis=1)[:, :k]
        so = np.take_along_axis(S, o, 1)
        oo = np.argsort(-so, axis=1, kind="stable")
        return np.take_along_axis(o, oo, 1)

    def evaluate(L, kdim, st, ks=(1, 5, 8), align=None, keep=None, basis="big", mask=None, alignw=None):
        """returns dict k -> (err, gmis, ahat) and optionally per-decision features (keep)"""
        errs = {k: np.zeros(N) for k in ks}
        gms = {k: np.zeros(N) for k in ks}
        feats = {} if keep else None
        KM = max(ks)
        TOP = np.zeros((N, KM), np.int64)
        SC = np.zeros((N, KM), np.float64)
        if keep:
            for nm in keep:
                feats[nm] = np.zeros(N)
        for tq in np.unique(task):
            qi, ci, S, parts, Dq = score_task(L, tq, kdim, st, align=align, basis=basis, mask=mask, alignw=alignw)
            o = topk_rows(S, KM)
            rows = ci[o]
            TOP[qi] = rows
            SC[qi] = np.take_along_axis(S, o, 1)
            H = L.head[rows]  # [nq, KM, 5, 7]
            for k in ks:
                ah = H[:, :k].mean(1)
                errs[k][qi] = err_of_rows(ah, qi)
                gms[k][qi] = ((ah[:, :, 6] >= 0) != (a_inf[qi][:, :, 6] >= 0)).mean(1)
            if keep:
                feats["s0"][qi] = SC[qi, 0]
                feats["s5"][qi] = SC[qi, :5].mean(1)
                feats["cos0_0"][qi] = np.take_along_axis(parts["v0"], o[:, :1], 1)[:, 0]
                feats["cos1_0"][qi] = np.take_along_axis(parts["v1"], o[:, :1], 1)[:, 0]
                feats["cosmax0"][qi] = parts["v0"].max(1)
                feats["cosmax1"][qi] = parts["v1"].max(1)
                if Dq is None:
                    q2 = np.einsum("ij,ij->i", rs[qi], rs[qi])[:, None]
                    c2 = np.einsum("ij,ij->i", L.rs[ci], L.rs[ci])[None, :]
                    Dq = np.sqrt(np.maximum(q2 - 2.0 * (rs[qi] @ L.rs[ci].T) + c2, 0.0))
                feats["d0"][qi] = np.take_along_axis(Dq, o[:, :1], 1)[:, 0]
                feats["dnn"][qi] = Dq.min(1)
                feats["d5"][qi] = np.take_along_axis(Dq, o[:, :5], 1).mean(1)
                H5 = H[:, :5].reshape(qi.size, 5, -1) / np.tile(sig, 5)[None, None, :]
                iu = np.triu_indices(5, 1)
                Dd = H5[:, iu[0]] - H5[:, iu[1]]
                feats["disp5"][qi] = np.sqrt((Dd * Dd).mean(2)).mean(1)
                feats["lag5"][qi] = step[qi] - L.step[rows[:, :5]].mean(1)
                feats["lagfrac5"][qi] = step[qi] / np.maximum(L.ep_len[rows[:, :5]].mean(1), 1) - L.progress[rows[:, :5]].mean(1)
                feats["libmotion5"][qi] = L.motion_next[rows[:, :5]].mean(1)
                feats["libstep5"][qi] = L.step[rows[:, :5]].mean(1)
                feats["succ5"][qi] = L.success[rows[:, :5]].mean(1)
                feats["libprog5"][qi] = L.progress[rows[:, :5]].mean(1)
                feats["ncand"][qi] = ci.size
        return errs, gms, TOP, SC, feats

    def err_of_rows(ah, qi):
        d = (ah.astype(np.float64) - a_inf[qi]) / sigma
        return np.sqrt((d * d).mean(axis=(1, 2)))

    # ================================================================ A. representation x library x k
    A = {}
    for lname, L in (("current", Lcur), ("big", Lbig)):
        for basis in ("big", "cur"):
            for kdim in kdims:
                for st in (0.0, 1.0):
                    errs, gms, _, _, _ = evaluate(L, kdim, st, basis=basis)
                    for k in errs:
                        tag = f"{lname}_b{basis}_pca{kdim}_st{st:g}_mean{k}"
                        A[tag] = summarize(errs[k], gms[k])
                    log(f"[{cell}] A {lname} basis={basis} pca{kdim} st{st:g}: " + " | ".join(
                        f"k{k} " + " ".join(f"{rn}={A[f'{lname}_b{basis}_pca{kdim}_st{st:g}_mean{k}'][rn]['err']:.3f}"
                                             for rn in A[f"{lname}_b{basis}_pca{kdim}_st{st:g}_mean{k}"]) for k in errs))
        bb = "big" if lname == "big" else "cur"
        # step-0 alignment variants (pca32, st1)
        for al in (0, 2):
            errs, gms, _, _, _ = evaluate(L, 32, 1.0, ks=(5,), align=al, basis=bb)
            A[f"{lname}_pca32_st1_mean5_align{al}"] = summarize(errs[5], gms[5])
            log(f"[{cell}] A {lname} pca32 st1 mean5 align<= {al}: step0 err {A[f'{lname}_pca32_st1_mean5_align{al}'].get('step0', {}).get('err')}")
        # step window at all steps
        for w in (3, 6):
            errs, gms, _, _, _ = evaluate(L, 32, 1.0, ks=(5,), alignw=w, basis=bb)
            A[f"{lname}_pca32_st1_mean5_alignw{w}"] = summarize(errs[5], gms[5])
            log(f"[{cell}] A {lname} pca32 st1 mean5 |dstep|<= {w}: " + " ".join(f"{rn}={v['err']:.3f}" for rn, v in A[f'{lname}_pca32_st1_mean5_alignw{w}'].items()))
        # success-only candidates
        errs, gms, _, _, _ = evaluate(L, 32, 1.0, ks=(5,), basis=bb, mask=L.success)
        A[f"{lname}_pca32_st1_mean5_succonly"] = summarize(errs[5], gms[5])
        log(f"[{cell}] A {lname} pca32 st1 mean5 success-only lib: " + " ".join(f"{rn}={v['err']:.3f}" for rn, v in A[f'{lname}_pca32_st1_mean5_succonly'].items()))
        # vision-only (st0) top1 vs state-only reference for the vision-contrast
    # state-only reference (st only, no vision): use score = -D
    for lname, L in (("current", Lcur), ("big", Lbig)):
        errs = np.zeros(N)
        for tq in np.unique(task):
            qi = np.nonzero(task == tq)[0]
            ci = np.nonzero(L.task == tq)[0]
            q2 = np.einsum("ij,ij->i", rs[qi], rs[qi])[:, None]
            c2 = np.einsum("ij,ij->i", L.rs[ci], L.rs[ci])[None, :]
            Dq = np.sqrt(np.maximum(q2 - 2.0 * (rs[qi] @ L.rs[ci].T) + c2, 0.0))
            o = topk_rows(-Dq, 5)
            ah = L.head[ci[o]].mean(1)
            errs[qi] = err_of_rows(ah, qi)
        A[f"{lname}_stateonly_mean5"] = summarize(errs)
        log(f"[{cell}] A {lname} state-only mean5: " + " ".join(f"{rn}={v['err']:.3f}" for rn, v in A[f'{lname}_stateonly_mean5'].items()))
    results["A"] = A

    # ================================================================ B. confidence features (stale + fresh), pca32 st1 mean5
    B = {}
    keep = ["s0", "s5", "cos0_0", "cos1_0", "cosmax0", "cosmax1", "d0", "dnn", "d5", "disp5", "lag5", "lagfrac5",
            "libmotion5", "libstep5", "succ5", "libprog5", "ncand"]
    for lname, L in (("current", Lcur), ("big", Lbig)):
        bb = "big" if lname == "big" else "cur"
        errs, gms, TOP, SC, F = evaluate(L, 32, 1.0, ks=(5,), keep=keep, basis=bb)
        err5 = errs[5]
        # overtime: query step / library median episode length of the task (online-legal)
        med_len = {int(t): float(np.median(L.ep_len[L.task == t])) for t in np.unique(L.task)}
        overtime = step / np.asarray([med_len[int(t)] for t in task])
        # history-based features
        motion_q = np.zeros(N)
        motion_q[prev_idx >= 0] = np.linalg.norm(rs[prev_idx >= 0] - rs[prev_idx[prev_idx >= 0]], axis=1)
        # visual self-change (cosine between consecutive query keys, pca32 task-centred is not needed; plain)
        vself = np.ones(N)
        pi = prev_idx >= 0
        for f in ("v0", "v1"):
            Zq = unit(QP[f][:, :32])
            vself[pi] = np.minimum(vself[pi], np.einsum("ij,ij->i", Zq[pi], Zq[prev_idx[pi]]))
        # stuck run length: consecutive decisions with motion below a library-derived threshold
        thr = np.percentile(L.motion_next[L.has_next], 10)   # library's own slowest decile of motion
        stuck = np.zeros(N)
        for i in range(N):
            if prev_idx[i] >= 0 and motion_q[i] < thr:
                stuck[i] = stuck[prev_idx[i]] + 1
        # window distance (m=3) to library windows
        win_d = np.full(N, np.nan)
        Lprev = L.prev
        has2 = (Lprev >= 0)
        has3 = has2.copy()
        has3[has2] = Lprev[Lprev[has2]] >= 0
        for tq in np.unique(task):
            qi = np.nonzero((task == tq) & (step >= 2))[0]
            if qi.size == 0:
                continue
            ci = np.nonzero((L.task == tq) & has3)[0]
            if ci.size == 0:
                continue
            W_c = np.concatenate([L.rs[ci], L.rs[Lprev[ci]], L.rs[Lprev[Lprev[ci]]]], 1)
            p1 = prev_idx[qi]; p2 = prev_idx[p1]
            W_q = np.concatenate([rs[qi], rs[p1], rs[p2]], 1)
            q2 = np.einsum("ij,ij->i", W_q, W_q)[:, None]
            c2 = np.einsum("ij,ij->i", W_c, W_c)[None, :]
            Dw = np.sqrt(np.maximum(q2 - 2.0 * (W_q @ W_c.T) + c2, 0.0))
            win_d[qi] = Dw.min(1)
        win_d[np.isnan(win_d)] = F["dnn"][np.isnan(win_d)]
        # library 1-NN medians (LOEO within task, other episodes) for scales
        d1 = []
        for tq in np.unique(L.task):
            ci = np.nonzero(L.task == tq)[0]
            sel = ci[::max(1, ci.size // 200)]
            q2 = np.einsum("ij,ij->i", L.rs[sel], L.rs[sel])[:, None]
            c2 = np.einsum("ij,ij->i", L.rs[ci], L.rs[ci])[None, :]
            Dq = np.sqrt(np.maximum(q2 - 2.0 * (L.rs[sel] @ L.rs[ci].T) + c2, 0.0))
            Dq[L.ep[sel][:, None] == L.ep[ci][None, :]] = np.inf
            d1.append(Dq.min(1))
        s_d = float(np.median(np.concatenate(d1)))
        motion_ratio = motion_q / np.maximum(F["libmotion5"], 1e-6)
        feats = {
            "s0": F["s0"], "s5": F["s5"], "cos0_0": F["cos0_0"], "cos1_0": F["cos1_0"],
            "cosmax_sum": F["cosmax0"] + F["cosmax1"],
            "-d0": -F["d0"], "-dnn": -F["dnn"], "-d5": -F["d5"], "-disp5": -F["disp5"],
            "-|lag5|": -np.abs(F["lag5"]), "-lag5": -F["lag5"], "-|lagfrac5|": -np.abs(F["lagfrac5"]),
            "-stuck": -stuck, "motion_q": motion_q, "motion_ratio": np.minimum(motion_ratio, 3.0),
            "-vself": -vself, "-win_d": -win_d, "succ5": F["succ5"], "-libprog5": -F["libprog5"],
            "-step": -step.astype(float), "-overtime": -overtime,
        }
        Bl = {"s_d": s_d, "motion_thr": float(thr)}
        for rn, r in (("stale", 2), ("fresh", 1), ("step0", 0)):
            m = regime == r
            if m.sum() < 10:
                continue
            e = err5[m]
            row = {"opt": aurc(-e, e), "n": int(m.sum()), "err": float(e.mean())}
            for nm, v in feats.items():
                vv = v[m].astype(np.float64)
                vv = np.where(np.isfinite(vv), vv, np.nanmin(vv))
                row[nm] = aurc(vv, e)
            # combos (z over the regime's decisions -- scale only; sign fixed a priori)
            def z(x):
                x = x[m].astype(np.float64); x = np.where(np.isfinite(x), x, np.nanmin(x))
                s = x.std(); return (x - x.mean()) / (s if s > 1e-12 else 1.0)
            combos = {
                "z(s0)+z(-d0)+z(-disp5)": z(feats["s0"]) + z(feats["-d0"]) + z(feats["-disp5"]),
                "z(-dnn)+z(-disp5)": z(feats["-dnn"]) + z(feats["-disp5"]),
                "z(s0)+z(-dnn)+z(-disp5)+z(motion_ratio)": z(feats["s0"]) + z(feats["-dnn"]) + z(feats["-disp5"]) + z(feats["motion_ratio"]),
                "z(s0)+z(-dnn)+z(-disp5)+z(-stuck)": z(feats["s0"]) + z(feats["-dnn"]) + z(feats["-disp5"]) + z(feats["-stuck"]),
                "z(s0)+z(-win_d)+z(-disp5)+z(-stuck)+z(-|lag5|)": z(feats["s0"]) + z(feats["-win_d"]) + z(feats["-disp5"]) + z(feats["-stuck"]) + z(feats["-|lag5|"]),
                "z(-win_d)+z(-disp5)+z(-stuck)": z(feats["-win_d"]) + z(feats["-disp5"]) + z(feats["-stuck"]),
                "z(s0)+z(-win_d)+z(-disp5)+z(-stuck)+z(motion_ratio)": z(feats["s0"]) + z(feats["-win_d"]) + z(feats["-disp5"]) + z(feats["-stuck"]) + z(feats["motion_ratio"]),
                "z(s0)+z(-win_d)+z(-disp5)+z(-stuck)+z(motion_ratio)+z(-|lag5|)": z(feats["s0"]) + z(feats["-win_d"]) + z(feats["-disp5"]) + z(feats["-stuck"]) + z(feats["motion_ratio"]) + z(feats["-|lag5|"]),
                "z(s0)+z(-win_d)+z(-disp5)+z(-stuck)+z(motion_ratio)+z(-|lag5|)+z(-step)": z(feats["s0"]) + z(feats["-win_d"]) + z(feats["-disp5"]) + z(feats["-stuck"]) + z(feats["motion_ratio"]) + z(feats["-|lag5|"]) + z(feats["-step"]),
                "z(-dnn)+z(-disp5)+z(-overtime)": z(feats["-dnn"]) + z(feats["-disp5"]) + z(feats["-overtime"]),
                "z(-dnn)+z(-disp5)+z(-overtime)+z(cosmax)": z(feats["-dnn"]) + z(feats["-disp5"]) + z(feats["-overtime"]) + z(feats["cosmax_sum"]),
                "z(-win_d)+z(-disp5)+z(-overtime)+z(cosmax)": z(feats["-win_d"]) + z(feats["-disp5"]) + z(feats["-overtime"]) + z(feats["cosmax_sum"]),
                "z(-win_d)+z(-disp5)+z(-overtime)+z(cosmax)+z(-stuck)": z(feats["-win_d"]) + z(feats["-disp5"]) + z(feats["-overtime"]) + z(feats["cosmax_sum"]) + z(feats["-stuck"]),
                "z(-dnn)+z(cosmax)": z(feats["-dnn"]) + z(feats["cosmax_sum"]),
                "z(-win_d)+z(-overtime)": z(feats["-win_d"]) + z(feats["-overtime"]),
                "z(-dnn)+z(-disp5)+z(-|lag5|)": z(feats["-dnn"]) + z(feats["-disp5"]) + z(feats["-|lag5|"]),
            }
            for nm, v in combos.items():
                row[nm] = aurc(v, e)
            # slices: stuck / drifted decisions and their err, and the confidence rank they get
            if r == 2:
                drift = win_d[m] > 3 * s_d
                row["frac_drift3x"] = float(drift.mean()); row["err_drift"] = float(e[drift].mean()) if drift.any() else None
                row["err_nodrift"] = float(e[~drift].mean()) if (~drift).any() else None
                st2 = stuck[m] >= 2
                row["frac_stuck>=2"] = float(st2.mean()); row["err_stuck>=2"] = float(e[st2].mean()) if st2.any() else None
                row["err_notstuck"] = float(e[~st2].mean()) if (~st2).any() else None
                mr = motion_ratio[m] < 0.3
                row["frac_motion_ratio<.3"] = float(mr.mean()); row["err_mr<.3"] = float(e[mr].mean()) if mr.any() else None
                lg = np.abs(F["lag5"][m]) > 5
                row["frac_|lag5|>5"] = float(lg.mean()); row["err_lag>5"] = float(e[lg].mean()) if lg.any() else None
                # cross-fitted (by episode parity) isotonic-like binned predicted err from a 4-feature combo
                ov = overtime[m] > 1.0
                row["frac_overtime>1"] = float(ov.mean()); row["err_overtime>1"] = float(e[ov].mean()) if ov.any() else None
                row["err_overtime<=1"] = float(e[~ov].mean()) if (~ov).any() else None
                X = np.stack([z(feats["s0"]), z(feats["-win_d"]), z(feats["-disp5"]), z(feats["-stuck"]), z(feats["motion_ratio"]), z(feats["-|lag5|"]), z(feats["-overtime"]), z(feats["cosmax_sum"])], 1)
                epm = ep[m]
                pred = np.zeros(m.sum())
                for par in (0, 1):
                    tr = (epm % 2) == par; te = ~tr
                    Xa = np.c_[X[tr], np.ones(tr.sum())]
                    w, *_ = np.linalg.lstsq(Xa, e[tr], rcond=None)
                    pred[te] = np.c_[X[te], np.ones(te.sum())] @ w
                row["xfit_linear8_predicted_err"] = aurc(-pred, e)
                row["xfit_corr"] = float(np.corrcoef(pred, e)[0, 1])
            Bl[rn] = row
        B[lname] = Bl
        log(f"[{cell}] B {lname} stale AURC: " + " ".join(f"{k}={v:.3f}" for k, v in Bl.get("stale", {}).items() if isinstance(v, float) and k not in ("err",)))
    results["B"] = B

    # ================================================================ C. flip-aware synthesis (pca32 st1, k in 5/8, big + current)
    C = {}
    for lname, L in (("current", Lcur), ("big", Lbig)):
        _, _, TOP, SC, _ = evaluate(L, 32, 1.0, ks=(8,), basis=("big" if lname == "big" else "cur"))
        for k in (5, 8):
            rows = TOP[:, :k]
            H = L.head[rows]                        # [N,k,5,7]
            G = np.where(H[:, :, :, 6] >= 0, 1.0, -1.0)   # [N,k,5]
            w = np.exp(SC[:, :k] - SC[:, :1])           # score kernel weights (relative to top-1)
            variants = {}
            variants["mean"] = H.mean(1)
            a = H.mean(1).copy(); a[:, :, 6] = np.where(a[:, :, 6] >= 0, 1.0, -1.0); variants["mean_snap"] = a
            # per-step gripper majority, dims 0..5 mean of all
            a = H.mean(1).copy(); a[:, :, 6] = np.where(G.sum(1) >= 0, 1.0, -1.0); variants["mean_gmaj"] = a
            # cluster by end-gripper sign (step 4): majority cluster -> mean within cluster
            end = G[:, :, 4]
            maj = np.where(end.sum(1) >= 0, 1.0, -1.0)
            inm = (end == maj[:, None]).astype(np.float64)
            a = (H * inm[:, :, None, None]).sum(1) / inm.sum(1)[:, None, None]; variants["cluster_end"] = a
            # cluster by full 5-step gripper pattern (most frequent pattern; ties -> pattern of top-1)
            pat = ((G > 0) * (2 ** np.arange(5))[None, None, :]).sum(2)   # [N,k] int
            a = np.zeros_like(H[:, 0])
            for i in range(N):
                vals, cnt = np.unique(pat[i], return_counts=True)
                best = vals[cnt == cnt.max()]
                p = pat[i, 0] if pat[i, 0] in best else best[0]
                sel = pat[i] == p
                a[i] = H[i, sel].mean(0)
            variants["cluster_pattern"] = a
            # score-weighted cluster (kernel weights) by end gripper
            wm = np.where((w * end).sum(1) >= 0, 1.0, -1.0)
            inm = (end == wm[:, None]).astype(np.float64) * w
            a = (H * inm[:, :, None, None]).sum(1) / inm.sum(1)[:, None, None]; variants["cluster_end_w"] = a
            # cluster by pattern of the executed segment, and gripper follows the cluster (already), plus prev-gripper prior:
            # if the candidates disagree, prefer the cluster consistent with gprev (no flip) unless flip cluster has >= 60%
            flipfrac = (end != gprev[:, None]).mean(1)
            choose = np.where(flipfrac >= 0.6, -gprev, gprev)
            inm = (end == choose[:, None]).astype(np.float64)
            ok = inm.sum(1) > 0
            a = variants["mean"].copy()
            a[ok] = (H[ok] * inm[ok][:, :, None, None]).sum(1) / inm[ok].sum(1)[:, None, None]
            variants["cluster_end_prior60"] = a
            Ck = {}
            for nm, ah in variants.items():
                e = err_of(ah); g = gmis_of(ah)
                row = {}
                for rn, r in (("step0", 0), ("fresh", 1), ("stale", 2)):
                    m = regime == r
                    if m.sum() == 0:
                        continue
                    row[rn] = {"err": float(e[m].mean()), "gmis": float(g[m].mean()),
                               "err_trans": float(e[m & trans].mean()) if (m & trans).any() else None,
                               "gmis_trans": float(g[m & trans].mean()) if (m & trans).any() else None,
                               "err_steady": float(e[m & ~trans].mean()), "gmis_steady": float(g[m & ~trans].mean()),
                               "frac_trans": float(trans[m].mean())}
                Ck[nm] = row
            C[f"{lname}_k{k}"] = Ck
            log(f"[{cell}] C {lname} k{k}: " + " | ".join(
                f"{nm} stale err {v.get('stale', {}).get('err', float('nan')):.3f} gm {v.get('stale', {}).get('gmis', float('nan')):.3f} "
                f"trans {v.get('stale', {}).get('err_trans') or float('nan'):.3f}" for nm, v in Ck.items()))
    results["C"] = C

    # ================================================================ D. fresh regime: continuity + vision fusion over ALL candidates
    D = {}
    if arm == "inf":
        for lname, L in (("big", Lbig), ("current", Lcur)):
            HD = (L.head / sig).reshape(L.L, 35)
            h2 = np.einsum("ij,ij->i", HD, HD)
            # library scale s_c: median 1-NN tail->head continuity across other episodes (subsample)
            TL = (L.tail / sig).reshape(L.L, 35)
            cs = []
            for tq in np.unique(L.task):
                ci = np.nonzero(L.task == tq)[0]
                sel = ci[::max(1, ci.size // 150)]
                Cq = np.sqrt(np.maximum(np.einsum("ij,ij->i", TL[sel], TL[sel])[:, None] - 2 * TL[sel] @ HD[ci].T + h2[ci][None, :], 0) / 35)
                Cq[L.ep[sel][:, None] == L.ep[ci][None, :]] = np.inf
                cs.append(Cq.min(1))
            s_c = float(np.median(np.concatenate(cs)))
            Dl = {"s_c": s_c}
            for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
                for alpha in (0.0, 0.25):
                    errs = np.zeros(N); conf_c = np.zeros(N); conf_v = np.zeros(N); disp = np.zeros(N)
                    for tq in np.unique(task):
                        qi, ci, S, parts, Dq = score_task(L, tq, 32, 1.0, basis=("big" if lname == "big" else "cur"))   # S = zv0 + zv1 + z(-d)
                        V = S - zrows(-Dq)                                    # vision only z-sum
                        T = (tail[qi] / sig).reshape(qi.size, 35)
                        Cc = np.sqrt(np.maximum(np.einsum("ij,ij->i", T, T)[:, None] - 2 * T @ HD[ci].T + h2[ci][None, :], 0) / 35)
                        F = -Cc / s_c + lam * V / 2.0 + alpha * zrows(-Dq)
                        o = topk_rows(F, 5)
                        rows = ci[o]
                        ah = L.head[rows].mean(1)
                        errs[qi] = err_of_rows(ah, qi)
                        conf_c[qi] = -np.take_along_axis(Cc, o[:, :1], 1)[:, 0]
                        conf_v[qi] = np.take_along_axis(V, o[:, :1], 1)[:, 0]
                        H5 = L.head[rows].reshape(qi.size, 5, -1) / np.tile(sig, 5)[None, None, :]
                        iu = np.triu_indices(5, 1)
                        Dd = H5[:, iu[0]] - H5[:, iu[1]]
                        disp[qi] = np.sqrt((Dd * Dd).mean(2)).mean(1)
                    m = regime == 1
                    e = errs[m]
                    def z(x):
                        x = x[m]; s = x.std(); return (x - x.mean()) / (s if s > 1e-12 else 1.0)
                    Dl[f"lam{lam:g}_a{alpha:g}"] = {
                        "err_fresh": float(e.mean()), "p50": float(np.median(e)),
                        "aurc_cont": aurc(conf_c[m], e), "aurc_cont_disp": aurc(z(conf_c) - z(disp), e),
                        "aurc_cont_disp_vis": aurc(z(conf_c) - z(disp) + z(conf_v), e), "opt": aurc(-e, e),
                        "err_trans": float(errs[m & trans].mean()), "err_steady": float(errs[m & ~trans].mean())}
                    log(f"[{cell}] D {lname} lam{lam:g} a{alpha:g}: fresh err {e.mean():.3f} aurc cont {Dl[f'lam{lam:g}_a{alpha:g}']['aurc_cont']:.3f} +disp {Dl[f'lam{lam:g}_a{alpha:g}']['aurc_cont_disp']:.3f} +vis {Dl[f'lam{lam:g}_a{alpha:g}']['aurc_cont_disp_vis']:.3f}")
            D[lname] = Dl
    results["D"] = D

    # ================================================================ E. timing (single thread, pca32 st1 mean5, per query)
    E = {}
    for lname, L in (("current", Lcur), ("big", Lbig)):
        bb = "big" if lname == "big" else "cur"
        LP = L.P if bb == "big" else L.Pc
        pcab = pca if bb == "big" else pca_cur
        pre = {}
        for tq in np.unique(L.task):
            ci = np.nonzero(L.task == tq)[0]
            d = {}
            for f in ("v0", "v1"):
                Pc = LP[f][ci, :32]; m = Pc.mean(0); d[f] = (np.ascontiguousarray(unit(Pc - m), np.float32), m.astype(np.float32))
            d["rs"] = np.ascontiguousarray(L.rs[ci]); d["rs2"] = np.einsum("ij,ij->i", d["rs"], d["rs"])
            d["head"] = np.ascontiguousarray(L.head[ci]); d["ci"] = ci
            pre[tq] = d
        Bn = {f: (pcab[f][0], np.ascontiguousarray(pcab[f][1][:, :32])) for f in ("v0", "v1")}          # [D,32]
        Bt = {f: (pcab[f][0], np.ascontiguousarray(pcab[f][1][:, :32].T)) for f in ("v0", "v1")}        # [32,D]
        K0 = np.load(qd / "key_v0.npy", mmap_mode="r"); K1 = np.load(qd / "key_v1.npy", mmap_mode="r")
        rng = np.random.default_rng(0)
        sel = rng.choice(N, size=min(300, N), replace=False)
        for i in sel[:50]:
            _ = np.asarray(K0[i]); _ = np.asarray(K1[i])
        tp_n, tp_t, tsc = [], [], []
        for i in sel:
            k0 = np.asarray(K0[i], np.float32); k1 = np.asarray(K1[i], np.float32); r = rs[i]
            d = pre[task[i]]
            # projection, layout [D,32]
            t1 = time.perf_counter_ns()
            xs = {}
            for f, kq in (("v0", k0), ("v1", k1)):
                mu, Bm = Bn[f]
                xs[f] = (kq - mu) @ Bm
            t2 = time.perf_counter_ns()
            # projection, layout [32,D]
            for f, kq in (("v0", k0), ("v1", k1)):
                mu, Bm = Bt[f]
                xs[f] = Bm @ (kq - mu)
            t3 = time.perf_counter_ns()
            S = np.zeros(d["ci"].size, np.float64)
            for f in ("v0", "v1"):
                x = xs[f] - d[f][1]
                x /= max(float(np.linalg.norm(x)), 1e-8)
                c = d[f][0] @ x
                S += (c - c.mean()) / max(c.std(), 1e-12)
            dv = np.sqrt(np.maximum(d["rs2"] - 2.0 * (d["rs"] @ r) + float(r @ r), 0))
            S += (-dv - (-dv).mean()) / max(dv.std(), 1e-12)
            o = np.argpartition(-S, 4)[:5]
            o = o[np.argsort(-S[o], kind="stable")]
            ah = d["head"][o].mean(0)
            t4 = time.perf_counter_ns()
            tp_n.append((t2 - t1) / 1e3); tp_t.append((t3 - t2) / 1e3); tsc.append((t4 - t3) / 1e3)
        E[lname] = {"proj_us_D32": float(np.median(tp_n)), "proj_us_32D": float(np.median(tp_t)),
                    "score_us_p50": float(np.median(tsc)), "score_us_p95": float(np.percentile(tsc, 95)),
                    "cands_mean": float(np.mean([pre[t]["ci"].size for t in pre])),
                    "threads": os.environ.get("OMP_NUM_THREADS")}
        log(f"[{cell}] E {lname}: {E[lname]}")
    results["E"] = E
    results["wall_s"] = time.time() - t0
    (out / f"diag_{cell}.json").write_text(json.dumps(results, indent=1))
    log(f"[{cell}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
