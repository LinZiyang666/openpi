"""state weight sweep, kernel-weighted mean, gripper hysteresis + chatter proxy (cache arms, both libraries)."""
import json, pathlib, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import diag as D
cell = sys.argv[1]; model, suite, arm = cell.split("_"); key = f"{model}_{suite}"
cache_dir = D.OUT / "cache"; t0 = time.time()
pca_cur = D.fit_pca_current(key, 32, cache_dir)
pca = {f: (np.load(D.PCA_DIR / key / D.BIG[model] / f / "mean.npy").astype(np.float32),
           np.ascontiguousarray(np.load(D.PCA_DIR / key / D.BIG[model] / f / "basis.npy", mmap_mode="r")[:, :32], np.float32)) for f in ("v0", "v1")}
libs = {"current": D.load_lib(key, "current", pca, 32, cache_dir, pca_cur), "big": D.load_lib(key, D.BIG[model], pca, 32, cache_dir, pca_cur)}
qd = D.ROOT / "queries" / cell
eps_json = json.load(open(qd / "episodes.json"))
ep = np.asarray(np.load(qd / "ep.npy"), np.int64); step = np.asarray(np.load(qd / "step.npy"), np.int64); N = ep.size
task = np.asarray([e["task_id"] for e in eps_json], np.int64)[ep]
rs = np.asarray(np.load(qd / "rs.npy", mmap_mode="r")[:, :8], np.float32)
a_inf = np.asarray(np.load(qd / "a_inf.npy", mmap_mode="r")[:, :5, :7], np.float32)
a_exec = np.asarray(np.load(qd / "a_exec.npy", mmap_mode="r")[:, :10, :7], np.float32)
QP = {"big": {f: np.load(cache_dir / f"qproj_{cell}_{f}.npy")[:, :32] for f in ("v0", "v1")},
      "cur": {f: np.load(cache_dir / f"qprojcur_{cell}_{f}.npy")[:, :32] for f in ("v0", "v1")}}
sigma = np.asarray(libs["current"].act[:, :5, :7], np.float64).std(axis=(0, 1))
prev_idx = np.full(N, -1, np.int64); same = np.r_[False, (ep[1:] == ep[:-1]) & (step[1:] == step[:-1] + 1)]; prev_idx[same] = np.arange(N)[same] - 1
pi = prev_idx >= 0
gprev = np.ones(N); gprev[pi] = np.where(a_exec[prev_idx[pi], 4, 6] >= 0, 1.0, -1.0)
stale = step >= 1; s0 = step == 0
gi = np.where(a_inf[:, :, 6] >= 0, 1.0, -1.0)
# teacher gripper stable across consecutive decisions (no flip in truth): a_inf prev end sign == a_inf now all steps
teach_stable = np.zeros(N, bool)
teach_stable[pi] = (gi[pi] == gi[prev_idx[pi], 4:5]).all(1)

def err_of(ah):
    d = (ah.astype(np.float64) - a_inf) / sigma; return np.sqrt((d * d).mean(axis=(1, 2)))
def gmis_of(ah): return ((ah[:, :, 6] >= 0) != (a_inf[:, :, 6] >= 0)).mean(1)
def topk(S, k):
    k = min(k, S.shape[1]); o = np.argpartition(-S, k - 1, axis=1)[:, :k]; so = np.take_along_axis(S, o, 1)
    return np.take_along_axis(o, np.argsort(-so, axis=1, kind="stable"), 1)

res = {"cell": cell}
for lname, L in libs.items():
    b = "big" if lname == "big" else "cur"; LP = L.P if b == "big" else L.Pc; Q = QP[b]
    out = {}
    for st in (0.5, 1.0, 2.0, 3.0):
        TOP = np.zeros((N, 8), np.int64); SC = np.zeros((N, 8))
        for tq in np.unique(task):
            qi = np.nonzero(task == tq)[0]; ci = np.nonzero(L.task == tq)[0]
            S = np.zeros((qi.size, ci.size))
            for f in ("v0", "v1"):
                Pc = LP[f][ci]; m = Pc.mean(0); S += D.zrows(D.unit(Q[f][qi] - m) @ D.unit(Pc - m).T)
            q2 = np.einsum("ij,ij->i", rs[qi], rs[qi])[:, None]; c2 = np.einsum("ij,ij->i", L.rs[ci], L.rs[ci])[None, :]
            Dq = np.sqrt(np.maximum(q2 - 2.0 * (rs[qi] @ L.rs[ci].T) + c2, 0.0)); S += st * D.zrows(-Dq)
            o = topk(S, 8); TOP[qi] = ci[o]; SC[qi] = np.take_along_axis(S, o, 1)
        H = L.head[TOP]
        row = {}
        for k in (5, 8):
            ah = H[:, :k].mean(1); e = err_of(ah)
            row[f"mean{k}"] = {"stale": float(e[stale].mean()), "step0": float(e[s0].mean()), "gmis_stale": float(gmis_of(ah)[stale].mean())}
            for T in (0.5, 1.0):
                w = np.exp((SC[:, :k] - SC[:, :1]) / T); w /= w.sum(1, keepdims=True)
                ah = (H[:, :k] * w[:, :, None, None]).sum(1); e = err_of(ah)
                row[f"kernel{k}_T{T:g}"] = {"stale": float(e[stale].mean()), "step0": float(e[s0].mean()), "keff_p50": float(np.median(1 / (w ** 2).sum(1)))}
        if st == 1.0:
            # gripper hysteresis on mean5: keep gprev unless flip fraction among top-5 end-grippers >= thr
            G = np.where(H[:, :5, :, 6] >= 0, 1.0, -1.0); end = G[:, :, 4]
            base = H[:, :5].mean(1)
            for thr in (0.5, 0.6, 0.7):
                flipfrac = (end != gprev[:, None]).mean(1)
                gsign = np.where(flipfrac >= thr, -gprev, gprev)
                ah = base.copy(); ah[:, :, 6] = gsign[:, None]          # whole executed block takes the decided sign
                e = err_of(ah); g = gmis_of(ah)
                # chatter proxy: synthesized end-sign differs from previous decision's synthesized end-sign while the teacher is stable
                mysign = gsign; prevsign = np.ones(N); prevsign[pi] = mysign[prev_idx[pi]]
                chat = (mysign != prevsign) & teach_stable & pi
                row[f"hyst{thr:g}"] = {"stale": float(e[stale].mean()), "gmis_stale": float(g[stale].mean()), "chatter_rate_stale": float(chat[stale].mean())}
            # reference chatter of plain mean sign
            msign = np.where(base[:, 4, 6] >= 0, 1.0, -1.0); prevsign = np.ones(N); prevsign[pi] = msign[prev_idx[pi]]
            chat = (msign != prevsign) & teach_stable & pi
            row["mean5_chatter_rate_stale"] = float(chat[stale].mean())
            # B0's own served gripper chatter (a_exec end sign vs previous) as the deployed reference
            bsign = np.where(a_exec[:, 4, 6] >= 0, 1.0, -1.0); prevb = np.ones(N); prevb[pi] = bsign[prev_idx[pi]]
            row["b0_served_chatter_rate_stale"] = float(((bsign != prevb) & teach_stable & pi)[stale].mean())
            row["teacher_flip_rate_stale"] = float((~teach_stable)[stale & pi].mean())
        out[f"st{st:g}"] = row
        print(f"[{cell}] {lname} st{st:g}: " + " ".join(f"{k}={v['stale']:.3f}/{v['step0']:.3f}" for k, v in row.items() if isinstance(v, dict) and 'step0' in v), flush=True)
    res[lname] = out
    print(f"[{cell}] {lname} hysteresis: " + json.dumps({k: v for k, v in out['st1'].items() if 'hyst' in k or 'chatter' in k or 'flip' in k}), flush=True)
(D.OUT / f"diag3_{cell}.json").write_text(json.dumps(res, indent=1))
print(f"[{cell}] done {time.time() - t0:.0f}s", flush=True)
