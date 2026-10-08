"""Library-only per-task camera signal: held-out B-pool episodes (not in the served subset) re-served by the frozen
ablation artifacts' layer-1 retrieval, both (deployed joint metric) vs wrist_sub (restricted principal block).
Uncorrected kernel-mean chunk vs the held-out row's own stored policy chunk: e10 (RMS over 10 steps x 6 motion
channels in sigma units) and first-5-step gripper-sign mismatch. Regime-2 distance (step > 0) / early (step 0).
Reads only offline_search_store/library (B pool) and the B-only ablation artifacts."""
import json, pickle, sys
from pathlib import Path
import numpy as np
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r10.analysis_opus.tools import core

ART = Path("/home/weiland/projects/openpi/exp/offline_search/rounds/r12/ablate/artifacts")
OUT = Path("/home/weiland/projects/openpi/exp/offline_search/rounds/r12/opus/out/design")
CELLS = [("pi05", "l10", 50), ("pi05", "spatial", 50), ("pi05", "spatial", 200), ("groot", "spatial", 50), ("groot", "l10", 50)]
EPS_PER_TASK, STEP_EVERY = 25, 4


def run(model, suite, size):
    cell = f"{model}_{suite}_{size}"
    W = pickle.load(open(ART / cell / "wrist_sub.pkl", "rb"))["method"]
    A = pickle.load(open(ART / cell / "agent_sub.pkl", "rb"))["method"]
    inn, inA = W.inner, A.inner
    sub_rows = np.asarray(W.row_subset)
    C = core.load_cell(model, suite)
    nrs = inn.camera_metrics[0].restricted_X.shape[1] - 64
    sig = np.asarray(inn.sig, np.float64).reshape(-1)[:7]
    rng = np.random.default_rng(0)
    in_sub = np.zeros(len(C["task"]), bool); in_sub[sub_rows] = True
    sub_eps = set(C["ep"][sub_rows].tolist())
    res = {}
    for t in range(10):
        Tk = inn.tasks[t]
        Xw = inn.camera_metrics[t].restricted_X            # [P1, rs] of library rows of task t
        Xa = inA.camera_metrics[t].restricted_X            # [P0, rs]
        Xf = np.c_[Xa[:, :64], Xw]                         # [P0, P1, rs]
        Wf, W0f = Tk.Wf.astype(np.float64), Tk.W0f.astype(np.float64)
        Mw, Mw0 = inn.camera_metrics[t].transform, inn.camera_metrics[t].early_transform
        lib_act = np.asarray(inn.act[Tk.rows][:, :10, :7], np.float64)
        eps = sorted({int(e) for e in C["ep"][(C["task"] == t) & ~in_sub]} - sub_eps)
        eps = sorted(rng.choice(eps, min(EPS_PER_TASK, len(eps)), replace=False).tolist())
        q = np.flatnonzero(np.isin(C["ep"], eps) & (C["step"] % STEP_EVERY == 0))
        q = np.sort(q)
        K0, K1 = core.gather_keys(model, suite, q)
        P0 = K0 @ inn.B0T.T - inn.muB0; P1 = K1 @ inn.B1T.T - inn.muB1
        rs = C["rs"][q][:, :nrs].astype(np.float64)
        xf = np.c_[P0, P1, rs].astype(np.float64); xw = np.c_[P1, rs].astype(np.float64)
        st = C["step"][q]; gt = C["act"][q][:, :10, :7].astype(np.float64)
        out = {k: [] for k in ("e_f", "e_w", "g_f", "g_w", "dF", "top1_same", "d1_f", "d1_w", "succ", "step", "ep")}
        for i in range(len(q)):
            early = st[i] == 0
            df = np.sqrt((((xf[i] - Xf) @ (W0f if early else Wf)) ** 2).sum(1))
            dw = np.sqrt((((xw[i] - Xw) @ (Mw0 if early else Mw)) ** 2).sum(1))
            ch = {}
            for name, d in (("f", df), ("w", dw)):
                k = min(inn.k, len(d)); idx = np.argpartition(d, k - 1)[:k]; idx = idx[np.lexsort((idx, d[idx]))]
                dk = d[idx]; w = _kernel_w(dk - dk[0], inn.kref); w = w / w.sum()
                a = np.tensordot(w, lib_act[idx], 1)
                ch[name] = (a, idx[0], dk[0])
            for name in ("f", "w"):
                a = ch[name][0]
                out[f"e_{name}"].append(np.sqrt(np.mean(((a[:, :6] - gt[i][:, :6]) / sig[:6]) ** 2)))
                out[f"g_{name}"].append(np.mean(np.sign(a[:5, 6]) != np.sign(gt[i][:5, 6])))
                out[f"d1_{name}"].append(ch[name][2])
            out["dF"].append(np.sqrt(np.mean(((ch["w"][0][:, :6] - ch["f"][0][:, :6]) / sig[:6]) ** 2)))
            out["top1_same"].append(ch["w"][1] == ch["f"][1])
            out["succ"].append(bool(C["success"][q[i]])); out["step"].append(int(st[i])); out["ep"].append(int(C["ep"][q[i]]))
        res[t] = {k: np.asarray(v) for k, v in out.items()}
        r = res[t]
        print(f"{cell} task {t}: n={len(q)} eps={len(eps)} e_f {r['e_f'].mean():.3f} e_w {r['e_w'].mean():.3f} "
              f"de {np.mean(r['e_w']-r['e_f']):+.4f} dg {np.mean(r['g_w']-r['g_f']):+.4f} dF {r['dF'].mean():.3f} "
              f"top1same {r['top1_same'].mean():.2f}", flush=True)
    np.savez(OUT / f"libsignal_{cell}.npz", **{f"t{t}_{k}": v for t in res for k, v in res[t].items()})


if __name__ == "__main__":
    which = sys.argv[1:] or [f"{m}_{s}_{n}" for m, s, n in CELLS]
    for m, s, n in CELLS:
        if f"{m}_{s}_{n}" in which:
            run(m, s, n)
