"""D3: recompute B0 field sims on a query subsample; test (a) per-task mu/sigma, (b) linear z-sum w/o tanh,
(c) continuity as a full-candidate field fused with B0, (d) cascade rs->vision, (e) tanh saturation stats. Read-only."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/home/weiland/projects/openpi")
import numpy as np
from exp.offline_search.harness import metrics, store

ROOT = "/dev/shm/offline_search_store"
RES = "/home/weiland/projects/openpi/exp/offline_search/results/r00"
CELLS = sys.argv[1].split(",") if len(sys.argv) > 1 else [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
EPS = np.float32(1e-8)


def zt(x, mu, sg):
    return np.float32(0.5) * (np.tanh((x - np.float32(mu)) / np.float32(sg)) + np.float32(1.0))


def aurc(err, conf):
    return metrics.risk_coverage(np.asarray(err, np.float64), np.asarray(conf, np.float64))["aurc"]


OUT = {}
for cell in CELLS:
    m, s, arm = cell.split("_"); key = f"{m}_{s}"
    p = store.current_params(m, s); w = np.array(p["weights"], np.float32); mu = p["mu"]; sg = p["sigma"]
    z = np.load(f"{RES}/B0_current/{cell}.npz"); N0 = len(z["row"])
    rng = np.random.default_rng(0); sel = np.sort(rng.choice(N0, size=min(NQ, N0), replace=False))
    row = z["row"][sel]; step = z["step"][sel].astype(int); qtask = z["task_id"][sel]; N = len(sel)
    Q = f"{ROOT}/queries/{cell}"; L = f"{ROOT}/library/{key}/current"
    K0 = np.load(f"{Q}/key_v0.npy", mmap_mode="r"); K1 = np.load(f"{Q}/key_v1.npy", mmap_mode="r")
    qrs = np.asarray(np.load(f"{Q}/rs.npy", mmap_mode="r")[row], np.float32)
    a_inf = np.load(f"{Q}/a_inf.npy", mmap_mode="r"); a_exec = np.load(f"{Q}/a_exec.npy", mmap_mode="r")
    LV0 = np.load(f"{L}/key_v0.npy", mmap_mode="r"); LV1 = np.load(f"{L}/key_v1.npy", mmap_mode="r")
    lrs = np.load(f"{L}/rs.npy"); ltask = np.load(f"{L}/task_id.npy").astype(int); A = metrics.seg(np.load(f"{L}/action.npy"))
    sigma = store.action_sigma(ROOT, key)
    gt = metrics.seg(a_inf[row])
    has_prev = step > 0; prow = np.where(has_prev, row - 1, row)
    tail = np.asarray(a_exec[prow], np.float64)[:, 5:10, :7]
    # per-task library stats of same-task pairs (library only): mu_t, sg_t per field
    stats = {}
    for t in np.unique(ltask):
        li = np.where(ltask == t)[0]
        V0 = np.asarray(LV0[li], np.float32); V1 = np.asarray(LV1[li], np.float32); R = lrs[li]
        n0 = np.linalg.norm(V0, axis=1); n1 = np.linalg.norm(V1, axis=1)
        C0 = (V0 @ V0.T) / np.outer(n0, n0); C1 = (V1 @ V1.T) / np.outer(n1, n1)
        D = np.sqrt(((R[:, None] - R[None]) ** 2).sum(-1))
        iu = np.triu_indices(len(li), 1)
        stats[t] = {"mu0": C0[iu].mean(), "sg0": C0[iu].std(), "mu1": C1[iu].mean(), "sg1": C1[iu].std(), "mur": (-D[iu]).mean(), "sgr": D[iu].std(),
                    "V0": V0, "n0": n0, "V1": V1, "n1": n1, "R": R, "rows": li}
    picks = {}; confs = {}; zstats = {"z0_top": [], "z1_top": [], "zr_top": [], "z0_all": [], "zr_all": []}
    names = ["B0", "pertask_zt", "linear_z", "linear_z_pertask", "pertask_zt_nors", "B0+cont", "linear_z+cont", "cont_only", "cascade_rs50_B0", "cascade_rs50_cont", "prod_zt", "min_zt", "B0_recal_top"]
    E_of = {n: np.zeros(N) for n in names}; conf_of = {n: np.zeros(N) for n in ["B0", "linear_z+cont", "B0+cont"]}
    for t in np.unique(qtask):
        qi = np.where(qtask == t)[0]; st = stats[t]
        q0 = np.asarray(K0[row[qi]], np.float32); q1 = np.asarray(K1[row[qi]], np.float32)
        c0 = (q0 @ st["V0"].T) / np.maximum(np.outer(np.linalg.norm(q0, axis=1), st["n0"]), EPS)
        c1 = (q1 @ st["V1"].T) / np.maximum(np.outer(np.linalg.norm(q1, axis=1), st["n1"]), EPS)
        d = np.sqrt(((qrs[qi][:, None] - st["R"][None]) ** 2).sum(-1))
        Eall = metrics.err_seg(A[st["rows"]][None], gt[qi][:, None], sigma)
        cont = metrics.err_seg(A[st["rows"]][None], tail[qi][:, None], sigma); cont[~has_prev[qi]] = 0.0
        s0 = zt(c0, mu[0], sg[0]); s1 = zt(c1, mu[1], sg[1]); s2 = zt(-d, mu[2], sg[2])
        fB0 = w[0] * s0 + w[1] * s1 + w[2] * s2
        # per-task recalibrated
        t0 = zt(c0, st["mu0"], st["sg0"]); t1 = zt(c1, st["mu1"], st["sg1"]); t2 = zt(-d, st["mur"], st["sgr"])
        fPT = w[0] * t0 + w[1] * t1 + w[2] * t2
        # linear z (no squash), global mu/sg
        z0 = (c0 - mu[0]) / sg[0]; z1 = (c1 - mu[1]) / sg[1]; zr = (-d - mu[2]) / sg[2]
        fLZ = w[0] * z0 + w[1] * z1 + w[2] * zr
        zz0 = (c0 - st["mu0"]) / st["sg0"]; zz1 = (c1 - st["mu1"]) / st["sg1"]; zzr = (-d - st["mur"]) / st["sgr"]
        fLZT = w[0] * zz0 + w[1] * zz1 + w[2] * zzr
        fPT_nors = w[0] * t0 + w[1] * t1
        # continuity as a field: z-scored by a fixed scale (per-cell median of the per-query min continuity), weight 1 vs the sum of B0 weights (=1)
        cz = -cont / (np.median(cont[has_prev[qi]].min(1)) + 1e-9) if has_prev[qi].any() else -cont
        fB0C = fB0 + 0.5 * np.where(has_prev[qi][:, None], cz, 0.0)
        fLZC = fLZ / 3.0 + np.where(has_prev[qi][:, None], cz, 0.0)
        fC = np.where(has_prev[qi][:, None], -cont, fB0)
        # cascade: rs top-50 (or all if fewer), then B0 / cont
        M = min(50, d.shape[1]); o = np.argsort(d, 1)[:, :M]; mask = np.full(d.shape, -np.inf); np.put_along_axis(mask, o, 0.0, 1)
        fCasB0 = fB0 + mask; fCasC = fC + mask
        fPR = np.log(s0 + 1e-6) * w[0] + np.log(s1 + 1e-6) * w[1] + np.log(s2 + 1e-6) * w[2]
        fMIN = np.minimum(np.minimum(s0, s1), s2)
        # z stats of the B0 top-10 candidates
        oB = np.argsort(-fB0, 1)[:, :10]
        zstats["z0_top"].append(np.take_along_axis(z0, oB, 1).ravel()); zstats["z1_top"].append(np.take_along_axis(z1, oB, 1).ravel()); zstats["zr_top"].append(np.take_along_axis(zr, oB, 1).ravel())
        zstats["z0_all"].append(z0.ravel()); zstats["zr_all"].append(zr.ravel())
        for n, f in [("B0", fB0), ("pertask_zt", fPT), ("linear_z", fLZ), ("linear_z_pertask", fLZT), ("pertask_zt_nors", fPT_nors), ("B0+cont", fB0C), ("linear_z+cont", fLZC), ("cont_only", fC), ("cascade_rs50_B0", fCasB0), ("cascade_rs50_cont", fCasC), ("prod_zt", fPR), ("min_zt", fMIN)]:
            i = np.argmax(f, 1); E_of[n][qi] = Eall[np.arange(len(qi)), i]
            if n in conf_of: conf_of[n][qi] = f[np.arange(len(qi)), i]
        # "B0_recal_top": B0 top-10 re-scored by per-task linear z (checks whether the gain is in the short list)
        f = np.full(fB0.shape, -np.inf); np.put_along_axis(f, oB, np.take_along_axis(fLZT, oB, 1), 1)
        i = np.argmax(f, 1); E_of["B0_recal_top"][qi] = Eall[np.arange(len(qi)), i]
    res = {"n": N, "err": {n: E_of[n].mean() for n in names}, "B0_check_vs_npz": float(np.abs(E_of["B0"] - z["err"][sel]).max()),
           "aurc": {n: aurc(E_of[n], conf_of[n]) for n in conf_of}}
    zs = {k: np.concatenate(v) for k, v in zstats.items()}
    res["z_top10_quantiles"] = {k: np.quantile(zs[k], [0.1, 0.5, 0.9]).round(2).tolist() for k in ("z0_top", "z1_top", "zr_top")}
    res["z_all_quantiles"] = {k: np.quantile(zs[k], [0.1, 0.5, 0.9]).round(2).tolist() for k in ("z0_all", "zr_all")}
    res["frac_top10_with_|z|>2"] = {k: float((np.abs(zs[k]) > 2).mean()) for k in ("z0_top", "z1_top", "zr_top")}
    res["pertask_mu_sg"] = {int(t): {k: round(float(v), 4) for k, v in st.items() if k in ("mu0", "sg0", "mu1", "sg1", "mur", "sgr")} for t, st in list(stats.items())[:3]}
    OUT[cell] = res
    print(cell, json.dumps(res, default=lambda x: round(float(x), 4)), flush=True)
tag = "_".join(CELLS) if len(CELLS) <= 2 else "all"
json.dump(OUT, open(f"/home/weiland/.claude/jobs/a607dd74/tmp/ideation_B/d3_out_{tag}.json", "w"), indent=1, default=float)
