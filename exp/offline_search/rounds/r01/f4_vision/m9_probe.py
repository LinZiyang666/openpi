"""M9 `vision_layout_probe` (owner-requested diagnostic): what is vision worth at the EARLY decisions, and in what
representation? CPU only, read-only on the store / trace, <= 16 worker processes (F4 CPU allotment).

Setup (per cell = model x suite x arm, per decision step s in {0, 1, 2}, per task):
  queries    : every query episode of the task at step s (500 episodes per cell); GT = a_inf of that row (metric side).
  candidates : STEP-ALIGNED rows of the BIG library (pi0.5 bpool_cs, GR00T bpool_all): the row at step s of every library
               episode of the same task (~50 per task). Secondary: the current library where init indices exist
               (pi05_spatial / groot_spatial (rule-derived) / groot_l10; pi05_l10 current has none) -- non-token reps only.
  err        : harness err (RMS over [:5, :7] in sigma_d units, sigma of library/current) of the picked candidate's chunk.
Pickers (argmax of a similarity over the candidates; "random" = mean err over candidates, "oracle" = min):
  privileged : object layout from the LIBERO init state (m9_init_layout_dump.py -> init_layout.json). The query init is
               row `init` of the A-pool file db_init/libero/<suite>_apool/<task>.init (== trace attr orig_init_state_idx);
               a library episode's init is row orig_init_state_idx of the B-pool file db_init/libero/<suite>/<task>.init
               (GR00T spatial bpool_all has it for 36/500 episodes only; the rest use init = episode_id - 50 * task_id,
               the rule that holds for 100 % of the 1,586 library episodes where both are known).
               lay_obj  = xy of every free-joint object + the fixture joints that vary (drawer / stove knob / microwave
                          door), each dim z-scored by its std over the task's A+B inits (z / quaternions are constant in
                          the init files: objects are placed upright at fixed heights before settling).
               lay_full = lay_obj + the 7 arm joint angles (init joint noise std ~0.02 rad), z-scored the same way.
               lay_ooi  = only the BDDL objects of interest (free objects' xy + their varying fixture joints), z-scored.
               rel      = (object-of-interest xy - end-effector xy at step s) for every free object of interest, plus
                          the eef height, in meters (query eef = raw_state; library eef = affine map of rs[:3]).
               priv_obj / priv_full / priv_ooi / priv_rel: -L2 on those; "+st": z(priv) + z(-d_state) (z within the
               candidate set).
  baselines  : state (-L2 rs[:8]), pool_v0 / pool_v1 (cosine of the stored 4x4-pooled keys), B0 (online fused formula).
  visual reps (training-free; tokens: full 256x2048 fp16 vision tokens, query side extracted from the trace h5 by
               m9_extract_early_tokens.py, library side library/<m>_<s>/<lib>/tok/):
     pool        cosine of pooled keys                       res_pool   cosine of (pooled key - candidate-set mean)
     res_pool_tm cosine after subtracting the task mean over ALL library rows of the task (what a method can use)
     tok         cosine of the flattened full token map      res_tok    same after subtracting the candidate-set mean
                                                                        token map ("task-mean token map on the library",
                                                                        computed per task AND step = the candidate set)
     topvarN_raw / topvarN_res   keep the N in {16, 32} token positions with the largest within-candidate-set variance,
                                 cosine of the concatenated (raw / centred) tokens there
     maxsim / maxsim_res         late interaction: mean over query tokens of the max cosine to any candidate token
     chamfer / chamfer_res       symmetric: 0.5 * (maxsim(q->c) + maxsim(c->q))
     pix16 / pix32 (pi0.5 only)  -L2 of area-downsampled RGB model-input images (input_images/*, same source both
                                 sides); pix32_rescos: cosine after subtracting the candidate-set mean image
     each for field v0 (agentview) and v1 (wrist); "_both" = z(v0 sim) + z(v1 sim) (z within the candidate set);
     "+st" = z(rep) + z(-d_state).
Also reported per rep: rho = mean over queries of Spearman(sim, -err) across the candidates (ranking quality beyond
the top-1). Context rows "*_all" / "*_alltask": the same pickers over ALL big-library rows of the task (not
step-aligned, i.e. what a normal retrieval faces): oracle, random, state, pool_v0/v1, B0, res_pool_tm, (+st).

Outputs: raw per-job json under <derived>/m9_raw/ (per-query err per picker; init-mapping check with --init-check);
m9_tables.py aggregates them into <family dir>/m9_results.json + m9_tables.md.
    taskset -c 18-25,62-69 .venv/bin/python exp/offline_search/rounds/r01/f4_vision/m9_probe.py --procs 16 \
        --cells pi05_spatial_inf,pi05_spatial_cache [--init-check]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from exp.offline_search.harness import dims, metrics, store  # noqa: E402

ROOT = pathlib.Path("/dev/shm/offline_search_store")
SSD = pathlib.Path("/home/weiland/trace_runs/offline_search_store")
DER = SSD / "derived" / "r01" / "f4_vision"
EARLY = DER / "early_tokens"
BDDL = pathlib.Path("/home/weiland/projects/openpi_ext/envs/libero_sim/lib/python3.8/site-packages/libero/libero/"
                    "bddl_files")
FAM = pathlib.Path(__file__).resolve().parent
BIG = {"pi05": "bpool_cs", "groot": "bpool_all"}
SUITE_FULL = {"spatial": "libero_spatial", "l10": "libero_10"}
STEPS = (0, 1, 2)
CELLS = [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]
_EPS = 1e-8


# ----------------------------------------------------------------------------------------------- layouts
def _tload(p):
    import torch

    return np.asarray(torch.load(p, weights_only=False), np.float64)


_LAYOUT_CACHE = {}


def layouts(suite_full):
    """task string -> dict(A=[50, D] A-pool inits, B=[50, D] B-pool inits, obj_idx, fix_idx, arm_idx, scale_obj,
    scale_full)."""
    if suite_full in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[suite_full]
    L = json.loads((DER / "init_layout.json").read_text())[suite_full]
    out = {}
    for rec in L.values():
        A, B = _tload(rec["apool_file"]), _tload(rec["bpool_file"])
        AB = np.concatenate([A, B])
        obj = [i for j in rec["joints"] if j["kind"] == "object_free" for i in j["state_idx"][:2]]
        fix = [j["state_idx"][0] for j in rec["joints"] if j["kind"] == "fixture_joint"
               and AB[:, j["state_idx"][0]].std() > 1e-9]
        arm = [j["state_idx"][0] for j in rec["joints"] if j["kind"] == "robot" and j["type"] == 3
               and j["name"].startswith("robot0_joint")]
        lo = obj + fix
        lf = lo + arm
        so = AB[:, lo].std(0) + 1e-9
        sf = AB[:, lf].std(0) + 1e-9
        # objects of interest (BDDL :obj_of_interest): free objects -> xy, fixtures -> their varying joints
        bddl = (BDDL / suite_full / f"{rec['name']}.bddl").read_text()
        ooi = bddl.split("(:obj_of_interest", 1)[1].split(")", 1)[0].split()
        ooi_xy = [j["state_idx"][:2] for j in rec["joints"] if j["kind"] == "object_free"
                  and j["name"][: -len("_joint0")] in ooi]
        ooi_fix = [j["state_idx"][0] for j in rec["joints"] if j["kind"] == "fixture_joint"
                   and any(j["name"].startswith(o) for o in ooi) and AB[:, j["state_idx"][0]].std() > 1e-9]
        li = [i for xy in ooi_xy for i in xy] + ooi_fix
        si = AB[:, li].std(0) + 1e-9
        out[rec["task"]] = {"A": A, "B": B, "lo": lo, "lf": lf, "so": so, "sf": sf, "arm": arm,
                            "n_obj": len(obj) // 2, "n_fix": len(fix), "ooi": ooi, "ooi_xy": ooi_xy,
                            "li": li, "si": si}
    _LAYOUT_CACHE[suite_full] = out
    return out


_AFF = {}


def eef_affine(key):
    """[3, 2] (slope, intercept) mapping rs[:3] -> raw eef xyz, least squares over the key's inf query cell."""
    if key not in _AFF:
        q = ROOT / "queries" / f"{key}_inf"
        rs = np.asarray(np.load(q / "rs.npy", mmap_mode="r")[:, :3], np.float64)
        raw = np.asarray(np.load(q / "raw_state.npy", mmap_mode="r")[:, :3], np.float64)
        cf = np.zeros((3, 2))
        for d in range(3):
            X = np.stack([rs[:, d], np.ones(len(rs))], 1)
            cf[d] = np.linalg.lstsq(X, raw[:, d], rcond=None)[0]
        _AFF[key] = cf
    return _AFF[key]


def lib_inits(libdir):
    eps = json.loads((libdir / "episodes.json").read_text())
    init = []
    for e in eps:
        i = e.get("orig_init_state_idx")
        init.append(int(i) if i is not None else int(e["episode_id"]) - 50 * int(e["task_id"]))
    return eps, np.asarray(init)


# ------------------------------------------------------------------------------------------------- sims
def _cos(Q, C):
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), _EPS)
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), _EPS)
    return Qn @ Cn.T


def _l2(Q, C):
    q2 = (Q * Q).sum(1)[:, None]
    c2 = (C * C).sum(1)[None]
    return np.sqrt(np.maximum(q2 - 2 * Q @ C.T + c2, 0))


def _z(S):
    """z-score each query row across its candidates."""
    mu = S.mean(1, keepdims=True)
    sd = S.std(1, keepdims=True)
    return (S - mu) / np.maximum(sd, 1e-12)


def _pool(T):
    n = T.shape[0]
    return T.reshape(n, 4, 4, 4, 4, T.shape[-1]).mean(axis=(2, 4)).reshape(n, -1)


def _maxsim(Qt, Ct, blk=8):
    """Qt [nq, 256, D], Ct [nc, 256, D] L2-normalized per token -> (maxsim q->c, maxsim c->q) [nq, nc]."""
    nq, nt, D = Qt.shape
    nc = Ct.shape[0]
    Cf = Ct.reshape(nc * nt, D)
    m_qc = np.empty((nq, nc), np.float32)
    m_cq = np.empty((nq, nc), np.float32)
    for lo in range(0, nq, blk):
        hi = min(nq, lo + blk)
        S = (Qt[lo:hi].reshape(-1, D) @ Cf.T).reshape(hi - lo, nt, nc, nt)
        m_qc[lo:hi] = S.max(axis=3).mean(axis=1)
        m_cq[lo:hi] = S.max(axis=1).mean(axis=2)
    return m_qc, m_cq


def _tnorm(T):
    return T / np.maximum(np.linalg.norm(T, axis=2, keepdims=True), _EPS)


def _down(img, k):
    """uint8 [n, 224, 224, 3] -> float32 [n, k*k*3] area mean."""
    n = img.shape[0]
    f = 224 // k
    x = np.asarray(img, np.float32).reshape(n, k, f, k, f, 3).mean(axis=(2, 4)) / 255.0
    return x.reshape(n, -1)


def _spearman_rows(S, E):
    """mean over rows of Spearman(S[i], -E[i])."""
    rs = np.argsort(np.argsort(S, axis=1), axis=1).astype(np.float64)
    re = np.argsort(np.argsort(-E, axis=1), axis=1).astype(np.float64)
    rs -= rs.mean(1, keepdims=True)
    re -= re.mean(1, keepdims=True)
    num = (rs * re).sum(1)
    den = np.sqrt((rs * rs).sum(1) * (re * re).sum(1))
    return float(np.mean(num / np.maximum(den, 1e-12)))


# -------------------------------------------------------------------------------------------------- job
def job(args):
    cell, s, t, libname, with_tok = args
    t0 = time.time()
    m, su, arm = store.parse_cell(cell)
    key = f"{m}_{su}"
    sigma = store.action_sigma(str(ROOT), key)
    qdir = ROOT / "queries" / cell
    qeps = json.loads((qdir / "episodes.json").read_text())
    qi = [i for i, e in enumerate(qeps) if e["task_id"] == t and e["end"] - e["start"] > s]
    if not qi:
        return None
    qrows = np.array([qeps[i]["start"] + s for i in qi])
    task = qeps[qi[0]]["task"]
    ldir = ROOT / "library" / key / libname
    leps, linit = lib_inits(ldir)
    li = [j for j, e in enumerate(leps) if e["task_id"] == t and e["num_steps"] > s]
    if libname == "current" and any(e["orig_init_state_idx"] is None for e in leps) and m == "pi05":
        return None
    lrows = np.array([leps[j]["start"] + s for j in li])
    lay = layouts(SUITE_FULL[su])[task]
    # --- GT errors
    a_inf = np.load(qdir / "a_inf.npy", mmap_mode="r")
    lact = np.load(ldir / "action.npy", mmap_mode="r")
    G = metrics.seg(a_inf[qrows])                          # [nq, 5, 7]
    A = metrics.seg(lact[lrows])                           # [nc, 5, 7]
    E = metrics.err_seg(A[None], G[:, None], sigma)        # [nq, nc]
    sims = {}
    # --- privileged layout
    qin = np.array([qeps[i]["init"] for i in qi])
    Xq_o = lay["A"][qin][:, lay["lo"]] / lay["so"]
    Xl_o = lay["B"][linit[li]][:, lay["lo"]] / lay["so"]
    Xq_f = lay["A"][qin][:, lay["lf"]] / lay["sf"]
    Xl_f = lay["B"][linit[li]][:, lay["lf"]] / lay["sf"]
    sims["priv_obj"] = -_l2(Xq_o, Xl_o)
    sims["priv_full"] = -_l2(Xq_f, Xl_f)
    sims["priv_ooi"] = -_l2(lay["A"][qin][:, lay["li"]] / lay["si"], lay["B"][linit[li]][:, lay["li"]] / lay["si"])
    # relative geometry (meters): object-of-interest xy minus the end-effector xy AT THIS STEP, plus the eef height.
    # query eef = raw_state[:3]; library eef = affine map of its rs[:3] (fitted on the query cells, exact for pi0.5,
    # <= 3 mm for GR00T)
    cf = eef_affine(key)
    eq = np.asarray(np.load(qdir / "raw_state.npy", mmap_mode="r")[qrows], np.float64)[:, :3]
    el = np.asarray(np.load(ldir / "rs.npy", mmap_mode="r")[lrows], np.float64)[:, :3] * cf[:, 0] + cf[:, 1]
    oq = np.concatenate([lay["A"][qin][:, xy] - eq[:, :2] for xy in lay["ooi_xy"]] + [eq[:, 2:3]], 1)
    ol = np.concatenate([lay["B"][linit[li]][:, xy] - el[:, :2] for xy in lay["ooi_xy"]] + [el[:, 2:3]], 1)
    sims["priv_rel"] = -_l2(oq, ol)
    # --- state / pooled keys / B0
    rs_q = dims.valid_state(np.asarray(np.load(qdir / "rs.npy", mmap_mode="r")[qrows], np.float64), m)
    rs_l = dims.valid_state(np.asarray(np.load(ldir / "rs.npy", mmap_mode="r")[lrows], np.float64), m)
    dst = _l2(rs_q, rs_l)
    sims["state"] = -dst
    sims["priv_obj+st"] = _z(sims["priv_obj"]) + _z(-dst)
    sims["priv_full+st"] = _z(sims["priv_full"]) + _z(-dst)
    sims["priv_ooi+st"] = _z(sims["priv_ooi"]) + _z(-dst)
    sims["priv_rel+st"] = _z(sims["priv_rel"]) + _z(-dst)
    K = {}
    for f in ("v0", "v1"):
        Kq = np.asarray(np.load(qdir / f"key_{f}.npy", mmap_mode="r")[qrows], np.float32)
        Kl = np.asarray(np.load(ldir / f"key_{f}.npy", mmap_mode="r")[lrows], np.float32)
        K[f] = (Kq, Kl)
        sims[f"pool_{f}"] = _cos(Kq, Kl)
        mu = Kl.mean(0, keepdims=True)
        sims[f"res_pool_{f}"] = _cos(Kq - mu, Kl - mu)
        ltid_ = np.asarray(np.load(ldir / "task_id.npy", mmap_mode="r"))
        mt = np.zeros((1, Kl.shape[1]), np.float64)                   # task mean over ALL task rows (chunked)
        ar_ = np.where(ltid_ == t)[0]
        Kall = np.load(ldir / f"key_{f}.npy", mmap_mode="r")
        for lo_ in range(0, len(ar_), 512):
            mt += np.asarray(Kall[ar_[lo_:lo_ + 512]], np.float64).sum(0, keepdims=True)
        mt = (mt / len(ar_)).astype(np.float32)
        sims[f"res_pool_tm_{f}"] = _cos(Kq - mt, Kl - mt)
    p = store.current_params(m, su)
    w, mu_, sg_ = p["weights"], p["mu"], p["sigma"]

    def zt(x, a, b):
        return 0.5 * (np.tanh((x - a) / b) + 1.0)
    rs_q32 = np.asarray(np.load(qdir / "rs.npy", mmap_mode="r")[qrows], np.float64)
    rs_l32 = np.asarray(np.load(ldir / "rs.npy", mmap_mode="r")[lrows], np.float64)
    sims["B0"] = (w[0] * zt(sims["pool_v0"], mu_[0], sg_[0]) + w[1] * zt(sims["pool_v1"], mu_[1], sg_[1])
                  + w[2] * zt(-_l2(rs_q32, rs_l32), mu_[2], sg_[2]))
    # --- token reps
    if with_tok:
        ltok_idx = np.load(ldir / "tok" / "rows.npy") if (ldir / "tok" / "rows.npy").exists() else None
        if ltok_idx is not None:
            pos = np.full(int(ltok_idx.max()) + 1, -1, np.int64)
            pos[ltok_idx] = np.arange(ltok_idx.shape[0])
            tpos = pos[lrows]
        else:
            tpos = lrows
        assert (tpos >= 0).all()
        erow = np.load(EARLY / cell / "rows.npy")
        assert (erow[qi, s] == qrows).all()
        for f in ("v0", "v1"):
            Tq = np.asarray(np.load(EARLY / cell / f"{f}.npy", mmap_mode="r")[qi, s], np.float32)
            Ltok = np.load(ldir / "tok" / f"{f}.npy", mmap_mode="r")
            o = np.argsort(tpos)
            Tl = np.empty((len(tpos), 256, 2048), np.float32)
            Tl[o] = np.asarray(Ltok[tpos[o]], np.float32)
            Mm = Tl.mean(0, keepdims=True)
            Rq, Rl = Tq - Mm, Tl - Mm
            nq, nc = Tq.shape[0], Tl.shape[0]
            sims[f"tok_{f}"] = _cos(Tq.reshape(nq, -1), Tl.reshape(nc, -1))
            sims[f"res_tok_{f}"] = _cos(Rq.reshape(nq, -1), Rl.reshape(nc, -1))
            var = (Rl ** 2).sum(axis=(0, 2))                  # [256] within-set variance per position
            for n in (16, 32):
                top = np.argsort(-var)[:n]
                sims[f"topvar{n}_raw_{f}"] = _cos(Tq[:, top].reshape(nq, -1), Tl[:, top].reshape(nc, -1))
                sims[f"topvar{n}_res_{f}"] = _cos(Rq[:, top].reshape(nq, -1), Rl[:, top].reshape(nc, -1))
            a, b = _maxsim(_tnorm(Tq), _tnorm(Tl))
            sims[f"maxsim_{f}"] = a
            sims[f"chamfer_{f}"] = 0.5 * (a + b)
            a, b = _maxsim(_tnorm(Rq), _tnorm(Rl))
            sims[f"maxsim_res_{f}"] = a
            sims[f"chamfer_res_{f}"] = 0.5 * (a + b)
            del Tq, Tl, Rq, Rl
        if m == "pi05":
            for f in ("img0", "img1"):
                fv = "v0" if f == "img0" else "v1"
                Iq = np.load(EARLY / cell / f"{f}.npy", mmap_mode="r")[qi, s]
                Il = np.load(ldir / "tok" / f"{f}.npy", mmap_mode="r")
                o = np.argsort(tpos)
                IL = np.empty((len(tpos), 224, 224, 3), np.uint8)
                IL[o] = Il[tpos[o]]
                for k in (16, 32):
                    xq, xl = _down(Iq, k), _down(IL, k)
                    sims[f"pix{k}_{fv}"] = -_l2(xq, xl)
                    mu = xl.mean(0, keepdims=True)
                    sims[f"pix{k}_rescos_{fv}"] = _cos(xq - mu, xl - mu)
    # --- combos: both fields, + state
    base = sorted({k[:-3] for k in sims if k.endswith("_v0") and k[:-3] + "_v1" in sims})
    for b in base:
        sims[f"{b}_both"] = _z(sims[f"{b}_v0"]) + _z(sims[f"{b}_v1"])
    for k in list(sims):
        if k.endswith(("_both", "_v1")) and not k.startswith(("priv", "state", "B0")):
            sims[f"{k}+st"] = _z(sims[k]) + _z(-dst)
    # --- evaluate
    ar = np.arange(len(qi))
    res = {"random": E.mean(1), "oracle": E.min(1)}
    rho = {}
    for k, S in sims.items():
        S = np.asarray(S, np.float64)
        pick = np.argmax(S + 1e-12 * 0, axis=1)
        res[k] = E[ar, pick]
        rho[k] = _spearman_rows(S, E)
    # context: pickers over ALL library rows of the task (not step aligned), as a normal retrieval would
    ltid = np.load(ldir / "task_id.npy", mmap_mode="r")
    allr = np.where(np.asarray(ltid) == t)[0]
    Aall = metrics.seg(lact[allr])
    Eall = metrics.err_seg(Aall[None], G[:, None], sigma)
    res["oracle_alltask"] = Eall.min(1)
    res["random_alltask"] = Eall.mean(1)
    rs_a = np.asarray(np.load(ldir / "rs.npy", mmap_mode="r")[allr], np.float64)
    dall = _l2(rs_q32, rs_a)
    sa = {"state_all": -dall}
    for f in ("v0", "v1"):
        KA = np.asarray(np.load(ldir / f"key_{f}.npy", mmap_mode="r")[allr], np.float32)
        sa[f"pool_{f}_all"] = _cos(K[f][0], KA)
        mu = KA.mean(0, keepdims=True)
        sa[f"res_pool_tm_{f}_all"] = _cos(K[f][0] - mu, KA - mu)
        del KA
    sa["B0_all"] = (w[0] * zt(sa["pool_v0_all"], mu_[0], sg_[0]) + w[1] * zt(sa["pool_v1_all"], mu_[1], sg_[1])
                    + w[2] * zt(-dall, mu_[2], sg_[2]))
    sa["pool_both_all"] = _z(sa["pool_v0_all"]) + _z(sa["pool_v1_all"])
    sa["res_pool_tm_both_all"] = _z(sa["res_pool_tm_v0_all"]) + _z(sa["res_pool_tm_v1_all"])
    for k in ("pool_v1_all", "pool_both_all", "res_pool_tm_v1_all", "res_pool_tm_both_all"):
        sa[k.replace("_all", "+st_all")] = _z(sa[k]) + _z(-dall)
    for k, S in sa.items():
        res[k] = Eall[ar, np.argmax(S, axis=1)]
        rho[k] = _spearman_rows(S, Eall)
    return {"cell": cell, "step": s, "task": int(t), "lib": libname, "nq": len(qi), "nc": len(li),
            "err": {k: v.astype(np.float32).tolist() for k, v in res.items()},
            "rho": rho, "qi": qi, "secs": time.time() - t0}


def init_check():
    """Does the mapping (query init -> A-pool row, library init -> B-pool row) look right? Spearman between the
    pairwise init arm-joint distance and the pairwise step-0 robot-state distance within each task."""
    out = {}
    for m in ("pi05", "groot"):
        for su in ("spatial", "l10"):
            key = f"{m}_{su}"
            laysu = layouts(SUITE_FULL[su])
            for side in ("query", "big"):
                vals = []
                for t in range(10):
                    if side == "query":
                        qeps = json.loads((ROOT / "queries" / f"{key}_inf" / "episodes.json").read_text())
                        ee = [e for e in qeps if e["task_id"] == t]
                        rows = np.array([e["start"] for e in ee])
                        rs = np.asarray(np.load(ROOT / "queries" / f"{key}_inf" / "rs.npy", mmap_mode="r")[rows])
                        lay = laysu[ee[0]["task"]]
                        J = lay["A"][[e["init"] for e in ee]][:, lay["arm"]]
                        Jwrong = lay["B"][[e["init"] for e in ee]][:, lay["arm"]]
                    else:
                        ldir = ROOT / "library" / key / BIG[m]
                        leps, linit = lib_inits(ldir)
                        idx = [j for j, e in enumerate(leps) if e["task_id"] == t]
                        rows = np.array([leps[j]["start"] for j in idx])
                        rs = np.asarray(np.load(ldir / "rs.npy", mmap_mode="r")[rows])
                        lay = laysu[leps[idx[0]]["task"]]
                        J = lay["B"][linit[idx]][:, lay["arm"]]
                        Jwrong = lay["A"][linit[idx]][:, lay["arm"]]
                    rs = dims.valid_state(rs.astype(np.float64), m)[:, :3]      # eef position
                    iu = np.triu_indices(len(rows), 1)
                    dr = _l2(rs, rs)[iu]

                    def sp(x, y):
                        rx = np.argsort(np.argsort(x)).astype(float)
                        ry = np.argsort(np.argsort(y)).astype(float)
                        return float(np.corrcoef(rx, ry)[0, 1])
                    vals.append((sp(_l2(J, J)[iu], dr), sp(_l2(Jwrong, Jwrong)[iu], dr)))
                v = np.array(vals)
                out[f"{key}_{side}"] = {"spearman_joint_vs_eef_dist": float(v[:, 0].mean()),
                                        "same_with_the_other_pool (control)": float(v[:, 1].mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--no-tok", action="store_true")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--init-check", action="store_true")
    ap.add_argument("--out", default=None, help="raw per-job json (default <derived>/m9_raw/<cells>.json)")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.tasks.split("-"))
    cells = a.cells.split(",")
    jobs = []
    for c in cells:
        m, su, _ = store.parse_cell(c)
        for s in STEPS:
            for t in range(lo, hi + 1):
                jobs.append((c, s, t, BIG[m], not a.no_tok))
                if not (m == "pi05" and su == "l10"):
                    jobs.append((c, s, t, "current", False))
    jobs.sort(key=lambda j: (not j[4], j[0].startswith("pi05")))   # heavy token jobs first
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(a.procs) as ex:
        for k, r in enumerate(ex.map(job, jobs, chunksize=1)):
            if r is not None:
                out.append(r)
            if k % 20 == 0:
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    chk = init_check() if a.init_check else None
    if chk:
        print(json.dumps(chk, indent=1), flush=True)
    if a.out is None:
        (DER / "m9_raw").mkdir(parents=True, exist_ok=True)
        a.out = str(DER / "m9_raw" / (a.cells.replace(",", "+") + f"_t{a.tasks}.json"))
    pathlib.Path(a.out).write_text(json.dumps({"jobs": out, "init_check": chk}))
    print("wrote", a.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
