"""V8 `awm_t2_metric` (R2, family g4_t2; ideation A-P4, tier T2) and its in-family AWM controls.

Retrieval = AWM (ideation A-P1) with the per-task linear whitening replaced by a learned metric phi:
  x = [PCA-64 key_v0, PCA-64 key_v1, rs[:8]] (136-d), z-scored per task exactly as AWM (fit-row mean / std);
  phi_t(x_n) = x_n @ Wl_t                                        ("lin": linear only)
  phi_t(x_n) = x_n @ Wl_t + relu(x_n @ W1 + b1 + c_t) @ W2        ("mlp": 2-layer MLP, 136 -> hidden -> 64)
  Wl_t is initialised with AWM's rank-64 whitening of task t (scaled by the LSQ factor alpha_t so distances are in
  head-RMS units) and W2 = 0, i.e. at initialisation phi reproduces AWM (rank-64, measured within .004 of the full
  136-d AWM by ideation A). Training (subprocess t2_train.py; CUDA if visible and >= 8 GB free, else CPU -- the model
  is tiny, 8 CPU threads train as fast as the 4090): pairs = every fit row with its 16 nearest rows (other episodes,
  initial metric) + 16 random same-task rows (other episodes); loss="abs" (the ideation-A P4 spec):
  Huber(||phi(x_i) - phi(x_j)|| - RMS_ij(head, sigma units)), Adam, 5 epochs. Other losses (t2_train.py): "off"
  (+ learned offset), "log" (relative), "mlkr" (metric learning for kernel regression = the retrieval's own objective,
  NOT the spec); remine=True re-mines the nearest pairs under the current metric every epoch; val_frac holds out
  library episodes (diagnostic: held-out pair loss + held-out LOEO kernel-retrieval error in the train log).
  Measured (g4_t2 hand-back): the spec's pairwise regression halves the held-out pair loss yet worsens held-out
  retrieval and stale err (objective misaligned with kNN retrieval); mlkr gives -.003..-.017 stale; none reaches the
  owner's T2 bar (>= .02 better than AWM in >= 3/4 cells).
  Per task vs per suite: the linear path stays per task (AWM measured a task-shared metric +.01..+.02 worse) while
  the nonlinear residual is ONE network per suite with a learned per-task hidden bias c_t (= one-hot task embedding):
  a task of the 50-episode library has only 74-417 rows (2-13 k pairs) against ~55 k residual parameters, so the
  residual has to share statistical strength across the suite's 10 tasks.
Controls (tier T1, no training): phi="awm" = full 136-d AWM codes (the reference the T2 must beat), "awm64" = the
  rank-64 initialisation.
Everything else is AWM-P1: step 0 uses the AWM early-step fit (rows with step <= 2, full 136-d) for query AND
  candidates; stale (prev_hit True / None): f = -d; fresh (prev_hit False): f = -d/median(d) - fresh_lam * c/s_c with
  c = RMS(head_cand - prev_a_exec[5:10, :7]/sigma); top-16 kernel mean (kref 5) of the full (H, 32) chunks.
  Confidence: step 0 -disp5; stale z(-d1)+z(-disp5)+z(-dst) with z-scales from library pseudo-queries (own episode
  excluded); fresh -d1/median(d) - disp5/s_a.
Vision is used in every regime (both camera PCA codes enter every distance). Query: numpy only, CPU.

kwargs: phi in {awm, awm64, lin, mlp}; lib in {current, big} (big = bpool_cs pi0.5 / bpool_all GR00T);
  fit in {same, big} (fit data: the candidate library itself, or the 10x library = "borrowed big-library info");
  PCA basis = fitted on the fit data (current -> derived/r02/g4_t2/pca, big -> r01 F4 basis).
Fitted T2 weights are cached (fingerprint of inputs + hyper-parameters + CODE_VERSION) in
  DERIVED/fits/<ms>/ so the inf / cache jobs of one model x suite, the timing pass and closed-loop servers share one
  canonical fit; set G4T2_REFIT=1 to force retraining. Train logs: DERIVED/fits/<ms>/*.json (train_s, device, losses).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from exp.offline_search.harness import api, dims

try:
    import g4t2_core as core
except ImportError:  # dotted import
    from exp.offline_search.rounds.r02.g4_t2 import g4t2_core as core

CODE_VERSION = "g4t2-v1"
PHIS = ("awm", "awm64", "lin", "mlp")
TRAINER = pathlib.Path(__file__).resolve().parent / "t2_train.py"
DEFAULTS = dict(hidden=256, epochs=5, lr_lin=1e-4, lr_mlp=1e-3, n_nn=16, n_rand=16, n_act=0, loss="abs", val_frac=0.0,
                remine=False, k=16, kref=5, lam=0.1, nn=3, early=True, fresh_lam=0.5, seed=0, batch=512)
SHORT = dict(hidden="h", epochs="ep", lr_lin="lrl", lr_mlp="lrm", n_nn="pnn", n_rand="prd", n_act="pact", loss="L",
             val_frac="val", remine="rm", k="k", kref="kr", lam="lam", nn="nn", early="early", fresh_lam="fl", seed="s", batch="bs")
TRAINED_ONLY = {"hidden", "epochs", "lr_lin", "lr_mlp", "n_nn", "n_rand", "n_act", "loss", "val_frac", "remine", "batch"}
LOSSES = ("abs", "off", "log", "mlkr")


def _fmt(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "1" if x else "0"
    return f"{x:g}".replace(".", "p").replace("-", "m").replace("+", "")


class _Task:
    __slots__ = ("rows", "ep", "step", "mean", "std", "Wl", "W1", "b1", "c", "W2", "Z", "z2", "mean0", "std0", "W0",
                 "Z0", "z02", "Za", "za2", "Wa", "act", "Hc", "h2", "RS", "rs2", "s_d", "s_c", "n")


class AWMT2(api.Method):
    family = "g4_t2"

    def __init__(self, phi="mlp", lib="big", fit="same", device="auto", **kw):
        if phi not in PHIS:
            raise ValueError(f"phi must be one of {PHIS}, got {phi!r}")
        if lib not in ("current", "big") or fit not in ("same", "big"):
            raise ValueError(f"lib in (current, big), fit in (same, big); got {lib!r}, {fit!r}")
        unknown = set(kw) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown kwargs {sorted(unknown)}")
        self.phi, self.lib, self.fit_data, self.device = phi, lib, fit, device
        self.cfg = {**DEFAULTS, **kw}
        for k_ in ("hidden", "epochs", "n_nn", "n_rand", "n_act", "k", "kref", "nn", "seed", "batch"):
            self.cfg[k_] = int(self.cfg[k_])
        if self.cfg["loss"] not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        self.cfg["early"] = bool(self.cfg["early"])
        self.cfg["remine"] = bool(self.cfg["remine"])
        self.tier = "T2" if phi in ("lin", "mlp") else "T1"
        name = f"V8_t2_{phi}_{'cur' if lib == 'current' else 'big'}" + ("_fitbig" if (fit == "big" and lib == "current") else "")
        for k_ in DEFAULTS:
            v = self.cfg[k_]
            if v == DEFAULTS[k_] or (k_ in TRAINED_ONLY and phi not in ("lin", "mlp")) or (k_ == "hidden" and phi != "mlp"):
                continue
            name += f"_{SHORT[k_]}{_fmt(v)}"
        self.name = name
        self.fit_info = {}

    # ---------------------------------------------------------------------------------------------- fit
    def fit(self, lib, ctx):
        t_fit = time.time()
        cfg = self.cfg
        self.model = ctx.model
        ms = ctx.lib_key
        root = ctx.root
        sig = np.asarray(ctx.action_sigma, np.float64)
        self.sig32 = sig.astype(np.float32)
        self.cand_name = "current" if self.lib == "current" else core.BIG[ctx.model]
        self.fit_name = self.cand_name if self.fit_data == "same" else core.BIG[ctx.model]
        self.basis = "current" if self.fit_name == "current" else "big"
        C = lib if self.cand_name == "current" else ctx.open_library(self.cand_name)
        F = C if self.fit_name == self.cand_name else (lib if self.fit_name == "current" else ctx.open_library(self.fit_name))
        with self.prof.section("fit_pca"):
            bs, bmeta = core.pca_basis(root, ms, self.basis)
            self.B0 = bs["v0"][1]
            self.B1 = bs["v1"][1]
            self.muB0 = (bs["v0"][0] @ self.B0).astype(np.float32)
            self.muB1 = (bs["v1"][0] @ self.B1).astype(np.float32)
            Pc = core.lib_proj(root, ms, self.cand_name, self.basis)
            Pf = Pc if F is C else core.lib_proj(root, ms, self.fit_name, self.basis)

        def feats(P, L):
            rs8 = dims.valid_state(np.asarray(L.rs, np.float32), ctx.model)
            return np.concatenate([P["v0"], P["v1"], rs8], 1).astype(np.float64)

        def heads(L):
            a = np.asarray(dims.valid_action(L.action), np.float64) / sig
            return a.reshape(a.shape[0], core.NH)

        Xc, Hc = feats(Pc, C), heads(C)
        Xf, Hf = (Xc, Hc) if F is C else (feats(Pf, F), heads(F))
        epf, stf = np.asarray(F.episode, np.int64), np.asarray(F.step, np.int64)
        tasks = [int(t) for t in C.tasks()]
        self.task_index = {t: i for i, t in enumerate(tasks)}
        # ---- per-task AWM fits on the fit rows (ideation A-P1)
        fits = {}
        with self.prof.section("fit_awm"):
            for t in tasks:
                rf = np.asarray(F.rows_of_task(t), np.int64)
                mean, std, W, Minv = core.awm_fit(Xf[rf], Hf[rf], epf[rf], cfg["nn"], cfg["lam"])
                fd = {"rf": rf, "mean": mean, "std": std, "W": W}
                if self.phi != "awm":
                    fd["W64"] = core.rank_w((Xf[rf] - mean) / std, Minv, core.PDIM)
                if cfg["early"]:
                    m = stf[rf] <= 2
                    fd["mean0"], fd["std0"], fd["W0"], _ = core.awm_fit(Xf[rf][m], Hf[rf][m], epf[rf][m], cfg["nn"], cfg["lam"])
                fits[t] = fd
        tw = None
        if self.phi in ("lin", "mlp"):
            with self.prof.section("fit_train"):
                tw = self._train(ctx, F, Xf, Hf, epf, fits, tasks)
        # ---- candidate structures
        act_all = C.action
        rs_c = dims.valid_state(np.asarray(C.rs, np.float32), ctx.model)
        epc, stc = np.asarray(C.episode, np.int64), np.asarray(C.step, np.int64)
        self.T = {}
        with self.prof.section("fit_index"):
            for t in tasks:
                fd = fits[t]
                rc = np.asarray(C.rows_of_task(t), np.int64)
                T = _Task()
                T.rows, T.ep, T.step, T.n = rc, epc[rc], stc[rc], len(rc)
                T.mean, T.std = fd["mean"].astype(np.float32), fd["std"].astype(np.float32)
                xn = ((Xc[rc] - fd["mean"]) / fd["std"]).astype(np.float32)
                T.Wl = T.W1 = T.b1 = T.c = T.W2 = None
                T.Wa = np.ascontiguousarray(fd["W"], np.float32)
                if self.phi == "awm":
                    T.Wl = T.Wa
                elif self.phi == "awm64":
                    T.Wl = np.ascontiguousarray(fd["W64"], np.float32)
                else:
                    ti = self.task_index[t]
                    T.Wl = np.ascontiguousarray(tw["Wl"][ti], np.float32)
                    if self.phi == "mlp":
                        T.W1, T.b1, T.W2 = tw["W1"], tw["b1"], tw["W2"]
                        T.c = np.ascontiguousarray(tw["c"][ti], np.float32)
                T.Z = np.ascontiguousarray(self._phi(T, xn), np.float32)
                T.z2 = np.einsum("ij,ij->i", T.Z, T.Z)
                if self.phi in ("lin", "mlp"):
                    T.Za = np.ascontiguousarray(xn @ T.Wa, np.float32)
                    T.za2 = np.einsum("ij,ij->i", T.Za, T.Za)
                else:
                    T.Za = T.za2 = None
                if cfg["early"]:
                    T.mean0, T.std0 = fd["mean0"].astype(np.float32), fd["std0"].astype(np.float32)
                    T.W0 = np.ascontiguousarray(fd["W0"], np.float32)
                    T.Z0 = np.ascontiguousarray(((Xc[rc] - fd["mean0"]) / fd["std0"]).astype(np.float32) @ T.W0, np.float32)
                    T.z02 = np.einsum("ij,ij->i", T.Z0, T.Z0)
                else:
                    T.mean0 = T.std0 = T.W0 = T.Z0 = T.z02 = None
                T.act = np.ascontiguousarray(np.asarray(act_all[rc], np.float32))
                T.Hc = np.ascontiguousarray(Hc[rc], np.float32)
                T.h2 = np.einsum("ij,ij->i", T.Hc, T.Hc)
                T.RS = np.ascontiguousarray(rs_c[rc], np.float32)
                T.rs2 = np.einsum("ij,ij->i", T.RS, T.RS)
                same = T.ep[:, None] == T.ep[None, :]
                DL = np.sqrt(core.sqdist(T.RS.astype(np.float64), T.RS.astype(np.float64)))
                DL[same] = np.inf
                T.s_d = float(np.median(DL.min(1))) + 1e-6
                tails = (np.asarray(act_all[rc][:, 5:10, dims.ACT_VALID], np.float64) / sig).reshape(len(rc), core.NH)
                CC = np.sqrt(core.sqdist(tails, T.Hc.astype(np.float64)) / core.NH)
                CC[same] = np.inf
                T.s_c = float(np.median(CC.min(1))) + 1e-6
                self.T[t] = T
        # ---- confidence scales from library pseudo-queries (own episode excluded)
        with self.prof.section("fit_scales"):
            rng = np.random.default_rng(cfg["seed"])
            rec = {"d1": [], "disp5": [], "dst": []}
            for t in tasks:
                T = self.T[t]
                sel = np.sort(rng.choice(T.n, size=min(64, T.n), replace=False))
                for i in sel:
                    z = T.Z[i]
                    d = np.sqrt(np.maximum(T.z2 - 2.0 * (T.Z @ z) + float(z @ z), 0.0))
                    other = T.ep != T.ep[i]
                    f = np.where(other, -d, -np.inf)
                    o, w = core.topk_kernel(f, cfg["k"], cfg["kref"])
                    head = (w @ T.Hc[o]) / w.sum()
                    rec["disp5"].append(float(np.sqrt(np.mean((T.Hc[o[:5]] - head) ** 2))))
                    rec["d1"].append(float(d[o[0]]))
                    ds = np.maximum(T.rs2 - 2.0 * (T.RS @ T.RS[i]) + float(T.RS[i] @ T.RS[i]), 0.0)
                    rec["dst"].append(float(np.sqrt(ds[other].min())) / T.s_d)
            self.zs = {k_: (float(np.mean(v)), float(np.std(v)) + 1e-9) for k_, v in rec.items()}
            self.s_a = float(np.median(rec["disp5"])) + 1e-9
        self.fit_info.update({"fit_s": time.time() - t_fit, "cand": self.cand_name, "fit_lib": self.fit_name,
                              "basis": self.basis, "basis_meta": bmeta, "L_cand": int(C.L), "L_fit": int(F.L),
                              "zs": self.zs, "s_a": self.s_a,
                              "s_c": {t: self.T[t].s_c for t in tasks}, "s_d": {t: self.T[t].s_d for t in tasks}})
        try:
            (ctx.scratch / f"{self.name}_fit.json").write_text(json.dumps(self.fit_info, indent=1, default=str))
        except Exception:
            pass

    def _phi(self, T, xn: np.ndarray) -> np.ndarray:
        z = xn @ T.Wl
        if T.W1 is not None:
            z = z + np.maximum(xn @ T.W1 + T.b1 + T.c, 0.0) @ T.W2
        return z

    def _train(self, ctx, F, Xf, Hf, epf, fits, tasks):
        cfg = self.cfg
        tcfg = {"phi": self.phi, "hidden": cfg["hidden"], "epochs": cfg["epochs"], "lr_lin": cfg["lr_lin"],
                "lr_mlp": cfg["lr_mlp"], "batch": cfg["batch"], "seed": cfg["seed"], "loss": cfg["loss"],
                "val_frac": cfg["val_frac"], **({"remine": True, "n_nn": cfg["n_nn"]} if cfg["remine"] else {})}
        lib_dir = pathlib.Path(ctx.root) / "library" / ctx.lib_key / self.fit_name
        key = {"v": CODE_VERSION, "ms": ctx.lib_key, "fit": self.fit_name, "basis": self.basis, "L": int(F.L),
               "src": [core.file_sig(lib_dir / f"{n}.npy") for n in ("action", "key_v0", "key_v1", "rs")],
               "sig": [round(float(s), 8) for s in self.sig32], "nn": cfg["nn"], "lam": cfg["lam"],
               "n_nn": cfg["n_nn"], "n_rand": cfg["n_rand"], "n_act": cfg["n_act"], **tcfg}
        h = core.fingerprint(key)
        out = core.DERIVED / "fits" / ctx.lib_key / f"{self.fit_name}_{self.phi}_{h}.npz"
        logp = out.with_suffix(".json")
        refit = os.environ.get("G4T2_REFIT", "") == "1"
        with core.file_lock(core.DERIVED / "locks" / f"fit_{ctx.lib_key}_{self.fit_name}_{self.phi}_{h}.lock"):
            if refit or not out.exists():
                t0 = time.time()
                Xn_all, tix, PI, PJ, R, W0s, B0s, VAL, KIND, HH, EP = [], [], [], [], [], [], [], [], [], [], []
                off = 0
                alphas = []
                vrng = np.random.default_rng(cfg["seed"] + 7)
                for ti, t in enumerate(tasks):
                    fd = fits[t]
                    rf = fd["rf"]
                    xn = (Xf[rf] - fd["mean"]) / fd["std"]
                    i, j, r, d0, kind = core.build_pairs(xn, fd["W64"], Hf[rf], epf[rf], cfg["n_nn"], cfg["n_rand"],
                                                   cfg["seed"] * 1000 + ti, n_act=cfg["n_act"])
                    alpha, b0 = core.lsq_scale(d0, r, intercept=cfg["loss"] != "abs")
                    Xn_all.append(xn.astype(np.float32))
                    tix.append(np.full(len(rf), ti, np.int64))
                    KIND.append(kind)
                    HH.append(Hf[rf].astype(np.float32))
                    EP.append(epf[rf])
                    ueps = np.unique(epf[rf])
                    vep = vrng.choice(ueps, size=int(round(cfg["val_frac"] * len(ueps))), replace=False)
                    VAL.append(np.isin(epf[rf], vep))
                    PI.append(i + off)
                    PJ.append(j + off)
                    R.append(r)
                    W0s.append((alpha * fd["W64"]).astype(np.float32))
                    B0s.append(b0)
                    alphas.append((alpha, b0))
                    off += len(rf)
                R = np.concatenate(R)
                tcfg_run = dict(tcfg, huber_delta=float(np.median(R)), device=self.device, min_free_gb=8.0,
                                cpu_threads=int(min(8, len(os.sched_getaffinity(0)))))
                prep_s = time.time() - t0
                with tempfile.TemporaryDirectory(prefix="g4t2_") as td:
                    inp, outp = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
                    np.savez(inp, Xn=np.concatenate(Xn_all), tix=np.concatenate(tix), pi=np.concatenate(PI),
                             pj=np.concatenate(PJ), r=R, Wl0=np.stack(W0s), b0=np.asarray(B0s, np.float32),
                             val=np.concatenate(VAL), kind=np.concatenate(KIND), H=np.concatenate(HH),
                             ep=np.concatenate(EP), cfg=np.array(json.dumps(tcfg_run)))
                    env = dict(os.environ)
                    env.pop("OMP_NUM_THREADS", None)
                    cp = subprocess.run([sys.executable, str(TRAINER), inp, outp], env=env, capture_output=True, text=True)
                    if cp.returncode != 0:
                        raise RuntimeError(f"T2 trainer failed (rc {cp.returncode}):\n{cp.stdout[-3000:]}\n{cp.stderr[-3000:]}")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp.npz")
                    shutil.copyfile(outp, tmp)
                    os.replace(tmp, out)
                with np.load(out) as zz:
                    tlog = json.loads(str(zz["log"]))
                tlog.update({"key": key, "fingerprint": h, "prep_s": prep_s, "alpha": alphas,
                             "fit_total_s": time.time() - t0, "created": time.strftime("%Y-%m-%d %H:%M:%S")})
                core.atomic_json(logp, tlog)
                self.fit_info["train_cache_hit"] = False
            else:
                self.fit_info["train_cache_hit"] = True
        with np.load(out) as zz:
            tw = {k_: zz[k_] for k_ in zz.files if k_ != "log"}
        try:
            tl = json.loads(logp.read_text())
            self.fit_info["train"] = {k_: tl.get(k_) for k_ in ("device", "train_s", "prep_s", "fit_total_s", "loss_init",
                                                                "loss_final", "epoch_loss", "steps", "P", "dead_units",
                                                                "residual_norm_ratio", "huber_delta", "fingerprint",
                                                                "val_loss_init", "val_loss_final", "b", "loeo_init",
                                                                "loeo", "remine_overlap")}
        except Exception:
            pass
        self.fit_info["train_file"] = str(out)
        return tw

    # ---------------------------------------------------------------------------------------------- query
    def reset(self, episode):
        pass

    def bytes_per_entry(self) -> float:
        # the entry stores the 136-d feature x (f32); main (64-d T2 / 136-d AWM) and step-0 codes are derived at load
        return float(core.FEAT_DIM * 4)

    def _feat(self, q) -> np.ndarray:
        with self.prof.section("proj"):
            p0 = np.asarray(q.key_v0, np.float32) @ self.B0 - self.muB0
            p1 = np.asarray(q.key_v1, np.float32) @ self.B1 - self.muB1
            rs8 = dims.valid_state(np.asarray(q.rs, np.float32), self.model)
        return np.concatenate([p0, p1, rs8]).astype(np.float32)

    def query(self, q):
        cfg = self.cfg
        T = self.T[int(q.task_id)]
        x = self._feat(q)
        step0 = q.step == 0
        ph = None if step0 else q.prev_hit
        regime = 0 if step0 else (2 if ph is False else 1)
        with self.prof.section("score"):
            if step0 and cfg["early"]:
                z = ((x - T.mean0) / T.std0) @ T.W0
                Z, z2 = T.Z0, T.z02
            else:
                z = self._phi(T, (x - T.mean) / T.std)
                Z, z2 = T.Z, T.z2
            d = np.sqrt(np.maximum(z2 - 2.0 * (Z @ z) + float(z @ z), 0.0))
            cont = None
            if regime == 2:
                tail = (np.asarray(q.prev_a_exec[5:10, dims.ACT_VALID], np.float32) / self.sig32).ravel()
                cont = np.sqrt(np.maximum(T.h2 - 2.0 * (T.Hc @ tail) + float(tail @ tail), 0.0) / core.NH)
                med = float(np.median(d)) + 1e-9
                f = -d / med - cfg["fresh_lam"] * cont / T.s_c
            else:
                f = -d
            o, w = core.topk_kernel(f, cfg["k"], cfg["kref"])
        with self.prof.section("synth"):
            ws = w / w.sum()
            a = (ws.astype(np.float32) @ T.act[o].reshape(len(o), -1)).reshape(T.act.shape[1:])
            head = (a[:dims.EXEC_STEPS, dims.ACT_VALID] / self.sig32).ravel()
            disp5 = float(np.sqrt(np.mean((T.Hc[o[:5]] - head) ** 2)))
            rsq = dims.valid_state(np.asarray(q.rs, np.float32), self.model)
            dst = float(np.sqrt(max(float((T.rs2 - 2.0 * (T.RS @ rsq)).min()) + float(rsq @ rsq), 0.0))) / T.s_d
        d1 = float(d[o[0]])
        med_d = float(np.median(d)) + 1e-9
        if regime == 0:
            conf = -disp5
        elif regime == 1:
            zs = self.zs
            conf = (-(d1 - zs["d1"][0]) / zs["d1"][1] - (disp5 - zs["disp5"][0]) / zs["disp5"][1]
                    - (dst - zs["dst"][0]) / zs["dst"][1])
        else:
            conf = -d1 / med_d - disp5 / self.s_a
        ex = {"regime": float(regime), "d1": d1, "d1_rel": d1 / med_d, "disp5": disp5, "dst": dst,
              "pick_ep": float(T.ep[o[0]]), "pick_step": float(T.step[o[0]]),
              "keff": float(w.sum() ** 2 / (w * w).sum())}
        if cont is not None:                      # fresh regime only (absent keys are NaN in the harness npz)
            ex["cont1"] = float(cont[o[0]] / T.s_c)
        if T.Za is not None and not step0:
            with self.prof.section("ov_awm"):
                xa = ((x - T.mean) / T.std) @ T.Wa
                da = np.sqrt(np.maximum(T.za2 - 2.0 * (T.Za @ xa) + float(xa @ xa), 0.0))
                fa = (-da / (float(np.median(da)) + 1e-9) - cfg["fresh_lam"] * cont / T.s_c) if regime == 2 else -da
                oa, _ = core.topk_kernel(fa, cfg["k"], cfg["kref"])
                ex["ov_awm"] = float(np.intersect1d(o, oa).size) / len(o)
                ex["same_top1_awm"] = float(o[0] == oa[0])
        return api.Result(topk=T.rows[o], scores=f[o].astype(np.float64), confidence=float(conf), action=a,
                          library=self.cand_name, extras=ex)

    # ------------------------------------------------------------------------------------ batch (diagnostics)
    def batch_eval(self, t: int, X: np.ndarray, step: np.ndarray, prev_hit: np.ndarray, tails: np.ndarray | None,
                   rs: np.ndarray, use_awm: bool = False):
        """Vectorized equivalent of query() for n decisions of task t (diag_eval.py). X [n, 136] f32 raw features,
        prev_hit int8 [n] (1 HIT, 0 MISS, -1 none), tails [n, 35] sigma-scaled prev tails (fresh rows), rs [n, 8].
        use_awm: score with the full AWM codes instead of phi (in-run control). Returns dict of per-decision arrays."""
        cfg = self.cfg
        T = self.T[int(t)]
        n = X.shape[0]
        regime = np.where(step == 0, 0, np.where(prev_hit == 0, 2, 1))
        out = {k_: np.full(n, np.nan) for k_ in ("d1", "disp5", "dst", "conf", "regime", "top1")}
        out["a5"] = np.full((n, dims.EXEC_STEPS, dims.ACT_DIMS), np.nan, np.float32)
        out["regime"] = regime.astype(np.float64)
        rs2 = np.einsum("ij,ij->i", rs, rs)
        dst = np.sqrt(np.maximum((T.rs2[None] - 2.0 * rs @ T.RS.T).min(1) + rs2, 0.0)) / T.s_d
        for reg in (0, 1, 2):
            m = np.flatnonzero(regime == reg)
            if m.size == 0:
                continue
            x = X[m]
            if reg == 0 and cfg["early"]:
                z = ((x - T.mean0) / T.std0) @ T.W0
                Z, z2 = T.Z0, T.z02
            elif use_awm:
                z = ((x - T.mean) / T.std) @ T.Wa
                Z, z2 = T.Za if T.Za is not None else T.Z, T.za2 if T.Za is not None else T.z2
            else:
                z = self._phi(T, (x - T.mean) / T.std)
                Z, z2 = T.Z, T.z2
            D = np.sqrt(np.maximum(z2[None] - 2.0 * (z @ Z.T) + np.einsum("ij,ij->i", z, z)[:, None], 0.0))
            med = np.median(D, axis=1) + 1e-9
            if reg == 2:
                tl = tails[m]
                cont = np.sqrt(np.maximum(T.h2[None] - 2.0 * (tl @ T.Hc.T) + np.einsum("ij,ij->i", tl, tl)[:, None], 0.0) / core.NH)
                S = -D / med[:, None] - cfg["fresh_lam"] * cont / T.s_c
            else:
                S = -D
            o, w = core.topk_kernel_batch(S, cfg["k"], cfg["kref"])
            ws = (w / w.sum(1, keepdims=True)).astype(np.float32)
            a5 = np.einsum("nk,nkij->nij", ws, T.act[o][:, :, :dims.EXEC_STEPS, dims.ACT_VALID])
            head = (a5 / self.sig32).reshape(len(m), core.NH)
            disp5 = np.sqrt(np.mean((T.Hc[o[:, :5]] - head[:, None]) ** 2, axis=(1, 2)))
            d1 = np.take_along_axis(D, o[:, :1], 1)[:, 0]
            if reg == 0:
                conf = -disp5
            elif reg == 1:
                zs = self.zs
                conf = (-(d1 - zs["d1"][0]) / zs["d1"][1] - (disp5 - zs["disp5"][0]) / zs["disp5"][1]
                        - (dst[m] - zs["dst"][0]) / zs["dst"][1])
            else:
                conf = -d1 / med - disp5 / self.s_a
            out["a5"][m] = a5
            out["d1"][m], out["disp5"][m], out["conf"][m] = d1, disp5, conf
            out["top1"][m] = T.rows[o[:, 0]]
        out["dst"] = dst
        return out
