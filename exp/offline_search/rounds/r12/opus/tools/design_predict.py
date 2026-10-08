"""Predictive simulation for the half-everywhere system (B-only primary; an 'A-informed' variant labelled as such).
Integrates over: the per-task camera effect (posterior from the B screen), the camdev data still to come (35 or 20
inits per task per arm), the frozen camera rule (camera_choice.py, without the optional covariate), the transfer
from pure cache to the full system (alpha), the guard swap on wrist tasks, and run noise."""
import json, math, sys
import numpy as np
sys.path.insert(0, "/home/weiland/projects/openpi")
from exp.offline_search.rounds.r12.opus.tools.camera_choice import dl, C_CELL

D = "/home/weiland/projects/openpi/exp/offline_search/rounds/r12/opus/out/design"
bs = json.load(open(f"{D}/bscreen_pertask.json"))
led = json.load(open(f"{D}/pertask_ledger.json"))
CELLS = ["pi05_l10_50", "pi05_spatial_50", "pi05_spatial_200", "groot_spatial_50", "groot_l10_50"]
KAPPA = {"pi05": .076, "groot": .074}
CAMDEV_N = {"pi05_spatial_200": 16}   # camdev HANDBACK: 27-32 (size 50), 15-19 (Sp-200) per task
# B pilot (B-dev, 100 eps) H2-vs-deployed IR change for the whole cell (all tasks with H2), from design_score_pilot.py
DH2 = {"pi05_l10_50": -.0140, "pi05_spatial_50": -.0104, "pi05_spatial_200": -.0041, "groot_spatial_50": -.0072, "groot_l10_50": -.0024}
# per-task guard-swap SR effect (pp at cell level if all tasks swap): frozen in PILOT_ANALYSIS §3 (500-ep estimate)
H_SWAP = {"pi05_l10_50": -.005, "pi05_spatial_50": 0., "pi05_spatial_200": 0., "groot_spatial_50": .003, "groot_l10_50": -.005}
# test-A paired SE of a cell-level SR difference (scorer CIs of R12 test-A method arms / 1.96), and B-dev deployed pools
SE_A = {"pi05_l10_50": .018, "pi05_spatial_50": .013, "pi05_spatial_200": .009, "groot_spatial_50": .017, "groot_l10_50": .022}
BDEP = {"pi05_l10_50": (.811, 700, .1678), "pi05_spatial_50": (.886, 800, .1464), "pi05_spatial_200": (.978, 400, .1016),
        "groot_spatial_50": (.88, 200, .1308), "groot_l10_50": (.70, 100, .2150)}   # GR00T: H100 replicates only
A_INFORMED_MU = {"pi05_l10_50": -.030, "pi05_spatial_50": .004, "pi05_spatial_200": -.008, "groot_spatial_50": .076, "groot_l10_50": -.110}


def sim(cell, rng, R=4000, a_informed=False, skeptic=False):
    r = bs[cell]
    nb = np.array([len(r[str(t)]["both"]) for t in range(10)], float)
    sb = np.array([sum(r[str(t)]["both"]) for t in range(10)], float)
    nw = np.array([len(r[str(t)]["wrist_sub"]) for t in range(10)], float)
    sw = np.array([sum(r[str(t)]["wrist_sub"]) for t in range(10)], float)
    d = sw / nw - sb / nb
    pw0, pb0 = (sw + 1) / (nw + 2), (sb + 1) / (nb + 2)
    se2 = pw0 * (1 - pw0) / nw + pb0 * (1 - pb0) / nb
    fit = dl(d, se2)
    tau2 = max(fit["tau2"], .05 ** 2)
    se_mu = math.sqrt(1 / np.sum(1 / (se2 + tau2)))
    mu_hat = fit["mu"]
    if a_informed and cell in A_INFORMED_MU:
        mu_hat, se_mu = A_INFORMED_MU[cell], .02
    L = led[cell]; om, u = np.array(L["omega"]), np.array(L["u"])
    model = cell.split("_")[0]
    pb_true = (sb + 1) / (nb + 2) * .5 + (sb.sum() / nb.sum()) * .5
    ncd = CAMDEV_N.get(cell, 28)
    out = {k: [] for k in ("nW", "dSR_A", "dIR_A", "dSR_B", "dIR_B", "cam", "dSR_true")}
    for _ in range(R):
        mu = rng.normal(mu_hat, se_mu)
        Bt = tau2 / (tau2 + se2)
        post_m = mu + Bt * (d - mu) if not (a_informed and cell in A_INFORMED_MU) else mu + Bt * (d - fit["mu"])
        delta = rng.normal(post_m, np.sqrt(tau2 * se2 / (tau2 + se2))) if not skeptic else np.full(10, mu)
        pb = np.clip(pb_true, .02, .98); pw = np.clip(pb + delta, .0, 1.)
        delta = pw - pb
        # camdev + screen pooled, rule
        cb = rng.binomial(ncd, pb); cw = rng.binomial(ncd, pw)
        Nb, Nw, Sb, Sw = nb + ncd, nw + ncd, sb + cb, sw + cw
        dd = Sw / Nw - Sb / Nb
        qw, qb = (Sw + 1) / (Nw + 2), (Sb + 1) / (Nb + 2)
        s2 = qw * (1 - qw) / Nw + qb * (1 - qb) / Nb
        f = dl(dd, s2)
        Bk = f["tau2"] / (f["tau2"] + s2) if f["tau2"] > 0 else np.zeros(10)
        Dk = f["mu"] + Bk * (dd - f["mu"])
        W = Dk > -C_CELL[cell]
        alpha = rng.uniform(.35, 1.0)
        fW = W.mean(); oW = om[W].sum()
        dsr_true = (alpha * delta[W]).sum() / 10 + fW * rng.normal(H_SWAP[cell], .015)
        # fewer (more) failures -> fewer (more) guard calls: dIR = -k * dSR, k ~ U(.1,.4) (B-dev calls per failing vs
        # winning episode, PILOT_ANALYSIS §1: (20.5-3.5)x.848/60 = .24 pi05 L10 ... .36 Sp-200)
        dir_true = -KAPPA[model] * (om[W] * u[W]).sum() + oW * rng.normal(DH2[cell], .004) - rng.uniform(.1, .4) * dsr_true
        out["nW"].append(W.sum()); out["dSR_true"].append(dsr_true)
        out["dSR_A"].append(dsr_true + rng.normal(0, SE_A[cell])); out["dIR_A"].append(dir_true + rng.normal(0, .003))
        p, n, _ = BDEP[cell]
        out["dSR_B"].append(dsr_true + rng.normal(0, math.sqrt(p * (1 - p) * (1 / 100 + 1 / n))))
        out["dIR_B"].append(dir_true + rng.normal(0, .005))
        out["cam"].append(-KAPPA[model] * (om[W] * u[W]).sum())
    return {k: np.asarray(v) for k, v in out.items()}


def q(x, lo=10, hi=90):
    return f"{np.mean(x):+.4f} [{np.percentile(x, lo):+.4f}, {np.percentile(x, hi):+.4f}]"


def main_unconditional():
    rng = np.random.default_rng(20261004)
    res = {}
    for variant, skeptic in ((False, False), (False, True), (True, False), (True, True)):
        tag = ("A_informed" if variant else "B_only") + ("_skeptic" if skeptic else "")
        print(f"\n######## {tag}")
        pooled = {k: [] for k in ("dSR_A", "dIR_A", "dSR_B", "dIR_B")}
        for cell in CELLS:
            s = sim(cell, rng, a_informed=variant, skeptic=skeptic)
            print(f"{cell:18s} wrist tasks {np.mean(s['nW']):.1f} (P0 {np.mean(s['nW']==0):.2f}, P10 {np.mean(s['nW']==10):.2f})  "
                  f"camera dIR {np.mean(s['cam']):+.4f}\n   A: dSR {q(s['dSR_A'])}  dIR {q(s['dIR_A'])}\n   B: dSR {q(s['dSR_B'])}  dIR {q(s['dIR_B'])}"
                  f"\n   true dSR {q(s['dSR_true'])}")
            res[f"{tag}|{cell}"] = {k: [float(np.mean(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))] for k, v in s.items()}
            for k in pooled:
                pooled[k].append(s[k])
        for k in pooled:
            arr = np.mean(np.vstack(pooled[k]), 0)
            print(f"   pooled (5 cells, equal weights) {k}: {q(arr)}")
            res[f"{tag}|pooled|{k}"] = [float(np.mean(arr)), float(np.percentile(arr, 10)), float(np.percentile(arr, 90))]
    # 50/50 mixture of the hierarchical and the skeptic model (B-only primary; A-informed secondary)
    rng2 = np.random.default_rng(7)
    for base in ("B_only", "A_informed"):
        print(f"\n######## MIXTURE {base} (50% heterogeneity transfers / 50% skeptic)")
        pool = {k: [] for k in ("dSR_A", "dIR_A", "dSR_B", "dIR_B", "nW")}
        for cell in CELLS:
            s1 = sim(cell, rng2, R=3000, a_informed=base == "A_informed")
            s2 = sim(cell, rng2, R=3000, a_informed=base == "A_informed", skeptic=True)
            m = {k: np.r_[s1[k], s2[k]] for k in s1}
            print(f"{cell:18s} wrist tasks {np.mean(m['nW']):.1f} [P(0)={np.mean(m['nW']==0):.2f}]\n   A: dSR {q(m['dSR_A'])}  dIR {q(m['dIR_A'])}\n   B: dSR {q(m['dSR_B'])}  dIR {q(m['dIR_B'])}")
            res[f"MIX_{base}|{cell}"] = {k: [float(np.mean(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))] for k, v in m.items()}
            for k in pool:
                pool[k].append(m[k])
        for k in ("dSR_A", "dIR_A", "dSR_B", "dIR_B"):
            arr = np.mean(np.vstack(pool[k]), 0)
            print(f"   pooled {k}: {q(arr)}")
            res[f"MIX_{base}|pooled|{k}"] = [float(np.mean(arr)), float(np.percentile(arr, 10)), float(np.percentile(arr, 90))]
    json.dump(res, open(f"{D}/predict_sim.json", "w"), indent=1)


def conditional(choice_path, R=20000, seed=11):
    """Frozen post-camdev prediction rule: given camera_choice.py's output (assignment + shrunk D_t, post_sd),
    predict new-minus-deployed dSR/dIR per cell for test A and the B screen. Mechanical; no discretion."""
    ch = json.load(open(choice_path))
    rng = np.random.default_rng(seed)
    out = {}
    print(f"conditional prediction from {choice_path}")
    for cell in CELLS:
        c = ch[cell]; model = cell.split("_")[0]
        W = np.array([x == "wrist" for x in c["camera"]])
        L = led[cell]; om, u = np.array(L["omega"]), np.array(L["u"])
        Dm, Ds = np.array(c["D"]), np.array(c["post_sd"])
        alpha = rng.uniform(.35, 1.0, R)
        Bt = np.array(c["shrink"]); zmu = rng.normal(0, c["se_mu"], R)[:, None]
        delta = Dm[None] + (1 - Bt)[None] * zmu + rng.normal(0, 1, (R, 10)) * Ds[None]
        fW, oW = W.mean(), om[W].sum()
        dsr = alpha * (delta[:, W].sum(1) / 10) + fW * rng.normal(H_SWAP[cell], .015, R)
        cam = -KAPPA[model] * (om[W] * u[W]).sum()
        dir_ = cam + oW * rng.normal(DH2[cell], .004, R) - rng.uniform(.1, .4, R) * dsr
        p, n, _ = BDEP[cell]
        res = dict(wrist_tasks=[int(t) for t in np.flatnonzero(W)], camera_dIR=float(cam),
                   dSR_A=dsr + rng.normal(0, SE_A[cell], R), dIR_A=dir_ + rng.normal(0, .003, R),
                   dSR_B=dsr + rng.normal(0, math.sqrt(p * (1 - p) * (1 / 100 + 1 / n)), R), dIR_B=dir_ + rng.normal(0, .005, R))
        print(f"{cell:18s} wrist {res['wrist_tasks']}  camera dIR {cam:+.4f}\n   A: dSR {q(res['dSR_A'])}  dIR {q(res['dIR_A'])}"
              f"\n   B: dSR {q(res['dSR_B'])}  dIR {q(res['dIR_B'])}")
        out[cell] = {k: ([float(np.mean(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))] if isinstance(v, np.ndarray) else v)
                     for k, v in res.items()}
    json.dump(out, open(choice_path.replace(".json", "_prediction.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "conditional":
        conditional(sys.argv[2])
    else:
        main_unconditional()
