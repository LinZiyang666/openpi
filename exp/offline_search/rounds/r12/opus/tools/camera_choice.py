"""Per-task camera choice for the half-everywhere system (layer 5), decided on B-dev pure-cache data only.

Inputs (B pool only): journals of the pure-cache ablation arms `both` and `wrist_sub` of
  - the B screen roots  rounds/r12/ablate/pilots/r12_ablate_<cell>/runs/r12_ablate_<cell>_{both,wrist_sub}
  - the camdev roots    r12_camdev_<cell> (under rounds/r12/ablate/camdev/ or trace_runs/os_closed_loop/), same arms
and the library-only per-task signal out/design/libsignal_<cell>.npz (design_libsignal.py).
Refuses any path containing r12_ablate_test_ / r12_test_ (test A).

Rule (frozen 2026-10-04, SPEC_HALF §L5):
  per task t: d_t = SR_wrist - SR_both over the pooled episodes (unpaired estimate; pairing ignored => conservative),
      se_t^2 = [p~w(1-p~w) + p~b(1-p~b)] / n_t,  p~ = (successes+1)/(n+2)
  random-effects (DerSimonian-Laird) per cell: tau^2, mu; optional library covariate (de_t, centred) enters as a
      meta-regression slope only if its WLS z <= -2 (expected sign: larger offline wrist error -> worse closed loop)
  shrunk estimate  D_t = m_t + B_t (d_t - m_t),  B_t = tau^2/(tau^2+se_t^2),  m_t = mu (+ beta*(de_t - mean de))
  camera_t = wrist  iff  D_t > -C_cell;  otherwise both (ties -> both)
  C_cell = max(.005, s_mix * kappa * u / alpha): the pure-cache SR loss that the IR saved by the wrist camera on that
      task could buy back with the knob, at the B-measured whole-episode mixing-line rate s_mix = (library pure-policy
      SR - B-dev deployed SR)/(.5 - B-dev deployed IR), kappa = .076/.074, u = B-dev non-call looks per decision,
      alpha = .65 (B-dev pure-cache -> full-system failure transfer); frozen values in C_CELL
Outputs JSON with every intermediate number."""
import glob, json, math, os, sys
import numpy as np

REPO = "/home/weiland/projects/openpi"
TRACE = "/home/weiland/trace_runs/os_closed_loop"
CELLS = ["pi05_l10_50", "pi05_spatial_50", "pi05_spatial_200", "groot_spatial_50", "groot_l10_50"]
C_TOL = 0.010   # legacy uniform value (used only by design_predict.py when no cell given)
C_CELL = {"pi05_l10_50": .008, "pi05_spatial_50": .012, "pi05_spatial_200": .005, "groot_spatial_50": .008, "groot_l10_50": .021}


def journal(run):
    if "r12_ablate_test_" in run or "/r12_test_" in run:
        raise ValueError(f"test-A path refused: {run}")
    comp = {}
    for line in open(run + "/client/journal.jsonl"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            t, i = r["task_uid"].split(":")[-2:]
            comp[(int(t), int(i))] = bool(r.get("success"))
    return comp


def arm_runs(cell, arm, sources):
    runs = []
    if "screen" in sources:
        runs += glob.glob(f"{REPO}/exp/offline_search/rounds/r12/ablate/pilots/r12_ablate_{cell}/runs/r12_ablate_{cell}_{arm}")
    if "camdev" in sources:
        runs += glob.glob(f"{REPO}/exp/offline_search/rounds/r12/ablate/camdev/r12_camdev_{cell}*/runs/r12_camdev_{cell}*_{arm}")
        runs += glob.glob(f"{TRACE}/r12_camdev_{cell}*/runs/r12_camdev_{cell}*_{arm}")
    return sorted({os.path.realpath(r) for r in runs if os.path.exists(r + "/client/journal.jsonl")})


def dl(d, se2, X=None):
    """DerSimonian-Laird random effects with optional single covariate. Returns mu (or (a,b)), tau2."""
    w = 1 / se2
    if X is None:
        mu_f = np.sum(w * d) / np.sum(w)
        Q = np.sum(w * (d - mu_f) ** 2)
        tau2 = max(0.0, (Q - (len(d) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
        ws = 1 / (se2 + tau2)
        return dict(mu=float(np.sum(ws * d) / np.sum(ws)), tau2=float(tau2), Q=float(Q))
    A = np.c_[np.ones_like(d), X]
    W = np.diag(w)
    beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ d)
    r = d - A @ beta
    Q = float(r @ W @ r)
    P = W - W @ A @ np.linalg.solve(A.T @ W @ A, A.T @ W)
    tau2 = max(0.0, (Q - (len(d) - 2)) / np.trace(P))
    Ws = np.diag(1 / (se2 + tau2))
    cov = np.linalg.inv(A.T @ Ws @ A)
    beta = cov @ A.T @ Ws @ d
    return dict(a=float(beta[0]), b=float(beta[1]), z_b=float(beta[1] / math.sqrt(cov[1, 1])), tau2=float(tau2), Q=Q)


def decide(cell, sources=("screen", "camdev"), use_cov=True):
    J = {}
    for arm in ("both", "wrist_sub"):
        J[arm] = {}
        for run in arm_runs(cell, arm, sources):
            for k, v in journal(run).items():
                J[arm].setdefault(k, []).append(v)
    tasks = range(10)
    n_w = np.array([sum(len(v) for k, v in J["wrist_sub"].items() if k[0] == t) for t in tasks], float)
    s_w = np.array([sum(sum(v) for k, v in J["wrist_sub"].items() if k[0] == t) for t in tasks], float)
    n_b = np.array([sum(len(v) for k, v in J["both"].items() if k[0] == t) for t in tasks], float)
    s_b = np.array([sum(sum(v) for k, v in J["both"].items() if k[0] == t) for t in tasks], float)
    d = s_w / n_w - s_b / n_b
    pw, pb = (s_w + 1) / (n_w + 2), (s_b + 1) / (n_b + 2)
    se2 = pw * (1 - pw) / n_w + pb * (1 - pb) / n_b
    out = dict(cell=cell, sources=list(sources), runs={a: arm_runs(cell, a, sources) for a in J},
               n_wrist=n_w.tolist(), n_both=n_b.tolist(), sr_wrist=(s_w / n_w).tolist(), sr_both=(s_b / n_b).tolist(),
               d=d.tolist(), se=np.sqrt(se2).tolist(), C=C_CELL[cell])
    base = dl(d, se2)
    out["re"] = base
    m = np.full(10, base["mu"]); tau2 = base["tau2"]
    lib = f"{REPO}/exp/offline_search/rounds/r12/opus/out/design/libsignal_{cell}.npz"
    if use_cov and os.path.exists(lib):
        z = np.load(lib)
        de = np.array([np.mean(z[f"t{t}_e_w"] - z[f"t{t}_e_f"]) for t in tasks])
        X = de - de.mean()
        reg = dl(d, se2, X)
        out["meta_regression"] = dict(reg, de=de.tolist())
        if reg["z_b"] <= -2:
            m = reg["a"] + reg["b"] * X; tau2 = reg["tau2"]; out["covariate_used"] = True
        else:
            out["covariate_used"] = False
    Bt = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros(10)
    D = m + Bt * (d - m)
    post_sd = np.sqrt(tau2 * se2 / (tau2 + se2)) if tau2 > 0 else np.zeros(10)
    se_mu = float(math.sqrt(1 / np.sum(1 / (se2 + tau2))))
    out.update(shrink=Bt.tolist(), D=D.tolist(), post_sd=post_sd.tolist(), se_mu=se_mu,
               camera=["wrist" if D[t] > -C_CELL[cell] else "both" for t in tasks])
    out["wrist_tasks"] = [t for t in tasks if out["camera"][t] == "wrist"]
    return out


if __name__ == "__main__":
    srcs = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("screen", "camdev")
    res = {c: decide(c, srcs) for c in CELLS}
    for c, r in res.items():
        print(f"\n== {c}  sources={r['sources']}  n/task wrist {r['n_wrist'][0]:.0f} both {r['n_both'][0]:.0f}")
        print(f"   mu {r['re']['mu']:+.3f}  tau {math.sqrt(r['re']['tau2']):.3f}  Q {r['re']['Q']:.1f}" +
              (f"   meta-reg b {r['meta_regression']['b']:+.2f} z {r['meta_regression']['z_b']:+.2f} used={r['covariate_used']}" if 'meta_regression' in r else ""))
        for t in range(10):
            print(f"   task {t}: both {r['sr_both'][t]:.2f} wrist {r['sr_wrist'][t]:.2f}  d {r['d'][t]:+.3f} se {r['se'][t]:.3f}  "
                  f"D {r['D'][t]:+.3f} (sd {r['post_sd'][t]:.3f})  -> {r['camera'][t]}")
        print(f"   wrist tasks: {r['wrist_tasks']}")
    tag = "_".join(srcs)
    json.dump(res, open(f"{REPO}/exp/offline_search/rounds/r12/opus/out/design/camera_choice_{tag}.json", "w"), indent=1)
