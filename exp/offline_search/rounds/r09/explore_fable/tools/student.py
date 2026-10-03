"""DAgger-style student feasibility: can a small network trained on the policy
shadows beat kernel retrieval at predicting the policy's chunk from the same
inputs the cache sees (PCA keys + robot state)?

The R8 collection is, in effect, DAgger data: at every decision of every arm the
policy labelled the state the *cache controller* visited. Pooling all arms of a
cell gives ~10^5 labelled look decisions. A student that maps
(keys_third, keys_wrist, state, task) -> policy chunk would run at look cost
(IR ~ .076, no stage 2/3) and could replace or correct the kernel synthesis.

Split by init: train on inits 0-19, validate on inits 20-29 (inits 30-49 are the
locked holdout and are never read). Validation is reported on the pure-cache
arm's look decisions (the deployment distribution) in library-sigma units and
gripper-sign disagreement, next to the deployed kernel synthesis (``served``)
and the policy's own draw-to-draw noise floor.

Caveats: these are test-task trajectories (same tasks, disjoint initial
states); a deployable student must be trained on B-pool recordings. Distance
to the shadow is a screening signal, not a success estimate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import common, shadow_gap

TRAIN_INITS = tuple(range(0, 20))
VAL_INITS = tuple(range(20, 30))


def cell_arms(model, suite, lib):
    return [r["arm"] for r in common.load_arms() if r["model"] == model and r["suite_short"] == suite
            and (r["r8"].get("library_size") == lib or r["r8"]["variant"] == "P10")]


def load_cell(model, suite, lib, derived=common.DERIVED, arms=None):
    """Look decisions (seq>0, inits 0-29) of every arm in the cell with finite shadow and keys."""
    arms = arms or cell_arms(model, suite, lib)
    parts = []
    for arm in arms:
        dec = pd.read_parquet(Path(derived) / "decisions" / f"{arm}.parquet")
        arr = np.load(Path(derived) / "arrays" / f"{arm}.npz")
        norm = np.load(Path(derived) / "arrays_norm" / f"{arm}.npz")
        keep = dec.is_look.values & (dec.decision_seq.values > 0) & common.discovery_mask(dec["init"].values)
        keep &= np.isfinite(arr["keys_third"]).all(1) & np.isfinite(arr["shadow"]).all((1, 2))
        idx = np.flatnonzero(keep)
        parts.append(dict(arm=np.array([arm] * len(idx)), task=dec.task_id.values[idx], init=dec["init"].values[idx],
                          seq=dec.decision_seq.values[idx], k3=arr["keys_third"][idx], kw=arr["keys_wrist"][idx],
                          st=norm["state_norm"][idx][:, :8], served=arr["served"][idx], shadow=arr["shadow"][idx],
                          src=dec.src.values[idx].astype(str)))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class MLP(nn.Module):
    def __init__(self, n_in, n_out, width=512, depth=3, dropout=0.1):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU(), nn.Dropout(dropout)]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def features(D, sigma, use_vision=True, use_cache=False):
    cols = [np.eye(10, dtype=np.float32)[D["task"]], D["st"]]
    if use_vision:
        cols = [D["k3"], D["kw"]] + cols
    if use_cache:
        cols.append((D["served"] / sigma).reshape(len(D["task"]), -1))
    return np.concatenate(cols, 1).astype(np.float32)


def fit_student(X, Y, Xv, epochs=40, lr=1e-3, batch=1024, seed=0, device="cuda", width=512, depth=3):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn, Xvn = (X - mu) / sd, (Xv - mu) / sd
    net = MLP(X.shape[1], Y.shape[1], width=width, depth=depth).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Yt = torch.tensor(Xn, device=device), torch.tensor(Y, device=device)
    n = len(Xt)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(torch.tensor(Xvn[i:i + 8192], device=device)) for i in range(0, len(Xvn), 8192)]).cpu().numpy()
    return pred


def evaluate_cell(model, suite, lib, out_dir, device="cuda", epochs=40, seed=0, val_arms=None):
    t0 = time.time()
    D = load_cell(model, suite, lib)
    sigma = shadow_gap.library_sigma(model, suite)
    a_arm = common.arm_name(model, suite, lib, "A")
    tr = np.isin(D["init"], TRAIN_INITS)
    va = np.isin(D["init"], VAL_INITS) & (D["arm"] == a_arm) & (D["src"] == "cache")
    Y = (D["shadow"] / sigma).reshape(len(D["task"]), -1).astype(np.float32)
    res = dict(cell=f"{model}_{suite}_{lib}", n_train=int(tr.sum()), n_val=int(va.sum()), arms=sorted(set(D["arm"])),
               train_inits=list(TRAIN_INITS), val_inits=list(VAL_INITS), rules={})
    shadow_v = D["shadow"][va]
    task_v = D["task"][va]

    def score(name, chunks):
        m, g, c = shadow_gap.gap_metrics(chunks, shadow_v, sigma)
        res["rules"][name] = dict(motion=float(m.mean()), grip=float(g.mean()), commit=float(c.mean()),
                                  per_task_motion=pd.Series(m).groupby(task_v).mean().round(4).to_dict())
        return m

    m_served = score("served", D["served"][va])
    variants = dict(student_vision=(True, False), student_vision_cache=(True, True), student_state_only=(False, False),
                    student_state_cache=(False, True))
    for name, (vis, cache) in variants.items():
        X = features(D, sigma, vis, cache)
        pred = fit_student(X[tr], Y[tr], X[va], epochs=epochs, seed=seed, device=device)
        chunks = (pred.reshape(-1, 10, 7) * sigma).astype(np.float32)
        m = score(name, chunks)
        boot = common.task_strat_bootstrap(m - m_served, task_v, n_boot=300, seed=seed)
        res["rules"][name]["motion_delta_vs_served"] = float((m - m_served).mean())
        res["rules"][name]["motion_delta_ci"] = [float(x) for x in common.ci(boot[:, 0])]
    # in-sample check (how much the student memorizes): score on train rows of the A arm
    res["seconds"] = round(time.time() - t0, 1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(out_dir / f"{model}_{suite}_{lib}.json", res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default="all")
    ap.add_argument("--out", default=str(common.OUT / "student"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    cells = common.CELLS if a.cells == "all" else [tuple(c.split(":")) for c in a.cells.split(",")]
    for model, suite, lib in cells:
        r = evaluate_cell(model, suite, str(lib) and int(lib), a.out, device=a.device, epochs=a.epochs)
        line = "  ".join(f"{k}: m {v['motion']:.4f} g {v['grip']:.4f}" + (f" ({v['motion_delta_vs_served']:+.4f} [{v['motion_delta_ci'][0]:+.4f},{v['motion_delta_ci'][1]:+.4f}])" if "motion_delta_vs_served" in v else "")
                        for k, v in r["rules"].items())
        print(f"{r['cell']} n_train={r['n_train']} n_val={r['n_val']} {r['seconds']}s | {line}", flush=True)
    return 0


# ----------------------------------------------------------------------------- kNN on the shadow-labelled set
def knn_predict(Xtr, Ytr, Xva, task_tr, task_va, k=16, kref=5, device="cuda", chunk=2048):
    """Same-task kernel kNN (AWM kernel, standardized Euclidean) on the shadow-labelled training rows.

    This is 'grow the library with shadow labels, keep the retrieval rule': the baseline that tells whether the
    student's gain comes from the data or from the function class."""
    from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd, device=device)
    B = torch.tensor((Xva - mu) / sd, device=device)
    Yt = torch.tensor(Ytr, device=device)
    ttr = torch.tensor(task_tr, device=device)
    out = np.empty((len(Xva), Ytr.shape[1]), np.float32)
    for i in range(0, len(B), chunk):
        q = B[i:i + chunk]
        d = torch.cdist(q, A)
        same = ttr[None, :] == torch.tensor(task_va[i:i + chunk], device=device)[:, None]
        d = torch.where(same, d, torch.full_like(d, float("inf")))
        dk, idx = torch.topk(d, k, dim=1, largest=False)
        w = torch.tensor(_kernel_w((dk - dk[:, :1]).cpu().numpy(), kref), device=device)
        w = w / w.sum(1, keepdim=True)
        out[i:i + chunk] = torch.einsum("nk,nkd->nd", w, Yt[idx]).cpu().numpy()
    return out


def learning_curve(model, suite, lib, out_dir, device="cuda", epochs=40, seed=0, fractions=(0.1, 0.25, 0.5, 1.0)):
    """Student and kNN accuracy versus the number of shadow-labelled training decisions (episode-level subsets)."""
    D = load_cell(model, suite, lib)
    sigma = shadow_gap.library_sigma(model, suite)
    a_arm = common.arm_name(model, suite, lib, "A")
    tr = np.isin(D["init"], TRAIN_INITS)
    va = np.isin(D["init"], VAL_INITS) & (D["arm"] == a_arm) & (D["src"] == "cache")
    Y = (D["shadow"] / sigma).reshape(len(D["task"]), -1).astype(np.float32)
    X = features(D, sigma, True, False)
    shadow_v, task_v = D["shadow"][va], D["task"][va]
    m_served, _, _ = shadow_gap.gap_metrics(D["served"][va], shadow_v, sigma)
    ep_id = pd.factorize(pd.Series(D["arm"]).astype(str) + ":" + pd.Series(D["task"]).astype(str) + ":" + pd.Series(D["init"]).astype(str))[0]
    rng = np.random.default_rng(seed)
    tr_eps = np.unique(ep_id[tr])
    res = dict(cell=f"{model}_{suite}_{lib}", served_motion=float(m_served.mean()), points=[])
    subsets = [("A_arm_only", tr & (D["arm"] == a_arm))] + [(f"frac_{f}", tr & np.isin(ep_id, rng.choice(tr_eps, int(len(tr_eps) * f), replace=False))) for f in fractions]
    for name, sel in subsets:
        pred_s = fit_student(X[sel], Y[sel], X[va], epochs=epochs, seed=seed, device=device)
        pred_k = knn_predict(X[sel], Y[sel], X[va], D["task"][sel], task_v, device=device)
        ms, _, _ = shadow_gap.gap_metrics((pred_s.reshape(-1, 10, 7) * sigma), shadow_v, sigma)
        mk, gk, _ = shadow_gap.gap_metrics((pred_k.reshape(-1, 10, 7) * sigma), shadow_v, sigma)
        res["points"].append(dict(subset=name, n_decisions=int(sel.sum()), n_episodes=int(len(np.unique(ep_id[sel]))),
                                  student_motion=float(ms.mean()), knn_motion=float(mk.mean()), knn_grip=float(gk.mean())))
        print(res["cell"], name, res["points"][-1], flush=True)
    common.write_json(Path(out_dir) / f"curve_{model}_{suite}_{lib}.json", res)
    return res


def curve_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="pi05:l10:50,groot:l10:50,pi05:spatial:50,pi05:l10:500")
    ap.add_argument("--out", default=str(common.OUT / "student"))
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(argv)
    for c in a.cells.split(","):
        m, s, l = c.split(":")
        learning_curve(m, s, int(l), a.out, epochs=a.epochs)


if __name__ == "__main__":
    import sys
    if "--curve" in sys.argv:
        sys.argv.remove("--curve")
        curve_main()
    else:
        main()
