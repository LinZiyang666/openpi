"""Round-2 corrector lab: ridge/RFF residual heads fitted on inits 0-19, scored on inits 20-29.

Rule-1 compliance: every loader filters to discovery inits (0-29) before anything is
computed; fitting uses ``TRAIN_INITS`` (0-19) and every evaluation uses ``EVAL_INITS``
(20-29). Nothing with init >= 30 is read into memory beyond the row filter.

Head (astra's recipe, re-implemented): inputs = PCA keys (128) + state (8) + cached
10x7 chunk (70) + capped decision index (1) [+ task one-hot (10) for the single
task-agnostic head]; features = standardized inputs + 384 random Fourier features;
ridge (alpha 100); target = 10-step motion residual (policy shadow - cached chunk),
optionally also the gripper channel. Equal weight per episode.

Phase labels for the analysis come from the executed gripper sign history of the
episode (approach / grasp / carry / release / post), so they need no simulator truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r09.explore_fable.tools import common, shadow_gap, student

TRAIN_INITS = tuple(range(0, 20))
EVAL_INITS = tuple(range(20, 30))
N_RFF = 384
ALPHA = 100.0
PHASES = ("approach", "grasp", "carry", "release", "post")


def load_cell_discovery(model, suite, lib, derived=common.DERIVED):
    """All look decisions of the cell's arms, inits 0-29 ONLY (rule 1), with episode keys and gripper history."""
    arms = student.cell_arms(model, suite, lib)
    parts = []
    for arm in arms:
        dec = pd.read_parquet(Path(derived) / "decisions" / f"{arm}.parquet")
        arr = np.load(Path(derived) / "arrays" / f"{arm}.npz")
        norm = np.load(Path(derived) / "arrays_norm" / f"{arm}.npz")
        disc = common.discovery_mask(dec["init"].values)          # inits 0-29 only
        keep = disc & dec.is_look.values & np.isfinite(arr["keys_third"]).all(1) & np.isfinite(arr["shadow"]).all((1, 2))
        phase = episode_phases(dec, arr["served"])                # uses executed gripper history (all decisions of the episode)
        idx = np.flatnonzero(keep)
        parts.append(dict(arm=np.array([arm] * len(idx)), ep=dec.episode_key.values[idx], task=dec.task_id.values[idx],
                          init=dec["init"].values[idx], seq=dec.decision_seq.values[idx], k3=arr["keys_third"][idx],
                          kw=arr["keys_wrist"][idx], st=norm["state_norm"][idx][:, :8], served=arr["served"][idx],
                          shadow=arr["shadow"][idx], src=dec.src.values[idx].astype(str), phase=phase[idx],
                          success=dec.journal_success.values[idx].astype(bool), ndec=dec.groupby("episode_key").decision_seq.transform("max").values[idx] + 1))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def episode_phases(dec, served):
    """Phase of every decision from the executed gripper sign sequence (5 controls per decision)."""
    phase = np.full(len(dec), "post", dtype=object)
    g = np.sign(served[:, :5, 6])                                    # LIBERO convention: +1 = close, -1 = open
    for ek, idx in dec.groupby("episode_key").indices.items():
        idx = idx[np.argsort(dec.decision_seq.values[idx])]
        closed = (g[idx] > 0).mean(1) > 0.5                         # decision-level gripper state
        n = len(idx)
        first_close = int(np.argmax(closed)) if closed.any() else n
        state = np.where(closed, "carry", "open")
        lab = np.empty(n, dtype=object)
        for i in range(n):
            if i < first_close - 1:
                lab[i] = "approach"
            elif closed[i]:
                lab[i] = "carry"
            else:
                lab[i] = "post"
        # transitions: open->close within +-1 decision = grasp; close->open = release
        for i in range(1, n):
            if not closed[i - 1] and closed[i]:
                for j in (i - 2, i - 1, i):
                    if 0 <= j < n:
                        lab[j] = "grasp"
            if closed[i - 1] and not closed[i]:
                for j in (i - 1, i, i + 1):
                    if 0 <= j < n:
                        lab[j] = "release"
        phase[idx] = lab
    return phase


def inputs(D, sigma, task_onehot=True):
    cols = [D["k3"], D["kw"], D["st"], (D["served"][:, :10, :7] / sigma).reshape(len(D["task"]), -1),
            (np.minimum(D["seq"], 120) / 120.0)[:, None]]
    if task_onehot:
        cols.append(np.eye(10, dtype=np.float32)[D["task"]])
    return np.concatenate(cols, 1).astype(np.float32)


def fit_head(X, Y, weights, seed=0, n_rff=N_RFF, alpha=ALPHA):
    """Standardize, append cos random features, weighted ridge. Returns a serving dict (numpy only)."""
    rng = np.random.default_rng(seed)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xn = np.clip((X - mean) / std, -8, 8)
    W = (rng.standard_normal((X.shape[1], n_rff)) / np.sqrt(X.shape[1])).astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, n_rff).astype(np.float32)
    F = np.concatenate([Xn, np.cos(Xn @ W + b) * np.sqrt(2)], 1).astype(np.float64)
    sw = np.sqrt(weights / weights.mean())[:, None]
    Fw, Yw = F * sw, Y.astype(np.float64) * sw
    mu_f, mu_y = (Fw * sw).sum(0) / (sw ** 2).sum(), (Yw * sw).sum(0) / (sw ** 2).sum()
    Fc, Yc = Fw - mu_f * sw, Yw - mu_y * sw
    A = Fc.T @ Fc + alpha * np.eye(F.shape[1])
    coef = np.linalg.solve(A, Fc.T @ Yc).T                        # (out, feat)
    intercept = mu_y - coef @ mu_f
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32), w=W, bias=b, coef=coef.astype(np.float32), intercept=intercept.astype(np.float32))


def predict(model, X):
    Xn = np.clip((X - model["mean"]) / model["std"], -8, 8)
    F = np.concatenate([Xn, np.cos(Xn @ model["w"] + model["bias"]) * np.sqrt(2)], 1)
    return F @ model["coef"].T + model["intercept"]


def train_cell(model_name, suite, lib, out_dir, with_gripper=False, task_onehot=True, per_task=False, seed=0):
    D = load_cell_discovery(model_name, suite, lib)
    sigma = shadow_gap.library_sigma(model_name, suite)
    tr = np.isin(D["init"], TRAIN_INITS)
    X = inputs(D, sigma, task_onehot)
    chans = 7 if with_gripper else 6
    Y = ((D["shadow"][:, :10, :chans] - D["served"][:, :10, :chans]) / sigma[:chans]).reshape(len(D["task"]), -1)
    ep_id = pd.factorize(D["ep"])[0]
    counts = np.bincount(ep_id)
    weights = 1.0 / counts[ep_id]
    heads = {}
    if per_task:
        for t in range(10):
            m = tr & (D["task"] == t)
            heads[t] = fit_head(X[m], Y[m], weights[m], seed=seed)
    else:
        heads["all"] = fit_head(X[tr], Y[tr], weights[tr], seed=seed)
    meta = dict(cell=f"{model_name}_{suite}_{lib}", train_inits=list(TRAIN_INITS), eval_inits=list(EVAL_INITS), n_train=int(tr.sum()),
                with_gripper=with_gripper, task_onehot=task_onehot, per_task=per_task, chans=chans, n_rff=N_RFF, alpha=ALPHA,
                arms=sorted(set(D["arm"])), sigma=sigma.tolist())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name}_{suite}_{lib}_{'grip' if with_gripper else 'motion'}_{'pertask' if per_task else 'single'}{'' if task_onehot else '_notask'}"
    arrays = {f"{k}_{name}": v for k, h in heads.items() for name, v in h.items()}
    np.savez(out_dir / f"head_{tag}.npz", **arrays, meta_json=json.dumps(meta))
    return D, X, Y, heads, meta, tag


def apply_head(heads, X, task, chans):
    out = np.zeros((len(X), 10 * chans), np.float32)
    if "all" in heads:
        return predict(heads["all"], X).astype(np.float32)
    for t, h in heads.items():
        m = task == t
        if m.any():
            out[m] = predict(h, X[m])
    return out


def evaluate(D, X, heads, meta, blends=(0.5, 1.0), eval_arm_only=True):
    """Held-out (inits 20-29) motion/gripper gap by phase for served vs corrected chunks."""
    sigma = np.asarray(meta["sigma"], np.float32)
    chans = meta["chans"]
    va = np.isin(D["init"], EVAL_INITS)
    if eval_arm_only:
        va &= (D["src"] == "cache") & np.char.endswith(D["arm"].astype(str), "_A")
    corr = apply_head(heads, X[va], D["task"][va], chans).reshape(-1, 10, chans) * sigma[:chans]
    served, shadow, phase = D["served"][va], D["shadow"][va], D["phase"][va]
    rows = []
    def score(name, ch):
        m, g, c10 = shadow_gap.gap_metrics(ch, shadow, sigma)
        row = dict(rule=name, motion=float(m.mean()), motion10=float(c10.mean()), grip=float(g.mean()))
        for p in PHASES:
            sel = phase == p
            row[f"motion_{p}"] = float(m[sel].mean()) if sel.any() else np.nan
            row[f"n_{p}"] = int(sel.sum())
        rows.append(row)
    score("served", served)
    for b in blends:
        ch = served.copy()
        ch[:, :10, :chans] += b * corr
        score(f"corr_{b}", ch)
        if chans == 7:   # gripper: use the corrected channel's sign
            ch2 = ch.copy()
            ch2[:, :, 6] = np.where(ch[:, :, 6] >= 0, 1.0, -1.0)
            score(f"corr_{b}_gripsign", ch2)
    return pd.DataFrame(rows).set_index("rule")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default="pi05:spatial:50,pi05:l10:50,groot:l10:50")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "out" / "corrector"))
    ap.add_argument("--gripper", action="store_true")
    ap.add_argument("--per-task", action="store_true")
    ap.add_argument("--no-task-input", action="store_true")
    a = ap.parse_args(argv)
    for c in a.cells.split(","):
        m, s, l = c.split(":")
        D, X, Y, heads, meta, tag = train_cell(m, s, int(l), a.out, with_gripper=a.gripper, task_onehot=not a.no_task_input, per_task=a.per_task)
        table = evaluate(D, X, heads, meta)
        table.round(3).to_csv(Path(a.out) / f"eval_{tag}.csv")
        print(f"\n== {tag} n_train={meta['n_train']} (inits 0-19), eval = A arm cache decisions inits 20-29")
        print(table.round(3).to_string())
    return 0


if __name__ == "__main__":
    main()
