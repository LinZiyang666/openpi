"""Counterfactual synthesis evaluator: re-synthesize the served chunk from the logged
16 neighbours with alternative rules and score every rule against the policy shadow.

At a look decision the pure cache logs the 16 retrieved rows and their kernel
weights; the deferred ``policy_shadow`` gives the policy's chunk for the same
observation. The fitted method artifact supplies the library chunks and the
per-task whitened codes, so any synthesis rule that uses only the neighbours,
their codes and the query code can be scored offline:

* ``kernel``      the deployed rule (reproduces the served chunk; sanity check)
* ``top1``        nearest neighbour only
* ``uniform_k``   plain mean of the k nearest
* ``kernel_kref`` the deployed kernel with another width
* ``vote_grip``   kernel motion + weighted majority sign for the gripper
* ``llr_state``   local linear correction: ridge regression of the neighbours'
                  chunks on their robot-state offset from the query
* ``llr_code``    the same in the leading principal directions of the codes

Score = RMS motion gap in library-sigma units and gripper-sign disagreement on
the executed 5 controls, relative to the shadow. Distance to the policy is a
screening signal (R8: a 90-query offline check under-predicted a 30 pp
closed-loop drop); it ranks candidates, it does not certify them.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler

from . import common, shadow_gap


def load_fit(arm):
    rows = {r["arm"]: r for r in common.load_arms()}
    path = rows[arm]["r8"]["source"]["artifact"]
    with open(path, "rb") as f:
        blob = FitUnpickler(f).load()
    return blob["method"], path


def query_codes(method, task_ids, keys_third, keys_wrist, state_norm):
    """Whitened codes of queries under the fitted main (stale) metric, per task."""
    x = np.concatenate([keys_third, keys_wrist, state_norm[:, :8]], axis=1).astype(np.float32)
    z = np.empty((len(x), method.tasks[int(task_ids[0])].Wf.shape[1]), np.float32)
    for t in np.unique(task_ids):
        T = method.tasks[int(t)]
        m = task_ids == t
        z[m] = x[m] @ T.Wf - T.shift
    return z


def neighbour_codes(method, task_ids, rows):
    """Codes of the logged neighbour rows (library-global row ids -> per-task code table)."""
    out = np.empty(rows.shape + (method.tasks[int(task_ids[0])].Z.shape[1],), np.float32)
    for t in np.unique(task_ids):
        T = method.tasks[int(t)]
        m = task_ids == t
        pos = np.searchsorted(T.rows, rows[m])
        if not (T.rows[pos] == rows[m]).all():
            raise ValueError("neighbour row not in task candidate set")
        out[m] = T.Z[pos]
    return out


def synth_rules(method, rows, dist, zq, zn, rs_q, rs_n, kref_list=(2, 3, 5, 8, 12), ks=(1, 4, 8, 16), ridge=1.0):
    """Return {rule: (n, 10, 7) chunks} for the logged neighbours (sorted by distance)."""
    act = method.act[:, :10, :7]                      # (L, 10, 7) normalized library chunks
    C = act[rows]                                     # (n, 16, 10, 7)
    out = {}
    rel = dist - dist[:, :1]
    for kr in kref_list:
        w = _kernel_w(rel, kr)
        w = w / w.sum(1, keepdims=True)
        out[f"kernel_kref{kr}"] = np.einsum("nk,nkhd->nhd", w, C)
    w_dep = _kernel_w(rel, method.kref)
    w_dep = w_dep / w_dep.sum(1, keepdims=True)
    base = np.einsum("nk,nkhd->nhd", w_dep, C)
    out["kernel"] = base
    for k in ks:
        out[f"uniform_{k}"] = C[:, :k].mean(1)
    # gripper by weighted majority of signs, motion from the deployed kernel
    vote = np.einsum("nk,nkh->nh", w_dep, np.where(C[:, :, :, 6] >= 0, 1.0, -1.0))
    g = base.copy()
    g[:, :, 6] = np.where(vote >= 0, 1.0, -1.0)
    out["vote_grip"] = g
    # local linear ridge corrections (weighted by the deployed kernel)
    def llr(X_n, X_q, lam):
        pred = np.empty_like(base)
        for i in range(len(rows)):
            dX = X_n[i] - X_q[i]                       # (16, p)
            W = w_dep[i]
            Xd = np.concatenate([np.ones((16, 1)), dX], 1)
            A = (Xd * W[:, None]).T @ Xd + lam * np.diag([0.0] + [1.0] * dX.shape[1])
            Y = C[i].reshape(16, -1)
            beta = np.linalg.solve(A, (Xd * W[:, None]).T @ Y)
            pred[i] = beta[0].reshape(10, 7)
        return pred
    out["llr_state"] = llr(rs_n, rs_q, ridge)
    # leading principal directions of the neighbour codes (per query), 3 dims
    pcs_n, pcs_q = np.empty(rows.shape + (3,), np.float32), np.empty((len(rows), 3), np.float32)
    for i in range(len(rows)):
        Zc = zn[i] - zn[i].mean(0)
        _, _, vt = np.linalg.svd(Zc, full_matrices=False)
        pcs_n[i] = zn[i] @ vt[:3].T
        pcs_q[i] = zq[i] @ vt[:3].T
    out["llr_code3"] = llr(pcs_n, pcs_q, ridge)
    out["llr_code3_r10"] = llr(pcs_n, pcs_q, 10.0)
    out["llr_state_r10"] = llr(rs_n, rs_q, 10.0)
    return out


def evaluate_arm(arm, derived=common.DERIVED, max_decisions=None, seed=0):
    model, suite, lib, variant = common.parse_arm(arm)
    method, fit_path = load_fit(arm)
    dec, arr = shadow_gap.load_arm(arm, derived)
    norm = np.load(Path(derived) / "arrays_norm" / f"{arm}.npz")
    if not (norm["decision_id"] == dec.decision_id.values).all():
        raise ValueError("state_norm order mismatch")
    sigma = shadow_gap.library_sigma(model, suite)
    look = dec.is_look.values & (dec.decision_seq.values > 0) & (arr["rows"] >= 0).all(1) & common.discovery_mask(dec["init"].values)
    idx = np.flatnonzero(look)
    if max_decisions and len(idx) > max_decisions:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_decisions, replace=False))
    task = dec.task_id.values[idx]
    rows = arr["rows"][idx].astype(np.int64)
    zq = query_codes(method, task, arr["keys_third"][idx], arr["keys_wrist"][idx], norm["state_norm"][idx])
    zn = neighbour_codes(method, task, rows)
    dist = np.sqrt(((zn - zq[:, None]) ** 2).sum(-1))
    # sanity: logged order is by increasing distance; logged weights are the deployed kernel
    order_ok = float(np.mean(np.all(np.diff(dist, axis=1) >= -1e-3, axis=1)))
    w_logged = arr["weights"][idx]
    w_re = _kernel_w(dist - dist[:, :1], method.kref)
    w_re = w_re / w_re.sum(1, keepdims=True)
    weight_err = float(np.abs(w_re - w_logged).max())
    rs_q = norm["state_norm"][idx][:, :8]
    rs_n = method.blind_rs[rows] if hasattr(method, "blind_rs") else None
    if rs_n is None:
        raise ValueError("fit lacks blind_rs (library robot state)")
    chunks = synth_rules(method, rows, dist, zq, zn, rs_q, rs_n)
    served, shadow = arr["served"][idx], arr["shadow"][idx]
    served_err = float(np.abs(chunks["kernel"][:, :5] - served[:, :5]).max())
    res = dict(arm=arm, fit=fit_path, n=int(len(idx)), order_ok=order_ok, weight_recon_maxerr=weight_err,
               served_recon_maxerr=served_err, rules={})
    m0, g0, _ = shadow_gap.gap_metrics(served, shadow, sigma)
    res["rules"]["served"] = dict(motion=float(m0.mean()), grip=float(g0.mean()))
    per_task = {}
    for name, ch in chunks.items():
        m, g, _ = shadow_gap.gap_metrics(ch, shadow, sigma)
        res["rules"][name] = dict(motion=float(m.mean()), grip=float(g.mean()),
                                  motion_delta_vs_served=float((m - m0).mean()),
                                  motion_delta_ci=[float(x) for x in _paired_ci(m - m0, task, seed)],
                                  grip_delta_vs_served=float((g - g0).mean()))
        per_task[name] = pd.DataFrame(dict(task=task, m=m)).groupby("task").m.mean().round(4).to_dict()
    res["per_task_motion"] = per_task
    return res


def _paired_ci(delta, task, seed=0, n_boot=500):
    boot = common.task_strat_bootstrap(delta, task, n_boot=n_boot, seed=seed)
    return common.ci(boot[:, 0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--out", default=str(common.OUT / "synth"))
    ap.add_argument("--max-decisions", type=int, default=6000)
    a = ap.parse_args(argv)
    arms = a.arms or [common.arm_name(m, s, l, "A") for m, s, l in common.CELLS]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        r = evaluate_arm(arm, max_decisions=a.max_decisions)
        common.write_json(out / f"{arm}.json", r)
        print(f"\n== {arm} n={r['n']} order_ok={r['order_ok']:.3f} weight_err={r['weight_recon_maxerr']:.2e} served_err={r['served_recon_maxerr']:.2e}")
        for k, v in sorted(r["rules"].items(), key=lambda kv: kv[1]["motion"]):
            d = v.get("motion_delta_vs_served")
            ci = v.get("motion_delta_ci")
            extra = f"  d_motion={d:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] d_grip={v['grip_delta_vs_served']:+.4f}" if d is not None else ""
            print(f"   {k:16s} motion {v['motion']:.4f}  grip {v['grip']:.4f}{extra}")
    return 0


if __name__ == "__main__":
    main()
