"""Grow a fitted pure-cache (BlindAWM) artifact with shadow-labelled rows.

Every look decision of a closed-loop arm carries the PCA keys and robot state the
cache saw and, from the deferred augmentation, the chunk the policy would have
produced there. Those (key, state) -> policy-chunk pairs are new library rows
taken exactly where the cache controller goes (DAgger-style labels). This tool
appends them to the frozen A fit as a *registered* library ``grown``:

* the per-task metric (mean/std/W), kernel, scales and confidence constants of
  the frozen fit are kept; only the candidate set grows;
* per-task code tables (Z, z2, HD, h2, RS, rs2, V0/V1, n20) and the global
  arrays (act, lib_ep, lib_step, blind_*) are extended consistently;
* the artifact keeps A's method spec/kwargs/cell, so the deployed arm differs from
  A only by ``--os-fit-artifact`` (no new code on the serving node).

Rows are taken from discovery inits in ``train_inits`` only (default 0-19), so a
closed-loop evaluation on inits 20-29 is held out at the init level. These are
test-task trajectories: the deployable version must regrow from B-pool rollouts.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler

from . import common, shadow_gap, student

GROWN = "grown"


def collect_rows(model, suite, lib, arms, train_inits, derived=common.DERIVED, include_step0=True):
    """Shadow-labelled look decisions: PCA keys, rs8, policy chunk, identity."""
    parts = []
    for arm in arms:
        dec = pd.read_parquet(Path(derived) / "decisions" / f"{arm}.parquet")
        arr = np.load(Path(derived) / "arrays" / f"{arm}.npz")
        norm = np.load(Path(derived) / "arrays_norm" / f"{arm}.npz")
        keep = dec.is_look.values & np.isin(dec["init"].values, list(train_inits))
        if not include_step0:
            keep &= dec.decision_seq.values > 0
        keep &= np.isfinite(arr["keys_third"]).all(1) & np.isfinite(arr["shadow"]).all((1, 2))
        idx = np.flatnonzero(keep)
        ep = pd.factorize(dec.episode_key.values[idx])[0]
        parts.append(dict(task=dec.task_id.values[idx].astype(np.int64), init=dec["init"].values[idx], seq=dec.decision_seq.values[idx],
                          P0=arr["keys_third"][idx], P1=arr["keys_wrist"][idx], rs=norm["state_norm"][idx][:, :8].astype(np.float32),
                          chunk=arr["shadow"][idx], ep=np.array([f"{arm}:{e}" for e in ep]), arm=np.array([arm] * len(idx))))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def extend_method(method, rows, H):
    """Return a deep copy of the fitted BlindAWM whose candidate set is the frozen library plus ``rows``."""
    m = copy.deepcopy(method)
    L0 = m.act.shape[0]
    n = len(rows["task"])
    act_new = np.zeros((n, H, 32), np.float32)
    act_new[:, :rows["chunk"].shape[1], :7] = rows["chunk"][:, :H, :7]
    if rows["chunk"].shape[1] < H:  # GR00T: hold the last labelled row through the unused horizon
        act_new[:, rows["chunk"].shape[1]:, :7] = rows["chunk"][:, -1:, :7]
    m.act = np.concatenate([np.asarray(m.act), act_new]).astype(np.float32)
    ep_codes = pd.factorize(rows["ep"])[0].astype(np.int32) + int(np.asarray(m.lib_ep).max()) + 1
    m.lib_ep = np.concatenate([np.asarray(m.lib_ep), ep_codes]).astype(np.int32)
    m.lib_step = np.concatenate([np.asarray(m.lib_step), rows["seq"].astype(np.int32)]).astype(np.int32)
    m.blind_next = np.concatenate([np.asarray(m.blind_next), np.full(n, -1, np.int32)]).astype(np.int32)
    m.blind_rs = np.concatenate([np.asarray(m.blind_rs), rows["rs"]]).astype(np.float32)
    m.blind_event = np.concatenate([np.asarray(m.blind_event), np.zeros(n, bool)])
    m.blind_terminal = np.concatenate([np.asarray(m.blind_terminal), np.zeros(n, bool)])
    sig = np.asarray(m.sig, np.float64)
    heads = (rows["chunk"][:, :5, :7] / sig).reshape(n, -1).astype(np.float32)
    for t, T in m.tasks.items():
        sel = np.flatnonzero(rows["task"] == t)
        if not len(sel):
            continue
        new_rows = (L0 + sel).astype(np.int64)
        X = np.concatenate([rows["P0"][sel], rows["P1"][sel], rows["rs"][sel]], 1).astype(np.float32)
        Z = (X @ T.Wf - T.shift).astype(np.float32)
        T.rows = np.concatenate([T.rows, new_rows])
        T.Z = np.concatenate([T.Z, Z]).astype(np.float32)
        T.z2 = (T.Z.astype(np.float64) ** 2).sum(1).astype(np.float32)
        T.HD = np.concatenate([T.HD, heads[sel]]).astype(np.float32)
        T.h2 = (T.HD.astype(np.float64) ** 2).sum(1).astype(np.float32)
        T.RS = np.concatenate([T.RS, rows["rs"][sel]]).astype(np.float32)
        T.rs2 = (T.RS.astype(np.float64) ** 2).sum(1).astype(np.float32)
        for fi, key in ((0, "P0"), (1, "P1")):
            Vm = getattr(T, f"Vm{fi}")
            V = rows[key][sel] - Vm
            V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
            setattr(T, f"V{fi}", np.concatenate([getattr(T, f"V{fi}"), V]).astype(np.float32))
        if T.A0 is not None:  # step-0 branch: early code = affine map of the main code
            Y = Z.astype(np.float64) @ T.A0
            if T.As0 is not None:
                Y = Y + rows["rs"][sel].astype(np.float64) @ T.As0
            T.n20 = np.concatenate([T.n20, (Y * Y).sum(1).astype(np.float32)]).astype(np.float32)
        elif T.Z0 is not None:
            raise NotImplementedError("rank-r early codes are not extended")
        order = np.argsort(T.rows, kind="stable")  # searchsorted callers rely on sorted task rows
        for name in ("rows", "Z", "z2", "HD", "h2", "RS", "rs2", "V0", "V1", "n20"):
            v = getattr(T, name, None)
            if v is not None:
                setattr(T, name, np.ascontiguousarray(v[order]))
    m.cand_name = GROWN
    m.os_library = GROWN
    m.n_cand = {t: int(len(T.rows)) for t, T in m.tasks.items()}
    m.name = f"GROWN{n}__" + m.name
    for a in [m.act, m.lib_ep, m.lib_step, m.blind_next, m.blind_rs, m.blind_event, m.blind_terminal] + \
             [x for T in m.tasks.values() for x in vars(T).values() if isinstance(x, np.ndarray)]:
        a.flags.writeable = False
    return m, L0


def emulate_query(method, task, P0, P1, rs, kref=None):
    """Regime-2 (stale) synthesis of the fitted method for extracted look decisions, pure numpy."""
    kref = kref or method.kref
    out = np.empty((len(task), 10, 7), np.float32)
    for t in np.unique(task):
        T = method.tasks[int(t)]
        sel = np.flatnonzero(task == t)
        X = np.concatenate([P0[sel], P1[sel], rs[sel]], 1).astype(np.float32)
        Z = X @ T.Wf - T.shift
        d2 = T.z2[None] - 2.0 * (Z @ T.Z.T) + (Z ** 2).sum(1)[:, None]
        d = np.sqrt(np.maximum(d2, 0))
        k = min(method.k, d.shape[1])
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(d, idx, 1)
        o = np.argsort(dk, axis=1, kind="stable")
        idx, dk = np.take_along_axis(idx, o, 1), np.take_along_axis(dk, o, 1)
        w = _kernel_w(dk - dk[:, :1], kref)
        w = w / w.sum(1, keepdims=True)
        rows = T.rows[idx]
        out[sel] = np.einsum("nk,nkhd->nhd", w.astype(np.float32), np.asarray(method.act)[rows][:, :, :10, :7])
    return out


def build(cell, arms, out_path, train_inits=range(0, 20), val_inits=range(20, 30), derived=common.DERIVED):
    model, suite, lib = cell
    a_arm = common.arm_name(model, suite, lib, "A")
    arms_rows = {r["arm"]: r for r in common.load_arms()}
    src = arms_rows[a_arm]["r8"]["source"]
    with open(src["artifact"], "rb") as f:
        blob = FitUnpickler(f).load()
    base = blob["method"]
    H = int(base.H)
    rows = collect_rows(model, suite, lib, arms, train_inits, derived)
    grown, L0 = extend_method(base, rows, H)
    # offline check on held-out inits of the A arm: served (frozen) vs grown synthesis vs shadow
    dec = pd.read_parquet(Path(derived) / "decisions" / f"{a_arm}.parquet")
    arr = np.load(Path(derived) / "arrays" / f"{a_arm}.npz")
    norm = np.load(Path(derived) / "arrays_norm" / f"{a_arm}.npz")
    va = dec.is_look.values & (dec.decision_seq.values > 0) & np.isin(dec["init"].values, list(val_inits))
    va &= np.isfinite(arr["keys_third"]).all(1) & np.isfinite(arr["shadow"]).all((1, 2))
    idx = np.flatnonzero(va)
    sigma = shadow_gap.library_sigma(model, suite)
    task = dec.task_id.values[idx]
    P0, P1, rs = arr["keys_third"][idx], arr["keys_wrist"][idx], norm["state_norm"][idx][:, :8]
    frozen = emulate_query(base, task, P0, P1, rs)
    grown_pred = emulate_query(grown, task, P0, P1, rs)
    m_served, g_served, _ = shadow_gap.gap_metrics(arr["served"][idx], arr["shadow"][idx], sigma)
    m_frozen, _, _ = shadow_gap.gap_metrics(frozen, arr["shadow"][idx], sigma)
    m_grown, g_grown, _ = shadow_gap.gap_metrics(grown_pred, arr["shadow"][idx], sigma)
    frac_new = float(np.mean([(np.asarray(grown.tasks[int(t)].rows) >= L0).mean() for t in np.unique(task)]))
    check = dict(n_val=int(len(idx)), served_motion=float(m_served.mean()), frozen_emulated_motion=float(m_frozen.mean()),
                 emulation_vs_served_maxabs=float(np.abs(frozen[:, :5] - arr["served"][idx][:, :5]).max()),
                 grown_motion=float(m_grown.mean()), served_grip=float(g_served.mean()), grown_grip=float(g_grown.mean()),
                 rows_added=int(len(rows["task"])), rows_original=int(L0), share_new_rows=frac_new,
                 per_task_grown_motion=pd.Series(m_grown).groupby(task).mean().round(4).to_dict(),
                 per_task_served_motion=pd.Series(m_served).groupby(task).mean().round(4).to_dict())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    registered = {GROWN: {"action": np.asarray(grown.act, np.float32),
                          "task_id": np.concatenate([np.asarray(np.load(Path(common.STORE) / "library" / f"{model}_{suite}" / "current" / "task_id.npy")), rows["task"]]).astype(np.int64),
                          "meta": dict(source_arms=sorted(set(arms)), train_inits=list(train_inits), rows_added=int(len(rows["task"])),
                                       label="deferred policy_shadow at cache look decisions", built=time.strftime("%Y-%m-%d %H:%M"))}}
    with open(out_path, "wb") as f:
        pickle.dump({"method": grown, "registered": registered, "spec": src["method"], "kwargs": src["kwargs"],
                     "cell": f"{model}_{suite}_cache", "fit_s": float(blob.get("fit_s", 0.0)),
                     "grown": dict(check=check, source_artifact=src["artifact"])}, f, protocol=4)
    common.write_json(out_path.with_suffix(".json"), dict(check=check, arms=sorted(set(arms)), source=src))
    return check


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="pi05:l10:50")
    ap.add_argument("--arms", default="A", help="'A' (pure-cache arm only), 'all' (every arm of the cell) or a comma list")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model, suite, lib = a.cell.split(":")
    lib = int(lib)
    if a.arms == "A":
        arms = [common.arm_name(model, suite, lib, "A")]
    elif a.arms == "all":
        arms = student.cell_arms(model, suite, lib)
    else:
        arms = a.arms.split(",")
    check = build((model, suite, lib), arms, a.out)
    print(json.dumps({k: v for k, v in check.items() if not k.startswith("per_task")}, indent=1))


if __name__ == "__main__":
    main()
