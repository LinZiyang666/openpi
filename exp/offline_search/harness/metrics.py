"""Metrics (protocol §4). GT = a_inf (full_inference) for BOTH arms.

Per decision (all on the executed, valid block a[:5, :7] -- see dims.py):
    err        = sqrt(mean_{t<5, d<7} (((a_hat - a_star) / sigma_d) ** 2)),  sigma_d = std of current-library
                 action[:, :5, d] (per model x suite)
    grip_mis   = mean_{t<5} [ (a_hat[t,6] >= 0) != (a_star[t,6] >= 0) ]      (dim 6 = gripper, bimodal at +-1)
    phase_err  = |step / (num_steps - 1) - progress[top1]|                     (NaN if the library has no progress)
    oracle_err = min over current-library candidates of the same task of err;  oracle_row = its argmin
    regret     = err - oracle_err
    bin        = min(2, 3 * step // num_steps)   (0 early / 1 mid / 2 late third)
    floor      = teacher-vs-teacher err from floor/<m>_<s>/ (step0 pairs -> early; resample -> by third);
                 indist = err <= floor median of the decision's bin; bad = err > floor p90 of the bin
Per cell: means / medians, per-bin, risk-coverage (sort by confidence desc, stable), AURC, risk@c, bad@c, flip rate
vs the recorded online top-1.
"""
from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import pathlib
import time

import numpy as np

from . import dims
from .store import LibraryView, action_sigma

BIN_NAMES = ("early", "mid", "late")
COVERAGES = (0.3, 0.5, 0.7, 0.9)
FLOOR_MIN_BIN = 20          # a bin needs this many floor samples, else the overall floor is used


# ------------------------------------------------------------------------------------------- basics
def seg(a) -> np.ndarray:
    """Executed valid block as float64: a[..., :5, :7]."""
    return np.asarray(dims.valid_action(a), np.float64)


def err_seg(a_hat_seg, a_star_seg, sigma) -> np.ndarray:
    d = (np.asarray(a_hat_seg, np.float64) - np.asarray(a_star_seg, np.float64)) / sigma
    return np.sqrt(np.mean(d * d, axis=(-2, -1)))


def grip_mis_seg(a_hat_seg, a_star_seg) -> np.ndarray:
    g = dims.GRIPPER_DIM
    return np.mean((a_hat_seg[..., g] >= 0) != (a_star_seg[..., g] >= 0), axis=-1)


def step_bin(step, num_steps) -> np.ndarray:
    step = np.asarray(step, np.int64)
    n = np.maximum(np.asarray(num_steps, np.int64), 1)
    return np.minimum(2, (3 * step) // n).astype(np.int8)


def phase_of(step, num_steps) -> np.ndarray:
    n = np.asarray(num_steps, np.float64)
    return np.where(n > 1, np.asarray(step, np.float64) / np.maximum(n - 1, 1), 0.0)


# ------------------------------------------------------------------------------------------- oracle
def oracle(a_star_seg: np.ndarray, task_id: np.ndarray, lib: LibraryView, sigma: np.ndarray, chunk: int = 512):
    """For each decision: min err over current-library rows of the same task, and the argmin row (first min).
    a_star_seg [n, 5, 7] float64; task_id [n]. Returns (oracle_err f64[n], oracle_row int64[n])."""
    n = a_star_seg.shape[0]
    oe = np.full(n, np.nan)
    orow = np.full(n, -1, np.int64)
    tid = np.asarray(task_id, np.int64)
    for t in np.unique(tid):
        cand = lib.rows_of_task(int(t))
        idx = np.flatnonzero(tid == t)
        if cand.size == 0:
            continue
        C = seg(lib.action[np.sort(cand)])            # rows ascending == library order
        cs = np.sort(cand)
        for s in range(0, idx.size, chunk):
            ii = idx[s:s + chunk]
            e = err_seg(C[None], a_star_seg[ii][:, None], sigma)       # [b, c]
            j = np.argmin(e, axis=1)
            oe[ii] = e[np.arange(ii.size), j]
            orow[ii] = cs[j]
    return oe, orow


def _fingerprint(qc, lib, sigma) -> str:
    import hashlib

    h = hashlib.blake2b(digest_size=10)
    h.update(str(pathlib.Path(qc.root).resolve()).encode() + qc.cell.encode())
    for p in (qc.dir / "a_inf.npy", qc.dir / "episodes.json", lib.dir / "action.npy", lib.dir / "task_id.npy"):
        st = p.stat() if p.exists() else None
        h.update(repr((p.name, st.st_size if st else -1, st.st_mtime_ns if st else -1)).encode())
    h.update(np.asarray(sigma, np.float64).tobytes())
    return h.hexdigest()


def oracle_cell(qc, lib, sigma, cache_dir=None):
    """oracle() over EVERY row of a cell (the oracle does not depend on the method), cached as
    <cache_dir>/oracle_<cell>_<fingerprint>.npz. The fingerprint covers the store paths, file sizes / mtimes and
    sigma. A per-cell flock makes concurrent jobs of the same cell compute it once. Returns (oe f64[N], orow i64[N])."""
    def compute():
        a_star = seg(qc.a_inf[:])
        task = np.array([e["task_id"] for e in qc.episodes], np.int64)[np.asarray(qc.ep, np.int64)]
        return oracle(a_star, task, lib, sigma)

    if cache_dir is None:
        return compute()
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"oracle_{qc.cell}_{_fingerprint(qc, lib, sigma)}.npz"
    lock = open(cache_dir / f".oracle_{qc.cell}.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if p.exists():
            z = np.load(p)
            return z["oracle_err"], z["oracle_row"]
        oe, orow = compute()
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp.npz")
        np.savez(tmp, oracle_err=oe, oracle_row=orow)
        os.replace(tmp, p)
        return oe, orow
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def oracle_one(a_star_seg: np.ndarray, cand_seg: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """err of every candidate for one decision (same arithmetic as oracle())."""
    return err_seg(cand_seg[None], a_star_seg[None, None], sigma)[0]


# -------------------------------------------------------------------------------------------- floor
def load_floor(root, key: str, sigma: np.ndarray) -> dict:
    """Teacher noise floor for one model x suite. Accepts the R0-D formats:
      step0_pairs.npz : a_inf_arm, a_cache_arm [P, H, 32] (err recomputed with sigma) or err [P]   -> early bin
      resample.npz    : a_fresh [S, K, H, 32] + a_recorded [S, H, 32] + third [S]  (err of every fresh sample vs
                        the recorded one), or err [S(,K)] + third [S]
    Returns {"available", "source", "samples": {bin: f64[]}, "median"/"p90": {bin|"all": float}, "n": {...}}."""
    d = pathlib.Path(root) / "floor" / key
    samples = {0: [], 1: [], 2: []}
    src = []
    p0 = d / "step0_pairs.npz"
    if p0.exists():
        z = np.load(p0, allow_pickle=False)
        if "a_inf_arm" in z and "a_cache_arm" in z:
            e = err_seg(seg(z["a_inf_arm"]), seg(z["a_cache_arm"]), sigma)
        else:
            e = np.asarray(z["err"], np.float64)
        samples[0].append(e.ravel())
        src.append("step0_pairs")
    p1 = d / "resample.npz"
    if p1.exists():
        z = np.load(p1, allow_pickle=False)
        third = np.asarray(z["third"], np.int64)
        if "a_fresh" in z and "a_recorded" in z:
            e = err_seg(seg(z["a_fresh"]), seg(z["a_recorded"])[:, None], sigma)     # [S, K]
        else:
            e = np.asarray(z["err"], np.float64)
        e = e.reshape(third.shape[0], -1)
        for b in range(3):
            samples[b].append(e[third == b].ravel())
        src.append("resample")
    samples = {b: (np.concatenate(v) if v else np.zeros(0)) for b, v in samples.items()}
    samples = {b: v[np.isfinite(v)] for b, v in samples.items()}
    allv = np.concatenate(list(samples.values()))
    out = {"available": allv.size > 0, "source": "+".join(src) or "none", "dir": str(d),
           "n": {BIN_NAMES[b]: int(samples[b].size) for b in range(3)}, "median": {}, "p90": {}, "bin_fallback": {}}
    if allv.size:
        out["median"]["all"] = float(np.median(allv))
        out["p90"]["all"] = float(np.percentile(allv, 90))
        out["n"]["all"] = int(allv.size)
        for b in range(3):
            if samples[b].size >= FLOOR_MIN_BIN:
                out["median"][BIN_NAMES[b]] = float(np.median(samples[b]))
                out["p90"][BIN_NAMES[b]] = float(np.percentile(samples[b], 90))
                out["bin_fallback"][BIN_NAMES[b]] = False
            else:
                out["median"][BIN_NAMES[b]] = out["median"]["all"]
                out["p90"][BIN_NAMES[b]] = out["p90"]["all"]
                out["bin_fallback"][BIN_NAMES[b]] = True
    return out


def floor_per_decision(floor: dict, bins: np.ndarray):
    if not floor["available"]:
        n = bins.shape[0]
        return np.full(n, np.nan), np.full(n, np.nan)
    med = np.array([floor["median"][b] for b in BIN_NAMES])
    p90 = np.array([floor["p90"][b] for b in BIN_NAMES])
    return med[bins], p90[bins]


# ------------------------------------------------------------------------------- per-decision metrics
def decision_metrics(qc, rows, top1, lib_code, lib_names, lib_tables, synth_seg, used_synth, sigma, cur_lib,
                     oracle_cache=None) -> dict:
    """Vectorized per-decision metrics for one cell (oracle_cache: directory for the per-cell oracle cache).
    qc: store.QueryCell; rows int64[n] query rows; top1 int64[n]; lib_code int8[n] indexes lib_names;
    lib_tables: {name: {"action": [L,H,32], "progress": f64[L] | None}}; synth_seg f64[n,5,7] or None;
    used_synth bool[n]."""
    n = rows.shape[0]
    a_star = seg(qc.a_inf[rows]) if n else np.zeros((0, dims.EXEC_STEPS, dims.ACT_DIMS))
    a_hat = np.empty_like(a_star)
    prog = np.full(n, np.nan)
    for c, name in enumerate(lib_names):
        m = lib_code == c
        if not m.any():
            continue
        tab = lib_tables[name]
        a_hat[m] = seg(tab["action"][top1[m]])
        if tab.get("progress") is not None:
            prog[m] = np.asarray(tab["progress"], np.float64)[top1[m]]
    if synth_seg is not None and used_synth.any():
        a_hat[used_synth] = synth_seg[used_synth]
    e = err_seg(a_hat, a_star, sigma)
    g = grip_mis_seg(a_hat, a_star)
    step = np.asarray(qc.step[rows], np.int64)
    ns = qc.num_steps_row[rows]
    ph = np.abs(phase_of(step, ns) - prog)
    task = np.array([qc.episodes[i]["task_id"] for i in np.asarray(qc.ep[rows], np.int64)], np.int64) if n else \
        np.zeros(0, np.int64)
    if oracle_cache is not None and 2 * n >= qc.N:     # whole-cell cache pays off; small runs compute their rows
        oe_all, orow_all = oracle_cell(qc, cur_lib, sigma, oracle_cache)
        oe, orow = oe_all[rows], orow_all[rows]
    else:
        oe, orow = oracle(a_star, task, cur_lib, sigma)
    return {"err": e, "grip_mis": g, "phase_err": ph, "oracle_err": oe, "oracle_row": orow, "regret": e - oe,
            "bin": step_bin(step, ns), "task_id": task, "step": step}


# ------------------------------------------------------------------------------------ risk-coverage
def risk_coverage(err: np.ndarray, conf: np.ndarray, bad: np.ndarray | None = None) -> dict:
    n = err.shape[0]
    if n == 0:
        return {"aurc": math.nan}
    order = np.argsort(-conf, kind="stable")
    k = np.arange(1, n + 1)
    risk = np.cumsum(err[order]) / k
    opt = np.cumsum(np.sort(err)) / k
    out = {"aurc": float(risk.mean()), "aurc_opt": float(opt.mean()), "aurc_rand": float(err.mean())}
    out["eaurc"] = out["aurc"] - out["aurc_opt"]
    for c in COVERAGES:
        j = max(1, math.ceil(c * n)) - 1
        out[f"risk_c{int(c * 100)}"] = float(risk[j])
    if bad is not None and np.isfinite(bad).all():
        br = np.cumsum(bad[order]) / k
        for c in COVERAGES + (1.0,):
            j = max(1, math.ceil(c * n)) - 1
            out[f"bad_c{int(c * 100)}"] = float(br[j])
        out["bad_aurc"] = float(br.mean())
    else:
        for c in COVERAGES + (1.0,):
            out[f"bad_c{int(c * 100)}"] = math.nan
        out["bad_aurc"] = math.nan
    grid = np.linspace(0.05, 1.0, 20)
    out["curve"] = [[float(c), float(risk[max(1, math.ceil(c * n)) - 1])] for c in grid]
    return out


def _f(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else math.nan


def _med(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else math.nan


def aggregate(dm: dict, conf: np.ndarray, floor: dict, top1: np.ndarray, lib_is_current: np.ndarray,
              rec_top1: np.ndarray, used_synth: np.ndarray) -> dict:
    e = dm["err"]
    fmed, fp90 = floor_per_decision(floor, dm["bin"].astype(np.int64))
    indist = (e <= fmed).astype(np.float64) if floor["available"] else np.full(e.shape, np.nan)
    bad = (e > fp90).astype(np.float64) if floor["available"] else None
    out = {"n": int(e.shape[0]),
           "err_mean": _f(e), "err_median": _med(e),
           "err_p90": float(np.percentile(e, 90)) if e.size else math.nan,
           "grip_mis": _f(dm["grip_mis"]), "phase_err_mean": _f(dm["phase_err"]), "phase_err_median": _med(dm["phase_err"]),
           "oracle_err_mean": _f(dm["oracle_err"]), "regret_mean": _f(dm["regret"]), "regret_median": _med(dm["regret"]),
           "indist": _f(indist) if floor["available"] else math.nan,
           "err_over_floor_median": _med(e / fmed) if floor["available"] else math.nan,
           "err_over_floor_mean": _f(e / fmed) if floor["available"] else math.nan,
           "floor_median": floor["median"].get("all", math.nan) if floor["available"] else math.nan,
           "floor_p90": floor["p90"].get("all", math.nan) if floor["available"] else math.nan,
           "synth_frac": float(used_synth.mean()) if used_synth.size else math.nan,
           "frac_current": float(lib_is_current.mean()) if lib_is_current.size else math.nan}
    cur = lib_is_current
    out["flip_rate"] = float((top1[cur] != rec_top1[cur]).mean()) if cur.any() else math.nan
    for b, nm in enumerate(BIN_NAMES):
        m = dm["bin"] == b
        out[f"n_{nm}"] = int(m.sum())
        out[f"err_{nm}"] = _f(e[m])
        out[f"indist_{nm}"] = _f(indist[m]) if floor["available"] else math.nan
    rc = risk_coverage(e, conf, bad)
    out.update({k: v for k, v in rc.items() if k != "curve"})
    out["rc_curve"] = rc.get("curve", [])
    return out


# --------------------------------------------------------------------------------------- scoreboard
SCOREBOARD_COLUMNS = (
    "round", "run_ts", "method", "family", "tier", "uses_gt", "uses_nonlibrary_action", "cell", "model", "suite", "arm", "subsample", "n",
    "library", "err_mean", "err_median", "err_p90", "err_early", "err_mid", "err_late",
    "indist", "indist_early", "indist_mid", "indist_late", "err_over_floor_median", "err_over_floor_mean",
    "grip_mis", "phase_err_mean", "phase_err_median", "oracle_err_mean", "regret_mean", "regret_median",
    "aurc", "aurc_opt", "eaurc", "risk_c30", "risk_c50", "risk_c70", "risk_c90",
    "bad_c30", "bad_c50", "bad_c70", "bad_c90", "bad_c100", "bad_aurc",
    "flip_rate", "frac_current", "synth_frac", "floor_median", "floor_p90",
    "bytes_per_entry", "ms_per_query", "ms_per_query_p50", "fit_s", "peak_rss_mb", "L", "out",
)


def _fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6g}"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return str(v)


def append_scoreboard(path, row: dict) -> None:
    """Append one row (locked; header written when the file is new). Unknown keys are ignored; the column set is
    SCOREBOARD_COLUMNS. A file written with an older column set is migrated in place first (under the lock): its
    rows are rewritten with the new header, new columns left empty. Re-runs append new rows -- consumers take the
    latest run_ts per (method, cell)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
    with os.fdopen(fd, "r+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            header = next(csv.reader([f.readline()]), [])
            cols = list(SCOREBOARD_COLUMNS)
            if header and header != cols:
                f.seek(0)
                old = list(csv.DictReader(f))
                cols += [c for c in header if c not in cols]
                f.seek(0)
                f.truncate()
                w = csv.writer(f)
                w.writerow(cols)
                for r in old:
                    w.writerow([r.get(c, "") or "" for c in cols])
            elif header:
                cols = header
            f.seek(0, os.SEEK_END)
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(cols)
            w.writerow([_fmt(row.get(c, "")) for c in cols])
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def summarize_method(method_dir) -> dict:
    """Collect <cell>.json of one method dir into summary.json, with inf - cache deltas per model x suite."""
    d = pathlib.Path(method_dir)
    cells = {}
    for p in sorted(d.glob("*.json")):
        if p.name in ("summary.json", "run_meta.json"):
            continue
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if "cell" in j and "metrics" in j:
            cells[j["cell"]] = j
    keys = ("err_mean", "err_median", "indist", "aurc", "eaurc", "grip_mis", "regret_mean", "flip_rate")
    headline = {c: {k: j["metrics"].get(k) for k in keys} | {"ms_per_query": j["timing"].get("ms_per_query")}
                for c, j in cells.items()}
    deltas = {}
    for ms in sorted({c.rsplit("_", 1)[0] for c in cells}):
        a, b = cells.get(f"{ms}_inf"), cells.get(f"{ms}_cache")
        if a and b:
            deltas[ms] = {k: (a["metrics"].get(k, math.nan) - b["metrics"].get(k, math.nan)) for k in keys}
    out = {"method_dir": str(d), "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cells": headline,
           "inf_minus_cache": deltas}
    (d / "summary.json").write_text(json.dumps(out, indent=1, default=_json_default))
    return out


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def sigma_for(root, key: str) -> np.ndarray:
    return action_sigma(str(root), key)
