"""Gripper-sign-faithful err for any harness run (analysis tooling, not a method).

LIBERO (robosuite PandaGripper.format_action) executes np.sign(gripper), so a synthesized (averaged) chunk's
fractional gripper value is only offline hedging. This recomputes the harness err with dim 6 (gripper) of the
EXECUTED segment [:5, :7] snapped to its sign (np.sign: -1 / 0 / +1; the normalized gripper maps +-1 <-> the executed
+-1 for both models) before scoring against a_inf (unchanged). grip_mis is unaffected by construction (>= 0 split).

Executed segment per decision (as the harness scored it): `synth_seg` when the method returned an action
(`used_synth`), else the library action of `top1` in the decision's library (`lib_names[lib_code]`, any stored
library of the model x suite; method-registered libraries are not in the store -> NaN, counted as n_missing).
Also re-derives the unsnapped err and reports max |err_recomputed - err_npz| as a consistency check.

    python exp/offline_search/rounds/r02/g4_t2/sign_faithful.py RUN [RUN ...] [--method M[,M]] [--cells all]
        [--root /dev/shm/offline_search_store] [--out DIR] [--procs 8] [--per-decision]
RUN = a method dir (results/rNN/<method>, holding <cell>.npz) or a round dir (then --method selects, default all).
Writes one csv per run: <out>/<round>__<method>.csv with rows (cell, regime, n, err, err_sf, d_sf, grip_mis,
used_synth, n_missing, max_recompute_diff); regime in {all, step0, fresh (inf arm, step >= 1), stale (cache arm,
step >= 1)}. --per-decision also writes <out>/<round>__<method>/<cell>.npz (row, step, err, err_sf).
Default --out: /home/weiland/trace_runs/offline_search_store/derived/r02/g4_t2/sign_faithful
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))
from exp.offline_search.harness import dims, store  # noqa: E402

DEFAULT_ROOT = "/dev/shm/offline_search_store"
DEFAULT_OUT = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r02/g4_t2/sign_faithful")
CELLS = [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]


def method_dirs(path, methods=None):
    p = pathlib.Path(path)
    if any((p / f"{c}.npz").exists() for c in CELLS):
        return [p]
    subs = sorted(d for d in p.iterdir() if d.is_dir() and any((d / f"{c}.npz").exists() for c in CELLS))
    if methods:
        subs = [d for d in subs if d.name in methods.split(",")]
    return subs


def snap(seg: np.ndarray) -> np.ndarray:
    s = np.array(seg, np.float64, copy=True)
    s[..., dims.GRIPPER_DIM] = np.sign(s[..., dims.GRIPPER_DIM])
    return s


def cell_sf(mdir, cell: str, root: str = DEFAULT_ROOT) -> dict:
    """Per-decision arrays: row, step, err (npz), err_re (recomputed), err_sf, grip_mis, used_synth, missing."""
    mdir = pathlib.Path(mdir)
    with np.load(mdir / f"{cell}.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in ("row", "step", "top1", "lib_code", "used_synth", "err", "grip_mis") if k in z.files}
        names = [str(x) for x in np.atleast_1d(z["lib_names"])] if "lib_names" in z.files else [str(z["library"])]
        synth = z["synth_seg"] if "synth_seg" in z.files else None
    model, suite, arm = store.parse_cell(cell)
    key = store.lib_key(cell)
    rows = d["row"].astype(np.int64)
    n = rows.shape[0]
    a_star = np.asarray(dims.valid_action(np.load(pathlib.Path(root) / "queries" / cell / "a_inf.npy", mmap_mode="r")[rows]), np.float64)
    a_hat = np.full((n, dims.EXEC_STEPS, dims.ACT_DIMS), np.nan)
    have = set(store.library_names(root, key))
    code = d.get("lib_code", np.zeros(n, np.int8)).astype(np.int64)
    used = d.get("used_synth", np.zeros(n, bool)).astype(bool)
    for c, name in enumerate(names):
        m = (code == c) & ~used
        if not m.any() or name not in have:
            continue
        act = np.load(pathlib.Path(root) / "library" / key / name / "action.npy", mmap_mode="r")
        a_hat[m] = np.asarray(dims.valid_action(act[d["top1"][m].astype(np.int64)]), np.float64)
    if synth is not None and used.any():
        a_hat[used] = synth[used].astype(np.float64)
    sig = store.action_sigma(root, key)
    miss = ~np.isfinite(a_hat).all(axis=(1, 2))

    def err(a):
        e = (a - a_star) / sig
        return np.sqrt(np.mean(e * e, axis=(1, 2)))

    return {"row": rows, "step": d["step"].astype(np.int64), "err": d["err"].astype(np.float64), "err_re": err(a_hat),
            "err_sf": err(snap(a_hat)), "grip_mis": d["grip_mis"].astype(np.float64), "used_synth": used, "missing": miss,
            "arm": arm}


def summarize(cell: str, r: dict) -> list[dict]:
    out = []
    regs = [("all", np.ones_like(r["step"], bool)), ("step0", r["step"] == 0),
            ("fresh" if r["arm"] == "inf" else "stale", r["step"] >= 1)]
    ok = ~r["missing"]
    diff = np.abs(r["err_re"][ok] - r["err"][ok])
    for name, m in regs:
        mm = m & ok
        out.append({"cell": cell, "regime": name, "n": int(m.sum()), "err": float(r["err"][mm].mean()) if mm.any() else float("nan"),
                    "err_sf": float(r["err_sf"][mm].mean()) if mm.any() else float("nan"),
                    "d_sf": float((r["err_sf"][mm] - r["err"][mm]).mean()) if mm.any() else float("nan"),
                    "grip_mis": float(r["grip_mis"][mm].mean()) if mm.any() else float("nan"),
                    "used_synth": float(r["used_synth"][m].mean()) if m.any() else float("nan"),
                    "n_missing": int((m & ~ok).sum()),
                    "max_recompute_diff": float(diff.max()) if diff.size else float("nan")})
    return out


def _job(args):
    mdir, cell, root, per_dec, outdir = args
    r = cell_sf(mdir, cell, root)
    if per_dec:
        p = pathlib.Path(outdir) / f"{pathlib.Path(mdir).parent.name}__{pathlib.Path(mdir).name}" / f"{cell}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, row=r["row"], step=r["step"], err=r["err"], err_sf=r["err_sf"], missing=r["missing"])
    return str(mdir), summarize(cell, r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--method", default=None)
    ap.add_argument("--cells", default="all")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--per-decision", action="store_true")
    a = ap.parse_args(argv)
    mds = [md for r in a.runs for md in method_dirs(r, a.method)]
    want = CELLS if a.cells == "all" else a.cells.split(",")
    jobs = [(str(md), c, a.root, a.per_decision, a.out) for md in mds for c in want if (md / f"{c}.npz").exists()]
    procs = max(1, min(a.procs, len(jobs), len(os.sched_getaffinity(0))))
    if procs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(procs) as pool:
            res = pool.map(_job, jobs, chunksize=1)
    else:
        res = [_job(j) for j in jobs]
    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cols = ["cell", "regime", "n", "err", "err_sf", "d_sf", "grip_mis", "used_synth", "n_missing", "max_recompute_diff"]
    for md in mds:
        rows = [row for m_, rr in res if m_ == str(md) for row in rr]
        rows.sort(key=lambda x: (CELLS.index(x["cell"]), ["all", "step0", "fresh", "stale"].index(x["regime"])))
        p = outdir / f"{md.parent.name}__{md.name}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in rows:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
        print(f"## {md.parent.name}/{md.name}  -> {p}")
        print(f"{'cell':22s} {'regime':6s} {'n':>6s} {'err':>7s} {'err_sf':>7s} {'d_sf':>7s} {'synth':>5s} {'miss':>5s} {'maxdiff':>8s}")
        for row in rows:
            if row["regime"] == "all":
                continue
            print(f"{row['cell']:22s} {row['regime']:6s} {row['n']:6d} {row['err']:7.4f} {row['err_sf']:7.4f} {row['d_sf']:+7.4f} "
                  f"{row['used_synth']:5.2f} {row['n_missing']:5d} {row['max_recompute_diff']:8.1e}")


if __name__ == "__main__":
    sys.exit(main())
