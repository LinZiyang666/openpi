"""Three-layer effect decomposition (protocol §8/§9: synthesis / method / library) with episode-bootstrap CIs.

Given, per regime, the runs
    B0     : the deployed ranking, top-1                   (e.g. results/r00/B0_current)
    SYNTH  : the same ranking, top-k synthesis             (e.g. results/r01/M4_b0cons_k5_mean)
    CUR    : the method on the current (~50-episode) library, fitted on it
    BIG    : the same method on the 10x library
    [FITBIG: the method on the current library but fitted on the 10x library = "borrowed big-library info"]
it reports per cell and regime (step0 / fresh = inf arm step >= 1 / stale = cache arm step >= 1), on the decisions
common to all runs (aligned by query row), the telescoping paired differences
    synthesis = SYNTH - B0 ;  method = CUR - SYNTH ;  [borrowed fit = FITBIG - CUR ;]  library = BIG - (FITBIG | CUR)
    total = BIG - B0 = sum of the layers,
each as mean d with a 95 % episode-bootstrap CI (profile/common.EpisodeBootstrap, the logic of profile/compare.py:
episodes resampled with replacement, fixed seed), plus the err level of every run and the library scale line
(library, episodes, entries, bytes/entry, entries x bytes) the protocol requires next to every conclusion.
--sign-faithful scores every run with the gripper-sign-faithful err (sign_faithful.py) instead of the harness err.

    python exp/offline_search/rounds/r02/g4_t2/decompose.py --b0 r00/B0_current --synth r01/M4_b0cons_k5_mean \
        --cur r02/<method_cur> --big r02/<method_big> [--fitbig r02/<method_cur_fitbig>] [--cells all]
        [--regimes step0,fresh,stale] [--reps 2000] [--seed 0] [--sign-faithful] [--out DIR] [--procs 8]
A RUN is a method dir path, or rNN/<method>, or a bare method name (searched in results/r02, r01, r00).
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

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from exp.offline_search.profile import common as C  # noqa: E402
import sign_faithful as SF  # noqa: E402

RESULTS = REPO / "exp" / "offline_search" / "results"
DEFAULT_ROOT = "/dev/shm/offline_search_store"
DEFAULT_OUT = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r02/g4_t2/decompose")


def resolve(spec: str) -> pathlib.Path:
    p = pathlib.Path(spec)
    if p.is_dir():
        return p.resolve()
    if (RESULTS / spec).is_dir():
        return (RESULTS / spec).resolve()
    for rnd in sorted((d.name for d in RESULTS.iterdir() if d.is_dir() and d.name.startswith("r")), reverse=True):
        if (RESULTS / rnd / spec).is_dir():
            return (RESULTS / rnd / spec).resolve()
    raise SystemExit(f"run {spec!r} not found (path, rNN/<method> or a method name under {RESULTS})")


def load_err(md: pathlib.Path, cell: str, root: str, sign_faithful: bool):
    if sign_faithful:
        r = SF.cell_sf(md, cell, root)
        return r["row"], r["err_sf"], r["step"]
    with np.load(md / f"{cell}.npz", allow_pickle=False) as z:
        return z["row"].astype(np.int64), z["err"].astype(np.float64), z["step"].astype(np.int64)


def scale_line(md: pathlib.Path, cell: str, root: str) -> dict:
    j = json.loads((md / f"{cell}.json").read_text()) if (md / f"{cell}.json").exists() else {}
    lib = j.get("library", "?")
    fit = j.get("fit", {})
    L = (fit.get("libraries") or {}).get(lib)
    eps = None
    try:
        e = np.load(pathlib.Path(root) / "library" / C.cell_ms(cell) / lib / "episode.npy", mmap_mode="r")
        eps = int(np.unique(e).size)
        L = L or int(e.shape[0])
    except Exception:
        pass
    bpe = fit.get("bytes_per_entry")
    tot = (L * bpe) if (L and bpe and np.isfinite(bpe)) else None
    return {"run": md.name, "library": lib, "episodes": eps, "entries": L, "bytes_per_entry": bpe,
            "key_MB": (tot / 2 ** 20) if tot else None}


LAYERS_3 = [("synthesis", "synth", "b0"), ("method", "cur", "synth"), ("library", "big", "cur")]
LAYERS_4 = [("synthesis", "synth", "b0"), ("method", "cur", "synth"), ("borrowed_fit", "fitbig", "cur"),
            ("library", "big", "fitbig")]


def cell_job(job):
    runs, cell, root, regimes, reps, seed, sf = job
    arm = C.parse_cell(cell)[2]
    data = {k: load_err(md, cell, root, sf) for k, md in runs.items()}
    rows = None
    for r, _e, _s in data.values():
        rows = r if rows is None else np.intersect1d(rows, r)
    E = {}
    step = None
    for k, (r, e, s) in data.items():
        idx = np.searchsorted(r, rows) if np.all(np.diff(r) > 0) else np.array([np.flatnonzero(r == x)[0] for x in rows])
        E[k] = e[idx]
        step = s[idx]
    qc = C.QueryCell(root, cell)
    ep = qc.ep[rows]
    layers = LAYERS_4 if "fitbig" in runs else LAYERS_3
    out = []
    for reg in regimes:
        if reg == "step0":
            m = step == 0
        elif (reg == "fresh" and arm == "inf") or (reg == "stale" and arm == "cache"):
            m = step >= 1
        else:
            continue
        if not m.any():
            continue
        bs = C.EpisodeBootstrap(ep[m], reps, seed)
        rec = {"cell": cell, "regime": reg, "n": int(m.sum()), "n_episodes": int(np.unique(ep[m]).size)}
        for k in runs:
            rec[f"err_{k}"] = float(E[k][m].mean())
        for name, a, b in layers + [("total", "big", "b0")]:
            ci = bs.mean_ci(E[a][m] - E[b][m])
            rec[name] = ci["est"]
            rec[f"{name}_lo"], rec[f"{name}_hi"] = ci["lo"], ci["hi"]
        out.append(rec)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b0", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--cur", required=True)
    ap.add_argument("--big", required=True)
    ap.add_argument("--fitbig", default=None)
    ap.add_argument("--cells", default="all")
    ap.add_argument("--regimes", default="step0,fresh,stale")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sign-faithful", action="store_true")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args(argv)
    runs = {"b0": resolve(a.b0), "synth": resolve(a.synth), "cur": resolve(a.cur), "big": resolve(a.big)}
    if a.fitbig:
        runs["fitbig"] = resolve(a.fitbig)
    want = C.expand_cells(a.cells)
    cells = [c for c in want if all((md / f"{c}.npz").exists() for md in runs.values())]
    missing = [c for c in want if c not in cells]
    regimes = a.regimes.split(",")
    jobs = [(runs, c, a.root, regimes, a.reps, a.seed, a.sign_faithful) for c in cells]
    procs = max(1, min(a.procs, len(jobs), len(os.sched_getaffinity(0))))
    res = [r for rr in C.pmap(cell_job, jobs, procs) for r in rr]
    layers = [n for n, _a, _b in (LAYERS_4 if a.fitbig else LAYERS_3)] + ["total"]
    tag = "__".join(md.name for md in runs.values()) + ("__sf" if a.sign_faithful else "")
    print(f"## effect decomposition ({'gripper-sign-faithful err' if a.sign_faithful else 'harness err'}; d < 0 = better; "
          f"episode bootstrap {a.reps} reps, seed {a.seed}; decisions common to all runs)")
    for k, md in runs.items():
        print(f"   {k:7s} = {md.parent.name}/{md.name}")
    if missing:
        print(f"   (cells skipped, not in every run: {', '.join(missing)})")
    hdr = f"| cell | regime | n | " + " | ".join(f"err {k}" for k in runs) + " | " + " | ".join(layers) + " |"
    print("\n" + hdr)
    print("|" + "---|" * (3 + len(runs) + len(layers)))
    for r in res:
        cells_ = [r["cell"], r["regime"], str(r["n"])] + [f"{r[f'err_{k}']:.3f}" for k in runs] + \
                 [f"{r[n]:+.3f} [{r[n + '_lo']:+.3f},{r[n + '_hi']:+.3f}]" for n in layers]
        print("| " + " | ".join(cells_) + " |")
    # library scale lines (protocol: every conclusion states the library scale)
    print("\nlibrary scale per run (from <cell>.json fit block + store; key_MB = entries x bytes_per_entry, fixed costs "
          "such as PCA bases not included):")
    scl = []
    for ms in sorted({C.cell_ms(c) for c in cells}):
        c0 = next(c for c in cells if C.cell_ms(c) == ms)
        for k, md in runs.items():
            s = scale_line(md, c0, a.root)
            s["ms"], s["role"] = ms, k
            scl.append(s)
            kb = f"{s['key_MB']:.1f} MB" if s["key_MB"] else "?"
            print(f"   {ms:14s} {k:7s} {s['run']:40s} lib={s['library']:9s} episodes={s['episodes']} entries={s['entries']} "
                  f"B/entry={s['bytes_per_entry']} -> {kb}")
    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cols = list(res[0].keys()) if res else []
    with open(outdir / f"{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in res:
            w.writerow(r)
    (outdir / f"{tag}.json").write_text(json.dumps({"runs": {k: str(v) for k, v in runs.items()}, "reps": a.reps,
                                                    "seed": a.seed, "sign_faithful": a.sign_faithful, "rows": res,
                                                    "scale": scl}, indent=1, default=float))
    print(f"\n[decompose] -> {outdir / tag}.csv/.json")


if __name__ == "__main__":
    sys.exit(main())
