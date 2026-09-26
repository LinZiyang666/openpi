"""Runner: fit once per cell in the main process, then fork workers over contiguous episode chunks.

    python -m exp.offline_search.harness.run --method <module_or_file>:<Class> [--kwargs JSON] \
        --cells all|<cell,...> --root <store root> --out <results/rNN> [--workers <#CPUs in affinity mask>] \
        [--subsample tok] [--seed 0] [--profile] [--round rNN] [--family NAME] [--scoreboard PATH|none] \
        [--allow-gpu-fit] [--no-timing] [--job] [--timing-only] [--pin-core K]

--job          one piece of a batch (batch.py): no DONE/ERROR / run_meta / summary / scoreboard writes; the
               per-cell outcome goes to <out>/<name>/_jobs/<cell>.json instead (batch.py writes the rest)
--timing-only  re-fit the method and run only the 300-query single-thread timing pass; merges the timing /
               profile fields into the existing <out>/<name>/<cell>.json (atomic replace); outcome in
               _jobs/<cell>.timing.json
--pin-core K   pin the process to logical CPU K (sched_setaffinity, applied before numpy loads)

Outputs in <out>/<method.name>/:
    <cell>.npz        per-decision arrays (see README "Output files")
    <cell>.json       summary: metrics, floor, risk-coverage curve, timing, fit stats, profile
    summary.json      headline metrics of every cell + inf - cache deltas
    run_meta.json     argv / spec / kwargs / times
    progress.jsonl    JSON lines: run/cell start+done, fit_done, per-worker progress every ~10 s, errors
    DONE | ERROR      written at the very end (ERROR lists the failed cells with tracebacks)
and one row per cell appended to the scoreboard CSV (default <out>/../scoreboard.csv).
"""
from __future__ import annotations

import os
import sys

if "--allow-gpu-fit" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
if __name__ == "__main__" and "--pin-core" in sys.argv:     # before numpy: every later thread inherits it
    os.sched_setaffinity(0, {int(sys.argv[sys.argv.index("--pin-core") + 1])})

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from . import api, dims, metrics, store  # noqa: E402

CPU_COUNT = len(os.sched_getaffinity(0)) or os.cpu_count() or 64   # respects taskset / --cpus masks
MAX_WORKERS = CPU_COUNT          # one job may use the whole machine
PROGRESS_EVERY_S = 10.0
TIMING_QUERIES = 300
TIMING_SEED = 12345

_G: dict = {}   # state inherited by forked workers


# -------------------------------------------------------------------------------------------- utils
def ep_seed(seed: int, uid: str) -> int:
    h = hashlib.blake2b(f"{int(seed)}:{uid}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") >> 1


def rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except Exception:
        return float("nan")


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def plog(path, **obj) -> None:
    """Append one JSON line (single write on an O_APPEND fd: lines from concurrent workers do not interleave)."""
    if path is None:
        return
    obj.setdefault("t", round(time.time(), 3))
    obj.setdefault("pid", os.getpid())
    line = (json.dumps(obj, default=metrics._json_default) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def load_method_class(spec: str):
    """'<dotted.module>:<Class>' or '<path/to/file.py>:<Class>'. File modules get their directory on sys.path so
    sibling imports (``import common``) work."""
    if ":" not in spec:
        raise ValueError(f"--method must be <module_or_file>:<ClassName>, got {spec!r}")
    mod, cls = spec.rsplit(":", 1)
    if mod.endswith(".py") or "/" in mod:
        p = pathlib.Path(mod).resolve()
        if not p.exists():
            raise FileNotFoundError(f"method file not found: {p}")
        name = "osm_" + hashlib.md5(str(p).encode()).hexdigest()[:12]
        if name in sys.modules:
            module = sys.modules[name]
        else:
            if str(p.parent) not in sys.path:
                sys.path.insert(0, str(p.parent))
            sp = importlib.util.spec_from_file_location(name, p)
            module = importlib.util.module_from_spec(sp)
            sys.modules[name] = module
            sp.loader.exec_module(module)
        src = str(p)
    else:
        module = importlib.import_module(mod)
        src = getattr(module, "__file__", mod)
    if not hasattr(module, cls):
        raise AttributeError(f"{mod} has no class {cls!r}")
    return getattr(module, cls), src


def infer_family(method, src: str) -> str:
    fam = getattr(method, "family", None)
    if fam:
        return str(fam)
    for part in reversed(pathlib.Path(src).parts):
        m = re.fullmatch(r"f\d+_(.+)", part)
        if m:
            return m.group(1)
    if "harness" in pathlib.Path(src).parts:
        return "baseline"
    return ""


def build_method(cls, kwargs):
    m = cls(**(kwargs or {}))
    api.check_method_attrs(m)
    return m


# ------------------------------------------------------------------------------------------- jobs
def episode_jobs(qc: store.QueryCell, subsample: str | None, episodes: list[int] | None):
    """[(episode index, rows int64[])] in episode order."""
    idx = range(len(qc.episodes)) if episodes is None else sorted(set(int(i) for i in episodes))
    tokmask = qc.tok_index >= 0 if subsample == "tok" else None
    jobs = []
    for i in idx:
        e = qc.episodes[i]
        rows = np.arange(e["start"], e["end"], dtype=np.int64)
        if tokmask is not None:
            rows = rows[tokmask[rows]]
        if rows.size:
            jobs.append((i, rows))
    if subsample not in (None, "tok"):
        raise ValueError(f"--subsample must be 'tok' or omitted, got {subsample!r}")
    return jobs


def split_chunks(jobs, n_chunks: int):
    """Contiguous episode chunks balanced by decision count (deterministic)."""
    if not jobs:
        return []
    n_chunks = max(1, min(n_chunks, len(jobs)))
    w = np.array([r.size for _, r in jobs], np.float64)
    cum = np.cumsum(w)
    tot = cum[-1]
    cuts = sorted(set(int(np.searchsorted(cum, tot * k / n_chunks, side="left")) + 1 for k in range(1, n_chunks)))
    cuts = [c for c in cuts if 0 < c < len(jobs)]
    bounds = [0] + cuts + [len(jobs)]
    return [jobs[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


# ------------------------------------------------------------------------------------------ worker
def _winit(counter):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if "torch" in sys.modules:
        try:
            sys.modules["torch"].set_num_threads(1)
        except Exception:
            pass
    with counter.get_lock():
        _G["wid"] = counter.value
        counter.value += 1
    _G["method"].prof = api.Profiler() if _G["profile"] else api.NULL_PROFILER


def _wstate():
    st = _G.get("wstate")
    if st is None:
        st = _G["wstate"] = {"eps": 0, "dec": 0, "t0": time.monotonic(), "tlast": time.monotonic()}
    return st


def _progress(force=False, **extra):
    st = _wstate()
    now = time.monotonic()
    if not force and now - st["tlast"] < PROGRESS_EVERY_S:
        return
    st["tlast"] = now
    el = now - st["t0"]
    plog(_G["progress"], ev="worker", cell=_G["cell"], worker=_G.get("wid", 0), episodes_done=st["eps"],
         episodes_total=_G["n_eps"], decisions_done=st["dec"], elapsed_s=round(el, 2),
         decisions_per_s=round(st["dec"] / el, 2) if el > 0 else None, rss_mb=round(rss_mb(), 1), **extra)


def _run_jobs(method, jobs, A, gt, lib_sizes, H, cell, seed, progress=True):
    """Core loop shared by workers, the in-process path, the timing pass and smoke. Returns a dict of lists."""
    episodes = _G["episodes"]
    out = {"rows": [], "top1": [], "lib": [], "topk": [], "scores": [], "conf": [], "synth": [], "used": [],
           "t_ns": [], "extras": []}
    pc = time.perf_counter_ns
    st = _wstate() if progress else None
    for ei, rows in jobs:
        e = episodes[ei]
        sd = ep_seed(seed, e["uid"])
        ev = api.EpisodeView(e["uid"], e["task"], e["task_id"], e["init"], ei, sd)
        np.random.seed(sd % 2**32)
        random.seed(sd)
        method.reset(ev)
        start = int(e["start"])
        tid = int(e["task_id"])
        for r in rows:
            r = int(r)
            q = api.QueryView(A, r, start, r - start, tid, ev, gt)
            t0 = pc()
            res = method.query(q)
            dt = pc() - t0
            topk, scores, conf, lib, act, ex = api.validate_result(
                res, lib_sizes=lib_sizes, H=H, where=f"cell={cell} uid={e['uid']} step={r - start} row={r}")
            k = min(topk.shape[0], api.TOPK_SAVE)
            out["rows"].append(r)
            out["top1"].append(int(topk[0]))
            out["lib"].append(lib)
            out["topk"].append(topk[:k])
            out["scores"].append(scores[:k])
            out["conf"].append(conf)
            out["used"].append(act is not None)
            out["synth"].append(None if act is None else np.asarray(dims.valid_action(act), np.float32))
            out["t_ns"].append(dt)
            out["extras"].append(ex)
            if st is not None:
                st["dec"] += 1
                _progress()
        if st is not None:
            st["eps"] += 1
    return out


class WorkerError(RuntimeError):
    """A forked worker failed; str() is the short cause, .worker_traceback the full worker traceback."""

    def __init__(self, cell, ci, err, tb):
        super().__init__(f"{cell}: chunk {ci} failed in a worker: {err}")
        self.worker_traceback = tb


def _run_chunk(ci, catch=True):
    if not catch:
        g = _G
        res = _run_jobs(g["method"], g["chunks"][ci], g["A"], g["gt"], g["lib_sizes"], g["H"], g["cell"], g["seed"])
        res["ci"] = ci
        res["prof"] = g["method"].prof.take() if g["profile"] else {}
        return res
    try:
        g = _G
        t0 = time.monotonic()
        res = _run_jobs(g["method"], g["chunks"][ci], g["A"], g["gt"], g["lib_sizes"], g["H"], g["cell"], g["seed"])
        res["ci"] = ci
        res["prof"] = g["method"].prof.take() if g["profile"] else {}
        el = time.monotonic() - t0
        plog(g["progress"], ev="chunk_done", cell=g["cell"], worker=g.get("wid", 0), chunk=ci,
             episodes=len(g["chunks"][ci]), decisions=len(res["rows"]), seconds=round(el, 3),
             decisions_per_s=round(len(res["rows"]) / el, 2) if el > 0 else None, rss_mb=round(rss_mb(), 1))
        return res
    except Exception as exc:  # report with the full worker traceback
        tb = traceback.format_exc()
        plog(_G.get("progress"), ev="error", cell=_G.get("cell"), worker=_G.get("wid"), chunk=ci, error=repr(exc),
             traceback=tb)
        return {"ci": ci, "error": f"{type(exc).__name__}: {exc}", "traceback": tb}


# ------------------------------------------------------------------------------------------ extras
def stack_extras(ex_list, n):
    """{key: array} for the npz. Same-shape values -> float32[n, *shape] (NaN rows where missing); ragged ->
    flattened, NaN-padded float32[n, maxlen] plus <key>__len int32[n] (-1 where missing)."""
    keys = sorted({k for d in ex_list if d for k in d})
    out = {}
    for k in keys:
        vals = [None if not d or k not in d else np.asarray(d[k], np.float32) for d in ex_list]
        shapes = {v.shape for v in vals if v is not None}
        if len(shapes) == 1:
            shp = shapes.pop()
            a = np.full((n,) + shp, np.nan, np.float32)
            for i, v in enumerate(vals):
                if v is not None:
                    a[i] = v
            out[f"x_{k}"] = a
        else:
            flat = [None if v is None else v.ravel() for v in vals]
            ml = max(v.size for v in flat if v is not None)
            a = np.full((n, ml), np.nan, np.float32)
            ln = np.full(n, -1, np.int32)
            for i, v in enumerate(flat):
                if v is not None:
                    a[i, :v.size] = v
                    ln[i] = v.size
            out[f"x_{k}"] = a
            out[f"x_{k}__len"] = ln
    return out


# --------------------------------------------------------------------------------------- timing pass
def timing_jobs(qc, subsample, n=TIMING_QUERIES, episodes=None):
    """Fixed query set for the single-thread timing pass: whole episodes in a seeded random order until n
    decisions (the last one truncated). Depends only on the cell (and subsample / episode restriction), not on the
    method."""
    jobs = episode_jobs(qc, subsample, episodes)
    order = np.random.default_rng(TIMING_SEED).permutation(len(jobs))
    sel, tot = [], 0
    for j in order:
        ei, rows = jobs[j]
        take = rows[: n - tot]
        sel.append((ei, take))
        tot += take.size
        if tot >= n:
            break
    return sel


def timing_pass(method, qc, A, gt, lib_sizes, H, cell, seed, subsample, episodes=None):
    jobs = timing_jobs(qc, subsample, episodes=episodes)
    method.prof = api.NULL_PROFILER
    cold = _run_jobs(method, jobs, A, gt, lib_sizes, H, cell, seed, progress=False)         # warm-up pass
    prof = api.Profiler()
    method.prof = prof
    hot = _run_jobs(method, jobs, A, gt, lib_sizes, H, cell, seed, progress=False)
    method.prof = api.NULL_PROFILER
    t = np.asarray(hot["t_ns"], np.float64) / 1e6
    tc = np.asarray(cold["t_ns"], np.float64) / 1e6
    return {"n_timing_queries": int(t.size), "ms_per_query": float(t.mean()) if t.size else math.nan,
            "ms_per_query_p50": float(np.median(t)) if t.size else math.nan,
            "ms_per_query_p95": float(np.percentile(t, 95)) if t.size else math.nan,
            "ms_per_query_cold": float(tc.mean()) if tc.size else math.nan,
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}, \
        api.Profiler.stats(prof.samples, n_queries=int(t.size))


def run_jobs_inprocess(method, qc, jobs, *, lib_sizes, seed=0, cell=None):
    """Run (episode, rows) jobs in this process in the given order (no metrics). Used by smoke's leak check."""
    _G.clear()
    _G.update(episodes=[{k: e[k] for k in ("uid", "task", "task_id", "init", "start")} for e in qc.episodes],
              progress=None, cell=cell or qc.cell, n_eps=len(jobs))
    A = api.QueryArrays(qc)
    gt = qc.a_inf if getattr(method, "uses_gt", False) else None
    return _run_jobs(method, jobs, A, gt, lib_sizes, dims.HORIZON[qc.model], cell or qc.cell, seed, progress=False)


# -------------------------------------------------------------------------------------------- cell
def _fit_cell(cls, kwargs, cell, *, root, out_dir, seed, profile, progress_path=None):
    """Build + fit the method for one cell; returns the pieces run_cell / timing_only_cell need."""
    root = pathlib.Path(root)
    qc = store.QueryCell(root, cell)
    lib = store.LibraryView(root, qc.lib_key, "current")
    H = dims.HORIZON[qc.model]
    if qc.H != H or lib.H != H:
        raise store.StoreError(f"{cell}: horizon mismatch queries H={qc.H} library H={lib.H} expected {H}")
    method = build_method(cls, kwargs)
    scratch = out_dir / "scratch" / cell
    scratch.mkdir(parents=True, exist_ok=True)
    fit_prof = api.Profiler() if profile else api.NULL_PROFILER
    ctx = api.Context(root=root, cell=cell, seed=seed, scratch=scratch, prof=fit_prof)
    method.prof = fit_prof
    plog(progress_path, ev="cell_start", cell=cell, method=method.name)
    t0 = time.perf_counter()
    method.fit(lib, ctx)
    fit_s = time.perf_counter() - t0
    fit_rss, fit_peak = rss_mb(), peak_rss_mb()
    method.prof = api.NULL_PROFILER
    bpe = float(method.bytes_per_entry())
    plog(progress_path, ev="fit_done", cell=cell, fit_s=round(fit_s, 3), rss_mb=round(fit_rss, 1))

    # libraries a Result may refer to: every stored library of this model x suite + the registered ones
    lib_sizes = {"current": lib.L}
    tables = {"current": {"action": lib.action, "progress": lib.progress if lib.has("progress") else None}}
    for lname in store.library_names(root, qc.lib_key):
        if lname == "current":
            continue
        bl = ctx.open_library(lname)
        lib_sizes[lname] = bl.L
        tables[lname] = {"action": bl.action, "progress": bl.progress if bl.has("progress") else None}
    for name, ent in ctx.registered.items():
        lib_sizes[name] = int(ent["action"].shape[0])
        tables[name] = {"action": ent["action"], "progress": ent.get("progress")}
    return dict(method=method, qc=qc, lib=lib, H=H, ctx=ctx, fit_s=fit_s, fit_rss=fit_rss, fit_peak=fit_peak, bpe=bpe,
                lib_sizes=lib_sizes, tables=tables, fit_prof=fit_prof)


def run_cell(cls, kwargs, cell, *, root, out_dir: pathlib.Path, workers, subsample=None, seed=0, profile=False,
             episodes=None, progress_path=None, do_timing=True, fam=None, src=""):
    """Fit + evaluate one cell; writes <cell>.npz / <cell>.json into out_dir and returns the json dict."""
    root = pathlib.Path(root)
    ph = {}
    F = _fit_cell(cls, kwargs, cell, root=root, out_dir=out_dir, seed=seed, profile=profile,
                  progress_path=progress_path)
    method, qc, lib, H = F["method"], F["qc"], F["lib"], F["H"]
    fit_s, lib_sizes, tables, fit_prof = F["fit_s"], F["lib_sizes"], F["tables"], F["fit_prof"]
    ph["fit_s"] = fit_s

    jobs = episode_jobs(qc, subsample, episodes)
    missing = sorted({qc.episodes[i]["task_id"] for i, _ in jobs} - set(lib.tasks()))
    if missing:
        raise store.StoreError(f"{cell}: query task_ids {missing} have no candidates in library {qc.lib_key}/current")
    A = api.QueryArrays(qc)
    gt = qc.a_inf if getattr(method, "uses_gt", False) else None
    W = max(1, min(int(workers), MAX_WORKERS))
    chunks = split_chunks(jobs, W * 4 if W > 1 else 1)
    _G.clear()
    _G.update(method=method, A=A, gt=gt, lib_sizes=lib_sizes, H=H, cell=cell, seed=seed, chunks=chunks,
              episodes=[{k: e[k] for k in ("uid", "task", "task_id", "init", "start")} for e in qc.episodes],
              profile=bool(profile), progress=progress_path, n_eps=len(jobs))
    tw = time.perf_counter()
    results = {}
    if W == 1 or len(chunks) <= 1:
        _G["wid"] = 0
        method.prof = api.Profiler() if profile else api.NULL_PROFILER
        for ci in range(len(chunks)):
            results[ci] = _run_chunk(ci, catch=False)      # in-process: the original exception propagates
        _progress(force=True, final=True)
    else:
        mctx = mp.get_context("fork")
        counter = mctx.Value("i", 0)
        with mctx.Pool(min(W, len(chunks)), initializer=_winit, initargs=(counter,)) as pool:
            for r in pool.imap_unordered(_run_chunk, range(len(chunks))):
                if "error" in r:
                    pool.terminate()
                    raise WorkerError(cell, r["ci"], r["error"], r["traceback"])
                results[r["ci"]] = r
    wall = time.perf_counter() - tw
    ph["workers_s"] = wall
    _G.pop("wstate", None)

    # gather in episode order
    t1 = time.perf_counter()
    cat = {k: [] for k in ("rows", "top1", "lib", "topk", "scores", "conf", "synth", "used", "t_ns", "extras")}
    prof_w: dict = {}
    for ci in range(len(chunks)):
        r = results[ci]
        for k in cat:
            cat[k].extend(r[k])
        for k, v in r.get("prof", {}).items():
            prof_w.setdefault(k, []).extend(v)
    n = len(cat["rows"])
    rows = np.asarray(cat["rows"], np.int64)
    top1 = np.asarray(cat["top1"], np.int64)
    lib_names = sorted(set(cat["lib"]), key=lambda s: (s != "current", s))
    code = {nm: i for i, nm in enumerate(lib_names)}
    lib_code = np.asarray([code[x] for x in cat["lib"]], np.int8)
    topk = np.full((n, api.TOPK_SAVE), -1, np.int32)
    tks = np.full((n, api.TOPK_SAVE), np.nan, np.float32)
    for i, (a, s) in enumerate(zip(cat["topk"], cat["scores"])):
        topk[i, :a.size] = a
        tks[i, :s.size] = s
    conf = np.asarray(cat["conf"], np.float64)
    used = np.asarray(cat["used"], bool)
    synth = None
    if used.any():
        synth = np.full((n, dims.EXEC_STEPS, dims.ACT_DIMS), np.nan, np.float32)
        for i, s in enumerate(cat["synth"]):
            if s is not None:
                synth[i] = s
    t_us = np.asarray(cat["t_ns"], np.float64) / 1e3
    ph["gather_s"] = time.perf_counter() - t1

    # metrics
    t1 = time.perf_counter()
    sigma = store.action_sigma(str(root), qc.lib_key)
    dm = metrics.decision_metrics(qc, rows, top1, lib_code, lib_names, tables,
                                  None if synth is None else synth.astype(np.float64), used, sigma, lib,
                                  oracle_cache=out_dir.parent / "_cache")
    floor = metrics.load_floor(root, qc.lib_key, sigma)
    rec_top1 = np.asarray(qc.rec_top1[rows], np.int64)
    is_cur = lib_code == code.get("current", -1)
    agg = metrics.aggregate(dm, conf, floor, top1, is_cur, rec_top1, used)
    flip = np.where(is_cur, (top1 != rec_top1).astype(np.int8), np.int8(-1)).astype(np.int8)
    ph["metrics_s"] = time.perf_counter() - t1

    timing, prof_t = ({}, {})
    if do_timing:
        t1 = time.perf_counter()
        _G["wstate"] = None
        timing, prof_t = timing_pass(method, qc, A, gt, lib_sizes, H, cell, seed, subsample, episodes)
        timing["timing_source"] = "in_run"
        ph["timing_s"] = time.perf_counter() - t1
    timing.update(fit_s=fit_s, worker_wall_s=wall, workers=W, n_chunks=len(chunks))

    t1 = time.perf_counter()
    cands = [lib.rows_of_task(t).size for t in sorted({qc.episodes[i]["task_id"] for i, _ in jobs})]
    npz = dict(row=rows, ep=np.asarray(qc.ep[rows], np.int32), step=dm["step"].astype(np.int16),
               task_id=dm["task_id"].astype(np.int32), top1=top1.astype(np.int32), topk=topk, topk_scores=tks,
               confidence=conf, err=dm["err"], grip_mis=dm["grip_mis"].astype(np.float32), phase_err=dm["phase_err"],
               oracle_err=dm["oracle_err"], oracle_row=dm["oracle_row"].astype(np.int32), regret=dm["regret"],
               bin=dm["bin"], used_synth=used, lib_code=lib_code, lib_names=np.asarray(lib_names),
               library=np.asarray(lib_names[0] if len(lib_names) == 1 else "mixed"), flip=flip,
               t_query_us=t_us.astype(np.float32),
               uses_nonlibrary_action=np.asarray(bool(getattr(method, "uses_nonlibrary_action", False))))
    if synth is not None:
        npz["synth_seg"] = synth
    npz.update(stack_extras(cat["extras"], n))
    np.savez_compressed(out_dir / f"{cell}.npz", **npz)
    ph["save_s"] = time.perf_counter() - t1
    timing["phases"] = ph

    summ = {
        "cell": cell, "model": qc.model, "suite": qc.suite, "arm": qc.arm, "method": method.name, "family": fam,
        "tier": method.tier, "uses_gt": bool(getattr(method, "uses_gt", False)),
        "uses_nonlibrary_action": bool(getattr(method, "uses_nonlibrary_action", False)), "method_src": src,
        "kwargs": kwargs or {}, "seed": seed, "subsample": subsample or "", "n_decisions": n, "n_episodes": len(jobs),
        "library": lib_names[0] if len(lib_names) == 1 else "mixed", "lib_names": lib_names,
        "metrics": {k: v for k, v in agg.items() if k != "rc_curve"}, "rc_curve": agg["rc_curve"],
        "floor": floor, "sigma": sigma.tolist(), "timing": timing,
        "fit": {"fit_s": fit_s, "rss_mb_after_fit": F["fit_rss"], "peak_rss_mb_after_fit": F["fit_peak"],
                "bytes_per_entry": F["bpe"], "L": lib.L, "libraries": lib_sizes,
                "cands_per_task": {"min": int(min(cands)) if cands else 0, "mean": float(np.mean(cands)) if cands else 0,
                                   "max": int(max(cands)) if cands else 0}},
        "profile": {"timing_pass": prof_t, "workers": api.Profiler.stats(prof_w, n_queries=n) if profile else None,
                    "fit": api.Profiler.stats(fit_prof.samples) if profile else None,
                    "query_total_us": {"mean": float(t_us.mean()) if n else math.nan,
                                       "p50": float(np.median(t_us)) if n else math.nan,
                                       "p95": float(np.percentile(t_us, 95)) if n else math.nan}},
        "extras_keys": sorted(k[2:] for k in npz if k.startswith("x_") and not k.endswith("__len")),
        "data_checks": {"exec_hit_counts": qc.exec_hit_counts()},
        "npz": str(out_dir / f"{cell}.npz"),
    }
    write_json_atomic(out_dir / f"{cell}.json", summ)
    plog(progress_path, ev="cell_done", cell=cell, n=n, err_mean=agg["err_mean"], wall_s=round(wall, 2),
         ms_per_query=timing.get("ms_per_query"), phases={k: round(v, 3) for k, v in ph.items()})
    return summ


def write_json_atomic(path, obj) -> None:
    path = pathlib.Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=metrics._json_default))
    os.replace(tmp, path)


TIMING_KEYS = ("n_timing_queries", "ms_per_query", "ms_per_query_p50", "ms_per_query_p95", "ms_per_query_cold",
               "threads")


def timing_only_cell(cls, kwargs, cell, *, root, out_dir: pathlib.Path, subsample=None, seed=0, episodes=None,
                     progress_path=None, pin_core=None):
    """Re-fit, run only the single-thread timing pass, and merge its fields into the existing <cell>.json."""
    jp = out_dir / f"{cell}.json"
    if not jp.exists():
        raise FileNotFoundError(f"{jp} does not exist: run the accuracy job of this cell first")
    F = _fit_cell(cls, kwargs, cell, root=root, out_dir=out_dir, seed=seed, profile=False, progress_path=None)
    method, qc = F["method"], F["qc"]
    A = api.QueryArrays(qc)
    gt = qc.a_inf if getattr(method, "uses_gt", False) else None
    _G.clear()
    _G.update(episodes=[{k: e[k] for k in ("uid", "task", "task_id", "init", "start")} for e in qc.episodes],
              progress=None, cell=cell, n_eps=0, wstate=None)
    t1 = time.perf_counter()
    timing, prof_t = timing_pass(method, qc, A, gt, F["lib_sizes"], F["H"], cell, seed, subsample, episodes)
    tsec = time.perf_counter() - t1
    j = json.loads(jp.read_text())
    j.setdefault("timing", {}).update({k: timing[k] for k in TIMING_KEYS if k in timing})
    j["timing"].update(timing_source="timing_only", timing_fit_s=F["fit_s"], timing_pass_s=tsec,
                       pinned_core=pin_core if pin_core is not None else sorted(os.sched_getaffinity(0)),
                       timing_concurrency=int(os.environ.get("OFFLINE_SEARCH_TIMING_CONCURRENCY", "1")))
    j.setdefault("profile", {})["timing_pass"] = prof_t
    write_json_atomic(jp, j)
    plog(progress_path, ev="timing_done", cell=cell, ms_per_query=timing.get("ms_per_query"),
         pinned_core=pin_core, fit_s=round(F["fit_s"], 3), seconds=round(tsec, 3))
    return j


def scoreboard_row(s: dict, *, rnd, run_ts, name, fam) -> dict:
    """Scoreboard row of one <cell>.json summary."""
    return dict(round=rnd, run_ts=run_ts, method=name, family=fam, tier=s["tier"], uses_gt=s["uses_gt"],
                uses_nonlibrary_action=s.get("uses_nonlibrary_action", False),
                cell=s["cell"], model=s["model"], suite=s["suite"], arm=s["arm"], subsample=s["subsample"],
                library=s["library"], bytes_per_entry=s["fit"]["bytes_per_entry"],
                ms_per_query=s["timing"].get("ms_per_query", math.nan),
                ms_per_query_p50=s["timing"].get("ms_per_query_p50", math.nan),
                fit_s=s["fit"]["fit_s"], peak_rss_mb=s["fit"]["peak_rss_mb_after_fit"], L=s["fit"]["L"],
                out=s["npz"], **s["metrics"])


# --------------------------------------------------------------------------------------------- run
def run(method_spec, kwargs=None, cells="all", *, root, out, workers=CPU_COUNT, subsample=None, seed=0, profile=False,
        round_=None, family=None, scoreboard="default", episodes=None, timing=True, verbose=True, job=False,
        timing_only=False, pin_core=None) -> dict:
    """Run one method over cells. job=True / timing_only=True are the batch.py modes: no markers, run_meta,
    summary or scoreboard; each cell's outcome is written to <out>/<name>/_jobs/<cell>[.timing].json."""
    root = pathlib.Path(root)
    out = pathlib.Path(out)
    cls, src = load_method_class(method_spec)
    probe = build_method(cls, kwargs)
    name = probe.name
    fam = family or infer_family(probe, src)
    del probe
    mdir = out / name
    mdir.mkdir(parents=True, exist_ok=True)
    batch_mode = job or timing_only
    if not batch_mode:
        for mk in ("DONE", "ERROR"):
            p = mdir / mk
            if p.exists():
                p.unlink()
    prog = mdir / "progress.jsonl"
    cell_list = store.resolve_cells(cells, root) if isinstance(cells, str) else list(cells)
    sb = (out.parent / "scoreboard.csv") if scoreboard == "default" else (None if scoreboard in (None, "none") else
                                                                          pathlib.Path(scoreboard))
    rnd = round_ or out.name
    meta = {"argv": sys.argv, "method_spec": method_spec, "method_src": src, "method": name, "family": fam,
            "kwargs": kwargs or {}, "cells": cell_list, "root": str(root), "out": str(out), "workers": workers,
            "subsample": subsample, "seed": seed, "profile": profile, "round": rnd, "scoreboard": str(sb),
            "host": socket.gethostname(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if not batch_mode:
        (mdir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    mode = "timing_only" if timing_only else ("job" if job else "run")
    plog(prog, ev="run_start", method=name, cells=cell_list, workers=workers, root=str(root), mode=mode)
    results, errors, skipped = {}, {}, {}
    run_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    jdir = mdir / "_jobs"
    if batch_mode:
        jdir.mkdir(exist_ok=True)
    for cell in cell_list:
        t0 = time.time()
        rec = {"cell": cell, "method": name, "mode": mode, "pid": os.getpid(), "workers": workers,
               "pinned_core": pin_core}
        try:
            if timing_only:
                s = timing_only_cell(cls, kwargs, cell, root=root, out_dir=mdir, subsample=subsample, seed=seed,
                                     episodes=episodes, progress_path=prog, pin_core=pin_core)
            else:
                s = run_cell(cls, kwargs, cell, root=root, out_dir=mdir, workers=workers, subsample=subsample,
                             seed=seed, profile=profile, episodes=episodes, progress_path=prog,
                             do_timing=timing and not job, fam=fam, src=src)
            results[cell] = s
            rec["status"] = "ok"
            if sb is not None and not batch_mode:
                metrics.append_scoreboard(sb, scoreboard_row(s, rnd=rnd, run_ts=run_ts, name=name, fam=fam))
            if verbose:
                m = s["metrics"]
                print(f"[{name}] {cell}{' (timing)' if timing_only else ''}: n={m['n']} err={m['err_mean']:.4f} "
                      f"(med {m['err_median']:.4f}) oracle={m['oracle_err_mean']:.4f} indist={m['indist']:.3f} "
                      f"aurc={m['aurc']:.4f} grip={m['grip_mis']:.4f} flip={m['flip_rate']:.4f} "
                      f"fit={s['fit']['fit_s']:.1f}s q={s['timing'].get('ms_per_query', math.nan):.3f}ms "
                      f"wall={s['timing'].get('worker_wall_s', math.nan):.1f}s", flush=True)
        except api.SkipCell as exc:
            skipped[cell] = str(exc)
            rec.update(status="skip", reason=str(exc))
            plog(prog, ev="skip", cell=cell, reason=str(exc))
            print(f"[{name}] {cell}: skipped ({exc})", flush=True)
        except Exception as exc:
            tb = traceback.format_exc() + ("\n-- worker traceback --\n" + exc.worker_traceback
                                           if isinstance(exc, WorkerError) else "")
            errors[cell] = tb
            rec.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=tb)
            plog(prog, ev="error", cell=cell, error=repr(exc), traceback=tb, mode=mode)
            print(f"[{name}] {cell}: ERROR {exc!r}\n{tb}", file=sys.stderr, flush=True)
        rec["seconds"] = round(time.time() - t0, 3)
        if batch_mode:
            write_json_atomic(jdir / f"{cell}{'.timing' if timing_only else ''}.json", rec)
    plog(prog, ev="run_done", method=name, ok=sorted(results), failed=sorted(errors), skipped=sorted(skipped),
         mode=mode)
    if batch_mode:
        return {"method": name, "dir": str(mdir), "results": results, "errors": errors, "skipped": skipped}
    if results:
        metrics.summarize_method(mdir)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["errors"] = sorted(errors)
    meta["skipped"] = skipped
    (mdir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    if errors:
        (mdir / "ERROR").write_text("failed cells: " + ", ".join(sorted(errors)) + "\n\n" +
                                    "\n\n".join(f"== {c}\n{tb}" for c, tb in errors.items()))
    else:
        (mdir / "DONE").write_text(f"ok {len(results)} cells, skipped {sorted(skipped)} {meta['finished']}\n")
    return {"method": name, "dir": str(mdir), "results": results, "errors": errors, "skipped": skipped}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, help="<dotted.module or path/to/file.py>:<ClassName>")
    ap.add_argument("--kwargs", default="{}", help="JSON dict passed to the method constructor")
    ap.add_argument("--cells", default="all", help="'all' or comma list, e.g. pi05_spatial_inf,groot_l10_cache")
    ap.add_argument("--root", default=str(store.FULL_ROOT))
    ap.add_argument("--out", required=True, help="round results dir, e.g. exp/offline_search/results/r01")
    ap.add_argument("--workers", type=int, default=CPU_COUNT, help=f"forked workers per cell (default {CPU_COUNT})")
    ap.add_argument("--subsample", choices=["tok"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="section profiler on in the workers (always on in timing)")
    ap.add_argument("--round", default=None, help="scoreboard round label (default: basename of --out)")
    ap.add_argument("--family", default=None)
    ap.add_argument("--scoreboard", default="default", help="CSV path, 'default' (<out>/../scoreboard.csv) or 'none'")
    ap.add_argument("--episodes-limit", type=int, default=None, help="debug: first K episodes of every cell")
    ap.add_argument("--no-timing", action="store_true", help="skip the 300-query single-thread timing pass")
    ap.add_argument("--allow-gpu-fit", action="store_true", help="keep CUDA visible in the main process (T2 fit)")
    ap.add_argument("--job", action="store_true", help="batch.py job mode (no markers / meta / summary / scoreboard)")
    ap.add_argument("--timing-only", action="store_true", help="re-fit + timing pass only, merged into <cell>.json")
    ap.add_argument("--pin-core", type=int, default=None, help="pin this process to logical CPU K")
    a = ap.parse_args(argv)
    kw = json.loads(a.kwargs)
    if not isinstance(kw, dict):
        ap.error("--kwargs must be a JSON object")
    if a.pin_core is not None:
        os.sched_setaffinity(0, {a.pin_core})
    eps = list(range(a.episodes_limit)) if a.episodes_limit else None
    r = run(a.method, kw, a.cells, root=a.root, out=a.out, workers=a.workers, subsample=a.subsample, seed=a.seed,
            profile=a.profile, round_=a.round, family=a.family, scoreboard=a.scoreboard, episodes=eps,
            timing=not a.no_timing, job=a.job, timing_only=a.timing_only, pin_core=a.pin_core)
    return 1 if r["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
