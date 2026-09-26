"""runprof -- where TIME and MEMORY go in a run, and how a method's latency grows with the library.

  summary  (default)  python -m exp.offline_search.profile.runprof summary RUN_OR_METHOD_DIR [--method M] [--cells all]
      per cell: fit_s, worker wall, throughput, single-thread ms/query (runner timing pass: mean/p50/p95/cold),
      multi-process t_query_us p50/p95/p99/max (npz), bytes_per_entry, L, RSS after fit / worker RSS peak;
      per-section us from <cell>.json "profile" (timing_pass, plus workers when the run used --profile) with the
      share of the query time; worker balance from progress.jsonl chunk_done events (busy seconds per worker,
      imbalance max/median, stragglers, slowest chunk rate vs median), errors / skips.
  scaling             python -m exp.offline_search.profile.runprof scaling --method <module_or_file>:<Class>
                          [--kwargs JSON] --cell pi05_spatial_inf [--root R] [--scales 0.5,1,2,4] [--n-queries 300]
      fits the method on library `current` rescaled to each factor (<1: whole episodes subsampled per task;
      >1: every row duplicated k times with remapped episode / prev / next; 1: an in-memory copy) and times
      query() single-threaded on the runner's fixed timing query set (warm-up pass + timed pass). The rescaled
      library is loaded into memory before fit, so fit_s excludes store IO (memory: scale x L x 256 KB for the
      two pooled keys, e.g. ~2.8 GB for l10 at 4x). Reports ms/query vs L and the log-log slope. Tokens are not
      rescaled (methods that need lib.tok() are not supported here).
  live                python -m exp.offline_search.profile.runprof live RUN_OR_METHOD_DIR [--stale 60]
      one-screen status of an in-flight run from progress.jsonl (+ DONE / ERROR markers). One-shot, no polling.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import common as C


# -------------------------------------------------------------------------------------------- utils
def read_progress(md: Path) -> list[dict]:
    p = Path(md) / "progress.jsonl"
    out = []
    if not p.exists():
        return out
    with open(p) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                out.append({"ev": "unparsable", "raw": ln[:200]})
    return out


def pct(x, q) -> float:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def progress_dirs(path, method=None) -> list[Path]:
    p = Path(path)
    if (p / "progress.jsonl").exists():
        return [p]
    subs = sorted(d for d in p.iterdir() if d.is_dir() and (d / "progress.jsonl").exists()) if p.exists() else []
    if method:
        subs = [d for d in subs if d.name in method.split(",")]
    return subs


# ------------------------------------------------------------------------------------------ summary
def worker_balance(events: list[dict], cell: str) -> dict:
    ch = [e for e in events if e.get("ev") == "chunk_done" and e.get("cell") == cell]
    wk = [e for e in events if e.get("ev") == "worker" and e.get("cell") == cell]
    if not ch and not wk:
        return {}
    busy, dec, rss = {}, {}, {}
    for e in ch:
        w = e.get("worker", e.get("pid"))
        busy[w] = busy.get(w, 0.0) + float(e.get("seconds") or 0.0)
        dec[w] = dec.get(w, 0) + int(e.get("decisions") or 0)
        rss[w] = max(rss.get(w, 0.0), float(e.get("rss_mb") or 0.0))
    for e in wk:
        w = e.get("worker", e.get("pid"))
        rss[w] = max(rss.get(w, 0.0), float(e.get("rss_mb") or 0.0))
    b = np.array(list(busy.values())) if busy else np.zeros(0)
    rates = np.array([e["decisions_per_s"] for e in ch if e.get("decisions_per_s")], np.float64)
    med = float(np.median(b)) if b.size else float("nan")
    strag = sorted([w for w, v in busy.items() if med > 0 and v > 1.25 * med], key=lambda w: -busy[w])
    slow = min(ch, key=lambda e: e.get("decisions_per_s") or np.inf) if ch else None
    return {"workers": len(busy) or len({e.get("worker") for e in wk}), "chunks": len(ch),
            "busy_max": float(b.max()) if b.size else None, "busy_med": med if b.size else None,
            "busy_min": float(b.min()) if b.size else None,
            "imbalance": float(b.max() / med) if b.size and med > 0 else None,
            "stragglers": strag[:5], "rss_peak_mb": max(rss.values()) if rss else None,
            "chunk_rate_med": float(np.median(rates)) if rates.size else None,
            "slowest_chunk": None if slow is None else {"chunk": slow.get("chunk"), "worker": slow.get("worker"),
                                                        "seconds": slow.get("seconds"), "decisions": slow.get("decisions"),
                                                        "rate": slow.get("decisions_per_s")},
            "dec_per_worker": {str(k): v for k, v in dec.items()}}


def cell_profile(md: Path, cell: str, events: list[dict]) -> dict:
    j = json.loads((md / f"{cell}.json").read_text()) if (md / f"{cell}.json").exists() else {}
    tm, ft, prof = j.get("timing", {}), j.get("fit", {}), j.get("profile") or {}
    row = {"cell": cell, "n": j.get("n_decisions"), "library": j.get("library"), "fit_s": ft.get("fit_s", tm.get("fit_s")),
           "wall_s": tm.get("worker_wall_s"), "workers": tm.get("workers"),
           "ms_q": tm.get("ms_per_query"), "ms_q_p50": tm.get("ms_per_query_p50"), "ms_q_p95": tm.get("ms_per_query_p95"),
           "ms_q_cold": tm.get("ms_per_query_cold"), "bpe": ft.get("bytes_per_entry"), "L": ft.get("L"),
           "rss_fit_mb": ft.get("rss_mb_after_fit"), "peak_rss_fit_mb": ft.get("peak_rss_mb_after_fit")}
    if row["n"] and row["wall_s"]:
        row["dec_per_s"] = row["n"] / row["wall_s"]
    npz = md / f"{cell}.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            if "t_query_us" in z.files:
                t = z["t_query_us"].astype(np.float64)
                row.update({"tq_p50": pct(t, 50), "tq_p95": pct(t, 95), "tq_p99": pct(t, 99), "tq_max": float(t.max()) if t.size else None})
    cs = [e for e in events if e.get("cell") == cell]
    t_start = next((e["t"] for e in cs if e.get("ev") == "cell_start"), None)
    t_done = next((e["t"] for e in cs if e.get("ev") == "cell_done"), None)
    if t_start and t_done:
        row["cell_wall_s"] = t_done - t_start
        row["overhead_s"] = row["cell_wall_s"] - (row.get("fit_s") or 0) - (row.get("wall_s") or 0)  # timing + metrics
    row["errors"] = sum(1 for e in cs if e.get("ev") == "error")
    sections = []
    q_us = 1000.0 * row["ms_q"] if row.get("ms_q") else None
    for src in ("timing_pass", "workers", "fit"):
        for name, st in (prof.get(src) or {}).items():
            sections.append({"cell": cell, "src": src, "section": name, "n": st.get("n"), "per_query_us": st.get("per_query_us"),
                             "mean_us": st.get("mean_us"), "p50_us": st.get("p50_us"), "p95_us": st.get("p95_us"),
                             "total_us": st.get("total_us"),
                             "share": (st.get("per_query_us") / q_us) if (src == "timing_pass" and q_us and st.get("per_query_us")) else None})
    return {"row": row, "sections": sections, "balance": worker_balance(events, cell)}


CELL_COLS = [("cell", "cell"), ("n", "n"), ("library", "lib"), ("fit_s", "fit_s", ".2f"), ("wall_s", "wall_s", ".1f"),
             ("dec_per_s", "dec/s", ".0f"), ("ms_q", "ms/q", ".3f"), ("ms_q_p50", "p50", ".3f"), ("ms_q_p95", "p95", ".3f"),
             ("ms_q_cold", "cold", ".3f"), ("tq_p50", "tq_p50us", ".0f"), ("tq_p95", "tq_p95us", ".0f"),
             ("tq_p99", "tq_p99us", ".0f"), ("tq_max", "tq_max_us", ".0f"), ("bpe", "bytes/entry", ".0f"), ("L", "L"),
             ("rss_fit_mb", "rss_fit_MB", ".0f"), ("overhead_s", "overhead_s", ".1f"), ("errors", "err")]
SEC_COLS = [("cell", "cell"), ("src", "src"), ("section", "section"), ("n", "n"), ("per_query_us", "us/query", ".2f"),
            ("share", "share", ".2f"), ("mean_us", "mean_us", ".2f"), ("p50_us", "p50_us", ".2f"), ("p95_us", "p95_us", ".2f")]
BAL_COLS = [("cell", "cell"), ("workers", "workers"), ("chunks", "chunks"), ("busy_max", "busy_max_s", ".2f"),
            ("busy_med", "busy_med_s", ".2f"), ("busy_min", "busy_min_s", ".2f"), ("imbalance", "max/med", ".2f"),
            ("stragglers", "stragglers"), ("chunk_rate_med", "chunk_dec/s_med", ".0f"), ("slow_rate", "slowest_dec/s", ".0f"),
            ("rss_peak_mb", "worker_rss_peak_MB", ".0f")]


def _cell_job(a):
    md, cell, events = a
    return cell_profile(Path(md), cell, events)


def summary(run_path, method=None, cells="all", root=C.DEFAULT_ROOT, out=None, quiet=False, procs=None) -> dict:
    res = {}
    mds = C.method_dirs(run_path, method)
    evs = {str(md): read_progress(md) for md in mds}
    jobs = [(str(md), c, [e for e in evs[str(md)] if e.get("cell") == c])
            for md in mds for c in C.run_cells(md, C.expand_cells(cells))]
    done = C.pmap(_cell_job, jobs, procs)                     # one pool over every (method, cell)
    for md in mds:
        ev = evs[str(md)]
        cs = [j[1] for j in jobs if j[0] == str(md)]
        per = [r for j, r in zip(jobs, done) if j[0] == str(md)]
        rows = [p["row"] for p in per]
        secs = [s for p in per for s in p["sections"]]
        bal = [{"cell": p["row"]["cell"], **p["balance"],
                "slow_rate": (p["balance"].get("slowest_chunk") or {}).get("rate")} for p in per if p["balance"]]
        tot_wall = sum((r.get("cell_wall_s") or 0) for r in rows)
        status = "DONE" if (md / "DONE").exists() else ("ERROR" if (md / "ERROR").exists() else "no marker")
        errs = [e for e in ev if e.get("ev") == "error"]
        skips = [e for e in ev if e.get("ev") == "skip"]
        L = [f"## runprof {md.name}  [{status}]  cells {len(cs)}  total cell wall {tot_wall:.1f}s"
             + (f"  errors {len(errs)}" if errs else "") + (f"  skipped {[e.get('cell') for e in skips]}" if skips else ""),
             "ms/q = single-thread timing pass (runner); tq_* = per-decision wall inside the worker pool (us);"
             " overhead = cell wall - fit - worker wall (timing pass + metrics + npz)."]
        L.append(C.md_table(rows, CELL_COLS, "per cell"))
        if secs:
            L.append("")
            L.append(C.md_table(secs, SEC_COLS, "sections (Profiler; share = of the timing-pass query time)"))
        if bal:
            L.append("")
            L.append(C.md_table(bal, BAL_COLS, "worker balance (progress.jsonl chunk_done)"))
        for e in errs[:5]:
            L.append(f"> error in {e.get('cell')}: {str(e.get('error'))[:300]}")
        s = "\n".join(L)
        if not quiet:
            print(s)
        rdir = C.report_dir(root, "runprof", md.name, out, multi=len(mds) > 1)
        C.write_json(rdir / "runprof.json", {"method": md.name, "status": status, "cells": rows, "sections": secs,
                                             "balance": bal, "errors": errs, "skipped": skips})
        C.write_csv(rdir / "runprof_cells.csv", rows)
        if secs:
            C.write_csv(rdir / "runprof_sections.csv", secs)
        (rdir / "runprof.md").write_text(s + "\n")
        if not quiet:
            print(f"\n[runprof] {md.name} -> {rdir}")
        res[md.name] = {"cells": rows, "sections": secs, "balance": bal, "status": status, "report_dir": str(rdir)}
    return res


# --------------------------------------------------------------------------------------------- live
def live(run_path, method=None, stale_s=60.0) -> str:
    now = time.time()
    L = [f"== live {Path(run_path)}  ({C.now_local()})"]
    for md in progress_dirs(run_path, method):
        ev = read_progress(md)
        status = "DONE" if (md / "DONE").exists() else ("ERROR" if (md / "ERROR").exists() else "running")
        start = next((e for e in ev if e.get("ev") == "run_start"), {})
        cells = start.get("cells") or []
        done = {e["cell"]: e for e in ev if e.get("ev") == "cell_done"}
        errs = [e for e in ev if e.get("ev") == "error"]
        skipped = {e["cell"] for e in ev if e.get("ev") == "skip"}
        last_t = max((e.get("t", 0) for e in ev), default=0)
        el = now - start.get("t", now)
        L.append(f"-- {md.name}: {status}  cells {len(done)}/{len(cells)} done"
                 + (f", {len(skipped)} skipped" if skipped else "") + (f", {len(errs)} ERRORS" if errs else "")
                 + f"  elapsed {el / 60:.1f} min  last event {now - last_t:.0f}s ago  workers {start.get('workers')}")
        for c, e in done.items():
            L.append(f"   done {c:20s} n={e.get('n')} err={C.fmt_val(e.get('err_mean'), '.4f')} wall={C.fmt_val(e.get('wall_s'), '.1f')}s "
                     f"ms/q={C.fmt_val(e.get('ms_per_query'), '.3f')}")
        cur = [e for e in ev if e.get("ev") == "cell_start" and e["cell"] not in done and e["cell"] not in skipped]
        if cur and status == "running":
            c = cur[-1]["cell"]
            ce = [e for e in ev if e.get("cell") == c]
            fit = next((e for e in ce if e.get("ev") == "fit_done"), None)
            wk = {}
            for e in ce:
                if e.get("ev") in ("worker", "chunk_done"):
                    wk.setdefault(e.get("worker"), []).append(e)
            if fit is None:
                L.append(f"   now  {c}: fitting for {now - cur[-1]['t']:.0f}s")
            else:
                last = {w: v[-1] for w, v in wk.items()}
                wev = {w: [x for x in v if x.get("ev") == "worker"] for w, v in wk.items()}
                eps_done = sum(v[-1].get("episodes_done", 0) for v in wev.values() if v)
                dec_done = sum(v[-1].get("decisions_done", 0) for v in wev.values() if v)
                tot_eps = next((x.get("episodes_total") for v in wev.values() for x in v if x.get("episodes_total")), None)
                run_s = now - fit["t"]
                eta = (tot_eps - eps_done) / (eps_done / run_s) if tot_eps and eps_done and run_s > 0 else None
                stale = [w for w, e in last.items() if now - e.get("t", now) > stale_s]
                rss = max((x.get("rss_mb") or 0) for v in wk.values() for x in v) if wk else None
                nch = sum(1 for e in ce if e.get("ev") == "chunk_done")
                L.append(f"   now  {c}: fit {fit.get('fit_s')}s; {len(wk)} workers reporting, chunks done {nch}, "
                         f"episodes {eps_done}/{tot_eps or '?'} decisions {dec_done}  "
                         f"rate {dec_done / run_s if run_s > 0 else 0:.0f} dec/s  "
                         f"ETA {'?' if eta is None else f'{eta / 60:.1f} min'}  rss peak {C.fmt_val(rss, '.0f')} MB"
                         + (f"  STALE (> {stale_s:.0f}s): workers {stale}" if stale else ""))
        pending = [c for c in cells if c not in done and c not in skipped and c not in {e["cell"] for e in cur}]
        if pending:
            L.append(f"   pending {pending}")
        for e in errs[-3:]:
            L.append(f"   ERROR {e.get('cell')}: {str(e.get('error'))[:200]}")
    s = "\n".join(L)
    print(s)
    return s


# ------------------------------------------------------------------------------------------ scaling
class ScaledLibrary:
    """Duck-typed harness LibraryView over library current rescaled by `factor` (in memory).
    factor < 1: keep round(factor * n) whole episodes per task (library order kept);
    factor >= 1: round(factor) copies of every row, episode ids offset per copy, prev/next remapped inside a copy."""

    PTR = ("prev", "next")
    IDS = ("episode", "traj")

    def __init__(self, base, factor: float, seed: int = 0):
        self._base = base
        L = base.L
        rng = np.random.default_rng(seed)
        ep = np.asarray(base.episode, np.int64) if base.has("episode") else np.arange(L)
        task = np.asarray(base.task_id, np.int64)
        if factor < 1:
            keep = []
            for t in np.unique(task):
                eps = np.unique(ep[task == t])
                k = max(1, int(round(factor * eps.size)))
                keep.append(rng.choice(eps, k, replace=False))
            idx = np.flatnonzero(np.isin(ep, np.concatenate(keep)))
            copy = np.zeros(idx.size, np.int64)
        else:
            k = max(1, int(round(factor)))
            idx = np.tile(np.arange(L), k)
            copy = np.repeat(np.arange(k), L)
        self._idx, self._copy = idx, copy
        pos = np.full((int(copy.max()) + 1, L), -1, np.int64)
        pos[copy, idx] = np.arange(idx.size)
        self._pos = pos
        self._epmax = int(ep.max()) + 1 if ep.size else 1
        self._a: dict = {}
        self.factor = factor
        self.root, self.key, self.model, self.suite, self.dir = base.root, base.key, base.model, base.suite, base.dir
        self.name = f"{base.name}_x{factor:g}"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        a = self.__dict__["_a"].get(name)
        if a is not None:
            return a
        base = self.__dict__["_base"]
        src = getattr(base, name)
        if not isinstance(src, np.ndarray) or src.ndim == 0 or src.shape[0] != base.L:
            return src
        idx, copy = self._idx, self._copy
        if name in self.PTR:
            p = np.asarray(src, np.int64)[idx]
            out = np.where(p >= 0, self._pos[copy, np.maximum(p, 0)], -1).astype(src.dtype)
        elif name in self.IDS:
            out = (np.asarray(src, np.int64)[idx] + copy * self._epmax).astype(np.int64)
        else:
            order = np.argsort(idx, kind="stable")
            tmp = np.empty((idx.size,) + src.shape[1:], src.dtype)
            tmp[order] = src[idx[order]]
            out = tmp
        out.setflags(write=False)
        self._a[name] = out
        return out

    def has(self, name):
        return self._base.has(name)

    def materialize(self) -> None:
        """Load every per-row array of the base library into memory (rescaled)."""
        for p in sorted(Path(self._base.dir).glob("*.npy")):
            n = p.name[:-4]
            n = n[5:] if n.startswith("rows.") else n
            try:
                getattr(self, n)
            except AttributeError:
                pass

    @property
    def L(self):
        return int(self._idx.size)

    @property
    def H(self):
        return self._base.H

    def tasks(self):
        return sorted(int(t) for t in np.unique(self.task_id))

    def rows_of_task(self, t):
        if "_tr" not in self.__dict__:
            tt = np.asarray(self.task_id, np.int64)
            self.__dict__["_tr"] = {int(k): np.flatnonzero(tt == k) for k in np.unique(tt)}
        return self.__dict__["_tr"].get(int(t), np.zeros(0, np.int64))

    @property
    def meta(self):
        return self._base.meta

    @property
    def episodes(self):
        return self._base.episodes

    @property
    def ids(self):
        b = self._base.ids
        return [f"{b[i]}#c{c}" for i, c in zip(self._idx, self._copy)] if b else []

    def tok(self, name):
        return None

    tok_rows = np.zeros(0, np.int64)


def scaling(method_spec, kwargs=None, cell="pi05_spatial_inf", root=C.DEFAULT_ROOT, scales=(0.5, 1, 2, 4), n_queries=300,
            seed=0, out=None, quiet=False) -> dict:
    from exp.offline_search.harness import api, run as hrun, store as hstore
    root = Path(root)
    cls, src = hrun.load_method_class(method_spec)
    qc = hstore.QueryCell(root, cell)
    base = hstore.LibraryView(root, qc.lib_key, "current")
    jobs = hrun.timing_jobs(qc, None, n=n_queries)
    scratch = Path(tempfile.mkdtemp(prefix="scaling_", dir=C.report_dir(root, "runprof", "_scratch", None)))
    rows = []
    name = None
    warm = ScaledLibrary(base, float(scales[0]), seed)      # untimed warm-up fit: first-call costs (imports, caches)
    warm.materialize()
    hrun.build_method(cls, kwargs).fit(warm, api.Context(root=root, cell=cell, seed=seed, scratch=scratch))
    del warm
    for f in scales:
        lib = ScaledLibrary(base, float(f), seed)
        lib.materialize()          # store IO happens here, outside the timed fit (same for every scale incl. 1x)
        m = hrun.build_method(cls, kwargs)
        name = name or m.name
        ctx = api.Context(root=root, cell=cell, seed=seed, scratch=scratch)
        r0 = hrun.rss_mb()
        t0 = time.perf_counter()
        m.fit(lib, ctx)
        fit_s = time.perf_counter() - t0
        r1 = hrun.rss_mb()
        sizes = {"current": lib.L}
        if hstore.library_available(root, qc.lib_key, "bpool_all"):
            sizes["bpool_all"] = hstore.LibraryView(root, qc.lib_key, "bpool_all").L
        sizes.update({n: int(e["action"].shape[0]) for n, e in ctx.registered.items()})
        m.prof = api.NULL_PROFILER
        cold = hrun.run_jobs_inprocess(m, qc, jobs, lib_sizes=sizes, seed=seed, cell=cell)
        prof = api.Profiler()
        m.prof = prof
        hot = hrun.run_jobs_inprocess(m, qc, jobs, lib_sizes=sizes, seed=seed, cell=cell)
        t = np.asarray(hot["t_ns"], np.float64) / 1e6
        tc = np.asarray(cold["t_ns"], np.float64) / 1e6
        secs = api.Profiler.stats(prof.samples, n_queries=int(t.size))
        rows.append({"scale": float(f), "L": int(lib.L), "fit_s": fit_s, "ms_q": float(t.mean()), "ms_q_p50": float(np.median(t)),
                     "ms_q_p95": float(np.percentile(t, 95)), "ms_q_cold": float(tc.mean()), "rss_fit_delta_mb": r1 - r0,
                     "bpe": float(m.bytes_per_entry()), "n_q": int(t.size),
                     "sections": {k: v.get("per_query_us") for k, v in secs.items()}})
        del m, lib, ctx
    for d in (scratch, scratch.parent):          # leave nothing behind unless the method wrote into its scratch
        try:
            d.rmdir()
        except OSError:
            pass
    Ls = np.array([r["L"] for r in rows], np.float64)
    ms = np.array([r["ms_q"] for r in rows], np.float64)
    slope = float(np.polyfit(np.log(Ls), np.log(ms), 1)[0]) if len(rows) > 1 and np.all(ms > 0) else float("nan")
    fs = np.array([r["fit_s"] for r in rows], np.float64)
    fslope = float(np.polyfit(np.log(Ls), np.log(np.maximum(fs, 1e-6)), 1)[0]) if len(rows) > 1 else float("nan")
    sec_names = sorted({k for r in rows for k in r["sections"]})
    tab = [{**{k: v for k, v in r.items() if k != "sections"}, **{f"us:{k}": r["sections"].get(k) for k in sec_names}} for r in rows]
    cols = [("scale", "scale", "g"), ("L", "L"), ("fit_s", "fit_s", ".3f"), ("ms_q", "ms/q", ".4f"), ("ms_q_p50", "p50", ".4f"),
            ("ms_q_p95", "p95", ".4f"), ("ms_q_cold", "cold", ".4f"), ("rss_fit_delta_mb", "dRSS_fit_MB", ".0f"),
            ("bpe", "bytes/entry", ".0f")] + [(f"us:{k}", f"us:{k}", ".2f") for k in sec_names]
    s = "\n".join([f"## scaling {method_spec} on {cell} (library current rescaled; {rows[0]['n_q'] if rows else 0} fixed "
                   f"timing queries, single thread)", C.md_table(tab, cols),
                   f"log-log slope ms/query ~ L^{slope:.2f}   fit_s ~ L^{fslope:.2f}"])
    if not quiet:
        print(s)
    rdir = C.report_dir(root, "runprof", f"scaling__{name}__{cell}", out)
    C.write_json(rdir / "scaling.json", {"method": method_spec, "kwargs": kwargs or {}, "cell": cell, "rows": rows,
                                         "slope_ms_vs_L": slope, "slope_fit_vs_L": fslope, "src": src})
    C.write_csv(rdir / "scaling.csv", tab)
    if not quiet:
        print(f"\n[runprof scaling] -> {rdir}")
    return {"rows": rows, "slope": slope, "fit_slope": fslope, "report_dir": str(rdir)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in ("summary", "scaling", "live"):
        argv = ["summary"] + argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("summary")
    a1.add_argument("run")
    a1.add_argument("--method", default=None)
    a1.add_argument("--cells", default="all")
    a1.add_argument("--root", default=str(C.DEFAULT_ROOT))
    a1.add_argument("--procs", type=int, default=C.DEFAULT_PROCS, help="worker processes (default: all cores)")
    a1.add_argument("--out", default=None, help="report dir (one subdir per method when several are reported)")
    a2 = sub.add_parser("scaling")
    a2.add_argument("--method", required=True, help="<module_or_file>:<Class> (harness method spec)")
    a2.add_argument("--kwargs", default="{}")
    a2.add_argument("--cell", default="pi05_spatial_inf")
    a2.add_argument("--root", default=str(C.DEFAULT_ROOT))
    a2.add_argument("--scales", default="0.5,1,2,4")
    a2.add_argument("--n-queries", type=int, default=300)
    a2.add_argument("--seed", type=int, default=0)
    a2.add_argument("--out", default=None)
    a3 = sub.add_parser("live")
    a3.add_argument("run")
    a3.add_argument("--method", default=None)
    a3.add_argument("--stale", type=float, default=60.0)
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        summary(a.run, a.method, a.cells, a.root, a.out, procs=a.procs)
    elif a.cmd == "scaling":
        scaling(a.method, json.loads(a.kwargs), a.cell, a.root, tuple(float(x) for x in a.scales.split(",")), a.n_queries,
                a.seed, a.out)
    else:
        live(a.run, a.method, a.stale)


if __name__ == "__main__":
    sys.exit(main())
