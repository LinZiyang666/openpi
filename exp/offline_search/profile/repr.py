"""repr -- diagnostics of a key representation (does its similarity know which actions fit?).

Library side (per model x suite x representation):
  within-task similarity distribution (p5/p50/p95, spread, saturation: fraction of cos > 0.99 / 0.999),
  cross-task similarity p50 and task separation, effective dimensionality (PCA spectrum on the library:
  participation ratio, #components for 90/99% variance, top-1 EVR; globally centered and task-centered).
Query side (per cell, stratified query sample):
  query-candidate similarity distribution + saturation, per-query spread (std over candidates),
  gap top1-top2 (raw and in std units), hubness (k-occurrence of library rows in the queries' kNN lists:
  skewness, fraction never retrieved, share held by the top 1% hubs), action-consistency (per-query
  Spearman between similarity and -err over the task candidates; per-query AUROC of similarity for good
  (err <= floor median, thresholds from the coverage cache when present) vs bad candidates, and a
  threshold-free variant with good = the query's best 10% candidates by err), and the
  representation's own 1-NN err / indist / regret plus the oracle's rank under this similarity.

Representations: built-ins key_v0, key_v1 (cosine), rs (L2 on the valid dims 0..7), tokmean_v0/_v1
(mean over the 256 tokens; token-subsample rows only), or a user callable
  --encode package.module:fn   or   --encode /path/file.py:fn
  fn(view) -> np.ndarray [len(view), D]; `view` is a RowsView: view.get(field) / view[field] (rows of the
  store part: library or query cell), view.kind ('library'|'query'), view.ms, view.cell, view.idx,
  view.task_id, view.step, view.lib (the library RowsView, e.g. for fitting whitening stats).
  (Encoders must not read GT fields such as a_inf / rec_* on the query side.) An object with
  .encode(view) and optional .lib_rows(lib) / .query_rows(qc) is accepted too.

  python -m exp.offline_search.profile.repr [--ms all] [--reprs key_v0,key_v1,rs] [--encode mod:fn --metric cos]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

from . import common as C

MAX_PAIR_SAMPLES = 200_000


class RowsView:
    def __init__(self, store, idx, kind: str, ms: str, cell: str | None = None, lib=None):
        self.store = store
        self.idx = np.asarray(idx, np.int64)
        self.kind, self.ms, self.cell = kind, ms, cell
        self.lib = lib if lib is not None else self
        if kind == "library":
            self.task_id = store.task_id[self.idx]
            self.step = np.asarray(store["step"])[self.idx] if store.has("step") else np.zeros(self.idx.size, np.int32)
        else:
            self.task_id = store.task_id[self.idx]
            self.step = store.step[self.idx]

    def __len__(self):
        return int(self.idx.size)

    def get(self, f: str) -> np.ndarray:
        a = self.store[f]
        o = np.argsort(self.idx, kind="stable")
        out = np.empty((self.idx.size,) + a.shape[1:], a.dtype)
        out[o] = a[self.idx[o]]
        return out

    __getitem__ = get

    def fields(self) -> list[str]:
        return self.store.rows.fields()


class Encoder:
    """name, metric, encode(view); lib_rows(lib) / query_rows(qc) restrict the rows (default: all)."""

    def __init__(self, name, fn, metric="cos", lib_rows=None, query_rows=None):
        self.name, self.fn, self.metric = name, fn, metric
        self._lib_rows, self._query_rows = lib_rows, query_rows

    def encode(self, view: RowsView) -> np.ndarray:
        return np.asarray(self.fn(view), np.float32)

    def lib_rows(self, lib: C.Library) -> np.ndarray:
        return self._lib_rows(lib) if self._lib_rows else np.arange(lib.n)

    def query_rows(self, qc: C.QueryCell) -> np.ndarray | None:
        return self._query_rows(qc) if self._query_rows else None


def _tok_rows_query(qc: C.QueryCell) -> np.ndarray:
    p = qc.dir / "tok" / "rows.npy"
    return np.load(p) if p.exists() else np.zeros(0, np.int64)


def _tok_rows_lib(lib: C.Library) -> np.ndarray:
    d = lib.dir / "tok"
    if (d / "rows.npy").exists():
        return np.load(d / "rows.npy")
    for n in ("v0.npy", "vision_0.npy"):
        if (d / n).exists():
            return np.arange(np.load(d / n, mmap_mode="r").shape[0])
    return np.zeros(0, np.int64)


def _tokmean(which: str):
    def fn(view: RowsView):
        d = (view.store.dir / "tok")
        arr = None
        for n in (f"{which}.npy", f"vision_{which[-1]}.npy"):
            if (d / n).exists():
                arr = np.load(d / n, mmap_mode="r")
                break
        if arr is None:
            raise FileNotFoundError(f"{d}: no token array for {which}")
        rows_file = d / "rows.npy"
        pos_of = {int(r): i for i, r in enumerate(np.load(rows_file))} if rows_file.exists() else None
        out = np.empty((len(view), arr.shape[-1]), np.float32)
        for j, r in enumerate(view.idx):
            i = pos_of[int(r)] if pos_of is not None else int(r)
            out[j] = np.asarray(arr[i], np.float32).mean(0)
        return out
    return fn


def builtin(name: str) -> Encoder:
    if name in ("key_v0", "key_v1"):
        return Encoder(name, lambda v, f=name: v.get(f), "cos")
    if name == "rs":
        return Encoder(name, lambda v: C.valid_rs(v.get("rs"), v.ms.split("_")[0]).astype(np.float32), "l2")
    if name in ("tokmean_v0", "tokmean_v1"):
        return Encoder(name, _tokmean(name[-2:]), "cos", _tok_rows_lib, _tok_rows_query)
    raise SystemExit(f"unknown representation {name!r} (built-ins: key_v0, key_v1, rs, tokmean_v0, tokmean_v1)")


def load_callable(spec: str):
    mod, _, attr = spec.rpartition(":")
    if mod.endswith(".py") or "/" in mod:
        s = importlib.util.spec_from_file_location(Path(mod).stem, mod)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, attr)


def user_encoder(spec: str, metric: str, name: str | None = None) -> Encoder:
    obj = load_callable(spec)
    if hasattr(obj, "encode"):
        o = obj() if isinstance(obj, type) else obj
        enc = Encoder(name or spec.rpartition(":")[2], o.encode, getattr(o, "metric", metric),
                      getattr(o, "lib_rows", None), getattr(o, "query_rows", None))
        return enc
    return Encoder(name or spec.rpartition(":")[2], obj, metric)


# ------------------------------------------------------------------------------------------- math
def prep(X: np.ndarray, metric: str) -> np.ndarray:
    X = np.asarray(X, np.float32)
    if metric == "cos":
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, 1e-12)
    return X


def sim(A: np.ndarray, B: np.ndarray, metric: str) -> np.ndarray:
    """Similarity (higher = closer). cos/dot: inner product of prepped rows; l2: -euclidean distance."""
    if metric in ("cos", "dot"):
        return (A @ B.T).astype(np.float64)
    a2 = (A.astype(np.float64) ** 2).sum(1)[:, None]
    b2 = (B.astype(np.float64) ** 2).sum(1)[None]
    d2 = a2 + b2 - 2.0 * (A.astype(np.float64) @ B.astype(np.float64).T)
    return -np.sqrt(np.maximum(d2, 0))


def dist_stats(v: np.ndarray, metric: str, prefix: str) -> dict:
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return {}
    q = np.quantile(v, [0.05, 0.5, 0.95])
    d = {f"{prefix}p05": q[0], f"{prefix}p50": q[1], f"{prefix}p95": q[2], f"{prefix}spread": q[2] - q[0],
         f"{prefix}std": float(v.std())}
    if metric == "cos":
        d[f"{prefix}sat99"] = float((v > 0.99).mean())
        d[f"{prefix}sat999"] = float((v > 0.999).mean())
    return d


def spectrum(X: np.ndarray, max_rows: int, rng) -> dict:
    """Effective dimensionality of the (already centered) rows X."""
    if X.shape[0] > max_rows:
        X = X[np.sort(rng.choice(X.shape[0], max_rows, replace=False))]
    X = X.astype(np.float64 if X.shape[1] <= 4096 else np.float32)
    G = (X.T @ X) if X.shape[1] < X.shape[0] else (X @ X.T)
    lam = np.clip(np.linalg.eigvalsh(G.astype(np.float64))[::-1], 0, None)
    tot = lam.sum()
    if tot <= 0:
        return {"pr": 0.0, "n90": 0, "n99": 0, "evr1": np.nan}
    c = np.cumsum(lam) / tot
    return {"pr": float(tot ** 2 / (lam ** 2).sum()), "n90": int(np.searchsorted(c, 0.90) + 1),
            "n99": int(np.searchsorted(c, 0.99) + 1), "evr1": float(lam[0] / tot),
            "evr5": [float(x) for x in (lam[:5] / tot)]}


def _pairs_sample(S: np.ndarray, rng, triu: bool, cap: int) -> np.ndarray:
    v = S[np.triu_indices(S.shape[0], 1)] if triu else S.ravel()
    if v.size > cap:
        v = v[rng.choice(v.size, cap, replace=False)]
    return v


def sample_queries(qc: C.QueryCell, n: int, rng, restrict: np.ndarray | None) -> np.ndarray:
    pool = np.arange(qc.n) if restrict is None else np.asarray(restrict, np.int64)
    if pool.size == 0:
        return pool
    tasks = np.unique(qc.task_id[pool])
    per = int(np.ceil(n / max(tasks.size, 1)))
    out = []
    for t in tasks:
        cand = pool[qc.task_id[pool] == t]
        out.append(cand if cand.size <= per else rng.choice(cand, per, replace=False))
    return np.sort(np.concatenate(out))


def subsample_episodes(lib: C.Library, rows: np.ndarray, cap: int, rng) -> np.ndarray:
    """Keep whole library episodes, the same fraction per task, until <= cap rows (large bpool_all libraries)."""
    ep = np.asarray(lib["episode"])[rows] if lib.has("episode") else rows
    t = lib.task_id[rows]
    frac = cap / rows.size
    keep = np.zeros(rows.size, bool)
    for k in np.unique(t):
        eps = np.unique(ep[t == k])
        pick = rng.choice(eps, max(1, int(np.floor(frac * eps.size))), replace=False)
        keep |= (t == k) & np.isin(ep, pick)
    return rows[keep]


# -------------------------------------------------------------------------------------------- job
def repr_job(job: dict) -> dict:
    root, ms, libname, enc_spec = job["root"], job["ms"], job["lib"], job["enc"]
    enc = builtin(enc_spec) if not enc_spec.startswith("user:") else user_encoder(enc_spec[5:], job["metric"], job.get("name"))
    if job.get("metric") and job.get("metric_forced"):
        enc.metric = job["metric"]
    rng = np.random.default_rng(job["seed"])
    lib = C.open_library(root, ms, libname)
    cur = C.open_library(root, ms, "current")
    if lib is None or cur is None:
        return {"ms": ms, "repr": enc.name, "error": f"library {libname!r}/current missing for {ms}"}
    lrows = np.asarray(enc.lib_rows(lib), np.int64)
    if lrows.size == 0:
        return {"ms": ms, "repr": enc.name, "error": "encoder selects no library rows"}
    n_full = int(lrows.size)
    if job.get("lib_max") and lrows.size > job["lib_max"]:
        lrows = subsample_episodes(lib, lrows, int(job["lib_max"]), rng)
    lview = RowsView(lib, lrows, "library", ms)
    X = prep(enc.encode(lview), enc.metric)
    lt = lib.task_id[lrows]
    sigma = C.lib_sigma(cur["action"])
    Ls = C.scaled_seg(lib["action"][np.sort(lrows)], sigma)[np.argsort(np.argsort(lrows))]
    # ---------------- library side
    within, cross = [], []
    tasks = np.unique(lt)
    for t in tasks:
        m = lt == t
        if m.sum() > 1:
            within.append(_pairs_sample(sim(X[m], X[m], enc.metric), rng, True, MAX_PAIR_SAMPLES // max(tasks.size, 1)))
    for _ in range(min(20, tasks.size * (tasks.size - 1))):
        a, b = rng.choice(tasks, 2, replace=False)
        ia = rng.choice(np.flatnonzero(lt == a), min(40, int((lt == a).sum())), replace=False)
        ib = rng.choice(np.flatnonzero(lt == b), min(40, int((lt == b).sum())), replace=False)
        cross.append(sim(X[ia], X[ib], enc.metric).ravel())
    w = np.concatenate(within) if within else np.zeros(0)
    cr = np.concatenate(cross) if cross else np.zeros(0)
    librow = {"ms": ms, "repr": enc.name, "metric": enc.metric, "lib": libname, "n_lib": int(lrows.size), "n_lib_full": n_full,
              "dim": int(X.shape[1]),
              **dist_stats(w, enc.metric, "w_")}
    if cr.size and w.size:
        librow["x_p50"] = float(np.median(cr))
        librow["task_sep"] = float((np.median(w) - np.median(cr)) / max(w.std(), 1e-12))
    Xc = X - X.mean(0, keepdims=True)
    sp = spectrum(Xc, job["pca_max"], rng)
    librow.update({"pr": sp["pr"], "n90": sp["n90"], "n99": sp["n99"], "evr1": sp["evr1"], "evr5": sp.get("evr5")})
    Xt = X.copy()
    for t in tasks:
        m = lt == t
        Xt[m] -= Xt[m].mean(0, keepdims=True)
    spt = spectrum(Xt, job["pca_max"], rng)
    librow.update({"pr_task": spt["pr"], "n90_task": spt["n90"]})
    del Xc, Xt
    # ---------------- query side
    cov = C.load_coverage(root, libname)
    qrows_out = []
    for arm in C.ARMS:
        cell = f"{ms}_{arm}"
        if cell not in job["cells"]:
            continue
        try:
            qc = C.QueryCell(root, cell)
        except FileNotFoundError:
            continue
        qsel = sample_queries(qc, job["n_queries"], rng, enc.query_rows(qc))
        if qsel.size == 0:
            qrows_out.append({"cell": cell, "repr": enc.name, "error": "no query rows"})
            continue
        Q = prep(enc.encode(RowsView(qc, qsel, "query", ms, cell, lview)), enc.metric)
        Qs = C.scaled_seg(qc["a_inf"][qsel], sigma)
        if cov is not None and cov.has(cell):
            thr = cov.get(cell, "thr")[qsel].astype(np.float64)
            thr_src = f"coverage_{libname}"
        else:
            fl = C.load_floor(root, ms, sigma)
            thr = fl.thr(qc.third[qsel]) if fl is not None else np.full(qsel.size, np.nan)
            thr_src = "floor" if fl is not None else "none"
        k = job["k"]
        occ = np.zeros(lrows.size, np.int64)
        seen_lib = np.zeros(lrows.size, bool)
        sims_sample, qstd, gap12, gapn, gapmed, rho, auc, auc_rel, nn_err, nn_ind, orc_err, orc_rank, n_both = \
            ([] for _ in range(13))
        for t in np.unique(qc.task_id[qsel]):
            qi = np.flatnonzero(qc.task_id[qsel] == t)
            ci = np.flatnonzero(lt == t)
            if ci.size < 2:
                continue
            seen_lib[ci] = True
            S = sim(Q[qi], X[ci], enc.metric)
            E = C.err_matrix(Qs[qi], Ls[ci])
            sims_sample.append(_pairs_sample(S, rng, False, MAX_PAIR_SAMPLES // 20))
            srt = -np.sort(-S, axis=1)
            sd = S.std(1)
            qstd.append(sd)
            gap12.append(srt[:, 0] - srt[:, 1])
            gapn.append((srt[:, 0] - srt[:, 1]) / np.maximum(sd, 1e-12))
            gapmed.append((srt[:, 0] - np.median(S, 1)) / np.maximum(sd, 1e-12))
            kk = min(k, ci.size)
            nn = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            np.add.at(occ, ci[nn.ravel()], 1)
            rho.append(C.spearman_rows(S, -E))
            good = E <= thr[qi][:, None]
            auc.append(C.auroc_rows(S, good))
            auc_rel.append(C.auroc_rows(S, E <= np.quantile(E, 0.1, axis=1, keepdims=True)))
            n_both.append(int(((good.any(1)) & ((~good).any(1))).sum()))
            top = np.argmax(S, 1)
            e_nn = E[np.arange(qi.size), top]
            nn_err.append(e_nn)
            nn_ind.append(e_nn <= thr[qi])
            oe = E.min(1)
            orc_err.append(oe)
            oj = np.argmin(E, 1)
            so = S[np.arange(qi.size), oj]
            orc_rank.append((S > so[:, None]).sum(1) + 1)
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
        from scipy.stats import skew
        occ_s = occ[seen_lib]
        top1pct = max(1, int(np.ceil(0.01 * occ_s.size)))
        rr = cat(rho)
        ne, oe_ = cat(nn_err), cat(orc_err)
        orank = cat(orc_rank)
        qrows_out.append({
            "cell": cell, "repr": enc.name, "metric": enc.metric, "nq": int(qsel.size), "thr_src": thr_src,
            **dist_stats(cat(sims_sample), enc.metric, "q_"),
            "qstd": float(cat(qstd).mean()), "gap12_p50": float(np.median(cat(gap12))), "gap12n_p50": float(np.median(cat(gapn))),
            "gapmed_n_p50": float(np.median(cat(gapmed))),
            "hub_skew": float(skew(occ_s)) if occ_s.size > 2 and occ_s.std() > 0 else 0.0,
            "antihub": float((occ_s == 0).mean()) if occ_s.size else np.nan,
            "hub1pct_share": float(np.sort(occ_s)[::-1][:top1pct].sum() / max(occ_s.sum(), 1)),
            "rho_mean": float(np.nanmean(rr)) if rr.size else np.nan, "rho_p50": float(np.nanmedian(rr)) if rr.size else np.nan,
            "auroc_mean": float(np.mean(cat(auc))) if cat(auc).size else np.nan, "auroc_n": int(cat(auc).size),
            "auroc_top10": float(np.mean(cat(auc_rel))) if cat(auc_rel).size else np.nan,
            "nn_err": float(ne.mean()) if ne.size else np.nan, "nn_indist": float(cat(nn_ind).mean()) if ne.size else np.nan,
            "oracle_err": float(oe_.mean()) if oe_.size else np.nan, "nn_regret": float((ne - oe_).mean()) if ne.size else np.nan,
            "orc_rank_p50": float(np.median(orank)) if orank.size else np.nan,
            "orc_top10": float((orank <= 10).mean()) if orank.size else np.nan})
    return {"ms": ms, "repr": enc.name, "lib_row": librow, "query_rows": qrows_out}


LIB_COLS = [("ms", "ms"), ("repr", "repr"), ("metric", "metric"), ("n_lib", "n_lib"), ("n_lib_full", "of"), ("dim", "dim"), ("w_p05", "w_p05", ".4g"),
            ("w_p50", "w_p50", ".4g"), ("w_p95", "w_p95", ".4g"), ("w_sat99", "sat>.99"), ("w_sat999", "sat>.999"),
            ("x_p50", "cross_p50", ".4g"), ("task_sep", "task_sep", ".2f"), ("pr", "PR", ".1f"), ("n90", "n90"), ("n99", "n99"),
            ("evr1", "evr1"), ("pr_task", "PR_task", ".1f"), ("n90_task", "n90_task")]
Q_COLS = [("cell", "cell"), ("repr", "repr"), ("nq", "nq"), ("q_p50", "sim_p50", ".4g"), ("q_spread", "spread", ".3g"),
          ("q_sat99", "sat>.99"), ("qstd", "q_std", ".3g"), ("gap12n_p50", "gap12/std"), ("gapmed_n_p50", "(top-med)/std"),
          ("hub_skew", "hub_skew", ".2f"), ("antihub", "antihub"), ("hub1pct_share", "top1%share"),
          ("rho_mean", "rho(sim,-err)"), ("auroc_mean", "AUROC good"), ("auroc_top10", "AUROC top10%"),
          ("nn_err", "1nn_err"), ("nn_indist", "1nn_ind"),
          ("oracle_err", "orc_err"), ("nn_regret", "regret"), ("orc_rank_p50", "orc_rank_p50", ".0f"), ("orc_top10", "orc@10")]


def run(root=C.DEFAULT_ROOT, ms="all", reprs="key_v0,key_v1,rs", encode=None, metric=None, name=None, lib="current",
        cells="all", n_queries=2000, k=10, pca_max=2000, procs=None, seed=0, out=None, quiet=False, lib_max=8000) -> dict:
    root = Path(root)
    mss = list(C.MODEL_SUITES) if ms == "all" else ms.split(",")
    cells_l = C.expand_cells(cells)
    specs = [r for r in (reprs.split(",") if reprs else []) if r]
    if encode:
        specs.append("user:" + encode)
    jobs = [dict(root=str(root), ms=m, lib=lib, enc=e, metric=metric or "cos", metric_forced=metric is not None,
                 name=name, seed=seed, n_queries=n_queries, k=k, pca_max=pca_max, cells=cells_l, lib_max=lib_max)
            for m in mss for e in specs if (root / "library" / m).exists()]
    res = C.pmap(repr_job, jobs, procs)
    lib_rows = [r["lib_row"] for r in res if "lib_row" in r]
    q_rows = [q for r in res for q in r.get("query_rows", [])]
    errs = [r["error"] for r in res if "error" in r]
    L = [f"## repr diagnostics (library `{lib}`; {n_queries} queries/cell stratified by task; kNN k={k}; seed {seed})",
         "w_* = within-task library-library similarity; task_sep = (within p50 - cross p50)/within std; PR = participation ratio;"
         " rho = per-query Spearman(sim, -err) over the task candidates; AUROC good = per-query AUROC of sim for err<=thr"
         " (queries with both classes only); AUROC top10% = same with good = the query's best 10% candidates;"
         " 1nn_* = this representation's own top-1; orc_rank = rank of the err-oracle under this similarity.", ""]
    L.append(C.md_table(lib_rows, LIB_COLS, "library side"))
    L.append("")
    L.append(C.md_table(sorted(q_rows, key=lambda r: (C.CELLS.index(r["cell"]), r["repr"])), Q_COLS, "query side"))
    if errs:
        L.append("errors: " + "; ".join(errs))
    s = "\n".join(L)
    if not quiet:
        print(s)
    tag = "_".join(sorted({r["repr"] for r in res if "repr" in r})) or "none"
    rdir = C.report_dir(root, "repr", f"{lib}__{tag}"[:120], out)
    C.write_json(rdir / "repr.json", {"lib": lib, "n_queries": n_queries, "k": k, "seed": seed, "library": lib_rows,
                                      "query": q_rows, "errors": errs})
    C.write_csv(rdir / "repr_library.csv", [{k_: v for k_, v in r.items() if k_ != "evr5"} for r in lib_rows])
    C.write_csv(rdir / "repr_query.csv", q_rows)
    (rdir / "repr.md").write_text(s + "\n")
    if not quiet:
        print(f"\n[repr] {len(res)} jobs -> {rdir}")
    return {"library": lib_rows, "query": q_rows, "errors": errs, "report_dir": str(rdir)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(C.DEFAULT_ROOT))
    ap.add_argument("--ms", default="all", help="model_suite list, e.g. pi05_spatial,groot_l10")
    ap.add_argument("--reprs", default="key_v0,key_v1,rs", help="built-ins (empty string for none)")
    ap.add_argument("--encode", default=None, help="user encoder module:fn or file.py:fn")
    ap.add_argument("--name", default=None, help="display name of the user encoder")
    ap.add_argument("--metric", default=None, choices=["cos", "l2", "dot"], help="force the metric (default: per repr)")
    ap.add_argument("--lib", default="current")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pca-max", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=C.DEFAULT_PROCS, help="worker processes (default: all cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lib-max", type=int, default=8000,
                    help="cap on library rows (whole episodes subsampled per task; bpool_all is ~10x current)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run(a.root, a.ms, a.reprs, a.encode, a.metric, a.name, a.lib, a.cells, a.n_queries, a.k, a.pca_max, a.procs, a.seed, a.out,
        lib_max=a.lib_max)


if __name__ == "__main__":
    sys.exit(main())
