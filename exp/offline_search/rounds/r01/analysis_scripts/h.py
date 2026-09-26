"""Shared helpers for the R1 analysis scripts (read-only on results)."""
import json, os, functools
import numpy as np

REPO = "/home/weiland/projects/openpi"
RES = f"{REPO}/exp/offline_search/results"
ROOT = "/dev/shm/offline_search_store"
OUT = "/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r01"
MODELS = ("pi05", "groot")
SUITES = ("spatial", "l10")
CELLS = [f"{m}_{s}_{a}" for m in MODELS for s in SUITES for a in ("inf", "cache")]
MS = [f"{m}_{s}" for m in MODELS for s in SUITES]
R0 = {"B0_current", "B1_lda", "B2_random", "B3_oracle", "B4_rs", "B4_v0", "B4_v1"}


def rundir(method):
    return f"{RES}/{'r00' if method in R0 else 'r01'}/{method}"


@functools.lru_cache(maxsize=None)
def load(method, cell):
    p = f"{rundir(method)}/{cell}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    d = {k: z[k] for k in z.files}
    return d


@functools.lru_cache(maxsize=None)
def cell_json(method, cell):
    p = f"{rundir(method)}/{cell}.json"
    return json.load(open(p)) if os.path.exists(p) else None


@functools.lru_cache(maxsize=None)
def episodes(cell):
    return json.load(open(f"{ROOT}/queries/{cell}/episodes.json"))


def aurc(err, conf, w=None):
    """Harness definition: stable sort by confidence descending (ties by row order); risk(c) = mean err of the
    accepted top fraction; AURC = mean of risk over c = 1/N..1. With weights w: weighted version."""
    err = np.asarray(err, np.float64); conf = np.asarray(conf, np.float64)
    o = np.lexsort((np.arange(len(conf)), -conf))
    e = err[o]
    if w is None:
        r = np.cumsum(e) / np.arange(1, len(e) + 1)
        return float(r.mean())
    ww = np.asarray(w, np.float64)[o]
    cw = np.cumsum(ww)
    r = np.cumsum(e * ww) / cw
    return float(np.sum(r * ww) / ww.sum())


def risk_at(err, conf, c, w=None):
    err = np.asarray(err, np.float64); conf = np.asarray(conf, np.float64)
    o = np.lexsort((np.arange(len(conf)), -conf))
    e = err[o]
    if w is None:
        k = max(1, int(round(c * len(e))))
        return float(e[:k].mean())
    ww = np.asarray(w, np.float64)[o]
    cw = np.cumsum(ww) / ww.sum()
    k = int(np.searchsorted(cw, c, side="left")) + 1
    return float(np.sum(e[:k] * ww[:k]) / ww[:k].sum())


def ep_boot(vals, ep, reps=1000, seed=0):
    """Episode-bootstrap 95% CI of the mean of vals (per-decision)."""
    vals = np.asarray(vals, np.float64); ep = np.asarray(ep)
    ue, inv = np.unique(ep, return_inverse=True)
    s = np.bincount(inv, weights=vals); n = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ue), size=(reps, len(ue)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
