"""Shared constants and helpers for the R9 fable exploration tools.

Everything here is read-only with respect to the capture roots. Derived tables
are written under ``DERIVED`` (``offline_search_store/derived/r09_fable``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

RUN_ROOT = Path(os.environ.get("R9F_RUN_ROOT", "/home/weiland/trace_runs/os_closed_loop/r08_main"))
STORE = Path(os.environ.get("R9F_STORE", "/home/weiland/trace_runs/offline_search_store"))
DERIVED = Path(os.environ.get("R9F_DERIVED", str(STORE / "derived" / "r09_fable")))
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"

# Owner prices (R8 REPORT §3): look (both cameras), wrist-only look, policy call.
PRICE = {"pi05": dict(look=0.152, wrist=0.0646, call=0.848),
         "groot": dict(look=0.148, wrist=0.0646, call=0.852)}
# Pure-policy references with 10-control commitment (R8 arm table, same topology).
P10_SR = {("pi05", "l10"): 0.908, ("pi05", "spatial"): 0.988, ("groot", "l10"): 0.898, ("groot", "spatial"): 0.940}
DISCOVERY_INITS = range(0, 30)   # inits 30-49 are the coordinator's locked holdout
MODELS = ("pi05", "groot")
SUITES = ("l10", "spatial")
LIBS = (50, 500)
CELLS = [(m, s, l) for m in MODELS for s in SUITES for l in LIBS]
EXEC = 5          # executed controls per decision
H10 = 10          # controls per look-commit (anchor chunk + blind tail)
VALID = slice(0, 7)
GRIP = 6


def arm_name(model, suite, lib, variant):
    """R8 arm naming: ``r8_<model>_<suite>_<lib>_<variant>``; pure policy has no library."""
    if variant == "P10":
        return f"r8_{model}_{suite}_P10"
    return f"r8_{model}_{suite}_{lib}_{variant}"


def parse_arm(name):
    """Inverse of :func:`arm_name`; returns (model, suite, lib|None, variant)."""
    parts = name.split("_")
    if len(parts) < 4 or parts[0] != "r8":
        raise ValueError(f"not an R8 arm name: {name}")
    model, suite = parts[1], parts[2]
    if parts[3] == "P10":
        return model, suite, None, "P10"
    return model, suite, int(parts[3]), "_".join(parts[4:])


def load_arms():
    return json.loads((RUN_ROOT / "arms.json").read_text())


def discovery_mask(init):
    init = np.asarray(init)
    return (init >= DISCOVERY_INITS.start) & (init < DISCOVERY_INITS.stop)


def task_strat_bootstrap(values, task, n_boot=2000, seed=0, stat=np.mean):
    """Task-stratified resampling of per-episode values (rows are (task, init) pairs).

    ``values`` may be 1-D (one statistic) or 2-D (n, k): every column is resampled
    with the same index draw so paired contrasts stay paired. Returns (n_boot, k).
    """
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    task = np.asarray(task)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(task == t) for t in np.unique(task)]
    out = np.empty((n_boot, values.shape[1]))
    for b in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        out[b] = stat(values[idx], axis=0)
    return out


def ci(boot, alpha=0.05):
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return lo, hi


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_default) + "\n")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
